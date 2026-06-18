import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import cv2
import numpy as np

from cryoemservices.util.image_processing.shared import is_image_stack

logger = logging.getLogger(__name__)


def drift_correct_image(
    array: np.ndarray,
    start_from: Literal["beginning", "middle", "end"] = "middle",
    max_iters: int = 100,
    eps: float = 1e-6,
    use_mask: bool = True,
    downsampling_factor: int = 2,
    num_procs: int = 1,
) -> np.ndarray:
    """
    Helper function that performs drift correction on an image stack using OpenCV's
    Enhanced Correlation Coefficient (ECC) maximisation routine, details of which
    can be found in the following paper (DOI: 10.1109/TPAMI.2008.113.)
    http://xanthippi.ceid.upatras.gr/people/evangelidis/george_files/PAMI_2008.pdf

    Parameters
    ----------
    array: np.ndarray
        The image stack to be aligned. This will be a grayscale or RGB image as
        a NumPy array.

    start_from: Literal["beginning", "middle", "end"] = "middle"
        The part of the array to use as the reference. For CLEM image stacks, the
        frames in the middle of the stack tend to be the most in focus, and are
        best used as the starting point for registration.

    max_iters: int = 100
        The maximum number of iterations before stopping the registration attempt.

    eps: float = 1e-6
        The convergence tolerance for ECC optimisation. The registration will stop
        when the relative improvement between iterations falls below this value.

    use_mask: bool = True,
        Applies a circular mask so that only the central part of the image is
        considered during registration.

    downsampling_factor: int = 2,
        The degree of downsampling to apply to the image.

    Returns
    -------
    aligned: np.ndarray
        The aligned image stack as a NumPy array.
    """

    def _register(
        frame_num: int,
        prev: np.ndarray,
        curr: np.ndarray,
        warp_init: np.ndarray,
        mask: np.ndarray | None,
        downsampling_factor: int,
    ):
        try:
            logger.info(f"Registering frame {frame_num}")
            # For RGB images, create a weighted grayscale image to use for registration
            scale = 1 / (downsampling_factor or 1)
            prev_gray = (
                (
                    # Luma-style weighted sum; avoids cv2.cvtColor copy
                    0.2126 * prev[..., 0]
                    + 0.7152 * prev[..., 1]
                    + 0.0722 * prev[..., 2]
                ).astype(np.float32)
                if prev.ndim == 3
                else prev.astype(np.float32)
            )
            curr_gray = (
                (
                    # Luma-style weighted sum; avoids cv2.cvtColor copy
                    0.2126 * curr[..., 0]
                    + 0.7152 * curr[..., 1]
                    + 0.0722 * curr[..., 2]
                ).astype(np.float32)
                if curr.ndim == 3
                else curr.astype(np.float32)
            )
            # Downsample arrays as needed
            if downsampling_factor > 1:
                prev_gray = cv2.resize(
                    prev_gray,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
                curr_gray = cv2.resize(
                    curr_gray,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
                if mask is not None:
                    mask = cv2.resize(
                        mask,
                        dsize=None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_NEAREST,
                    )

            # Compute and return the transform
            warp = warp_init.copy()
            cv2.findTransformECC(
                prev_gray,
                curr_gray,
                warp,
                motionType=cv2.MOTION_EUCLIDEAN,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    max_iters,
                    eps,
                ),
                inputMask=mask,
                gaussFiltSize=5,
            )  # Updated in-place
            warp[:2, 2] *= downsampling_factor
            return warp, frame_num
        except cv2.error:
            logger.warning(
                f"Error registering frame {frame_num}. Using identity matrix",
                exc_info=True,
            )
            return warp_init, frame_num

    def _make_homogeneous(M: np.ndarray):
        H = np.eye(3, dtype=np.float32)
        H[:2] = M
        return H

    # Validate that this is a grayscale or RGB array to begin with
    if not is_image_stack(array):
        logger.warning(
            f"Image provided likely not an image stack (has dimensions {array.shape});"
            "returning original array"
        )
        return array

    def _warp_frame(
        frame_num: int,
        frame: np.ndarray,
        M: np.ndarray,
    ):
        try:
            logger.info(f"Transforming frame {frame_num}")
            out = cv2.warpAffine(
                frame,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=float(vmin),
            )
            np.clip(out, a_min=vmin, a_max=vmax, out=out)
            return out, frame_num
        except Exception:
            logger.warning(
                "Could not apply transformation to frame. Returning original frame"
            )
            return frame, frame_num

    # Start of function
    cv2.setNumThreads(1)
    start_time = time.perf_counter()

    # Extract dimensions, dtype, vmin, and vmax
    z, h, w = array.shape[:3]
    dtype = array.dtype
    vmin, vmax = array.min(), array.max()
    logger.debug(
        f"shape: {array.shape}\ndtype: {array.dtype}\nvmin: {vmin}\nvmax: {vmax}"
    )

    # Set the reference index
    if start_from == "beginning":
        ref_idx = 0
    elif start_from == "middle":
        ref_idx = z // 2
    elif start_from == "end":
        ref_idx = z - 1
    else:
        raise ValueError(f"Invalid input for 'start_from' parameter: {start_from}")

    # Preallocate empty arrays
    placeholder_start_time = time.perf_counter()

    # Output stack with original dimensions
    aligned = np.empty(array.shape, dtype=dtype)
    aligned[ref_idx] = array[ref_idx]

    # Create identity Euclidean transform
    I = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )

    # Create placeholder for the per-frame transforms
    transforms = np.zeros((z, 2, 3), dtype=np.float32)
    transforms[ref_idx] = I

    placeholder_end_time = time.perf_counter()
    logger.info("Allocated placeholder arrays")
    logger.debug(
        f"Allocated placeholder arrays in {placeholder_end_time - placeholder_start_time}s"
    )

    # Create a mask
    mask: np.ndarray | None = None
    if use_mask:
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (w // 2, h // 2), min(h, w) // 3, 255, -1)  # Updated in-place
        logger.debug("Created mask")

    # Construct the list of current and previous frames to use for the alignment
    frames_to_align = [
        # Forward pass from reference frame
        *[(i - 1, i) for i in range(ref_idx + 1, z)],
        # Backward pass from reference frame
        *[(i + 1, i) for i in range(ref_idx - 1, -1, -1)],
    ]
    # Register and collect warps relative to reference frames
    registration_start_time = time.perf_counter()
    logger.info("Performing image registration")
    with ThreadPoolExecutor(max_workers=num_procs) as pool:
        futures = [
            pool.submit(
                _register,
                curr,
                array[prev],
                array[curr],
                I,
                mask,
                downsampling_factor,
            )
            for prev, curr in frames_to_align
        ]
        for future in as_completed(futures):
            warp, frame_num = future.result()
            transforms[frame_num] = warp

    # Update the per-frame transforms with the cumulative ones
    logger.info("Calculating cumulative transformations")
    for i in range(ref_idx + 1, z):
        transforms[i] = (
            _make_homogeneous(transforms[i - 1]) @ _make_homogeneous(transforms[i])
        )[:2]
    for i in range(ref_idx - 1, -1, -1):
        transforms[i] = (
            _make_homogeneous(transforms[i + 1]) @ _make_homogeneous(transforms[i])
        )[:2]
    registration_end_time = time.perf_counter()
    logger.debug(
        f"Registration completed in {registration_end_time - registration_start_time}s"
    )

    # Warp the frames and assign them to the aligned array
    warp_start_time = time.perf_counter()
    logger.info("Applying transformations to frames")
    with ThreadPoolExecutor(max_workers=num_procs) as pool:
        futures = [
            pool.submit(
                _warp_frame,
                f,
                array[f],
                transforms[f],
            )
            for f in range(z)
            if f != ref_idx
        ]
        for future in as_completed(futures):
            frame, frame_num = future.result()
            aligned[frame_num] = frame.astype(dtype, copy=False)
    warp_end_time = time.perf_counter()
    logger.debug(
        f"Completed transformation of frames in {warp_end_time - warp_start_time}s"
    )

    end_time = time.perf_counter()
    logger.debug(
        f"Completed drift correction of current image in {end_time - start_time}s"
    )
    return aligned.astype(dtype, copy=False)
