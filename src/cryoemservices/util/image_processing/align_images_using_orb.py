import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast

import cv2
import numpy as np

from cryoemservices.util.image_processing.shared import (
    apply_sobel_edge_filter,
    create_hanning_window,
    is_image_stack,
)

logger = logging.getLogger(__name__)


def align_images_using_orb(
    reference_array: np.ndarray,
    moving_array: np.ndarray,
    # Preprocessing parameters
    sigma: int = 3,
    hanning_window: bool = True,
    kernel_size: int = 3,
    # Hole detection parameters
    min_area: int = 100,
    max_area: int = 5000,
    # ORB detection parameters
    patch_size: int = 31,
    # Parallel processing parameters,
    num_procs=1,
):
    """
    Align a moving image or image stack to a reference using ORB (Oriented FAST and
    Rotated BRIEF) feature matching (DOI: https://doi.org/10.1109/ICCV.2011.6126544).

    The function looks for holes in both images using Sobel edge filtering followed
    by thresholding and contour analysis. It calculates the centroids of the detected
    holes, and uses them as fixed keypoint locations to compute ORB descriptors with.

    The descriptors are matched between the reference and moving images, and an affine
    transformation is estimated using RANSAC and applied to the moving image.

    THis function supports both single images and image stacks, along with grayscale
    and RGB ones.

    Parameters
    ----------
    reference_array : np.ndarray
        Reference image or image stack to which the moving image(s) will be aligned.
        Must have the same shape as `moving_array`.

    moving_array : np.ndarray
        Image or image stack to be aligned with the reference.

    sigma : int, optional
        Standard deviation for the Gaussian blur applied during preprocessing.
        Default is 3.

    hanning_window : bool, optional
        If True, applies a 2D Hanning window prior to edge detection to reduce
        boundary artifacts. Default is True.

    kernel_size : int, optional
        Kernel size used for the Sobel edge filter. Default is 3.

    min_area : int, optional
        Minimum contour area used to identify candidate hole regions.
        Default is 100.

    max_area : int, optional
        Maximum contour area used to identify candidate hole regions.
        Default is 5000.

    patch_size : int, optional
        Patch size used when creating ORB keypoints at detected hole centroids.
        Default is 31.

    num_procs : int, optional
        Number of worker threads used to process image frames in parallel.
        Default is 1.

    Returns
    -------
    np.ndarray
        The aligned moving image or image stack with the same shape as the input.

    """

    def _extract_keypoints(
        img: np.ndarray,
        window: np.ndarray | None,
        sigma: int,  # Gaussian blur
        kernel_size: int,  # Sobel filter
        struct: np.ndarray,  # Structuring element
        min_area: int,  # Contour evaluation
        max_area: int,  # Contour evaluation
        patch_size: int,  # Keypoints
    ):
        # Preprocess image
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        windowed = (
            (blurred * window).astype(np.float32)
            if window is not None
            else blurred.astype(np.float32)
        )
        sobel = apply_sobel_edge_filter(windowed, kernel_size)

        # Threshold the image
        _, thres = cv2.threshold(
            sobel,
            sobel.max() * 0.3,
            sobel.max(),
            cv2.THRESH_BINARY,
        )
        # Apply morphological closing
        # Connect nearby edges and fill holes inside detected shapes
        cleaned = cv2.morphologyEx(
            thres,
            cv2.MORPH_CLOSE,
            struct,
        )

        # Extract external boundaries from the cleaned image
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        keypoints: list[cv2.KeyPoint] = []
        for contour in contours:
            # Filter out overly large or small contours
            if (area := cv2.contourArea(contour)) <= min_area or area >= max_area:
                continue
            # Filter out artefacts
            if cv2.arcLength(contour, True) == 0:
                continue
            # Calculate spatial moments of the contours
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            # Store as keypoint objects
            keypoints.append(cv2.KeyPoint(float(cx), float(cy), patch_size))
        logger.debug(f"{len(keypoints)} hole centroids detected")
        return sobel, keypoints

    def _register(
        frame_num: int,
        ref: np.ndarray,
        mov: np.ndarray,
        # Preprocessing parameters
        sigma: int,
        window: np.ndarray | None,
        kernel_size: int,
        # Hole detection parameters
        min_area: int,
        max_area: int,
        # ORB detection parameters
        patch_size: int,
        # Instantiated classes and objects
        struct: np.ndarray,  # Structuring element
        orb: cv2.ORB,  # ORB feature matcher instance
        matcher: cv2.BFMatcher,  # Brute-force matcher instance
    ):
        try:
            # Convert RGB to grayscale for image registration
            ref_gray = (
                (
                    # Luma-style weighted sum; avoids cv2.cvtColor copy
                    0.2126 * ref[..., 0] + 0.7152 * ref[..., 1] + 0.0722 * ref[..., 2]
                ).astype(np.float32)
                if ref.ndim == 3
                else ref.astype(np.float32)
            )
            mov_gray = (
                (
                    # Luma-style weighted sum; avoids cv2.cvtColor copy
                    0.2126 * mov[..., 0] + 0.7152 * mov[..., 1] + 0.0722 * mov[..., 2]
                ).astype(np.float32)
                if mov.ndim == 3
                else mov.astype(np.float32)
            )

            args = [
                window,
                sigma,
                kernel_size,
                struct,
                min_area,
                max_area,
                patch_size,
            ]
            ref_sobel, ref_kps = _extract_keypoints(ref_gray, *args)
            mov_sobel, mov_kps = _extract_keypoints(mov_gray, *args)

            # Compute descriptors at the keypoints
            ref_kps, ref_des = orb.compute(ref_sobel, ref_kps)
            mov_kps, mov_des = orb.compute(mov_sobel, mov_kps)
            if ref_des is None or mov_des is None:
                raise RuntimeError("ORB descriptor computation failed")

            # Match the descriptors with 'k' nearest neighbours
            matches = matcher.knnMatch(ref_des, mov_des, k=2)

            # Conduct Lowe ratio test
            # Keep 'm' if it's significantly closer than the second-best match 'n'
            good_matches = [m for m, n in matches if m.distance < 0.9 * n.distance]
            logger.debug("Number of ORB matches after ratio test:", len(good_matches))

            # Estimate the affine transform with RANSAC
            if not len(good_matches) >= 3:
                raise RuntimeError("Not enough matches for affine estimation")
            # Extract (x,y) coordinates of the matched keypoints
            ref_pts = np.float32([ref_kps[m.queryIdx].pt for m in good_matches])
            mov_pts = np.float32([mov_kps[m.trainIdx].pt for m in good_matches])
            # Estimate the affine transformation
            M, inliers = cv2.estimateAffinePartial2D(
                mov_pts,
                ref_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=3,
            )
            logger.debug("Affine matrix:\n", M)
            logger.debug("Inliers:", np.sum(inliers), "/", len(inliers))

            # Apply transform to the original moving image
            aligned = cv2.warpAffine(mov, M, (w, h), flags=cv2.INTER_LINEAR)
            return aligned, frame_num
        except (cv2.error, Exception):
            logger.warning(f"Error registering frame {frame_num}", exc_info=True)
            return mov, frame_num

    # Start of main function
    cv2.setNumThreads(1)
    start_time = time.perf_counter()

    if not (ref := reference_array).shape == (mov := moving_array).shape:
        raise RuntimeError("Input images do not have the same dimensions")

    # Standardise as image stacks before further processing
    is_stack = is_image_stack(ref)
    ref = ref if is_stack else ref[np.newaxis, ...]
    mov = mov if is_stack else mov[np.newaxis, ...]
    z, h, w = ref.shape[:3]
    dtype = ref.dtype

    # Create reusable objects
    window = create_hanning_window(w, h).astype(np.float32) if hanning_window else None
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    orb = cast(cv2.ORB, cv2.ORB_create(nfeatures=5000))
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)  # Brute-force Hamming matcher

    # Create placeholder array for aligned image
    aligned = np.empty(ref.shape, dtype=dtype)

    # Process frames in parallel
    with ThreadPoolExecutor(max_workers=num_procs) as pool:
        futures = [
            pool.submit(
                _register,
                f,
                ref[f],
                mov[f],
                sigma,
                window,
                kernel_size,
                min_area,
                max_area,
                patch_size,
                struct,
                orb,
                matcher,
            )
            for f in range(z)
        ]
        for future in as_completed(futures):
            frame, frame_num = future.result()
            aligned[frame_num] = frame

    # If it wasn't initially a stack, flatten it
    if not is_stack:
        aligned = aligned[0]

    end_time = time.perf_counter()
    logger.debug(f"Completed registration of image stack in {end_time - start_time}")
    return aligned
