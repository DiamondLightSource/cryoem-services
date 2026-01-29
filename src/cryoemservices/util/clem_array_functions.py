"""
Array manipulation functions to process and manipulate the images acquired via the Leica
light microscope.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from enum import Enum
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

import cv2
import numpy as np
import SimpleITK as sitk
from readlif.reader import LifFile
from tifffile import imwrite

# Create logger object to output messages with
logger = logging.getLogger("cryoemservices.util.clem_array_functions")


"""
HELPER CLASSES AND FUNCTIONS
"""


def get_valid_dtypes() -> tuple[str, ...]:
    """
    Use NumPy's in-built dtype dictionary to get list of available, valid dtypes.
    Major dtype groups are "int", "uint", "float", and "complex"
    """
    # Extract list of NumPy dtype classes
    dtype_class_list = list(
        itertools.chain.from_iterable(
            (
                (str(value) for value in np.sctypes[key])
                for key in np.sctypes.keys()
                if key not in ("others",)
            )
        )
    )
    # Use regex matching to get just the dtype portion of the class
    valid_dtypes: list[str] = []
    pattern = r"<[a-z]+ '[a-z]+\.([a-z0-9]+)'>"
    for dtype_class in dtype_class_list:
        match = re.fullmatch(pattern, dtype_class)
        if match is not None:
            dtype = str(np.dtype(match.group(1)))
            valid_dtypes.append(dtype)
    if len(valid_dtypes) == 0:
        logger.error("Unable to get list of NumPy dtypes from NumPy module")
        raise Exception

    return tuple(valid_dtypes)


# Load valid dtypes for future use
valid_dtypes = get_valid_dtypes()
additional_dtype_keywords = (
    "int",
    "uint",
    "float",
    "complex",
    "longdouble",
    "clongdouble",
)


def get_dtype_info(dtype: str):
    """
    Returns NumPy's built-in dtype info object.

    See the docs for:
    numpy.finfo - https://numpy.org/doc/stable/reference/generated/numpy.finfo.html
    numpy.iinfo = https://numpy.org/doc/stable/reference/generated/numpy.iinfo.html
    """

    # Validate input
    if dtype not in valid_dtypes and not any(
        # dtypes without numbers should also be accepted
        dtype == key
        for key in additional_dtype_keywords
    ):
        logger.error(f"{dtype} is not a valid NumPy dtype")
        raise ValueError

    # Load corresponding dictionary from NumPy
    dtype_info = (
        np.iinfo(dtype) if dtype.startswith(("int", "uint")) else np.finfo(dtype)
    )
    return dtype_info


def stretch_image_contrast(
    array: np.ndarray,
    percentile_range: tuple[float, float] = (0.5, 99.5),  # Lower and upper percentiles
    target_dtype: Optional[str] = None,
    debug: bool = False,
) -> np.ndarray:
    """
    Changes the range of pixel values occupied by the data, rescaling it across the
    entirety of the array's bit depth.

    The target dtype should belong to the "int" or "uint" groups, excluding 'int64'
    and 'uint64'. np.int64(np.float64(2**63 - 1)) and np.uint64(np.float64(2**64 - 1))
    cannot be represented exactly in np.float64 or Python's float due to the loss of
    precision at such large values. The leads to overflow errors when trying to cast
    to np.int64 and np.uint64. 32-bit floats and below are thus also not supported as
    input arrays, as the loss of precision occurs even earlier, leading to casting
    issues at smaller NumPy dtypes.
    """

    # Use shorter variable names
    arr: np.ndarray = array

    # Check that input array dtype is supported by NumPy
    dtype = str(arr.dtype)
    if dtype not in valid_dtypes:
        logger.error(f"{dtype} is not a valid NumPy dtype")
        raise ValueError

    # Reject 'float16', 'float32', and 'complex64' input arrays
    if dtype in ("float16", "float32", "complex64"):
        logger.error(f"{dtype} input array not currently supported")
        raise NotImplementedError

    # Handle 'complex' dtypes
    if dtype.startswith("complex"):
        # Accept 'complex' dtypes with no imaginary component
        if np.all(arr.imag == 0):
            dtype = "float64"  # Overwrite initial dtype
            arr = arr.real.astype(dtype)
        # Reject otherwise
        else:
            logger.error(
                "Complex numbers with imaginary components not currently supported"
            )
            raise NotImplementedError

    # Parse target dtype
    if target_dtype is None:
        target_dtype = dtype

    # Validate target_dtype
    if (
        target_dtype not in valid_dtypes
        and target_dtype not in additional_dtype_keywords
    ):
        logger.warning("Invalid target dtype provided; using array's own dtype")
        target_dtype = dtype

    # Reject float, complex, and int64/uint64 target dtypes
    if target_dtype.startswith(("float", "complex")) or target_dtype in (
        "int64",
        "uint64",
    ):
        logger.error(f"{target_dtype} output array not currently supported")
        raise NotImplementedError

    # Get key values
    b_lo: float | int = np.percentile(arr, percentile_range[0])
    b_up: float | int = np.percentile(arr, percentile_range[1])
    diff: float | int = b_up - b_lo
    dtype_info = get_dtype_info(target_dtype)
    vmin = int(dtype_info.min)
    vmax = int(dtype_info.max)

    if debug:
        logger.debug(f"Using {vmax} as maximum array value")

    for f in range(arr.shape[0]):
        # Overwrite outliers and normalise to new range
        frame: np.ndarray = arr[f]
        frame[frame <= b_lo] = b_lo
        frame[frame >= b_up] = b_up

        # DEBUG: Check array properties immediately after truncation
        if debug:
            # Calculate positions of mins and maxes in array
            coords_max = np.unravel_index(np.argmax(frame), frame.shape)
            coords_min = np.unravel_index(np.argmin(frame), frame.shape)

            logger.debug(
                "Frame properties after truncation: \n"
                f"dtype: {frame.dtype} \n"
                f"Shape: {frame.shape} \n"
                f"Min: {frame.min()} \n"
                f"Min coords: {coords_min} \n"
                f"Max: {frame.max()} \n"
                f"Max coords: {coords_max} \n"
            )

        # Normalise differently depending on whether dtype supports negative values
        frame = (
            # Scale between 0 and max positive value if no negative values are present
            (((frame / diff) - (b_lo / diff)) * vmax)
            if (vmin == 0 or b_lo >= 0)
            # Keep 0 as center; scale values by largest scalar present
            else (frame / max(abs(b_lo), abs(b_up)) * vmax)
        )

        # DEBUG: Check array properties immediately after calculation
        if debug:
            # Calculate positions of mins and maxes in array
            coords_max = np.unravel_index(np.argmax(frame), frame.shape)
            coords_min = np.unravel_index(np.argmin(frame), frame.shape)

            logger.debug(
                "Frame properties after contrast stretching: \n"
                f"dtype: {frame.dtype} \n"
                f"Shape: {frame.shape} \n"
                f"Min: {frame.min()} \n"
                f"Min coords: {coords_min} \n"
                f"Max: {frame.max()} \n"
                f"Max coords: {coords_max} \n"
            )

        # Catch negative numbers after contrast stretching
        #   NOTE: For some reason (maybe NumPy inaccuracies when rounding), it's still
        #   possible for some pixel values to go negative
        if vmin == 0 or b_lo >= 0:
            frame[frame <= 0] = 0
        # Catch floats that potentially exceed limits of target dtype
        frame[frame >= vmax] = vmax
        frame[frame <= vmin] = vmin

        if debug:
            # Calculate positions of mins and maxes in array
            coords_max = np.unravel_index(np.argmax(frame), frame.shape)
            coords_min = np.unravel_index(np.argmin(frame), frame.shape)

            logger.debug(
                "Frame properties after catching negative numbers: \n"
                f"dtype: {frame.dtype} \n"
                f"Shape: {frame.shape} \n"
                f"Min: {frame.min()} \n"
                f"Min coords: {coords_min} \n"
                f"Max: {frame.max()} \n"
                f"Max coords: {coords_max} \n"
            )

        # Round values if dtype is integer-based
        #   NOTE: np.round() rounds values that are exactly halfway towards the nearest
        #   even number (e.g. 0.5 -> 0)
        frame = frame.round(0) if dtype.startswith(("int", "uint")) else frame

        if debug:
            # Calculate positions of mins and maxes in array
            coords_max = np.unravel_index(np.argmax(frame), frame.shape)
            coords_min = np.unravel_index(np.argmin(frame), frame.shape)

            logger.debug(
                "Frame properties after rounding: \n"
                f"dtype: {frame.dtype} \n"
                f"Shape: {frame.shape} \n"
                f"Min: {frame.min()} \n"
                f"Min coords: {coords_min} \n"
                f"Max: {frame.max()} \n"
                f"Max coords: {coords_max} \n"
            )

        # Restore original dtype
        frame = frame.astype(target_dtype)

        if debug:
            # Calculate positions of mins and maxes in array
            coords_max = np.unravel_index(np.argmax(frame), frame.shape)
            coords_min = np.unravel_index(np.argmin(frame), frame.shape)

            logger.debug(
                "Frame properties after resetting dtype: \n"
                f"dtype: {frame.dtype} \n"
                f"Shape: {frame.shape} \n"
                f"Min: {frame.min()} \n"
                f"Min coords: {coords_min} \n"
                f"Max: {frame.max()} \n"
                f"Max coords: {coords_max} \n"
            )

        # DEBUG: Check array properties after rounding
        if debug:
            if frame.min() <= vmin:
                logger.debug(
                    "There are values below the minimum bit range for this dtype"
                )
            if frame.max() >= vmax:
                logger.debug("There are values above the max bit range for this dtype")

        # Append to array
        if f == 0:
            arr_new = np.array([frame])
        else:
            arr_new = np.append(arr_new, [frame], axis=0)

    return arr_new


def is_image_stack(
    array: np.ndarray,
) -> bool:
    """
    Helper function to check if an incoming array can be treated as an image stack
    """
    # Check for 2 dimensions or less
    if len(array.shape) == 2:
        return False
    # Check if it's a 2D RGB/RGBA image
    if len(array.shape) == 3 and array.shape[-1] in (3, 4):
        return False

    # Check for valid 3D image stacks
    # Grayscale image stack
    if len(array.shape) == 3 and array.shape[-1] not in (3, 4):
        return True
    # RGB/RGBA image stack
    if len(array.shape) == 4 and array.shape[-1] in (3, 4):
        return True

    # Raise exception for everything else
    raise ValueError(f"Unexpected image shape: {array.shape}")


def put_arrays_in_shared_memory(arrays: list[np.ndarray]):
    """
    Copy a list of NumPy arrays into shared memory blocks and return their references
    """

    shm_blocks: list[SharedMemory] = []
    shm_metadata: list[dict[str, Any]] = []

    for i, arr in enumerate(arrays):
        shm = SharedMemory(create=True, size=arr.nbytes)
        shm_arr = np.ndarray(
            shape=arr.shape,
            dtype=arr.dtype,
            buffer=shm.buf,
        )
        shm_arr[:] = arr  # Allocate array to shared memory block

        shm_blocks.append(shm)
        shm_metadata.append(
            {
                "name": shm.name,
                "shape": arr.shape,
                "dtype": arr.dtype,
            }
        )
    return shm_blocks, shm_metadata


def align_image_to_self(
    array: np.ndarray,
    start_from: Literal["beginning", "middle", "end"] = "middle",
    max_iters: int = 100,
    eps: float = 1e-6,
    use_mask: bool = True,
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

    Returns
    -------
    aligned: np.ndarray
        The aligned image stack as a NumPy array.
    """

    def _register(
        prev: np.ndarray,
        curr: np.ndarray,
        warp_init: np.ndarray,
        mask: np.ndarray | None,
    ):
        warp = warp_init.copy()
        cv2.findTransformECC(
            prev,
            curr,
            warp,
            motionType=cv2.MOTION_EUCLIDEAN,
            criteria=criteria,
            inputMask=mask,
            gaussFiltSize=5,
        )  # Updated in-place
        return warp

    def _warp_frame(
        frame: np.ndarray,
        M: np.ndarray,
    ) -> np.ndarray:
        def _warp(
            frame: np.ndarray,
            M: np.ndarray,
        ) -> np.ndarray:
            return cv2.warpAffine(
                frame,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=vmin,
            )

        if frame.ndim == 2:
            return _warp(frame, M)
        else:
            out = np.empty_like(frame)
            for c in range(frame.shape[-1]):
                out[..., c] = _warp(frame[..., c], M)
            return out

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

    # Extract dimensions, dtype, vmin, and vmax
    z, h, w = array.shape[:3]
    dtype = array.dtype
    vmin, vmax = float(array.min()), float(array.max())
    logger.debug(
        f"shape: {array.shape}\ndtype: {array.dtype}\nvmin: {vmin}\nvmax: {vmax}\n"
    )

    # For RGB images, create a weighted grayscale image to use for registration
    reg = (
        (
            # Luma-style weighted sum; avoids cv2.cvtColor copy
            0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
        ).astype(dtype)
        if array.ndim == 4
        else array.copy()
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
    # Output stack with original dimensions
    aligned = np.empty(array.shape, dtype=dtype)
    aligned[ref_idx] = array[ref_idx]
    # Array of transforms
    transforms = np.zeros((z, 2, 3), dtype=np.float32)

    # Create identity Euclidean transform and assign to reference slice
    I = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )
    transforms[ref_idx] = I
    logger.debug("Allocated placeholder arrays")

    # Create a mask
    mask: np.ndarray | None = None
    if use_mask:
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (w // 2, h // 2), min(h, w) // 3, 255, -1)  # Updated in-place
        logger.debug("Created mask")

    # Set the conditions to use to stop the registration
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max_iters,
        eps,
    )

    # Forward pass
    # 'warp' maps current frame to previous aligned frame
    for i in range(ref_idx + 1, z):
        logger.info(f"Aligning frame {i + 1}/{z}")
        try:
            warp = _register(
                reg[i - 1].astype(np.float32, copy=False),
                reg[i].astype(np.float32, copy=False),
                I,
                mask,
            )
            logger.debug(f"Registered frame {i + 1}")
        except cv2.error:
            logger.warning(
                f"Error registering frame {i}; reverting to using identity matrix"
            )
            warp = I
        new_transform = _make_homogeneous(warp) @ _make_homogeneous(transforms[i - 1])
        transforms[i] = new_transform[:2]
        reg[i] = _warp_frame(reg[i], transforms[i]).astype(dtype, copy=False)
        aligned[i] = _warp_frame(array[i], transforms[i]).astype(dtype, copy=False)
        logger.debug(f"Applied transformation for frame {i + 1}")

    # Backward pass
    for i in range(ref_idx - 1, -1, -1):
        logger.info(f"Aligning frame {i + 1}/{z}")
        try:
            warp = _register(
                reg[i + 1].astype(np.float32, copy=False),
                reg[i].astype(np.float32, copy=False),
                I,
                mask,
            )
            logger.debug(f"Registered frame {i + 1}")
        except cv2.error:
            logger.warning(
                f"Error registering frame {i}; reverting to using identity matrix"
            )
            warp = I
        new_transform = _make_homogeneous(warp) @ _make_homogeneous(transforms[i + 1])
        transforms[i] = new_transform[:2]
        reg[i] = _warp_frame(reg[i], transforms[i]).astype(dtype, copy=False)
        aligned[i] = _warp_frame(array[i], transforms[i]).astype(dtype, copy=False)
        logger.debug(f"Applied transformation for frame {i + 1}")

    if np.issubdtype(dtype, np.integer):
        vmin, vmax = int(vmin), int(vmax)
    np.clip(aligned, a_min=vmin, a_max=vmax, out=aligned)
    return aligned.astype(dtype, copy=False)


def align_image_to_reference(
    reference_array: np.ndarray, moving_array: np.ndarray, downsample_factor: int = 2
) -> np.ndarray:
    """
    Uses SimpleITK's image registration methods to align images to a reference.

    Currently, this method works poorly for defocused images, which can lead to a
    lot of jitter in the beginning and tail frames if the images being aligned are
    a defocus series.
    """
    # Restrict number of threads used by SimpleITK
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

    # Get initial dtype
    if reference_array.dtype != moving_array.dtype:
        logger.error("The image stacks provided do not have the same dtype")
        raise ValueError
    dtype = moving_array.dtype
    vmin, vmax = moving_array.min(), moving_array.max()

    if reference_array.shape != moving_array.shape:
        logger.error("The image stacks provided are not of the same shape")

    # Standardise frames and stacks as stacks
    if was_a_stack := is_image_stack(moving_array):
        num_frames = moving_array.shape[0]
    else:
        num_frames = 1
        reference_array = reference_array[np.newaxis, ...]
        moving_array = moving_array[np.newaxis, ...]

    # Convert to arrays to SITK objects
    fixed_sitk = sitk.Cast(sitk.GetImageFromArray(reference_array), sitk.sitkFloat32)
    moving_sitk = sitk.Cast(sitk.GetImageFromArray(moving_array), sitk.sitkFloat32)

    # Pre-allocate output NumPy array
    aligned = np.empty(moving_array.shape, dtype=dtype)

    prev_transform = None
    for f in range(num_frames):
        logger.debug(f"Aligning frame {f + 1}/{num_frames}")
        # Extract frame (SITK uses x, y, z ordering)
        fixed_frame = fixed_sitk[:, :, f]
        moving_frame = moving_sitk[:, :, f]

        # Downsample frames for registration
        if downsample_factor > 1:
            fixed_small = sitk.Shrink(fixed_frame, [downsample_factor] * 2)
            moving_small = sitk.Shrink(moving_frame, [downsample_factor] * 2)
        else:
            fixed_small, moving_small = fixed_frame, moving_frame

        # Set up registration method
        registration = sitk.ImageRegistrationMethod()
        registration.SetInterpolator(sitk.sitkLinear)

        # Set the metric to use
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
        registration.SetMetricSamplingPercentage(0.05)  # Sample 5% of pixels
        registration.SetMetricSamplingStrategy(registration.RANDOM)

        # Set the optimiser to use
        # registration.SetOptimizerAsGradientDescent(
        #     learningRate=0.5,
        #     numberOfIterations=300,
        #     convergenceMinimumValue=1e-6,
        #     convergenceWindowSize=10,
        # )
        registration.SetOptimizerAsGradientDescentLineSearch(
            learningRate=1.0,
            numberOfIterations=200,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=5,
        )
        # registration.SetOptimizerAsAmoeba(
        #     simplexDelta=1.0,
        #     numberOfIterations=300,
        #     parametersConvergenceTolerance=1e-8,
        #     functionConvergenceTolerance=1e-4,
        #     withRestarts=False,
        # )
        # registration.SetOptimizerAsOnePlusOneEvolutionary(
        #     epsilon=1e-6,
        #     initialRadius=1.0,
        #     numberOfIterations=300,
        #     growthFactor=-1.0,
        #     shrinkFactor=-1.0,
        # )
        registration.SetOptimizerScalesFromIndexShift()

        # Register over a multi-resolution pyramid
        registration.SetShrinkFactorsPerLevel([4, 2])
        registration.SetSmoothingSigmasPerLevel([2, 1])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Initilaise transform or reuse previous one, if available
        initial_transform = prev_transform or sitk.CenteredTransformInitializer(
            fixed_small,
            moving_small,
            sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        registration.SetInitialTransform(initial_transform, inPlace=False)

        # Execute registration on downsampled images
        final_transform = registration.Execute(fixed_small, moving_small)
        prev_transform = final_transform

        # Apply transform to full resolution frame
        aligned_frame = sitk.GetArrayFromImage(
            sitk.Resample(
                moving_frame,
                transform=final_transform,
                interpolator=sitk.sitkLinear,
                outputPixelType=sitk.sitkFloat32,
            )
        )

        # Clip and round to original dtype
        np.clip(aligned_frame, a_min=vmin, a_max=vmax, out=aligned_frame)
        if np.issubdtype(dtype, np.integer):
            np.rint(aligned_frame, out=aligned_frame)
        aligned[f] = aligned_frame.astype(dtype, copy=False)

    # If the image was not initially a stack, flatten it
    if not was_a_stack:
        aligned = aligned[0]

    return aligned


def is_grayscale_image(array: np.ndarray):
    """
    Helper function to check if an incoming array can be treated as a grayscale image
    or image stack
    """
    # Rule out RGB/RGBA images first
    if len(array.shape) in (3, 4) and array.shape[-1] in (3, 4):
        return False

    # Check for 2D grayscale
    if len(array.shape) == 2:
        return True
    # Check for grayscale image stacks
    if len(array.shape) == 3:
        return True

    # Raise error otherwise
    raise ValueError(f"Unexpected image shape: {array.shape}")


class LUT(Enum):
    """
    3-channel color lookup tables to use when colorising image stacks. They are binary
    values, making them potentially compatible with images of any bit depth.
    """

    # (R, G, B)
    red = (1, 0, 0)
    green = (0, 1, 0)
    blue = (0, 0, 1)
    cyan = (0, 1, 1)
    magenta = (1, 0, 1)
    yellow = (1, 1, 0)
    gray = (1, 1, 1)


def convert_to_rgb(
    array: np.ndarray,
    color: str,
) -> np.ndarray:
    # Set up variables
    try:
        lut = LUT[color.lower()].value
    except KeyError:
        raise KeyError(f"No lookup table found for the colour {color!r}")

    # Pre-allocate empty output array
    out = np.empty((array.shape + (3,)), dtype=array.dtype)

    # Write to channels directly
    for c, value in enumerate(lut):
        np.multiply(array, value, out=out[..., c], casting="unsafe")

    return out


def flatten_image(
    array: np.ndarray,
    mode: Literal["mean", "min", "max"] = "mean",
) -> np.ndarray:
    # Flatten along first (outermost) axis
    axis = 0
    dtype = array.dtype
    out: np.ndarray
    if mode in ("min", "max"):
        # Pre-allocate empty output array
        out = np.empty(array.shape[1:], dtype=dtype)
        if mode == "min":
            np.minimum.reduce(array, axis=axis, out=out)
        else:
            np.maximum.reduce(array, axis=axis, out=out)
    elif mode == "mean":
        # Use float32 when calculating integer arrays to keep footprint small
        is_int_dtype = np.issubdtype(dtype, np.integer)
        out = array.mean(axis=axis, dtype=np.float32 if is_int_dtype else dtype)
        if is_int_dtype:
            np.rint(out, out=out)
            out = out.astype(dtype, copy=False)
    # Raise error if the mode provided is incorrect
    else:
        logger.error(f"{mode} is not a valid image flattening option")
        raise ValueError

    return out


def merge_images(
    arrays: np.ndarray | list[np.ndarray],
) -> np.ndarray:
    """
    Takes a list of arrays and returns a composite image averaged across every image in
    the list.
    """

    # Check that an array was provided
    if not arrays:
        raise ValueError("No input arrays provided")

    # Standardise to a list of arrays
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    # Validate that arrays have the same shape
    shape, dtype = arrays[0].shape, arrays[0].dtype
    for arr in arrays:
        if arr.shape != shape:
            logger.error("Input arrays do not have the same shape")
            raise ValueError
        if arr.dtype != dtype:
            logger.error("Input arrays do not have the same dtype")
            raise ValueError

    is_int_dtype = np.issubdtype(dtype, np.integer)
    # Add values in float32 for int dtypes
    out = np.zeros(
        shape,
        dtype=np.float32 if is_int_dtype else dtype,
    )
    for arr in arrays:
        out += arr
    out /= len(arrays)
    if is_int_dtype:
        np.rint(out, out=out)
        out = out.astype(dtype, copy=False)

    return out


@dataclass(frozen=True)
class ImageLoader(Protocol):
    """
    This a type hinting stub used by the CLEM array functions below to represent
    the image loader classes for LIF and TIFF files. These classes will contain
    a 'load()' function that uses the appropriate packages to load and return
    the image as a NumPy array.
    """

    def load(self) -> np.ndarray: ...


@dataclass(frozen=True)
class LIFImageLoader(ImageLoader):
    lif_file: Path
    scene_num: int
    channel_num: int
    frame_num: int
    tile_num: int

    def load(self) -> np.ndarray:
        return np.asarray(
            LifFile(self.lif_file)
            .get_image(self.scene_num)
            .get_frame(
                z=self.frame_num,
                c=self.channel_num,
                m=self.tile_num,
            )
        ).copy()


@dataclass(frozen=True)
class TIFFImageLoader(ImageLoader):
    tiff_file: Path

    def load(self) -> np.ndarray:
        return np.asarray(cv2.imread(self.tiff_file, flags=cv2.IMREAD_UNCHANGED))


@dataclass(frozen=True)
class HistogramResult:
    data: np.ndarray | None
    error: dict | None


def get_histogram(
    image_loader: ImageLoader,
    bins: int = 65536,
    pixel_sampling_step: int = 4,
):
    """
    Helper function that opens an image and compiles a histogram of the intensity
    distribution. This is then used to estimate suitable values to normalise the
    images with.

    Parameters
    ----------
    image_loader: ImageLoader
        ImageLoader class that returns the image as a NumPy array.

    bins: int
        Number of bins to use for the histogram. The default is 65536, which is a
        16-bit channel.

    pixel_sampling_step: int
        The number of pixels along each axes of the array to skip during sampling.
        Improves speed while still returning a relatively representative statistical
        sample of the image.

    Returns
    -------
    histogram: np.ndarray
        A one-dimensional NumPy array containing the counts
    """
    try:
        arr = image_loader.load()[::pixel_sampling_step, ::pixel_sampling_step]
        return HistogramResult(
            data=np.bincount(arr.ravel(), minlength=bins), error=None
        )
    # If it errors, return the file it errored on and why
    except Exception as e:
        return HistogramResult(
            data=None,
            error={
                "data": asdict(image_loader),
                "error": str(e),
            },
        )


def get_percentiles(
    image_loaders: list[ImageLoader],
    percentiles: tuple[float, float] = (1, 99),
    bins: int = 65536,
    pixel_sampling_step: int = 4,
    num_procs: int = 1,
) -> tuple[int | float, int | float]:
    """
    Helper function to sample the list of images provided and globally determine lower
    and upper percentile values to use when normalising and converting the images to
    8-bit NumPy arrays.

    Due to the I/O-heavy and statistical nature of this step, multithreading has been
    implemented in addition pixel sampling so that relatively representative values
    can be estimated rapidly.

    Parameters
    ----------
    image_loaders: ImageLoader
        List of ImageLoader objects from which images can be extracted as NumPy arrays.

    percentiles: tuple[float, float]
        The lower and upper percentiles to be extracted. Used to normalise the contrast
        before converting the image to an 8-bit NumPy array.

    bins: int
        Number of bins to group the image into. The default is 65536 bins, which is a
        16-bit channel.

    pixel_sampling_step: int
        Number of pixels along each of the axes to skip when sampling the images. The
        default is 4.

    num_procs: int
        Number of threads to use when running this function

    Returns
    -------
    values: tuple[int | float, int | float]
        The pixel values corresponding to the lower and upper percentiles specified.
    """

    def _percentile(p: float):
        return np.searchsorted(cumsum, p / 100 * total)

    p_lo, p_hi = percentiles
    hist = np.zeros(bins)

    with ThreadPoolExecutor(max_workers=num_procs) as pool:
        futures = [
            pool.submit(
                get_histogram,
                image_loader,
                bins,
                pixel_sampling_step,
            )
            for image_loader in image_loaders
        ]
        for future in as_completed(futures):
            result = future.result()
            if result.data is not None:
                hist += result.data
            else:
                logger.warning(
                    "Failed to get histogram for the following image: \n"
                    f"{json.dumps(result.error, indent=2, default=str)}"
                )

    # Calculate the cumulative sum
    cumsum = np.cumsum(hist)
    total = cumsum[-1]

    return _percentile(p_lo), _percentile(p_hi)


@dataclass(frozen=True)
class LoadImageResult:
    data: np.ndarray | None
    frame_num: int | None
    error: dict | None


def load_and_convert_image(
    image_loader: ImageLoader,
    frame_num: int,
    vmin: int | float,
    vmax: int | float,
):
    """
    Helper function that loads images and converts them into an 8-bit NumPy array.

    Parameters
    ----------
    image_loader: ImageLoader
        An ImageLoader dataclass which opens the specified image file and returns it
        as a NumPy array.

    frame_num: int
        The frame number the image corresponds to. This is forwarded as part of the
        returned dataclass so that the array can be correctly allocated to its frame.

    vmin: int | float
        The minimum pixel value to clip the array at before normalising to an 8-bit
        NumPy array.

    vmax: int | float
        The maximum pixel value to clip the array at before normalising to an 8-bit
        NumPy array.

    Returns
    -------
    image_frame: np.ndarray
        A single image returned as a 2D image in either grayscale or RGB.
    """

    try:
        arr = image_loader.load()
        scale = 255 / (vmax - vmin)  # Downscale to 8-bit
        np.clip(arr, a_min=vmin, a_max=vmax, out=arr)
        np.subtract(arr, vmin, out=arr, casting="unsafe")
        np.multiply(arr, scale, out=arr, casting="unsafe")
        return LoadImageResult(
            data=arr.astype(np.uint8),
            frame_num=frame_num,
            error=None,
        )
    except Exception as e:
        return LoadImageResult(
            data=None,
            frame_num=None,
            error={
                "data": asdict(image_loader),
                "error": str(e),
            },
        )


@dataclass(frozen=True)
class ResizeTileResult:
    data: np.ndarray | None
    frame_num: int | None
    x0: int | None
    x1: int | None
    y0: int | None
    y1: int | None
    error: dict | None


def resize_tile(
    image_loader: ImageLoader,
    frame_num: int,
    tile_extent: tuple[float, float, float, float],
    parent_extent: tuple[float, float, float, float],
    parent_pixel_size: float,
    parent_shape: tuple[int, int],  # NumPy row-column order
    vmin: int | float,
    vmax: int | float,
):
    """
    Helper function that loads the image, uses the provided stage information to
    calculate its size on the wider frame it belongs to, resizes and converts it
    to an 8-bit NumPy array, then returns it along with the coordinates for the
    space it occupies in the frame.

    Parameters
    ----------
    image_loader: ImageLoader
        An ImageLoader class with a 'load()' function that will return a NumPy array.

    frame_num: int
        The frame number the image corresponds to

    tile_extent: tuple[float, float, float, float]
        The span in real space that the image occupies. This takes a tuple of 4 values
        in the order x0, x1, y0, y1, which correspond to the left-most, right-most,
        upper-most, and bottom-most boundary of the image

    parent_extent: tuple[float, float, float, float]
        The extent of the parent image to which this tile belongs to. It takes a tuple
        of 4 values in the same format as the tile extent.

    parent_pixel_size: float
        The pixel size of the parent image this tile will be mapped to.

    parent_shape: tuple[int, int]
        The shape of the parent image this tile will be mapped to.

    vmin: int | float
        The minimum pixel value to clip the image to before converting to 8-bit.

    vmax: int | float
        The maximum pixel value to clip the image to before converting to 8-bit.
    """
    try:
        # Find array size in parent frame the tile corresponds to
        x0, x1, y0, y1 = tile_extent
        px0, px1, py0, py1 = parent_extent
        parent_y_pixels, parent_x_pixels = parent_shape
        tile_x_pixels = int(round((x1 - x0) / parent_pixel_size))
        tile_y_pixels = int(round((y1 - y0) / parent_pixel_size))

        # Position on master image (Use top left as reference)
        pos_x = int(round((x0 - px0) / (px1 - px0) * parent_x_pixels))
        pos_y = int(round((y0 - py0) / (py1 - py0) * parent_y_pixels))

        # Load image and resize
        img = image_loader.load()
        resized = cv2.resize(
            img,
            dsize=(tile_x_pixels, tile_y_pixels),
            interpolation=cv2.INTER_AREA,
        )
        # Normalise to 8-bit
        scale = 255 / (vmax - vmin)
        np.clip(resized, vmin, vmax, out=resized)
        np.subtract(resized, vmin, out=resized, casting="unsafe")
        np.multiply(resized, scale, out=resized, casting="unsafe")
        resized = resized.astype(np.uint8)

        return ResizeTileResult(
            data=resized,
            frame_num=frame_num,
            x0=pos_x,
            x1=pos_x + tile_x_pixels,
            y0=pos_y,
            y1=pos_y + tile_y_pixels,
            error=None,
        )
    except Exception as e:
        return ResizeTileResult(
            data=None,
            frame_num=None,
            x0=None,
            x1=None,
            y0=None,
            y1=None,
            error={
                "data": asdict(image_loader),
                "frame_num": frame_num,
                "tile_extent": tile_extent,
                "parent_extent": parent_extent,
                "parent_pixel_size": parent_pixel_size,
                "parent_shape": parent_shape,
                "vmin": vmin,
                "vmax": vmax,
                "error": str(e),
            },
        )


def write_stack_to_tiff(
    array: np.ndarray,
    save_dir: Path,
    file_name: str,
    # Resolution information
    x_res: Optional[float] = None,
    y_res: Optional[float] = None,
    z_res: Optional[float] = None,
    units: Optional[str] = None,
    # Array properties
    axes: Optional[str] = None,
    image_labels: Optional[list[str] | str] = None,
    # Colour properties
    photometric: Optional[str] = None,  # Valid options listed below
    color_map: Optional[np.ndarray] = None,
    extended_metadata: Optional[str] = None,  # Stored as an extended string
) -> Path:
    """
    Writes the NumPy array as a calibrated, ImageJ-compatible TIFF image stack.
    """
    # Get resolution
    if z_res is not None:
        z_size = (1 / z_res) if z_res > 0 else float(0)
    else:
        z_size = None

    if x_res is not None and y_res is not None:
        resolution = (x_res * 10**6 / 10**6, y_res * 10**6 / 10**6)
    else:
        resolution = None

    resolution_unit = 1 if units is not None else None
    use_bigtiff = (
        False
        if not is_image_stack(array) or (is_image_stack(array) and array.shape[0] == 1)
        else True
    )

    # Get photometric
    valid_photometrics = (
        "minisblack",
        "miniswhite",
        "rgb",
        "ycbcr",  # Y: Luminance | Cb: Blue chrominance | Cr: Red chrominance
        "palette",
    )
    if photometric is not None and photometric not in valid_photometrics:
        photometric = None
        logger.warning("Incorrect photometric value provided; defaulting to 'None'")

    # Process extended metadata
    if extended_metadata is None:
        extended_metadata = ""

    # Save as a greyscale TIFF
    save_name = save_dir.joinpath(file_name + ".tiff")

    # With 'bigtiff=True', they have to be pure Python class instances
    imwrite(
        save_name,
        array,
        bigtiff=True,
        # Array properties,
        shape=array.shape,
        dtype=str(array.dtype),
        resolution=resolution,
        resolutionunit=resolution_unit,
        # Colour properties
        photometric=photometric,  # Greyscale image
        colormap=color_map,
        # ImageJ compatibility
        imagej=use_bigtiff,
        metadata={
            "axes": axes,
            "unit": units,
            "spacing": z_size,
            "loop": False,
            # Coerce NumPy's min() and max() return values into Python floats
            # Follow ImageJ's precision level
            "min": round(float(array.min()), 1),
            "max": round(float(array.max()), 1),
            "Info": extended_metadata,
            "Labels": image_labels,
        },
    )
    logger.info(f"{file_name} image saved as {save_name}")

    return save_name
