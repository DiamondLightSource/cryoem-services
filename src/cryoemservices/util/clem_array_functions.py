"""
Array manipulation functions to process and manipulate the images acquired via the Leica
light microscope.
"""

from __future__ import annotations

import itertools
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import SimpleITK as sitk
from pystackreg import StackReg
from SimpleITK.SimpleITK import Image
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


def estimate_int_dtype(array: np.ndarray, bit_depth: Optional[int] = None) -> str:
    """
    Finds the smallest NumPy integer dtype that can contain the range of values
    present in a particular image.
    """

    # Define helper sub-functions
    def _round_from_zero(x: float) -> int:
        """
        Round values AWAY from zero.
        """
        x_round = int(np.sign(x) * np.ceil(abs(x)))
        return x_round

    def _by_bit_depth(
        array: np.ndarray,
        dtype_group: str,
        bit_depth: int,
    ) -> Optional[str]:

        # Set up variables
        arr = array
        dtype_subset = [
            dtype for dtype in valid_dtypes if dtype.startswith(dtype_group)
        ]

        # Get dtypes with bit values greater than the provided one
        bit_list: list[int] = []
        for dtype in dtype_subset:
            match = re.fullmatch("[a-z]+([0-9]+)", dtype)
            if match is not None:
                value = int(match.group(1))
                if value >= bit_depth:
                    bit_list.append(value)
            else:
                continue
        if len(bit_list) == 0:
            logger.error("No suitable dtypes found based on provided bit depth")
            return None

        # Use the minimum viable dtype
        dtype_final = f"{dtype_group}{min(bit_list)}"

        # Raise error if dtype calculated using provided bit depth can't accommodate array
        dtype_info = get_dtype_info(dtype_final)
        vmin = int(dtype_info.min)
        vmax = int(dtype_info.max)
        # Use the rounded, int values of the array to ensure no rounding errors
        arr_min = _round_from_zero(arr.min())
        arr_max = _round_from_zero(arr.max())
        if (arr_max > vmax) or (arr_min < vmin):
            logger.warning(
                "Array values still exceed those supported by the estimated dtype"
            )
            return None

        # Return estimated dtype otherwise
        return dtype_final

    def _by_array_values(array: np.ndarray, dtype_group: str) -> Optional[str]:

        # Set up variables
        arr = array

        # Get the list of dtypes that can accommodate the array's contents
        dtype_subset = []
        for dtype in valid_dtypes:
            if dtype.startswith(dtype_group):
                dtype_info = get_dtype_info(dtype)
                vmin = int(dtype_info.min)
                vmax = int(dtype_info.max)
                arr_min = (
                    int(arr.min())
                    if str(arr.dtype).startswith(("int", "uint"))
                    else _round_from_zero(arr.min())
                )
                arr_max = (
                    int(arr.max())
                    if str(arr.dtype).startswith(("int", "uint"))
                    else _round_from_zero(arr.max())
                )
                if vmax >= arr_max and vmin <= arr_min:
                    dtype_subset.append(dtype)
                else:
                    continue
        # Get the numerical portion of the suitable dtypes
        bit_list: list[int] = []
        for dtype in dtype_subset:
            match = re.fullmatch("[a-z]+([0-9]+)", dtype)
            if match is not None:
                value = int(match.group(1))
                bit_list.append(value)
            else:
                continue
        if len(bit_list) == 0:
            logger.error(
                "No suitable dtypes found that can accommodate the array's values"
            )
            return None
        # Use the smallest value
        dtype_final = f"{dtype_group}{min(bit_list)}"

        return dtype_final

    # Set up variables
    arr = array
    dtype = str(arr.dtype)

    # Validate initial dtype (this should never be triggered, in principle)
    if dtype not in valid_dtypes:
        logger.error(f"{dtype} is not a valid NumPy dtype")
        raise ValueError

    # Reject complex numbers with imaginary components
    if dtype.startswith("complex") and np.any(arr.imag != 0):
        logger.error(
            "Complex numbers with imaginary components not currently supported"
        )
        raise NotImplementedError
    else:
        arr = arr.real

    # Use "int" if negative values are present, and "uint" if not
    dtype_group = "uint" if arr.min() >= 0 else "int"

    estimate: Optional[str] = None
    # Make an estimate using the provided bit depth if set
    estimate = (
        _by_bit_depth(
            array=arr,
            dtype_group=dtype_group,
            bit_depth=bit_depth,
        )
        if bit_depth is not None
        else None
    )

    # Estimate using array depth instead
    estimate = (
        _by_array_values(array=arr, dtype_group=dtype_group)
        if estimate is None
        else estimate
    )

    if estimate is None:
        raise ValueError("Unable to find an appropriate dtype for the array")
    else:
        return estimate


def convert_array_dtype(
    array: np.ndarray,
    target_dtype: str,
    initial_dtype: Optional[str] = None,
) -> np.ndarray:
    """
    Rescales the pixel values of the array to fit within the allowed range of the
    desired array dtype while preserving the existing contrast.

    The target dtype should belong to the "int" or "uint" groups, excluding 'int64'
    and 'uint64'. np.int64(np.float64(2**63 - 1)) and np.uint64(np.float64(2**64 - 1))
    cannot be represented exactly in np.float64 or Python's float due to the loss of
    precision at such large values. The leads to overflow errors when trying to cast
    to np.int64 and np.uint64. 32-bit floats and below are thus also not supported as
    input arrays, as the loss of precision occurs even earlier, leading to casting
    issues at smaller NumPy dtypes.
    """

    # Use shorter names for variables
    arr: np.ndarray = array
    dtype_final = target_dtype
    dtype_init = initial_dtype

    # Parse target dtype provided
    if dtype_final not in valid_dtypes and not any(
        dtype_final == key for key in additional_dtype_keywords
    ):
        logger.error(f"{dtype_final} is not a valid NumPy dtype")
        raise ValueError

    # Reject float, complex, and int64/uint64 as target dtypes
    if dtype_final.startswith(("float", "complex")) or dtype_final in (
        "int64",
        "uint64",
    ):
        logger.error(f"{dtype_final} output array not currently supported")
        raise NotImplementedError

    # Parse initial dtype estimate provided
    # Estimate initial corresponding integer dtype if None provided
    if dtype_init is None:
        dtype_init = estimate_int_dtype(arr)

    # Estimate dtype if invalid one provided
    if dtype_init not in valid_dtypes and dtype_init not in additional_dtype_keywords:
        logger.warning(
            f"{dtype_init} is not a valid NumPy dtype; estimating dtype from the array"
        )
        dtype_init = estimate_int_dtype(arr)

    # Find closest equivalent integer dtype that encompasses floats
    if dtype_init.startswith(("float", "complex")):
        logger.warning(
            "Unsupported dtype estimate provided; estimating dtype from the array"
        )
        dtype_init = estimate_int_dtype(arr)

    # Validate that the initial array is supported
    # Reject float16, float32, complex64 input arrays
    if str(arr.dtype) in ("float16", "float32", "complex64"):
        logger.error(f"{arr.dtype} input array not currently supported")
        raise NotImplementedError

    # Accept 'complex' dtypes if no imaginary component
    if str(arr.dtype).startswith("complex") and np.any(arr.imag != 0):
        logger.error(
            "Complex numbers with imaginary components not currently supported"
        )
        raise NotImplementedError
    else:
        arr = arr.real
        dtype_init = estimate_int_dtype(arr)

    # Get max supported values of initial and final arrays
    dtype_info_init = get_dtype_info(dtype_init)
    min_init = int(dtype_info_init.min)
    max_init = int(dtype_info_init.max)
    range_init = max_init - min_init

    dtype_info_final = get_dtype_info(dtype_final)
    min_final = int(dtype_info_final.min)
    max_final = int(dtype_info_final.max)
    range_final = max_final - min_final

    # Rescale
    for f in range(arr.shape[0]):
        # Map from old range to new range without exceeding maximum bit depth
        frame: np.ndarray = (
            ((arr[f] / range_init) - (min_init / range_init)) * range_final
        ) + min_final

        # Catch numbers exceeding thresholds when going between dtypes
        if frame.min() < min_final:
            logger.warning(
                f"{np.sum(frame < min_final)} values below allowed target minimum value"
            )
            frame[frame < min_final] = min_final
        if frame.max() > max_final:
            logger.warning(
                f"{np.sum(frame > max_final)} values above allowed target maximum value"
            )
            frame[frame > max_final] = max_final

        # Preserve dtype and round values if dtype is integer-based
        frame = frame.round(0) if dtype_final.startswith(("int", "uint")) else frame
        frame = frame.astype(dtype_final)

        # Append to array
        if f == 0:
            arr_new = np.array([frame])
        else:
            arr_new = np.append(arr_new, [frame], axis=0)

    return arr_new


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


def align_image_to_self(
    array: np.ndarray,
    start_from: Literal["beginning", "middle", "end"] = "beginning",
) -> np.ndarray:
    """
    Use PyStackReg to correct for drift in an image stack.
    """
    # Record initial dtype and intensities
    dtype = str(array.dtype)
    vmin: int | float = array.min()
    vmax: int | float = array.max()

    # Check if an image or a stack has been passed to the function
    shape = array.shape
    if len(shape) <= 2 or (
        len(shape) == 3 and shape[-1] in (3, 4)  # Check for 2D RGB or RGBA images
    ):
        logger.warning(
            f"Image provided likely not an image stack (has dimensions {shape});"
            "returning original array"
        )
        return array

    # Set up StackReg object to facilitate image processing
    sr = StackReg(StackReg.RIGID_BODY)

    # Standard method for aligning images
    if start_from == "beginning":
        aligned = np.array(sr.register_transform_stack(array, reference="previous"))

    # Align from the middle
    # Useful for aligning defocus series, where the plane of focus is in the middle
    elif start_from == "middle":

        # Align both halves independently
        idx = len(array) // 2  # Floor division
        aligned_front = np.flip(
            sr.register_transform_stack(
                np.flip(array[: idx + 1], axis=0), reference="previous"
            ),
            axis=0,
        )
        aligned_back = np.array(
            sr.register_transform_stack(array[idx:], reference="previous")
        )

        # Rejoin halves
        aligned = np.concatenate((aligned_front[:idx], aligned_back), axis=0)

    # Align from the end
    elif start_from == "end":
        aligned = np.flip(
            sr.register_transform_stack(np.flip(array, axis=0), reference="previous"),
            axis=0,
        )
    else:
        logger.error(f"Invalid parameter {start_from!r} provided")
        raise ValueError

    # Crop intensities that have been shifted outside of the initial range
    aligned[aligned < vmin] = vmin
    aligned[aligned > vmax] = vmax

    # Revert array back to initial dtype
    aligned = aligned.astype(dtype) if str(aligned.dtype) != dtype else aligned

    return aligned


def align_image_to_reference(
    reference_array: np.ndarray,
    moving_array: np.ndarray,
) -> np.ndarray:
    """
    Uses SimpleITK's image registration methods to align images to a reference.

    Currently, this method works poorly for defocused images, which can lead to a
    lot of jitter in the beginning and tail frames if the images being aligned are
    a defocus series.
    """
    # Get initial dtype
    if str(reference_array.dtype) != str(moving_array.dtype):
        logger.error("The image stacks provided do not have the same dtype")
        raise ValueError
    dtype = str(moving_array.dtype)
    vmin: int | float = moving_array.min()
    vmax: int | float = moving_array.max()

    # Check array shape
    if reference_array.shape != moving_array.shape:
        logger.error("The image stacks provided do not have the same dimensions")
        raise ValueError

    # SimpleITK's image registration prefers to work with floats
    fixed: Image = sitk.Cast(sitk.GetImageFromArray(reference_array), sitk.sitkFloat64)
    moving: Image = sitk.Cast(sitk.GetImageFromArray(moving_array), sitk.sitkFloat64)

    # Check if an image or a stack has been passed to the function, and handle accordingly
    shape: tuple[int, ...] = (
        fixed.GetSize()
    )  # In (x, y, z) order; SITK's Size object omits the RGB dimension, if present
    num_frames = 1 if len(shape) == 2 else shape[-1]

    # Iterate through stacks, aligning corresponding frames
    aligned_frames: list[Image] = []
    for f in range(num_frames):
        fixed_frame: Image
        moving_frame: Image
        if len(shape) == 2:
            fixed_frame = fixed
            moving_frame = moving
        else:
            fixed_frame = fixed[:, :, f]
            moving_frame = moving[:, :, f]

        # Set up the registration parameters
        registration = sitk.ImageRegistrationMethod()

        # Choose the metric
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)

        # Choose the type of interpolation to use
        registration.SetInterpolator(sitk.sitkLinear)

        # Register over a multi-resolution pyramid
        registration.SetShrinkFactorsPerLevel([4, 2, 1])
        registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        # registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Choose the type of optimiser to use
        # registration.SetOptimizerAsGradientDescent(
        #     learningRate=0.5,
        #     numberOfIterations=300,
        #     convergenceMinimumValue=1e-6,
        #     convergenceWindowSize=10,
        # )
        registration.SetOptimizerAsGradientDescentLineSearch(
            learningRate=1.0,
            numberOfIterations=300,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
            lineSearchLowerLimit=0,
            lineSearchUpperLimit=5.0,
            lineSearchMaximumIterations=20,
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

        # Initialise a rigid-body transform
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_frame,
            moving_frame,
            sitk.Euler2DTransform(),  # Restricted to in-plane translation and rotation
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        registration.SetInitialTransform(initial_transform, inPlace=False)

        # Register the frame
        final_transform = registration.Execute(fixed_frame, moving_frame)

        # Transform the moving frame
        aligned_frame: Image = sitk.Resample(
            moving_frame,
            transform=final_transform,
            interpolator=sitk.sitkLinear,
            defaultPixelValue=0.0,
            outputPixelType=moving_frame.GetPixelID(),
            useNearestNeighborExtrapolator=True,
        )
        aligned_frames.append(aligned_frame)

    # Recombine as a single image stack
    aligned: np.ndarray = (
        sitk.GetArrayFromImage(aligned_frames[0])
        if len(aligned_frames) == 1
        else sitk.GetArrayFromImage(sitk.JoinSeries(aligned_frames))
    )

    # Crop values that exceed initial range after transformation
    aligned[aligned < vmin] = vmin
    aligned[aligned > vmax] = vmax
    aligned = aligned.astype(dtype) if str(aligned.dtype) != dtype else aligned

    return aligned


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
    arr: np.ndarray = array
    dtype = str(arr.dtype)
    try:
        lut = LUT[color.lower()].value
    except KeyError:
        raise KeyError(f"No lookup table found for the colour {color!r}")

    # Calculate pixel values for each channel
    arr_list: list[np.ndarray] = [arr * c for c in lut]

    # Stack arrays along last axis and preserve dtype
    arr_new = np.stack(arr_list, axis=-1).astype(dtype)

    return arr_new


def flatten_image(
    array: np.ndarray,
    mode: Literal["mean", "min", "max"] = "mean",
) -> np.ndarray:

    # Flatten along first (outermost) axis
    axis = 0
    if mode == "min":
        arr_new: np.ndarray = array.min(axis=axis)
    elif mode == "max":
        arr_new = array.max(axis=axis)
    elif mode == "mean":
        dtype = str(array.dtype)
        arr_new = array.mean(axis=axis)
        arr_new = arr_new.round(0) if dtype.startswith(("int", "uint")) else arr_new
        arr_new = arr_new.astype(dtype)

    # Raise error if the mode provided is incorrect
    else:
        logger.error(f"{mode} is not a valid image flattening option")
        raise ValueError

    return arr_new


def merge_images(
    arrays: np.ndarray | list[np.ndarray],
) -> np.ndarray:
    """
    Takes a list of arrays and returns a composite image averaged across every image in
    the list.
    """

    # Standardise to a list of arrays
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    # Validate that arrays have the same shape
    if len({arr.shape for arr in arrays}) > 1:
        logger.error("Input arrays do not have the same shape")
        raise ValueError

    # Validate that arrays have the same dtype
    if len({str(arr.dtype) for arr in arrays}) > 1:
        logger.error("Input arrays do not have the same dtype")
        raise ValueError
    dtype = str(arrays[0].dtype)

    # Calculate average across all arrays
    arr_new: np.ndarray = np.mean(arrays, axis=0)

    # Revert to input array dtype
    arr_new = arr_new.round(0) if dtype.startswith(("int", "uint")) else arr_new
    arr_new = arr_new.astype(dtype)

    return arr_new


def preprocess_img_stk(
    array: np.ndarray,
    target_dtype: str = "uint8",
    initial_dtype: Optional[str] = None,
    adjust_contrast: Optional[str] = None,
) -> np.ndarray:
    """
    Preprocessing routine for the image stacks extracted from raw data, rescaling
    intensities and converting the arrays to the desired dtypes as needed.

    The target dtype should belong to the "int" or "uint" groups, excluding 'int64'
    and 'uint64'. np.int64(np.float64(2**63 - 1)) and np.uint64(np.float64(2**64 - 1))
    cannot be represented exactly in np.float64 or Python's float due to the loss of
    precision at such large values. The leads to overflow errors when trying to cast
    to np.int64 and np.uint64. 32-bit floats and below are thus also not supported as
    input arrays, as the loss of precision occurs even earlier, leading to casting
    issues at smaller NumPy dtypes.

    """

    # Use shorter aliases in function
    arr: np.ndarray = array
    dtype_init = initial_dtype
    dtype_final = target_dtype

    # Validate target dtype
    if dtype_final not in valid_dtypes:
        logger.error(f"{dtype_final} is not a valid NumPy dtype")
        raise ValueError

    # Reject float, complex, and int64/uint64 output dtypes
    if dtype_final.startswith(("float", "complex")) or dtype_final in (
        "int64",
        "uint64",
    ):
        logger.error(f"{dtype_final} output dtype not currently supported")
        raise NotImplementedError

    # Check that input array dtype is supported
    if str(arr.dtype) in ("float16", "float32", "complex64"):
        logger.error(f"{arr.dtype} input array not currently supported")
        raise NotImplementedError

    # Reject "complex" arrays with imaginary values
    if str(arr.dtype).startswith("complex"):
        if not np.all(arr.imag == 0):
            logger.error(f"{str(arr.dtype)} not supported by this workflow")
            raise ValueError
        else:
            arr = arr.real

    # Estimate initial dtype if none provided
    if dtype_init is None or not dtype_init.startswith(("int", "uint")):
        if dtype_init is None:
            pass  # No warning needed for None
        elif dtype_init not in valid_dtypes:
            logger.warning(
                f"{dtype_init} is not a valid NumPy dtype; converting to most appropriate dtype"
            )
        elif not dtype_init.startswith(("int", "uint")):
            logger.warning(
                f"{dtype_init} is not supported by this workflow; converting to most appropriate dtype"
            )
        # Convert to suitable dtype
        dtype_init = estimate_int_dtype(arr)
        arr = (
            convert_array_dtype(
                array=arr,
                target_dtype=dtype_final,
                initial_dtype=dtype_init,
            )
            if not np.all(arr == 0)
            else arr.astype(dtype_final)
        )
        dtype_init = dtype_final

    # Rescale intensity values
    # List of currently implemented methods (can add more as needed)
    contrast_adustment_methods = ("stretch",)

    if adjust_contrast is not None and adjust_contrast in contrast_adustment_methods:
        if adjust_contrast == "stretch":
            logger.info("Stretching image contrast across channel range")
            arr = (
                stretch_image_contrast(
                    array=arr,
                    percentile_range=(0.5, 99.5),
                )
                if not np.all(arr == 0)
                else arr
            )

    # Convert to desired bit depth
    if dtype_init != dtype_final:
        logger.info(f"Converting to {dtype_final} array")
        arr = (
            convert_array_dtype(
                array=arr,
                target_dtype=dtype_final,
                initial_dtype=dtype_init,
            )
            if not np.all(arr == 0)
            else arr.astype(dtype_final)
        )

    return arr


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

    # Use shorter aliases and calculate what is needed
    arr: np.ndarray = array

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

    # Get photometric
    valid_photometrics = [
        "minisblack",
        "miniswhite",
        "rgb",
        "ycbcr",  # Y: Luminance | Cb: Blue chrominance | Cr: Red chrominance
        "palette",
    ]
    if photometric is not None and photometric not in valid_photometrics:
        photometric = None
        logger.warning("Incorrect photometric value provided; defaulting to 'None'")

    # Process extended metadata
    if extended_metadata is None:
        extended_metadata = ""

    # Save as a greyscale TIFF
    save_name = save_dir.joinpath(file_name + ".tiff")
    logger.info(f"Saving {file_name} image as {save_name}")
    imwrite(
        save_name,
        arr,
        # Array properties,
        shape=arr.shape,
        dtype=str(arr.dtype),
        resolution=resolution,
        resolutionunit=resolution_unit,
        # Colour properties
        photometric=photometric,  # Greyscale image
        colormap=color_map,
        # ImageJ compatibility
        imagej=True,
        metadata={
            "axes": axes,
            "unit": units,
            "spacing": z_size,
            "loop": False,
            "min": round(arr.min(), 1),  # Round according to ImageJ precision
            "max": round(arr.max(), 1),
            "Info": extended_metadata,
            "Labels": image_labels,
        },
    )

    return save_name
