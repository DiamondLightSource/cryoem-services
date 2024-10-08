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
from typing import Optional

import numpy as np
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


def get_dtype_info(dtype: str) -> np.finfo | np.iinfo:
    """
    Returns NumPy's built-int dtype info object, which contains useful information about
    the dtype that can be called for use in other functions.

    See the docs for:
    numpy.finfo - https://numpy.org/doc/stable/reference/generated/numpy.finfo.html
    numpy.iinfo = https://numpy.org/doc/stable/reference/generated/numpy.iinfo.html
    """

    # Additional keywords to support for this function

    # Validate input
    if dtype not in valid_dtypes and not any(
        # dtypes without numbers should also be accepted
        dtype == key
        for key in additional_dtype_keywords
    ):
        logger.error(f"{dtype} is not a valid or supported NumPy dtype")
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
        if get_dtype_info(dtype_final).max < max(abs(arr.min()), abs(arr.max())):
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
        dtype_subset = [
            dtype
            for dtype in valid_dtypes
            if dtype.startswith(dtype_group)
            and (
                get_dtype_info(dtype).max >= arr.max()
                and get_dtype_info(dtype).min <= arr.min()
            )
        ]
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
    dtype_init = str(arr.dtype)

    # Validate initial dtype (this should never be triggered, in principle)
    if dtype_init not in valid_dtypes:
        logger.error(f"{dtype_init} is not a valid or supported NumPy dtype")
        raise ValueError

    # Reject complex dtypes if imaginary components are present
    if dtype_init.startswith("complex") and np.any(arr.imag != 0):
        logger.error("Complex numbers not currently supported")
        raise NotImplementedError
    else:
        arr = arr.real

    # Use "int" if negative values are present, and "uint" if not
    dtype_group = "uint" if arr.min() >= 0 else "int"

    result: Optional[str] = None
    # Make an estimate using the provided bit depth if set
    result = (
        _by_bit_depth(
            array=arr,
            dtype_group=dtype_group,
            bit_depth=bit_depth,
        )
        if bit_depth is not None
        else None
    )

    # Estimate using array depth instead
    result = (
        _by_array_values(array=arr, dtype_group=dtype_group)
        if result is None
        else result
    )

    if result is None:
        raise ValueError("Unable to find an appropriate dtype for the array")
    else:
        return result


def shrink_value(value: int) -> int:
    """
    When converting between floats and int dtypes in NumPy, there are cases where the
    float value will be rounded inexactly to be larger than that of its corresponding
    int, leading to issues when casting arrays with large values.

    This function replaces the value of the integer with one that can be correctly
    represented in both float and int without a further change in info.
    """

    # Validate input
    if not isinstance(value, (int)):
        raise TypeError(f"Input is not an integer: {str(type(value))}")

    vfloat = float(value)

    # Process number if float conversion increases its value
    if abs(vfloat) > abs(value):
        vstr = str(vfloat)
        if "e" in vstr:
            # Separate the numerical bit from the exponent
            num, exp = vstr.split("e", 1)

            # Remove negative sign for subsequent stage
            if num.startswith("-"):
                num = num.split("-", 1)[-1]
                neg = "-"
            else:
                neg = ""

            # Reduce precision and shrink value

            #   NOTE: Python rounds floats in scientific notation mode to 15 decimal
            #   places, so subtract 2 from that last decimal place.

            #   NOTE: Python rounds values that are exactly halfway (i.e. 0.5) to the
            #   nearest EVEN number, so 2 needs to be subtracted to ensure it doesn't
            #   round up to above the maximum value allowed by the array dtype again.
            num = str(round((float(num) - (2 / 10**15)), 15))

            # Rebuild new value as string
            vstr = "e".join(["".join([neg, num]), exp])
            vfloat = float(vstr)
            value = int(vfloat)
        else:
            # Can't think of any cases where a number not in scientific notation might
            # be bigger than its integer representation, so raising ValueError for now.
            raise ValueError(f"{vfloat} is larger than {value}")

    # Skip if it can be represented exactly as a float
    elif abs(vfloat) == abs(value):
        pass

    else:
        raise Exception("Unexpected exception occurred")

    # Returns value unchanged if it can be presented correctly as a float
    return value


def convert_array_dtype(
    array: np.ndarray,
    target_dtype: str,
    initial_dtype: Optional[str] = None,
) -> np.ndarray:
    """
    Rescales the pixel values of the array to fit within the allowed range of the
    desired array dtype while preserving the existing contrast.

    The target dtypes should belong to the "int" or "uint" groups.
    """

    # Use shorter names for variables
    arr: np.ndarray = array
    dtype_final = target_dtype
    dtype_init = initial_dtype

    # Validate the final dtype to convert to
    if dtype_final not in valid_dtypes and not any(
        dtype_final == key for key in additional_dtype_keywords
    ):
        logger.error(f"{dtype_final} is not a valid or supported NumPy dtype")
        raise ValueError

    # Support only conversion to "int" or "uint" dtypes for now
    if not dtype_final.startswith(("int", "uint")):
        logger.error(f"Array conversion to {dtype_final} is not currently supported")
        raise NotImplementedError

    # Parse initial dtype provided
    # Estimate dtype if None provided
    if dtype_init is None:
        dtype_init = estimate_int_dtype(arr)

    # Estimate dtype if invalid one provided
    if dtype_init not in valid_dtypes and not any(
        dtype_init == key for key in additional_dtype_keywords
    ):
        logger.warning(
            f"{dtype_init} is not a valid or supported NumPy dtype; estimating the dtype from the array"
        )
        dtype_init = estimate_int_dtype(arr)

    # Find closest equivalent integer dtype that encompasses floats
    if dtype_init.startswith("float"):
        dtype_init = estimate_int_dtype(arr)

    # Accept complex dtypes if imaginary components are zero
    if dtype_init.startswith("complex") and np.any(arr.imag != 0):
        logger.error("Complex numbers not currently supported")
        raise NotImplementedError
    else:
        arr = arr.real
        dtype_init = estimate_int_dtype(arr)

    # Get max supported values of initial and final arrays
    min_init = get_dtype_info(dtype_init).min
    max_init = get_dtype_info(dtype_init).max
    range_init = max_init - min_init

    min_final = get_dtype_info(dtype_final).min
    max_final = get_dtype_info(dtype_final).max
    range_final = max_final - min_final

    # Rescale
    for f in range(arr.shape[0]):
        # Map from old range to new range without exceeding maximum bit depth
        frame = np.array(
            (
                (((arr[f] / range_init) - (min_init / range_init)) * range_final)
                + min_final
            )
        )

        # Preserve dtype and round values if dtype is integer-based
        frame = frame.round(0) if dtype_final.startswith(("int", "uint")) else frame
        frame = frame.astype(dtype_final)

        # Append to array
        if f == 0:
            arr_new = np.array([frame])
        else:
            arr_new = np.append(arr_new, [frame], axis=0)

    # Do any pixels exceed the limits?
    if arr_new.min() < min_final:
        logger.warning(f"Lower limit of target array dtype exceeded: {arr_new.min()}")
    if arr_new.max() > max_final:
        logger.warning(f"Upper limit of target array dtype exceeded: {arr_new.max()}")

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

    This function should be applied to arrays of the "int" and "uint" dtypes.
    """

    # Use shorter variable names
    arr: np.ndarray = array

    # Check that dtype is supported by NumPy
    dtype = str(arr.dtype)
    if dtype not in valid_dtypes:
        logger.error(f"{dtype} is not a valid or supported NumPy dtype")
        raise ValueError

    # Handle "complex" dtypes
    if dtype.startswith("complex"):
        # Accept "complex" dtypes with no imaginary component
        if np.all(arr.imag == 0):
            dtype = "float64"  # Overwrite initial dtype
            arr = arr.real.astype(dtype)
        # Reject "complex" dtypes
        else:
            logger.error(
                f"Contrast stretching for {dtype} arrays is not currently supported"
            )
            raise NotImplementedError
        # By this point, "complex" dtypes should be eliminated
        # Only "float", "int", and "uint" should be left

    # Reject "float" dtypes if no "int"/"uint" target dtype is provided
    if dtype.startswith("float"):
        if target_dtype is None:
            logger.error(f"No target integer dtype provided for initial {dtype} array")
            raise ValueError
        if not target_dtype.startswith(("int", "uint")):
            logger.error(
                f"No valid target integer dtype provided for initial {dtype} array"
            )
            raise ValueError
        # By this point, target dtype should be "int" or "uint"

    # Handle input parameters when dtype is valid to begin with
    if dtype.startswith(("int", "uint")):
        # Raise warning if the target dtype differs from the initial array dtype and both are valid options
        if (
            target_dtype is not None
            and target_dtype.startswith(("int", "uint"))
            and target_dtype != dtype
        ):
            logger.warning(
                "Target integer dtype different from initial array dtype; using array dtype"
            )
        target_dtype = dtype
        # By this point, the target dtype should be "int" or "uint"

    # Raise exception if the target dtype is still not set by this point
    if target_dtype is None:
        logger.error("Unable to determine dtype to stretch image contrast to")
        raise Exception

    # Get key values
    b_lo: float | int = np.percentile(arr, percentile_range[0])
    b_up: float | int = np.percentile(arr, percentile_range[1])
    diff: float | int = b_up - b_lo
    dtype_info = get_dtype_info(target_dtype)
    vmax = shrink_value(dtype_info.max)

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
            np.array(
                # Scale between 0 and max positive value if no negative values are present
                ((frame / diff) - (b_lo / diff))
                * vmax
            )
            if (dtype_info.min == 0 or b_lo >= 0)
            # Keep 0 as center; scale values by largest scalar present
            else np.array(frame / max(abs(b_lo), abs(b_up)) * vmax)
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
        if dtype_info.min == 0 or b_lo >= 0:
            frame[frame <= 0] = 0

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
        frame = frame.astype(dtype)

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
            if frame.min() <= dtype_info.min:
                logger.debug(
                    "There are values below the minimum bit range for this dtype"
                )
            if frame.max() >= dtype_info.max:
                logger.debug("There are values above the max bit range for this dtype")

        # Append to array
        if f == 0:
            arr_new = np.array([frame])
        else:
            arr_new = np.append(arr_new, [frame], axis=0)

    return arr_new


class LUT(Enum):
    """
    3-channel color lookup tables to use when colorising image stacks. They are placed
    on a continuous scale from 0 to 1, making them potentially compatible with images
    of any bit depth.
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
    mode: str = "mean",
) -> np.ndarray:

    # Flatten along first (outermost) axis
    axis = 0
    if mode == "min":
        arr_new = np.array(array.min(axis=axis))
    elif mode == "max":
        arr_new = np.array(array.max(axis=axis))
    elif mode == "mean":
        dtype = str(array.dtype)
        arr_new = np.array(array.mean(axis=axis))
        arr_new = (
            arr_new.round(0) if str(dtype).startswith(("int", "uint")) else arr_new
        )
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

    # Preserve dtype of array
    # This is an averaging operation, so we can safely switch the dtype back from
    # float64 without encountering an overflow
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
    """

    # Use shorter aliases in function
    arr: np.ndarray = array
    dtype_init = initial_dtype
    dtype_final = target_dtype

    # Validate that function inputs are correct
    if dtype_final not in valid_dtypes:
        logger.error(f"{dtype_final} is not a valid or supported NumPy dtype")
        raise ValueError

    # Handle complex arrays differently
    if str(arr.dtype).startswith("complex"):
        # Reject "complex" arrays with imaginary values
        if not np.all(arr.imag == 0):
            logger.error(f"{str(arr.dtype)} not supported by this workflow")
            raise ValueError
        # Keep only the real component
        else:
            arr = arr.real

    # Estimate initial dtype if none provided
    if dtype_init is None or not dtype_init.startswith(("int", "uint")):
        if dtype_init is None:
            pass  # No warning needed for None
        elif dtype_init not in valid_dtypes:
            logger.warning(
                f"{dtype_init} is not a valid or supported NumPy dtype; converting to most appropriate dtype"
            )
        elif not dtype_init.startswith(("int", "uint")):
            logger.warning(
                f"{dtype_init} is not supported by this workflow; converting to most appropriate dtype"
            )

        dtype_init = estimate_int_dtype(arr)
        arr = (
            convert_array_dtype(
                array=arr,
                target_dtype=dtype_final,
                initial_dtype=dtype_init,
            )
            if np.all(arr == 0)
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
                if np.all(arr == 0)
                else arr
            )
        else:
            logger.warning("Invalid contrast adjustment method provided; skipping step")

    # Convert to desired bit depth
    if dtype_init != dtype_final:
        logger.info(f"Converting to {dtype_final} array")
        arr = (
            convert_array_dtype(
                array=arr,
                target_dtype=dtype_final,
                initial_dtype=dtype_init,
            )
            if np.all(arr == 0)
            else arr.astype(dtype_final)
        )
    else:
        logger.info(f"Image is already a {dtype_final} array")

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
):
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
