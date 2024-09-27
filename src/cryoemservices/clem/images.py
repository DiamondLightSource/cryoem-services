"""
Array manipulation and image processing functions for the images acquired via the Leica
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
from tifffile import imwrite

# Create logger object to output messages with
logger = logging.getLogger("cryoemservices.clem.images")


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
        raise Exception("Unable to get list of NumPy dtypes from NumPy module")

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
        raise ValueError(f"{dtype} is not a valid or supported NumPy dtype")

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
    ) -> str:

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
            raise Exception("No suitable dtypes found based on provided bit depth")

        # Use the minimum viable dtype
        dtype_final = f"{dtype_group}{min(bit_list)}"

        # Return None if dtype calculated using provided bit depth can't accommodate array
        if get_dtype_info(dtype_final).max < max(abs(arr.min()), abs(arr.max())):
            raise Exception(
                "Array values still exceed those supported by the estimated dtype"
            )

        # Return estimated dtype otherwise
        return dtype_final

    def _by_array_values(array: np.ndarray, dtype_group: str) -> str:

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
            raise Exception(
                "No suitable dtypes found that can accommodate the array's values"
            )
        # Use the smallest value
        dtype_final = f"{dtype_group}{min(bit_list)}"

        return dtype_final

    # Set up variables
    arr = array
    dtype_init = str(arr.dtype)

    # Validate initial dtype (this should never be triggered, in principle)
    if dtype_init not in valid_dtypes:
        raise ValueError(f"{dtype_init} is not a valid or supported NumPy dtype")

    # Reject complex dtypes if imaginary components are present
    if dtype_init.startswith("complex") and np.any(arr.imag != 0):
        raise NotImplementedError("Complex numbers not currently supported")
    else:
        arr = arr.real

    # Use "int" if negative values are present, and "uint" if not
    dtype_group = "uint" if arr.min() >= 0 else "int"

    result = None
    if bit_depth is not None:
        try:
            # Make an estimate using the provided bit depth
            result = _by_bit_depth(
                array=arr,
                dtype_group=dtype_group,
                bit_depth=bit_depth,
            )
        except Exception:
            pass

    dtype_final = (
        _by_array_values(array=arr, dtype_group=dtype_group)
        if result is None
        else result
    )

    return dtype_final


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
        raise ValueError(f"{dtype_final} is not a valid or supported NumPy dtype")

    # Support only conversion to "int" or "uint" dtypes for now
    if not dtype_final.startswith(("int", "uint")):
        raise NotImplementedError(
            f"Array conversion to {dtype_final} is not currently supported"
        )

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
        raise NotImplementedError("Complex numbers not currently supported")
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
) -> np.ndarray:
    """
    Changes the range of pixel values occupied by the data, rescaling it across the
    entirety of the array's bit depth.

    This function should be applied to arrays of the "int" and "uint" dtypes.
    """

    # Use shorter variable names
    arr: np.ndarray = array
    b_lo: float | int = np.percentile(arr, percentile_range[0])
    b_up: float | int = np.percentile(arr, percentile_range[1])

    # Check that dtype is supported by NumPy
    dtype = str(array.dtype)
    if dtype not in valid_dtypes:
        raise ValueError(f"{dtype} is not a valid or supported NumPy dtype")

    # Reject "float" and "complex" dtype inputs
    if dtype.startswith(("complex", "float")):
        raise NotImplementedError(
            f"Contrast stretching for {dtype} arrays is not currently supported"
        )

    dtype_info = get_dtype_info(dtype)

    for f in range(arr.shape[0]):
        # Overwrite outliers and normalise to new range
        frame: np.ndarray = arr[f]
        frame[frame < b_lo] = b_lo
        frame[frame > b_up] = b_up
        # Normalise differently depending on whether dtype supports negative values
        frame = (
            np.array(
                # Scale between 0 and max positive value if no negative values are present
                ((frame / (b_up - b_lo)) - (b_lo / (b_up - b_lo)))
                * dtype_info.max
            )
            if (dtype_info.min == 0 or b_lo >= 0)
            # Keep 0 as center; scale values by largest scalar present
            else np.array(frame / max(abs(b_lo), abs(b_up)) * dtype_info.max)
        )

        # Debug information
        # print(
        #     f"dtype: {frame.dtype} \n"
        #     f"Shape: {frame.shape} \n"
        #     f"Min: {frame.min()} \n"
        #     f"Max: {frame.max()} \n"
        # )

        # Preserve dtype and round values if dtype is integer-based
        frame = frame.round(0) if dtype.startswith(("int", "uint")) else frame
        frame = frame.astype(dtype)

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
    mode: Literal["min", "max", "mean"] = "mean",
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
        raise ValueError(f"{mode} is not a valid image flattening option")

    return arr_new


def create_composite_image(
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
        raise ValueError("Input arrays do not have the same shape")

    # Validate that arrays have the same dtype
    if len({str(arr.dtype) for arr in arrays}) > 1:
        raise ValueError("Input arrays do not have the same dtype")
    dtype = str(arrays[0].dtype)

    # Calculate average across all arrays
    arr_new: np.ndarray = np.mean(arrays, axis=0)

    # Preserve dtype of array
    # This is an averaging operation, so we can safely switch the dtype back from
    # float64 without encountering an overflow
    arr_new = arr_new.round(0) if dtype.startswith(("int", "uint")) else arr_new
    arr_new = arr_new.astype(dtype)

    return arr_new


"""
FUNCTIONS FOR PRE-PROCESSING OF IMAGE STACKS
"""


def process_img_stk(
    array: np.ndarray,
    initial_dtype: str,
    target_dtype: str = "uint8",
    adjust_contrast: Optional[str] = None,
) -> np.ndarray:
    """
    Processes the NumPy array, rescaling intensities and converting to the desired
    dtype as needed.
    """

    # Use shorter aliases in function
    arr: np.ndarray = array
    dtype_init = initial_dtype
    dtype_final = target_dtype

    # Validate that function inputs are correct
    if dtype_final not in valid_dtypes:
        raise ValueError(f"{dtype_final} is not a valid or supported NumPy dtype")

    if dtype_init not in valid_dtypes:
        logger.info(
            f"{dtype_init} is not a valid or supported NumPy dtype; converting to most appropriate dtype"
        )
        arr = (
            convert_array_dtype(
                array=arr,
                target_dtype=dtype_final,
                initial_dtype=dtype_init,
            )
            if np.max(arr) > 0
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
                if np.max(arr) > 0
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
            if np.max(arr) > 0
            else arr.astype(dtype_final)
        )
    else:
        logger.info(f"Image is already a {dtype_final} array")

    return arr


def write_stack_to_tiff(
    array: np.ndarray,
    save_dir: Path,
    series_name: str,
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
    save_name = save_dir.joinpath(series_name + ".tiff")
    logger.info(f"Saving {series_name} image as {save_name}")
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
