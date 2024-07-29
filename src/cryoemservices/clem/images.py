"""
Array manipulation and image processing functions for the images acquired via the Leica
light microscope.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from tifffile import imwrite

# Create logger object to output messages with
logger = logging.getLogger("cryoemservices.clem.images")

# Accepted bit depths and corresponding NumPy dtypes
# For use by various functions in the script
valid_bit_depths = (8, 16, 32, 64)
valid_dtypes = tuple(f"uint{n}" for n in valid_bit_depths)


class UnsignedIntegerError(Exception):
    """
    Raised if the bit depth value provided is not one that NumPy can interpret as an
    unsigned integer dtype.
    """

    def __init__(
        self,
        bit_depth: int,
    ):
        self.bit_depth = bit_depth
        self.message = (
            f"The bit depth provided ({bit_depth}) is not a NumPy-compatible unsigned integer dtype. "
            "Only 8, 16, 32, and 64 bits are allowed. "
        )
        super().__init__(self.message)


def estimate_bit_depth(array: np.ndarray) -> int:
    """
    Returns the smallest bit depth that will enclose the range of values present in
    an array.
    """

    bit_depth = np.ceil(np.log2(array.max()))
    # Raise error if value is too large
    if bit_depth > 64:
        raise UnsignedIntegerError(bit_depth)

    # Return bit_depth if corresponding to one of the accepted values
    if bit_depth in valid_bit_depths:
        return bit_depth
    # Return smallest value that is larger than the specified one
    else:
        new_bit_depth = min(n for n in valid_bit_depths if n > bit_depth)
        return new_bit_depth


def stretch_image_contrast(
    array: np.ndarray,
    percentile_range: tuple[float, float] = (0.5, 99.5),  # Lower and upper percentiles
) -> np.ndarray:
    """
    Changes the range of pixel values occupied by the data, rescaling it across the
    entirety of the array's bit depth.
    """

    # Check that dtype is supported by NumPy
    dtype = str(array.dtype)
    bit_depth = int("".join([char for char in dtype if char.isdigit()]))
    if dtype not in valid_dtypes:
        raise UnsignedIntegerError(bit_depth)
    max_int = 2**bit_depth - 1

    # Use shorter variable names
    arr: np.ndarray = array
    b_lo: Union[float, int] = np.percentile(arr, percentile_range[0])
    b_up: Union[float, int] = np.percentile(arr, percentile_range[1])

    # Overwrite outliers and normalise values to range
    for f in range(arr.shape[0]):
        frame: np.ndarray = arr[f]
        frame[frame < b_lo] = b_lo
        frame[frame > b_up] = b_up
        frame = np.array((frame - b_lo) / (b_up - b_lo) * max_int)  # Normalise
        frame = frame.round(0).astype(dtype)  # Round and convert to dtype
        if f == 0:
            arr_new = np.array([frame])
        else:
            arr_new = np.append(arr_new, [frame], axis=0)

    return arr_new


def convert_array_bit_depth(
    array: np.ndarray,
    target_bit_depth: int,
    initial_bit_depth: Optional[int] = None,
) -> np.ndarray:
    """
    Rescales the pixel values of the array to fit within the desired array bit depth
    WITHOUT modifying the contrast.

    If the array has a bit depth not compatible with NumPy, one can be provided
    """

    # Use shorter names for variables
    arr: np.ndarray = array
    bit_final: int = target_bit_depth
    dtype_final = f"uint{bit_final}"

    # Validate the final dtype to convert to
    if bit_final not in valid_bit_depths:
        raise UnsignedIntegerError(bit_final)

    # Use initial bit depth if provided
    if initial_bit_depth is not None:
        bit_init = initial_bit_depth
    # Otherwise, get it from the array
    else:
        dtype_init = str(arr.dtype)
        bit_init = int("".join([char for char in dtype_init if char.isdigit()]))

    # Get max pixel values of initial and final arrays
    int_init = int(2**bit_init - 1)
    int_final = int(2**bit_final - 1)

    # Rescale (DIVIDE BEFORE MULTIPLY)
    for f in range(arr.shape[0]):
        frame_new = np.array(arr[f] / int_init * int_final).round(0).astype(dtype_final)
        if f == 0:
            arr_new = np.array([frame_new])
        else:
            arr_new = np.append(arr_new, [frame_new], axis=0)

    return arr_new


def process_img_stk(
    array: np.ndarray,
    initial_bit_depth: int,
    target_bit_depth: int = 8,
    adjust_contrast: Optional[str] = None,
) -> np.ndarray:
    """
    Processes the NumPy array, rescaling intensities and converting to the desired bit
    depth as needed.
    """

    # Use shorter aliases in function
    arr: np.ndarray = array
    bdi = initial_bit_depth
    bdt = target_bit_depth

    # Validate that function inputs are correct
    if bdt not in valid_bit_depths:
        raise UnsignedIntegerError(bdt)

    if bdi not in valid_bit_depths:
        logger.info(f"{bdi}-bit is not supported by NumPy; converting to 16-bit")
        arr = (
            convert_array_bit_depth(
                array=arr,
                target_bit_depth=16,
                initial_bit_depth=bdi,
            )
            if np.max(arr) > 0
            else arr.astype(f"uint{16}")
        )
        bdi = 16  # Overwrite

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
    if not bdi == bdt:
        logger.info(f"Converting to {bdt}-bit image")
        arr = (
            convert_array_bit_depth(
                array=arr,
                target_bit_depth=bdt,
                initial_bit_depth=bdi,
            )
            if np.max(arr) > 0
            else arr.astype(f"uint{bdt}")
        )
    else:
        logger.info(f"Image is already {bdt}-bit")

    return arr


def write_to_tiff(
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
    image_labels: Optional[Union[list[str], str]] = None,
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

    return arr
