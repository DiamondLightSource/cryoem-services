"""
Array manipulation functions to process and manipulate the images acquired via the Leica
light microscope.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Protocol, cast

import cv2
import numpy as np
import tifffile as tf
from readlif.reader import LifFile

# Create logger object to output messages with
logger = logging.getLogger("cryoemservices.util.image_processing")


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
        return tf.imread(self.tiff_file)


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
    result: HistogramResult
        Python dataclass object containing the data, which is a 1D NumPy array
        showing the counts for each bin, and information about the error, if one
        occurs.
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

    percentiles: tuple[float, float] = (1, 99)
        The lower and upper percentiles to be extracted. Used to normalise the contrast
        before converting the image to an 8-bit NumPy array.

    bins: int = 65536
        Number of bins to group the image into. The default is 65536 bins, which is a
        16-bit channel.

    pixel_sampling_step: int = 4
        Number of pixels along each of the axes to skip when sampling the images. The
        default is 4.

    num_procs: int = 1
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
    new_shape: tuple[int, int] | None = None,
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

    new_shape: tuple[int, int] | None
        The new shape to resize the image to, if provided.

    Returns
    -------
    result: LoadImageResult
        A dataclass object containing the loaded image (2D NumPy array), the frame
        the image corresponds to, and details about any error that occurs during
        the process.
    """

    try:
        arr = image_loader.load().astype(np.float32)
        if new_shape:
            new_y, new_x = new_shape
            arr = cv2.resize(
                arr,
                dsize=(new_x, new_y),  # cv2 takes (x, y) order
                interpolation=cv2.INTER_AREA,
            )
        scale = 255 / ((vmax - vmin) or 1)  # Downscale to 8-bit
        np.subtract(arr, vmin, out=arr)
        np.multiply(arr, scale, out=arr)
        np.clip(arr, a_min=0, a_max=255, out=arr)
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


def load_and_resize_tile(
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

    Returns
    -------
    result: ResizeTileResult
        A dataclass containing the resized image and the array slicing information
        needed to assign it to its location in the parent image stack.
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
        img = image_loader.load().astype(np.float32)
        resized = cv2.resize(
            img,
            dsize=(tile_x_pixels, tile_y_pixels),
            interpolation=cv2.INTER_AREA,
        )
        # Normalise to 8-bit
        scale = 255 / ((vmax - vmin) or 1)
        np.subtract(resized, vmin, out=resized)
        np.multiply(resized, scale, out=resized)
        np.clip(resized, a_min=0, a_max=255, out=resized)

        return ResizeTileResult(
            data=resized.astype(np.uint8),
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
    x_res: float | None = None,
    y_res: float | None = None,
    z_res: float | None = None,
    units: str | None = None,
    # Array properties
    axes: str | None = None,
    image_labels: list[str] | str | None = None,
    # Colour properties
    photometric: str | None = None,  # Valid options listed below
    color_map: np.ndarray | None = None,
    extended_metadata: str | None = None,  # Stored as an extended string
) -> Path:
    """
    Writes the NumPy array as a calibrated, ImageJ-compatible TIFF image stack.
    It will save image stacks in BigTIFF format, while 2D images will be saved
    as conventional TIFFs.

    Parameters
    ----------
    array: np.ndarray
        The image to be saved.

    save_dir: Path
        The file path (absolute or relative) to where the image is to be saved.

    file_name: str
        The name (not including the file suffix) to assign to the image.

    x_res: float | None = None
        The resolution (pixels per unit length) of the x-axis. It should match
        the SI unit used in the 'units' parameter. i.e. For a pixel size of
        12 um, the resolution will be 83.3333... if 'mm' is used.

    y_res: float | None = None
        The resolution of the y-axis.

    z_res: float | None = None
        The resolution of the z-axis.

    units: str | None = None
        The SI unit of the resolution values that have been provided above. Most
        conventional imaging units are accepted (e.g., m, mm, um, micron).

    axes: str | None = NOne
        A string sequence telling the TIFF writer what the axes in the incoming
        image corespond to. They should be passed in the order TZCYXS.

    image_labels: list[str] | str | None = None
        A string (for a single frame) or list of strings (for image stacks), of
        the labels to assign to each image.

    photometric: str | None
        Information on the colouring mode to associate the image with. The currently
        supported options include:
            * "minisblack",
            * "miniswhite",
            * "rgb",
            * "ycbcr",
            * "palette",

    color_map: np.ndarray = None
        The color map used to determine how the colour should be scaled according
        to pixel value.

    extended_metadata: str | None = None
        Optional additional metadata that can be inserted into the image header as
        a stringified object.

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
    tf.imwrite(
        save_name,
        array,
        bigtiff=use_bigtiff,
        # Array properties,
        shape=array.shape,
        dtype=str(array.dtype),
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


def write_image(
    img: np.ndarray,
    save_dir: Path,
    name: str,
):
    """
    Helper function to quickly save a 2D grayscale/RGB image in most common file
    formats (e.g. PNG, JPG, TIFF).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_dir / name, img)


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
    """
    Converts a grayscale image into an RGB image with the desired colour format.
    """
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
    """
    Flattens an image along its outermost axis, turning an image stack into a 2D image.

    Parameters
    ----------
    array: np.ndarray
        The incoming image. The input can technically be any NumPy array with 2 or more
        dimensions, but this should be used for grayscale or RGB image stacks.

    mode: Literal["mean", "min", "max"] = "mean"
        The method with which the pixels along the axis being flattened are sampled.
        "min" and "max" would respectively return the minimum and maximum intensity
        projections for each pixel along that axis, while "mean" computes the average
        pixel value along that axis.
    """
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
    Takes a list of images and returns a composite image averaged across every image
    in the list.
    """

    # Standardise inputs as a list
    if not isinstance(arrays, list):
        arrays = [arrays]

    # Check that an array was provided
    if not arrays:
        raise ValueError("No input arrays provided")

    # Raise error if any of the list inputs aren't an array
    if any(not isinstance(arr, np.ndarray) for arr in arrays):
        raise ValueError("One or more of the provided inputs is not a NumPy array")

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


def create_hanning_window(width: int, height: int):
    """
    Generate a 2D Hanning window of the specified dimensions. This array will
    smoothly taper towards zero at the edges of the image, which helps with
    reducing boundary artifacts and edge discontinuities.
    """
    return np.outer(np.hanning(height), np.hanning(width))


def apply_sobel_edge_filter(
    array: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """
    Calculate the Sobel edge magnitude of a 2D grayscale image.

    Horizontal and vertical Sobel gradients are calculated and combined to produce
    thegradient magnitude image. The result is then normalised and returned as an
    8-bit array.

    Parameters
    ----------
    array : np.ndarray
        Input 2D image array.

    kernel_size : int
        Size of the Sobel kernel used to compute the gradients.

    Returns
    -------
    np.ndarray
        Normalized gradient magnitude image highlighting edge strength.
    """
    gx = cv2.Sobel(array, cv2.CV_32F, 1, 0, ksize=kernel_size)
    gy = cv2.Sobel(array, cv2.CV_32F, 0, 1, ksize=kernel_size)

    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)


def threshold_image(
    array: np.ndarray,
    percentile: float | None = None,
) -> np.ndarray:
    """
    Helper function to threshold an input image and return it as a binary 8-bit one
    If a percentile is provided, uses the percentile to determine the threshold value.
    Otherwise, it will default to using the Otsu method to automatically select a
    suitable threshold.
    """
    # Use the percentile if provided
    _, binary = cv2.threshold(
        array,
        thresh=(np.percentile(array, percentile) if percentile is not None else 0),
        maxval=255,
        type=cv2.THRESH_BINARY + (0 if percentile is not None else cv2.THRESH_OTSU),
    )
    return cast(np.ndarray, binary).astype(np.uint8)
