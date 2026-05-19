"""
Array manipulation functions to process and manipulate the images acquired via the Leica
light microscope.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Literal, Protocol, cast

import cv2
import numpy as np
import SimpleITK as sitk
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


def align_images_using_mmi(
    reference_array: np.ndarray,
    moving_array: np.ndarray,
    downsample_factor: int = 2,
    sampling_fraction: float = 0.5,
    shrink_factors_per_level: list[int] = [2, 1],
    smoothing_sigmas_per_level: list[float] = [1.0, 0.5],
    num_procs: int = 1,
) -> np.ndarray:
    """
    Align images to a reference using SimpleITK's implementation of the Mattes
    Mutual Information image registration method. This workflow handles 2D images
    or image stacks in both grayscale and RGB formats.

    Currently, this method works poorly for defocused images, which can lead to a
    lot of jitter in the beginning and tail frames if the images being aligned are
    a defocus series.

    Parameters
    ----------
    reference_array: np.ndarray
        The image being used as a reference.

    moving_array: np.ndarray
        The image to align.

    downsample_factor: int = 2
        The degree of binning to apply to the image during the registration process.
        While resizing, SITK preserves the dimensions of the image when it is first
        converted into an SITK Image, and will adjust the pixel size associated with
        the Image when it is subsequently resized.

    sampling_fraction: float = 0.5,
        The fraction of pixels to sample when calculating the transformation matrix.

    shrink_factors_per_level: list[int] = [2, 1]
        The degree of shrinking to apply to the image per pyramid level, with the
        registration being repeated to fine-tune the transformation matrix.

    smoothing_sigmas_per_level: list[float] = [1.0, 0.5]
        The intensity of the Gaussian blurring to apply at each pyramid level.

    num_procs: int = 1
        The number of threads to run this function with. The code has been optimised
    """

    def _register_frame(frame_num: int, ref: np.ndarray, mov: np.ndarray):
        try:
            logger.info("Setting up SITK image objects")
            ref_sitk = sitk.Cast(sitk.GetImageFromArray(ref), sitk.sitkFloat32)
            mov_sitk = sitk.Cast(sitk.GetImageFromArray(mov), sitk.sitkFloat32)
            # Downsample the frame
            if downsample_factor > 1:
                ref_small = sitk.Shrink(ref_sitk, [downsample_factor] * 2)
                mov_small = sitk.Shrink(mov_sitk, [downsample_factor] * 2)
            else:
                ref_small, mov_small = ref_sitk, mov_sitk

            # Set up registration method
            logger.info("Setting up registration method")
            registration = sitk.ImageRegistrationMethod()
            registration.SetInterpolator(sitk.sitkLinear)

            # Set the metric to use
            registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
            registration.SetMetricSamplingPercentage(sampling_fraction)
            registration.SetMetricSamplingStrategy(registration.RANDOM)

            # Use Mattes Mutual Information as the metric
            registration.SetOptimizerAsGradientDescentLineSearch(
                learningRate=1.0,
                numberOfIterations=200,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=5,
            )
            registration.SetOptimizerScalesFromIndexShift()

            # Register over a multi-resolution pyramid
            registration.SetShrinkFactorsPerLevel(shrink_factors_per_level)
            registration.SetSmoothingSigmasPerLevel(smoothing_sigmas_per_level)
            registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

            # Initilaise transform or reuse previous one, if available
            initial_transform = sitk.CenteredTransformInitializer(
                ref_small,
                mov_small,
                sitk.Euler2DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
            registration.SetInitialTransform(initial_transform, inPlace=False)

            # Execute registration on downsampled images
            logger.info(f"Registering frame {frame_num}")
            final_transform = registration.Execute(ref_small, mov_small)

            # Apply transform to full resolution frame
            logger.info(f"Applying transformation to frame {frame_num}")
            aligned = sitk.GetArrayFromImage(
                sitk.Resample(
                    mov_sitk,
                    transform=final_transform,
                    interpolator=sitk.sitkLinear,
                    outputPixelType=sitk.sitkFloat32,
                )
            )
            np.clip(aligned, a_min=vmin, a_max=vmax, out=aligned)
            if np.issubdtype(dtype, np.integer):
                np.rint(aligned, out=aligned)
            return aligned.astype(dtype, copy=False), frame_num
        except Exception:
            logger.warning(
                f"Error registering frame {frame_num} to reference. Returning original image",
                exc_info=True,
            )
            return mov, frame_num

    # Restrict number of threads used by SimpleITK
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    start_time = time.perf_counter()
    logger.debug(
        f"SITK image alignment settings: \n"
        f"Downsample factor: {downsample_factor}\n"
        f"Sampling percentage: {sampling_fraction}\n"
        f"Shrink factors per level: {shrink_factors_per_level}\n"
        f"Smoothing sigmas per level: {smoothing_sigmas_per_level}"
    )

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

    # Pre-allocate output NumPy array
    aligned = np.empty(moving_array.shape, dtype=dtype)

    with ThreadPoolExecutor(max_workers=num_procs) as pool:
        futures = [
            pool.submit(
                _register_frame,
                f,
                reference_array[f],
                moving_array[f],
            )
            for f in range(num_frames)
        ]
        for future in as_completed(futures):
            frame, frame_num = future.result()
            aligned[frame_num] = frame

    # If the image was not initially a stack, flatten it
    if not was_a_stack:
        aligned = aligned[0]

    end_time = time.perf_counter()
    logger.debug(f"Completed registration of image stack in {end_time - start_time}s")
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


def align_images_using_neighbors(
    reference_array: np.ndarray,
    moving_array: np.ndarray,
    # Preprocessing
    median_blur: int | None = None,
    gaussian_blur: float | None = 3,
    sobel_kernel: int | None = 3,
    use_hanning: bool = False,
    threshold_percentile: float | None = 99,
    min_component_area: int | None = 200,
    morph_close_kernel: int | None = 19,
    morph_open_kernel: int | None = 3,
    # Feature detection
    min_feature_area: int | None = 400,
    max_feature_area: int | None = 5000,
    min_solidity: float | None = 0.6,
    max_aspect_ratio: float | None = 0.9,
    # Similarity calculation and registration
    max_neighbor_distance: float | None = 400,
    min_score: float | None = 0.2,
    ransac_threshold: float = 10,
    # Debug options
    save_tables: bool = False,
    save_images: bool = False,
    save_dir: Path | None = None,
):
    """
    Image registration function that identifies features in the reference and moving
    images and aligns them by using the geometry of the features' neighbours to work
    out the transformation matrix.

    Parameters
    ----------
    reference_array: np.ndarray
        The reference image. Currently, only 2D grayscale images are supported.
    moving_array: np.ndarray
        The image to be aligned.
    median_blur: int | None
        The kernel size of the median blur to be applied to the image.
        The default is None.
    gaussian_blur: float | None
        The sigma value to use to Gaussian blur the image.
        The default is 3.
    sobel_kernel: int | None
        The kernel size to use for the Sobel edge filter. Larger kernel sizes
        mean that the pixel intensity is computed over a larger radius.
        The default is 3.
    use_hanning: bool
        Decide whether to apply a Hanning window to the image. This dampens the
        intensity of the image smoothly towards the edges, emphasising features
        towards the centre of the image.
        The default is False.
    threshold_percentile: float | None
        The nth percentile of the pixel values to threshold the image to. If
        set to None, The function will use the Otsu method to attempt to find
        a suitable threshold value.
        The default is 99.0.
    min_component_area: int | None
        The minimum pixel area a feature in the thresholded image must have to
        be considered for feature detection. This helps reduce noisy artifacts
        left over after the initial thresholding operation.
        The default is 200.
    morph_close_kernel: int | None
        The kernel size to use when joining fragmented features and filling in
        holes formed by the features after thresholding.
        The default is 19.
    morph_open_kernel: int | None
        The kernel size to use when separating features in the thresholded image.
        This is used to separate features that might have been joined together
        by the previous step due to their close proximity to one another.
        The default is 3
    min_feature_area: int | None
        The minimimum pixel area of the feature for it to be used for computing
        the transformation matrix.
        The default is 400.
    max_feature_area: int | None
        The maximum pixel area of the feature to be used.
        The default is 5000.
    min_solidity: float | None
        The minimum ratio between the feature's actual area and the area of the
        smallest convex hull it is bound by. This is used to reject features
        that are highly irregular in shape.
        The default is 0.6.
    max_aspect_ratio: float | None
        The maximum aspect ratio of the ellipse fitted around a feature, beyond
        which the feature will not be used to compute the transformation matrix
        with. A ratio of 1 indicates that the ellipse is a perfect sphere, and
        a value of 0.5 indicates that its major axis is twice as long as its
        minor one.
        The default is 0.9.
    max_neighbor_distance: float | None
        The maximum pixel distance between two features for them to be included
        in the similarity score computation.
        The default is 400.0.
    min_score: float | None
        The minimum similarity score between features in the reference and moving
        images for their coordinates to be considered when computing the final
        transformation matrix.
        The default is 0.2
    ransac_threshold: float
        The maximum pixel distance the points between the reference and moving
        images are allowed to differ from one another before the computed
        transform is considered bad.
        The default is 5.
    save_tables: bool
        Toggle whether to save intermediate tables.
        The default is False.
    save_images: bool
        Toggle whether to save intermediate images.
        The default is False.
    save_dir: Path | None
        If either 'save_tables' or 'save_images' is set tot True, this determines
        the folder to save the output files to. If this is not set, a warning will
        be logged and the files will not be generated.
        The default is None.

    Returns
    -------
    result: dict[str, np.ndarray]
        Contains the aligned image and the transformation matrix, stored under the
        keys "aligned" and "transform" respectively.

    """

    def _save_image(
        name: str,
        img: np.ndarray,
    ):
        if save_images and save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_dir / name, img)

    def _filter_components(
        binary: np.ndarray,
    ):
        """
        Removes components in a binary image with an area smaller than the threshold.
        """
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            image=binary,
            labels=8,
        )
        filtered = np.zeros_like(binary)
        for i in range(1, n):
            if (
                min_component_area is not None
                and stats[i, cv2.CC_STAT_AREA] >= min_component_area
            ):
                filtered[labels == i] = 255
        return filtered

    def _fill_holes(
        binary: np.ndarray,
    ):
        """
        Performs morphological closing to connect disjointed components in the binary
        image using the kernel size provided, before performing morphological opening
        to separate components that have been joined together by thin bridges using a
        different, usually smaller, kernel size.
        """
        if morph_close_kernel is not None:
            kernel_close = cv2.getStructuringElement(
                shape=cv2.MORPH_ELLIPSE,
                ksize=(morph_close_kernel, morph_close_kernel),
            )
            binary = cv2.morphologyEx(
                binary,
                op=cv2.MORPH_CLOSE,
                kernel=kernel_close,
            )
        if morph_open_kernel is not None:
            kernel_open = cv2.getStructuringElement(
                shape=cv2.MORPH_ELLIPSE,
                ksize=(morph_open_kernel, morph_open_kernel),
            )
            binary = cv2.morphologyEx(
                binary,
                op=cv2.MORPH_OPEN,
                kernel=kernel_open,
            )
        # Attempt to fill in holes after connecting components
        filled = np.zeros_like(binary)
        contours, _ = cv2.findContours(
            binary,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            filled,
            contours=contours,
            contourIdx=-1,
            color=255,
            thickness=cv2.FILLED,
        )
        return filled

    def _preprocess(
        img: np.ndarray,
        name: str,
    ):
        height, width = img.shape[:2]
        processed = img
        # Apply median blur
        if median_blur is not None:
            processed = cv2.medianBlur(processed, median_blur)
            _save_image(f"{name}_median.png", processed)
        # Apply Gaussian blur
        if gaussian_blur is not None:
            processed = cv2.GaussianBlur(processed, ksize=(0, 0), sigmaX=gaussian_blur)
            _save_image(f"{name}_gaussian.png", processed)
        # Apply Hanning window
        if use_hanning:
            window = create_hanning_window(width=width, height=height).astype(
                np.float32
            )
            processed = processed.astype(np.float32) * window
            cv2.normalize(
                src=processed,
                dst=processed,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
            )
            processed = processed.astype(np.uint8)
            _save_image(f"{name}_hanning.png", processed)
        # Apply Sobel filter
        if sobel_kernel is not None:
            processed = apply_sobel_edge_filter(processed, kernel_size=sobel_kernel)
            _save_image(f"{name}_sobel.png", processed)
        # Threshold image
        binary = threshold_image(
            processed,
            percentile=threshold_percentile,
        )
        _save_image(f"{name}_threshold.png", binary)
        # Filter out small features after thresholding
        if min_component_area is not None:
            binary = _filter_components(binary)
            _save_image(f"{name}_filtered.png", binary)
        # Connect disjointed components in the binary image
        binary = _fill_holes(binary)
        _save_image(f"{name}_filled.png", binary)

        return binary

    def _detect_features(
        binary: np.ndarray,
        name: str,
    ) -> np.ndarray:
        """
        Identifies features in a thresholded image that fulfil the criteria specified.
        Returns a list of descriptors for each feature, along with either None if
        'save_images' is False or an annotated version of the image if True.
        """
        height, width = binary.shape[:2]
        contours, _ = cv2.findContours(
            binary,
            mode=cv2.RETR_LIST,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        # Create RGB version of binary image for annotation or a None placeholder
        annotated = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) if save_images else None

        # Keep only features that match criteria, and extract descriptors
        features = []
        index = 0
        for contour in contours:
            # Remove features that cannot be used for ellipse fitting
            if len(contour) < 5:
                continue
            # Check contour area feature
            area = cv2.contourArea(contour)
            if min_feature_area is not None and area < min_feature_area:
                continue
            if max_feature_area is not None and area > max_feature_area:
                continue

            # Check convex hull area
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            # Filter out bad fits
            if hull_area <= 0:
                continue

            # Estimate solidity (how much of the hull the feature takes up)
            solidity = area / hull_area
            if solidity < min_solidity:
                continue

            # Elliptical fit for the feature
            ellipse = cv2.fitEllipse(contour)
            (x, y), (w, h), angle = ellipse
            # w = short axis
            # h = long axis

            # Exclude fits where the center is outside the image
            if x < 0 or x > width:
                continue
            if y < 0 or y > height:
                continue

            # Adjust the returned angle so that:
            # up = 0,
            # clockwise = positive
            # anticlockwise = negative
            if angle > 90:
                angle -= 180

            # Check the aspect ratio
            aspect = min(w, h) / max(w, h)
            # Reject fits that are too circular
            if aspect > max_aspect_ratio:
                continue

            # Append results
            features.append((x, y, w, h, angle, area, hull_area))

            # Annotate image
            if annotated is not None:
                # Draw the outline of the fitted contour and convex hull
                cv2.drawContours(
                    annotated,
                    contours=[contour, hull],
                    contourIdx=-1,
                    color=(0, 0, 255),
                    thickness=line_thickness,
                )
                # Draw the fitted ellipse
                cv2.ellipse(
                    annotated,
                    box=ellipse,
                    color=(0, 255, 255),
                    thickness=line_thickness,
                )
                # Add a marker
                cv2.circle(
                    annotated,
                    center=(int(x), int(y)),
                    radius=marker_size,
                    color=(0, 255, 0),
                    thickness=line_thickness,
                )
                # Add the index number
                cv2.putText(
                    annotated,
                    text=f"{index}",
                    org=(int(x) + 40, int(y)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(255, 255, 255),
                    thickness=line_thickness,
                    lineType=cv2.LINE_AA,
                )

            # Increment the index for next loop once successful
            index += 1

        if annotated is not None:
            _save_image(f"{name}_features.png", annotated)
        if save_tables and save_dir is not None:
            np.savetxt(
                save_dir / f"{name}_features.tsv",
                np.column_stack((np.arange(len(features)), features)),
                fmt=[
                    "%-5d",
                    "%-10.3f",
                    "%-10.3f",
                    "%-10.3f",
                    "%-10.3f",
                    "%-10.3f",
                    "%-10.3f",
                    "%-10.3f",
                ],
                header=(
                    f"{'':<3} "
                    f"{'x':<10} "
                    f"{'y':<10} "
                    f"{'w':<10} "
                    f"{'h':<10} "
                    f"{'angle':<10} "
                    f"{'cont_area':<10} "
                    f"{'hull_area':<10} "
                ),
            )
        return np.array(features, dtype=np.float32)

    def _match_features(
        ref_features: np.ndarray,
        mov_features: np.ndarray,
    ):
        """
        Constructs descriptors for each feature in the arrays provided, which are then
        used to compute a similarity score between the two features. Only the features
        for which they are each other's best match. For instance, for Point A in the
        reference image and Point B in the moving image, Point B must be the highest
        scoring candidate for Point A, and Point A must be Point B's highest scoring
        candidate.
        """

        def _build_descriptor(
            features: np.ndarray,
        ):
            """
            Build geometric descriptors describing the arrangement of neighbouring
            features around each parent feature.
            - Use pairwise distance matrices instead of repeated norm calculations
            - Use combinations() to quickly iterate unique neighbor combinations
            - Precomputes logarithms before comparing similarities
            """
            # Safe early exit if no features were provided
            if len(features) == 0:
                return np.empty((0, 13), dtype=np.float32)

            # Precompute distances between points via matrix operations
            coords = features[:, :2]  # x, y
            distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

            descriptors = []
            num_features = len(features)
            for o in range(num_features):
                # Only look at nearest neighbours
                if max_neighbor_distance is not None:
                    neighbors = np.where(
                        (distances[o] <= max_neighbor_distance) & (distances[o] > 0)
                    )[0]
                else:
                    neighbors = np.array([i for i in range(num_features) if i != o])

                # Skip if there aren't at least 2 neighboirs
                if len(neighbors) < 2:
                    continue

                O = coords[o]

                # Iterate through all unique neighbour combinations
                for a, b in combinations(neighbors, 2):
                    dA = distances[o, a]
                    dB = distances[o, b]

                    # Order neighbours from nearest to furthest
                    if dA < dB:
                        n, f = a, b
                        d_near, d_far = dA, dB
                    else:
                        n, f = b, a
                        d_near, d_far = dB, dA

                    # Calculate the signed angle of vector NOF
                    N = coords[n]
                    F = coords[f]

                    NO = O - N
                    OF = F - O
                    cross = (NO[0] * OF[1]) - (NO[1] * OF[0])
                    dot = NO @ OF
                    angle = np.arctan2(cross, dot)

                    # Extract ellipse and contour descriptors
                    feat_near = features[n][[2, 3, 5, 6]]
                    feat_far = features[f][[2, 3, 5, 6]]

                    # Precompute their logs and extra feature names
                    wn, hn, area_n, hull_n = np.log(
                        np.where(feat_near <= 0, eps, feat_near)
                    )
                    wf, hf, area_f, hull_f = np.log(
                        np.where(feat_far <= 0, eps, feat_far)
                    )

                    # Append
                    descriptors.append(
                        [
                            o,
                            n,
                            f,
                            d_near / (d_far + eps),
                            angle,
                            wn,
                            hn,
                            area_n,
                            hull_n,
                            wf,
                            hf,
                            area_f,
                            hull_f,
                        ]
                    )
            if len(descriptors) == 0:
                return np.empty((0, 13), dtype=np.float32)
            return np.asarray(descriptors, dtype=np.float32)

        def _calculate_similarity(
            ref_desc: np.ndarray,
            mov_desc: np.ndarray,
        ):
            """
            Compare descriptor sets using vectorised NumPy operations.

            This replaces extremely slow Python nested loops.
            """

            # Expand dimensions so that every descriptor is compared against every other
            ref = ref_desc[:, None, :]  # 'Ref' will be the outermost axis
            mov = mov_desc[None, :, :]  # 'Mov' will be the next axis in

            # Compare neighbour distance ratios
            ratio_ref = ref[..., 0]
            ratio_mov = mov[..., 0]
            d_ratio = np.abs(ratio_mov - ratio_ref) / (ratio_ref + ratio_mov + eps)

            # Compare signed angles
            angle_ref = ref[..., 1]
            angle_mov = mov[..., 1]
            d_theta = (
                np.abs(
                    np.arctan2(
                        np.sin(angle_mov - angle_ref),
                        np.cos(angle_mov - angle_ref),
                    )
                )
                / np.pi
            )

            # Compare ellipse dimensions
            d_elps_near = np.linalg.norm(mov[..., 2:4] - ref[..., 2:4], axis=-1)
            d_elps_far = np.linalg.norm(mov[..., 6:8] - ref[..., 6:8], axis=-1)
            d_elps = 0.5 * (d_elps_near + d_elps_far)

            # Compare combined contour area-hull area differences
            d_area_near = np.linalg.norm(mov[..., 4:6] - ref[..., 4:6], axis=-1)
            d_area_far = np.linalg.norm(mov[..., 8:10] - ref[..., 8:10], axis=-1)
            d_area = 0.5 * (d_area_near + d_area_far)

            # Calculate the combined feature-space distance
            # Smaller distances indicate larger similarity scores
            D = 4.0 * d_theta + 2.0 * d_ratio + 0.5 * d_elps + 1.0 * d_area
            return np.exp(-D)

        # =============================================================================
        # Start of '_match_features'
        # =============================================================================

        ref_desc = _build_descriptor(ref_features)
        mov_desc = _build_descriptor(mov_features)

        if len(ref_desc) == 0 or len(mov_desc) == 0:
            logger.warning("Could not compile descriptors from the detected features")
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )

        # Save tables if set
        if save_tables and save_dir is not None:
            for desc, file_name in (
                (ref_desc, "ref_desc.tsv"),
                (mov_desc, "mov_desc.tsv"),
            ):
                desc = np.column_stack((np.arange(len(desc)), desc))
                np.savetxt(
                    save_dir / file_name,
                    desc,
                    fmt=[
                        "%-5d",
                        "%-3d",
                        "%-3d",
                        "%-3d",
                        "%-10.3f",
                        "%-10.3f",
                        "%-10.3f",
                        "%-10.3f",
                        "%-10.3f",
                        "%-10.3f",
                        "%-10.3f",
                        "%-10.3f",
                        "%-10.3f",
                        "%-10.3f",
                    ],
                    header=(
                        f"{'':<3} "
                        f"{'o':<3} "
                        f"{'n':<3} "
                        f"{'f':<3} "
                        f"{'d_ratio':<10} "
                        f"{'angle':<10} "
                        f"{'w_near':<10} "
                        f"{'h_near':<10} "
                        f"{'area_near':<10} "
                        f"{'hull_near':<10} "
                        f"{'w_far':<10} "
                        f"{'h_far':<10} "
                        f"{'area_far':<10} "
                        f"{'hull_far':<10} "
                    ),
                )

        # Calculate the similarity scorees between all descriptors
        len_ref, len_mov = len(ref_features), len(mov_features)
        scores = np.zeros((len_ref, len_mov), dtype=np.float32)
        for i in range(len_ref):
            ref_desc_curr = ref_desc[ref_desc[:, 0] == i][:, 3:]
            # Early skip if there are no descriptors for this feature
            if len(ref_desc_curr) == 0:
                continue
            for j in range(len_mov):
                mov_desc_curr = mov_desc[mov_desc[:, 0] == j][:, 3:]
                # Early skip if there are no descriptors for this feature
                if len(mov_desc_curr) == 0:
                    continue

                # Calculate the similarity score between each feature's descriptors
                score_curr = _calculate_similarity(
                    ref_desc_curr,
                    mov_desc_curr,
                )

                # Pick the highest score in mov for each descriptor in ref, then average
                # Do the reverse for mov
                score_ref = np.mean(np.max(score_curr, axis=1))
                score_mov = np.mean(np.max(score_curr, axis=0))
                # Add the average to the score matrix
                scores[i, j] = 0.5 * (score_ref + score_mov)

        if save_tables and save_dir is not None:
            np.savetxt(
                save_dir / "scores.tsv",
                np.column_stack((np.arange(len(scores)), scores)),
                fmt=(["%-5d"] + ["%-10.3f"] * len(mov_features)),
                header=(
                    f"{'':<3} " + "".join(f"{i:<10} " for i in range(len(mov_features)))
                ),
            )

        # Keep only mutual best matches and those above min threshold
        ref_best = np.argmax(scores, axis=1)
        mov_best = np.argmax(scores, axis=0)
        ref_list: list[int] = []
        mov_list: list[int] = []
        score_list: list[float] = []
        for ref_idx, mov_idx in enumerate(ref_best):
            if mov_best[mov_idx] == ref_idx:
                score = float(scores[ref_idx, mov_idx])
                if min_score is not None and score < min_score:
                    continue
                ref_list.append(ref_idx)
                mov_list.append(mov_idx)
                score_list.append(score)
        ref_matches = ref_features[ref_list][:, :2]
        mov_matches = mov_features[mov_list][:, :2]
        logger.info(
            (
                "Found matches:\n"
                + "".join(
                    f"{i} -> {j} ({score})\n"
                    for i, j, score in zip(ref_list, mov_list, score_list)
                )
            )
        )
        return ref_matches, mov_matches

    def _draw_matches(
        ref_features: np.ndarray,
        mov_features: np.ndarray,
        ref_match: np.ndarray,  # x- and y-coordinates
        mov_match: np.ndarray,
    ):
        """
        Overlay the original reference and moving images on top of one another and add
        annotations showing how the features from the reference image map onto those in
        the moving image.
        """

        annotated = cv2.addWeighted(
            src1=reference_array,
            alpha=0.5,
            src2=moving_array,
            beta=0.5,
            gamma=0,
        )
        annotated = cv2.cvtColor(annotated, code=cv2.COLOR_GRAY2BGR)

        # Mark the points
        for (x0, y0), (x1, y1) in zip(ref_match, mov_match):
            x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
            cv2.line(
                annotated,
                pt1=(x0, y0),
                pt2=(x1, y1),
                color=(0, 255, 255),
                thickness=line_thickness,
            )
        for features, color in (
            (ref_features, (0, 255, 0)),
            (mov_features, (0, 0, 255)),
        ):
            for i, (x, y) in enumerate(features[:, :2]):
                x, y = int(x), int(y)
                cv2.circle(
                    annotated,
                    center=(x, y),
                    radius=marker_size,
                    color=color,
                    thickness=line_thickness,
                )
                cv2.putText(
                    annotated,
                    text=f"{i}",
                    org=(x + 20, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=color,
                    thickness=line_thickness,
                    lineType=cv2.LINE_AA,
                )
        _save_image("matched_holes.png", annotated)

    # =================================================================================
    # Start of function
    # =================================================================================

    # Let Python control multithreading
    cv2.setNumThreads(1)

    # Check if save parameters have been set correctly
    if (save_tables or save_images) and save_dir is None:
        logger.warning(
            "No save directory provided even though saving of intermediate results "
            "was specified. Intermediate results will not be saved."
        )

    # Shared variables for use by the nested functions
    eps = 1e-9
    line_thickness = 2
    marker_size = 3

    # Preprocess images to get binaries
    ref_bin = _preprocess(reference_array, name="ref")
    mov_bin = _preprocess(moving_array, name="mov")

    ref_features = _detect_features(ref_bin, name="ref")
    mov_features = _detect_features(mov_bin, name="mov")

    # Run the feature matching algorith
    ref_match, mov_match = _match_features(ref_features, mov_features)
    if len(ref_match) == 0 or len(mov_match) == 0:
        logger.warning("Could not identify matching features between the images")
        return {}
    if save_images:
        _draw_matches(ref_features, mov_features, ref_match, mov_match)

    # Use the matched points to estimate the similarity transform
    M, _ = cv2.estimateAffinePartial2D(
        from_=np.ascontiguousarray(mov_match),
        to=np.ascontiguousarray(ref_match),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
    )
    if M is None:
        raise RuntimeError("Affine transform estimation failed")
    aligned = cv2.warpAffine(
        moving_array, M=M, dsize=(moving_array.shape[1], moving_array.shape[0])
    )
    if save_images:
        overlay = cv2.addWeighted(
            src1=reference_array,
            alpha=0.5,
            src2=aligned,
            beta=0.5,
            gamma=0,
        )
        _save_image("overlay.png", overlay)

    return {
        "aligned": aligned,
        "transform": M,
    }
