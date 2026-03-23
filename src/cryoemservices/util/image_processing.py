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
        np.clip(arr, a_min=vmin, a_max=vmax, out=arr)
        np.subtract(arr, vmin, out=arr)
        np.multiply(arr, scale, out=arr)
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
        np.clip(resized, vmin, vmax, out=resized)
        np.subtract(resized, vmin, out=resized)
        np.multiply(resized, scale, out=resized)
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
