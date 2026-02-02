"""
Contains functions needed in order to run a service to align the image stacks
generated from CLEM data. This service will include:
    1. Image alignment within individual stacks
    2. Flattening of 3D image stacks into 2D images (optional)
    3. Image alignment across stacks (optional)
    3. Colorisation of images according to their channels
    4. Merging of images to create a colored composite

"""

from __future__ import annotations

import json
import logging
import time
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from defusedxml.ElementTree import parse
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from tifffile import TiffFile, imwrite

from cryoemservices.util.clem_array_functions import (
    align_image_to_reference,
    align_image_to_self,
    convert_to_rgb,
    flatten_image,
    is_grayscale_image,
    is_image_stack,
    merge_images,
)
from cryoemservices.util.clem_metadata import get_channel_info

# Create logger object to output messages with
logger = logging.getLogger("cryoemservices.wrappers.clem_align_and_merge")


def _align_image_to_self_worker(
    shm_metadata: dict[str, Any],
    start_from: Literal["beginning", "middle", "end"] = "beginning",
):
    shm = SharedMemory(name=shm_metadata["name"])
    array = np.ndarray(
        shape=shm_metadata["shape"],
        dtype=shm_metadata["dtype"],
        buffer=shm.buf,
    )
    result = align_image_to_self(array, start_from=start_from)
    shm.close()
    return result


def _flatten_image_worker(
    shm_metadata,
    mode: Literal["mean", "min", "max"] = "mean",
):
    shm = SharedMemory(name=shm_metadata["name"])
    array = np.ndarray(
        shape=shm_metadata["shape"],
        dtype=shm_metadata["dtype"],
        buffer=shm.buf,
    )
    result = flatten_image(array, mode)
    shm.close()
    return result


def _align_image_to_reference_worker(
    reference_shm_metadata: dict[str, Any],
    moving_shm_metadata: dict[str, Any],
    downsample_factor: int = 2,
):
    # Load shared arrays from memory
    ref_shm = SharedMemory(name=reference_shm_metadata["name"])
    ref = np.ndarray(
        shape=reference_shm_metadata["shape"],
        dtype=reference_shm_metadata["dtype"],
        buffer=ref_shm.buf,
    )
    mov_shm = SharedMemory(name=moving_shm_metadata["name"])
    mov = np.ndarray(
        shape=moving_shm_metadata["shape"],
        dtype=moving_shm_metadata["dtype"],
        buffer=mov_shm.buf,
    )
    result = align_image_to_reference(ref, mov, downsample_factor)
    ref_shm.close()
    mov_shm.close()
    return result


def _convert_to_rgb_worker(
    shm_metadata: dict[str, Any],
    color: str,
):
    shm = SharedMemory(name=shm_metadata["name"])
    array = np.ndarray(
        shape=shm_metadata["shape"],
        dtype=shm_metadata["dtype"],
        buffer=shm.buf,
    )
    result = convert_to_rgb(array, color)
    shm.close()
    return result


def align_and_merge_stacks(
    images: Path | list[Path],
    metadata: Optional[Path] = None,
    crop_to_n_frames: Optional[int] = None,
    align_self: Literal["enabled", ""] = "",
    flatten: Optional[Literal["min", "max", "mean", ""]] = "mean",
    align_across: Literal["enabled", ""] = "",
    num_procs: int = 4,
) -> dict[str, Any]:
    """
    A cryoemservices wrapper to create composite images from component image stack in
    the series.

    This will (eventually) take a message (JSON dictionary) containing the image stack
    and metadata files associated with an image series, along with settings for how
    the stacks should be processed (a flattened 2D image or a 3D image stack).

    The order of processing will be as follows:
    - Crop image stacks to the middle n frames
    - Align the images within a stack
    - Flatten the stack (depends on whether a 2D or 3D composite image is needed)
    - Align image stacks for a given position to one another
    - Colourise the image stacks
    - Merge them together to create a composite image/stack
    """

    start_time = time.perf_counter()

    # Validate inputs before proceeding further
    if crop_to_n_frames is not None and not isinstance(crop_to_n_frames, int):
        message = "Incorrect value provided for 'crop_to_n_frames' parameter"
        logger.error(message)
        raise ValueError(message)

    if align_self not in ("enabled", ""):
        message = "Incorrect value provided for 'align_self' parameter"
        logger.error(message)
        raise ValueError(message)

    if flatten not in ("mean", "min", "max", ""):
        message = "Incorrect value provided for 'flatten' parameter"
        logger.error(message)
        raise ValueError(message)

    if align_across not in ("enabled", ""):
        message = "Incorrect value provided for 'align_across' parameter"
        logger.error(message)
        raise ValueError(message)

    # Standardise into a list of Paths
    files = [images] if isinstance(images, (Path, str)) else images
    files = [Path(file) if isinstance(file, str) else file for file in files]

    # Check that a value has been provided
    if len(files) == 0:
        logger.error("No image stack file paths have been provided")
        raise ValueError

    # Check that files have the same parent directories
    if len({file.parents[0] for file in files}) > 1:
        logger.error(
            "The files provided come from different directories, and might not "
            "be part of the same series"
        )
        raise ValueError

    # Get parent directory
    parent_dir = files[0].parent
    # Validate parent directory
    logger.info(f"Setting {str(parent_dir)!r} as the working directory")

    # Find metadata file if none was provided
    if metadata is None:
        # Raise error if no files found
        if (
            len(
                (
                    metadata_search_result := list(
                        (parent_dir / "metadata").glob("*.xml")
                    )
                )
            )
            == 0
        ):
            logger.error("No metadata file was found at the default directory")
            raise FileNotFoundError
        # Raise error if too many files found
        if len(metadata_search_result) > 1:
            logger.error(
                "More than one metadata file was found at the default directory"
            )
            raise Exception
        # Load metadata file
        metadata = metadata_search_result[0]
        logger.info(f"Using metadata from {str(metadata)!r}")

    # Load color channel information
    channels = get_channel_info(parse(metadata))

    # Use grey image as reference and put that first in the list
    # If a grey image is not present, the first image will be used as reference
    colors = sorted(
        channels.keys(),
        key=lambda c: (0, c) if c.lower() in ("gray", "grey") else (1, c),
    )
    logger.info(f"Successfully loaded metadata from {str(metadata)!r}")

    # Load image stacks according to their order in the XML file
    arrays: list[np.ndarray] = []

    # ImageJ metadata to load and pass on
    resolution_list: set[tuple[float, float]] = set()
    spacing_list: set[float] = set()
    units_list: set[str] = set()
    colors_to_process: list[str] = []

    load_start_time = time.perf_counter()
    for c, color in enumerate(colors):
        color = colors[c]
        file_search = [file for file in files if color in file.stem]
        # Handle exceptions in search results
        if len(file_search) == 0:
            logger.info(f"No file provided for {color!r} channel; omitting it")
            continue
        if len(file_search) > 1:
            logger.error("More than one file provided that matches this colour")
            raise Exception
        file = file_search[0]

        logger.info(f"Loading {str(file)!r}")
        with TiffFile(file) as tiff_file:
            # Check that there aren't multiple series in the image stack
            if len(tiff_file.series) > 1:
                raise ValueError(
                    "The image stack provided contains more than one series"
                )

            # Load array
            array = tiff_file.series[0].pages.asarray()

            # Crop array to middle n frames if selected and a stack is provided
            if is_image_stack(array) and isinstance(crop_to_n_frames, int):
                m = len(array) // 2
                n1 = crop_to_n_frames // 2
                n2 = n1 + crop_to_n_frames % 2
                f1 = m - n1 if (m - n1) >= 0 else 0
                f2 = m + n2 if (m + n2) <= len(array) else len(array)
                array = array[f1:f2]

            # Append array and its corresponding colour
            arrays.append(array)
            colors_to_process.append(color)
            logger.info(f"Loaded {str(file)!r}")

            # Load and append ImageJ metadata
            ij_metadata = tiff_file.imagej_metadata
            resolution_list.add(tiff_file.series[0][0].resolution)
            spacing_list.add(ij_metadata.get("spacing", float(0)))
            units_list.add(ij_metadata.get("unit", ""))
    load_end_time = time.perf_counter()
    logger.debug(
        f"Loaded {len(arrays)} images with shape {arrays[0].shape} in "
        f"{load_end_time - load_start_time}s"
    )

    # Use shortened list for subsequent processing
    if len(colors_to_process) == 0:
        raise ValueError(
            "No files corresponded to the colour channels present in this series"
        )
    colors = colors_to_process

    # Add file name component to describe type of composite image being generated
    if any(color in colors for color in ("gray", "grey")):
        img_type = "BF" if len(colors) == 1 else "BF_FL"
    else:
        img_type = "FL"

    # Check that images have the same pixel calibration
    if len(resolution_list) > 1:
        logger.error("The image stacks provided do not have the same resolution")
        raise ValueError

    if len(spacing_list) > 1:
        logger.error("The image stacks provided do not have the same z-spacing")
        raise ValueError

    if len(units_list) > 1:
        logger.error("The image stacks provided do not have the same units")
        raise ValueError

    # Validate that the stacks provided are of the same shape
    if len({arr.shape for arr in arrays}) > 1:
        logger.error("The image stacks provided do not have the same shape")
        raise ValueError

    # Get the dtype of the image
    if len({str(arr.dtype) for arr in arrays}) > 1:
        logger.error("The image stacks provided do not have the same dtype")
        raise ValueError

    # Align frames within each image stack
    if align_self:
        # Perform drift correction only if they are image stacks
        if all(is_image_stack(array) for array in arrays):
            logger.info("Correcting for drift in images")
            drift_correction_start_time = time.perf_counter()

            # Determine a suitable downsampling factor to use
            if (
                num_pixels := np.prod(
                    (
                        arrays[0][0].shape[:2]
                        if is_image_stack(arrays[0])
                        else arrays[0].shape[:2]
                    )
                )
            ) > 4096**2:
                downsample_factor = 8
            elif num_pixels > 2048**2:
                downsample_factor = 4
            elif num_pixels > 1024**2:
                downsample_factor = 2
            else:
                downsample_factor = 1

            # Align image with multithreading
            arrays = [
                align_image_to_self(
                    array,
                    "middle",
                    num_procs=num_procs,
                )
                for array in arrays
            ]

            drift_correction_end_time = time.perf_counter()
            logger.info("Successfully applied drift correction")
            logger.debug(
                f"Applied drift correction to {len(arrays)} images "
                f"of shape {arrays[0].shape} "
                f"in {drift_correction_end_time - drift_correction_start_time}s"
            )
        else:
            logger.info(
                "Skipping drift correction step as no image stacks were provided"
            )

    # Flatten images if the option is selected
    if flatten:
        # Only flatten if they are image stacks
        if all(is_image_stack(array) for array in arrays):
            logger.info("Flattening image stacks")
            flatten_start_time = time.perf_counter()

            with ThreadPoolExecutor(max_workers=num_procs) as pool:
                arrays = list(
                    pool.map(
                        lambda p: flatten_image(*p),
                        [(arr, flatten) for arr in arrays],
                    )
                )
            if len({arr.shape for arr in arrays}) > 1:
                logger.error("The flattened arrays do not have the same shape")
                raise ValueError
            flatten_end_time = time.perf_counter()
            logger.info("Successfully flattened images")
            logger.debug(
                f"Flattened {len(arrays)} images in {flatten_end_time - flatten_start_time}s"
            )
            logger.debug(f"Images now have the following shape: {arrays[0].shape}")
        else:
            logger.info(
                "Skipping image flattening step as no image stacks were provided"
            )

    # Align other stacks to reference stack
    if align_across:
        if len(arrays) > 1:
            logger.info(
                f"Aligning images using {colors[0]!r} channel as the reference image"
            )
            align_across_start_time = time.perf_counter()

            reference = arrays[0]  # First image in list is the reference image
            to_align = arrays[1:]

            # Determine suitable combinations of parameters to use
            # Ensure that there are at least 8192 pixels sampled at coarsest level
            if (
                num_pixels := np.prod(
                    (
                        arrays[0][0].shape[:2]
                        if is_image_stack(arrays[0])
                        else arrays[0].shape[:2]
                    )
                )
            ) > 4096**2:
                downsample_factor = 4
                sampling_percentage = 0.125
                shrink_factors_per_level = [4, 2]
            elif num_pixels > 2048**2:
                downsample_factor = 2
                sampling_percentage = 0.125
                shrink_factors_per_level = [4, 2]
            elif num_pixels > 1024**2:
                downsample_factor = 2
                sampling_percentage = 0.5
                shrink_factors_per_level = [4, 2]
            elif num_pixels > 512**2:
                downsample_factor = 2
                sampling_percentage = 0.5
                shrink_factors_per_level = [2, 1]
            elif num_pixels > 256**2:
                downsample_factor = 2
                sampling_percentage = 0.5
                shrink_factors_per_level = [1]
            # Image is small enough to sample everything
            else:
                downsample_factor = 1
                sampling_percentage = 1.0
                shrink_factors_per_level = [1]
            smoothing_sigmas_per_level = [s / 2 for s in shrink_factors_per_level]

            # Align images to reference with multithreading
            aligned = [
                align_image_to_reference(
                    reference_array=reference,
                    moving_array=arr,
                    downsample_factor=downsample_factor,
                    sampling_percentage=sampling_percentage,
                    shrink_factors_per_level=shrink_factors_per_level,
                    smoothing_sigmas_per_level=smoothing_sigmas_per_level,
                    num_procs=num_procs,
                )
                for arr in to_align
            ]

            arrays = [reference, *aligned]
            align_across_end_time = time.perf_counter()
            logger.info(f"Successfully aligned images to {colors[0]!r} channel")
            logger.debug(
                f"Aligned {len(to_align)} images of shape {reference.shape} "
                f" in {align_across_end_time - align_across_start_time}s"
            )
        elif len(arrays) == 1:
            logger.info("Skipping image alignment step as there is only one image")
        else:
            logger.warning("No image arrays are present")

    # Colourise images
    if all(is_grayscale_image(array) for array in arrays):
        convert_rgb_start_time = time.perf_counter()
        logger.info("Converting images from grayscale to RGB")
        with ThreadPoolExecutor(max_workers=num_procs) as pool:
            arrays = list(
                pool.map(
                    lambda p: convert_to_rgb(*p),
                    [(arrays[c], color) for c, color in enumerate(colors)],
                )
            )

        convert_rgb_end_time = time.perf_counter()
        logger.info("Successfully colorised images")
        logger.debug(
            f"Converted {len(arrays)} images to RGB "
            f"in {convert_rgb_end_time - convert_rgb_start_time}s"
        )
    else:
        logger.info("Skipping image colorisation step as they are not grayscale")

    # Convert to a composite image
    logger.info("Creating a composite image")
    merge_start_time = time.perf_counter()
    composite_img = merge_images(arrays)
    merge_end_time = time.perf_counter()
    logger.info("Successfully merged images")
    logger.debug(f"Completed merge in {merge_end_time - merge_start_time}s")

    # Adjust image contrast and convert to 8-bit
    logger.info("Applying contrast correction and converting to 8-bit")
    contrast_correction_start_time = time.perf_counter()
    vmin, vmax = composite_img.min(), composite_img.max()
    scale = 255 / (vmax - vmin)
    np.clip(composite_img, a_min=vmin, a_max=vmax, out=composite_img)
    np.subtract(composite_img, vmin, out=composite_img)
    np.multiply(composite_img, scale, out=composite_img, casting="unsafe")
    composite_img = composite_img.astype(dtype=np.uint8, copy=False)
    contrast_correction_end_time = time.perf_counter()
    logger.info("Successfully adjusted image contrast")
    logger.debug(
        f"Converted image in {contrast_correction_end_time - contrast_correction_start_time}s"
    )

    # Prepare to save image as a TIFF file
    logger.info("Saving composite image")

    # Set up metadata properties
    final_shape = composite_img.shape
    resolution = list(resolution_list)[0]
    units = list(units_list)[0]
    extended_metadata = ""
    image_labels = [str(f) for f in range((1 if flatten else composite_img.shape[0]))]

    if flatten:
        axes = "YXS"
        z_size = None
    else:
        axes = "ZYXS"
        z_size = list(spacing_list)[0]

    # Save image as a TIFF file
    save_name = (parent_dir / f"composite_{img_type}.tiff").resolve()
    imwrite(
        save_name,
        composite_img,
        bigtiff=False if flatten else True,
        # Array properties
        shape=final_shape,
        dtype=str(composite_img.dtype),
        resolution=resolution,
        resolutionunit=None,
        # Colour properties
        photometric="rgb",
        colormap=None,
        # ImageJ compatibility
        imagej=True,
        metadata={
            "axes": axes,
            "unit": units,
            "spacing": z_size,
            "loop": False,
            "min": round(float(composite_img.min()), 1),
            "max": round(float(composite_img.max()), 1),
            "Info": extended_metadata,
            "Labels": image_labels,
        },
    )
    logger.info(f"Composite image saved as {str(save_name)!r}")

    # Collect and return parameters and result
    result: dict[str, Any] = {
        "image_stacks": [str(file) for file in files],  # Convert Path to str
        "align_self": align_self,
        "flatten": flatten,
        "align_across": align_across,
        "output_file": str(save_name),
        "thumbnail": str(save_name.parent / ".thumbnails" / f"{save_name.stem}.png"),
        "thumbnail_size": (512, 512),  # height, row
    }
    logger.debug(
        "Will return the following result: \n"
        f"{json.dumps(result, indent=2, default=str)}"
    )

    end_time = time.perf_counter()
    logger.debug(f"Completed align and merge job in {end_time - start_time}s")
    return result


class AlignAndMergeParameters(BaseModel):
    series_name: str
    images: Path | list[Path]
    metadata: Optional[Path] = Field(default=None)
    crop_to_n_frames: Optional[int] = Field(default=None)
    align_self: Literal["enabled", ""] = Field(default="")
    flatten: Literal["mean", "min", "max", ""] = Field(default="mean")
    align_across: Literal["enabled", ""] = Field(default="")

    @field_validator("images", mode="before")
    @classmethod
    def parse_images(cls, value):
        if isinstance(value, str):
            # Check for stringified list
            if value.startswith("[") and value.endswith("]"):
                try:
                    eval_images: list[str] = literal_eval(value)
                    images = [Path(file) for file in eval_images]
                    return images
                except (SyntaxError, ValueError):
                    logger.error("Unable to parse stringified list for file paths")
                    raise ValueError
        # Leave the value as-is; if it fails, it fails
        return value

    @model_validator(mode="after")
    @classmethod
    def wrap_images(cls, model: AlignAndMergeParameters):
        """
        Wrap single images in a list to standardise what's passed on to the align-and-
        merge function.
        """
        if isinstance(model.images, Path):
            model.images = [model.images]
        return model

    @field_validator("crop_to_n_frames", mode="before")
    @classmethod
    def parse_for_None(cls, value):
        """
        Convert incoming "None" into None.
        """
        if value == "None":
            return None
        return value


class AlignAndMergeWrapper:
    def __init__(self, recwrap):
        self.log = logging.LoggerAdapter(logger)
        self.recwrap = recwrap

    def run(self) -> bool:
        """
        Reads the Zocalo wrapper recipe, loads the parameters, and pass them to the
        alignment and merging function. Upon collecting the results, it then passes
        them back to Murfey for the next stage in the workflow.
        """

        params_dict = self.recwrap.recipe_step["job_parameters"]
        try:
            params = AlignAndMergeParameters(**params_dict)
        except (ValidationError, TypeError) as error:
            logger.error(
                "AlignAndMergeParameters validation failed for parameters: "
                f"{params_dict} with exception: {error}"
            )
            return False

        # Process files and collect output
        try:
            result = align_and_merge_stacks(
                images=params.images,
                metadata=params.metadata,
                crop_to_n_frames=params.crop_to_n_frames,
                align_self=params.align_self,
                flatten=params.flatten,
                align_across=params.align_across,
            )
        # Log error and return False if the command fails to execute
        except Exception:
            logger.error(
                f"Exception encountered while aligning and merging images for series {params.series_name!r}: \n",
                exc_info=True,
            )
            return False
        if not result:
            logger.error(
                "No image alignment and merging results were returned for series "
                f"{params.series_name!r}"
            )
            return False

        # Request for PNG image to be created
        images_params = {
            "image_command": "tiff_to_apng",
            "input_file": result["output_file"],
            "output_file": result["thumbnail"],
            "target_size": result["thumbnail_size"],
        }
        self.recwrap.send_to(
            "images",
            images_params,
        )
        logger.info(f"Submitted the following job to Images service: \n{images_params}")

        # Send results to Murfey for registration
        result["series_name"] = params.series_name
        murfey_params = {
            "register": "clem.register_align_and_merge_result",
            "result": result,
        }
        self.recwrap.send_to("murfey_feedback", murfey_params)
        logger.info(
            f"Submitted alignment and merging result for {result['series_name']!r} "
            "to Murfey for registration"
        )
        return True
