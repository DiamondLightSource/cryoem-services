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

import logging
from ast import literal_eval
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Literal, Optional
from xml.etree import ElementTree as ET

import numpy as np
from defusedxml.ElementTree import parse
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from tifffile import TiffFile, imwrite

from cryoemservices.util.clem_array_functions import (
    align_image_to_reference,
    align_image_to_self,
    convert_to_rgb,
    flatten_image,
    merge_images,
    stretch_image_contrast,
)

# Create logger object to output messages with
logger = logging.getLogger("cryoemservices.wrappers.clem_align_and_merge")


def align_and_merge_stacks(
    images: Path | list[Path],
    metadata: Optional[Path] = None,
    crop_to_n_frames: Optional[int] = None,
    align_self: Literal["enabled", ""] = "",
    flatten: Optional[Literal["min", "max", "mean", ""]] = "mean",
    align_across: Literal["enabled", ""] = "",
    # Print messages only if run as a CLI
    print_messages: bool = False,
    debug: bool = False,
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

    # Use shorter inputs in function
    files = images

    # Turn single entry into a list
    if isinstance(files, (Path, str)):
        files = [files]

    # Check that a value has been provided
    if len(files) == 0:
        logger.error("No image stack file paths have been provided")
        raise ValueError

    # Convert to Path object if a string was provided
    files = [Path(file) if isinstance(file, str) else file for file in files]

    # Check that files have the same parent directories
    if len({file.parents[0] for file in files}) > 1:
        logger.error(
            "The files provided come from different directories, and might not "
            "be part of the same series"
        )
        raise ValueError

    # Get parent directory
    parent_dir = list({file.parents[0] for file in files})[0]
    # Validate parent directory
    if print_messages is True:
        print(f"Setting {str(parent_dir)!r} as the working directory")

    # Find metadata file if none was provided
    if metadata is None:
        # Raise error if no files found
        if len(list((parent_dir / "metadata").glob("*.xml"))) == 0:
            logger.error("No metadata file was found at the default directory")
            raise FileNotFoundError
        # Raise error if too many files found
        if len(list((parent_dir / "metadata").glob("*.xml"))) > 1:
            logger.error(
                "More than one metadata file was found at the default directory"
            )
            raise Exception
        # Load metadata file
        metadata = list((parent_dir / "metadata").glob("*.xml"))[0]
        if print_messages is True:
            print(f"Using metadata from {str(metadata)!r}")

    # Load metadata for series from XML file
    xml_metadata: ET.ElementTree = parse(metadata).getroot()

    # Get order of colors as shown in metadata
    channels = xml_metadata.findall(
        "Data/Image/ImageDescription/Channels/ChannelDescription"
    )
    colors: list[str] = [
        channels[c].attrib["LUTName"].lower() for c in range(len(channels))
    ]
    # Use grey image as reference and put that first in the list
    # If a grey image is not present, the first image will be used as reference
    colors = sorted(
        colors, key=lambda c: (0, c) if c.lower() in ("gray", "grey") else (1, c)
    )
    if print_messages is True:
        print("Loaded metadata from file")

    # Load image stacks according to their order in the XML file
    arrays: list[np.ndarray] = []

    # ImageJ metadata to load and pass on
    resolution_list: list[tuple[float, float]] = []
    spacing_list: list[float] = []
    units_list: list[str] = []
    colors_to_process: list[str] = []
    for c in range(len(colors)):
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

        with TiffFile(file) as tiff_file:
            # Check that there aren't multiple series in the image stack
            if len(tiff_file.series) > 1:
                raise ValueError(
                    "The image stack provided contains more than one series"
                )

            # Load array
            array = tiff_file.series[0].pages.asarray()

            # Crop array to middle n frames if selected and a stack is provided
            if (
                not len(array.shape) < 3  # 2D grayscale image
                or not (
                    len(array.shape) == 3  # 2D RGB/RGBA image
                    and array.shape[-1] in (3, 4)
                )
            ) and isinstance(crop_to_n_frames, int):
                m = len(array) // 2
                n1 = crop_to_n_frames // 2
                n2 = n1 + 1 if crop_to_n_frames % 2 == 1 else n1
                f1 = m - n1 if (m - n1) >= 0 else 0
                f2 = m + n2 if (m + n2) <= len(array) else len(array)
                array = array[f1:f2]
                logger.debug(
                    f"Image stack cropped to central {crop_to_n_frames} frames"
                )

            # Append array and its corresponding colour
            arrays.append(array)
            colors_to_process.append(color)
            if print_messages is True:
                print(f"Loaded {str(file)!r}")

            # Load and append ImageJ metadata
            ij_metadata = tiff_file.imagej_metadata

            resolution_list.append(tiff_file.series[0][0].resolution)
            spacing_list.append(ij_metadata.get("spacing", float(0)))
            units_list.append(ij_metadata.get("unit", ""))

    # Use shortened list for subsequent processing
    if len(colors_to_process) == 0:
        raise ValueError(
            "No files corresponded to the colour channels present in this series"
        )
    colors = colors_to_process

    # Add file name component to describe type of composite image being generated
    img_type = (
        "BF_FL"  # Bright field + fluorescent
        if any(color in colors for color in ("gray", "grey"))
        else "FL"  # Fluorescent only
    )

    # Check that images have the same pixel calibration
    if len(set(resolution_list)) > 1:
        logger.error("The image stacks provided do not have the same resolution")
        raise ValueError

    if len(set(spacing_list)) > 1:
        logger.error("The image stacks provided do not have the same z-spacing")
        raise ValueError

    if len(set(units_list)) > 1:
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

    # Debug
    if debug and print_messages:
        print(
            "Initial properties of image stacks: \n",
            f"Shape: {arrays[0].shape} \n",
            f"dtype: {arrays[0].dtype} \n",
            f"Min: {np.min(arrays)} \n",
            f"Max: {np.max(arrays)} \n",
        )
    logger.debug(
        "Initial properties of image stacks: \n",
        f"Shape: {arrays[0].shape} \n",
        f"dtype: {arrays[0].dtype} \n",
        f"Min: {np.min(arrays)} \n",
        f"Max: {np.max(arrays)} \n",
    )

    # Align frames within each image stack
    if align_self:
        if print_messages is True:
            print("Correcting for drift in images...")
        with Pool(len(arrays)) as pool:
            arrays = pool.starmap(
                align_image_to_self,
                [(arr, "middle") for arr in arrays],
            )
        if print_messages is True:
            print(" Done")

    # Flatten images if the option is selected
    if flatten:
        if print_messages is True:
            print("Flattening image stacks...")
        with Pool(len(arrays)) as pool:
            arrays = pool.starmap(
                flatten_image,
                [(arr, flatten) for arr in arrays],
            )
        # Validate that image flattening was done correctly
        if len({arr.shape for arr in arrays}) > 1:
            logger.error("The flattened arrays do not have the same shape")
            raise ValueError
        if print_messages is True:
            print(" Done")

        # # Debug
        if debug and print_messages:
            print(
                "Properties of array after flattening: \n",
                f"Shape: {arrays[0].shape} \n",
                f"dtype: {arrays[0].dtype} \n",
                f"Min: {np.min(arrays)} \n",
                f"Max: {np.max(arrays)} \n",
            )
        logger.debug(
            "Properties of array after flattening: \n",
            f"Shape: {arrays[0].shape} \n",
            f"dtype: {arrays[0].dtype} \n",
            f"Min: {np.min(arrays)} \n",
            f"Max: {np.max(arrays)} \n",
        )

    # Align other stacks to reference stack
    if align_across:
        if print_messages is True:
            print("Aligning images to reference image...")
        reference = arrays[0]  # First image in list is the reference image
        to_align = arrays[1:]
        with Pool(len(to_align)) as pool:
            aligned = pool.starmap(
                align_image_to_reference, [(reference, moving) for moving in to_align]
            )
        arrays = [reference, *aligned]
        if print_messages is True:
            print(" Done")

    # Colourise images
    if print_messages is True:
        print("Converting images from grayscale to RGB...")
    with Pool(len(arrays)) as pool:
        arrays = pool.starmap(
            convert_to_rgb, [(arrays[c], colors[c]) for c in range(len(colors))]
        )
    if print_messages is True:
        print(" Done")

    # Debug
    if debug and print_messages:
        print(
            "Properties of array after colourising: \n",
            f"Shape: {[arr.shape for arr in arrays]} \n",
            f"dtype: {[arr.dtype for arr in arrays]} \n",
            f"Min: {[np.min(arr) for arr in arrays]} \n",
            f"Max: {[np.max(arr) for arr in arrays]} \n",
        )
    logger.debug(
        "Properties of array after colourising: \n",
        f"Shape: {[arr.shape for arr in arrays]} \n",
        f"dtype: {[arr.dtype for arr in arrays]} \n",
        f"Min: {[np.min(arr) for arr in arrays]} \n",
        f"Max: {[np.max(arr) for arr in arrays]} \n",
    )

    # Convert to a composite image
    if print_messages is True:
        print("Creating a composite image...")
    composite_img = merge_images(arrays)
    if print_messages is True:
        print(" Done")

    # # Debug
    if debug and print_messages:
        print(
            "Properties of array after merging: \n",
            f"Shape: {arrays[0].shape} \n",
            f"dtype: {arrays[0].dtype} \n",
            f"Min: {np.min(arrays)} \n",
            f"Max: {np.max(arrays)} \n",
        )
    logger.debug(
        "Properties of array after merging: \n",
        f"Shape: {arrays[0].shape} \n",
        f"dtype: {arrays[0].dtype} \n",
        f"Min: {np.min(arrays)} \n",
        f"Max: {np.max(arrays)} \n",
    )

    # Adjust image contrast after merging images
    if print_messages is True:
        print("Applying contrast correction...")
    composite_img = stretch_image_contrast(
        composite_img,
        percentile_range=(0, 100),
    )
    if print_messages is True:
        print(" Done")

    # Debug
    if debug and print_messages:
        print(
            "Properties of array after stretching: \n",
            f"Shape: {composite_img.shape} \n",
            f"dtype: {composite_img.dtype} \n",
            f"Min: {np.min(composite_img)} \n",
            f"Max: {np.max(composite_img)} \n",
        )
    logger.debug(
        "Properties of array after stretching: \n",
        f"Shape: {composite_img.shape} \n",
        f"dtype: {composite_img.dtype} \n",
        f"Min: {np.min(composite_img)} \n",
        f"Max: {np.max(composite_img)} \n",
    )

    # Prepare to save image as a TIFF file
    if print_messages is True:
        print("Saving composite image...")

    # Set up metadata properties
    final_shape = composite_img.shape
    resolution = resolution_list[0]
    units = units_list[0]
    extended_metadata = ""
    # image_labels = None

    if flatten:
        axes = "YXS"
        z_size = None
    else:
        axes = "ZYXS"
        z_size = spacing_list[0]

    # Save image as a TIFF file
    save_name = parent_dir / f"composite_{img_type}.tiff"
    imwrite(
        save_name,
        composite_img,
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
            "min": round(composite_img.min(), 1),
            "max": round(composite_img.max(), 1),
            "Info": extended_metadata,
            # "Labels": image_labels,
        },
    )
    if print_messages is True:
        print(f"Composite image saved as {str(save_name)!r}")

    # Collect and return parameters and result
    result: dict[str, Any] = {
        "image_stacks": [str(file) for file in files],  # Convert Path to str
        "align_self": align_self,
        "flatten": flatten,
        "align_across": align_across,
        "composite_image": str(save_name.resolve()),
    }
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
    def wrap_images(cls, model: AlignAndMergeParameters):
        """
        Wrap single images in a list to standardise what's passed on to the align-and-
        merge function.
        """
        if isinstance(model.images, Path):
            model.images = [model.images]
        return model

    @field_validator("crop_to_n_frames", mode="before")
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
        result = align_and_merge_stacks(
            images=params.images,
            metadata=params.metadata,
            crop_to_n_frames=params.crop_to_n_frames,
            align_self=params.align_self,
            flatten=params.flatten,
            align_across=params.align_across,
        )
        if not result.keys():
            logger.error(
                "Failed to complete the aligning and merging process for "
                f"{params.series_name!r}"
            )
            return False

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
