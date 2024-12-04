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
from pathlib import Path
from typing import Any, Literal, Optional
from xml.etree import ElementTree as ET

import numpy as np
from defusedxml.ElementTree import parse
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from tifffile import TiffFile, imwrite
from zocalo.wrapper import BaseWrapper

from cryoemservices.util.clem_array_functions import (
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
    align_self: Optional[str] = None,
    flatten: Optional[Literal["min", "max", "mean"]] = "mean",
    align_across: Optional[str] = None,
    # Print messages only if run as a CLI
    print_messages: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    """
    A cryoEM service (eventually) to create composite images from component image
    stacks in the series.

    This will (eventually) take a message (JSON dictionary) containing the image stack
    and metadata files associated with an image series, along with settings for how
    the stacks should be processed (a flattened 2D image or a 3D image stack).

    The order of processing will be as follows:
    - Align the images within a stack
    - Flatten the stack (depends on whether a 2D or 3D composite image is needed)
    - Align image stacks for a given position to one another
    - Colourise the image stacks
    - Merge them together to create a composite image/stack
    """

    # Validate inputs before proceeding further
    if flatten is not None and flatten not in ("min", "max", "mean"):
        logger.error("Incorrect value provided for 'flatten' parameter")
        raise ValueError

    # Use shorter inputs in function
    files = images

    # Turn single entry into a list
    if isinstance(files, Path):
        files = [files]
    # Check that a value has been provided
    if len(files) == 0:
        logger.error("No image stack file paths have been provided")
        raise ValueError

    # Check that files have the same parent directories
    if len({file.parents[0] for file in files}) > 1:
        raise Exception(
            "The files provided come from different directories, and might not "
            "be part of the same series"
        )
    # Get parent directory
    parent_dir = list({file.parents[0] for file in files})[0]
    # Validate parent directory
    ## TO DO
    if print_messages is True:
        print(f"Setting {parent_dir!r} as the working directory")

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
            print(f"Using metadata from {metadata!r}")

    # Load metadata for series from XML file
    xml_metadata: ET.ElementTree = parse(metadata).getroot()

    # Get order of colors as shown in metadata
    channels = xml_metadata.findall(
        "Data/Image/ImageDescription/Channels/ChannelDescription"
    )
    colors: list[str] = [
        channels[c].attrib["LUTName"].lower() for c in range(len(channels))
    ]
    if print_messages is True:
        print("Loaded metadata from file")

    # Load image stacks according to their order in the XML file
    arrays: list[np.ndarray] = []
    for c in range(len(colors)):
        color = colors[c]
        file_search = [file for file in files if color in file.stem]
        # Handle exceptions in search results
        if len(file_search) == 0:
            logger.error("No files provided that match this colour")
            raise FileNotFoundError
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

            # Load array and apend it to stack
            arrays.append(tiff_file.series[0].pages.asarray())
            if print_messages is True:
                print(f"Loaded {file!r}")

    # Validate that the stacks provided are of the same shape
    if len({arr.shape for arr in arrays}) > 1:
        logger.error("The image stacks provided do not have the same shape")
        raise Exception

    # Get the dtype of the image
    if len({str(arr.dtype) for arr in arrays}) > 1:
        logger.error("The image stacks do not have the same dtype")
        raise Exception

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
    #    TO DO
    #    NOTE: This could potentially be a separate cluster submission job.
    #    Image alignment could potentially take a while, which is not ideal
    #    for a container service.

    # Flatten images if the option is selected
    if flatten is not None:
        if print_messages is True:
            print("Flattening image stacks...")
        arrays = [flatten_image(arr, mode=flatten) for arr in arrays]
        # Validate that image flattening was done correctly
        if len({arr.shape for arr in arrays}) > 1:
            logger.error("The flattened arrays do not have the same shape")
            raise Exception
        if print_messages is True:
            print("Done")

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
    ## TO DO
    ## NOTE: This could also potentially be a separate cluster submission job
    ## as well due to how long image alignment could take
    ## NOTE: Having this step after the flattening step could potentially save on compute
    ## if only 2D images are required

    # Colourise images
    if print_messages is True:
        print("Converting images from grayscale to RGB...")
    arrays = [convert_to_rgb(arrays[c], colors[c]) for c in range(len(colors))]
    if print_messages is True:
        print("Done")

    # # Debug
    if debug and print_messages:
        print(
            "Properties of array after colourising: \n",
            f"Shape: {arrays[0].shape} \n",
            f"dtype: {arrays[0].dtype} \n",
            f"Min: {np.min(arrays)} \n",
            f"Max: {np.max(arrays)} \n",
        )
    logger.debug(
        "Properties of array after colourising: \n",
        f"Shape: {arrays[0].shape} \n",
        f"dtype: {arrays[0].dtype} \n",
        f"Min: {np.min(arrays)} \n",
        f"Max: {np.max(arrays)} \n",
    )

    # Convert to a composite image
    if print_messages is True:
        print("Creating a composite image...")
    composite_img = merge_images(arrays)
    if print_messages is True:
        print("Done")

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
        print("Done")

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
    # Save the image to a TIFF file
    if print_messages is True:
        print("Saving composite image...")

    # Set up metadata properties
    final_shape = composite_img.shape
    units = "micron"
    extended_metadata = ""
    # image_labels = None

    if flatten is None:
        axes = "ZYXS"
        z_size = 1
    else:
        axes = "YXS"
        z_size = None

    save_name = parent_dir / "composite.tiff"
    imwrite(
        save_name,
        composite_img,
        # Array properties
        shape=final_shape,
        dtype=str(composite_img.dtype),
        resolution=None,
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
        print(f"Composite image saved as {save_name}")

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
    align_self: Optional[str] = Field(default=None)
    flatten: Optional[Literal["min", "max", "mean"]] = Field(default="mean")
    align_across: Optional[str] = Field(default=None)

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

    @field_validator("align_self", "flatten", "align_across", mode="before")
    def parse_for_null(cls, value):
        """
        Convert incoming "null" keyword into None.
        """
        if value == "null":
            return None
        return value


class AlignAndMergeWrapper(BaseWrapper):
    def run(self) -> bool:
        """
        Reads the Zocalo wrapper recipe, loads the parameters, and pass them to the
        alignment and merging function. Upon collecting the results, it then passes
        them back to Murfey for the next stage in the workflow.
        """

        if not hasattr(self, "recwrap"):
            logger.error("No RecipeWrapper object found")
            return False
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
