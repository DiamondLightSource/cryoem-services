"""
Contains functions needed in order to run a service to process the image stacks
generated from CLEM data. This service will include:
    1. Flattening of 3D image stacks into 2D images (optional)
    2. Image alignment within and across stacks (optional)
    3. Colorisation of images according to their channels
    4. Merging of images to create a colored composite

"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union
from xml.etree import ElementTree as ET

import numpy as np
from defusedxml.ElementTree import parse
from tifffile import TiffFile, imwrite

from cryoemservices.clem.images import (
    convert_array_bit_depth,
    convert_to_rgb,
    create_composite_image,
    flatten_image,
    stretch_image_contrast,
)
from cryoemservices.clem.xml import get_image_elements


def merge_image_stacks(
    image_files: Union[Path, list[Path]],
    metadata_file: Optional[Path] = None,
    flatten: Optional[Literal["min", "max", "mean"]] = "mean",
    registration: Optional[str] = None,
):
    """
    A cryoEM service (eventually) to create composite images from component image
    stacks in the series.

    This will (eventually) take a message containing the image stack and metadata
    files associated with an image series, along with settings for how the stacks
    should be processed (a flattened 2D image or a 3D image stack).
    """

    # Validate inputs before proceeding further
    if flatten is not None and flatten not in ("min", "max", "mean"):
        raise ValueError("Incorrect value provided for 'flatten' parameter")

    # Use shorter inputs in funciton
    files = image_files

    # Turn single entry into a list
    if isinstance(files, Path):
        files = [files]
    # Check that a value has been provided
    if len(files) == 0:
        raise ValueError("No image stack file paths have been provided")

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

    # Find metadata file if none was provided
    if metadata_file is None:
        # Raise error if no files found
        if len(list((parent_dir / "metadata").glob("*.xml"))) == 0:
            raise FileNotFoundError(
                "No metadata file was found at the default directory"
            )
        # Raise error if too many files found
        if len(list((parent_dir / "metadata").glob("*.xml"))) > 1:
            raise Exception(
                "More than one metadata file was found at the default directory"
            )
        metadata_file = list((parent_dir / "metadata").glob("*.xml"))[0]

    # Load metadata for series from XML file
    element_list = get_image_elements(
        parse(metadata_file).getroot()
    )  # Check if multiple elements are present in the XML file first
    metadata: ET.ElementTree = (
        element_list[0] if len(element_list) > 0 else parse(metadata_file).getroot()
    )

    # Get order of colors as shown in metadata
    channels = metadata.findall(
        "Data/Image/ImageDescription/Channels/ChannelDescription"
    )
    colors: list[str] = [
        channels[c].attrib["LUTName"].lower() for c in range(len(channels))
    ]

    # Load image stacks according to their order in the XML file
    arrays: list[np.ndarray] = []
    for c in range(len(colors)):
        color = colors[c]
        file_search = [file for file in files if color in file.stem]
        # Handle exceptions in search results
        if len(file_search) == 0:
            raise FileNotFoundError("No files provided that match this colour")
        if len(file_search) > 1:
            raise Exception("More than one file provided that matches this colour")
        file = file_search[0]

        with TiffFile(file) as tiff_file:

            # Check that there aren't multiple series in the image stack
            if len(tiff_file.series) > 1:
                raise ValueError(
                    "The image stack provided contains more than one series"
                )

            # Load array and apend it to stack
            arrays.append(tiff_file.series[0].pages.asarray())
            pass

    # Validate that the stacks provided are of the same shape
    if len({arr.shape for arr in arrays}) > 1:
        raise Exception("The image stacks provided do not have the same shape")

    # Get the dtype of the image
    if len({str(arr.dtype) for arr in arrays}) > 1:
        raise Exception("The image stacks do not have the same dtype")
    dtype_init = list({str(arr.dtype) for arr in arrays})[0]
    bit_init = int("".join([char for char in dtype_init if char.isdigit()]))

    # Flatten images
    if flatten is not None:
        arrays = [flatten_image(arr, mode=flatten) for arr in arrays]
        # Validate that image flattening was done correctly
        if len({arr.shape for arr in arrays}) > 1:
            raise Exception("The flattened arrays do not have the same shape")

    # Align reference stack
    ## TO DO

    # Align other stacks to reference stack
    ## TO DO

    # Colourise images
    arrays = [convert_to_rgb(arrays[c], colors[c]) for c in range(len(colors))]

    # Convert to a composite image
    composite_img = create_composite_image(arrays)
    composite_img = convert_array_bit_depth(
        composite_img,
        target_bit_depth=bit_init,
        initial_bit_depth=bit_init,
    )
    composite_img = stretch_image_contrast(
        composite_img,
        percentile_range=(0, 100),
    )
    # Debug information
    # print(f"dtype: {composite_img.dtype}")
    # print(f"Shape: {composite_img.shape}")
    # print(f"Max: {composite_img.max()}")
    # print(f"Min: {composite_img.min()}")

    # Save the image to a TIFF file
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
    print(f"Composite image saved as {save_name}")

    return composite_img
