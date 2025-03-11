"""
Contains functions needed in order to run a service to convert the raw CLEM data (from
either a LIF or TIFF file) into image stacks for use in subsequent stages of the CLEM
processing workflow.
"""

from __future__ import annotations

import logging
from ast import literal_eval
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np
from defusedxml.ElementTree import parse
from PIL import Image
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from cryoemservices.util.clem_array_functions import (
    estimate_int_dtype,
    preprocess_img_stk,
    write_stack_to_tiff,
)
from cryoemservices.util.clem_raw_metadata import (
    get_axis_resolution,
    get_image_elements,
)

# Create logger object to output messages with
logger = logging.getLogger("cryoemservices.services.clem_process_raw_tiffs")


def process_tiff_files(
    tiff_list: list[Path],
    metadata_file: Path,
    series_name_short: str,
    series_name_long: str,
    save_dir: Path,
) -> list[dict] | None:
    """
    Opens the TIFF files as NumPy arrays and stacks them.
    """

    # Validate metadata
    # Convert to list for Python 3.9 compatibility
    if list(metadata_file.parents)[-2] != list(tiff_list[0].parents)[-2]:
        logger.error("The base paths of the metadata and TIFF files do not match")
        return None

    # Load relevant metadata
    elem_list = get_image_elements(parse(metadata_file).getroot())
    metadata = elem_list[0]

    # Create save directory for image metadata
    metadata_dir = save_dir / "metadata"
    if not metadata_dir.exists():
        metadata_dir.mkdir(parents=True)
        logger.info(f"Created metadata directory at {metadata_dir}")
    else:
        logger.info(f"{metadata_dir} already exists")

    # Save image metadata
    img_xml_file = metadata_dir / (series_name_short.replace(" ", "_") + ".xml")
    metadata_tree = ET.ElementTree(metadata)
    ET.indent(metadata_tree, "  ")
    metadata_tree.write(img_xml_file, encoding="utf-8")
    logger.info(f"Metadata for image stack saved to {img_xml_file}")

    # Load channels
    channels = metadata.findall(
        "Data/Image/ImageDescription/Channels/ChannelDescription"
    )
    colors = [channels[c].attrib["LUTName"].lower() for c in range(len(channels))]

    # Get x, y, and z resolution (pixels per um)
    dimensions = metadata.findall(
        "Data/Image/ImageDescription/Dimensions/DimensionDescription"
    )
    x_res = get_axis_resolution(dimensions[0])
    y_res = get_axis_resolution(dimensions[1])

    # Process z-axis if it exists
    z_res = get_axis_resolution(dimensions[2]) if len(dimensions) > 2 else float(0)

    # Load timestamps (might be useful in the future)
    # timestamps = elem.find("Data/Image/TimeStampList")

    # Generate slice labels for later
    num_frames = (
        int(dimensions[2].attrib["NumberOfElements"]) if len(dimensions) > 2 else 1
    )
    image_labels = [f"{f}" for f in range(num_frames)]

    # Process channels as individual TIFFs
    results: list[dict] = []
    for c in range(len(colors)):

        # Get color
        color = colors[c]
        logger.info(f"Processing {color} channel")

        # Find TIFFs from relevant channel and series
        #   Replace " " with "_" when comparing file name against series name as
        #   found in metadata
        tiff_sublist = [
            f
            for f in tiff_list
            if (f"C{str(c).zfill(2)}" in f.stem or f"C{str(c).zfill(3)}" in f.stem)
            and (
                series_name_short.replace(" ", "_")
                == f.stem.split("--")[0].replace(" ", "_")
            )
        ]
        tiff_sublist.sort(
            key=lambda f: (int(f.stem.split("--")[1].replace("Z", "")),)
        )  # Sort by Z as an int, not str

        # Return error message if the list of TIFFs is empty for some reason
        if not tiff_sublist:
            logger.error(
                f"Error processing {color!r} channel for {series_name_long!r}; "
                "no TIFF files found"
            )
            continue

        # Load image stack
        logger.info("Loading image stack")
        for t in range(len(tiff_sublist)):
            img = Image.open(tiff_sublist[t])
            if t == 0:
                arr = np.array([img])  # Store as 3D array
            else:
                arr = np.append(arr, [img], axis=0)
        logger.debug(
            f"{series_name_long} {color} array properties: \n"
            f"Shape: {arr.shape} \n"
            f"dtype: {arr.dtype} \n"
            f"Min value: {arr.min()} \n"
            f"Max value: {arr.max()} \n"
        )

        # Estimate initial NumPy dtype
        bit_depth = int(channels[c].attrib["Resolution"])
        dtype_init = estimate_int_dtype(arr, bit_depth=bit_depth)

        # Rescale intensity values for fluorescent channels
        adjust_contrast = (
            "stretch"
            if color
            in (
                "blue",
                "cyan",
                "green",
                "magenta",
                "red",
                "yellow",
            )
            else None
        )

        # Process the image stack
        logger.info("Processing image stack")
        arr = preprocess_img_stk(
            array=arr,
            initial_dtype=dtype_init,
            target_dtype="uint8",
            adjust_contrast=adjust_contrast,
        )
        logger.debug(
            f"{series_name_long} {color} array properties: \n"
            f"Shape: {arr.shape} \n"
            f"dtype: {arr.dtype} \n"
            f"Min value: {arr.min()} \n"
            f"Max value: {arr.max()} \n"
        )

        # Save as a greyscale TIFF
        logger.info("Processing image stack")
        img_stk_file = write_stack_to_tiff(
            array=arr,
            save_dir=save_dir,
            file_name=color,
            x_res=x_res,
            y_res=y_res,
            z_res=z_res,
            units="micron",
            axes="ZYX",
            image_labels=image_labels,
            photometric="minisblack",
        )
        # Create dictionary for the image stack created
        result = {
            "image_stack": str(img_stk_file.resolve()),
            "metadata": str(img_xml_file.resolve()),
            "series_name": series_name_long,
            "channel": color,
            "number_of_members": len(channels),
            "parent_tiffs": str([str(f) for f in tiff_sublist]),
        }
        results.append(result)

    # Collect and return files that have been generated
    return results


def convert_tiff_to_stack(
    tiff_list: list[Path],  # List of files associated with this series
    root_folder: str,  # Name of the folder to treat as the root folder
    metadata_file: Optional[Path] = None,  # Option to manually provide metadata file
) -> list[dict] | None:
    """
    Takes a list of TIFF files for a distinct image series and converts them into a
    TIFF image stack with the key metadata embedded. Stacks are saved in a folder
    called "processed", which preserves the directory structure of the raw files.

    FOLDER STRUCTURE:
    The file structure when using "auto-save" differs slightly from that when saving
    as a single .LIF file:

    parent_folder
    |__ images              <- Raw data stored here
    |   |__ position_1      <- Folder for a defined set of images
    |   |   |__ metadata    <- Metadata for all images for that position
    |   |   |   |__ position1.xlif          <- Actually an XML file
    |   |   |__ position1--Z00--C00.tiff    <- Individual channels and frames
    |   |   |__ position1--Z00--C01.tiff
    |   |   |   ...
    |   |__ position_2
    |       ...
    |__ processed           <- Processed data goes here
        |__ position_1
        |   |__ metadata
        |   |   |__ position_1.xml
        |   |__ gray.tiff   <- Individual TIFFs collected into image stacks
        |   |__ red.tiff
        |   |   ... Mimics "images" folder structure
    """

    # Set variables and shorter names for use within function
    new_root_folder = "processed"

    # Use the names of the TIFF files to get the unique path to it
    #   Remove the "--Z##--C##.tiff" end of the file path strings
    series_path = tiff_list[0].parent / tiff_list[0].stem.split("--")[0]
    series_name_short = (
        series_path.stem
    )  # For finding and parsing metadata file; may contain spaces

    # Parse path parts to construct parameters
    path_parts = list(series_path.parts)
    path_parts[0] = (
        "" if path_parts[0] == "/" else path_parts[0]
    )  # Remove leading "/" in Unix paths
    path_parts = [
        p.replace(" ", "_") if " " in p else p for p in path_parts
    ]  # Replace spaces in path
    path_parts = (
        path_parts[:-1] if path_parts[-1] == path_parts[-2] else path_parts
    )  # Remove last level if redundant
    try:
        # Search for root folder with case-insensitivity and point to new location
        root_index = [p.lower() for p in path_parts].index(root_folder.lower())
        path_parts[root_index] = new_root_folder
    except ValueError:
        logger.error(
            f"Subpath {root_folder!r} was not found in image path "
            f"{str(series_path.parent)!r}"
        )
        return None

    # Construct long name of the series for database records
    series_name_long = "--".join(path_parts[root_index + 1 :]).replace(" ", "_")
    logger.info(f"Processing {series_name_long} TIFF files")

    # Construct save directory
    save_dir = Path("/".join(path_parts))  # Folder where this series will be saved
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
        logger.info(f"Created {save_dir}")
    else:
        logger.info(f"{str(save_dir)} already exists")

    # Get associated XML file
    if metadata_file is None:  # Search for it using relative paths if not provided
        xml_file = series_path.parent / "Metadata" / (series_name_short + ".xlif")
        if xml_file.exists():
            logger.info(f"Metadata file found at {xml_file}")
        else:
            logger.error(f"No metadata file found at {xml_file}")
            return None
    else:
        xml_file = metadata_file

    # Process TIFF files and collect results
    results = process_tiff_files(
        tiff_list=tiff_list,
        metadata_file=xml_file,
        series_name_short=series_name_short,
        series_name_long=series_name_long,
        save_dir=save_dir,
    )

    return results


class TIFFToStackParameters(BaseModel):
    """
    Pydantic model for validating the received message for the TIFF file conversion
    workflow.

    The keys under the "job_parameters" key in the Zocalo wrapper recipe must match
    the attributes present in this model. Attributes in the model with a default
    value don't have to be provided in the wrapper recipe.
    """

    # One of 'tiff_list' or 'tiff_file' should be provided
    # The other one should be None (pass "null" in the CLI)
    tiff_list: Optional[list[Path]]
    tiff_file: Optional[Path]
    root_folder: str
    metadata: Path

    @field_validator("tiff_list", mode="before")
    def parse_tiff_list(cls, value):
        if isinstance(value, str):
            # Check for "null" keyword
            if value == "null":
                return None
            # Check for stringified list
            if value.startswith("[") and value.endswith("]"):
                try:
                    eval_tiff_list: list[str] = literal_eval(value)
                    tiff_list = [Path(p) for p in eval_tiff_list]
                    return tiff_list
                except (SyntaxError, TypeError, ValueError):
                    logger.error("Unable to parse stringified list for file paths")
                    raise ValueError
        # Leave the value as-is; if it fails, it fails
        return value

    @field_validator("tiff_file", mode="before")
    def parse_tiff_file(cls, value):
        # Check for "null" keyword
        if value == "null":
            return None
        # Convert to Path otherwise
        if isinstance(value, str):
            return Path(value.strip())
        return value

    @model_validator(mode="after")
    def construct_tiff_list(cls, model: TIFFToStackParameters):
        if model.tiff_list and model.tiff_file:
            raise ValueError(
                "Only one of 'tiff_list' or 'tiff_file' should be provided, not both"
            )
        if not model.tiff_list and not model.tiff_file:
            raise ValueError("One of 'tiff_list' or 'tiff_file' has to be provided")
        if not model.tiff_list and model.tiff_file:
            series_name = model.tiff_file.stem.split("--")[0]
            model.tiff_list = [
                f.resolve()
                for f in model.tiff_file.parent.glob("./*")
                if f.suffix in (".tif", ".tiff")
                and f.stem.startswith(f"{series_name}--")
            ]
        # Return updated model
        return model


class TIFFToStackWrapper:
    def __init__(self, recwrap):
        self.log = logging.LoggerAdapter(logger)
        self.recwrap = recwrap

    def run(self) -> bool:
        """
        Reads the Zocalo wrapper recipe, loads the parameters, and passes them to the
        TIFF file processing function. Upon collecting the results, it sends them back
        to Murfey for the next stage of the workflow.
        """
        params_dict = self.recwrap.recipe_step["job_parameters"]
        try:
            params = TIFFToStackParameters(**params_dict)
        except (ValidationError, TypeError) as e:
            logger.error(
                "TIFFToStackParameters validation failed for parameters: "
                f"{params_dict} with exception: {e}"
            )
            return False

        # Catch no TIFF files case
        if not params.tiff_list:  # Will catch None and []
            logger.error("No list of TIFF files found after data validation")
            return False

        # Reconstruct series name using reference file from list
        ref_file = params.tiff_list[0]
        path_parts = list((ref_file.parent / ref_file.stem.split("--")[0]).parts)
        try:
            root_index = path_parts.index(params.root_folder)
        except ValueError:
            logger.error(
                f"Subpath {params.root_folder!r} was not found in file path "
                f"{str(ref_file.parent / ref_file.stem.split('--')[0])!r}"
            )
            return False
        series_name = "--".join(
            [p.replace(" ", "_") if " " in p else p for p in path_parts][
                root_index + 1 :
            ]
        )

        # Process files and collect output
        results = convert_tiff_to_stack(
            tiff_list=params.tiff_list,
            root_folder=params.root_folder,
            metadata_file=params.metadata,
        )

        # Log errors and warnings
        if results is None:
            logger.error(f"Process failed for TIFF series {series_name!r}")
            return False
        for result in results:
            # Send results to Murfey's "feedback_callback" function
            murfey_params = {
                "register": "clem.register_tiff_preprocessing_result",
                "result": result,
            }
            self.recwrap.send_to("murfey_feedback", murfey_params)
            logger.info(
                f"Submitted {result['series_name']!r} {result['channel']!r} "
                "image stack and associated metadata for registration"
            )

        return True
