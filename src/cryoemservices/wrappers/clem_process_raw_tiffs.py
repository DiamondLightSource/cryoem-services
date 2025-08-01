"""
Contains functions needed in order to run a service to convert the raw CLEM data (from
either a LIF or TIFF file) into image stacks for use in subsequent stages of the CLEM
processing workflow.
"""

from __future__ import annotations

import logging
from ast import literal_eval
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET

import numpy as np
from defusedxml.ElementTree import parse
from matplotlib import pyplot as plt
from PIL import Image
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from cryoemservices.util.clem_array_functions import (
    estimate_int_dtype,
    preprocess_img_stk,
    write_stack_to_tiff,
)
from cryoemservices.util.clem_metadata import find_image_elements, get_axis_resolution

# Create logger object to output messages with
logger = logging.getLogger("cryoemservices.services.clem_process_raw_tiffs")


def process_tiff_files(
    tiff_list: list[Path],  # List of files associated with this series
    root_folder: str,  # Name of the folder to treat as the root folder
    metadata_file: Optional[Path] = None,  # Option to manually provide metadata file
) -> dict:
    """
    Takes a list of TIFF files for a distinct image series and uses them to construct
    an image stack or atlas as appropriate. Stacks are saved in a directory called
    "processed" with a folder structure mirroring how they were stored as raw data.

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

    # Set variables for use within function
    new_root_folder = "processed"

    # Use the names of the TIFF files to get the unique path to it
    #   Remove the "--Z##--C##.tiff" end of the file path strings
    series_path = tiff_list[0].parent / tiff_list[0].stem.split("--")[0]

    # Construct series identifier from unique path
    series_identifier = series_path.stem

    # Parse path parts to construct parameters
    path_parts = list(series_path.parts)
    # Remove leading "/" in Unix paths
    path_parts[0] = "" if path_parts[0] == "/" else path_parts[0]
    # Remove last level if redundant
    path_parts = path_parts[:-1] if path_parts[-1] == path_parts[-2] else path_parts

    # Search for root folder with case-insensitivity and point to new location
    try:
        root_index = [p.lower() for p in path_parts].index(root_folder.lower())
        path_parts[root_index] = new_root_folder
    except ValueError:
        logger.error(
            f"Subpath {root_folder!r} was not found in image path "
            f"{str(series_path.parent)!r}"
        )
        return {}

    # Sanitise everything after the root folder
    path_parts = [
        part if p <= root_index else part.replace(" ", "_")
        for p, part in enumerate(path_parts)
    ]

    # Construct extended series name for database records
    series_name = "--".join(path_parts[root_index + 1 :]).replace(" ", "_")
    logger.info(f"Processing {series_name} TIFF files")

    # Search for metadata file using relative paths if not provided
    if metadata_file is None:
        metadata_file = series_path.parent / "Metadata" / (series_identifier + ".xlif")
        if metadata_file.exists():
            logger.info(f"Metadata file found at {metadata_file}")
        else:
            logger.error(f"No metadata file found at {metadata_file}")
            return {}

    # Load relevant metadata
    metadata_dict = find_image_elements(parse(metadata_file).getroot())
    if len(metadata_dict) < 1:
        logger.error(f"No image metadata found in file {metadata_file}")
        return {}
    elif len(metadata_dict) > 1:
        logger.error(
            f"More than one image metadata element found in file {metadata_file}"
        )
        return {}
    metadata = list(metadata_dict.values())[0]

    # Create save directory for images and  metadata
    save_dir = Path("/".join(path_parts))  # Folder where this series will be saved
    save_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = save_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created save directory for images and metadata for {series_name}")

    # Save image metadata
    img_xml_file = metadata_dir / (series_identifier.replace(" ", "_") + ".xml")
    metadata_tree = ET.ElementTree(metadata)
    ET.indent(metadata_tree, "  ")
    metadata_tree.write(img_xml_file, encoding="utf-8")
    logger.info(f"Metadata for image stack saved to {img_xml_file}")

    # Load channels
    channels = metadata.findall(".//ChannelDescription")
    colors = [channels[c].attrib["LUTName"].lower() for c in range(len(channels))]

    # Get x, y, and z resolution (pixels per um)
    dims = metadata.findall(".//DimensionDescription")
    try:
        x_dim = [dim for dim in dims if dim.get("DimID", "") == "1"][0]
        y_dim = [dim for dim in dims if dim.get("DimID", "") == "2"][0]
        x_res = get_axis_resolution(x_dim)
        y_res = get_axis_resolution(y_dim)

        # Get the end-to-end width and height of image
        x_len = float(x_dim.get("Length", ""))
        x_pix = int(x_dim.get("NumberOfElements", ""))
        y_len = float(y_dim.get("Length", ""))
        y_pix = int(y_dim.get("NumberOfElements", ""))

        # Assumption 1: 'Length' is midpoint-to-midpoint length
        w = x_len * x_pix / (x_pix - 1)
        h = y_len * y_pix / (y_pix - 1)

    except (IndexError, ValueError):
        logger.error(f"Unable to extract dimensional information for {series_name}")

    # Construct results template for use in h
    result: dict[str, Any] = {
        "series_name": series_name,
        "number_of_members": len(channels),
        "data_type": "",
        "channel_data": {},  # Dictionary for different color channels
        "metadata": str(img_xml_file.resolve()),
        "extent": [],  # [x_min, x_max, y_min, y_max] in real space
    }

    # if-block to construct atlas from images
    if "10" in (dim.get("DimID", "") for dim in dims):
        result["data_type"] = "atlas"
        logger.info(f"Processing {series_name} as a montage")

        tile_scan_info = next(
            (
                node
                for node in metadata.findall(".//Attachment")
                if node.get("Name") == "TileScanInfo"
            ),
            None,
        )
        if tile_scan_info is None:
            logger.error(f"No tile scan information found for series {series_name!r}")
            return {}
        tile_scans = list(tile_scan_info)

        for c, color in enumerate(colors):
            logger.info(f"Processing {color} channel")

            # Select only the desired colour if multiple channels are present
            tiff_sublist = (
                [
                    file
                    for file in tiff_list
                    if f"--C{str(c).zfill(2)}" in file.stem.split("--", maxsplit=1)[-1]
                ]
                if len(colors) > 1
                else tiff_list
            )
            tiff_sublist.sort(
                key=lambda f: (int(f.stem.split("--")[1].replace("Stage", "")),)
            )  # Sort by 'Stage' value
            print(f"Found the following files for {color} channel:", tiff_sublist)
            if not tiff_sublist:
                logger.error(
                    f"Error processing {color!r} channel for {series_name!r}: "
                    "No TIFF files found"
                )
                continue

            # Initial arbitrary limits of the atlas in real space
            x_min = 10e10
            x_max = float(0)
            y_min = 10e10
            y_max = float(0)

            # Create the figure to save the image onto
            fig, ax = plt.subplots()

            # Load the coordinates
            for t, tile in enumerate(tile_scans):
                try:
                    logger.info(f"Processing tile {t}")
                    # Convert tile coordinates into floats
                    x = float(tile.get("PosX", ""))
                    y = float(tile.get("PosY", ""))
                    # Find the coordinates of the image's 4 corners in real space
                    tile_extent = [x, x + w, y, y + h]

                    # Load image and add it to the plot
                    img = Image.open(tiff_sublist[t])
                    ax.imshow(
                        img,
                        extent=tile_extent,
                        origin="lower",
                        cmap="gray",
                    )

                    # Update atlas limits
                    x_min = x if x < x_min else x_min
                    y_min = y if y < y_min else y_min
                    x_max = x + w if x + w > x_max else x_max
                    y_max = y + h if y + h > y_max else y_max

                except (TypeError, ValueError):
                    logger.warning(
                        f"Unable to extract coordinate information for tile {t}"
                    )
                    continue

            # Crop plotted image to just the populated space
            logger.info("Adjusting plotting settings...")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.invert_yaxis()  # Set origin to top left
            ax.set_aspect("equal")  # Ensure x- and y-axis have the same scale
            ax.axis("off")  # Turn off axis ticks and labels

            # Save just the contents of the Axes object
            logger.info("Rendering image...")
            fig.canvas.draw()  # Render the figure
            renderer = fig.canvas.get_renderer()
            bbox = ax.get_tightbbox(renderer).transformed(
                fig.dpi_scale_trans.inverted()
            )
            fig_name = save_dir / f"{color}.png"
            fig.savefig(
                fig_name, bbox_inches=bbox, pad_inches=0, dpi=400, facecolor="black"
            )
            plt.close()
            logger.info(f"Atlas image saved as {fig_name}")

            result["channel_data"][color] = {
                "atlas": str(fig_name),
                "parent_tiffs": str([str(f) for f in tiff_sublist]),
            }

    # if-block to construct image stacks from images
    else:
        result["data_type"] = "image"
        logger.info(f"Processing {series_name} as an image stack")

        # Process z-axis if it exists
        z_res = get_axis_resolution(dims[2]) if len(dims) > 2 else float(0)

        # Load timestamps (might be useful in the future)
        # timestamps = elem.find("Data/Image/TimeStampList")

        # Generate slice labels for later
        num_frames = [
            int(dim.get("NumberOfElements", "")) if dim else 1
            for dim in dims
            if dim.get("DimID", "") == "3"
        ][0]
        image_labels = [f"{f}" for f in range(num_frames)]

        # Process channels as individual TIFFs
        for c, color in enumerate(colors):
            logger.info(f"Processing {color} channel")

            # Find TIFFs from relevant channel and series
            #   Replace " " with "_" when comparing file name against series name as
            #   found in metadata
            tiff_sublist = [
                f
                for f in tiff_list
                if (f"C{str(c).zfill(2)}" in f.stem or f"C{str(c).zfill(3)}" in f.stem)
                and (
                    series_identifier.replace(" ", "_")
                    == f.stem.split("--")[0].replace(" ", "_")
                )
            ]
            tiff_sublist.sort(
                key=lambda f: (int(f.stem.split("--")[1].replace("Z", "")),)
            )  # Sort by Z as an int, not str

            # Return error message if the list of TIFFs is empty for some reason
            if not tiff_sublist:
                logger.error(
                    f"Error processing {color!r} channel for {series_name!r}; "
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
                f"{series_name} {color} array properties: \n"
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
                f"{series_name} {color} array properties: \n"
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
            result["channel_data"][color] = {
                "image_stack": str(img_stk_file.resolve()),
                "parent_tiffs": str([str(f) for f in tiff_sublist]),
            }

    # Collect and return files that have been generated
    return result


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
        result = process_tiff_files(
            tiff_list=params.tiff_list,
            root_folder=params.root_folder,
            metadata_file=params.metadata,
        )

        # Log errors and warnings
        if not result:
            logger.error(f"Process failed for TIFF series {series_name!r}")
            return False
        # Send results to Murfey's "feedback_callback" function
        murfey_params = {
            "register": "clem.register_tiff_preprocessing_result",
            "result": result,
        }
        self.recwrap.send_to("murfey_feedback", murfey_params)
        logger.info(
            f"Submitted processed data for {result['series_name']!r} "
            "and associated metadata to Murfey for registration"
        )

        return True
