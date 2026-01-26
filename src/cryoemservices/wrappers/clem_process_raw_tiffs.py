"""
Contains functions needed in order to run a service to convert the raw CLEM data (from
either a LIF or TIFF file) into image stacks for use in subsequent stages of the CLEM
processing workflow.
"""

from __future__ import annotations

import json
import logging
import re
import time
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET

import numpy as np
from defusedxml.ElementTree import parse
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from cryoemservices.util.clem_array_functions import (
    TIFFImageLoader,
    get_percentiles,
    load_and_convert_image,
    resize_tile,
    write_stack_to_tiff,
)
from cryoemservices.util.clem_metadata import (
    find_image_elements,
    get_channel_info,
    get_dimension_info,
    get_tile_scan_info,
)

# Create logger object to output messages with
logger = logging.getLogger("cryoemservices.services.clem_process_raw_tiffs")


def process_tiff_files(
    tiff_list: list[Path],  # List of files associated with this series
    root_folder: str,  # Name of the folder to treat as the root folder
    metadata_file: Optional[Path] = None,  # Option to manually provide metadata file
    number_of_processes: int = 1,  # Number of processing threads to run
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

    # Start the timer
    start_time = time.perf_counter()

    # Validate processor count input
    num_procs = number_of_processes  # Use shorter phrase in script
    if num_procs < 1:
        logger.warning("Processor count set to zero or less; resetting to 1")
        num_procs = 1

    # Set variables for use within function
    new_root_folder = "processed"

    # Omit fragmented TIFF files ('--' followed by digits only) from consideration
    tiff_list = [file for file in tiff_list if not re.search(r"--(\d+)", file.stem)]

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

    # Create save directory for images and  metadata
    save_dir = Path("/".join(path_parts))  # Folder where this series will be saved
    save_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = save_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created save directory for images and metadata for {series_name}")

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

    # Save image metadata
    img_xml_file = metadata_dir / (series_identifier.replace(" ", "_") + ".xml")
    metadata_tree = ET.ElementTree(metadata)
    ET.indent(metadata_tree, "  ")
    metadata_tree.write(img_xml_file, encoding="utf-8")
    logger.info(f"Metadata for image stack saved to {img_xml_file}")

    # Extract metadata info
    try:
        channels = get_channel_info(metadata)
        dims = get_dimension_info(metadata)
        tile_scan_info = get_tile_scan_info(metadata)
    except Exception:
        logger.error(
            f"Failed to parse metadata file for {series_name!r}", exc_info=True
        )
        return {}

    # Get x, y, and z resolution (pixels per um)
    x_len = float(dims["x"]["length"])
    y_len = float(dims["y"]["length"])
    x_pixels = int(dims["x"]["num_pixels"])
    y_pixels = int(dims["y"]["num_pixels"])

    # Get number of frames
    num_frames = int(dims["z"].get("num_frames", 1))
    num_tiles = int(dims["m"].get("num_tiles", 1))

    # Calculate the full extent of the image
    # Initial arbitrary limits of the atlas in real space
    x_min = 10e10
    x_max = float(0)  # Comparing an int against a float will cause MyPy warning
    y_min = 10e10
    y_max = float(0)

    tile_extents: dict[int, tuple[float, float, float, float]] = {}
    for tile_num, tile_info in tile_scan_info.items():
        # Calculate and store the extent of the tile
        x = float(tile_info["pos_x"])
        y = float(tile_info["pos_y"])
        x0 = x - (x_len / 2)
        x1 = x + (x_len / 2)
        y0 = y - (y_len / 2)
        y1 = y + (y_len / 2)
        tile_extents[tile_num] = (x0, x1, y0, y1)

        # Update the atlas limits
        x_min = x - x_len / 2 if x - x_len / 2 < x_min else x_min
        x_max = x + x_len / 2 if x + x_len / 2 > x_max else x_max
        y_min = y - y_len / 2 if y - y_len / 2 < y_min else y_min
        y_max = y + y_len / 2 if y + y_len / 2 > y_max else y_max

    extent: tuple[float, float, float, float] = (x_min, x_max, y_min, y_max)

    # Construct results template for use in murfey
    result: dict[str, Any] = {
        "series_name": series_name,
        "number_of_members": len(channels),
        "is_stack": num_frames > 1,
        "is_montage": num_tiles > 1,
        "output_files": {},
        "thumbnails": {},
        "thumbnail_size": (512, 512),  # height, row
        "metadata": str(img_xml_file.resolve()),
        "parent_tiffs": {},
        "pixels_x": None,
        "pixels_y": None,
        "units": "",
        "pixel_size": None,
        "resolution": None,
        "extent": (),  # [x_min, x_max, y_min, y_max] in real space
    }

    # Iterate by color, then z-frame, then tile
    for c, color in enumerate(channels.keys()):
        logger.info(f"Processing {color} channel for {series_name!r}")
        # Get the subset of files to work with
        # Retain only the files associated with the current colour
        # Remove TIFF file fragments
        tiff_color_subset = (
            [
                file
                for file in tiff_list
                if f"C{str(c).zfill(2)}" in file.stem.split("--")
            ]
            if len(channels) > 1
            else tiff_list
        )

        # Sort list before iterating
        tiff_color_subset.sort(
            key=lambda f: (
                int(m.group(1)) if (m := re.search(r"--Z(\d+)", f.stem)) else -1
            )
        )

        # Estimate suitable contrast limits for the dataset
        logger.info("Estimating global intensity range")
        estimate_intensity_start_time = time.perf_counter()
        global_vmin, global_vmax = get_percentiles(
            image_loaders=[
                TIFFImageLoader(file)
                for f, file in enumerate(tiff_color_subset)
                if not f % 4
            ],
            percentiles=(0.5, 99.5),
            num_procs=num_procs,
        )
        estimate_intensity_end_time = time.perf_counter()
        logger.debug(
            f"Estimated contrast limits in {estimate_intensity_end_time - estimate_intensity_start_time}s"
        )

        # If it's a montage, stitch the tiles together in Matplotlib
        if num_tiles > 1:
            # Calculate the pixel size and shape of the final tiled image
            frame_x_len = x_max - x_min
            frame_y_len = y_max - y_min
            # Fit the tiled image within 2400 x 2400 pixels
            frame_pixel_size = min(frame_x_len / 2400, frame_y_len / 2400)
            frame_x_pixels = int(round(frame_x_len / frame_pixel_size))
            frame_y_pixels = int(round(frame_y_len / frame_pixel_size))
            frame_shape = frame_y_pixels, frame_x_pixels

            # Construct scaled down images for tiling
            logger.info("Constructing image stack from tiles")
            tile_stitching_start_time = time.perf_counter()
            arr = np.zeros((num_frames, frame_y_pixels, frame_x_pixels), dtype=np.uint8)
            with ThreadPoolExecutor(max_workers=num_procs) as pool:
                futures = [
                    pool.submit(
                        resize_tile,
                        TIFFImageLoader(file),
                        f,
                        tile_extents[t],
                        extent,
                        frame_pixel_size,
                        frame_shape,
                        global_vmin,
                        global_vmax,
                    )
                    for f in range(num_frames)
                    for t in tile_scan_info.keys()
                    for file in tiff_color_subset
                    if f"Stage{str(t).zfill(2)}" in file.stem.split("--")
                    and f"Z{str(f).zfill(2)}" in file.stem.split("--")
                ]
                for future in as_completed(futures):
                    r = future.result()
                    if (
                        r.data is not None
                        and r.frame_num is not None
                        and r.x0 is not None
                        and r.x1 is not None
                        and r.y0 is not None
                        and r.y1 is not None
                    ):
                        arr[
                            r.frame_num,
                            r.y0 : r.y1,
                            r.x0 : r.x1,
                        ] = r.data
                    else:
                        logger.warning(
                            "Failed to resize tile for the following image: \n"
                            f"{json.dumps(r.error, indent=2, default=str)}"
                        )

            # Update resolution and pixel size in dimensions dictionary
            y_pixels_new, x_pixels_new = arr.shape[-2:]
            dims["x"]["num_pixels"] = x_pixels_new
            dims["x"]["resolution"] = x_pixels_new / (x_max - x_min)
            dims["x"]["pixel_size"] = (x_max - x_min) / x_pixels_new
            dims["y"]["num_pixels"] = y_pixels_new
            dims["y"]["resolution"] = y_pixels_new / (y_max - y_min)
            dims["y"]["pixel_size"] = (y_max - y_min) / y_pixels_new

            tile_stitching_end_time = time.perf_counter()
            logger.debug(
                "Constructed image stack of tiled dataset in "
                f"{tile_stitching_end_time - tile_stitching_start_time}s"
            )

        # Otherwise, load the image as an array directly
        else:
            logger.info("Constructing image stack")
            array_loading_start_time = time.perf_counter()
            arr = np.zeros((num_frames, y_pixels, x_pixels), dtype=np.uint8)
            with ThreadPoolExecutor(max_workers=num_procs) as pool:
                futures = [
                    pool.submit(
                        load_and_convert_image,
                        TIFFImageLoader(tiff_color_subset[z]),
                        z,
                        global_vmin,
                        global_vmax,
                    )
                    for z in range(num_frames)
                ]
                for future in as_completed(futures):
                    r = future.result()
                    if r.data is not None and r.frame_num is not None:
                        arr[r.frame_num] = r.data
                    else:
                        logger.warning(
                            f"Failed to load the following image: \n"
                            f"{json.dumps(r.error, indent=2, default=str)}"
                        )
            array_loading_end_time = time.perf_counter()
            logger.debug(
                f"Loaded image stack of {num_frames} frames in "
                f"{array_loading_end_time - array_loading_start_time}s"
            )

        # Update results dictionary once per dataset
        if c == 0:
            # Update results dictionary
            result["pixels_x"] = dims["x"]["num_pixels"]
            result["pixels_y"] = dims["y"]["num_pixels"]
            result["units"] = dims["x"]["units"]
            result["pixel_size"] = dims["x"]["pixel_size"]
            result["resolution"] = dims["x"]["resolution"]
            extent = (x_min, x_max, y_min, y_max)
            result["extent"] = extent

        # Extract metadata needed for saving the image
        image_labels = [str(z) for z in range(dims["z"].get("num_frames", 1))]
        x_res = float(dims["x"]["resolution"])
        y_res = float(dims["y"]["resolution"])
        z_res = float(dims["z"]["resolution"]) if dims.get("z", {}) else float(0)
        units = dims["x"]["units"]

        # Convert units to microns just for the image
        if units == "m":
            units = "micron"
            x_res /= 10**6
            y_res /= 10**6
            z_res /= 10**6

        # Save as a greyscale TIFF
        img_stk_file = write_stack_to_tiff(
            array=arr,
            save_dir=save_dir,
            file_name=color,
            x_res=x_res,
            y_res=y_res,
            z_res=z_res,
            units=units,
            axes="ZYX",
            image_labels=image_labels,
            photometric="minisblack",
        ).resolve()

        # Collect the images created
        result["output_files"][color] = str(img_stk_file)
        result["parent_tiffs"][color] = [str(file) for file in tiff_color_subset][0:1]

        # Append path to where the PNG images should be created
        result["thumbnails"][color] = str(
            img_stk_file.parent / ".thumbnails" / f"{img_stk_file.stem}.png"
        )

    end_time = time.perf_counter()
    logger.debug(f"Completed processing of {series_name} in {end_time - start_time}s")
    logger.debug(
        "Returning the following processing results: \n"
        f"{json.dumps(result, indent=2, default=str)}"
    )
    return result


class ProcessRawTIFFsParameters(BaseModel):
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
    num_procs: int = 20

    @field_validator("tiff_list", mode="before")
    @classmethod
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
    @classmethod
    def parse_tiff_file(cls, value):
        # Check for "null" keyword
        if value == "null":
            return None
        # Convert to Path otherwise
        if isinstance(value, str):
            return Path(value.strip())
        return value

    @model_validator(mode="after")
    @classmethod
    def construct_tiff_list(cls, model: ProcessRawTIFFsParameters):
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


class ProcessRawTIFFsWrapper:
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
            params = ProcessRawTIFFsParameters(**params_dict)
        except (ValidationError, TypeError) as e:
            logger.error(
                "ProcessRawTIFFsParameters validation failed for the following parameters: \n"
                f"{json.dumps(params_dict, indent=2, default=str)}\n"
                f"with exception: {e}"
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
        try:
            result = process_tiff_files(
                tiff_list=params.tiff_list,
                root_folder=params.root_folder,
                metadata_file=params.metadata,
                number_of_processes=params.num_procs,
            )
        # Log error and return False if the command fails to execute
        except Exception:
            logger.error(
                f"Exception encountered while processing TIFF files for series {series_name!r}: \n",
                exc_info=True,
            )
            return False
        if not result:
            logger.error(
                f"No processing results were returned for TIFF series {series_name!r}"
            )
            return False

        # Request for PNG images to be created
        for color in result["output_files"].keys():
            images_params = {
                "image_command": "tiff_to_apng",
                "input_file": result["output_files"][color],
                "output_file": result["thumbnails"][color],
                "target_size": result["thumbnail_size"],
                "color": color,
            }
            self.recwrap.send_to(
                "images",
                images_params,
            )
            logger.info(
                f"Submitted the following job to Images service: \n{images_params}"
            )

        # Send results to Murfey's "feedback_callback" function
        murfey_params = {
            "register": "clem.register_preprocessing_result",
            "result": result,
        }
        self.recwrap.send_to("murfey_feedback", murfey_params)
        logger.info(
            f"Submitted processed data for {result['series_name']!r} "
            "and associated metadata to Murfey for registration"
        )

        return True
