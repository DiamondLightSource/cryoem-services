"""
Contains functions needed in order to run a service to convert the raw CLEM data (from
either a LIF or TIFF file) into image stacks for use in subsequent stages of the CLEM
processing workflow.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np
from pydantic import BaseModel, ValidationError
from readlif.reader import LifFile

from cryoemservices.util.clem_array_functions import (
    LIFImageLoader,
    get_percentiles,
    load_and_convert_image,
    load_and_resize_tile,
    write_stack_to_tiff,
)
from cryoemservices.util.clem_metadata import (
    find_image_elements,
    get_channel_info,
    get_dimension_info,
    get_tile_scan_info,
)

# Create logger object to output messages with
logger = logging.getLogger("cryoemservices.wrappers.clem_process_raw_lifs")


def get_lif_xml_metadata(
    file: LifFile,
    save_xml: Optional[Path] = None,
) -> ET.Element:
    """
    Extracts and returns the metadata from the LIF file as a formatted XML Element.
    It can be optionally saved as an XML file to the specified file path.
    """

    # Use readlif function to get XML metadata
    xml_root: ET.Element = file.xml_root  # This one for navigating
    xml_tree = ET.ElementTree(xml_root)  # This one for saving

    # Skip saving the metadata if save_xml not provided
    if save_xml:
        xml_file = str(save_xml)  # Convert Path to string
        ET.indent(xml_tree, "  ")  # Format with proper indentation
        xml_tree.write(xml_file, encoding="utf-8")  # Save
        logger.info(f"File metadata saved to {xml_file!r}")

    return xml_root


def process_lif_subimage(
    file: Path,
    scene_num: int,
    metadata: ET.Element,
    root_save_dir: Path,
    save_path: str = "",
    num_procs: int = 4,
) -> dict:
    """
    Takes the LIF file and its corresponding metadata and loads the relevant subimage.
    For image stacks, it will load each colour channel as its own image stack, rescale
    the intensity values to utilise the whole channel, convert it into 8-bit grayscale,
    and save them as individual images or image stacks.

    For montages, it will stitch the different subimages together and save the final
    composite image as an atlas.
    """

    start_time = time.perf_counter()

    # Get name of subimage
    file_name = file.stem.replace(" ", "_")  # Remove spaces
    img_name = metadata.attrib["Name"].replace(" ", "_")  # Remove spaces

    # Construct path to save images and metadata to
    save_dir = (  # Save directory for all subimages from this LIF file
        root_save_dir
        / "/".join(file.relative_to(root_save_dir.parent).parts[1:-1])
        / (save_path if save_path else Path(file_name) / img_name)
    )

    # Create a name for this series
    series_name = (
        save_dir.relative_to(root_save_dir)
        .as_posix()
        .replace("/", "--")
        .replace(" ", "_")
    )
    logger.info(f"Processing {series_name!r}")

    # Save metadata relative to the subimage
    img_xml_dir = save_dir / "metadata"
    img_xml_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created folders to save image and metadata for {series_name!r} to")

    # Save image XML metadata (all channels together)
    img_xml_file = img_xml_dir / (img_name + ".xml")
    metadata_tree = ET.ElementTree(metadata)  # For saving
    ET.indent(metadata_tree, "  ")
    metadata_tree.write(img_xml_file, encoding="utf-8")
    logger.info(f"Image metadata saved to {img_xml_file}")

    # Extract metadata using helper functions
    try:
        channels = get_channel_info(metadata)
        dims = get_dimension_info(metadata)
        tile_scan_info = get_tile_scan_info(metadata)
    except Exception:
        logger.error(
            f"Failed to parse metadata file for {series_name!r}", exc_info=True
        )
        return {}

    # Get width and height for a single frame
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
    x_max = float(0)
    y_min = 10e10
    y_max = float(0)

    tile_extents: dict[int, tuple[float, float, float, float]] = {}
    for tile_num, tile_info in tile_scan_info.items():
        x = float(tile_info["pos_x"])
        y = float(tile_info["pos_y"])
        x0 = x - (x_len / 2)
        x1 = x + (x_len / 2)
        y0 = y - (y_len / 2)
        y1 = y + (y_len / 2)
        tile_extents[tile_num] = (x0, x1, y0, y1)

        # Update the atlas limits
        x_min = x0 if x0 <= x_min else x_min
        x_max = x1 if x1 >= x_max else x_max
        y_min = y0 if y1 <= y_min else y_min
        y_max = y1 if y1 >= y_max else y_max

    extent: tuple[float, float, float, float] = (x_min, x_max, y_min, y_max)

    # Template of results dictionary
    result: dict = {
        "series_name": series_name,
        "number_of_members": len(channels),
        "is_stack": num_frames > 1,
        "is_montage": num_tiles > 1,
        "output_files": {},
        "thumbnails": {},
        "thumbnail_size": (512, 512),  # height, row
        "metadata": str(img_xml_file.resolve()),
        "parent_lif": str(file.resolve()),
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

        # Estimate suitable contrast limits for the dataset
        logger.info("Estimating global intensity range")
        estimate_intensity_start_time = time.perf_counter()
        global_vmin, global_vmax = get_percentiles(
            image_loaders=[
                LIFImageLoader(
                    lif_file=file,
                    scene_num=scene_num,
                    channel_num=c,
                    frame_num=z,
                    tile_num=t,
                )
                for z in range(num_frames)
                for t in range(num_tiles)
                if not z % 2 and not t % 2
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
            frame_pixel_size = max(frame_x_len / 2400, frame_y_len / 2400)
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
                        load_and_resize_tile,
                        LIFImageLoader(
                            lif_file=file,
                            scene_num=scene_num,
                            channel_num=c,
                            frame_num=f,
                            tile_num=t,
                        ),
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

        else:
            logger.info("Constructing image stack")
            array_loading_start_time = time.perf_counter()
            arr = np.zeros((num_frames, y_pixels, x_pixels), dtype=np.uint8)
            with ThreadPoolExecutor(max_workers=num_procs) as pool:
                futures = [
                    pool.submit(
                        load_and_convert_image,
                        LIFImageLoader(
                            lif_file=file,
                            scene_num=scene_num,
                            channel_num=c,
                            frame_num=z,
                            tile_num=0,
                        ),
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
                            "Failed to load the following image: \n"
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

        # Create and append paths to PNG files to create
        result["thumbnails"][color] = str(
            img_stk_file.parent / ".thumbnails" / f"{img_stk_file.stem}.png"
        )

    end_time = time.perf_counter()
    logger.info(f"Completed processing of {series_name} in {end_time - start_time}s")
    logger.debug(
        "Returning the following processing results: \n"
        f"{json.dumps(result, indent=2, default=str)}"
    )
    return result


def process_lif_file(
    file: Path,
    root_folder: str,  # Name of the folder to treat as the root folder for LIF files
    number_of_processes: int = 1,  # Number of processing threads to run
) -> list[dict]:
    """
    Takes a LIF file, extracts its metadata as an XML tree, then parses through the
    subimages stored inside it, saving each channel in the subimage as a separate
    image or image stack. It uses information stored in the metadata to name the
    individual series.

    FOLDER STRUCTURE:
    parent_folder
    |__ images          <- Raw data stored here
    |   |__ sample_name     <- Folders for samples
    |       |__ lif files   <- LIF files of specific sample
    |       |__ metadata    <- Save raw XML metadata file here
    |__ processed       <- Processed data goes here
        |__ sample_name
            |__ lif_file_names      <- Folders for data from the same LIF file
                |__ sub_image       <- Folders for individual subimages
                |   |__ tiffs       <- Save channels as individual images or image stacks
                |   |__ metadata    <- Individual XML files saved here (not yet implemented)
    """

    start_time = time.perf_counter()

    # Validate processor count input
    num_procs = number_of_processes  # Use shorter phrase in script
    if num_procs < 1:
        logger.warning("Processor count set to zero or less; resetting to 1")
        num_procs = 1

    # Folder for processed files with same structure as old one
    new_root_folder = "processed"
    file_name = file.stem.replace(" ", "_")  # Replace spaces
    path_parts = list(file.parts)

    # Build path to processed directory
    # Remove leading "/" in Unix paths
    path_parts[0] = "" if path_parts[0] == "/" else path_parts[0]
    try:
        # Search for root folder with case-insensitivity
        root_index = [p.lower() for p in path_parts].index(root_folder.lower())
        path_parts[root_index] = new_root_folder  # Point to new folder
    except ValueError:
        logger.error(
            f"Subpath {root_folder!r} was not found in image path {str(file)!r}"
        )
        return []
    processed_dir = Path("/".join(path_parts[: root_index + 1]))

    # Create folders if not already present
    raw_xml_dir = file.parent / "metadata"
    for folder in (processed_dir, raw_xml_dir):
        folder.mkdir(parents=True, exist_ok=True)
        logger.info("Created processing directory and folder to store raw metadata in")

    # Load LIF file as a LifFile class
    logger.info(f"Loading {file.name!r}")
    lif_file = LifFile(str(file))  # Stack of scenes
    scene_list = list(lif_file.get_iter_image())  # List of scene names

    # Save original metadata as XML tree
    logger.info("Extracting image metadata")
    xml_root = get_lif_xml_metadata(
        file=lif_file,
        save_xml=raw_xml_dir.joinpath(file_name + ".xml"),
    )

    # Recursively generate list of metadata-containing elements
    metadata_dict = find_image_elements(xml_root)

    # Check that elements match number of images
    if not len(metadata_dict) == len(scene_list):
        logger.error(
            "Error matching metadata list to list of subimages. "
            # Show what went wrong
            f"Metadata entries: {len(metadata_dict)} "
            f"Sub-images: {len(scene_list)} "
        )
        return []

    # Iterate through scenes
    logger.info(f"Examining subimages in {file.name!r}")

    # Iterate across the series in the pool
    results = [
        process_lif_subimage(
            file,
            i,
            metadata,
            processed_dir,
            series_path,
            num_procs,
        )
        for i, (series_path, metadata) in enumerate(metadata_dict.items())
    ]

    end_time = time.perf_counter()
    logger.debug(f"Processed LIF file {file} in {end_time - start_time}s")
    return results


class ProcessRawLIFsParameters(BaseModel):
    """
    Pydantic model for validating the received message for the LIF file conversion
    workflow.

    The keys under the "job_parameters" key in the Zocalo wrapper recipe must match
    the attributes present in this model. Attributes in the model with a default
    value don't have to be provided in the wrapper recipe.
    """

    lif_file: Path
    root_folder: str  # The root folder under which all LIF files are saved
    num_procs: int = 20  # Number of processing threads to run


class ProcessRawLIFsWrapper:
    def __init__(self, recwrap):
        self.log = logging.LoggerAdapter(logger)
        self.recwrap = recwrap

    def run(self) -> bool:
        """
        Reads the Zocalo wrapper recipe, loads the parameters, and passes them to the
        LIF file processing function. Upon collecting the results, it then passes them
        back to Murfey for the next stage in the workflow.
        """

        params_dict = self.recwrap.recipe_step["job_parameters"]
        try:
            params = ProcessRawLIFsParameters(**params_dict)
        except (ValidationError, TypeError) as error:
            logger.error(
                "ProcessRawLIFsParameters validation failed for the following parameters: \n"
                f"{json.dumps(params_dict, indent=2, default=str)}\n"
                f"with exception: {error}"
            )
            return False

        # Process files and collect output
        try:
            results = process_lif_file(
                file=params.lif_file,
                root_folder=params.root_folder,
                number_of_processes=params.num_procs,
            )
        # Log error and return False if the command fails to execute
        except Exception:
            logger.error(
                f"Exception encontered while processing LIF file {str(params.lif_file)!r}: \n",
                exc_info=True,
            )
            return False
        if not results:
            logger.error(f"Failed to extract subimages from {str(params.lif_file)!r}")
            return False

        # Send each subset of output files to Murfey for registration
        for result in results:
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

            # Create dictionary and send it to Murfey's "feedback_callback" function
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
