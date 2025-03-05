"""
Contains functions needed in order to run a service to convert the raw CLEM data (from
either a LIF or TIFF file) into image stacks for use in subsequent stages of the CLEM
processing workflow.
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np
from pydantic import BaseModel, ValidationError
from readlif.reader import LifFile

from cryoemservices.util.clem_array_functions import (
    estimate_int_dtype,
    preprocess_img_stk,
    write_stack_to_tiff,
)
from cryoemservices.util.clem_raw_metadata import get_image_elements

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


def process_lif_substack(
    file: Path,
    scene_num: int,
    metadata: ET.Element,
    root_save_dir: Path,
) -> list[dict]:
    """
    Takes the LIF file and its corresponding metadata and loads the relevant sub-stack,
    with each channel as its own array. Rescales their intensity values to utilise the
    whole channel, scales them down to 8-bit, then saves each each array as a separate
    TIFF image stack.
    """

    # Load LIF file
    image = LifFile(str(file)).get_image(scene_num)

    # Get name of sub-image
    file_name = file.stem.replace(" ", "_")  # Remove spaces
    img_name = metadata.attrib["Name"].replace(" ", "_")  # Remove spaces
    logger.info(f"Processing {file_name}-{img_name}")

    # Create save dirs for TIFF files and their metadata
    save_dir = (  # Save directory for all substacks from this LIF file
        root_save_dir
        / "/".join(file.relative_to(root_save_dir.parent).parts[1:-1])
        / file_name
    )
    img_dir = save_dir / img_name  # Save directory for this specific substack
    img_xml_dir = img_dir / "metadata"  # Save metadata relative to the substack
    for folder in (img_dir, img_xml_dir):
        if not folder.exists():
            # Potential race condition when generating folders from multiple pools
            folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created {folder}")
        else:
            logger.info(f"{folder} already exists")

    # Create a name for this series
    series_name = (
        img_dir.relative_to(root_save_dir)
        .as_posix()
        .replace("/", "--")
        .replace(" ", "_")
    )

    # Save image stack XML metadata (all channels together)
    img_xml_file = img_xml_dir / (img_name + ".xml")
    metadata_tree = ET.ElementTree(metadata)
    ET.indent(metadata_tree, "  ")
    metadata_tree.write(img_xml_file, encoding="utf-8")
    logger.info(f"Image stack metadata saved to {img_xml_file}")

    # Load channels
    channels = metadata.findall(
        "Data/Image/ImageDescription/Channels/ChannelDescription"
    )
    colors = [channels[c].attrib["LUTName"].lower() for c in range(len(channels))]

    # Generate slice labels for later
    num_frames = image.dims.z
    image_labels = [f"{f}" for f in range(num_frames)]

    # Get x, y, and z resolution (pixels per um)
    x_res = image.scale[0]
    y_res = image.scale[1]

    # Process z-axis if it exists
    z_res: float = image.scale[2] if num_frames > 1 else float(0)

    # Load timestamps (might be useful in the future)
    # timestamps = metadata.find("Data/Image/TimeStampList")

    # Process channels as individual TIFFs
    results: list[dict] = []
    for c in range(len(colors)):

        # Get color
        color = colors[c]
        logger.info(f"Processing {color} channel")

        # Load image stack to array
        logger.info("Loading image stack")
        for z in range(num_frames):
            frame = image.get_frame(z=z, t=0, c=c)  # PIL object; array-like
            if z == 0:
                arr = np.array([frame])
            else:
                arr = np.append(arr, [frame], axis=0)
        logger.debug(
            f"{img_name} {color} array properties: \n"
            f"Shape: {arr.shape} \n"
            f"dtype: {arr.dtype} \n"
            f"Min value: {arr.min()} \n"
            f"Max value: {arr.max()} \n"
        )

        # Estimate initial NumPy dtype
        bit_depth = image.bit_depth[c]
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
            f"{img_name} {color} array properties: \n"
            f"Shape: {arr.shape} \n"
            f"dtype: {arr.dtype} \n"
            f"Min value: {arr.min()} \n"
            f"Max value: {arr.max()} \n"
        )

        # Save as a greyscale TIFF
        logger.info("Processing image stack")
        img_stk_file = write_stack_to_tiff(
            array=arr,
            save_dir=img_dir,
            file_name=color,
            x_res=x_res,
            y_res=y_res,
            z_res=z_res,
            units="micron",
            axes="ZYX",
            image_labels=image_labels,
            photometric="minisblack",
        )
        # Collect the image stacks created
        result = {
            "image_stack": str(img_stk_file.resolve()),
            "metadata": str(img_xml_file.resolve()),
            "series_name": series_name,
            "channel": color,
            "number_of_members": len(channels),
            "parent_lif": str(file.resolve()),
        }
        results.append(result)

    return results


def convert_lif_to_stack(
    file: Path,
    root_folder: str,  # Name of the folder to treat as the root folder for LIF files
    number_of_processes: int = 1,  # Number of processing threads to run
) -> list[dict]:
    """
    Takes a LIF file, extracts its metadata as an XML tree, then parses through the
    sub-images stored inside it, saving each channel in the sub-image as a separate
    image stack. It uses information stored in the XML metadata to name the individual
    image stacks.

    FOLDER STRUCTURE:
    parent_folder
    |__ images          <- Raw data stored here
    |   |__ sample_name     <- Folders for samples
    |       |__ lif files   <- LIF files of specific sample
    |       |__ metadata    <- Save raw XML metadata file here
    |__ processed       <- Processed data goes here
        |__ sample_name
            |__ lif_file_names      <- Folders for data from the same LIF file
                |__ sub_image       <- Folders for individual sub-images
                |   |__ tiffs       <- Save channels as individual image stacks
                |   |__ metadata    <- Individual XML files saved here (not yet implemented)
    """

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
            f"Subpath {root_folder!r} was not found in image path " f"{str(file)!r}"
        )
        return []
    processed_dir = Path("/".join(path_parts[: root_index + 1]))

    # Save master metadata relative to raw file
    raw_xml_dir = file.parent / "metadata"

    # Create folders if not already present
    for folder in (processed_dir, raw_xml_dir):
        if not folder.exists():
            folder.mkdir(parents=True)
            logger.info(f"Created {str(folder)!r}")
        else:
            logger.info(f"{str(folder)!r} already exists")

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
    metadata_list = get_image_elements(xml_root)

    # Check that elements match number of images
    if not len(metadata_list) == len(scene_list):
        logger.error(
            "Error matching metadata list to list of sub-images. "
            # Show what went wrong
            f"Metadata entries: {len(metadata_list)} "
            f"Sub-images: {len(scene_list)} "
        )
        return []

    # Iterate through scenes
    logger.info("Examining sub-images")

    # Set up multiprocessing arguments
    pool_args = []
    for i in range(len(scene_list)):
        pool_args.append(
            # Arguments need to be pickle-able; no complex objects allowed
            #   Follow order of args in the function
            [
                file,
                i,
                metadata_list[i],
                processed_dir,
            ]
        )

    # Parallel process image stacks and return results
    with mp.Pool(processes=num_procs) as pool:
        # Each thread will return a list of dicts
        results_map = pool.starmap(process_lif_substack, pool_args)

    # Return flattened list of dicts
    results = list(itertools.chain.from_iterable(results_map))
    return results


class LIFToStackParameters(BaseModel):
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


class LIFToStackWrapper:
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
            params = LIFToStackParameters(**params_dict)
        except (ValidationError, TypeError) as error:
            logger.error(
                "LIFToStackParameters validation failed for parameters: "
                f"{params_dict} with exception: {error}"
            )
            return False

        # Process files and collect output
        results = convert_lif_to_stack(
            file=params.lif_file,
            root_folder=params.root_folder,
            number_of_processes=params.num_procs,
        )

        # Return False and log error if the command fails to execute
        if results is None:
            logger.error(
                f"Failed to extract image stacks from {str(params.lif_file)!r}"
            )
            return False
        # Send each subset of output files to Murfey for registration
        for result in results:
            # Create dictionary and send it to Murfey's "feedback_callback" function
            murfey_params = {
                "register": "clem.register_lif_preprocessing_result",
                "result": result,
            }
            self.recwrap.send_to("murfey_feedback", murfey_params)
            logger.info(
                f"Submitted {result['series_name']!r} {result['channel']!r} "
                "image stack and associated metadata to Murfey for registration"
            )

        return True
