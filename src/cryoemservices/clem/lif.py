"""
Supporting functions to extract TIFF image stacks from the LIF files produced by the
Leica light microscope.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np
from readlif.reader import LifFile

from cryoemservices.clem.images import process_img_stk, write_to_tiff
from cryoemservices.clem.xml import get_image_elements

# Create logger object to output messages with
logger = logging.getLogger("cryoemservices.clem.lif")


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


def process_lif_file(
    file: Path,
    scene_num: int,
    metadata: ET.Element,
    save_dir: Path,
) -> bool:
    """
    Takes the LIF file and its corresponding metadata and loads the relevant sub-stack,
    with each channel as its own array. Rescales their intensity values to utilise the
    whole channel, scales them down to 8-bit, then saves each each array as a separate
    TIFF image stack.
    """

    # Load LIF file
    file_name = file.stem.replace(" ", "_")
    image = LifFile(str(file)).get_image(scene_num)

    # Get name of sub-image
    img_name = metadata.attrib["Name"].replace(" ", "_")  # Remove spaces
    logger.info(f"Processing {file_name}-{img_name}")

    # Create save dirs for TIFF files and their metadata
    img_dir = save_dir / img_name
    img_xml_dir = img_dir / "metadata"
    for folder in (img_dir, img_xml_dir):
        if not folder.exists():
            # Potential race condition when generating folders from multiple pools
            folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created {folder}")
        else:
            logger.info(f"{folder} already exists")

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
    for c in range(len(colors)):

        # Get color
        color = colors[c]
        logger.info(f"Processing {color} channel")

        # Get bit depth
        bit_depth = image.bit_depth[c]

        # Load image stack to array
        logger.info("Loading image stack")
        for z in range(num_frames):
            frame = image.get_frame(z=z, t=0, c=c)  # PIL object; array-like
            if z == 0:
                arr = np.array([frame])
            else:
                arr = np.append(arr, [frame], axis=0)
        logger.info(
            f"{img_name} {color} array has the dimensions {np.shape(arr)} \n"
            f"Min value: {np.min(arr)} \n"
            f"Max value: {np.max(arr)} \n"
        )

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
        arr = process_img_stk(
            array=arr,
            initial_bit_depth=bit_depth,
            target_bit_depth=8,
            adjust_contrast=adjust_contrast,
        )

        # Save as a greyscale TIFF
        arr = write_to_tiff(
            array=arr,
            save_dir=img_dir,
            series_name=color,
            x_res=x_res,
            y_res=y_res,
            z_res=z_res,
            units="micron",
            axes="ZYX",
            image_labels=image_labels,
            photometric="minisblack",
        )

    return True


def convert_lif_to_stack(
    file: Path,
    root_folder: str,  # Name of the folder to treat as the root folder for LIF files
    number_of_processes: int = 1,  # Number of processing threads to run
) -> bool:
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
    file_name = file.stem.replace(" ", "_")  # Replace spaces
    path_parts = list(file.parts)
    new_root_folder = "processed"
    # Rewrite string in-place
    counter = 0
    for p in range(len(path_parts)):
        part = path_parts[p]
        # Omit initial "/" in Linux file systems for subsequent rejoining
        if part == "/":
            path_parts[p] = ""
        # Rename designated root folder to "processed"
        if (
            part.lower() == root_folder.lower() and counter < 1
        ):  # Remove case-sensitivity
            path_parts[p] = new_root_folder
            counter += 1  # Do for first instance only
    # Check that "processed" has been inserted into file path
    if new_root_folder not in path_parts:
        logger.error(
            f"Subpath {root_folder!r} was not found in image path " f"{str(file)!r}"
        )
        return False

    # Create folders if not already present
    processed_dir = Path("/".join(path_parts)).parent / file_name  # Processed images
    raw_xml_dir = file.parent / "metadata"  # Raw metadata
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
        raise IndexError(
            "Error matching metadata list to list of sub-images. \n"
            # Show what went wrong
            f"Metadata entries: {len(metadata_list)} \n"
            f"Sub-images: {len(scene_list)}"
        )

    # Iterate through scenes
    logger.info("Examining sub-images")

    # Set up multiprocessing arguments
    pool_args = []
    for i in range(len(scene_list)):
        pool_args.append(
            # Arguments need to be pickle-able; no complex objects allowed
            [  # Follow order of args in the function
                file,  # Load LIF file in the sub-process
                i,
                metadata_list[i],  # Corresponding metadata
                processed_dir,
            ]
        )

    # Parallel process image stacks
    with mp.Pool(processes=num_procs) as pool:
        result = pool.starmap(process_lif_file, pool_args)

    if result:
        return all(result)
    return False
