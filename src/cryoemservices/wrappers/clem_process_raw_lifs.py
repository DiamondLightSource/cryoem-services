"""
Contains functions needed in order to run a service to convert the raw CLEM data (from
either a LIF or TIFF file) into image stacks for use in subsequent stages of the CLEM
processing workflow.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np
from matplotlib import pyplot as plt
from pydantic import BaseModel, ValidationError
from readlif.reader import LifFile

from cryoemservices.util.clem_array_functions import (
    estimate_int_dtype,
    preprocess_img_stk,
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


def get_percentiles(
    file: Path,
    scene_num: int,
    channel_num: int,
    frame_num: int,
    tile_num: int,
    percentiles: tuple[float, float] = (1, 99),
):
    # Load the subimage
    image = LifFile(str(file)).get_image(scene_num)
    try:
        arr = image.get_frame(
            z=frame_num,
            c=channel_num,
            m=tile_num,
        )
    except Exception:
        logger.warning(f"Unable to load frame {frame_num}-{tile_num}")
        return (None, None)
    p_lo, p_hi = np.percentile(arr, percentiles)
    return p_lo, p_hi


def stitch_image_frames(
    file: Path,
    scene_num: int,
    channel_num: int,
    frame_num: int,
    tile_scan_info: dict[int, dict],
    image_width: float,
    image_height: float,
    extent: tuple[float, float, float, float],
    dpi: int | float = 300,
    color_limits: tuple[Optional[float], Optional[float]] = (None, None),
) -> np.ndarray:
    """
    Helper function to dynamically stitch together image tiles to create a montage.

    This function makes use of Matplotlib's 'imshow' to define the position and size
    of the image tiles and render them on a blank canvas spatially relative to one
    another. The plot is then converted into a NumPy array, which is returned and
    used to build up the final image stack.
    """

    # Create the figure to stitch the tiles in
    fig, ax = plt.subplots(facecolor="black")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.set_dpi(dpi)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.invert_yaxis()  # Set origin to top left of figure
    ax.set_aspect("equal")  # Ensure same scales on x- and y-axis
    ax.axis("off")
    ax.set_facecolor("black")

    # Unpack colour limits
    vmin, vmax = color_limits

    for tile_num, tile_info in tile_scan_info.items():
        logger.info(f"Loading tile {tile_num}")
        try:
            # Get extent of current tile
            x = float(tile_info["pos_x"])
            y = float(tile_info["pos_y"])
            tile_extent = [
                x,
                x + image_width,
                y,
                y + image_height,
            ]

            # Add tile to the montage
            arr = (
                LifFile(file)
                .get_image(scene_num)
                .get_frame(z=frame_num, c=channel_num, m=tile_num)
            )
            ax.imshow(
                arr,
                extent=tile_extent,
                origin="lower",
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
            )
        except Exception:
            logger.warning(
                f"Unable to add file {str(file)!r} to the montage",
                exc_info=True,
            )
            continue

    # Extract the stitched image as an array
    fig.canvas.draw()  # Render the figure using Matplotlib's engine
    canvas = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    canvas_width, canvas_height = fig.canvas.get_width_height()
    canvas = canvas.reshape(canvas_height, canvas_width, 4)
    logger.debug(f"Rendered canvas has shape {canvas.shape}")

    # Find the extent of the stitched image on the rendered canvas
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    x0, y0, x1, y1 = [int(coord * fig.dpi) for coord in bbox.extents]
    logger.debug(f"Image extents on canvas are {[x0, x1, y0, y1]}")

    # Extract and return the stitched image as an array
    frame = np.array(canvas[y0:y1, x0:x1, :3].mean(axis=-1), dtype=np.uint8)
    plt.close()
    logger.debug(f"Image frame has shape {frame.shape}")
    return frame


def process_lif_subimage(
    file: Path,
    scene_num: int,
    metadata: ET.Element,
    root_save_dir: Path,
    save_path: str = "",
    num_procs: int = 1,
) -> dict:
    """
    Takes the LIF file and its corresponding metadata and loads the relevant subimage.
    For image stacks, it will load each colour channel as its own image stack, rescale
    the intensity values to utilise the whole channel, convert it into 8-bit grayscale,
    and save them as individual images or image stacks.

    For montages, it will stitch the different subimages together and save the final
    composite image as an atlas.
    """

    # Load LIF file
    image = LifFile(str(file)).get_image(scene_num)

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

    # Get width and height for a single frame
    w = float(dims["x"]["length"])
    h = float(dims["y"]["length"])

    # Get number of frames
    num_frames = int(dims["z"].get("num_frames", 1))
    num_tiles = int(dims["m"].get("num_tiles", 1))

    # Calculate the full extent of the image
    # Initial arbitrary limits of the atlas in real space
    x_min = 10e10
    x_max = float(0)
    y_min = 10e10
    y_max = float(0)

    for tile in tile_scan_info.values():
        x = float(tile["pos_x"])
        y = float(tile["pos_y"])

        # Update the atlas limits
        x_min = x if x < x_min else x_min
        y_min = y if y < y_min else y_min
        x_max = x + w if x + w > x_max else x_max
        y_max = y + h if y + h > y_max else y_max
    extent = [x_min, x_max, y_min, y_max]

    # Template of results dictionary
    result: dict = {
        "series_name": series_name,
        "number_of_members": len(channels),
        "is_stack": num_frames > 1,
        "is_montage": num_tiles > 1,
        "output_files": {},
        "metadata": str(img_xml_file.resolve()),
        "parent_lif": str(file.resolve()),
        "pixels_x": None,
        "pixels_y": None,
        "units": "",
        "pixel_size": None,
        "resolution": None,
        "extent": [],  # [x_min, x_max, y_min, y_max] in real space
    }

    # Iterate by color, then z-frame, then tile
    for c, color in enumerate(channels.keys()):
        logger.info(f"Processing {color} channel for {series_name!r}")

        # If it's a montage, stitch the tiles together in Matplotlib
        if num_tiles > 1:
            logger.info("Estimating global intensity range")
            with mp.Pool(processes=num_procs) as pool:
                min_max_list = pool.starmap(
                    get_percentiles,
                    # Construct pool arguments in-line
                    [
                        (
                            file,
                            scene_num,
                            c,
                            z,
                            t,
                            (0.5, 99.5),
                        )
                        for z in range(num_frames)
                        for t in range(num_tiles)
                    ],
                )
            global_vmin = int(np.percentile([m for m, _ in min_max_list if m], [5])[0])
            global_vmax = int(np.percentile([m for _, m in min_max_list if m], [95])[0])

            # Stitch frames together
            with mp.Pool(processes=num_procs) as pool:
                frame_list = pool.starmap(
                    stitch_image_frames,
                    # Construct pool arguments in-line
                    [
                        (
                            file,
                            scene_num,
                            c,
                            z,
                            tile_scan_info,
                            w,
                            h,
                            extent,
                            500,
                            (global_vmin, global_vmax),
                        )
                        for z in range(num_frames)
                    ],
                )
            arr = np.array(frame_list, dtype=np.uint8)

            # Update resolution and pixel size in dimensions dictionary
            h_new, w_new = arr.shape[-2:]
            dims["x"]["num_pixels"] = w_new
            dims["x"]["resolution"] = w_new / (x_max - x_min)
            dims["x"]["pixel_size"] = (x_max - x_min) / w_new
            dims["y"]["num_pixels"] = h_new
            dims["y"]["resolution"] = h_new / (y_max - y_min)
            dims["y"]["pixel_size"] = (y_max / y_min) / h_new

        else:
            for z in range(num_frames):
                frame = image.get_frame(z=z, c=c)
                logger.info(f"Loaded frame {z}")

                if z == 0:
                    arr = np.array([frame])
                else:
                    arr = np.append(arr, [frame], axis=0)

        # Update results dictionary once per dataset
        if c == 0:
            # Update results dictionary
            result["pixels_x"] = dims["x"]["num_pixels"]
            result["pixels_y"] = dims["y"]["num_pixels"]
            result["units"] = dims["x"]["units"]
            result["pixel_size"] = dims["x"]["pixel_size"]
            result["resolution"] = dims["x"]["resolution"]
            extent = [x_min, x_max, y_min, y_max]
            result["extent"] = extent

        # Estimate initial NumPy dtype
        dtype_init = estimate_int_dtype(
            arr,
            bit_depth=8 if num_tiles > 1 else channels[color]["bit_depth"],
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
            and num_tiles == 1  # Contrast adjustment already applied during stitching
            else None
        )
        logger.info(f"Contrast adjustment set to {adjust_contrast}")

        # Process the subimage
        logger.info("Applying image processing routine to subimage")
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
        )
        # Collect the images created
        result["output_files"][color] = str(img_stk_file.resolve())

    logger.debug(f"Processing results are as follows: {result}")
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
        results = process_lif_file(
            file=params.lif_file,
            root_folder=params.root_folder,
            number_of_processes=params.num_procs,
        )

        # Return False and log error if the command fails to execute
        if not results:
            logger.error(f"Failed to extract subimages from {str(params.lif_file)!r}")
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
                f"Submitted processed data for {result['series_name']!r} "
                "and associated metadata to Murfey for registration"
            )

        return True
