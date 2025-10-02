"""
=======================================================================================
XML-RELATED FUNCTIONS
=======================================================================================

Functions to handle file types that can be read in as XML Element objects.
These include, but are not limited to:
    1.  XML (self-explanatory)
    2.  XLIF (used when reconstructing image stacks from TIFFs)
    3.  XLEF
    4.  XLCF
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

logger = logging.getLogger("cryoemservice.util.clem_raw_metadata")


def find_image_elements(
    node: ET.Element, path: str = "", result: Optional[dict] = None
) -> dict[str, ET.Element]:
    """
    Searches the XML metadata recursively to find the nodes tagged as "Element" that
    have image-related tags. Some LIF datasets have layers of nested elements, so a
    recursive approach is needed to avoid certain datasets breaking it.
    """

    if result is None:
        result = {}

    # Look for Element nodes
    if node.tag == "Element":
        name = node.get("Name")
        # Only process the Element node if it has a name
        if name:
            current_name = name.strip().replace(" ", "_")
            # Remove file extension if present
            current_name = Path(current_name).stem
            # Construct full path to current node
            new_path = f"{path}/{current_name}" if path else current_name
            # Add to dictionary if current node contains image-related data
            if node.find("./Data/Image"):
                result[new_path] = node
            # Use updated path for child traversal
            path = new_path

    # Run function recursively until no more child nodes are found
    for child in node:
        find_image_elements(child, path, result)
    return result


def get_channel_info(node: ET.Element) -> dict[str, dict]:
    """
    Parses the XML metadata of a single dataset (this will raise an error if
    the XML metadata contains multiple datasets) to extract information about
    the colour chnanels present in the dataset.
    """

    # Load channels
    channels = node.findall(".//ChannelDescription")

    # Raise error if multiple datasets are present
    colors = [channel.get("LUTName", "").lower() for channel in channels]
    if len(colors) != len(set(colors)):
        raise ValueError(
            "More than one node found describing the same colour channel. "
            "Metadata for multiple datasets are likely present."
        )

    # Extract channel information
    channel_info = {}
    for channel in channels:
        try:
            channel_info[channel.get("LUTName", "").lower()] = {
                "bit_depth": int(channel.get("Resolution", "")),
                "min": float(channel.get("Min", "")),
                "max": float(channel.get("Max", "")),
            }
        except (ValueError, TypeError):
            logger.error("Unable to extract channel information")
            continue
    return channel_info


def get_dimension_info(node: ET.Element) -> dict[str, dict]:
    """
    Parses the XML metadata from a single dataset (this will raise an error
    if the XML metadata contains multiple datasets) to calculate and return
    dimension information.

    This information will include the length, pixel size, resolution, and
    units of the x, y, and z axes, along with the number of images present
    in an overview.

    """

    dimensions_key = (
        ("1", "x"),
        ("2", "y"),
        ("3", "z"),
        ("10", "m"),
    )

    dims = node.findall(".//DimensionDescription")

    dims_info: dict[str, dict] = {}
    for dim_id, dim_name in dimensions_key:
        # Check that the metadata is for one dataset only
        search_results = [dim for dim in dims if dim.get("DimID") == dim_id]
        if len(search_results) > 1:
            raise ValueError(
                f"More than one node found for the {dim_name}-axis. "
                "Metadata for multiple datasets are likely present."
            )
        if len(search_results) == 0:
            dims_info[dim_name] = {}
            continue
        dim_info = search_results[0]
        num_elements = int(dim_info.get("NumberOfElements", ""))

        # Handle x, y, and z axes differently
        if dim_id != "10":
            origin = float(dim_info.get("Origin", ""))
            length = float(dim_info.get("Length", ""))
            units: str = dim_info.get("Unit", "")

            # Get the end-to-end length of the axis
            # 'Length' is midpoint-to-midpoint length
            end_to_end_length = (length - origin) * num_elements / (num_elements - 1)

            # Calculate resolution and pixel size
            resolution = num_elements / end_to_end_length  # Pixels per unit
            pixel_size = 1 / resolution  # Units per pixel

            # Update dictionary
            dims_info[dim_name] = (
                {
                    "num_frames": num_elements,
                }
                if dim_id == "3"
                else {
                    "num_pixels": num_elements,
                }
            )
            dims_info[dim_name].update(
                {
                    "length": end_to_end_length,
                    "units": units,
                    "resolution": resolution,
                    "pixel_size": pixel_size,
                }
            )
        else:
            dims_info[dim_name] = {"num_tiles": num_elements}
    return dims_info


def get_tile_scan_info(node: ET.Element):
    # Placeholder dict
    tile_scan_info: dict[int, dict] = {}

    # Look for nodes named "TileScanInfo"
    search_results = [
        child
        for child in node.findall(".//Attachment")
        if child.get("Name", "") == "TileScanInfo"
    ]
    if search_results:
        # Raise error if more than one is found
        if len(search_results) > 1:
            raise ValueError(
                "More than one 'TileScanInfo' node found. "
                "Metadata for multiple datasets are likely present."
            )
        # Extract tile position information
        for t, tile in enumerate(search_results[0]):
            tile_scan_info[t] = {
                "field_x": int(tile.get("FieldX", "")),
                "field_y": int(tile.get("FieldY", "")),
                "pos_x": float(tile.get("PosX", "")),
                "pos_y": float(tile.get("PosY", "")),
            }
    # If "TileScanInfo" is not found, look for "ATLCameraSettingDefinition"
    else:
        search_results = node.findall(".//ATLCameraSettingDefinition")
        if not search_results:
            raise KeyError(
                "No tile scan information was found in the provided metadata"
            )
        # Raise error if more than one is found
        if len(search_results) > 1:
            raise ValueError(
                "More than one 'ATLCameraSettingDefinition' node found. "
                "Metadata for multiple datasets are likely present."
            )
        camera_settings = search_results[0]
        tile_scan_info[0] = {
            "field_x": 0,
            "field_y": 0,
            "pos_x": float(camera_settings.get("StagePosX", "")),
            "pos_y": float(camera_settings.get("StagePosY", "")),
        }
    return tile_scan_info
