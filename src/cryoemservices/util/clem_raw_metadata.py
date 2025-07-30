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


def get_axis_resolution(node: ET.Element) -> float:
    """
    Calculates the resolution (pixels per unit length) for the x-, y-, and z-axes.
    Follows "readlif" convention of subtracting 1 from the number of frames/pixels
    to maintain consistency with its output.
    """
    # Verify
    if node.tag != "DimensionDescription" and node.attrib.get("Unit", "") != "m":
        message = "This node does not have dimensional information"
        logger.error(message)
        raise ValueError(message)

    # Calculate
    length = (
        float(node.get("Length", "")) - float(node.get("Origin", ""))
    ) * 10**6  # Convert to um
    pixels = int(node.get("NumberOfElements", ""))
    resolution = (pixels - 1) / length  # Pixels per um

    return resolution
