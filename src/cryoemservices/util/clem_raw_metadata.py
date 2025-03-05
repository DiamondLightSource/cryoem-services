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
from typing import Generator
from xml.etree import ElementTree as ET

logger = logging.getLogger("cryoemservice.util.clem_raw_metadata")


def get_image_elements(root: ET.Element) -> list[ET.Element]:
    """
    Searches the XML metadata recursively to find the nodes tagged as "Element" that
    have image-related tags. Some LIF datasets have layers of nested elements, so a
    recursive approach is needed to avoid certain datasets breaking it.
    """

    def _find_elements_recursively(
        node: ET.Element,
    ) -> Generator[ET.Element, None, None]:

        # Check if the node itself is labelled "Element"
        if node.tag == "Element":
            yield node
        # Find all descendants that are labelled "Element"
        for elem in node.findall(".//Element"):
            yield elem

    # Find all element nodes, but keep only ones that have image-related tags
    elem_list = [
        elem for elem in _find_elements_recursively(root) if elem.find("./Data/Image")
    ]
    return elem_list


def get_axis_resolution(element: ET.Element) -> float:
    """
    Calculates the resolution (pixels per unit length) for the x-, y-, and z-axes.
    Follows "readlif" convention of subtracting 1 from the number of frames/pixels
    to maintain consistency with its output.
    """
    # Use shortened variables
    elem = element

    # Verify
    if elem.tag != "DimensionDescription" and elem.attrib.get("Unit", "") != "m":
        logger.error("This element does not have dimensional information")
        raise ValueError("This element does not have dimensional information")

    # Calculate
    length = (
        float(elem.attrib.get("Length", "")) - float(elem.attrib.get("Origin", ""))
    ) * 10**6  # Convert to um
    pixels = int(elem.attrib.get("NumberOfElements", ""))
    resolution = (pixels - 1) / length  # Pixels per um

    return resolution
