"""
Recurring helper functions used when generating the fixtures needed to test the
CLEM workflow.
"""

from __future__ import annotations

import math
import uuid
import xml.etree.ElementTree as ET
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Literal

import numpy as np


def gaussian_2d(
    shape: tuple[int, int],
    amplitude: float,
    centre: tuple[int, int],
    sigma: tuple[float, float],
    theta: float,
    offset: float,
):
    """
    Helper function to create Gaussian peaks
    """
    x0, y0 = centre
    sig_x, sig_y = sigma

    # Create meshgrid
    rows, cols = shape
    y, x = np.meshgrid(np.arange(cols), np.arange(rows), indexing="ij")

    x_rot: np.ndarray = (x - x0) * np.cos(np.deg2rad(theta)) + (y - y0) * np.sin(
        np.deg2rad(theta)
    )
    y_rot: np.ndarray = (y - y0) * np.cos(np.deg2rad(theta)) - (x - x0) * np.sin(
        np.deg2rad(theta)
    )

    # Compute and return Gaussian
    gaussian = (
        amplitude * np.exp(-(x_rot**2 / (2 * sig_x**2)) - (y_rot**2 / (2 * sig_y**2)))
        + offset
    )

    return gaussian


def create_grayscale_image(
    shape: tuple[int, int],
    num_frames: int,
    dtype: str,
    peaks: list[dict[str, Any]],
    peak_shift_per_frame: tuple[int, int],
    intensity_offset_per_frame: int,
):
    """
    Creates a grayscale image with peaks that are offset from frame-to-frame
    """

    x_shift, y_shift = peak_shift_per_frame
    c_off = intensity_offset_per_frame
    if num_frames == 1:
        arr = np.zeros(shape, dtype=dtype)
        for peak in peaks:
            arr += gaussian_2d(**peak).astype(dtype)
    else:
        arr = np.zeros((num_frames, *shape), dtype=dtype)
        for f in range(num_frames):
            for peak in peaks:
                # Adjust the peak offset per frame
                centre: tuple[int, int] = peak["centre"]
                x, y = centre
                peak["centre"] = (x + x_shift, y + y_shift)
                peak["offset"] += c_off
                arr[f] += gaussian_2d(**peak).astype(dtype)

    arr = arr.astype(dtype)

    return arr


def to_scientific_notation(value: float, precision=6):
    """
    Helper function to return stringified floats with the same degree of precision in
    both the base and exponent as that found in the XML metadata of the LIF files.

    E.g.:
    2.661100e-004
    0.000000e+000
    """
    scientific_str = f"{value:.{precision}e}"
    base, exponent = scientific_str.split("e")
    return f"{base}e{int(exponent):+04d}"


def get_hexadecimal_timestamp(time_ns: int):
    """
    Helper function to convert a timestamp (in ns) into its hexadecimal Windows FILETIME
    counterpart, which is what is stored in the XML metadata of the LIF files.
    """

    # Windows FILETIME epoch: Jan 1, 1601
    filetime_epoch_ns = 116444736000000000
    filetime = (time_ns - filetime_epoch_ns) // 100  # FILETIME is in 100-ns intervals

    # Convert to hexadecimal string
    return format(filetime, "x")


def add_tile_scan_info(
    parent_node: ET.Element,
    num_tiles: int,
    tile_offset: float,
    collection_mode: Literal["linear", "grid", ""],
):
    def pad_decimals(value: float):
        """Pad decimals to 10 decimal places and return them as a string."""
        return str(
            Decimal(value).quantize(Decimal("1.0000000000"), rounding=ROUND_HALF_UP)
        )

    # Don't add tile scan info if num_tiles is 0
    if num_tiles >= 1:
        # TileScanInfo stored as Attachement under Image
        tile_scan_info = ET.SubElement(
            parent_node,
            "Attachment",
            {
                "Name": "TileScanInfo",
            },
        )

        # Arrange tiles as close to a square as possible
        num_cols = math.ceil(math.sqrt(num_tiles))
        for t in range(num_tiles):
            col = t % num_cols
            row = t // num_cols
            ET.SubElement(
                tile_scan_info,
                "Tile",
                {
                    "FieldX": str(col) if collection_mode == "grid" else str(t),
                    "FieldY": str(row) if collection_mode == "grid" else "0",
                    "PosX": str(pad_decimals(tile_offset * (col + 1))),
                    "PosY": str(pad_decimals(tile_offset * (row + 1))),
                },
            )

    # HardwareSetting stored as Attachment under Image
    hardware_setting = ET.SubElement(
        parent_node,
        "Attachment",
        {
            "Name": "HardwareSetting",
        },
    )

    # ATLCameraSettingDefinition stored under HardwareSetting
    ET.SubElement(
        hardware_setting,
        "ATLCameraSettingDefinition",
        {
            "VersionNumber": "5",
            "StagePosX": pad_decimals(tile_offset),
            "StagePosY": pad_decimals(tile_offset),
        },
    )
    return parent_node


def add_channel_info(
    parent_node: ET.Element,
    colors: list[str],
    bit_depth: int,
    x_pix: int,
    y_pix: int,
    num_frames: int,
):
    # Channels stored under ImageDescription
    channels = ET.SubElement(
        parent_node,
        "Channels",
    )
    # Incrementally attach channels
    for c, color in enumerate(colors):
        ET.SubElement(
            channels,
            "ChannelDescription",
            {
                "Resolution": str(bit_depth),
                "Min": to_scientific_notation(float(0)),
                "Max": to_scientific_notation(float(2**bit_depth)),
                "Unit": "",
                "LUTName": color.title(),
                "IsLUTInverted": "0",
                "BytesInc": str(c * (bit_depth // 8) * x_pix * y_pix * num_frames),
                "BitInc": "0",
            },
        )
    return parent_node


def add_dimension_info(
    parent_node: ET.Element,
    x_pix: int,
    y_pix: int,
    pixel_size: float,
    num_frames: int,
    z_size: float,
    num_tiles: int,
):
    # Construct list to iteratively construct XML tree with
    dims = [
        # DimID | Num Elements | Pixel Size | Unit | BytesInc
        (1, x_pix, pixel_size * (x_pix - 1), "m", 2),
        (2, y_pix, pixel_size * (y_pix - 1), "m", 2 * x_pix),
    ]
    if num_frames > 1:
        dims.append((3, num_frames, z_size * (num_frames - 1), "m", 2 * x_pix * y_pix))
    if num_tiles > 1:
        dims.append(
            (
                10,
                num_tiles,
                float(num_tiles - 1),
                "",
                2 * x_pix * y_pix * num_frames * 2,
            )
        )

    # Dimensions stored under ImageDescription
    dimensions = ET.SubElement(
        parent_node,
        "Dimensions",
    )

    # DimensionDescription stored under Dimensions
    for dim_id, num_elem, length, unit, bytes_inc in dims:
        ET.SubElement(
            dimensions,
            "DimensionDescription",
            {
                "DimID": str(dim_id),
                "NumberOfElements": str(num_elem),
                "Origin": to_scientific_notation(float(0)),
                "Length": to_scientific_notation(length),
                "Unit": unit,
                "BytesInc": str(bytes_inc),
                "BitInc": "0",
            },
        )
    return parent_node


def add_dataset_element(
    parent_node: ET.Element | None = None,
    series_name: str = "",
    colors: list[str] = [],
    bit_depth: int = 16,
    x_pix: int = 2048,
    y_pix: int = 2048,
    pixel_size: float = 0.000000125,
    num_frames: int = 10,
    z_size: float = 0.00000040,
    num_tiles: int = 1,
    tile_offset: float = 0.00025,
    collection_mode: Literal["linear", "grid", ""] = "",
):
    """
    Helper function to construct a CLEM dataset Element that can be inserted as
    needed into the relevant metadata file, or to be used directly in tests
    """
    # Add Element to Root
    elem_args = (
        "Element",
        {
            "Name": series_name,
            "UniqueID": str(uuid.uuid4()),
        },
    )
    element = (
        ET.SubElement(parent_node, *elem_args)
        if parent_node is not None
        else ET.Element(*elem_args)
    )

    # Add Data to Element
    data = ET.SubElement(element, "Data")

    # Add Image to Data
    image = ET.SubElement(
        data,
        "Image",
        {"TextDescription": ""},
    )

    # Add TileScanInfo to Image
    image = add_tile_scan_info(
        image,
        num_tiles=num_tiles,
        tile_offset=tile_offset,
        collection_mode=collection_mode,
    )

    # Add ImageDescription to Image
    image_description = ET.SubElement(
        image,
        "ImageDescription",
    )

    # Add Channels to ImageDescription
    image_description = add_channel_info(
        image_description,
        colors=colors,
        bit_depth=bit_depth,
        x_pix=x_pix,
        y_pix=y_pix,
        num_frames=num_frames,
    )
    # Add Dimensions to ImageDescription
    image_description = add_dimension_info(
        image_description,
        x_pix=x_pix,
        y_pix=y_pix,
        pixel_size=pixel_size,
        num_frames=num_frames,
        z_size=z_size,
        num_tiles=num_tiles,
    )

    return parent_node if parent_node is not None else element


# Create a mock LIF file metadata
def create_lif_xml_metadata(
    lif_file_path: Path,
    datasets: list[dict],
):
    """
    Helper function to construct an XML metadata file for a CLEM dataset collected
    in the LIF file format with the desired values and fields.

    SOME NOTES:
    ==========================================================================
    Overview, Series, and TileScan are Elements that will be indented to the same
    level.

    Overview: Contains Elements with tiled scans of the sample
    Series: Is a single dataset of a ROI on the sample
    TileScan: Contains Elements with datasets of various ROIs on the sample
    """

    # Create outer shell of the LIF XML metadata
    root = ET.Element("LMSDataContainerHeader")

    # Add Element describing LIF file to Root
    lif_file_element = ET.SubElement(
        root,
        "Element",
        {
            "Name": lif_file_path.name,
            "UniqueID": str(uuid.uuid4()),
        },
    )
    # Add Data for LIF file to LIF file Element
    lif_file_data = ET.SubElement(
        lif_file_element,
        "Data",
    )
    # Add Experiment Element to Data
    ET.SubElement(
        lif_file_data,
        "Experiment",
        {"Path": str(lif_file_path)},
    )
    # Add placeholder Attributes and Memory to LIF file Element
    for node_name in ("Attributes", "Memory"):
        ET.SubElement(
            lif_file_element,
            node_name,
        )

    # Add Children to LIF file Element
    lif_file_children = ET.SubElement(
        lif_file_element,
        "Children",
    )

    # Unpack dataset and add the desired child datasets
    for dataset in datasets:
        data_type: str = dataset["dataset_type"]
        data_name: str = dataset["dataset_name"]
        data_list: list[dict] = dataset["datasets"]
        # Add datasets to LIF file metadata according to data type
        if data_type in ("overview", "tile_scan"):
            # Create Element and attach it to LIF file Children
            dataset_element = ET.SubElement(
                lif_file_children, "Element", {"Name": data_name}
            )
            # Attach placeholder Data and Memory nodes to Overview
            for node_name in ("Data", "Memory"):
                ET.SubElement(dataset_element, node_name)
            # Add Children node to Element
            dataset_children = ET.SubElement(
                dataset_element,
                "Children",
            )
            # Add datasets to Children node
            for data in data_list:
                dataset_children = add_dataset_element(
                    dataset_children,
                    **data,
                )
        elif data_type == "series":
            # Add dataset directly to LIF file Children
            for data in data_list:
                lif_file_children = add_dataset_element(lif_file_children, **data)

    return root


# Create a mock TIFF file metadata
def create_tiff_xml_metadata(
    series_name: str,
    colors: list[str],
    bit_depth: int,
    x_pix: int,
    y_pix: int,
    pixel_size: float,
    num_frames: int,
    z_size: float,
    num_tiles: int,
    tile_offset: float,
    collection_mode: Literal["linear", "grid", ""],
):
    """
    Helper function to construct an XML metadata file for a CLEM dataset collected
    in the TIFF file format with the desired values and fields.

    The layout of the TIFF file format metadata, and the currently used parts is
    as follows:

    <LMSDataContainerHeader ...>
        <Element Name="Series Name" ...>
            <Data>
                <Image>
                    <Attachment Name="TileScanInfo">
                        <Tile ... />
                        ...
                    </Attachment>
                    <ImageDescription>
                        <Channels>
                            <ChannelDescription ... />
                            ...
                        </Channels>
                        <Dimensions>
                            <DimensionDescription ... />
                            ...
                        </Dimensions>
                    </ImageDescription>
                    ...
            <Memory />
            <Children />
        </Element>
    </LMSDataContainerHeader>
    """
    # Add Root with its atttributes
    root = ET.Element(
        "LMSDataContainerHeader",
    )

    # Add dataset Element to Root
    root = add_dataset_element(
        root,
        series_name=series_name,
        colors=colors,
        bit_depth=bit_depth,
        x_pix=x_pix,
        y_pix=y_pix,
        pixel_size=pixel_size,
        num_frames=num_frames,
        z_size=z_size,
        num_tiles=num_tiles,
        tile_offset=tile_offset,
        collection_mode=collection_mode,
    )
    return root
