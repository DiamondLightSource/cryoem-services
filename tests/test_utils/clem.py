"""
Recurring helper functions used when generating the fixtures needed to test the
CLEM workflow.
"""

from __future__ import annotations

import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

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
                peak["centre"] = (x + (f * x_shift), y + (f * y_shift))
                peak["offset"] += f * c_off
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


def create_xml_metadata(
    series_names: str | list[str],
    colors: list[str],
    num_z: int,
    lif_file: Path | None = None,
):
    """
    Fixture to create a bare bones XML metadata object for use in the tests below
    """

    def create_element(series_name: str, colors: list[str]):
        """
        Helper function to create a single XML element
        """

        def create_color_channel(order: int, color: str):
            """
            Formats a single 'Channel' item to be inserted into the element
            """
            return f'          <ChannelDescription DataType="0" ChannelTag="0" Resolution="16" NameOfMeasuredQuantity="" Min="0.000000e+000" Max="6.553500e+004" Unit="" LUTName="{color.title()}" IsLUTInverted="0" BytesInc="{int(order * 713031680)}" BitInc="0" />'

        # Bare bones example of a single XML element
        element = [
            f'<Element Name="{series_name}" Visibility="1" CopyOption="1" UniqueID="{str(uuid.uuid4())}">',
            "  <Data>",
            '    <Image TextDescription="">',
            "      <ImageDescription>",
            "        <Channels>",
            # Individual colour channels go here
            "        </Channels>",
            "        <Dimensions>",
            '          <DimensionDescription DimID="1" NumberOfElements="2048" Origin="0.000000e+000" Length="2.661100e-004" Unit="m" BitInc="0" BytesInc="2" />',
            '          <DimensionDescription DimID="2" NumberOfElements="2048" Origin="0.000000e+000" Length="2.661100e-004" Unit="m" BitInc="0" BytesInc="4096" />',
            f'          <DimensionDescription DimID="3" NumberOfElements="{num_z}" Origin="6.999377e-003" Length="-{to_scientific_notation(float((num_z - 1) * 0.00000003))}" Unit="m" BitInc="0" BytesInc="8388608" />',
            "        </Dimensions>",
            "      </ImageDescription>",
            f'      <TimeStampList NumberOfTimeStamps="{num_z * num_channels}">{" ".join([get_hexadecimal_timestamp(time.time_ns()) for t in range(num_z * num_channels)])} </TimeStampList>',
            "    </Image>",
            "  </Data>",
            "  <Children />",
            "</Element>",
        ]
        # Insert channels in-place
        for c, color in enumerate(colors):
            # Find the new index after any previous channel insertions
            channel_index = element.index("        </Channels>")
            element.insert(channel_index, create_color_channel(c, color))
        return element

    series_names = [series_names] if isinstance(series_names, str) else series_names
    num_channels = len(colors)

    # If a LIF file is provided, net elements in additional metadata
    if lif_file:
        # External shell surrounding XML metadata elements for each scene in the LIF file
        xml_metadata = [
            '<LMSDataContainerHeader Version="2">',
            f'  <Element Name="{lif_file.name}" Visibility="1" CopyOption="1" UniqueID="{str(uuid.uuid4())}">',
            "    <Children>",
            # Elements (corresponding to individual image substacks) go here
            "    </Children>",
            "  </Element>",
            "</LMSDataContainerHeader>",
        ]

        for series_name in series_names:
            # Find new index to insert next element at after previous insertion

            element_index = xml_metadata.index("    </Children>")
            element = create_element(series_name, colors)
            element = [
                f"{(' ' * 6)}{line}" for line in element
            ]  # Indent as appropriate
            xml_metadata[element_index:element_index] = element
    # Otherwise, the element is standalone
    else:
        xml_metadata = create_element(series_names[0], colors)

    return ET.fromstring("\n".join(xml_metadata))
