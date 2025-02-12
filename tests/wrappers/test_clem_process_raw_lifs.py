from __future__ import annotations

import time
import uuid
import xml.etree.ElementTree as ET
from collections import namedtuple
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock
from xml.etree.ElementTree import Element

import numpy as np
import pytest
from readlif.reader import LifFile, LifImage
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.util.clem_raw_metadata import (
    get_axis_resolution,
    get_image_elements,
)
from cryoemservices.wrappers.clem_process_raw_lifs import (
    LIFToStackWrapper,
    convert_lif_to_stack,
    get_lif_xml_metadata,
    process_lif_substack,
)

# Directory structure
visit_name = "test_visit"
raw_folder = "images"
processed_folder = "processed"

# LIF file properties to mock
series_name = "test_series"
num_scenes = 8
scene_num = 0
num_z = 5
colors = [
    "gray",
    "green",
    "red",
    "blue",
]
num_channels = len(colors)


# Create fixtures to represent the directory structure and raw data
@pytest.fixture
def raw_dir(tmp_path: Path):
    raw_dir = tmp_path / visit_name / raw_folder
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True)
    return raw_dir


@pytest.fixture
def processed_dir(raw_dir: Path):
    processed_dir = raw_dir.parent / processed_folder
    if not processed_dir.exists():
        processed_dir.mkdir(parents=True)
    return processed_dir


@pytest.fixture
def lif_file(raw_dir: Path):
    lif_file = raw_dir / "test_file.lif"
    if not lif_file.exists():
        lif_file.touch()
    return lif_file


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


@pytest.fixture
def raw_xml_metadata(lif_file: Path):
    """
    Fixture to create a bare bones XML metadata object for use in the tests below
    """

    def create_element(order: int, colors: list[str]):
        def create_color_channel(order: int, color: str):
            return f'                <ChannelDescription DataType="0" ChannelTag="0" Resolution="16" NameOfMeasuredQuantity="" Min="0.000000e+000" Max="6.553500e+004" Unit="" LUTName="{color.title()}" IsLUTInverted="0" BytesInc="{int(order * 713031680)}" BitInc="0" />'

        # Bare bones example of a single scene's XML element
        element = [
            f'      <Element Name="test_series_{order}" Visibility="1" CopyOption="1" UniqueID="{str(uuid.uuid4())}">',
            "        <Data>",
            '          <Image TextDescription="">',
            "            <ImageDescription>",
            "              <Channels>",
            # Individual colour channels go here
            "              </Channels>",
            "              <Dimensions>",
            '                <DimensionDescription DimID="1" NumberOfElements="2048" Origin="0.000000e+000" Length="2.661100e-004" Unit="m" BitInc="0" BytesInc="2" />',
            '                <DimensionDescription DimID="2" NumberOfElements="2048" Origin="0.000000e+000" Length="2.661100e-004" Unit="m" BitInc="0" BytesInc="4096" />',
            f'                <DimensionDescription DimID="3" NumberOfElements="{num_z}" Origin="6.999377e-003" Length="-{to_scientific_notation(float((num_z - 1) * 0.00000003))}" Unit="m" BitInc="0" BytesInc="8388608" />',
            "              </Dimensions>",
            "            </ImageDescription>",
            f'            <TimeStampList NumberOfTimeStamps="{num_z * num_channels}">{" ".join([get_hexadecimal_timestamp(time.time_ns()) for t in range(num_z * num_channels)])} </TimeStampList>',
            "          </Image>",
            "        </Data>",
            "        <Children />",
            "      </Element>",
        ]
        # Insert channels in-place
        for c, color in enumerate(colors):
            # Find the new index after any previous channel insertions
            channel_index = element.index("              </Channels>")
            element.insert(channel_index, create_color_channel(c, color))
        return element

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

    for n in range(num_scenes):
        # Find new index to insert next element at after previous insertion
        element_index = xml_metadata.index("    </Children>")
        xml_metadata[element_index:element_index] = create_element(n, colors)

    return ET.fromstring("\n".join(xml_metadata))


def test_get_lif_xml_metadata(
    tmp_path: Path,
    raw_xml_metadata: Element,
):
    # Mock out the XML Element object extracted from the LIF file
    mock_lif_file = MagicMock(spec=LifFile)
    mock_lif_file.xml_root = raw_xml_metadata

    # Save to an arbitrary Path
    xml_file = tmp_path / "test_file.xml"
    get_lif_xml_metadata(mock_lif_file, xml_file)

    # Assert that the xml file was created
    assert xml_file.exists()


def dummy_result(
    lif_file: Path,
    series_name: str,
    color: str,
    processed_dir: Path,
):
    """
    Helper function to populate the dummy result with the needed variables
    """

    return {
        "image_stack": str(processed_dir / series_name / f"{color}.tiff"),
        "metadata": str(
            processed_dir / series_name / "metadata" / f"{series_name}.xml"
        ),
        "series_name": series_name,
        "channel": color,
        "number_of_members": num_channels,
        "parent_lif": str(lif_file),
    }


@mock.patch("cryoemservices.wrappers.clem_process_raw_lifs.LifFile")
def test_process_lif_substack(
    mock_lif_file,
    lif_file: Path,
    raw_xml_metadata: Element,
    processed_dir: Path,
):
    scene_num = 0

    metadata = get_image_elements(raw_xml_metadata)[scene_num]

    # Mock out the LifImage object generated in the function
    mock_lif_image = MagicMock(spec=LifImage)

    # Generate values for the 'dims' attribute
    Dims = namedtuple("Dims", "x y z t m")
    mock_lif_image.dims = Dims(2048, 2048, num_z, 0, 0)

    # Generate values for the 'scale' attribute
    dimensions = metadata.findall(
        "Data/Image/ImageDescription/Dimensions/DimensionDescription"
    )
    mock_lif_image.scale = [get_axis_resolution(element) for element in dimensions]

    # Create a NumPy array for the 'get_frame' attribute
    mock_lif_image.get_frame.return_value = np.random.randint(
        0, 256, (2048, 2048), dtype="uint16"
    )

    # Assign a return value to the 'bit_depth' attribute
    mock_lif_image.bit_depth = [16 for c in range(num_channels)]

    # Assign LifImage mock object LifFile function
    mock_lif_file.return_value.get_image.return_value = mock_lif_image

    # Run the function
    results = process_lif_substack(
        lif_file,
        scene_num,
        metadata,
        processed_dir,
    )
    assert results


@mock.patch("multiprocessing.Pool")
@mock.patch("cryoemservices.wrappers.clem_process_raw_lifs.get_lif_xml_metadata")
@mock.patch("cryoemservices.wrappers.clem_process_raw_lifs.LifFile")
def test_convert_lif_to_stack(
    mock_load_lif_file,
    mock_get_lif_xml_metadata,
    mock_pool_class,
    lif_file: Path,
    raw_dir: Path,
    raw_xml_metadata: Element,
):
    """
    Tests the LIF-to-stack conversion function
    """

    # Reconstruct the path to the processed directory
    path_parts = list(raw_dir.parts)
    path_parts[0] = "" if path_parts[0] == "/" else path_parts[0]
    try:
        root_index = path_parts.index(raw_folder)
        path_parts[root_index] = processed_folder
        processed_folder
    except ValueError:
        raise ValueError(f"{raw_folder} not found in file path")
    processed_dir = Path("/".join(path_parts[: root_index + 1]))

    # Mock out LifFile object and its dependents
    mock_lif_file = MagicMock(spec=LifFile)
    mock_load_lif_file.return_value = mock_lif_file
    mock_lif_file.get_iter_image.return_value = [
        f"scene_{i}" for i in range(num_scenes)
    ]

    # Mock out XML metadata extracted from LIF file
    mock_get_lif_xml_metadata.return_value = raw_xml_metadata
    metadata_list = get_image_elements(raw_xml_metadata)

    # Mock out the multiprocessing function and its outputs
    mock_pool_instance = MagicMock()
    mock_pool_instance.starmap.return_value = [
        [
            dummy_result(
                lif_file,
                f"test_series_{n}",
                color,
                processed_dir,
            )
            for color in colors
        ]
        for n in range(num_scenes)
    ]
    mock_pool_class.return_value.__enter__.return_value = mock_pool_instance

    # Run the function
    results = convert_lif_to_stack(
        lif_file,
        root_folder=raw_folder,
        number_of_processes=1,
    )

    # Check that arguments were fed into the multiprocessing function correctly
    mock_pool_instance.starmap.assert_called_once_with(
        process_lif_substack,
        [
            [
                lif_file,
                n,
                metadata_list[n],
                processed_dir,
            ]
            for n in range(num_scenes)
        ],
    )

    # Check that nested list of results was collapsed correctly
    assert len(results) == num_scenes * num_channels


# Set up a mock transport object
@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")  # Observe what happens to the 'send' call
    return transport


# Patches are matched to variables on last in, first out basis
@mock.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")  # = mock_send_to
@mock.patch(
    "cryoemservices.wrappers.clem_process_raw_lifs.convert_lif_to_stack"
)  # = mock_lif_to_stack
def test_lif_to_stack_wrapper(
    mock_lif_to_stack,
    mock_send_to,
    offline_transport,
    tmp_path,  # Pytest default working directory
):
    # Construct a dictionary to pass to the wrapper
    lif_file = tmp_path / "images" / "image.lif"
    root_folder = "images"
    num_procs = 20

    message = {
        "recipe": {
            "start": [[1, []]],
            "1": {
                "job_parameters": {
                    "lif_file": str(lif_file),
                    "root_folder": root_folder,
                },
                "parameters": {},
            },
        },
        "recipe-pointer": 1,
        "environment": {"ID": "envID"},
    }

    # The result from the mocked function is also a mock object
    # Construct the expected output result
    colors = ("gray", "green", "red")  # Typical color channels present in the LIF file
    series_name = "test_series"
    series_dir = tmp_path / "processed" / series_name
    metadata = series_dir / "metadata" / f"{series_name}.xml"

    outputs = [
        {
            "image_stack": str(series_dir / f"{color}.tiff"),
            "metadata": str(metadata),
            "series_name": series_name,
            "channel": color,
            "number_of_members": len(colors),
            "parent_lif": str(lif_file),
        }
        for color in colors
    ]
    mock_lif_to_stack.return_value = outputs

    # Set up a recipe wrapper with the defined recipe
    recipe_wrapper = RecipeWrapper(message=message, transport=offline_transport)

    # Manually start up the function wrapper
    lif_to_stack_wrapper = LIFToStackWrapper()
    lif_to_stack_wrapper.set_recipe_wrapper(recipe_wrapper)
    return_code = lif_to_stack_wrapper.run()

    # Start checking the calls that take place when running the function
    # Check that the LIF-to-stack wrapper is called correctly
    mock_lif_to_stack.assert_called_once_with(
        file=lif_file,
        root_folder=root_folder,
        number_of_processes=num_procs,
    )

    # Check that all the results set up are sent out at the end of the function
    for output in outputs:
        # Generate the dictionary to be sent out
        murfey_params = {
            "register": "clem.register_lif_preprocessing_result",
            "result": output,
        }
        # Check that the message is sent out correctly
        mock_send_to.assert_any_call("murfey_feedback", murfey_params)

    # Check that the wrapper ran through to completion
    assert return_code
