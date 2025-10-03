from __future__ import annotations

import math
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock
from xml.etree.ElementTree import Element

import numpy as np
import pytest
from readlif.reader import LifFile, LifImage
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.util.clem_metadata import find_image_elements
from cryoemservices.wrappers.clem_process_raw_lifs import (
    ProcessRawLIFsWrapper,
    get_lif_xml_metadata,
    process_lif_file,
    process_lif_subimage,
)
from tests.test_utils.clem import create_lif_xml_metadata

# Directory structure
visit_name = "test_visit"
raw_folder = "images"
processed_folder = "processed"

# Default LIF file metadata to mock
series_name = "Position"
num_scenes = 2
scene_num = 0
colors = [
    "gray",
    "green",
    "red",
    "blue",
]
num_channels = len(colors)
num_pixels = 512
pixel_size = 0.0000005
num_frames = 5
z_size = 0.00000040
num_tiles = [6 if i % 2 else 1 for i in range(num_scenes)]
tile_offset = 0.00025


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


@pytest.fixture
def raw_xml_metadata(lif_file: Path):
    # Set up the information to be used
    datasets = [
        {
            # Create tile scan datasets
            "dataset_type": "tile_scan",
            "dataset_name": "TileScan 1",
            "datasets": [
                {
                    "series_name": f"{series_name} {n}",
                    "colors": colors,
                    "bit_depth": 16,
                    "x_pix": num_pixels,
                    "y_pix": num_pixels,
                    "pixel_size": pixel_size,
                    "num_frames": num_frames,
                    "z_size": z_size,
                    "num_tiles": num_tiles[n],
                    "tile_offset": tile_offset,
                    "collection_mode": "linear",
                }
                for n in range(num_scenes)
            ],
        },
    ]

    xml_metadata = create_lif_xml_metadata(
        lif_file_path=lif_file,
        datasets=datasets,
    )
    return xml_metadata


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


def create_dummy_result(
    lif_file: Path,
    series_name: str,
    processed_dir: Path,
    scene_num: int,
):
    """
    Helper function to populate the dummy result with the needed variables
    """
    series_name = series_name.replace(" ", "_")

    # Calculate the image's extent
    num_cols = math.ceil(math.sqrt(num_tiles[scene_num]))
    num_rows = math.ceil(num_tiles[scene_num] / num_cols)
    extent = [
        # [x_min, x_max, y_min, y_max] in real space
        tile_offset,
        (tile_offset * num_cols) + (num_pixels * pixel_size),
        tile_offset,
        (tile_offset * num_rows) + (num_pixels * pixel_size),
    ]

    # After calculating the extent, update num_pixels, pixel_size, and resolution
    x_pix = num_pixels
    y_pix = num_pixels
    actual_pixel_size = pixel_size

    # Tiles are set up such that it will be square or wider than it is tall.
    # The Figure used for stitching has a default size of 6 inches x 6 inches.
    # The final image will thus have a fixed height of (6 * dpi)
    # 'dpi' currently set to 400
    if num_tiles[scene_num] > 1:
        stitched_height = extent[3] - extent[2]
        stitched_width = extent[1] - extent[0]
        x_to_y_ratio = stitched_width / stitched_height
        # if x:y > 4:3, set width to 3200, otherwise set height to 2400
        if x_to_y_ratio > 1:
            x_pix = 2400
            actual_pixel_size = stitched_width / x_pix
            y_pix = int(stitched_height / actual_pixel_size)
        else:
            y_pix = 2400
            actual_pixel_size = stitched_height / y_pix
            x_pix = int(stitched_width / actual_pixel_size)

    return {
        "series_name": f"{lif_file.stem}--{series_name}",
        "number_of_members": num_channels,
        "is_stack": num_frames > 1,
        "is_montage": num_tiles[scene_num] > 1,
        "output_files": {
            color: str(processed_dir / lif_file.stem / series_name / f"{color}.tiff")
            for color in colors
        },
        "metadata": str(
            processed_dir
            / lif_file.stem
            / series_name
            / "metadata"
            / f"{series_name}.xml"
        ),
        "parent_lif": str(lif_file),
        "pixels_x": x_pix,
        "pixels_y": y_pix,
        "units": "m",
        "pixel_size": actual_pixel_size,
        "resolution": 1 / actual_pixel_size,
        "extent": extent,
    }


scene_num_to_test = [
    [
        n,
    ]
    for n in range(num_scenes)
]


@pytest.mark.parametrize("test_params", scene_num_to_test)
@mock.patch("cryoemservices.wrappers.clem_process_raw_lifs.LifFile")
def test_process_lif_subimage(
    mock_lif_file,
    lif_file: Path,
    raw_xml_metadata: Element,
    processed_dir: Path,
    test_params: tuple[int],
):

    # Pick a single scene from the LIF file to analyse
    (scene_num,) = test_params
    metadata = list(find_image_elements(raw_xml_metadata).values())[scene_num]

    # Mock out the LifImage object
    mock_lif_image = MagicMock(spec=LifImage)

    # Create a NumPy array for the 'get_frame' attribute
    mock_lif_image.get_frame.return_value = np.random.randint(
        0, 256, (num_pixels, num_pixels), dtype="uint16"
    )

    # Assign a return value to the 'bit_depth' attribute
    mock_lif_image.bit_depth = [16 for c in range(num_channels)]

    # Assign LifImage mock object LifFile function
    mock_lif_file.return_value.get_image.return_value = mock_lif_image

    # Run the function
    result = process_lif_subimage(
        file=lif_file,
        scene_num=scene_num,
        metadata=metadata,
        root_save_dir=processed_dir,
        num_procs=5,
    )
    assert result  # Verify that function completed successfully

    # Verify against expected results
    expected_result = create_dummy_result(
        lif_file=lif_file,
        series_name=f"{series_name}_{scene_num}",
        processed_dir=processed_dir,
        scene_num=scene_num,
    )

    # Order of list of dictionaries should match exactly
    for key, value in expected_result.items():
        if key == "extent":
            for c, coord in enumerate(value):
                assert math.isclose(coord, result[key][c])
        elif key in ("pixels_y",):
            assert math.isclose(value, result[key], abs_tol=2)
        elif key in ("pixels_x",):
            assert math.isclose(value, result[key], abs_tol=2)
        elif key == "pixel_size":
            assert math.isclose(value, result[key], abs_tol=1e-9)
        elif key == "resolution":
            assert math.isclose(value, result[key], abs_tol=1e-9)
        else:
            assert value == result[key]


@mock.patch("cryoemservices.wrappers.clem_process_raw_lifs.process_lif_subimage")
@mock.patch("cryoemservices.wrappers.clem_process_raw_lifs.get_lif_xml_metadata")
@mock.patch("cryoemservices.wrappers.clem_process_raw_lifs.LifFile")
def test_process_lif_file(
    mock_load_lif_file,
    mock_get_lif_xml_metadata,
    mock_process_lif_subimage,
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

    # Mock out the sub-image processing function to return results iteratively
    mock_process_lif_subimage.side_effect = [
        create_dummy_result(
            lif_file=lif_file,
            series_name=series_name,
            processed_dir=processed_dir,
            scene_num=scene_num,
        )
        for scene_num in range(num_scenes)
    ]

    # Run the function
    results = process_lif_file(
        lif_file,
        root_folder=raw_folder,
        number_of_processes=5,
    )

    # Check that nested list of results was collapsed correctly
    assert mock_process_lif_subimage.call_count == num_scenes
    assert len(results) == num_scenes


# Set up a mock transport object
@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")  # Observe what happens to the 'send' call
    return transport


# Patches are matched to variables on last in, first out basis
@mock.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")  # = mock_send_to
@mock.patch("cryoemservices.wrappers.clem_process_raw_lifs.process_lif_file")
def test_lif_to_stack_wrapper(
    mock_process_lif_file,
    mock_send_to,
    offline_transport,
    lif_file: Path,
    processed_dir: Path,
):
    # Set the number of simultaneous processes to run
    num_procs = 20

    # Construct a dictionary to pass to the wrapper
    message = {
        "recipe": {
            "start": [[1, []]],
            "1": {
                "job_parameters": {
                    "lif_file": str(lif_file),
                    "root_folder": raw_folder,
                },
                "parameters": {},
            },
        },
        "recipe-pointer": 1,
        "environment": {"ID": "envID"},
    }

    # Construct the expected output result
    outputs = [
        create_dummy_result(
            lif_file=lif_file,
            series_name=series_name,
            processed_dir=processed_dir,
            scene_num=scene_num,
        )
        for scene_num in range(num_scenes)
    ]
    mock_process_lif_file.return_value = outputs

    # Set up a recipe wrapper with the defined recipe
    recipe_wrapper = RecipeWrapper(message=message, transport=offline_transport)

    # Manually start up the function wrapper
    lif_to_stack_wrapper = ProcessRawLIFsWrapper(recipe_wrapper)
    return_code = lif_to_stack_wrapper.run()

    # Start checking the calls that take place when running the function
    # Check that the LIF-to-stack wrapper is called correctly
    mock_process_lif_file.assert_called_once_with(
        file=lif_file,
        root_folder=raw_folder,
        number_of_processes=num_procs,
    )

    # Check that all the results set up are sent out at the end of the function
    for output in outputs:
        # Generate the dictionary to be sent out
        murfey_params = {
            "register": "clem.register_preprocessing_result",
            "result": output,
        }
        # Check that the message is sent out correctly
        mock_send_to.assert_any_call("murfey_feedback", murfey_params)

    # Check that the wrapper ran through to completion
    assert return_code
