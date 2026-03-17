from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture
from readlif.reader import LifFile, LifImage
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.util.metadata.clem import (
    find_image_elements,
)
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
colors = [
    "gray",
    "green",
    "red",
    "blue",
]
num_channels = len(colors)
z_size = 0.00000040
tile_offset = 0.0001024

scene_values = (
    # x pixels | y pixels | Pixel size | No. frames | No. tiles
    (256, 256, 0.0000004, 5, 6),
    (256, 256, 0.0000004, 5, 1),
    (5120, 5120, 0.00000002, 1, 6),
    (5120, 5120, 0.00000002, 1, 1),
)
num_scenes = len(scene_values)


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
                    "x_pix": x_pixels,
                    "y_pix": y_pixels,
                    "pixel_size": pixel_size,
                    "num_frames": num_frames,
                    "z_size": z_size,
                    "num_tiles": num_tiles,
                    "tile_offset": tile_offset,
                    "collection_mode": "linear",
                }
                for n, (
                    x_pixels,
                    y_pixels,
                    pixel_size,
                    num_frames,
                    num_tiles,
                ) in enumerate(scene_values)
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
    raw_xml_metadata: ET.Element,
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
    x_pixels, y_pixels, pixel_size, num_frames, num_tiles = scene_values[scene_num]

    # Calculate the image's extent
    num_cols = math.ceil(math.sqrt(num_tiles))
    num_rows = math.ceil(num_tiles / num_cols)
    extent = [
        # [x_min, x_max, y_min, y_max] in real space
        tile_offset / 2,
        tile_offset / 2 + (tile_offset * (num_cols - 1)) + (x_pixels * pixel_size),
        tile_offset / 2,
        tile_offset / 2 + (tile_offset * (num_rows - 1)) + (y_pixels * pixel_size),
    ]

    # After calculating the extent, update num_pixels, pixel_size, and resolution
    x_pixels_new = x_pixels
    y_pixels_new = y_pixels
    pixel_size_new = pixel_size

    # For tiled images or images larger than 4096 x 4096 pixels, they will be
    # forcibly resized to fit within a box of size 4096 x 4096 pixels
    if num_tiles > 1 or x_pixels * y_pixels > 4096**2:
        stitched_height = extent[3] - extent[2]
        stitched_width = extent[1] - extent[0]
        pixel_size_new = max(stitched_height / 4096, stitched_width / 4096)
        x_pixels_new = int(stitched_width / pixel_size_new)
        y_pixels_new = int(stitched_height / pixel_size_new)

    return {
        "series_name": f"{lif_file.stem}--{series_name}",
        "number_of_members": num_channels,
        "is_stack": num_frames > 1,
        "is_montage": num_tiles > 1,
        "output_files": {
            color: str(processed_dir / lif_file.stem / series_name / f"{color}.tiff")
            for color in colors
        },
        "thumbnails": {
            color: str(
                processed_dir
                / lif_file.stem
                / series_name
                / ".thumbnails"
                / f"{color}.png"
            )
            for color in colors
        },
        "thumbnail_size": (512, 512),
        "metadata": str(
            processed_dir
            / lif_file.stem
            / series_name
            / "metadata"
            / f"{series_name}.xml"
        ),
        "parent_lif": str(lif_file),
        "pixels_x": x_pixels_new,
        "pixels_y": y_pixels_new,
        "units": "m",
        "pixel_size": pixel_size_new,
        "resolution": 1 / (pixel_size_new or 1),
        "extent": extent,
    }


@pytest.mark.parametrize(
    "test_params",
    (
        [
            n,
        ]
        for n in range(num_scenes)
    ),
)
def test_process_lif_subimage(
    mocker: MockerFixture,
    lif_file: Path,
    raw_xml_metadata: ET.Element,
    processed_dir: Path,
    test_params: tuple[int],
):
    # Pick a single scene from the LIF file to analyse
    (scene_num,) = test_params
    x_pixels, y_pixels, _, _, _ = scene_values[scene_num]
    metadata = list(find_image_elements(raw_xml_metadata).values())[scene_num]

    # Mock the LifFile object and assign the necessary return values
    mock_lif_file = mocker.patch("cryoemservices.util.image_processing.LifFile")
    mock_lif_image = MagicMock(spec=LifImage)
    mock_lif_image.get_frame.return_value = np.random.randint(
        0, 256, (y_pixels, x_pixels), dtype="uint16"
    )
    mock_lif_image.bit_depth = [16 for c in range(num_channels)]
    mock_lif_file.return_value.get_image.return_value = mock_lif_image

    # Run the function
    result = process_lif_subimage(
        file=lif_file,
        scene_num=scene_num,
        metadata=metadata,
        root_save_dir=processed_dir,
        num_procs=1,
    )

    # Verify against expected results
    expected_result = create_dummy_result(
        lif_file=lif_file,
        series_name=f"{series_name}_{scene_num}",
        processed_dir=processed_dir,
        scene_num=scene_num,
    )

    # Order of list of dictionaries should match exactly
    for key, value in expected_result.items():
        # Permit some leeway for float values
        if key == "extent":
            for c, coord in enumerate(value):
                assert math.isclose(coord, result[key][c])
        elif key in ("pixels_x",):
            assert math.isclose(value, result[key], abs_tol=2)
        elif key in ("pixels_y",):
            assert math.isclose(value, result[key], abs_tol=2)
        elif key == "pixel_size":
            assert math.isclose(value, result[key], abs_tol=1e-9)
        elif key == "resolution":
            assert math.isclose(value, result[key], abs_tol=1e-9)
        # Check that image stacks were created
        elif key == "output_files":
            for file in result[key].values():
                assert Path(file).exists()
        # Match everything else exactly
        else:
            assert value == result[key]


def test_process_lif_file(
    mocker: MockerFixture,
    lif_file: Path,
    raw_dir: Path,
    raw_xml_metadata: ET.Element,
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
    mocker.patch(
        "cryoemservices.wrappers.clem_process_raw_lifs.LifFile",
        return_value=mock_lif_file,
    )
    mock_lif_file.get_iter_image.return_value = [
        f"scene_{i}" for i in range(num_scenes)
    ]

    # Mock out XML metadata extracted from LIF file
    mocker.patch(
        "cryoemservices.wrappers.clem_process_raw_lifs.get_lif_xml_metadata",
        return_value=raw_xml_metadata,
    )

    # Mock out the sub-image processing function to return results iteratively
    mock_process_lif_subimage = mocker.patch(
        "cryoemservices.wrappers.clem_process_raw_lifs.process_lif_subimage"
    )
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
def test_lif_to_stack_wrapper(
    mocker: MockerFixture,
    offline_transport,
    lif_file: Path,
    processed_dir: Path,
):
    # Patch out functions
    mock_process_lif_file = mocker.patch(
        "cryoemservices.wrappers.clem_process_raw_lifs.process_lif_file"
    )
    mock_send_to = mocker.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")

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
    results = [
        create_dummy_result(
            lif_file=lif_file,
            series_name=series_name,
            processed_dir=processed_dir,
            scene_num=scene_num,
        )
        for scene_num in range(num_scenes)
    ]
    mock_process_lif_file.return_value = results

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
    for result in results:
        # Check that the call to 'images' was sent out
        for color in result["output_files"].keys():
            mock_send_to.assert_any_call(
                "images",
                {
                    "image_command": "tiff_to_apng",
                    "input_file": result["output_files"][color],
                    "output_file": result["thumbnails"][color],
                    "target_size": result["thumbnail_size"],
                    "color": color,
                },
            )

        # Check that the message was sent to 'murfey_feedback' correctly
        mock_send_to.assert_any_call(
            "murfey_feedback",
            {
                "register": "clem.register_preprocessing_result",
                "result": result,
            },
        )

    assert mock_send_to.call_count == (
        len(results)
        * (
            len(colors)  # 'images' calls
            + 1  # 'murfey_feedback' call
        )
    )

    # Check that the wrapper ran through to completion
    assert return_code
