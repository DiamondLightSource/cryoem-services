from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture
from readlif.reader import LifFile, LifImage
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.clem_process_raw_lifs import (
    ProcessRawLIFsService,
    get_lif_xml_metadata,
    process_lif_file,
    process_lif_subimage,
)
from cryoemservices.util.metadata.clem import (
    find_image_elements,
)
from cryoemservices.util.models import MockRW
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
        "series_name": f"{lif_file.stem}/{series_name}",
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
    mock_lif_file = mocker.patch("cryoemservices.util.image_processing.shared.LifFile")
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
        "cryoemservices.services.clem_process_raw_lifs.LifFile",
        return_value=mock_lif_file,
    )
    mock_lif_file.get_iter_image.return_value = [
        f"scene_{i}" for i in range(num_scenes)
    ]

    # Mock out XML metadata extracted from LIF file
    mocker.patch(
        "cryoemservices.services.clem_process_raw_lifs.get_lif_xml_metadata",
        return_value=raw_xml_metadata,
    )

    # Store dummy result in mock Future objects
    mock_futures = []
    for scene_num in range(num_scenes):
        mock_future = MagicMock()
        mock_future.result.return_value = create_dummy_result(
            lif_file=lif_file,
            series_name=series_name,
            processed_dir=processed_dir,
            scene_num=scene_num,
        )
        mock_futures.append(mock_future)

    # Mock out the ProcessPoolExecutor used to run 'process_lif_subimage'
    mock_executor = mocker.patch(
        "cryoemservices.services.clem_process_raw_lifs.ProcessPoolExecutor"
    )
    mock_pool = mock_executor.return_value.__enter__.return_value
    mock_pool.submit.side_effect = mock_futures

    # Run the function
    results = process_lif_file(
        lif_file,
        root_folder=raw_folder,
        number_of_processes=5,
    )

    # Check that nested list of results was collapsed correctly
    assert mock_pool.submit.call_count == num_scenes
    assert len(results) == num_scenes


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    mocker.spy(transport, "nack")
    return transport


lif_to_stack_params_matrix = (
    # Use Recipe Wrapper?
    (True,),
    (False,),
)


@pytest.mark.parametrize("test_params", lif_to_stack_params_matrix)
def test_lif_to_stack_service(
    mocker: MockerFixture,
    test_params: tuple[bool],
    lif_file: Path,
    raw_dir: Path,
    offline_transport: OfflineTransport,
):
    """
    Sends a test message to the LIF processing service, which should execute the
    function with the parameters present in the message, then send messages with
    the expected outputs back to Murfey.
    """

    # Unpack test params
    (use_recwrap,) = test_params

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    lif_to_stack_test_message = {
        "lif_file": str(lif_file),
        "root_folder": raw_dir.stem,
    }

    # Set up the expected mock values
    # mock_convert.return_value = processing_results
    dummy_results = [
        {
            # Include only keys used for tests
            "series_name": f"dummy_{i}",
            "output_files": {
                "gray": "dummy",
                "red": "dummy",
                "green": "dummy",
                "blue": "dummy",
            },
            "thumbnails": {
                "gray": "dummy",
                "red": "dummy",
                "green": "dummy",
                "blue": "dummy",
            },
            "thumbnail_size": (512, 512),
        }
        for i in range(4)
    ]
    mock_convert = mocker.patch(
        "cryoemservices.services.clem_process_raw_lifs.process_lif_file",
        return_value=dummy_results,
    )

    # Set up and run the service
    service = ProcessRawLIFsService(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    if use_recwrap:
        recwrap = MockRW(offline_transport)
        recwrap.recipe_step = {"parameters": lif_to_stack_test_message}
        service.call_process_raw_lifs(
            recwrap,
            header=header,
            message=None,
        )
    else:
        service.call_process_raw_lifs(
            None,
            header=header,
            message=lif_to_stack_test_message,
        )

    # Check that the expected calls are made
    mock_convert.assert_called_with(
        file=lif_file,
        root_folder=raw_dir.stem,
        number_of_processes=16,
        logger=service.log,
    )
    for result in dummy_results:
        for color in cast(dict[str, Any], result["output_files"]).keys():
            offline_transport.send.assert_any_call(
                "images",
                {
                    "image_command": "tiff_to_apng",
                    "input_file": cast(
                        dict[str, dict[str, str]], result["output_files"]
                    )[color],
                    "output_file": cast(
                        dict[str, dict[str, str]], result["output_files"]
                    )[color],
                    "target_size": result["thumbnail_size"],
                    "color": color,
                },
            )
        offline_transport.send.assert_any_call(
            "murfey_feedback",
            {
                "register": "clem.register_preprocessing_result",
                "result": result,
            },
        )
    assert offline_transport.send.call_count == (
        len(dummy_results)
        * (
            4  # One for each colour
            + 1  # One for 'murfey_feedback'
        )
    )


def test_lif_to_stack_bad_messsage(
    offline_transport: OfflineTransport,
):
    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    bad_message = "This is a bad message"

    # Set up and run the service
    service = ProcessRawLIFsService(
        environment={"queue": ""},
        transport=offline_transport,
    )
    service.initializing()
    service.call_process_raw_lifs(
        None,
        header=header,
        message=bad_message,
    )

    # Check that message was nacked with the expected parameters
    offline_transport.nack.assert_called_once_with(header, requeue=False)

    # Check that the message wasn't erronerously sent
    offline_transport.send.assert_not_called()


# Introduce invalid parameters field-by-field
lif_to_stack_params_bad_validation_matrix = (
    # LIF file | Root folder | Num procs
    (False, True, True),
    (True, False, True),
    (True, True, False),
)


@pytest.mark.parametrize("test_params", lif_to_stack_params_bad_validation_matrix)
def test_lif_to_stack_service_validation_failed(
    test_params: tuple[bool, bool, bool],
    lif_file: Path,
    raw_dir: Path,
    offline_transport: OfflineTransport,
):
    """
    Sends a test message to the image alignment and merging service, which should
    excecute the function using the parameters present in the message, then send
    messages with the expected outputs back to Murfey.
    """

    # Unpack test params
    (
        valid_lif_file,
        valid_root_folder,
        valid_num_procs,
    ) = test_params

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    lif_file_value: Any = str(lif_file) if valid_lif_file else 123456789
    root_folder: Any = raw_dir.stem if valid_root_folder else 123456789
    num_procs: Any = 16 if valid_num_procs else "This is a string"
    lif_to_stack_test_message = {
        "lif_file": lif_file_value,
        "root_folder": root_folder,
        "num_procs": num_procs,
    }

    # Set up and run the service
    service = ProcessRawLIFsService(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    service.call_process_raw_lifs(
        None,
        header=header,
        message=lif_to_stack_test_message,
    )

    # Check that the message was nacked with the expected parameters
    offline_transport.nack.assert_called_once_with(header, requeue=False)

    # Check that the message wasn't erronerously sent
    offline_transport.send.assert_not_called()


def test_lif_to_stack_service_processing_failed(
    mocker: MockerFixture,
    lif_file: Path,
    raw_dir: Path,
    offline_transport: OfflineTransport,
):
    """
    Sends a test message to the LIF processing service, which should execute the
    function with the parameters present in the message, then send messages with
    the expected outputs back to Murfey.
    """

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    lif_to_stack_test_message = {
        "lif_file": str(lif_file),
        "root_folder": raw_dir.stem,
    }

    # Set up the expected mock values
    mocker.patch(
        "cryoemservices.services.clem_process_raw_lifs.process_lif_file",
        return_value={},
    )

    # Set up and run the service
    service = ProcessRawLIFsService(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    service.call_process_raw_lifs(
        None,
        header=header,
        message=lif_to_stack_test_message,
    )

    # Check that message was nacked with expected parameters
    offline_transport.nack.assert_called_once_with(header, requeue=True)

    # Check that the message wasn't erronerously sent
    offline_transport.send.assert_not_called()
