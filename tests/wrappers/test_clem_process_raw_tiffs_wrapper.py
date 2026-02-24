from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from pydantic_core import ValidationError
from pytest_mock import MockerFixture
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.wrappers.clem_process_raw_tiffs import (
    ProcessRawTIFFsParameters,
    ProcessRawTIFFsWrapper,
    process_tiff_files,
)
from tests.test_utils.clem import create_tiff_xml_metadata

# Common settings
visit_name = "test_visit"
raw_folder = "images"
processed_folder = "processed"
area_name = "test_area"

# TIFF file settings
num_procs = 4
colors = [
    "gray",
    "green",
    "red",
]
num_channels = len(colors)
z_size = 0.0000004
tile_offset = 0.0001024
dataset_values = (
    # x pixels | y pixels | Pixel size | No. frames | No. tiles
    (256, 256, 0.0000004, 5, 6),
    (256, 256, 0.0000004, 5, 1),
    (5120, 5120, 0.00000002, 1, 6),
    (5120, 5120, 0.00000002, 1, 1),
)
num_datasets = len(dataset_values)


# Create fixtures for use in subsequent tests
@pytest.fixture
def raw_dir(tmp_path):
    raw_dir = tmp_path / visit_name / raw_folder
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


@pytest.fixture
def tiff_lists(raw_dir: Path):
    # Generate list of file names
    tiff_folder = raw_dir / area_name
    tiff_folder.mkdir(parents=True, exist_ok=True)

    tiff_lists: list[list[Path]] = []
    for n, (_, _, _, num_frames, num_tiles) in enumerate(dataset_values):
        series_name = f"Position {n}"
        tiff_list: list[Path] = []
        for t in range(num_tiles):
            for z in range(num_frames):
                for c in range(num_channels):
                    file_stem = series_name
                    if num_tiles > 1:
                        file_stem += f"--Stage{str(t).zfill(2)}"
                    if num_frames > 1:
                        file_stem += f"--Z{str(z).zfill(2)}"
                    if num_channels > 1:
                        file_stem += f"--C{str(c).zfill(2)}"
                    tiff_file = tiff_folder / f"{file_stem}.tif"
                    tiff_file.touch(exist_ok=True)
                    tiff_list.append(tiff_file)
        tiff_lists.append(tiff_list)
    return tiff_lists


@pytest.fixture
def raw_metadata_files(raw_dir: Path):
    metadata_list: list[Path] = []
    for n in range(num_datasets):
        # Create parent directory for metadata file
        metadata_folder = raw_dir / area_name / "Metadata"
        metadata_folder.mkdir(parents=True, exist_ok=True)
        # Create metadata file
        metadata = metadata_folder / f"Position {n}.xlif"
        metadata.touch(exist_ok=True)
        metadata_list.append(metadata)
    return metadata_list


@pytest.fixture
def processed_dir(raw_dir: Path):
    processed_dir = raw_dir.parent / processed_folder
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def create_dummy_result(
    tiff_list: list[Path],
    series_name: str,
    processed_dir: Path,
    test_num: int,
):
    """
    Helper function to generate the expected results for the TIFF file
    processing workflow
    """

    series_name = series_name.replace(" ", "_")
    x_pixels, y_pixels, pixel_size, num_frames, num_tiles = dataset_values[test_num]

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
        "series_name": f"{area_name}--{series_name}",
        "number_of_members": num_channels,
        "is_stack": num_frames > 1,
        "is_montage": num_tiles > 1,
        "output_files": {
            color: str(processed_dir / area_name / series_name / f"{color}.tiff")
            for color in colors
        },
        "thumbnails": {
            color: str(
                processed_dir / area_name / series_name / ".thumbnails" / f"{color}.png"
            )
            for color in colors
        },
        "thumbnail_size": (512, 512),
        "parent_tiffs": (
            {
                color: sorted(
                    [
                        str(file)
                        for file in tiff_list
                        if f"--C{str(c).zfill(2)}" in file.stem
                    ]
                )[0:1]
                for c, color in enumerate(colors)
            }
            if num_channels > 1
            else {
                color: sorted([str(file) for file in tiff_list])[0:1]
                for color in colors
            }
        ),
        "metadata": str(
            processed_dir / area_name / series_name / "metadata" / f"{series_name}.xml"
        ),
        "pixels_x": x_pixels_new,
        "pixels_y": y_pixels_new,
        "units": "m",
        "pixel_size": pixel_size_new,
        "resolution": 1 / (pixel_size_new or 1),
        "extent": extent,
    }


@pytest.mark.parametrize("test_params", [(n,) for n in range(num_datasets)])
def test_process_tiff_files(
    mocker: MockerFixture,
    processed_dir: Path,
    tiff_lists,
    raw_metadata_files,
    test_params: tuple[int],
):
    # Unpack test params
    (test_num,) = test_params
    x_pixels, y_pixels, pixel_size, num_frames, num_tiles = dataset_values[test_num]

    series_name = f"Position {test_num}"
    tiff_list = tiff_lists[test_num]
    raw_metadata = raw_metadata_files[test_num]

    # Create XML metadata for test dataset
    xml_metadata = create_tiff_xml_metadata(
        series_name=series_name,
        colors=colors,
        bit_depth=16,
        x_pix=x_pixels,
        y_pix=y_pixels,
        pixel_size=pixel_size,
        num_frames=num_frames,
        z_size=z_size,
        num_tiles=num_tiles,
        tile_offset=tile_offset,
        collection_mode="linear",
    )
    mock_parse = mocker.patch("cryoemservices.wrappers.clem_process_raw_tiffs.parse")
    mock_parse.return_value.getroot.return_value = xml_metadata

    # Mock the result of 'cv2.imread'
    mocker.patch(
        "cryoemservices.util.clem_array_functions.cv2.imread",
        return_value=np.random.randint(0, 65536, (y_pixels, x_pixels), dtype="uint16"),
    )

    # Run the function
    result = process_tiff_files(
        tiff_list=tiff_list,
        root_folder=raw_folder,
        metadata_file=raw_metadata,
        number_of_processes=5,
    )
    assert result  # Verify that function completed successfully

    # Construct the expected results
    expected_result = create_dummy_result(
        tiff_list=tiff_list,
        series_name=series_name,
        processed_dir=processed_dir,
        test_num=test_num,
    )

    # Order of list of dictionaries should match exactly
    for key, value in expected_result.items():
        # Allow deviation for float values
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
        # Check that image stacks were created
        elif key == "output_files":
            for file in result[key].values():
                assert Path(file).exists()
        # Assert everything else exactly
        else:
            assert value == result[key]


@pytest.mark.parametrize(
    "test_params",
    (
        # Use 'tiff_list'? Build from 'tiff_file' if False | Stringify file path?
        (True, False),
        # Check that list of strings is converted correctly
        (True, True),
        (False, True),
    ),
)
def test_process_raw_tiffs_parameters(
    test_params: tuple[bool, bool],
    tiff_lists: list[list[Path]],
    raw_metadata_files: list[Path],
    raw_folder=raw_folder,
):
    # Unpack test params
    use_tiff_list, stringify = test_params
    metadata = raw_metadata_files[0]

    # Modify 'tiff_list' and 'tiff_file' for the test
    tiff_list = [str(file) if stringify else file for file in tiff_lists[0]]
    tiff_file = "null" if use_tiff_list else tiff_list[0]

    # Construct dictionary and validate it with the Pydantic model
    params = {
        "tiff_list": (tiff_list if use_tiff_list else "null"),
        "tiff_file": tiff_file,
        "root_folder": raw_folder,
        "metadata": (str(metadata) if stringify else metadata),
        "num_procs": num_procs,
    }
    validated_params = ProcessRawTIFFsParameters(**params)

    # Check that parameters were validated correctly
    assert validated_params.tiff_list is not None
    for file in validated_params.tiff_list:
        assert isinstance(file, Path)
    assert validated_params.root_folder == raw_folder
    assert isinstance(validated_params.metadata, Path)


@pytest.mark.parametrize(
    "test_params",
    (
        # Use 'tiff_list' | Use 'tiff_file' | Garbled string
        # tiff_list and tiff_file cannot both be populated or absent
        (True, True, ""),
        (False, False, ""),
        # Cannot evaluate stringified list
        (True, False, "[asdflkajsdlfkj]"),
        (True, False, "[1, 2, 3, 4]"),
    ),
)
def test_process_raw_tiffs_parameters_fail(
    test_params: tuple[bool, bool, str],
    tiff_lists: list[list[Path]],
    raw_metadata_files: list[Path],
    raw_folder=raw_folder,
):
    # Unpack test params
    use_tiff_list, use_tiff_file, garbled_string = test_params
    metadata = raw_metadata_files[0]
    # Modify 'tiff_file' and 'tiff_list' accordingly
    tiff_file = tiff_lists[0][0] if use_tiff_file else "null"
    tiff_list = tiff_lists[0] if use_tiff_list else "null"

    # Construct the dictionary and validate it with the Pydantic model
    params = {
        "tiff_list": (garbled_string if garbled_string else tiff_list),
        "tiff_file": tiff_file,
        "root_folder": raw_folder,
        "metadata": metadata,
        "num_procs": num_procs,
    }
    with pytest.raises(ValidationError):
        ProcessRawTIFFsParameters(**params)


# Set up a mock transport object
@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport
    mocker.spy(transport, "send")  # Observe what happens to the 'send' call
    return transport


# Patches are matched to input parameters on a last in, first out basis
@pytest.mark.parametrize("test_params", [(n,) for n in range(num_datasets)])
def test_process_raw_tiffs_wrapper(
    mocker: MockerFixture,
    test_params: tuple[int],
    offline_transport,  # 'offline_transport' fixture defined above
    tiff_lists: list[list[Path]],
    processed_dir: Path,
    raw_metadata_files: list[Path],
):
    # Unpack test params and load relevant test datasets
    (dataset_num,) = test_params
    tiff_list = tiff_lists[dataset_num]
    metadata = raw_metadata_files[dataset_num]
    series_name = f"Position {dataset_num}"

    # Construct mock objects for use in this test
    mock_send_to = mocker.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")
    mock_process_raw_tiffs = mocker.patch(
        "cryoemservices.wrappers.clem_process_raw_tiffs.process_tiff_files"
    )

    # Construct a dictionary to pass to the wrapper
    message = {
        "recipe": {
            "start": [[1, []]],
            "1": {
                "job_parameters": {
                    "tiff_list": str([str(file) for file in tiff_list]),
                    "tiff_file": None,
                    "root_folder": raw_folder,
                    "metadata": metadata,
                    "num_procs": num_procs,
                },
                "parameters": {},
            },
        },
        "recipe-pointer": 1,
        "environment": {"ID": "envID"},
    }

    # Generate the expected output result of the TIFF processing function
    # series_dir = processed_dir / area_name / series_name.replace(" ", "_")
    # processed_metadata = series_dir / "metadata" / f"{series_name.replace(" ", "_")}.xml"
    result = create_dummy_result(
        tiff_list=tiff_list,
        series_name=series_name,
        processed_dir=processed_dir,
        test_num=dataset_num,
    )
    mock_process_raw_tiffs.return_value = result

    # Set up a recipe wrapper with the defined message
    recipe_wrapper = RecipeWrapper(message=message, transport=offline_transport)

    # Manually start up the function wrapper
    process_raw_tiffs_wrapper = ProcessRawTIFFsWrapper(recipe_wrapper)
    return_code = process_raw_tiffs_wrapper.run()

    # Start checking the calls that take place when running the function
    mock_process_raw_tiffs.assert_called_once_with(
        tiff_list=tiff_list,
        root_folder=raw_folder,
        metadata_file=metadata,
        number_of_processes=num_procs,
    )

    # Check that 'images' was called for each colour
    for color in cast(dict[str, str], result["output_files"]).keys():
        mock_send_to.assert_any_call(
            "images",
            {
                "image_command": "tiff_to_apng",
                "input_file": cast(dict[str, str], result["output_files"])[color],
                "output_file": cast(dict[str, str], result["thumbnails"])[color],
                "target_size": result["thumbnail_size"],
                "color": color,
            },
        )
    # Check that the messag is sent out correctly
    mock_send_to.assert_any_call(
        "murfey_feedback",
        {
            "register": "clem.register_preprocessing_result",
            "result": result,
        },
    )
    # Check that 'send_to' was called the expected number of times
    assert mock_send_to.call_count == 1 + len(colors)

    # Check that the wrapper ran through to completion
    assert return_code
