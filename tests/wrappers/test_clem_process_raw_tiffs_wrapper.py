from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic_core import ValidationError
from pytest_mock import MockerFixture
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.util.clem_array_functions import (
    get_percentiles,
    stitch_image_frames,
)
from cryoemservices.util.clem_metadata import get_dimension_info, get_tile_scan_info
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
num_datasets = 2  # For iterating metadata values with
series_names = [f"Position {n}" for n in range(num_datasets)]
colors = [
    "gray",
    "green",
    "red",
]
num_channels = len(colors)
num_pixels = 256
pixel_size = 0.0000004
num_frames = 5
z_size = 0.0000004
num_tiles = [6 if i % 2 else 1 for i in range(num_datasets)]
tile_offset = 0.0001024


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
    for n in range(num_datasets):
        series_name = series_names[n]
        tiff_list: list[Path] = []
        for t in range(num_tiles[n]):
            for z in range(num_frames):
                for c in range(num_channels):
                    file_stem = series_name
                    if num_tiles[n] > 1:
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
        metadata = metadata_folder / f"{series_names[n]}.xlif"
        metadata.touch(exist_ok=True)
        metadata_list.append(metadata)
    return metadata_list


@pytest.fixture
def processed_dir(raw_dir: Path):
    processed_dir = raw_dir.parent / processed_folder
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


percentiles_test_matrix = (
    (0, 100),
    (1, 99),
    (5, 95),
    (10, 90),
)


@pytest.mark.parametrize("test_params", percentiles_test_matrix)
def test_get_percentiles(
    mocker: MockerFixture,
    test_params: tuple[float, float],
    tiff_lists: list[list[Path]],
):
    # Unpack test params
    p_lo, p_hi = test_params

    # Construct return values for the mocked PIL Image
    mock_image = mocker.patch(
        "cryoemservices.util.clem_array_functions.PILImage", autospec=True
    )
    arr = np.arange(256).reshape(16, 16).astype("uint8")
    mock_image.open.return_value = arr

    v_min, v_max = get_percentiles(
        tiff_file=tiff_lists[0][0],
        percentiles=(p_lo, p_hi),
    )
    expected_min, expected_max = np.percentile(arr, (p_lo, p_hi))
    assert v_min is not None and v_max is not None
    assert math.isclose(v_min, expected_min)
    assert math.isclose(v_max, expected_max)


def test_get_percentiles_fails(
    tiff_lists: list[list[Path]],
):
    # It should return (None, None) if the TIFF file cannot be read
    assert get_percentiles(tiff_file=tiff_lists[0][0]) == (None, None)


def test_stitch_image_frames(
    mocker: MockerFixture,
    tiff_lists: list[list[Path]],
):
    dataset_num = 1  # Dataset with tiles
    series_name = series_names[dataset_num]
    tiff_list = tiff_lists[dataset_num]
    metadata = create_tiff_xml_metadata(
        series_name=series_name,
        colors=colors,
        bit_depth=16,
        x_pix=num_pixels,
        y_pix=num_pixels,
        pixel_size=pixel_size,
        num_frames=num_frames,
        z_size=z_size,
        num_tiles=num_tiles[dataset_num],
        tile_offset=tile_offset,
        collection_mode="grid",
    )

    # Verify that we've loaded the metadata with multiple tiles
    assert len(metadata.findall(".//Tile")) > 1
    tile_scan_info = get_tile_scan_info(metadata)
    assert len(tile_scan_info) > 1

    # Calculate the extent that will be covered by the test metadata
    x_min, x_max, y_min, y_max = 1e10, 0.0, 1e10, 0.0
    for tile_scan in tile_scan_info.values():
        x0 = tile_scan["pos_x"] - tile_offset / 2
        x1 = tile_scan["pos_x"] + tile_offset / 2
        y0 = tile_scan["pos_y"] - tile_offset / 2
        y1 = tile_scan["pos_y"] + tile_offset / 2
        x_min = x0 if x0 < x_min else x_min
        x_max = x1 if x1 > x_max else x_max
        y_min = y0 if y0 < y_min else y_min
        y_max = y1 if y1 > y_max else y_max

    # Calculate the expected shape of the final image
    width = x_max - x_min
    height = y_max - y_min
    if width / height > 1:
        x_pixels = 2400
        y_pixels = int(x_pixels / width * height)
    else:
        y_pixels = 2400
        x_pixels = int(y_pixels * width / height)

    # Patch the PIL Image object to return an array of ones
    mock_image = mocker.patch(
        "cryoemservices.util.clem_array_functions.PILImage", autospec=True
    )
    arr = np.ones((num_pixels, num_pixels), dtype="uint8")
    mock_image.open.return_value = arr

    # Run the function
    frame = stitch_image_frames(
        tiff_list=tiff_list,
        tile_scan_info=tile_scan_info,
        image_width=num_pixels,
        image_height=num_pixels,
        extent=(x_min, x_max, y_min, y_max),
        dpi=400,
        contrast_limits=(0, 255),
    )

    # Check that size and contents are as expected
    assert (frame == 1).all()
    np.testing.assert_allclose(frame.shape, (y_pixels, x_pixels), atol=2)


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

    # Calculate the image's extent
    num_cols = math.ceil(math.sqrt(num_tiles[test_num]))
    num_rows = math.ceil(num_tiles[test_num] / num_cols)
    extent = [
        # [x_min, x_max, y_min, y_max] in real space
        tile_offset / 2,
        tile_offset / 2 + (tile_offset * (num_cols - 1)) + (num_pixels * pixel_size),
        tile_offset / 2,
        tile_offset / 2 + (tile_offset * (num_rows - 1)) + (num_pixels * pixel_size),
    ]

    # After calculating the extent, update num_pixels, pixel_size, and resolution
    x_pix = num_pixels
    y_pix = num_pixels
    actual_pixel_size = pixel_size

    # Tiles are set up such that it will be square or wider than it is tall.
    # The Figure used for stitching has a default size of 6 inches x 6 inches.
    # The final image will thus have a fixed height of (6 * dpi)
    # 'dpi' currently set to 400
    if num_tiles[test_num] > 1:
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
        "series_name": f"{area_name}--{series_name}",
        "number_of_members": num_channels,
        "is_stack": num_frames > 1,
        "is_montage": num_tiles[test_num] > 1,
        "output_files": {
            color: str(processed_dir / area_name / series_name / f"{color}.tiff")
            for color in colors
        },
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
        "pixels_x": x_pix,
        "pixels_y": y_pix,
        "units": "m",
        "pixel_size": actual_pixel_size,
        "resolution": 1 / actual_pixel_size,
        "extent": extent,
    }


tiff_dataset_to_test = [(n,) for n in range(num_datasets)]


@pytest.mark.parametrize("test_params", tiff_dataset_to_test)
def test_process_tiff_files(
    mocker: MockerFixture,
    processed_dir: Path,
    tiff_lists,
    raw_metadata_files,
    test_params: tuple[int],
):
    # Unpack test params
    (test_num,) = test_params
    series_name = series_names[test_num]
    tiff_list = tiff_lists[test_num]
    raw_metadata = raw_metadata_files[test_num]

    # Create XML metadata for test dataset
    xml_metadata = create_tiff_xml_metadata(
        series_name=series_name,
        colors=colors,
        bit_depth=16,
        x_pix=num_pixels,
        y_pix=num_pixels,
        pixel_size=pixel_size,
        num_frames=num_frames,
        z_size=z_size,
        num_tiles=num_tiles[test_num],
        tile_offset=tile_offset,
        collection_mode="linear",
    )
    mock_parse = mocker.patch("cryoemservices.wrappers.clem_process_raw_tiffs.parse")
    mock_parse.return_value.getroot.return_value = xml_metadata

    # Extract metadata dictionaries
    dims = get_dimension_info(xml_metadata)
    tile_scan_info = get_tile_scan_info(xml_metadata)

    # Get width and height for a single frame
    w = float(dims["x"]["length"])
    h = float(dims["y"]["length"])

    # Mock the result of 'Image.open()'
    mock_image = mocker.patch("cryoemservices.wrappers.clem_process_raw_tiffs.Image")
    mock_image.open.return_value = np.random.randint(
        0, 255, (num_pixels, num_pixels), dtype="uint16"
    )

    # Calculate the image's extent
    num_cols = math.ceil(math.sqrt(num_tiles[test_num]))
    num_rows = math.ceil(num_tiles[test_num] / num_cols)
    extent = [
        # [x_min, x_max, y_min, y_max] in real space
        tile_offset / 2,
        tile_offset / 2 + (tile_offset * (num_cols - 1)) + (num_pixels * pixel_size),
        tile_offset / 2,
        tile_offset / 2 + (tile_offset * (num_rows - 1)) + (num_pixels * pixel_size),
    ]

    # After calculating the extent, update num_pixels, pixel_size, and resolution
    x_pix = num_pixels
    y_pix = num_pixels
    actual_pixel_size = pixel_size

    # Calculate the shape of the final image for use when mocking the subprocess call
    if num_tiles[test_num] > 1:
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

        # Mock the subprocess calls used for contrast measurment and image stitching
        starmap_results: list[Any] = []
        starmap_args: list[Any] = []
        for c in range(num_channels):
            tiff_color_subset = [
                file
                for file in tiff_list
                if f"C{str(c).zfill(2)}" in file.stem.split("--")
            ]

            # Add results and args for percentile measurment
            starmap_results.append(
                [
                    (0, 255)
                    for z in range(num_frames)
                    for t in range(num_tiles[test_num])
                ]
            )
            starmap_args.append(
                [
                    get_percentiles,
                    [
                        (
                            # LIF file-based parameters
                            None,
                            None,
                            None,
                            None,
                            None,
                            # TIFF file-based parameters
                            file,
                            # Common parameters
                            (0.5, 99.5),
                        )
                        for file in tiff_color_subset
                    ],
                ]
            )
            # Add results and args for image stitching
            starmap_results.append(
                [np.ones((y_pix, x_pix), dtype="uint8") for f in range(num_frames)]
            )
            starmap_args.append(
                [
                    stitch_image_frames,
                    [
                        (
                            # LIF file-based parameters
                            None,
                            None,
                            None,
                            None,
                            # TIFF file-based parameters
                            [
                                file
                                for file in tiff_color_subset
                                if f"Z{str(z).zfill(2)}" in file.stem.split("--")
                            ],
                            # Common parameters
                            tile_scan_info,
                            w,
                            h,
                            extent,
                            400,
                            (0, 255),
                        )
                        for z in range(num_frames)
                    ],
                ]
            )
        pool_mocks = []
        for result in starmap_results:
            mock_pool = MagicMock()
            mock_pool.__enter__.return_value = mock_pool
            mock_pool.starmap.return_value = result
            pool_mocks.append(mock_pool)
        mock_pool_constructor = mocker.patch(
            "cryoemservices.wrappers.clem_process_raw_tiffs.Pool", autospec=True
        )
        mock_pool_constructor.side_effect = pool_mocks

    # Run the function
    result = process_tiff_files(
        tiff_list=tiff_list,
        root_folder=raw_folder,
        metadata_file=raw_metadata,
        number_of_processes=5,
    )
    if num_tiles[test_num] > 1:
        for p, pool in enumerate(pool_mocks):
            pool.starmap.assert_called_once_with(*starmap_args[p])
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


process_raw_tiffs_params_matrix = (
    # Use 'tiff_list'? Build from 'tiff_file' if False | Stringify file path?
    (True, False),
    # Check that list of strings is converted correctly
    (True, True),
    (False, True),
)


@pytest.mark.parametrize("test_params", process_raw_tiffs_params_matrix)
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
    }
    validated_params = ProcessRawTIFFsParameters(**params)

    # Check that parameters were validated correctly
    assert validated_params.tiff_list is not None
    for file in validated_params.tiff_list:
        assert isinstance(file, Path)
    assert validated_params.root_folder == raw_folder
    assert isinstance(validated_params.metadata, Path)


process_raw_tiffs_params_failure_matrix = (
    # Use 'tiff_list' | Use 'tiff_file' | Garbled string
    # tiff_list and tiff_file cannot both be populated or absent
    (True, True, ""),
    (False, False, ""),
    # Cannot evaluate stringified list
    (True, False, "[asdflkajsdlfkj]"),
    (True, False, "[1, 2, 3, 4]"),
)


@pytest.mark.parametrize("test_params", process_raw_tiffs_params_failure_matrix)
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
@pytest.mark.parametrize("test_params", tiff_dataset_to_test)
def test_process_raw_tiffs_wrapper(
    mocker: MockerFixture,
    test_params: tuple[int],
    offline_transport,  # 'offline_transport' fixture defined above
    tiff_lists: list[list[Path]],
    raw_metadata_files: list[Path],
):
    # Unpack test params and load relevant test datasets
    (dataset_num,) = test_params
    tiff_list = tiff_lists[dataset_num]
    metadata = raw_metadata_files[dataset_num]
    series_name = series_names[dataset_num]

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
    output = {"series_name": series_name, "dummy": "dummy"}
    mock_process_raw_tiffs.return_value = output

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
        number_of_processes=20,
    )

    # Check that all the results set up are sent out at the end of the function
    # Generate the dictionary to be sent out
    murfey_params = {
        "register": "clem.register_preprocessing_result",
        "result": output,
    }
    # Check that the messag is sent out correctly
    mock_send_to.assert_called_once_with("murfey_feedback", murfey_params)

    # Check that the wrapper ran through to completion
    assert return_code
