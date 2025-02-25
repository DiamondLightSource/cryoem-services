from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from xml.etree.ElementTree import Element

import numpy as np
import pytest
from pydantic_core import ValidationError
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.wrappers.clem_process_raw_tiffs import (
    TIFFToStackParameters,
    TIFFToStackWrapper,
    convert_tiff_to_stack,
    process_tiff_files,
)
from tests.test_utils.clem import create_xml_metadata

# Common settings
visit_name = "test_visit"
raw_folder = "images"
processed_folder = "processed"
area_name = "test_area"
series_name = "Test Series"

# TIFF file settings
colors = [
    "gray",
    "green",
    "red",
]
num_channels = len(colors)
num_z = 5


# Create fixtures for use in subsequent tests
@pytest.fixture
def raw_dir(tmp_path):
    raw_dir = tmp_path / visit_name / raw_folder
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


@pytest.fixture
def tiff_list(raw_dir: Path):
    # Generate list of file names
    tiff_folder = raw_dir / area_name
    tiff_folder.mkdir(parents=True, exist_ok=True)
    tiff_list = [
        tiff_folder / f"{series_name}--Z{str(z).zfill(2)}--C{str(c).zfill(2)}.tif"
        for z in range(num_z)
        for c in range(num_channels)
    ]
    # Create files
    for tiff_file in tiff_list:
        tiff_file.touch(exist_ok=True)

    return tiff_list


@pytest.fixture
def metadata(raw_dir: Path):
    # Create parent directory for metadata file
    metadata_folder = raw_dir / area_name / "Metadata"
    metadata_folder.mkdir(parents=True, exist_ok=True)
    # Create metadata file
    metadata = metadata_folder / f"{series_name}.xlif"
    metadata.touch(exist_ok=True)
    return metadata


@pytest.fixture
def raw_xml_metadata():
    xml_metadata = create_xml_metadata(
        series_names=series_name,
        colors=colors,
        num_z=num_z,
    )
    return xml_metadata


@pytest.fixture
def processed_dir(raw_dir: Path):
    processed_dir = raw_dir.parent / processed_folder
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def dummy_result(
    save_dir: Path,
    series_name_short: str,
    series_name_long: str,
    parent_tiffs: list[Path],
    color: str,
    num_channels: int,
):
    """
    Helper function to generate the expected results for the TIFF file
    processing workflow
    """
    return {
        "image_stack": str(save_dir / f"{color}.tiff"),
        "metadata": str(save_dir / "metadata" / f"{series_name_short}.xml"),
        "series_name": series_name_long,
        "channel": color,
        "number_of_members": num_channels,
        "parent_tiffs": str([str(f) for f in parent_tiffs]),
    }


@patch("cryoemservices.wrappers.clem_process_raw_tiffs.Image")
@patch("cryoemservices.wrappers.clem_process_raw_tiffs.parse")
def test_process_tiff_files(
    mock_parse,
    mock_image,
    tiff_list: list[Path],
    metadata: Path,
    processed_dir: Path,
    raw_xml_metadata: Element,
):

    # Build short and long series names
    series_name_short = series_name
    series_name_long = f"{area_name}--{series_name.replace(' ', '_')}"

    # Construct save directory
    series_dir = processed_dir / area_name / series_name

    # Mock results of the 'parse()' function
    mock_parse.return_value.getroot.return_value = raw_xml_metadata

    # Mock the result of 'Image.open()'
    mock_image.open.return_value = np.random.randint(0, 255, (2048, 2048)).astype(
        "uint16"
    )

    # Run the function
    results = process_tiff_files(
        tiff_list=tiff_list,
        metadata_file=metadata,
        series_name_short=series_name_short,
        series_name_long=series_name_long,
        save_dir=series_dir,
    )

    # Construct the expected results
    dummy_results = [
        dummy_result(
            save_dir=series_dir,
            series_name_short=series_name_short.replace(" ", "_"),
            series_name_long=series_name_long,
            parent_tiffs=[
                f for f in tiff_list if f"--C{str(c).zfill(2)}.tif" in str(f)
            ],
            color=color,
            num_channels=len(colors),
        )
        for c, color in enumerate(colors)
    ]

    # Assert that the results match
    for r, result in enumerate(results):
        assert result == dummy_results[r]


@patch("cryoemservices.wrappers.clem_process_raw_tiffs.process_tiff_files")
def test_convert_tiff_to_stack(
    mock_process_tiffs,
    tiff_list: list[Path],
    metadata: Path,
    processed_dir: Path,
):
    # Build short and long series names
    series_name_short = series_name
    series_name_long = f"{area_name}--{series_name.replace(' ', '_')}"

    # Construct save directory
    series_dir = processed_dir / area_name / series_name.replace(" ", "_")

    # Mock out the result of the TIFF processing function
    mock_process_tiffs.return_value = [
        dummy_result(
            save_dir=series_dir,
            series_name_short=series_name_short.replace(" ", "_"),
            series_name_long=series_name_long,
            parent_tiffs=[
                f for f in tiff_list if f"--C{str(c).zfill(2)}.tif" in str(f)
            ],
            color=color,
            num_channels=len(colors),
        )
        for c, color in enumerate(colors)
    ]

    # Run the functino with the mocked objects
    results = convert_tiff_to_stack(
        tiff_list=tiff_list,
        root_folder=raw_folder,
        metadata_file=metadata,
    )

    # Check that it was called with the correct parameters
    mock_process_tiffs.assert_called_once_with(
        tiff_list=tiff_list,
        metadata_file=metadata,
        series_name_short=series_name_short,
        series_name_long=series_name_long,
        save_dir=series_dir,
    )
    assert results


tiff_to_stack_params_matrix = (
    # Use TIFF list? | Stringify file path? |
    (True, False),
    (True, True),
    (False, True),
)


@pytest.mark.parametrize("test_params", tiff_to_stack_params_matrix)
def test_tiff_to_stack_parameters(
    test_params: tuple[bool, bool],
    tiff_list: list[Path | str],
    metadata: Path,
    raw_folder=raw_folder,
):

    # Unpack test params
    use_tiff_list, stringify = test_params

    tiff_list = [str(file) if stringify else file for file in tiff_list]
    tiff_file = "null" if use_tiff_list else tiff_list[0]

    params = {
        "tiff_list": (tiff_list if use_tiff_list else "null"),
        "tiff_file": tiff_file,
        "root_folder": raw_folder,
        "metadata": (str(metadata) if stringify else metadata),
    }
    validated_params = TIFFToStackParameters(**params)

    # Validate parameters that are used in the wrapper
    for file in validated_params.tiff_list:
        assert isinstance(file, Path)
    assert validated_params.root_folder == raw_folder
    assert isinstance(validated_params.metadata, Path)


tiff_to_stack_params_failure_matrix = (
    # Use TIFF list? | Use TIFF file? | Garbled string
    (True, True, ""),
    (False, False, ""),
    (True, False, "[asdflkajsdlfkj]"),
    (True, False, "[1, 2, 3, 4]"),
)


@pytest.mark.parametrize("test_params", tiff_to_stack_params_failure_matrix)
def test_tiff_to_stack_parameters_fail(
    test_params: tuple[bool, bool, str],
    tiff_list: str | list[Path],
    metadata: Path,
    raw_folder=raw_folder,
):

    # Unpack test params
    use_tiff_list, use_tiff_file, garbled_string = test_params

    tiff_file = tiff_list[0] if use_tiff_file else "null"
    tiff_list = tiff_list if use_tiff_list else "null"

    params = {
        "tiff_list": (garbled_string if garbled_string else tiff_list),
        "tiff_file": tiff_file,
        "root_folder": raw_folder,
        "metadata": metadata,
    }
    with pytest.raises(ValidationError):
        TIFFToStackParameters(**params)


# Set up a mock transport object
@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport
    mocker.spy(transport, "send")  # Observe what happens to the 'send' call
    return transport


# Patches are matched to input parameters on a last in, first out basis
@patch("workflows.recipe.wrapper.RecipeWrapper.send_to")
@patch("cryoemservices.wrappers.clem_process_raw_tiffs.convert_tiff_to_stack")
def test_tiff_to_stack_wrapper(
    mock_tiff_to_stack,
    mock_send_to,
    offline_transport,  # 'offline_transport' fixture defined above
    tiff_list: list[Path],
    metadata: Path,
    processed_dir: Path,
):
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
    series_dir = processed_dir / area_name / series_name
    processed_metadata = series_dir / "metadata" / f"{series_name}.xml"
    outputs = [
        {
            "image_stack": str(series_dir / f"{color}.tiff"),
            "metadata": str(processed_metadata),
            "series_name": series_name,
            "channel": color,
            "number_of_members": len(colors),
            "parents_tiffs": str([str(file) for file in tiff_list]),
        }
        for c, color in enumerate(colors)
    ]
    mock_tiff_to_stack.return_value = outputs

    # Set up a recipe wrapper with the defined message
    recipe_wrapper = RecipeWrapper(message=message, transport=offline_transport)

    # Manually start up the function wrapper
    tiff_to_stack_wrapper = TIFFToStackWrapper()
    tiff_to_stack_wrapper.set_recipe_wrapper(recipe_wrapper)
    return_code = tiff_to_stack_wrapper.run()

    # Start checking the calls that take place when running the function
    mock_tiff_to_stack.assert_called_once_with(
        tiff_list=tiff_list,
        root_folder=raw_folder,
        metadata_file=metadata,
    )

    # Check that all the results set up are sent out at the end of the function
    for output in outputs:
        # Generate the dictionary to be sent out
        murfey_params = {
            "register": "clem.register_tiff_preprocessing_result",
            "result": output,
        }
        # Check that the messag is sent out correctly
        mock_send_to.assert_any_call("murfey_feedback", murfey_params)

    # Check that the wrapper ran through to completion
    assert return_code
