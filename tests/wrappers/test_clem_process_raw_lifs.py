from __future__ import annotations

from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock
from xml.etree.ElementTree import Element

import pytest
from readlif.reader import LifFile
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.wrappers.clem_process_raw_lifs import (
    LIFToStackWrapper,
    convert_lif_to_stack,
    process_lif_substack,
)

visit_name = "test_visit"
root_folder = "images"
processed_folder = "processed"


# Create fixtures to represent raw data directory and contents
@pytest.fixture
def raw_dir(tmp_path: Path):
    raw_dir = tmp_path / visit_name / root_folder
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True)
    return raw_dir


@pytest.fixture
def lif_file(raw_dir: Path):
    lif_file = raw_dir / "test_file.lif"
    if not lif_file.exists():
        lif_file.touch()
    return lif_file


@mock.patch("multiprocessing.Pool")
@mock.patch("cryoemservices.wrappers.clem_process_raw_lifs.get_image_elements")
@mock.patch("cryoemservices.wrappers.clem_process_raw_lifs.get_lif_xml_metadata")
@mock.patch("cryoemservices.wrappers.clem_process_raw_lifs.LifFile")
def test_convert_lif_to_stack(
    mock_load_lif_file,
    mock_get_lif_xml_metadata,
    mock_get_image_elements,
    mock_pool_class,
    lif_file: Path,
    raw_dir: Path,
):
    """
    Tests the LIF-to-stack conversion function
    """

    # LIF file properties to mock
    num_scenes = 8
    num_channels = 3

    # Reconstruct the path to the processed directory
    path_parts = list(raw_dir.parts)
    path_parts[0] = "" if path_parts[0] == "/" else path_parts[0]
    try:
        root_index = path_parts.index(root_folder)
        path_parts[root_index] = processed_folder
        processed_folder
    except ValueError:
        raise ValueError(f"{root_folder} not found in file path")
    processed_dir = Path("/".join(path_parts[: root_index + 1]))

    # Mock out LifFile object and its dependents
    mock_lif_file = MagicMock(spec=LifFile)
    mock_load_lif_file.return_value = mock_lif_file
    mock_lif_file.get_iter_image.return_value = [
        f"scene_{i}" for i in range(num_scenes)
    ]

    # Mock out XML metadata extracted from LIF file and its dependents
    mock_xml_root = MagicMock(spec=Element)
    mock_get_lif_xml_metadata.return_value = mock_xml_root
    metadata = MagicMock(spec=Element)
    mock_metadata_list = [metadata for i in range(num_scenes)]
    mock_get_image_elements.return_value = mock_metadata_list

    # Mock out the multiprocessing function and its outputs
    dummy_result = {  # Dummy for the image stack of a single chanel
        "image_stack": "test_img.tiff",
        "metadata": "test_metadata.xml",
        "series_name": "test_series",
        "channel": "gray",
        "number_of_members": num_channels,
        "parent_lif": str(lif_file),
    }
    mock_pool_instance = MagicMock()
    mock_pool_instance.starmap.return_value = [
        [dummy_result for c in range(num_channels)] for i in range(num_scenes)
    ]
    mock_pool_class.return_value.__enter__.return_value = mock_pool_instance

    # Run the function
    results = convert_lif_to_stack(
        lif_file,
        root_folder=root_folder,
        number_of_processes=1,
    )

    # Check that arguments were fed into the multiprocessing function the correct number of times
    mock_pool_instance.starmap.assert_called_once_with(
        process_lif_substack,
        [
            [
                lif_file,
                n,
                mock_metadata_list[n],
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
