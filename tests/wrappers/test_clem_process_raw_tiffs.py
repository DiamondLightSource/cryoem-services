from __future__ import annotations

from unittest import mock

import pytest
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.wrappers.clem_process_raw_tiffs import TIFFToStackWrapper


# Set up a mock transport object
@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport
    mocker.spy(transport, "send")  # Observe what happens to the 'send' call
    return transport


# Patches are matched to input parameters on a last in, first out basis
@mock.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")  # = mock_send_to
@mock.patch(
    "cryoemservices.wrappers.clem_process_raw_tiffs.convert_tiff_to_stack"
)  # = mock_tiff_to_stack
def test_tiff_to_stack_wrapper(
    mock_tiff_to_stack,
    mock_send_to,
    offline_transport,  # 'offline_transport' fixture defined above
    tmp_path,  # Pytest default working directory
):
    # Construct a dictionary to pass to the wrapper
    root_folder = "images"
    series_name = "test_series"
    raw_dir = tmp_path / root_folder
    colors = ("gray", "green", "red")  # The most common colour channels used
    num_frames = 120
    tiff_list = [
        raw_dir / f"{series_name}--Z{str(z).zfill(2)}--C{str(c).zfill(2)}.tiff"
        for z in range(num_frames)
        for c in range(len(colors))
    ]
    raw_metadata = raw_dir / "Metadata" / f"{series_name}.xlif"

    message = {
        "recipe": {
            "start": [[1, []]],
            "1": {
                "job_parameters": {
                    "tiff_list": str([str(file) for file in tiff_list]),
                    "tiff_file": None,
                    "root_folder": root_folder,
                    "metadata": raw_metadata,
                },
                "parameters": {},
            },
        },
        "recipe-pointer": 1,
        "environment": {"ID": "envID"},
    }

    # Generate the expected output result of the TIFF processing function
    series_dir = tmp_path / "processed" / series_name
    processed_metadata = series_dir / "metadata" / f"{series_name}.xml"
    outputs = [
        {
            "image_stack": str(series_dir / f"{color}.tiff"),
            "metadata": str(processed_metadata),
            "series_name": series_name,
            "channel": color,
            "number_of_members": len(colors),
            "parents_tiffs": str(
                [
                    str(
                        raw_dir
                        / f"{series_name}--Z{str(z).zfill(2)}--C{str(c).zfill(2)}.tiff"
                    )
                    for z in range(num_frames)
                ]
            ),
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
        root_folder=root_folder,
        metadata_file=raw_metadata,
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
