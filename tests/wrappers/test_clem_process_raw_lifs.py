from __future__ import annotations

from unittest import mock

import pytest
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.wrappers.clem_process_raw_lifs import LIFToStackWrapper


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
