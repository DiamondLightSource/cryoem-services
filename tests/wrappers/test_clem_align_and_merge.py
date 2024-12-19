from __future__ import annotations

from unittest import mock

import pytest
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.wrappers.clem_align_and_merge import AlignAndMergeWrapper


# Sets up a mock transport object
@pytest.fixture
def offline_transport(mocker):  # 'mocker' is a keyword associated with unittest.mock
    transport = OfflineTransport()
    mocker.spy(transport, "send")  # Observe what happens to the 'send' call
    return transport


# Patches are matched on a last in, first out basis
@mock.patch("workflows.recipe.wrapper.RecipeWrapper.send_to")  # = mock_send_to
@mock.patch(
    "cryoemservices.wrappers.clem_align_and_merge.align_and_merge_stacks"
)  # = mock_align_and_merge
def test_align_and_merge_wrapper(
    mock_align_and_merge,
    mock_send_to,
    offline_transport,  # The fixture defined earlier
    tmp_path,  # Where pytest runs by default
):

    # Feed it a dictionary
    message = {
        "recipe": {
            "start": [[1, []]],
            "1": {
                "job_parameters": {
                    "series_name": "test_series",
                    "images": [
                        f"{tmp_path}/gray.tiff",
                        f"{tmp_path}/green.tiff",
                        f"{tmp_path}/red.tiff",
                    ],
                    "metadata": f"{tmp_path}/metadata/test_series.xml",
                    "crop_to_n_frames": 30,
                    "align_self": "enabled",
                    "flatten": "max",
                    "align_across": "enabled",
                },
                "parameters": {},
            },
        },
        "recipe-pointer": 1,
        "environment": {"ID": "envID"},
    }

    # Set a recipe wrapper with the correct dictionaries
    recipe_wraper = RecipeWrapper(message=message, transport=offline_transport)

    # Manually start up a recipe wrapper
    align_and_merge_wrapper = AlignAndMergeWrapper()
    align_and_merge_wrapper.set_recipe_wrapper(recipe_wraper)
    result = align_and_merge_wrapper.run()

    # Start checking the calls that take place within the function

    # Step 1: Validating the message via Pydantic
    # This can be skipped

    # Step 2: Check the align and merge function
    mock_align_and_merge.assert_called_once()
    mock_align_and_merge.assert_called_with(
        images=[tmp_path / "gray.tiff", tmp_path / "green.tiff", tmp_path / "red.tiff"],
        metadata=tmp_path / "metadata/test_series.xml",
        crop_to_n_frames=30,
        align_self="enabled",
        flatten="max",
        align_across="enabled",
    )

    # Step 3: Check that the wrapper sends the correct output message
    mock_send_to.assert_called_once()  # Check that it's only called once

    # Generate the dictionary we expect to get
    murfey_params = {
        "register": "clem.register_align_and_merge_result",
        "result": mock.ANY,  # Checks that an object exists
    }
    mock_send_to.assert_called_with(
        "murfey_feedback",
        murfey_params,
    )

    # Check at the end that wrapper ran to completion successfully
    assert result
