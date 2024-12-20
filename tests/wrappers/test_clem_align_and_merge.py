from __future__ import annotations

from unittest import mock

import pytest
from workflows.recipe.wrapper import RecipeWrapper
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.wrappers.clem_align_and_merge import AlignAndMergeWrapper


# Set up a mock transport object
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
    offline_transport,  # Fixture defined above
    tmp_path,  # Pytest default working directory
):

    # Construct a dictionary to pass to the wrapper
    image_list = [tmp_path / f"{color}.tiff" for color in ("gray", "green", "red")]
    metadata = tmp_path / "metadata" / "test_series.xml"
    crop_to_n_frames = 30
    align_self = "enabled"
    flatten = "max"
    align_across = "enabled"

    message = {
        "recipe": {
            "start": [[1, []]],
            "1": {
                "job_parameters": {
                    "series_name": "test_series",
                    "images": [str(image) for image in image_list],
                    "metadata": str(metadata),
                    "crop_to_n_frames": crop_to_n_frames,
                    "align_self": align_self,
                    "flatten": flatten,
                    "align_across": align_across,
                },
                "parameters": {},
            },
        },
        "recipe-pointer": 1,
        "environment": {"ID": "envID"},
    }

    # Set a recipe wrapper with the correct dictionaries
    recipe_wraper = RecipeWrapper(message=message, transport=offline_transport)

    # Manually start up the function wrapper
    align_and_merge_wrapper = AlignAndMergeWrapper()
    align_and_merge_wrapper.set_recipe_wrapper(recipe_wraper)
    result = align_and_merge_wrapper.run()

    # Start checking the calls that take place within the function
    # Check the align-and-merge function is called correctly
    mock_align_and_merge.assert_called_once_with(
        images=image_list,
        metadata=metadata,
        crop_to_n_frames=crop_to_n_frames,
        align_self=align_self,
        flatten=flatten,
        align_across=align_across,
    )

    # Check that the wrapper sends the correct output message
    # Generate the dictionary to be sent out
    murfey_params = {
        "register": "clem.register_align_and_merge_result",
        "result": mock.ANY,  # Checks that an object exists
    }
    # Check that the message is sent out correctly
    mock_send_to.assert_called_once_with(
        "murfey_feedback",
        murfey_params,
    )

    # Check that the wrapper ran through to completion
    assert result
