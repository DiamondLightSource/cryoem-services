from __future__ import annotations

import sys
from unittest import mock

import pytest

from cryoemservices.services import images


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.images_plugins.picked_particles")
def test_images_call(mock_picker_image, tmp_path):
    """
    Send a test message to the images service
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    images_test_message = {"image_command": "picked_particles"}

    # Set up the mock service
    service = images.Images(environment={"queue": ""})
    service.transport = mock.Mock()
    service.start()

    mock_recipe_wrapper = mock.Mock()
    mock_recipe_wrapper.recipe_step = {"parameters": {"param1": "value1"}}

    # Send a message to the service
    service.image_call(mock_recipe_wrapper, header=header, message=images_test_message)

    # Check the correct calls were made
    mock_picker_image.assert_called_with(
        images.PluginInterface(mock_recipe_wrapper, mock.ANY, images_test_message)
    )
    mock_recipe_wrapper.transport.ack.assert_called()


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.images_plugins.picked_particles")
def test_images_failed_call(mock_picker_image, tmp_path):
    """
    Send a test message to the images service for a call that fails
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    images_test_message = {}

    # Set up the mock service
    service = images.Images(environment={"queue": ""})
    service.transport = mock.Mock()
    service.start()

    mock_recipe_wrapper = mock.Mock()
    mock_recipe_wrapper.recipe_step = {
        "parameters": {"image_command": "picked_particles"}
    }

    mock_picker_image.return_value = False

    # Send a message to the service
    service.image_call(mock_recipe_wrapper, header=header, message=images_test_message)

    # Check the correct calls were made
    mock_picker_image.assert_called_with(
        images.PluginInterface(mock_recipe_wrapper, mock.ANY, images_test_message)
    )
    mock_recipe_wrapper.transport.nack.assert_called()


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.images_plugins.picked_particles")
def test_images_file_not_found(mock_picker_image, tmp_path):
    """
    Send a test message to the images service for a call that can't find a file
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    images_test_message = {}

    # Set up the mock service
    service = images.Images(environment={"queue": ""})
    service.transport = mock.Mock()
    service.start()

    mock_recipe_wrapper = mock.Mock()
    mock_recipe_wrapper.recipe_step = {
        "parameters": {"image_command": "picked_particles"}
    }

    mock_picker_image.side_effect = FileNotFoundError

    # Send a message to the service
    service.image_call(mock_recipe_wrapper, header=header, message=images_test_message)

    # Check the correct calls were made
    mock_picker_image.assert_called_with(
        images.PluginInterface(mock_recipe_wrapper, mock.ANY, images_test_message)
    )
    mock_recipe_wrapper.transport.nack.assert_called()


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_images_call_unknown_function(tmp_path):
    """
    Send a test message to the images service, for a plugin that does not exist
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    images_test_message = {"image_command": "unknown_function"}

    # Set up the mock service
    service = images.Images(environment={"queue": ""})
    service.transport = mock.Mock()
    service.start()

    mock_recipe_wrapper = mock.Mock()

    # Send a message to the service
    service.image_call(mock_recipe_wrapper, header=header, message=images_test_message)

    # Check the correct calls were made
    mock_recipe_wrapper.transport.nack.assert_called()
