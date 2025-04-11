from __future__ import annotations

import sys
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import images


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "ack")
    mocker.spy(transport, "nack")
    return transport


def test_plugins_exist():
    """Check the correct images plugins exist"""
    # Set up the mock service
    service = images.Images(environment={"queue": ""})
    service.transport = mock.Mock()
    service.start()

    # Check the expected images plugins are present
    assert len(service.image_functions.keys()) == 6
    assert service.image_functions.get("mrc_central_slice", "")
    assert service.image_functions.get("mrc_to_apng", "")
    assert service.image_functions.get("mrc_to_jpeg", "")
    assert service.image_functions.get("picked_particles", "")
    assert service.image_functions.get("picked_particles_3d_apng", "")
    assert service.image_functions.get("picked_particles_3d_central_slice", "")


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.images_plugins.picked_particles")
def test_images_call_rw(mock_picker_image):
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
    mock_picker_image.assert_called_once()
    mock_recipe_wrapper.transport.ack.assert_called()


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.images_plugins.picked_particles")
def test_images_call_simple_message(mock_picker_image, offline_transport):
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
    service.transport = offline_transport
    service.start()

    # Send a message to the service
    service.image_call(None, header=header, message=images_test_message)

    # Check the correct calls were made
    mock_picker_image.assert_called_once()
    offline_transport.ack.assert_called()


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.images_plugins.picked_particles")
def test_images_call_invalid_message(mock_picker_image, offline_transport):
    """
    Send a test message to the images service for a call that fails
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }

    # Set up the mock service
    service = images.Images(environment={"queue": ""})
    service.transport = offline_transport
    service.start()

    # Send a message to the service
    service.image_call(None, header=header, message="string message")

    # Check the correct calls were made
    mock_picker_image.assert_not_called()
    offline_transport.nack.assert_called()


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.images_plugins.picked_particles")
def test_images_failed_call(mock_picker_image):
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
    mock_picker_image.assert_called_once()
    mock_recipe_wrapper.transport.nack.assert_called()


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.images_plugins.picked_particles")
def test_images_file_not_found(mock_picker_image):
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
    mock_picker_image.assert_called_once()
    mock_recipe_wrapper.transport.nack.assert_called()


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_images_call_unknown_function():
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
