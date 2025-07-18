from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from cryoemservices.cli import resubmit_wrapper


@mock.patch("cryoemservices.cli.resubmit_wrapper.PikaTransport")
@mock.patch("cryoemservices.cli.resubmit_wrapper.RecipeWrapper")
def test_run_wrapper(mock_rw, mock_transport, tmp_path):
    """Test that failed wrappers can be restarted"""
    # Create a sample json recipe wrapper
    recipe_wrapper = tmp_path / "recipe_wrapper.json"
    recipe_data = {
        "payload": ["payload"],
        "recipe-pointer": 1,
        "recipe": {"start": [[1, []]], "1": {"parameters": {"param1": "value1"}}},
    }
    with open(recipe_wrapper, "w") as rw:
        json.dump(recipe_data, rw)
    mock_rw().payload = recipe_data["payload"]
    mock_rw().recipe_pointer = recipe_data["recipe-pointer"]

    # Create a sample config file
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as cf:
        cf.write("rabbitmq_credentials: rmq_creds\n")
        cf.write(f"recipe_directory: {tmp_path}\n")

    # Run the wrapper starter
    sys.argv = [
        "cryoemservices.resubmit_wrapper",
        "--wrapper",
        str(recipe_wrapper),
        "--config_file",
        str(config_file),
    ]
    resubmit_wrapper.run()

    # Check the calls the resubmission should have made
    mock_transport.assert_called()
    mock_transport().load_configuration_file.assert_called_with(Path("rmq_creds"))
    mock_transport().connect.assert_called()

    mock_rw.assert_called_with(message=recipe_data, transport=mock_transport())
    mock_rw()._send_to_destination.assert_called_with(1, None, ["payload"], {})
    mock_transport().disconnect.assert_called()


@mock.patch("cryoemservices.cli.resubmit_wrapper.PikaTransport")
def test_run_wrapper_no_wrapper(mock_transport, tmp_path):
    """Test that non-existant wrappers don't get processed"""
    # Create a sample config file
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as cf:
        cf.write("rabbitmq_credentials: rmq_creds\n")
        cf.write(f"recipe_directory: {tmp_path}\n")

    # Run the wrapper starter
    sys.argv = [
        "cryoemservices.resubmit_wrapper",
        "--wrapper",
        str(tmp_path / "recipe_wrapper.json"),
        "--config_file",
        str(config_file),
    ]
    with pytest.raises(FileNotFoundError):
        resubmit_wrapper.run()

    # Check nothing got called
    mock_transport.assert_not_called()


@mock.patch("cryoemservices.cli.resubmit_wrapper.PikaTransport")
def test_run_wrapper_no_config(mock_transport, tmp_path):
    """Test that wrappers don't get processed without a config file"""
    # Touch a dummy wrapper file
    recipe_wrapper = tmp_path / "recipe_wrapper.json"
    recipe_wrapper.touch()

    # Run the wrapper starter
    sys.argv = [
        "cryoemservices.resubmit_wrapper",
        "--wrapper",
        str(recipe_wrapper),
        "--config_file",
        str(tmp_path / "config.yaml"),
    ]
    with pytest.raises(FileNotFoundError):
        resubmit_wrapper.run()

    # Check nothing got called
    mock_transport.assert_not_called()
