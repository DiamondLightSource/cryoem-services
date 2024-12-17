from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

from cryoemservices.cli import run_wrapper


@mock.patch("cryoemservices.wrappers.class2d_wrapper.Class2DWrapper")
@mock.patch("cryoemservices.cli.run_wrapper.PikaTransport")
@mock.patch("cryoemservices.cli.run_wrapper.RecipeWrapper")
def test_run_wrapper(mock_rw, mock_transport, mock_class2d, tmp_path):
    """Test that wrappers can be started and run"""
    # Create a sample json recipe and config file
    recipe_wrapper = tmp_path / "recipe_wrapper.json"
    config_file = tmp_path / "config.yaml"
    with open(recipe_wrapper, "w") as rw:
        rw.write('{\n    "test": "test"\n}')
    with open(config_file, "w") as cf:
        cf.write("rabbitmq_credentials: rmq_creds\n")
        cf.write(f"recipe_directory: {tmp_path}\n")

    # Run the wrapper starter
    sys.argv = [
        "cryoemservices.wrap",
        "--wrap",
        "Class2D",
        "--recipe_wrap",
        str(recipe_wrapper),
        "--config_file",
        str(config_file),
    ]
    run_wrapper.run()

    mock_transport.assert_called()
    mock_transport().load_configuration_file.assert_called_with(Path("rmq_creds"))
    mock_transport().connect.assert_called()
    mock_rw.assert_called_with(message={"test": "test"}, transport=mock_transport())

    mock_class2d.assert_called()
    mock_class2d().set_recipe_wrapper.assert_called()
    mock_class2d().prepare.assert_called_with("Starting processing")
    mock_class2d().run.assert_called()
    mock_class2d().success.assert_called_with("Finished processing")
    mock_class2d().done.assert_called_with("Finished processing")

    mock_transport().disconnect.assert_called()
