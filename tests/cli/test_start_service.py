from __future__ import annotations

import sys
from unittest import mock

from cryoemservices.cli import start_service


@mock.patch("workflows.frontend.Frontend")
def test_start_service(mock_frontend, tmp_path):
    """Test that wrappers can be started and run"""
    # Create a sample config file
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as cf:
        cf.write("rabbitmq_credentials: rmq_creds\n")
        cf.write(f"recipe_directory: {tmp_path}\n")

    # Run the wrapper starter
    sys.argv = [
        "cryoemservices.wrap",
        "--service",
        "MotionCorr",
        "--config_file",
        str(config_file),
    ]
    start_service.run()

    mock_frontend.assert_called_with(
        **{
            "service": "MotionCorr",
            "transport": mock.ANY,
            "transport_command_channel": "command",
            "verbose_service": True,
            "environment": {"config": f"{tmp_path}/config.yaml"},
        }
    )
    mock_frontend().run.assert_called()
