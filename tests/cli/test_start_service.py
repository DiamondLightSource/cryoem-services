from __future__ import annotations

import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path
from unittest import mock

from cryoemservices.cli import start_service


@mock.patch("cryoemservices.cli.start_service.PikaTransport")
@mock.patch("cryoemservices.services.motioncorr.MotionCorr")
def test_start_service_with_optional_args(mock_service, mock_transport, tmp_path):
    """Test that wrappers can be started and run"""
    # Create a sample config file
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as cf:
        cf.write("rabbitmq_credentials: rmq_creds\n")
        cf.write(f"recipe_directory: {tmp_path}\n")

    # Run the wrapper starter
    sys.argv = [
        "cryoemservices.service",
        "--service",
        "MotionCorr",
        "--config_file",
        str(config_file),
        "--slurm",
        "extra",
        "--queue",
        "motioncorr",
    ]
    start_service.run()

    # Check the calls which should be made
    mock_transport.assert_called()
    mock_transport().load_configuration_file.assert_called_with(Path("rmq_creds"))

    mock_service.assert_called_with(
        environment={
            "config": f"{tmp_path}/config.yaml",
            "slurm_cluster": "extra",
            "queue": "motioncorr",
        },
        transport=mock.ANY,
        single_message_mode=False,
    )
    mock_service().start.assert_called_once()


@mock.patch("cryoemservices.cli.start_service.PikaTransport")
@mock.patch("cryoemservices.services.motioncorr.MotionCorr")
def test_start_service_with_default_args(mock_service, mock_transport, tmp_path):
    """Test that wrappers can be started and run"""

    # Create a sample config file
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as cf:
        cf.write("rabbitmq_credentials: rmq_creds\n")
        cf.write(f"recipe_directory: {tmp_path}\n")

    # Run the wrapper starter
    sys.argv = [
        "cryoemservices.service",
        "--service",
        "MotionCorr",
        "--config_file",
        str(config_file),
        "--single_message",
    ]
    start_service.run()

    # Check the calls which should be made
    mock_transport.assert_called()
    mock_transport().load_configuration_file.assert_called_with(Path("rmq_creds"))

    mock_service.assert_called_with(
        environment={
            "config": f"{tmp_path}/config.yaml",
            "slurm_cluster": "default",
            "queue": "",
        },
        transport=mock.ANY,
        single_message_mode=True,
    )
    mock_service().start.assert_called_once()


def test_start_service_exists():
    """Test the service CLI is made"""
    result = subprocess.run(
        [
            "cryoemservices.service",
            "--help",
        ],
        capture_output=True,
    )
    assert not result.returncode

    known_services = [e.name for e in entry_points(group="cryoemservices.services")]

    # Find the first line of the help and strip out all the spaces and newlines
    stdout_as_string = result.stdout.decode("utf8", "replace")
    cleaned_help_line = (
        stdout_as_string.split("\n\n")[0].replace("\n", "").replace(" ", "")
    )
    assert cleaned_help_line == (
        "usage:cryoemservices.service[-h]-s{"
        + ",".join(sorted(known_services))
        + "}-cCONFIG_FILE[--slurmSLURM][--queueQUEUE][--single_message]"
    )
