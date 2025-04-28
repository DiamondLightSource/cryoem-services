from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

from cryoemservices.cli import start_service


@mock.patch("cryoemservices.cli.start_service.PikaTransport")
@mock.patch("cryoemservices.cli.start_service.multiprocessing.Process")
@mock.patch("cryoemservices.services.motioncorr.MotionCorr")
def test_start_service_with_optional_args(
    mock_service, mock_multiprocess, mock_transport, tmp_path
):
    """Test that wrappers can be started and run"""
    mock_multiprocess().is_alive.side_effect = [True, False]

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
        "--slurm",
        "extra",
        "--queue",
        "motioncorr",
    ]
    start_service.run()

    # Check the calls which should be made
    mock_transport.assert_called()
    mock_transport().load_configuration_file.assert_called_with(Path("rmq_creds"))
    mock_transport().connect.assert_called()
    mock_transport().disconnect.assert_called()
    mock_transport().is_connected.assert_called()

    mock_service.assert_called_with(
        environment={
            "config": f"{tmp_path}/config.yaml",
            "slurm_cluster": "extra",
            "queue": "motioncorr",
        },
        transport=mock.ANY,
    )

    mock_multiprocess.assert_called_with(target=mock.ANY)
    mock_multiprocess().start.assert_called()
    mock_multiprocess().is_alive.assert_called()
    mock_multiprocess().terminate.assert_called()
    mock_multiprocess().join.assert_called()


@mock.patch("cryoemservices.cli.start_service.PikaTransport")
@mock.patch("cryoemservices.cli.start_service.multiprocessing.Process")
@mock.patch("cryoemservices.services.motioncorr.MotionCorr")
def test_start_service_with_default_args(
    mock_service, mock_multiprocess, mock_transport, tmp_path
):
    """Test that wrappers can be started and run"""
    mock_multiprocess().is_alive.side_effect = [True, False]

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

    # Check the calls which should be made
    mock_transport.assert_called()
    mock_transport().load_configuration_file.assert_called_with(Path("rmq_creds"))
    mock_transport().connect.assert_called()
    mock_transport().disconnect.assert_called()
    mock_transport().is_connected.assert_called()

    mock_service.assert_called_with(
        environment={
            "config": f"{tmp_path}/config.yaml",
            "slurm_cluster": "default",
            "queue": "",
        },
        transport=mock.ANY,
    )

    mock_multiprocess.assert_called_with(target=mock.ANY)
    mock_multiprocess().start.assert_called()
    mock_multiprocess().is_alive.assert_called()
    mock_multiprocess().terminate.assert_called()
    mock_multiprocess().join.assert_called()
