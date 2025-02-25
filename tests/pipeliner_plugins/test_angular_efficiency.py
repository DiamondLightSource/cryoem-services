from __future__ import annotations

import subprocess
import sys
from unittest import mock

import numpy as np

from cryoemservices.pipeliner_plugins import angular_efficiency


def test_efficiency_from_map():
    psf_map = np.arange(1000).reshape((10, 10, 10))
    predicted_eff = angular_efficiency.efficiency_from_map(psf_map, 10)
    assert np.isclose(predicted_eff, 0.67, rtol=0.01)


def test_find_efficiency():
    theta = np.arange(0, 45, 45 / 180)
    phi = np.arange(180)

    predicted_eff = angular_efficiency.find_efficiency(theta, phi, 10, 160)

    # cryoEF gives 0.75 for these angles, the python implementation gives 0.77
    assert np.abs(predicted_eff - 0.75) < 0.021


@mock.patch("cryoemservices.pipeliner_plugins.angular_efficiency.find_efficiency")
def test_run_find_efficiency(mock_efficiency, tmp_path):
    data_file = tmp_path / "angles.star"
    with open(data_file, "w") as f:
        f.write(
            "data_global\n\nloop_\n_dummy\n1\n\n"
            "data_particles\n\nloop_\n_rlnAngleTilt\n_rlnAngleRot\n_rlnClassNumber\n"
            "1 0.1 1\n2 0.2 1\n3 0.3 2\n"
        )

    sys.argv = [
        "cryoemservices.angular_efficiency",
        "--file",
        str(data_file),
        "--class_id",
        "1",
        "--boxsize",
        "128",
        "--bfactor",
        "80",
    ]
    angular_efficiency.run()

    mock_efficiency.assert_called_once()
    assert (mock_efficiency.call_args.kwargs["theta_degrees"] == [1, 2]).all()
    assert (mock_efficiency.call_args.kwargs["phi_degrees"] == [0.1, 0.2]).all()
    assert mock_efficiency.call_args.kwargs["boxsize"] == 128
    assert mock_efficiency.call_args.kwargs["bfactor"] == 80


def test_find_efficiency_exists():
    """Test the DLQ check CLI is made"""
    result = subprocess.run(
        [
            "cryoemservices.angular_efficiency",
            "--help",
        ],
        capture_output=True,
    )
    assert not result.returncode

    # Find the first line of the help and strip out all the spaces and newlines
    stdout_as_string = result.stdout.decode("utf8", "replace")
    cleaned_help_line = (
        stdout_as_string.split("\n\n")[0].replace("\n", "").replace(" ", "")
    )
    assert cleaned_help_line == (
        "usage:cryoemservices.angular_efficiency[-h]"
        "-fFILE[-cCLASS_ID][--boxsizeBOXSIZE][--bfactorBFACTOR]"
    )
