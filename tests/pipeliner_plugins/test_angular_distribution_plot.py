from __future__ import annotations

import subprocess
import sys
from unittest import mock

import numpy as np

from cryoemservices.pipeliner_plugins import angular_distribution_plot


@mock.patch("cryoemservices.pipeliner_plugins.angular_distribution_plot.hp")
def test_angular_distribution_plot(mock_healpy, tmp_path):
    theta = np.array([1.0]) / np.pi * 180
    phi = np.array([2.0]) / np.pi * 180

    mock_healpy.pixelfunc.ang2pix.return_value = [0, 0]
    mock_healpy.nside2npix.return_value = 1

    angular_distribution_plot.angular_distribution_plot(
        theta, phi, 2, tmp_path / "output.jpeg", "label"
    )

    mock_healpy.pixelfunc.ang2pix.assert_called_once_with(
        8,
        np.array([1.0]),
        np.array([2.0]),
    )
    mock_healpy.nside2npix.assert_called_once_with(8)
    mock_healpy.mollview.assert_called_once_with(
        np.array([2.0]),
        title="Angular distribution of particles for class: label",
        unit="Number of particles",
        flip="geo",
    )
    mock_healpy.graticule.assert_called_once()

    assert (tmp_path / "output.jpeg").is_file()


@mock.patch(
    "cryoemservices.pipeliner_plugins.angular_distribution_plot.angular_distribution_plot"
)
def test_run_angular_distribution_plot(mock_dist_plot, tmp_path):
    data_file = tmp_path / "angles.star"
    with open(data_file, "w") as f:
        f.write(
            "data_global\n\nloop_\n_dummy\n1\n\n"
            "data_particles\n\nloop_\n_rlnAngleTilt\n_rlnAngleRot\n_rlnClassNumber\n"
            "1 0.1 1\n2 0.2 1\n3 0.3 2\n"
        )

    sys.argv = [
        "cryoemservices.angular_distribution_plot",
        "--file",
        str(data_file),
        "--output",
        "output.jpeg",
        "--class_id",
        "1",
        "--healpix_order",
        "4",
    ]
    angular_distribution_plot.run()

    mock_dist_plot.assert_called_once()
    assert (mock_dist_plot.call_args.kwargs["theta_degrees"] == [1, 2]).all()
    assert (mock_dist_plot.call_args.kwargs["phi_degrees"] == [0.1, 0.2]).all()
    assert mock_dist_plot.call_args.kwargs["healpix_order"] == 4
    assert mock_dist_plot.call_args.kwargs["output_jpeg"] == "output.jpeg"
    assert mock_dist_plot.call_args.kwargs["class_label"] == "1"


def test_angular_distribution_plot_exists():
    """Test the distribution plot CLI is made"""
    result = subprocess.run(
        [
            "cryoemservices.angular_distribution_plot",
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
        "usage:cryoemservices.angular_distribution_plot[-h]"
        "-fFILE-oOUTPUT[-cCLASS_ID][--healpix_orderHEALPIX_ORDER]"
    )
