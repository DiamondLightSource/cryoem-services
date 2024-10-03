from __future__ import annotations

from pathlib import Path
from unittest import mock

import mrcfile
import numpy as np

from cryoemservices.pipeliner_plugins import symmetry_finder


def _make_test_mrc_file(mrc_name: Path):
    # Random mrcfile creation as reference
    randmrc = np.random.logistic(loc=0, scale=0.1, size=(162, 162, 162)).astype(
        np.float32
    )
    with mrcfile.new(mrc_name, overwrite=True) as mrc:
        mrc.set_data(randmrc)


def test_find_difference(tmp_path):
    """Test that the difference code returns zero if given a simgle file"""
    volume_file = tmp_path / "random_mrcfile.mrc"
    _make_test_mrc_file(volume_file)

    assert symmetry_finder.find_difference(volume_file, volume_file) == 0


@mock.patch("cryoemservices.pipeliner_plugins.symmetry_finder.subprocess.run")
@mock.patch("cryoemservices.pipeliner_plugins.symmetry_finder.find_difference")
def test_determine_symmetry_random(mock_difference, mock_subprocess, tmp_path):
    """Test the symmetry determination code runs all the checks"""
    mock_subprocess().returncode = 0
    mock_difference.return_value = 1

    volume_file = tmp_path / "random_mrcfile.mrc"
    symmetry_result = symmetry_finder.determine_symmetry(volume_file)

    assert symmetry_result[0] == "I"
    assert symmetry_result[1] == f"{tmp_path}/random_mrcfile_aligned_I_symmetrised.mrc"

    for symmetry in ["C2", "C3", "C4", "C5", "C6", "C7", "C8", "T", "O", "I"]:
        mock_subprocess.assert_any_call(
            [
                "relion_align_symmetry",
                "--i",
                f"{tmp_path}/random_mrcfile.mrc",
                "--o",
                f"{tmp_path}/random_mrcfile_aligned_{symmetry}.mrc",
                "--sym",
                symmetry,
            ],
            capture_output=True,
        )
        mock_subprocess.assert_any_call(
            [
                "relion_image_handler",
                "--i",
                f"{tmp_path}/random_mrcfile_aligned_{symmetry}.mrc",
                "--o",
                f"{tmp_path}/random_mrcfile_aligned_{symmetry}_symmetrised.mrc",
                "--sym",
                symmetry,
            ],
            capture_output=True,
        )
        mock_difference.assert_any_call(
            tmp_path / f"random_mrcfile_aligned_{symmetry}.mrc",
            tmp_path / f"random_mrcfile_aligned_{symmetry}_symmetrised.mrc",
        )
