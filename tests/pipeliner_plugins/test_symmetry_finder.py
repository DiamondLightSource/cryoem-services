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
def test_determine_symmetry_random_precompute(
    mock_difference, mock_subprocess, tmp_path
):
    """Test the symmetry determination code runs all the checks"""
    mock_subprocess().returncode = 0
    mock_difference.return_value = 1

    volume_file = tmp_path / "random_mrcfile.mrc"
    symmetry_result = symmetry_finder.determine_symmetry(
        volume=volume_file, use_precomputed_scores=True
    )

    assert symmetry_result[0] == "I"
    assert symmetry_result[1] == f"{tmp_path}/random_mrcfile_aligned_I_symmetrised.mrc"

    assert mock_subprocess.call_count == 21
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


@mock.patch("cryoemservices.pipeliner_plugins.symmetry_finder.subprocess.run")
@mock.patch("cryoemservices.pipeliner_plugins.symmetry_finder.find_difference")
@mock.patch("cryoemservices.pipeliner_plugins.symmetry_finder.mrcfile")
def test_determine_symmetry_random_do_noise(
    mock_mrcfile, mock_difference, mock_subprocess, tmp_path
):
    """Test the symmetry determination code runs all the checks"""
    mock_subprocess().returncode = 0
    mock_difference.return_value = 1
    mock_mrcfile().__enter__().header = {"nx": 10, "ny": 10, "nz": 10}

    volume_file = tmp_path / "random_mrcfile.mrc"
    symmetry_finder.determine_symmetry(volume=volume_file, use_precomputed_scores=False)

    assert mock_subprocess.call_count == 41
    for symmetry in ["C2", "C3", "C4", "C5", "C6", "C7", "C8", "T", "O", "I"]:
        mock_subprocess.assert_any_call(
            [
                "relion_align_symmetry",
                "--i",
                f"{tmp_path}/random_reference.mrc",
                "--o",
                f"{tmp_path}/random_reference_aligned_{symmetry}.mrc",
                "--sym",
                symmetry,
            ],
            capture_output=True,
        )
        mock_subprocess.assert_any_call(
            [
                "relion_image_handler",
                "--i",
                f"{tmp_path}/random_reference_aligned_{symmetry}.mrc",
                "--o",
                f"{tmp_path}/random_reference_aligned_{symmetry}_symmetrised.mrc",
                "--sym",
                symmetry,
            ],
            capture_output=True,
        )
        mock_difference.assert_any_call(
            tmp_path / f"random_reference_aligned_{symmetry}.mrc",
            tmp_path / f"random_reference_aligned_{symmetry}_symmetrised.mrc",
        )
        assert not (tmp_path / f"random_reference_aligned_{symmetry}.mrc").is_file()
        assert not (
            tmp_path / f"random_reference_aligned_{symmetry}_symmetrised.mrc"
        ).is_file()

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
