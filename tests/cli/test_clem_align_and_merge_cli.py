from __future__ import annotations

import subprocess
import sys
from unittest import mock

from cryoemservices.cli import clem_align_and_merge


@mock.patch("cryoemservices.wrappers.clem_align_and_merge.align_and_merge_stacks")
def test_align_and_merge(mock_align_and_merge, tmp_path):
    """Test that the cli runs the expected default args"""
    (tmp_path / "file1").touch()

    # Run the cli
    sys.argv = [
        "clem.align_and_merge",
        f"{tmp_path}/file1",
    ]
    clem_align_and_merge.run()

    mock_align_and_merge.assert_called_with(
        images=[tmp_path / "file1"],
        metadata=None,
        crop_to_n_frames=None,
        align_self="",
        flatten="mean",
        align_across="",
    )


@mock.patch("cryoemservices.wrappers.clem_align_and_merge.align_and_merge_stacks")
def test_align_and_merge_string_list(mock_align_and_merge, tmp_path):
    """Test that the cli runs the expected default args"""
    (tmp_path / "file1").touch()
    (tmp_path / "file2").touch()

    # Run the cli
    sys.argv = [
        "clem.align_and_merge",
        f"['{tmp_path}/file1', '{tmp_path}/file2']",
    ]
    clem_align_and_merge.run()

    mock_align_and_merge.assert_called_with(
        images=[tmp_path / "file1", tmp_path / "file2"],
        metadata=None,
        crop_to_n_frames=None,
        align_self="",
        flatten="mean",
        align_across="",
    )


def test_align_and_merge_exists():
    """Test the clem.align_and_merge CLI is made"""
    result = subprocess.run(
        [
            "clem.align_and_merge",
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
        "usage:clem.align_and_merge[-h][--metadataMETADATA]"
        "[--crop-to-n-framesCROP_TO_N_FRAMES][--align-selfALIGN_SELF]"
        "[--flattenFLATTEN][--align-acrossALIGN_ACROSS][--debug]images[images...]"
    )
