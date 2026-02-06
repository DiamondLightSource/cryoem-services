from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pytest_mock import MockerFixture

from cryoemservices.cli import clem_align_and_merge


def test_align_and_merge_with_optional_args(mocker: MockerFixture, tmp_path: Path):
    """Test that the cli runs with all args provided"""
    (tmp_path / "file1").touch()
    (tmp_path / "file2").touch()

    # Set up mocked objects
    mock_setup = mocker.patch("cryoemservices.cli.clem_align_and_merge.set_up_logging")
    mock_align_and_merge = mocker.patch(
        "cryoemservices.wrappers.clem_align_and_merge.align_and_merge_stacks"
    )

    # Run the cli
    sys.argv = [
        "clem.align_and_merge",
        f"{tmp_path}/file1",
        f"{tmp_path}/file2",
        "--metadata",
        "metadata.xml",
        "--crop-to-n-frames",
        "5",
        "--align-self",
        "enabled",
        "--flatten",
        "min",
        "--align-across",
        "enabled",
        "--num-procs",
        "4",
        "--debug",
    ]
    clem_align_and_merge.run()

    mock_setup.assert_called_with(debug=True)
    mock_align_and_merge.assert_called_with(
        images=[tmp_path / "file1", tmp_path / "file2"],
        metadata=Path("metadata.xml"),
        crop_to_n_frames=5,
        align_self="enabled",
        flatten="min",
        align_across="enabled",
        num_procs=4,
    )


def test_align_and_merge_with_default_args(mocker: MockerFixture, tmp_path: Path):
    """Test that the cli runs the expected default args"""
    (tmp_path / "file1").touch()

    # Set up mocked objects
    mock_setup = mocker.patch("cryoemservices.cli.clem_align_and_merge.set_up_logging")
    mock_align_and_merge = mocker.patch(
        "cryoemservices.wrappers.clem_align_and_merge.align_and_merge_stacks"
    )

    # Run the cli
    sys.argv = [
        "clem.align_and_merge",
        f"{tmp_path}/file1",
    ]
    clem_align_and_merge.run()

    mock_setup.assert_called_once_with(debug=False)
    mock_align_and_merge.assert_called_with(
        images=[tmp_path / "file1"],
        metadata=None,
        crop_to_n_frames=None,
        align_self="",
        flatten="mean",
        align_across="",
        num_procs=1,
    )


def test_align_and_merge_string_list(mocker: MockerFixture, tmp_path):
    """Test that the cli runs the expected default args"""
    (tmp_path / "file1").touch()
    (tmp_path / "file2").touch()

    # Set up mocked objects
    mock_align_and_merge = mocker.patch(
        "cryoemservices.wrappers.clem_align_and_merge.align_and_merge_stacks"
    )

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
        num_procs=1,
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
        "[--flattenFLATTEN][--align-acrossALIGN_ACROSS]"
        "[--num-procsNUM_PROCS][--debug]images[images...]"
    )
