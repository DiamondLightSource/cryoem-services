from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest import mock

from cryoemservices.cli import clem_tiff_to_stack


@mock.patch("cryoemservices.cli.clem_tiff_to_stack.convert_tiff_to_stack")
def test_tiff_to_stack_with_optional_args(mock_convert_tiff_to_stack, tmp_path):
    """Test that the cli runs with all args provided"""
    (tmp_path / "file--1.tiff").touch()
    (tmp_path / "file--2.tif").touch()

    # Run the cli
    sys.argv = [
        "clem_tiff_to_stack",
        f"{tmp_path}/file--1.tiff",
        "--root-folder",
        "root",
        "--metadata",
        "file.xlif",
    ]
    clem_tiff_to_stack.run()

    try:
        mock_convert_tiff_to_stack.assert_called_with(
            tiff_list=[tmp_path / "file--2.tif", tmp_path / "file--1.tiff"],
            root_folder="root",
            metadata_file=Path("file.xlif"),
        )
    except AssertionError:
        mock_convert_tiff_to_stack.assert_called_with(
            tiff_list=[tmp_path / "file--1.tiff", tmp_path / "file--2.tif"],
            root_folder="root",
            metadata_file=Path("file.xlif"),
        )


@mock.patch("cryoemservices.cli.clem_tiff_to_stack.convert_tiff_to_stack")
def test_tiff_to_stack_with_default_args(mock_convert_tiff_to_stack, tmp_path):
    """Test that the cli runs the expected default args"""
    (tmp_path / "file--1.tiff").touch()

    # Run the cli
    sys.argv = [
        "clem_tiff_to_stack",
        f"{tmp_path}/file.tiff",
    ]
    clem_tiff_to_stack.run()

    mock_convert_tiff_to_stack.assert_called_with(
        tiff_list=[tmp_path / "file--1.tiff"],
        root_folder="images",
        metadata_file=None,
    )


def test_tiff_to_stack_exists():
    """Test the clem.tiff_to_stack CLI is made"""
    result = subprocess.run(
        [
            "clem.tiff_to_stack",
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
        "usage:clem.tiff_to_stack[-h][--root-folderROOT_FOLDER][--metadataMETADATA]tiff_file"
    )
