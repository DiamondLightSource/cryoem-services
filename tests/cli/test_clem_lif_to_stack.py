from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest import mock

from cryoemservices.cli import clem_lif_to_stack


@mock.patch("cryoemservices.cli.clem_lif_to_stack.convert_lif_to_stack")
def test_lif_to_stack_with_optional_args(mock_convert_lif_to_stack, tmp_path):
    """Test that the cli runs with all args provided"""
    # Run the cli
    sys.argv = [
        "clem.lif_to_stack",
        "file.lif",
        "--root-folder",
        "root",
        "--num-procs",
        "10",
    ]
    clem_lif_to_stack.run()

    mock_convert_lif_to_stack.assert_called_with(
        file=Path("file.lif"),
        root_folder="root",
        number_of_processes=10,
    )


@mock.patch("cryoemservices.cli.clem_lif_to_stack.convert_lif_to_stack")
def test_lif_to_stack_with_default_args(mock_convert_lif_to_stack, tmp_path):
    """Test that the cli runs the expected default args"""
    # Run the cli
    sys.argv = [
        "clem.lif_to_stack",
        "file.lif",
    ]
    clem_lif_to_stack.run()

    mock_convert_lif_to_stack.assert_called_with(
        file=Path("file.lif"),
        root_folder="images",
        number_of_processes=1,
    )


def test_lif_to_stack_exists():
    """Test the clem.lif_to_stack CLI is made"""
    result = subprocess.run(
        [
            "clem.lif_to_stack",
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
        "usage:clem.lif_to_stack[-h][--root-folderROOT_FOLDER][-nNUM_PROCS]lif_file"
    )
