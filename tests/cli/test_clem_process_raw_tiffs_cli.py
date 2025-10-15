from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pytest_mock import MockerFixture

from cryoemservices.cli import clem_process_raw_tiffs


def test_process_raw_tiffs_with_optional_args(mocker: MockerFixture, tmp_path: Path):
    """Test that the cli runs with all args provided"""
    tiff_files: list[Path] = sorted(
        [
            (tmp_path / "file--1.tiff"),
            (tmp_path / "file--2.tif"),
        ]
    )
    for file in tiff_files:
        file.touch()
    metadata_file = Path("file.xlif")
    num_procs = 20

    # Set up mock objects and results
    mock_setup = mocker.patch(
        "cryoemservices.cli.clem_process_raw_tiffs.set_up_logging"
    )
    mock_convert = mocker.patch(
        "cryoemservices.wrappers.clem_process_raw_tiffs.process_tiff_files"
    )
    mock_print = mocker.patch("cryoemservices.cli.clem_process_raw_tiffs.print")

    dummy_result = [{"dummy": i} for i in range(3)]
    mock_convert.return_value = dummy_result

    # Run the cli
    sys.argv = [
        "clem_process_raw_tiffs",
        str(tiff_files[0]),
        "--root-folder",
        "root",
        "--metadata",
        str(metadata_file),
        "-n",
        str(num_procs),
        "--debug",
    ]
    clem_process_raw_tiffs.run()

    # Check that calls were made with the expected values
    mock_setup.assert_called_once_with(debug=True)
    args, kwargs = mock_convert.call_args
    assert sorted(kwargs["tiff_list"]) == tiff_files
    assert kwargs["root_folder"] == "root"
    assert kwargs["metadata_file"] == metadata_file
    assert kwargs["number_of_processes"] == num_procs

    mock_print.assert_called_with("TIFF processing workflow successfully completed")


def test_process_raw_tiffs_with_default_args(mocker: MockerFixture, tmp_path: Path):
    """Test that the cli runs the expected default args"""
    tiff_file = tmp_path / "file--1.tiff"
    tiff_file.touch()

    mock_convert = mocker.patch(
        "cryoemservices.wrappers.clem_process_raw_tiffs.process_tiff_files"
    )
    mock_setup = mocker.patch(
        "cryoemservices.cli.clem_process_raw_tiffs.set_up_logging"
    )

    # Run the cli
    sys.argv = [
        "clem_process_raw_tiffs",
        str(tiff_file),
    ]
    clem_process_raw_tiffs.run()

    # Check that calls were mae with the expected values
    mock_setup.assert_called_once_with(debug=False)
    mock_convert.assert_called_with(
        tiff_list=[tiff_file],
        root_folder="images",
        metadata_file=None,
        number_of_processes=1,
    )


def test_process_raw_tiffs_exists():
    """Test that clem.process_raw_tiffs CLI is made"""
    result = subprocess.run(
        [
            "clem.process_raw_tiffs",
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
        "usage:clem.process_raw_tiffs[-h][--root-folderROOT_FOLDER][--metadataMETADATA][-nNUM_PROCS][--debug]tiff_file"
    )
