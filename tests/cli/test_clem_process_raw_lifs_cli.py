from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pytest_mock import MockerFixture

from cryoemservices.cli import clem_process_raw_lifs


def test_process_raw_lifs_with_optional_args(mocker: MockerFixture):
    """Test that the cli runs with all args provided"""
    # Set up necessary mock objects
    mock_convert = mocker.patch(
        "cryoemservices.wrappers.clem_process_raw_lifs.process_lif_file"
    )
    mock_print = mocker.patch("cryoemservices.cli.clem_process_raw_lifs.print")
    mock_setup = mocker.patch("cryoemservices.cli.clem_process_raw_lifs.set_up_logging")

    # Create some dummy results
    dummy_results = [{"result": i} for i in range(3)]
    mock_convert.return_value = dummy_results

    # Run the cli
    sys.argv = [
        "clem.process_raw_lifs",
        "file.lif",
        "--root-folder",
        "root",
        "--num-procs",
        "10",
        "--debug",
    ]
    clem_process_raw_lifs.run()

    # Check that it was called with the correct args
    mock_setup.assert_called_with(debug=True)
    mock_convert.assert_called_with(
        file=Path("file.lif"),
        root_folder="root",
        number_of_processes=10,
    )

    # Check that the dummy results are printed at the end
    mock_print.assert_called_with("LIF processing workflow successfully completed")


def test_process_raw_lifs_with_default_args(mocker: MockerFixture):
    """Test that the cli runs with all args provided"""
    # Set up necessary mock objects
    mock_convert = mocker.patch(
        "cryoemservices.wrappers.clem_process_raw_lifs.process_lif_file"
    )
    mock_setup = mocker.patch("cryoemservices.cli.clem_process_raw_lifs.set_up_logging")

    # Create some dummy results
    dummy_results = [{"result": i} for i in range(3)]
    mock_convert.return_value = dummy_results

    # Run the cli
    sys.argv = [
        "clem.process_raw_lifs",
        "file.lif",
    ]
    clem_process_raw_lifs.run()

    # Check that it was called with the correct args
    mock_setup.assert_called_with(debug=False)
    mock_convert.assert_called_with(
        file=Path("file.lif"),
        root_folder="images",
        number_of_processes=1,
    )


def test_process_raw_lifs_exists():
    """Test that clem.process_raw_lifs CLI is made"""
    result = subprocess.run(
        [
            "clem.process_raw_lifs",
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
        "usage:clem.process_raw_lifs[-h][--root-folderROOT_FOLDER][-nNUM_PROCS][--debug]lif_file"
    )
