from __future__ import annotations

import sys
from unittest import mock

from cryoemservices.cli import generate_session_report


@mock.patch("cryoemservices.cli.generate_session_report.ispyb")
@mock.patch("cryoemservices.cli.generate_session_report.sqlalchemy")
def test_generate_session_report_dataless(mock_sqlalchemy, mock_ispyb, tmp_path):
    """Test the session report if processing stages are not found"""
    test_session_results = generate_session_report.SessionResults(
        dc_id=1, logo="logo.jpg"
    )
    test_session_results.gather_preprocessing_ispyb_results()
    test_session_results.gather_classification_ispyb_results()
    test_session_results.write_report()


@mock.patch("cryoemservices.cli.generate_session_report.ispyb")
@mock.patch("cryoemservices.cli.generate_session_report.sqlalchemy")
@mock.patch("cryoemservices.cli.generate_session_report.pylatex")
def test_generate_session_report_with_processing(
    mock_pylatex, mock_sqlalchemy, mock_ispyb, tmp_path
):
    """Test the session report if processing stages are present, but without database"""
    test_session_results = generate_session_report.SessionResults(
        dc_id=1, logo="logo.jpg"
    )

    test_session_results.processing_stages = [
        "em-spa-preprocess",
        "em-spa-class2d",
        "em-spa-class3d",
        "em-spa-refine",
    ]
    test_session_results.autoproc_ids = [1, 2, 3, 4]
    test_session_results.raw_name = "raw"
    test_session_results.image_directory = f"{tmp_path}/images"
    (tmp_path / "tmp").mkdir()

    test_session_results.gather_preprocessing_ispyb_results()
    test_session_results.gather_classification_ispyb_results()
    test_session_results.write_report()

    mock_pylatex.Document.assert_any_call(
        geometry_options={
            "tmargin": "2cm",
            "bmargin": "2cm",
            "lmargin": "2cm",
            "rmargin": "2cm",
        }
    )
    mock_pylatex.Document(
        geometry_options={
            "tmargin": "2cm",
            "bmargin": "2cm",
            "lmargin": "2cm",
            "rmargin": "2cm",
        }
    ).generate_pdf.assert_any_call(f"{tmp_path}/tmp/report_raw")


@mock.patch("cryoemservices.cli.generate_session_report.SessionResults")
def test_session_report_cli(mock_session_report, tmp_path):
    """Test that the command line function works"""
    # Run the session report maker
    sys.argv = [
        "cryoemservices.generate_session_report",
        "--dc_id",
        "10",
        "--logo",
        "/test/logo.jpg",
    ]
    generate_session_report.run()

    mock_session_report.assert_any_call(10, "/test/logo.jpg")
    mock_session_report().gather_preprocessing_ispyb_results.assert_any_call()
    mock_session_report().gather_classification_ispyb_results.assert_any_call()
    mock_session_report().write_report.assert_any_call()
