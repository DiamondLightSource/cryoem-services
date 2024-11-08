from __future__ import annotations

from unittest import mock

from cryoemservices.cli.generate_session_report import SessionResults


@mock.patch("cryoemservices.cli.generate_session_report.ispyb")
@mock.patch("cryoemservices.cli.generate_session_report.sqlalchemy")
def test_generate_session_report_dataless(mock_sqlalchemy, mock_ispyb, tmp_path):
    test_session_results = SessionResults(dc_id=1, logo="logo.jpg")
    test_session_results.gather_preprocessing_ispyb_results()
    test_session_results.gather_classification_ispyb_results()
    test_session_results.write_report()


@mock.patch("cryoemservices.cli.generate_session_report.ispyb")
@mock.patch("cryoemservices.cli.generate_session_report.sqlalchemy")
def test_generate_session_report_with_processing(mock_sqlalchemy, mock_ispyb, tmp_path):
    test_session_results = SessionResults(dc_id=1, logo="logo.jpg")

    test_session_results.processing_stages = [
        "em-spa-preprocess",
        "em-spa-class2d",
        "em-spa-class3d",
        "em-spa-refine",
    ]
    test_session_results.autoproc_ids = [0, 1, 2, 3]

    test_session_results.gather_preprocessing_ispyb_results()
    test_session_results.gather_classification_ispyb_results()
    test_session_results.write_report()
