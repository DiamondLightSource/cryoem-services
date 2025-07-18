from __future__ import annotations

from unittest import mock

import ispyb.sqlalchemy as models
from workflows.recipe import Recipe

from cryoemservices.util import process_recipe_tools
from cryoemservices.util.config import ServiceConfig


@mock.patch("cryoemservices.util.process_recipe_tools.select")
def test_get_processing_info(mock_select):
    """Test the lookup calls for processing parameters"""

    class MockParameters:
        recipe = "example"
        dataCollectionId = 10
        parameterKey = "parameter_a"
        parameterValue = "A"

    # A mock for the query results
    mock_session = mock.MagicMock()
    mock_session.execute().scalars().first.return_value = MockParameters()
    mock_session.execute().scalars().all.return_value = [MockParameters()]

    output_parameters = process_recipe_tools.get_processing_info(1, mock_session)

    # Check the sqlalchemy calls
    assert mock_session.execute.call_count == 4
    assert mock_session.execute().scalars.call_count == 4
    mock_session.execute().scalars().first.assert_called_once()
    mock_session.execute().scalars().all.assert_called_once()

    mock_select.assert_any_call(models.ProcessingJob)
    mock_select.assert_any_call(models.ProcessingJobParameter)
    assert mock_select().where.call_count == 2

    # Check the return value
    assert list(output_parameters.keys()) == [
        "recipe",
        "ispyb_dcid",
        "ispyb_reprocessing_parameters",
    ]
    assert output_parameters["recipe"] == "example"
    assert output_parameters["ispyb_dcid"] == 10
    assert output_parameters["ispyb_reprocessing_parameters"] == {"parameter_a": "A"}


@mock.patch("cryoemservices.util.process_recipe_tools.select")
def test_get_image_directory(mock_select):
    """Test the lookup calls for image directory from data collections"""

    class MockParameters:
        imageDirectory = "/path/to/images/"

    # A mock for the query results
    mock_session = mock.MagicMock()
    mock_session.execute().scalars().first.return_value = MockParameters()

    output_parameters = process_recipe_tools.get_image_directory(1, mock_session)

    # Check the sqlalchemy calls
    assert mock_session.execute.call_count == 2
    assert mock_session.execute().scalars.call_count == 2
    mock_session.execute().scalars().first.assert_called_once()

    mock_select.assert_called_with(models.DataCollection)
    mock_select().where.assert_called_once()

    # Check the return value
    assert output_parameters == "/path/to/images/"


@mock.patch("cryoemservices.util.process_recipe_tools.models")
@mock.patch("cryoemservices.util.process_recipe_tools.sessionmaker")
@mock.patch("cryoemservices.util.process_recipe_tools.create_engine")
@mock.patch("cryoemservices.util.process_recipe_tools.get_processing_info")
@mock.patch("cryoemservices.util.process_recipe_tools.get_image_directory")
def test_ispyb_filter(
    mock_image_directory,
    mock_processing_info,
    mock_engine,
    mock_sessionmaker,
    mock_ispyb_api,
    tmp_path,
):
    """Test the filter returns the expected parameters and message"""
    config = ServiceConfig(
        rabbitmq_credentials=tmp_path / "rmq_creds",
        recipe_directory=tmp_path / "recipes",
        ispyb_credentials=tmp_path / "ispyb.cfg",
    )

    sample_message = {
        "parameters": {
            "ispyb_process": "dummy_process",
            "guid": "id_string",
        },
        "recipe": Recipe(),
    }

    mock_image_directory.return_value = "/facility/microscope/data/year/visit/raw"
    mock_processing_info.return_value = {
        "recipe": "example",
        "ispyb_dcid": 10,
        "ispyb_reprocessing_parameters": {
            "parameter_a": "A",
            "parameter_b": "B",
        },
    }

    output_message, output_parameters = process_recipe_tools.ispyb_filter(
        message=sample_message, parameters=sample_message["parameters"], config=config
    )

    # Check the sqlalchemy calls
    mock_ispyb_api.url.assert_called_with(credentials=tmp_path / "ispyb.cfg")
    mock_engine.assert_called()
    mock_sessionmaker.assert_called()
    mock_sessionmaker()().__enter__.assert_called()

    # Check the calls for the mocked functions
    mock_processing_info.assert_called_with("dummy_process", mock.ANY)
    mock_image_directory.assert_called_with(10, mock.ANY)

    # Check the outputs
    assert list(output_message.keys()) == ["parameters", "recipe", "recipes"]
    assert output_message["recipes"] == ["ispyb-example"]

    assert list(output_parameters.keys()) == [
        "ispyb_process",
        "guid",
        "recipe",
        "ispyb_dcid",
        "ispyb_reprocessing_parameters",
        "ispyb_beamline",
        "ispyb_image_directory",
        "ispyb_visit_directory",
        "ispyb_working_directory",
        "ispyb_results_directory",
    ]

    assert output_parameters["recipe"] == "example"
    assert output_parameters["ispyb_dcid"] == 10
    assert output_parameters["ispyb_reprocessing_parameters"] == {
        "parameter_a": "A",
        "parameter_b": "B",
    }
    assert (
        output_parameters["ispyb_image_directory"]
        == "/facility/microscope/data/year/visit/raw"
    )
    assert output_parameters["ispyb_beamline"] == "microscope"
    assert (
        output_parameters["ispyb_visit_directory"]
        == "/facility/microscope/data/year/visit"
    )
    assert (
        output_parameters["ispyb_working_directory"]
        == "/facility/microscope/data/year/visit/tmp/wrapper/raw/id_string"
    )
    assert (
        output_parameters["ispyb_results_directory"]
        == "/facility/microscope/data/year/visit/processed/raw/id_string"
    )
