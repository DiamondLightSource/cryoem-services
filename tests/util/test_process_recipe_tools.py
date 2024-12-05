from __future__ import annotations

from unittest import mock

from workflows.recipe import Recipe

from cryoemservices.util import process_recipe_tools
from cryoemservices.util.config import ServiceConfig


@mock.patch("cryoemservices.util.process_recipe_tools.models")
@mock.patch("cryoemservices.util.process_recipe_tools.sessionmaker")
@mock.patch("cryoemservices.util.process_recipe_tools.create_engine")
@mock.patch("cryoemservices.util.process_recipe_tools.get_processing_info")
@mock.patch("cryoemservices.util.process_recipe_tools.get_dc_info")
def test_ispyb_filter(
    mock_dc_info,
    mock_processing_info,
    mock_engine,
    mock_sessionmaker,
    mock_ispyb_api,
    tmp_path,
):

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

    mock_dc_info.return_value = {
        "imageDirectory": "/facility/microscope/data/year/visit/raw",
        "fileTemplate": "GridSquare_*/Data/*.tiff",
    }
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

    mock_ispyb_api.url.assert_called_with(credentials=tmp_path / "ispyb.cfg")
    mock_engine.assert_called()
    mock_sessionmaker.assert_called()
    mock_sessionmaker()().__enter__.assert_called()

    mock_processing_info.assert_called_with("dummy_process", mock.ANY)
    mock_dc_info.assert_called_with(10, mock.ANY)

    assert list(output_message.keys()) == ["parameters", "recipe", "recipes"]
    assert output_message["recipes"] == ["ispyb-example"]

    assert list(output_parameters.keys()) == [
        "ispyb_process",
        "guid",
        "recipe",
        "ispyb_dcid",
        "ispyb_reprocessing_parameters",
        "ispyb_beamline",
        "ispyb_dc_info",
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
    assert output_parameters["ispyb_dc_info"] == {
        "imageDirectory": "/facility/microscope/data/year/visit/raw",
        "fileTemplate": "GridSquare_*/Data/*.tiff",
    }
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
