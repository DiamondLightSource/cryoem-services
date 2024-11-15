from __future__ import annotations

import os
import sys
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import process_recipe


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("workflows.recipe.RecipeWrapper")
@mock.patch("workflows.recipe.Recipe")
def test_process_recipe_service(mock_recipe, mock_rw, offline_transport, tmp_path):
    """
    Send a test message to CTFFind
    This should call the mock subprocess then send messages on to the
    cryolo, node_creator, ispyb_connector and images services
    """
    # Create a config file
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as cf:
        cf.write("rabbitmq_credentials: rmq_creds\n")
        cf.write(f"recipe_directory: {tmp_path}/recipes\n")
    os.environ["CRYOEMSERVICES_CONFIG"] = str(config_file)

    # Create a recipe
    (tmp_path / "recipes").mkdir()
    with open(tmp_path / "recipes/test-recipe.json", "w") as recipe:
        recipe.write(
            '{\n"1": {\n"parameters": {\n"param_out": "{param_in}"\n}\n},\n"start": [[1, []]]\n}'
        )

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    recipe_test_message = {
        "parameters": {
            "param_in": "parameter_example",
        },
        "recipes": ["test-recipe"],
    }

    # Set up the mock service
    service = process_recipe.ProcessRecipe()
    service.transport = offline_transport
    service.start()
    service.process(None, header=header, message=recipe_test_message)

    # Check the recipe was read in
    assert mock_recipe.call_count == 2
    mock_recipe.assert_any_call(
        recipe='{\n"1": {\n"parameters": {\n"param_out": "{param_in}"\n}\n},\n"start": [[1, []]]\n}'
    )
    mock_recipe().validate.assert_called()
    mock_recipe().merge.assert_called_with(mock_recipe())
    mock_recipe().merge().apply_parameters.assert_called_with(
        {
            "param_in": "parameter_example",
            "guid": mock.ANY,
        }
    )

    # Check the wrapper was started
    mock_rw.assert_called_with(
        recipe=mock_recipe().merge(), transport=offline_transport
    )
    mock_rw().start.assert_called()
