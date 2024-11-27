from __future__ import annotations

import sys
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import ispyb_connector


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb_commands.insert_movie")
def test_ispyb_service_run(
    mock_command, mock_sqlalchemy, mock_ispyb_api, offline_transport, tmp_path
):
    """
    Send a test message to the ispyb service
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    ispyb_test_message = {
        "parameters": {
            "ispyb_command": "insert_movie",
        },
        "content": "dummy",
    }

    mock_command.return_value = {"success": True, "return_value": "dummy_result"}

    # Set up the mock service and call it
    service = ispyb_connector.EMISPyB()
    service.transport = offline_transport
    service.start()
    service.insert_into_ispyb(None, header=header, message=ispyb_test_message)

    mock_ispyb_api.url.assert_called()
    mock_sqlalchemy.create_engine.assert_called()
    mock_sqlalchemy.orm.sessionmaker.assert_called()
    mock_sqlalchemy.orm.sessionmaker()().__enter__.assert_called()

    mock_command.assert_called_with(
        message="dummy",
        parameters=mock.ANY,
        session=mock_sqlalchemy.orm.sessionmaker()().__enter__(),
    )

    # Check that the correct messages were sent
    offline_transport.send.assert_any_call(
        destination="output",
        message={"result": "dummy_result"},
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb_commands.insert_movie")
def test_ispyb_service_store_result(
    mock_command, mock_sqlalchemy, mock_ispyb_api, offline_transport, tmp_path
):
    """
    Send a test message to the ispyb service with a stored result
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    ispyb_test_message = {
        "parameters": {
            "ispyb_command": "insert_movie",
            "store_result": "full_result",
        },
        "content": "dummy",
    }

    mock_command.return_value = {
        "success": True,
        "return_value": {
            "store_result": "result_name",
            "store_value": "result_value",
        },
    }

    mock_rw = mock.MagicMock()
    mock_rw.recipe_step = {"parameters": ispyb_test_message["parameters"]}
    mock_rw.environment = {}

    # Set up the mock service and call it
    service = ispyb_connector.EMISPyB()
    service.transport = offline_transport
    service.start()
    service.insert_into_ispyb(rw=mock_rw, header=header, message=ispyb_test_message)

    # Check that the correct messages were sent
    mock_command.assert_called()
    mock_rw.send_to.assert_called_with(
        "output",
        {
            "result": {
                "store_result": "result_name",
                "store_value": "result_value",
            }
        },
    )

    # Check the results were stores in the environment
    assert mock_rw.environment["full_result"] == {
        "store_result": "result_name",
        "store_value": "result_value",
    }
    assert mock_rw.environment["result_name"] == "result_value"


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb_commands.insert_movie")
def test_ispyb_service_checkpoint(
    mock_command, mock_sqlalchemy, mock_ispyb_api, offline_transport, tmp_path
):
    """
    Send a test message to the ispyb service with a checkpoint return
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    ispyb_test_message = {
        "parameters": {
            "ispyb_command": "insert_movie",
        },
        "content": "dummy",
    }

    mock_command.return_value = {"checkpoint": True, "return_value": "dummy_result"}

    mock_rw = mock.MagicMock()
    mock_rw.recipe_step = {"parameters": ispyb_test_message["parameters"]}

    # Set up the mock service and call it
    service = ispyb_connector.EMISPyB()
    service.transport = offline_transport
    service.start()
    service.insert_into_ispyb(rw=mock_rw, header=header, message=ispyb_test_message)

    # Check that the correct messages were sent - this checkpoints but does not send
    mock_rw.send_to.assert_not_called()
    offline_transport.send.assert_not_called()
    mock_rw.checkpoint.assert_called_with("dummy_result")
