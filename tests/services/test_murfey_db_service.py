from __future__ import annotations

from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import murfey_db_connector


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@mock.patch("cryoemservices.services.ispyb_connector.MockRW")
@mock.patch("cryoemservices.services.murfey_db_connector.get_murfey_db_session")
@mock.patch(
    "cryoemservices.services.murfey_db_connector.murfey_db_commands.insert_movie"
)
def test_murfey_db_service_override(
    mock_command, mock_murfey_db_session, mock_rw, offline_transport, tmp_path
):
    """
    Send a test message to the murfey database connector service
    for a function with a murfey-specific version
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    ispyb_test_message = {
        "ispyb_command": "insert_movie",
    }
    mock_command.return_value = {"success": True, "return_value": "dummy_result"}

    # Set up the mock service and call it
    service = murfey_db_connector.MurfeyDBConnector(
        environment={"config": f"{tmp_path}/config.yaml", "queue": ""},
        transport=offline_transport,
    )
    service.initializing()
    service.insert_into_ispyb(None, header=header, message=ispyb_test_message)

    mock_murfey_db_session.assert_called()
    mock_murfey_db_session().__enter__.assert_called()

    mock_command.assert_called_with(
        message={
            "expiry_time": mock.ANY,
            "ispyb_command": "insert_movie",
        },
        parameters=mock.ANY,
        session=mock_murfey_db_session().__enter__(),
    )

    # Check that the correct messages were sent
    mock_rw.assert_called_with(offline_transport)
    mock_rw().set_default_channel.assert_called_with("output")
    mock_rw().send.assert_called_with({"result": "dummy_result"})


@mock.patch("cryoemservices.services.ispyb_connector.MockRW")
@mock.patch("cryoemservices.services.murfey_db_connector.get_murfey_db_session")
@mock.patch(
    "cryoemservices.services.murfey_db_connector.ispyb_commands.insert_motion_correction"
)
def test_murfey_db_service_use_ispyb_version(
    mock_command, mock_murfey_db_session, mock_rw, offline_transport, tmp_path
):
    """
    Send a test message to the murfey database connector service
    for a function that is not overridden for murfey
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    ispyb_test_message = {
        "ispyb_command": "insert_motion_correction",
    }
    mock_command.return_value = {"success": True, "return_value": "dummy_result"}

    # Set up the mock service and call it
    service = murfey_db_connector.MurfeyDBConnector(
        environment={"config": f"{tmp_path}/config.yaml", "queue": ""},
        transport=offline_transport,
    )
    service.initializing()
    service.insert_into_ispyb(None, header=header, message=ispyb_test_message)

    mock_murfey_db_session.assert_called()
    mock_murfey_db_session().__enter__.assert_called()

    mock_command.assert_called_with(
        message={
            "expiry_time": mock.ANY,
            "ispyb_command": "insert_motion_correction",
        },
        parameters=mock.ANY,
        session=mock_murfey_db_session().__enter__(),
    )

    # Check that the correct messages were sent
    mock_rw.assert_called_with(offline_transport)
    mock_rw().set_default_channel.assert_called_with("output")
    mock_rw().send.assert_called_with({"result": "dummy_result"})
