from __future__ import annotations

import sys
from unittest import mock

import pytest
from workflows.transport.pika_transport import PikaTransport

from cryoemservices.services import common_service


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.common_service.queue")
def test_common_service(mock_queue, tmp_path):
    """
    Start the common service to check it works
    """
    mock_transport = mock.Mock()
    mock_transport.is_connected.side_effect = [True, False]

    mock_service = mock.Mock()
    mock_queue.PriorityQueue().get.return_value = (
        mock_service,
        "test_header",
        "test_message",
    )

    # Set up the mock service and send the message to it
    service = common_service.CommonService(
        environment={"queue": ""}, transport=mock_transport
    )
    service.start()

    mock_transport.connect.assert_called()
    mock_transport.subscription_callback_set_intercept.assert_called_with(
        service._transport_interceptor
    )
    assert mock_transport.is_connected.call_count == 2
    mock_transport.disconnect.assert_called()

    mock_queue.PriorityQueue.assert_called()
    mock_queue.PriorityQueue().get.assert_called_with(True, 2)

    mock_service.assert_called_with("test_header", "test_message")


@mock.patch("cryoemservices.services.common_service.partial")
def test_reject_message(mock_partial):
    """
    Check message rejection works
    """
    # Set up mock transports for both offline and pika
    mock_transport = mock.Mock()
    mock_pika = mock.Mock(spec=PikaTransport)
    mock_thread = mock.Mock()
    mock_channel = mock.Mock()
    mock_thread._pika_channels = {2: mock_channel}
    mock_pika._pika_thread = mock_thread

    # Set up the mock service and send the message to it
    service = common_service.CommonService(
        environment={"queue": ""}, transport=mock_transport
    )
    header = {"message-id": 1, "subscription": 2}

    # Reject using the basic transport
    service._reject_message(header)
    mock_transport.nack.assert_called_once_with(header, requeue=True)

    # Reject using a pika instance
    service._reject_message(header, transport=mock_pika, requeue=False)
    mock_thread._connection.add_callback_threadsafe.assert_called_once()
    mock_partial.assert_called_once_with(
        mock_channel.basic_reject, delivery_tag=1, requeue=False
    )
