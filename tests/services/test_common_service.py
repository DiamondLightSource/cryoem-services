from __future__ import annotations

import sys
from unittest import mock

import pytest

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
