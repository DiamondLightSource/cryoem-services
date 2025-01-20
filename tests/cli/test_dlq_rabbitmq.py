from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from cryoemservices.cli import dlq_rabbitmq


@pytest.fixture
def rabbitmq_credentials(tmp_path):
    rabbitmq_file = tmp_path / "rabbitmq-credentials.yaml"
    with open(rabbitmq_file, "w") as rmq_creds:
        rmq_creds.write(
            "[rabbit]\n"
            "host: 0.0.0.0\n"
            "port: 5672\n"
            "base_url: http://rabbitmq-dummy.com/api\n"
            "username: dummy-user\n"
            "password: dummy-pass\n"
            "vhost: host\n"
        )
    return rabbitmq_file


@pytest.fixture
def config_file(rabbitmq_credentials, tmp_path):
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as cf:
        cf.write(
            f"rabbitmq_credentials: {tmp_path}/rabbitmq-credentials.yaml\n"
            f"recipe_directory: {tmp_path}\n"
        )
    return config_file


@mock.patch("cryoemservices.cli.dlq_rabbitmq.RabbitMQAPI")
def test_check_rabbitmq_dlq(mock_rmq_api, rabbitmq_credentials, tmp_path):
    mock_queue = mock.Mock()
    mock_queue.name = "dlq.dummy"
    mock_queue.vhost = "host"
    mock_queue.messages = 2
    mock_rmq_api().queues.return_value = [mock_queue]

    queue_info = dlq_rabbitmq.check_dlq_rabbitmq(rabbitmq_credentials)
    assert len(queue_info.keys()) == 1
    assert "dlq.dummy" in queue_info.keys()
    assert queue_info["dlq.dummy"] == 2


def test_check_rabbitmq_dlq_fails_no_creds(tmp_path):
    queue_info = dlq_rabbitmq.check_dlq_rabbitmq(tmp_path / "not_a_file")
    assert len(queue_info.keys()) == 0


@mock.patch("cryoemservices.cli.dlq_rabbitmq.PikaTransport")
@mock.patch("cryoemservices.cli.dlq_rabbitmq.Queue")
def bad_test_dlq_purge(mock_queue, mock_transport, tmp_path):
    mock_queue.get.return_value = False

    dlq_rabbitmq.dlq_purge("dummy", tmp_path / "config_file")
    mock_transport.assert_called_once()
    mock_transport().load_configuration_file.assert_called_with(
        tmp_path / "config_file"
    )
    mock_transport().connect.assert_called_once()
    mock_transport().subscribe.assert_called_with(
        "dlq.dummy",
        mock.ANY,
        acknowledgement=True,
    )
    mock_transport().disconnect.assert_called_once()

    mock_queue.assert_called_once()
    mock_queue().get.assert_any_call(True, 3)
    mock_queue().get.assert_called_with(True, 0.1)


@mock.patch("cryoemservices.cli.dlq_rabbitmq.time")
@mock.patch("cryoemservices.cli.dlq_rabbitmq.PikaTransport")
def test_dlq_reinject(mock_transport, mock_time, tmp_path):
    """Reinject some example messages, skipping anything invalid"""
    # Create two sample messages, and two invalid messages
    msg1_info = {
        "header": {
            "x-death": [{"queue": "queue_msg1"}],
            "message-id": 1,
            "routing_key": "dlq.queue_msg1",
            "redelivered": True,
            "exchange": "",
            "consumer_tag": "1",
            "delivery_mode": 2,
            "other_key": "value",
        },
        "message": {"parameters": "msg1"},
    }
    with open(tmp_path / "msg1", "w") as msgs1_file:
        json.dump(msg1_info, msgs1_file)
    msg2_info = {
        "header": {"x-death": [{"queue": "queue_msg2"}]},
        "message": {"content": "msg2"},
    }
    with open(tmp_path / "msg2", "w") as msgs2_file:
        json.dump(msg2_info, msgs2_file)
    msg3_info = {
        "header": {},
    }  # Won't send, no message
    with open(tmp_path / "msg3", "w") as msgs3_file:
        json.dump(msg3_info, msgs3_file)
    msg4_info = {"message": {}}  # Won't send, no header
    with open(tmp_path / "msg4", "w") as msgs4_file:
        json.dump(msg4_info, msgs4_file)

    # Send the four messages, plus a file that is not a message
    dlq_rabbitmq.dlq_reinject(
        messages_path=[
            tmp_path / "msg1",
            tmp_path / "msg2",
            tmp_path / "msg3",
            tmp_path / "msg4",
            tmp_path / "not_a_message",
        ],
        wait_time=1,
        rabbitmq_credentials=tmp_path / "config_file",
        remove=True,
    )

    mock_transport.assert_called_once()
    mock_transport().load_configuration_file.assert_called_with(
        tmp_path / "config_file"
    )
    mock_transport().connect.assert_called_once()

    # Only two messages should have been sent, the rest are invalid so are skipped
    assert mock_transport().send.call_count == 2
    mock_transport().send.assert_any_call(
        "queue_msg1",
        {"parameters": "msg1"},
        headers={
            "x-death": "[{'queue': 'queue_msg1'}]",
            "other_key": "value",
            "dlq-reinjected": "True",
        },
    )
    mock_transport().send.assert_any_call(
        "queue_msg2",
        {"content": "msg2"},
        headers={"x-death": "[{'queue': 'queue_msg2'}]", "dlq-reinjected": "True"},
    )

    # Removal and waiting
    assert not (tmp_path / "msg1").is_file()
    assert not (tmp_path / "msg2").is_file()
    assert mock_time.sleep.call_count == 2
    mock_time.sleep.assert_called_with(1)
    mock_transport().disconnect.assert_called_once()


@mock.patch("cryoemservices.cli.dlq_rabbitmq.RabbitMQAPI")
def test_rabbitmq_dlq_check(mock_rmq_api, capsys, config_file, tmp_path):
    mock_queue = mock.Mock()
    mock_queue.name = "dlq.dummy"
    mock_queue.vhost = "host"
    mock_queue.messages = 2
    mock_rmq_api().queues.return_value = [mock_queue]

    sys.argv = [
        "cryoemservices.dlq_rabbitmq",
        "--config_file",
        str(config_file),
    ]
    dlq_rabbitmq.run()

    mock_rmq_api.assert_called_with(
        url="http://rabbitmq-dummy.com/api",
        user="dummy-user",
        password="dummy-pass",
    )
    mock_rmq_api().queues.assert_called_once()
    captured = capsys.readouterr()
    assert captured.out == (
        "Connecting to: http://rabbitmq-dummy.com/api\n"
        "dlq.dummy contains 2 entries\n"
        "Total of 2 DLQ messages found\n"
    )


def no_test_rabbitmq_dlq_purge_only(config_file, tmp_path):
    sys.argv = [
        "cryoemservices.dlq_rabbitmq",
        "--config_file",
        str(config_file),
        "--queue",
        "dummy",
    ]
    dlq_rabbitmq.run()


def no_test_rabbitmq_dlq_purge_reinject(config_file, tmp_path):
    sys.argv = [
        "cryoemservices.dlq_rabbitmq",
        "--config_file",
        str(config_file),
        "--queue",
        "dummy",
        "--reinject",
    ]
    dlq_rabbitmq.run()


def no_test_rabbitmq_dlq_purge_reinject_remove(config_file, tmp_path):
    sys.argv = [
        "cryoemservices.dlq_rabbitmq",
        "--config_file",
        str(config_file),
        "--queue",
        "dummy",
        "--reinject",
        "--remove",
        "--wait",
        "1",
    ]
    dlq_rabbitmq.run()


@mock.patch("cryoemservices.cli.dlq_rabbitmq.dlq_reinject")
@mock.patch("cryoemservices.cli.dlq_rabbitmq.check_dlq_rabbitmq")
def test_rabbitmq_dlq_reinject_extras(mock_check, mock_reinject, config_file, tmp_path):
    """Test that the messages glob happens as expected"""
    os.chdir(tmp_path)
    (tmp_path / "DLQ").mkdir()
    for i in range(4):
        (tmp_path / f"DLQ/msg0{i}").touch()

    sys.argv = [
        "cryoemservices.dlq_rabbitmq",
        "--config_file",
        str(config_file),
        "--messages",
        "DLQ/msg*",
    ]
    dlq_rabbitmq.run()

    # DLQ checks are always run
    mock_check.assert_called_once()

    # The provided messages should have been reinjected
    mock_reinject.assert_called_once()
    mock_reinject.assert_called_with(
        [Path("DLQ/msg00"), Path("DLQ/msg01"), Path("DLQ/msg02"), Path("DLQ/msg03")],
        0,
        tmp_path / "rabbitmq-credentials.yaml",
        False,
    )
