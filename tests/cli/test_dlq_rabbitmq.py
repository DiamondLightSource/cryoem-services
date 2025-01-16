from __future__ import annotations

import sys
from unittest import mock

import pytest

from cryoemservices.cli import dlq_rabbitmq


@pytest.fixture
def config_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as cf:
        cf.write(
            f"rabbitmq_credentials: {tmp_path}/rabbitmq-credentials.yaml\n"
            f"recipe_directory: {tmp_path}\n"
        )

    with open(tmp_path / "rabbitmq-credentials.yaml", "w") as rmq_creds:
        rmq_creds.write(
            "[rabbit]\n"
            "host: 0.0.0.0\n"
            "port: 5672\n"
            "base_url: http://rabbitmq-dummy.com/api\n"
            "username: dummy-user\n"
            "password: dummy-pass\n"
            "vhost: host\n"
        )
    return str(config_file)


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
        config_file,
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
        config_file,
        "--queue",
        "dummy",
    ]
    dlq_rabbitmq.run()


def no_test_rabbitmq_dlq_purge_reinject(config_file, tmp_path):
    sys.argv = [
        "cryoemservices.dlq_rabbitmq",
        "--config_file",
        config_file,
        "--queue",
        "dummy",
        "--reinject",
    ]
    dlq_rabbitmq.run()


def no_test_rabbitmq_dlq_purge_reinject_remove(config_file, tmp_path):
    sys.argv = [
        "cryoemservices.dlq_rabbitmq",
        "--config_file",
        config_file,
        "--queue",
        "dummy",
        "--reinject",
        "--remove",
        "--wait",
        "1",
    ]
    dlq_rabbitmq.run()


def no_test_rabbitmq_dlq_reinject_extras(config_file, tmp_path):
    sys.argv = [
        "cryoemservices.dlq_rabbitmq",
        "--config_file",
        config_file,
        "--messages",
        "DLQ/date/msg*",
    ]
    dlq_rabbitmq.run()
