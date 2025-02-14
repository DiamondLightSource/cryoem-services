from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from queue import Empty, Queue

import requests
from pydantic import BaseModel
from workflows.transport.pika_transport import PikaTransport

from cryoemservices.util.config import config_from_file

dlq_dump_path = Path("./DLQ")


class QueueInfo(BaseModel):
    name: str
    vhost: str
    messages: int


class RabbitMQAPI:
    def __init__(self, url: str, user: str, password: str):
        self._url = url
        self._session = requests.Session()
        self._session.auth = (user, password)

    def queues(self) -> list[QueueInfo]:
        response = self._session.get(f"{self._url}/queues")
        return [QueueInfo(**qi) for qi in response.json()]


def check_dlq_rabbitmq(rabbitmq_credentials: Path) -> dict:
    if not rabbitmq_credentials.is_file():
        return {}

    with open(rabbitmq_credentials) as fp:
        rabbitmq_vals = fp.read().split("\n")
    rabbitmq_connection = {}
    for rmq_val in rabbitmq_vals:
        if ": " in rmq_val:
            split_rmq = rmq_val.split(": ")
            rabbitmq_connection[split_rmq[0]] = split_rmq[1]
    print("Connecting to:", rabbitmq_connection["base_url"])

    rmq = RabbitMQAPI(
        url=rabbitmq_connection["base_url"],
        user=rabbitmq_connection["username"],
        password=rabbitmq_connection["password"],
    )
    return {
        q.name: q.messages
        for q in rmq.queues()
        if q.name.startswith("dlq.")
        and q.vhost == rabbitmq_connection["vhost"]
        and q.messages
    }


def dlq_purge(queue: str, rabbitmq_credentials: Path) -> list[Path]:
    transport = PikaTransport()
    transport.load_configuration_file(rabbitmq_credentials)
    transport.connect()

    queue_to_purge = "dlq." + queue
    idlequeue: Queue = Queue()
    exported_messages = []

    def receive_dlq_message(header: dict, message: dict) -> None:
        idlequeue.put_nowait("start")
        header["x-death"][0]["time"] = datetime.timestamp(header["x-death"][0]["time"])

        timestamp = time.localtime(int(header["x-death"][0]["time"]))
        filepath = dlq_dump_path / time.strftime("%Y-%m-%d", timestamp)
        filepath.mkdir(parents=True, exist_ok=True)
        filename = filepath / (
            f"{queue}-"
            + time.strftime("%Y%m%d-%H%M%S", timestamp)
            + "-"
            + str(header["message-id"])
        )

        dlqmsg = {
            "exported": {
                "date": time.strftime("%Y-%m-%d"),
                "time": time.strftime("%H:%M:%S"),
            },
            "header": header,
            "message": message,
        }

        with filename.open("w") as fh:
            json.dump(dlqmsg, fh, indent=2, sort_keys=True)
        print(f"Message {header['message-id']} exported to {filename}")
        exported_messages.append(filename)
        transport.ack(header)
        idlequeue.put_nowait("done")

    print("Looking for DLQ messages in " + queue_to_purge)
    transport.subscribe(
        queue_to_purge,
        partial(receive_dlq_message),
        acknowledgement=True,
    )
    try:
        idlequeue.get(True, 3)
        while True:
            idlequeue.get(True, 0.1)
    except Empty:
        print("Done.")
    transport.disconnect()
    return exported_messages


def dlq_reinject(
    messages_path: list[Path],
    wait_time: float,
    rabbitmq_credentials: Path,
    remove: bool,
):
    transport = PikaTransport()
    transport.load_configuration_file(rabbitmq_credentials)
    transport.connect()

    for f, dlqfile in enumerate(messages_path):
        if not f == 0:
            time.sleep(wait_time)

        if not Path(dlqfile).is_file():
            print(f"Ignoring missing file {dlqfile}")
            continue
        with open(dlqfile) as fh:
            dlqmsg = json.load(fh)
        print(f"Parsing message from {dlqfile}")
        if (
            not isinstance(dlqmsg, dict)
            or not dlqmsg.get("header")
            or not dlqmsg.get("message")
        ):
            print(f"File {dlqfile} is not a valid DLQ message.")
            continue

        header = dlqmsg["header"]
        header["dlq-reinjected"] = "True"

        drop_keys = {
            "message-id",
            "routing_key",
            "redelivered",
            "exchange",
            "consumer_tag",
            "delivery_mode",
        }
        clean_header = {k: str(v) for k, v in header.items() if k not in drop_keys}

        destination = header.get("x-death", [{}])[0].get("queue")
        transport.send(
            destination,
            dlqmsg["message"],
            headers=clean_header,
        )
        if remove:
            dlqfile.unlink()
        print(f"Done {dlqfile}\n")

    transport.disconnect()


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Manage rejected messages for rabbitmq. Will check for messages, then optionally purge and reinject them."
    )
    parser.add_argument(
        "-c",
        "--config_file",
        action="store",
        required=True,
        help="Config file for cryoem-services",
    )
    parser.add_argument(
        "-q",
        "--queue",
        required=False,
        default="",
        help="Queue to purge of rejected messages. Do not include the dlq. prefix",
    )
    parser.add_argument(
        "--reinject",
        action="store_true",
        default=False,
        help="Reinject purged messages to rabbitmq?",
    )
    parser.add_argument(
        "-m",
        "--messages",
        nargs="*",
        required=False,
        help="Path pattern to extra messages to be reinjected",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        default=False,
        help="Remove message files after reinjection?",
    )
    parser.add_argument(
        "-w",
        "--wait",
        default=0,
        dest="wait",
        help="Wait this many seconds between reinjections",
    )
    parser.add_argument(
        "--skip_checks",
        action="store_true",
        default=False,
        help="Skip checking the status of the dead letter queues",
    )
    args = parser.parse_args()

    if Path(args.config_file).is_file():
        service_config = config_from_file(Path(args.config_file))
    else:
        exit(f"Cannot find config file {args.config_file}")

    if not args.skip_checks:
        dlq_checks = check_dlq_rabbitmq(service_config.rabbitmq_credentials)
        for queue, count in dlq_checks.items():
            print(f"{queue} contains {count} entries")
        total = sum(dlq_checks.values())
        if total:
            print(f"Total of {total} DLQ messages found")
        else:
            print("No DLQ messages found")

    if args.queue:
        exported_messages = dlq_purge(args.queue, service_config.rabbitmq_credentials)

        if args.reinject:
            dlq_reinject(
                exported_messages,
                float(args.wait),
                service_config.rabbitmq_credentials,
                args.remove,
            )

    if args.messages:
        if len(args.messages) == 1:
            extra_messages = list(Path(".").glob(args.messages[0]))
        else:
            extra_messages = [Path(msg) for msg in args.messages]
        dlq_reinject(
            extra_messages,
            float(args.wait),
            service_config.rabbitmq_credentials,
            args.remove,
        )

    if args.remove:
        for date_directory in dlq_dump_path.glob("*"):
            try:
                date_directory.rmdir()
            except OSError:
                print(f"Cannnot remove {date_directory} as it is not empty")
        try:
            dlq_dump_path.rmdir()
        except OSError:
            print(f"Cannnot remove {dlq_dump_path} as it is not empty")
