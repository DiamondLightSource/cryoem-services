from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from queue import Empty, Queue

from workflows.transport.pika_transport import PikaTransport

from cryoemservices.util.config import config_from_file


def dlq_purge(queue: str, rabbitmq_credentials: Path):
    transport = PikaTransport()
    transport.load_configuration_file(rabbitmq_credentials)
    transport.connect()

    queue_to_purge = "dlq." + queue
    dlq_dump_path = "./DLQ"
    idlequeue: Queue = Queue()
    exported_messages = []

    def receive_dlq_message(header: dict, message: dict) -> None:
        idlequeue.put_nowait("start")
        msg_time = int(datetime.timestamp(header["x-death"][0]["time"])) * 1000
        header["x-death"][0]["time"] = datetime.timestamp(header["x-death"][0]["time"])

        timestamp = time.localtime(msg_time / 1000)
        filepath = Path(dlq_dump_path, time.strftime("%Y-%m-%d", timestamp))
        filepath.mkdir(parents=True, exist_ok=True)
        filename = filepath / (
            f"{queue}-"
            + time.strftime("%Y%m%d-%H%M%S", timestamp)
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


def dlq_reinject(messages_path: Path, wait_time: float, rabbitmq_credentials: Path):
    transport = PikaTransport()
    transport.load_configuration_file(rabbitmq_credentials)
    transport.connect()

    for dlqfile in messages_path.glob("*"):
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
        print(f"Done {dlqfile}\n")
        if wait_time:
            time.sleep(wait_time)

    transport.disconnect()


def run() -> None:
    parser = argparse.ArgumentParser(
        usage="cryoemservices.dlq_handling [options] [queue [queue ...]]"
    )
    parser.add_argument(
        "-c",
        "--config_file",
        action="store",
        required=True,
        help="Config file",
    )
    parser.add_argument(
        "-q",
        "--queue",
        required=True,
        help="Queue to purge of dead letters. Do not include the dlq. prefix",
    )
    parser.add_argument(
        "-m",
        "--messages",
        required=False,
        help="Path pattern of extra messages to be reinjected",
    )
    parser.add_argument(
        "-w",
        "--wait",
        default=None,
        dest="wait",
        help="Wait this many seconds between reinjections",
    )
    args = parser.parse_args()

    service_config = config_from_file(args.config_file)

    exported_messages = dlq_purge(args.queue, service_config.rabbitmq_credentials)
    dlq_reinject(
        exported_messages, float(args.wait), service_config.rabbitmq_credentials
    )
