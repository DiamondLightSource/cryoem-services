from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from queue import Queue

from workflows.transport.pika_transport import PikaTransport

from cryoemservices.util.config import config_from_file


def run() -> None:
    parser = argparse.ArgumentParser(
        usage="cryoemservices.dlq_purge [options] [queue [queue ...]]"
    )

    parser.add_argument("-?", action="help", help=argparse.SUPPRESS)
    dlq_dump_path = "./DLQ"
    parser.add_argument(
        "queue",
        required=True,
        help="Queue to purge of dead letters. Do not include the dlq. prefix",
    )
    parser.add_argument(
        "-c",
        "--config_file",
        action="store",
        required=True,
        help="Config file",
    )
    args = parser.parse_args()

    service_config = config_from_file(args.config_file)
    queue_to_purge = "dlq." + args.queue
    transport = PikaTransport()
    transport.load_configuration_file(service_config.rabbitmq_credentials)

    idlequeue: Queue = Queue()

    def receive_dlq_message(header: dict, message: dict) -> None:
        idlequeue.put_nowait("start")
        msg_time = int(datetime.timestamp(header["x-death"][0]["time"])) * 1000
        header["x-death"][0]["time"] = datetime.timestamp(header["x-death"][0]["time"])

        timestamp = time.localtime(msg_time / 1000)
        filepath = Path(dlq_dump_path, time.strftime("%Y-%m-%d", timestamp))
        filepath.mkdir(parents=True, exist_ok=True)
        filename = filepath / (
            f"{args.queue}-"
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
        print(
            f"Message {header['message-id']} ({time.strftime('%Y-%m-%d %H:%M:%S', timestamp)}) exported:\n  {filename}"
        )
        transport.ack(header)
        idlequeue.put_nowait("done")

    transport.connect()
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
    except queue_to_purge.Empty:
        print("Done.")
    transport.disconnect()
