from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from workflows.transport.pika_transport import PikaTransport

from cryoemservices.util.config import config_from_file


def run() -> None:
    parser = argparse.ArgumentParser(
        usage="cryoemservices.dlq_reinject [options] file [file [..]]"
    )
    parser.add_argument(
        "-w",
        "--wait",
        default=None,
        dest="wait",
        help="Wait this many seconds between reinjections",
    )
    parser.add_argument(
        "-c",
        "--config_file",
        action="store",
        required=True,
        help="Config file",
    )
    parser.add_argument(
        "files", nargs="*", help="File(s) containing DLQ messages to be reinjected"
    )
    args = parser.parse_args()

    if not args.files:
        sys.exit("No DLQ message files given.")

    service_config = config_from_file(args.config_file)
    transport = PikaTransport()
    transport.load_configuration_file(service_config.rabbitmq_credentials)
    transport.connect()

    for dlqfile in args.files:
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
            sys.exit("File is not a valid DLQ message.")

        header = dlqmsg["header"]
        header["dlq-reinjected"] = "True"

        destination = header.get("x-death", [{}])[0].get("queue")
        transport.send(
            destination,
            dlqmsg["message"],
            headers=_rabbit_prepare_header(header),
        )
        print(f"Done {dlqfile}\n")
        if args.wait:
            time.sleep(float(args.wait))

    transport.disconnect()


def _rabbit_prepare_header(header: dict) -> dict:
    drop = {
        "message-id",
        "routing_key",
        "redelivered",
        "exchange",
        "consumer_tag",
        "delivery_mode",
    }
    return {k: str(v) for k, v in header.items() if k not in drop}
