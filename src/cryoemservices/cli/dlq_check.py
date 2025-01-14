from __future__ import annotations

import argparse
from pathlib import Path

from zocalo.util.rabbitmq import RabbitMQAPI

from cryoemservices.util.config import config_from_file


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


def run() -> None:
    parser = argparse.ArgumentParser(usage="cryoemservices.dlq_check [options]")
    parser.add_argument(
        "-c",
        "--config_file",
        action="store",
        required=True,
        help="Config file",
    )
    args = parser.parse_args()

    if Path(args.config_file).is_file():
        config = config_from_file(Path(args.config_file))
    else:
        exit(f"Cannot find config file {args.config_file}")

    dlqs = check_dlq_rabbitmq(config.rabbitmq_credentials)
    for queue, count in dlqs.items():
        print(f"{queue} contains {count} entries")

    total = sum(dlqs.values())
    if total:
        exit(f"Total of {total} DLQ messages found")
    else:
        print("No DLQ messages found")
