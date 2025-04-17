from __future__ import annotations

import argparse
import logging

import graypy
from workflows.services import get_known_services

from cryoemservices.services.service_frontend import ServiceFrontend
from cryoemservices.util.config import config_from_file


def run():
    # Enumerate all known services
    known_services = sorted(get_known_services())

    # Set up parser
    parser = argparse.ArgumentParser(usage="cryoemservices.service [options]")
    parser.add_argument(
        "-s",
        "--service",
        required=True,
        choices=list(known_services),
        help=f"Name of the service to start. Known services: {', '.join(known_services)}",
    )
    parser.add_argument(
        "-c",
        "--config_file",
        required=True,
        help="Config file specifying the location of other credentials to read",
    )
    parser.add_argument(
        "--slurm",
        required=False,
        default="default",
        help="Optional slurm cluster name, matching a key present in the config file",
    )
    parser.add_argument(
        "--queue",
        required=False,
        default="",
        help="Optional override for the default queue used by the service",
    )
    args = parser.parse_args()

    service_config = config_from_file(args.config_file)

    # Initialize logging
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    if service_config.graylog_host:
        graylog_handler = graypy.GELFUDPHandler(
            service_config.graylog_host, service_config.graylog_port, level_names=True
        )
        logging.getLogger().addHandler(graylog_handler)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("pika").setLevel(logging.WARN)
    log = logging.getLogger("cryoemservices.service")

    # Create and start workflows Frontend object
    log.info(f"Launching service {args.service}")
    frontend = ServiceFrontend(
        service=args.service,
        rabbitmq_credentials=service_config.rabbitmq_credentials,
        environment={
            "config": args.config_file,
            "slurm_cluster": args.slurm,
            "queue": args.queue,
        },
    )
    try:
        frontend.run()
    except KeyboardInterrupt:
        log.info("\nShutdown via Ctrl+C")
