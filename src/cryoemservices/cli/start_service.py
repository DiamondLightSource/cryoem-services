from __future__ import annotations

import argparse
import logging

import graypy
import workflows.frontend
from workflows.services import get_known_services
from workflows.transport.pika_transport import PikaTransport

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

    # Create Transport factory using given rabbitmq credentials
    def transport_factory():
        transport_type = PikaTransport()
        transport_type.load_configuration_file(service_config.rabbitmq_credentials)
        return transport_type

    frontend_args: dict = {
        "service": args.service,
        "transport": transport_factory,
        "transport_command_channel": "command",
        "verbose_service": True,
        "environment": {
            "config": args.config_file,
            "slurm_cluster": args.slurm,
            "queue": args.queue,
        },
    }

    # Create and start workflows Frontend object
    log.info(f"Launching service {args.service}")
    frontend = workflows.frontend.Frontend(**frontend_args)
    try:
        frontend.run()
    except KeyboardInterrupt:
        log.info("\nShutdown via Ctrl+C")
