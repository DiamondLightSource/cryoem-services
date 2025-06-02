from __future__ import annotations

import argparse
import logging
from importlib.metadata import entry_points

import graypy
from workflows.transport.pika_transport import PikaTransport

from cryoemservices.util.config import config_from_file


def run():
    # Enumerate all known services
    known_services = {
        e.name: e.load for e in entry_points(group="cryoemservices.services")
    }

    # Set up parser
    parser = argparse.ArgumentParser(usage="cryoemservices.service [options]")
    parser.add_argument(
        "-s",
        "--service",
        required=True,
        choices=sorted(known_services),
        help=f"Name of the service to start. Known services: {', '.join(sorted(known_services))}",
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
    log.info(f"Launching service {args.service}")

    # Create Transport factory using given rabbitmq credentials and connect to it
    def transport_factory():
        transport_type = PikaTransport()
        transport_type.load_configuration_file(service_config.rabbitmq_credentials)
        return transport_type

    # Start new service in a separate process
    service_factory = known_services.get(args.service)()
    service_instance = service_factory(
        environment={
            "config": args.config_file,
            "slurm_cluster": args.slurm,
            "queue": args.queue,
        },
        transport=transport_factory(),
    )
    log.info(f"Started service {args.service}")
    try:
        service_instance.start()
    except KeyboardInterrupt:
        log.info("Shutdown via Ctrl+C")
    log.info(f"Terminated service {args.service}")
