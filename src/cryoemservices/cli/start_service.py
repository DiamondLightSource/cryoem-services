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
    parser = argparse.ArgumentParser(description="Start up a service")
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
        "--extra_config",
        required=False,
        default="",
        help="Optional extra config option needed by a service",
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
    parser.add_argument(
        "--single_message",
        action="store_true",
        help="Consume only a single message then shut down?",
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

    # Create the transport for rabbitmq
    transport_type = PikaTransport()
    transport_type.load_configuration_file(service_config.rabbitmq_credentials)

    # Start new service in a separate process
    service_factory = known_services.get(args.service)()
    service_instance = service_factory(
        environment={
            "config": args.config_file,
            "extra_config": args.extra_config,
            "slurm_cluster": args.slurm,
            "queue": args.queue,
        },
        transport=transport_type,
        single_message_mode=args.single_message,
    )
    log.info(f"Started service {args.service}")
    try:
        service_instance.start()
    except KeyboardInterrupt:
        log.info("Shutdown via Ctrl+C")
    log.info(f"Terminated service {args.service}")
