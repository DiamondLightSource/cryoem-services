from __future__ import annotations

import argparse
import logging

import workflows.frontend
from workflows.services import get_known_services
from workflows.transport.pika_transport import PikaTransport

from cryoemservices.util.config import config_from_file

# Initialize logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger("cryoemservices.service")


def start_service():
    # Enumerate all known services
    known_services = sorted(get_known_services())

    # Set up parser
    parser = argparse.ArgumentParser(usage="cryoemservices.service [options]")
    parser.add_argument(
        "-s",
        "--service",
        dest="service",
        required=True,
        choices=list(known_services),
        help="Name of the service to start. Known services: "
        + ", ".join(known_services),
    )
    parser.add_argument(
        "-c",
        "--config_file",
        action="store",
        required=True,
        help="Config file specifying the location of other credentials to read",
    )
    args = parser.parse_args()

    # Create Transport factory using given rabbitmq credentials
    service_config = config_from_file(args.config_file)

    def transport_factory():
        transport_type = PikaTransport()
        transport_type.load_configuration_file(service_config.rabbitmq_credentials)
        return transport_type

    frontend_args: dict = {
        "service": args.service,
        "transport": transport_factory,
        "transport_command_channel": "command",
        "verbose_service": True,
    }
    frontend_args.setdefault("environment", {})

    # Create and start workflows Frontend object
    frontend = workflows.frontend.Frontend(**frontend_args)
    try:
        frontend.run()
    except KeyboardInterrupt:
        print("\nShutdown via Ctrl+C")
