from __future__ import annotations

import argparse
import logging

import workflows.frontend
from workflows.services import get_known_services
from workflows.transport.pika_transport import PikaTransport

from cryoemservices.util.config import config_from_file


def start_service():
    ServiceStarter().run(
        transport_command_channel="command",
    )


class ServiceStarter:
    """Starts a service"""

    def __init__(self):
        # Initialize logging
        self.console = logging.StreamHandler()
        self.console.setLevel(logging.INFO)
        logging.getLogger().addHandler(self.console)
        logging.getLogger().setLevel(logging.INFO)
        self.log = logging.getLogger("zocalo.service")

    def run(
        self,
        **kwargs,
    ):
        """Example command line interface to start services.
        :param cmdline_args: List of command line arguments to pass to parser
        """

        # Enumerate all known services
        known_services = sorted(get_known_services())

        # Set up parser
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-s",
            "--service",
            dest="service",
            required=True,
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

        # Check if service exists
        if args.service not in known_services:
            self.log.error(
                f"Unknown service {args.service}. Valid options are {known_services}"
            )
            return

        service_config = config_from_file(args.config_file)

        # Create Transport factory using given rabbitmq credentials
        def transport_factory():
            transport_type = PikaTransport()
            transport_type.load_configuration_file(service_config.rabbitmq_credentials)
            return transport_type

        kwargs.update(
            {
                "service": args.service,
                "transport": transport_factory,
                "verbose_service": True,
            },
        )
        kwargs.setdefault("environment", {})

        # Create Frontend object
        frontend = workflows.frontend.Frontend(**kwargs)

        # Start Frontend
        try:
            frontend.run()
        except KeyboardInterrupt:
            print("\nShutdown via Ctrl+C")
