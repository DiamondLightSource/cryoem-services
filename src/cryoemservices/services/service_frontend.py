from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path
from typing import Optional

from workflows.services import lookup as lookup_services
from workflows.transport.pika_transport import PikaTransport


class ServiceFrontend:
    """
    The frontend class encapsulates the actual service.
    It controls the service process and keeps the connection to the transport layer.
    """

    def __init__(
        self,
        service: str,
        rabbitmq_credentials: Path,
        environment: Optional[dict] = None,
    ):
        """Connect to the transport layer and start service"""
        self._service_name = service
        self.log = logging.getLogger("workflows.frontend")

        # Create Transport factory using given rabbitmq credentials and connect to it
        def transport_factory():
            transport_type = PikaTransport()
            transport_type.load_configuration_file(rabbitmq_credentials)
            return transport_type

        self._transport = transport_factory()
        self._transport.connect()

        # Find service class if necessary
        service_factory = lookup_services(service)
        if not service_factory:
            raise RuntimeError(f"Cannot start service {service}")

        # Start new service in a separate process
        service_instance = service_factory(
            environment=environment,
            transport=transport_factory(),
        )
        started_service = multiprocessing.Process(target=service_instance.start)
        started_service.start()
        self.log.info(f"Started service {self._service_name}")
        self._service = started_service

    def run(self):
        """The main loop of the frontend"""
        try:
            while self._service.is_alive():
                if not self._transport.is_connected():
                    raise RuntimeError("Lost transport layer connection")
        finally:
            self._service.terminate()
            self._service.join()
            self._transport.disconnect()
            self.log.info(f"Terminated service {self._service_name}")
