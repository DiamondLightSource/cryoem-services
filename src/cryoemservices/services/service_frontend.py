from __future__ import annotations

import logging
import multiprocessing
import threading
from pathlib import Path
from typing import Optional

from workflows import Disconnected
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
        """Connect to the transport layer and start service.
        :param service:
            Name of the class to be instantiated in a subprocess as service.
        :param rabbitmq_credentials:
            Path to rabbitmq credentials.
        :param environment:
            An optional dictionary that is passed to started services.
        """
        self.__lock = threading.RLock()
        self._service_name = service
        self.log = logging.getLogger("workflows.frontend")

        # Create Transport factory using given rabbitmq credentials and connect to it
        def transport_factory():
            transport_type = PikaTransport()
            transport_type.load_configuration_file(rabbitmq_credentials)
            return transport_type

        self._transport_factory = transport_factory
        self._transport = transport_factory()
        self._transport.connect()

        # Start service
        self._service, self._pipe_service = self.setup_service(
            new_service=service, service_environment=environment
        )

    def run(self):
        """The main loop of the frontend"""
        try:
            while True:
                self._check_for_service_messages()
                if not self._transport.is_connected():
                    raise Disconnected("Lost transport layer connection")
        finally:
            with self.__lock:
                self._service.terminate()
                self._service.join()
            self._transport.disconnect()
            self.log.info(f"Terminated service {self._service_name}")

    def _check_for_service_messages(self):
        """Check for incoming messages from the service"""
        if self._pipe_service and self._pipe_service.poll(1):
            try:
                message = self._pipe_service.recv()
                self.log.info(message.getMessage())
            except EOFError:
                raise Disconnected("Service died unexpectedly")

    def setup_service(self, new_service: str, service_environment: Optional[dict]):
        """Start a new service in a subprocess"""
        with self.__lock:
            # Find service class if necessary
            service_factory = lookup_services(new_service)
            if not service_factory:
                self.log.error(f"Cannot start service {new_service}")
                return False

            # Set up pipes and connect a new service object
            pipe_service, svc_tofrontend = multiprocessing.Pipe(False)
            service_instance = service_factory(
                environment=service_environment,
                transport=self._transport_factory(),
                frontend_pipe=svc_tofrontend,
            )

            # Start new service in a separate process
            started_service = multiprocessing.Process(
                target=service_instance.start,
                args=(),
            )
            started_service.start()

            # At this point the passed pipe objects must be closed
            svc_tofrontend.close()
        self.log.info(f"Started service {self._service_name}")
        return started_service, pipe_service
