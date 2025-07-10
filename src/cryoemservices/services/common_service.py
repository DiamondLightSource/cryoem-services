from __future__ import annotations

import logging
import queue
from pathlib import Path
from typing import Optional

from workflows.transport.pika_transport import PikaTransport


class CommonService:
    """
    Base class for services.
    """

    # Logger name
    _logger_name = "cryoemservices.service"

    def initializing(self):
        """Service initialization for individual service setup and subscriptions"""
        self.log.warning("Initializing is not implemented for the common service")
        pass

    def __init__(self, environment: dict, rabbitmq_credentials: Path):
        self._environment: dict = environment
        self._rabbitmq_credentials: Path = rabbitmq_credentials
        self._transport: Optional[PikaTransport] = None
        self._queue: queue.Queue = queue.PriorityQueue()
        self.log = logging.getLogger(self._logger_name)
        self.log.setLevel(logging.INFO)

    def _transport_factory(self):
        transport_type = PikaTransport()
        transport_type.load_configuration_file(self._rabbitmq_credentials)
        return transport_type

    def get_new_transport(self):
        new_transport = self._transport_factory()
        new_transport.connect()
        new_transport.subscription_callback_set_intercept(self._transport_interceptor)
        return new_transport

    def send_with_new_connection(self, destination_queue: str, message_to_send: dict):
        self.log.info(f"Sending to {destination_queue}")
        new_transport = self.get_new_transport()
        new_transport.send(destination_queue, message_to_send)
        new_transport.disconnect()

    def _transport_interceptor(self, callback):
        """Takes a callback function and adds headers and messages"""

        def add_item_to_queue(header, message):
            queue_item = (callback, header, message)
            self._queue.put(queue_item)

        return add_item_to_queue

    def start(self):
        """Start listening and process commands in main loop"""
        try:
            # Setup
            self._transport = self.get_new_transport()
            self.initializing()

            # Main loop
            while self._transport.is_connected():
                try:
                    callback, header, message = self._queue.get(True, 2)
                except queue.Empty:
                    continue
                callback(header, message)
        except Exception as e:
            self.log.critical(f"Unhandled service exception: {e}", exc_info=True)
        try:
            self._transport.disconnect()
        except Exception as e:
            self.log.error(f"Could not disconnect transport: {e}", exc_info=True)
