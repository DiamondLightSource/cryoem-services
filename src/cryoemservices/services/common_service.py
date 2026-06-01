from __future__ import annotations

import logging
import queue
from functools import partial

from workflows.transport.common_transport import CommonTransport
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

    def __init__(
        self,
        environment: dict,
        transport: CommonTransport,
        single_message_mode: bool = False,
    ):
        self._environment: dict = environment
        self._transport: CommonTransport = transport
        self._queue: queue.Queue = queue.PriorityQueue()
        self.log = logging.getLogger(self._logger_name)
        self.log.setLevel(logging.INFO)
        self.single_message_mode: bool = single_message_mode

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
            self._transport.connect()
            self._transport.subscription_callback_set_intercept(
                self._transport_interceptor
            )
            self.initializing()

            # Main loop
            while self._transport.is_connected():
                try:
                    callback, header, message = self._queue.get(True, 2)
                except queue.Empty:
                    continue
                callback(header, message)
                if self.single_message_mode:
                    break
        except Exception as e:
            self.log.critical(f"Unhandled service exception: {e}", exc_info=True)
        try:
            self._transport.disconnect()
        except Exception as e:
            self.log.error(f"Could not disconnect transport: {e}", exc_info=True)

    def _reject_message(
        self,
        header: dict,
        transport: CommonTransport | None = None,
        requeue: bool = True,
    ):
        """Reject failed messages back to rabbitmq"""
        message_id = header.get("message-id")
        subscription_id = header.get("subscription")
        if transport is None:
            transport = self._transport
        if (
            isinstance(transport, PikaTransport)
            and message_id is not None
            and subscription_id is not None
        ):
            pika_thread = transport._pika_thread
            channel = pika_thread._pika_channels[subscription_id]
            pika_thread._connection.add_callback_threadsafe(
                partial(channel.basic_reject, delivery_tag=message_id, requeue=requeue)
            )
        else:
            # Resort back to nacking if this isn't pika or the header is invalid
            # Mostly just for tests compatibility
            self.log.warning(
                f"Message {message_id} in {subscription_id} is not valid for rabbitmq"
            )
            transport.nack(header)
