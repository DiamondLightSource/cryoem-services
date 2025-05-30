from __future__ import annotations

from importlib.metadata import entry_points
from typing import Callable

from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW


class Images(CommonService):
    """
    A service that generates images and thumbnails.
    Plugin functions can be registered under the entry point
    'cryoemservices.services.images.plugins'. The contract is that a plugin function
    takes a parameters callable, and returns a truthy value
    to acknowledge success, and a falsy value to reject the related message.
    """

    # Logger name
    _logger_name = "cryoemservices.services.images"

    # Dictionary to contain functions from plugins
    image_functions: dict[str, Callable]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_functions = {}

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Image service starting")
        self.image_functions.update(
            {
                e.name: e.load()
                for e in entry_points(group="cryoemservices.services.images.plugins")
            }
        )
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "images",
            self.image_call,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def image_call(self, rw, header, message):
        """Pass incoming message to the relevant plugin function."""
        if not rw:
            self.log.info("Received a simple message")
            if not isinstance(message, dict):
                self.log.error("Rejected invalid simple message")
                self._transport.nack(header)
                return

            # Create a wrapper-like object that can be passed to functions
            # as if a recipe wrapper was present.
            rw = MockRW(self._transport)
            rw.recipe_step = {"parameters": message}

        def parameters(key: str):
            if isinstance(message, dict) and message.get(key):
                return message[key]
            return rw.recipe_step.get("parameters", {}).get(key)

        command = parameters("image_command")
        if command not in self.image_functions:
            self.log.error(f"Unknown command: {command!r}")
            rw.transport.nack(header)
            return

        try:
            result = self.image_functions[command](parameters)
        except (PermissionError, FileNotFoundError) as e:
            self.log.error(f"Command {command!r} raised {e}", exc_info=True)
            rw.transport.nack(header)
            return

        if result:
            self.log.info(f"Command {command!r} completed")
            rw.transport.ack(header)
        else:
            self.log.error(f"Command {command!r} returned {result!r}")
            rw.transport.nack(header)
