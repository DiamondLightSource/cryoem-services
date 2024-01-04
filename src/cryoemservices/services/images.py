from __future__ import annotations

import time
from typing import Any, Callable, Dict, NamedTuple, Protocol

import workflows.recipe
from importlib_metadata import entry_points
from workflows.services.common_service import CommonService


class _CallableParameter(Protocol):
    def __call__(self, key: str, default: Any = ...) -> Any:
        ...


class PluginInterface(NamedTuple):
    rw: workflows.recipe.wrapper.RecipeWrapper
    parameters: _CallableParameter
    message: Dict[str, Any]


class Images(CommonService):
    """
    A service that generates images and thumbnails.
    Plugin functions can be registered under the entry point
    'cryoemservices.services.images.plugins'. The contract is that a plugin function
    takes a single argument of type PluginInterface, and returns a truthy value
    to acknowledge success, and a falsy value to reject the related message.
    If a falsy value is returned that is not False then, additionally, an error
    is logged.
    Functions may choose to return a list of files that were generated, but
    this is optional at this time.
    """

    # Human readable service name
    _service_name = "Images"

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
        workflows.recipe.wrap_subscribe(
            self._transport,
            "images",
            self.image_call,
            acknowledgement=True,
            log_extender=self.extend_log,
        )

    def image_call(self, rw, header, message):
        """Pass incoming message to the relevant plugin function."""

        def parameters(key: str, default=None):
            if isinstance(message, dict) and message.get(key):
                return message[key]
            return rw.recipe_step.get("parameters", {}).get(key, default)

        command = parameters("image_command")
        if command not in self.image_functions:
            self.log.error(f"Unknown command: {command!r}")
            rw.transport.nack(header)
            return

        start = time.perf_counter()
        try:
            result = self.image_functions[command](
                PluginInterface(rw, parameters, message)
            )
        except (PermissionError, FileNotFoundError) as e:
            self.log.error(f"Command {command!r} raised {e}", exc_info=True)
            rw.transport.nack(header)
            return
        runtime = time.perf_counter() - start

        if result:
            self.log.info(f"Command {command!r} completed in {runtime:.1f} seconds")
            rw.transport.ack(header)
        elif result is False:
            # The assumption here is that if a function returns explicit
            # 'False' then it has already taken care of logging, so we
            # don't need yet another log record.
            rw.transport.nack(header)
        else:
            self.log.error(
                f"Command {command!r} returned {result!r} after {runtime:.1f} seconds"
            )
            rw.transport.nack(header)
