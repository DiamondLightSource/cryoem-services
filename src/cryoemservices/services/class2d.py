from __future__ import annotations

from pydantic import ValidationError
from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW
from cryoemservices.wrappers.class2d_wrapper import Class2DParameters, run_class2d


class Class2D(CommonService):
    """
    A service for running Relion Class2D, which does not hold a RMQ connection open
    """

    # Logger name
    _logger_name = "cryoemservices.services.class2d"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Class2D service starting")
        self.subscription_id = wrap_subscribe(
            self._transport,
            self._environment["queue"] or "class2d",
            self.class2d,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def class2d(self, rw, header: dict, message: dict):
        """Main function which interprets and processes received messages"""
        if not rw:
            self.log.info("Received a simple message")
            if not isinstance(message, dict):
                self.log.error("Rejected invalid simple message")
                self._reject_message(header, requeue=False)
                return

            # Create a wrapper-like object that can be passed to functions
            # as if a recipe wrapper was present.
            rw = MockRW(self._transport)
            rw.recipe_step = {"parameters": message}

        try:
            if isinstance(message, dict):
                class2d_params = Class2DParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                class2d_params = Class2DParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
                message = {}
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"Class2D parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            self._reject_message(header, transport=rw.transport, requeue=False)
            return

        # Run the class2d job
        self.log.info(f"Running Class2D job for {class2d_params.particles_file}")
        try:
            successful_run = run_class2d(class2d_params, send_to_rabbitmq=rw.send_to)
        except Exception as e:
            self.log.error(f"Failed to run class2d due to {e}", exc_info=True)
            successful_run = False
        if successful_run:
            rw.transport.ack(header)
        else:
            self._reject_message(header, transport=rw.transport, requeue=True)
