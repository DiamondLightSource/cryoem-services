from __future__ import annotations

from pydantic import ValidationError
from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW
from cryoemservices.wrappers.class3d_wrapper import Class3DParameters, run_class3d


class Class3D(CommonService):
    """
    A service for running Relion Class3D, which does not hold a RMQ connection open
    """

    # Logger name
    _logger_name = "cryoemservices.services.class3d"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Class3D service starting")
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "class3d",
            self.class3d,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def class3d(self, rw, header: dict, message: dict):
        """Main function which interprets and processes received messages"""
        if not rw:
            self.log.info("Received a simple message")
            if not isinstance(message, dict):
                self.log.error("Rejected invaid simple message")
                self._reject_message(header, requeue=False)
                return

            # Create a wrapper-like object that can be passed to functions
            # as if a recipe wrapper was present.
            rw = MockRW(self._transport)
            rw.recipe_step = {"parameters": message}

        try:
            if isinstance(message, dict):
                class3d_params = Class3DParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                class3d_params = Class3DParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
                message = {}
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"Class3D parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            self._reject_message(header, transport=rw.transport, requeue=False)
            return

        # In this setup we cannot reject messages on failure, so instead check here
        if message.get("requeue", 0) >= 5:
            self.log.warning(f"Rejecting requeued file {class3d_params.particles_file}")
            self._reject_message(header, transport=rw.transport, requeue=False)
            return

        # Acknowledge the message and disconnect from rabbitmq
        self.log.info(
            f"Running disconnected Class3D job for {class3d_params.particles_file}"
        )
        rw.transport.ack(header)

        # Run the class3d job
        try:
            successful_run = run_class3d(class3d_params, send_to_rabbitmq=rw.send_to)
        except Exception as e:
            self.log.error(f"Failed to run class3d due to {e}", exc_info=True)
            successful_run = False
        except KeyboardInterrupt:
            rw.send_to("class3d", message)
            raise KeyboardInterrupt
        if successful_run:
            self.log.error(f"Class3D job completed for {class3d_params.particles_file}")
        else:
            self.log.error(f"Class3D job failed for {class3d_params.particles_file}")
            # Send back to the queue but mark a failure in the message
            message["requeue"] = message.get("requeue", 0) + 1
            rw.send_to("class3d", message)
