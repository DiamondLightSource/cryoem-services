from __future__ import annotations

from pydantic import ValidationError
from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW
from cryoemservices.wrappers.refine3d_wrapper import RefineParameters, run_refinement


class Refine3D(CommonService):
    """
    A service for running Relion Refine3D, which does not hold a RMQ connection open
    """

    # Logger name
    _logger_name = "cryoemservices.services.refine3d"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Refine3D service starting")
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "refine3d",
            self.refine3d,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def refine3d(self, rw, header: dict, message: dict):
        """Main function which interprets and processes received messages"""
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

        try:
            if isinstance(message, dict):
                refine_params = RefineParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                refine_params = RefineParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
                message = {}
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"Refine3D parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        # In this setup we cannot nack messages on failure, so instead check here
        if message.get("requeue", 0) >= 5:
            self.log.warning(f"Nacking requeued file {refine_params.particles_file}")
            rw.transport.nack(header)
            return False

        # Acknowledge the message and disconnect from rabbitmq
        self.log.info(
            f"Running disconnected Refine3D job for {refine_params.particles_file}"
            f"Running disconnected Refine3D job for {refine_params.particles_file}"
        )
        rw.transport.ack(header)

        # Run the refinement job
        try:
            successful_run = run_refinement(refine_params, send_to_rabbitmq=rw.send_to)
        except Exception as e:
            self.log.error(f"Failed to run refinement due to {e}", exc_info=True)
            successful_run = False
        except KeyboardInterrupt:
            rw.send_to("refine3d", message)
            raise KeyboardInterrupt
        if successful_run:
            self.log.error(
                f"Refinement job completed for {refine_params.particles_file}"
            )
        else:
            self.log.error(f"Refinement job failed for {refine_params.particles_file}")
            # Send back to the queue but mark a failure in the message
            message["requeue"] = message.get("requeue", 0) + 1
            rw.send_to("refine3d", message)
