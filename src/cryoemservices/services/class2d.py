import json
from pathlib import Path

from pydantic import ValidationError
from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import InterruptHandler, MockRW
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

        # In this setup we cannot reject messages on failure, so instead check here
        if message.get("requeue", 0) >= 5:
            self.log.warning(f"Rejecting requeued file {class2d_params.particles_file}")
            self._reject_message(header, transport=rw.transport, requeue=False)
            return

        Path(class2d_params.class2d_dir).mkdir(exist_ok=True, parents=True)
        with open(f"{class2d_params.class2d_dir}/recipe.json", "w") as recipe_file:
            json.dump(
                {"header": header, "message": class2d_params.model_dump(mode="json")},
                recipe_file,
            )

        # Acknowledge the message and disconnect from rabbitmq
        self.log.info(
            f"Running disconnected Class2D job for {class2d_params.particles_file}"
        )
        rw.transport.ack(header)
        rw.transport.unsubscribe(self.subscription_id)
        rw.transport.drop_callback_reference(self.subscription_id)

        # Run the class2d job
        with InterruptHandler() as handler:
            try:
                successful_run = run_class2d(
                    class2d_params, send_to_rabbitmq=rw.send_to
                )
            except Exception as e:
                self.log.error(f"Failed to run class2d due to {e}", exc_info=True)
                successful_run = False
            if handler.interrupted:
                self.log.warning("Process was interrupted")
                successful_run = False

            # Reconnect to rabbitmq
            self.initializing()
            if successful_run:
                self.log.info(
                    f"Class2D job completed for {class2d_params.particles_file}"
                )
            else:
                self.log.error(
                    f"Class2D job failed for {class2d_params.particles_file}"
                )
                # Send back to the queue but mark a failure in the message
                message["requeue"] = message.get("requeue", 0) + 1
                # Create a new transport object of the same type as before
                rw._transport = type(rw.transport)()
                rw.transport.connect()
                rw.transport.send("class2d", message, headers=header)

            if handler.interrupted:
                raise RuntimeError("Process was interrupted")
        return True
