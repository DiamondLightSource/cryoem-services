from __future__ import annotations

from pydantic import ValidationError
from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW
from cryoemservices.wrappers.clem_process_raw_lifs import (
    LIFToStackParameters,
    convert_lif_to_stack,
)


class LIFToStackService(CommonService):
    """
    A service version of the LIF file-processing wrapper in the CLEM workflow
    """

    # log name
    _log_name = "cryoemservices.services.clem_process_raw_lifs"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("CLEM LIF processing service starting")
        # Subscribe service to RMQ queue
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "clem.process_raw_lifs",
            self.call_process_raw_lifs,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def call_process_raw_lifs(self, rw, header, message):
        """Pass incoming message to the relevant plugin function."""
        # Encase message in RecipeWrapper if none was provided
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
                params = LIFToStackParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                params = LIFToStackParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"LIFToStackParameters validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        # Process files and collect output
        results = convert_lif_to_stack(
            file=params.lif_file,
            root_folder=params.root_folder,
            number_of_processes=params.num_procs,
        )

        # Log error if the command fails to execute
        if not results:
            self.log.error(
                f"Failed to extract image stacks from {str(params.lif_file)!r}"
            )
            rw.transport.nack(header)
            return

        # Send each subset of output files to Murfey for registration
        for result in results:
            # Create dictionary and send it to Murfey's "feedback_callback" function
            murfey_params = {
                "register": "clem.register_lif_preprocessing_result",
                "result": result,
            }
            rw.send_to("murfey_feedback", murfey_params)
            self.log.info(
                f"Submitted {result['series_name']!r} {result['channel']!r} "
                "image stack and associated metadata to Murfey for registration"
            )

        rw.transport.ack(header)
        return
