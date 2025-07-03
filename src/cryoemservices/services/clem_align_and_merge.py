from __future__ import annotations

from pydantic import ValidationError
from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW
from cryoemservices.wrappers.clem_align_and_merge import (
    AlignAndMergeParameters,
    align_and_merge_stacks,
)


class AlignAndMergeService(CommonService):
    """
    A service version of the image alignment and merging process in the CLEM workflow
    """

    # log name
    _log_name = "cryoemservices.services.clem_align_and_merge"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("CLEM image alignment and merging service starting")
        # Subscribe service to RMQ queue
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "clem.align_and_merge",
            self.call_align_and_merge,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def call_align_and_merge(self, rw, header, message):
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
                params = AlignAndMergeParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                params = AlignAndMergeParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"AlignAndMergeParameters validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        # Process files and collect output
        results = align_and_merge_stacks(
            images=params.images,
            metadata=params.metadata,
            crop_to_n_frames=params.crop_to_n_frames,
            align_self=params.align_self,
            flatten=params.flatten,
            align_across=params.align_across,
        )

        if not results:
            self.log.error(
                "Failed to complete the aligning and merging process for "
                f"{params.series_name!r}"
            )
            rw.transport.nack(header)
            return

        # Send results to Murfey for registration
        results["series_name"] = params.series_name
        murfey_params = {
            "register": "clem.register_align_and_merge_result",
            "result": results,
        }
        rw.send_to("murfey_feedback", murfey_params)
        self.log.info(
            f"Submitted alignment and merging result for {results['series_name']!r} "
            "to Murfey for registration"
        )
        rw.transport.ack(header)
        return
