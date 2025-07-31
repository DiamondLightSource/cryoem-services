from __future__ import annotations

from pydantic import ValidationError
from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW
from cryoemservices.wrappers.clem_process_raw_tiffs import (
    TIFFToStackParameters,
    process_tiff_files,
)


class TIFFToStackService(CommonService):
    """
    A service version of the TIFF file-processing wrapper in the CLEM workflow
    """

    # log name
    _log_name = "cryoemservices.services.clem_process_raw_tiffs"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("CLEM TIFF processing service starting")
        # Subscribe service to RMQ queue
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "clem.process_raw_tiffs",
            self.call_process_raw_tiffs,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def call_process_raw_tiffs(self, rw, header, message):
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
                params = TIFFToStackParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                params = TIFFToStackParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"TIFFToStackParameters validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        # Reconstruct series name using reference file from list
        ref_file = params.tiff_list[0]
        path_parts = list((ref_file.parent / ref_file.stem.split("--")[0]).parts)
        try:
            root_index = path_parts.index(params.root_folder)
        except ValueError:
            self.log.error(
                f"Subpath {params.root_folder!r} was not found in file path "
                f"{str(ref_file.parent / ref_file.stem.split('--')[0])!r}"
            )
            rw.transport.nack(header)
            return
        series_name = "--".join(
            [p.replace(" ", "_") if " " in p else p for p in path_parts][
                root_index + 1 :
            ]
        )

        # Process files and collect output
        result = process_tiff_files(
            tiff_list=params.tiff_list,
            root_folder=params.root_folder,
            metadata_file=params.metadata,
        )

        # Nack message and log error if the command fails to execute
        if not result:
            self.log.error(f"Process failed for TIFF series {series_name!r}")
            rw.transport.nack(header)
            return

        # Create dictionary and send it to Murfey's "feedback_callback" function
        murfey_params = {
            "register": "clem.register_tiff_preprocessing_result",
            "result": result,
        }
        rw.send_to("murfey_feedback", murfey_params)
        self.log.info(
            f"Submitted processed data for {result['series_name']!r} "
            "and associated metadata to Murfey for registration"
        )

        rw.transport.ack(header)
        return
