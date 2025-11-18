import os
from pathlib import Path
from typing import Optional

import mrcfile
import tensorflow as tf
from easymode.core.distribution import cache_model, load_model
from easymode.segmentation import inference
from pydantic import BaseModel, Field, ValidationError
from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW
from cryoemservices.util.relion_service_options import RelionServiceOptions


def fix_header(input_tomogram, output_tomogram):
    """Set header of output tomogram equal to that of input"""
    with mrcfile.open(input_tomogram) as mrc:
        header = mrc.header

    with mrcfile.open(output_tomogram, "r+") as mrc:
        mrc.header = header


class EasymodeParameters(BaseModel):
    tomogram: str = Field(..., min_length=1)
    output_dir: str = Field(..., min_length=1)
    feature: str
    pixel_size: Optional[float] = None
    relion_options: RelionServiceOptions


class Easymode(CommonService):
    """
    A service for segmenting cryoEM tomogram components using easymode
    """

    # Logger name
    _logger_name = "cryoemservices.services.easymode"

    # Job name
    job_type = "easymode.segment"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("easymode service starting")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        for device in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(device, True)
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "segmentation",
            self.easymode,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def easymode(self, rw, header: dict, message: dict):
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
                easymode_params = EasymodeParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                easymode_params = EasymodeParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"easymode parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        print(f"GPU - loading model ({easymode_params.feature}).")
        model_path = cache_model(easymode_params.feature)
        model = load_model(model_path)
        print("GPU - model loaded")

        output_tomogram = Path(easymode_params.output_dir) / (
            Path(easymode_params.tomogram).stem + "_easymode_{feature}.mrc"
        )
        print(f"Running for output {output_tomogram}")

        binning = 1
        batch_size = 1
        tta = 1
        segmented_volume = inference.segment_tomogram(
            model, output_tomogram, tta, batch_size, binning
        )
        with mrcfile.new(output_tomogram, overwrite=True) as m:
            m.set_data(segmented_volume)

        fix_header(easymode_params.tomogram, output_tomogram)
