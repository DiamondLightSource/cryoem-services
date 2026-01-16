import os
from pathlib import Path
from typing import Optional

import mrcfile
import numpy as np
import tensorflow as tf
from easymode.core.distribution import get_model, load_model
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
        mrc.header.cella = header.cella
        mrc.header.mx = header.mx
        mrc.header.my = header.my
        mrc.header.mz = header.mz


class EasymodeParameters(BaseModel):
    tomogram: str = Field(..., min_length=1)
    output_dir: str = Field(..., min_length=1)
    segmentation_apng: str = Field(..., min_length=1)
    membrain_segmention: Optional[str] = None
    feature_list: list[str] = ["ribosome", "microtubule", "tric"]
    mask: Optional[str] = "void"
    pixel_size: float
    batch_size: int = 1
    tta: int = 1
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

        output_tomograms: dict[str, Path] = {}
        if easymode_params.mask:
            easymode_params.feature_list.append(easymode_params.mask)
        for feature in easymode_params.feature_list:
            self.log.info(f"Loading model for {feature}.")
            model_path, model_metadata = get_model(feature)
            model = load_model(model_path)
            self.log.info("Model loaded")

            output_tomograms[feature] = Path(easymode_params.output_dir) / (
                Path(easymode_params.tomogram).stem + f"_easymode_{feature}.mrc"
            )
            self.log.info(f"Running for output {output_tomograms[feature]}")
            segmented_volume, volume_apix = inference.segment_tomogram(
                model=model,
                tomogram_path=easymode_params.tomogram,
                tta=easymode_params.tta,
                batch_size=easymode_params.batch_size,
                model_apix=model_metadata["apix"],
                input_apix=easymode_params.pixel_size,
            )

            # Convert to int8 and save mrc
            segmented_volume = (segmented_volume * 127).astype(np.int8)
            with mrcfile.new(output_tomograms[feature], overwrite=True) as m:
                m.set_data(segmented_volume)
            fix_header(easymode_params.tomogram, output_tomograms[feature])

        # Forward results to images service
        self.log.info(f"Sending to images service {easymode_params.segmentation_apng}")
        full_segmentation_list = (
            [easymode_params.membrain_segmentation]
            if easymode_params.membrain_segmention
            else []
        )
        full_segmentation_list += [
            str(output_tomograms[feat])
            for feat in output_tomograms.keys()
            if feat != easymode_params.mask
        ]
        rw.send_to(
            "movie",
            {
                "image_command": "mrc_to_apng_colour",
                "file_list": full_segmentation_list,
                "outfile": easymode_params.segmentation_apng,
                "mask": str(output_tomograms[easymode_params.mask])
                if easymode_params.mask
                else None,
            },
        )

        self.log.info(f"Finished segmenting {easymode_params.tomogram}")
        rw.transport.ack(header)
