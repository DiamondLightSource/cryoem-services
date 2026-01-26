import os
from pathlib import Path
from typing import Optional

import mrcfile
import numpy as np
import tensorflow as tf
from easymode.core import config as easymode_config
from easymode.core.distribution import get_model, load_model
from easymode.segmentation.inference import segment_tomogram
from pydantic import BaseModel, Field, ValidationError
from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW
from cryoemservices.util.relion_service_options import RelionServiceOptions


class EasymodeParameters(BaseModel):
    tomogram: str = Field(..., min_length=1)
    output_dir: str = Field(..., min_length=1)
    segmentation_apng: str = Field(..., min_length=1)
    membrain_segmentation: Optional[str] = None
    feature_list: list[str] = ["ribosome", "microtubule", "tric"]
    mask: Optional[str] = None  # "void"
    pixel_size: Optional[float] = None
    batch_size: int = 1
    tta: int = 1
    display_binning: int = 4
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
        if self._environment["extra_config"]:
            easymode_config.edit_setting(
                "MODEL_DIRECTORY", self._environment["extra_config"]
            )
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

        try:
            with mrcfile.open(easymode_params.tomogram) as mrc:
                tomogram_header = mrc.header
        except (FileNotFoundError, ValueError):
            self.log.error(f"Input tomogram {easymode_params.tomogram} cannot be read")
            rw.transport.nack(header)
            return

        output_tomograms: dict[str, Path] = {}
        ispyb_tomograms: dict[str, Path] = {}
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
            segmented_volume, volume_apix = segment_tomogram(
                model=model,
                tomogram_path=easymode_params.tomogram,
                tta=easymode_params.tta,
                batch_size=easymode_params.batch_size,
                model_apix=model_metadata["apix"],
                input_apix=easymode_params.pixel_size,
            )

            # Convert to int8 and save mrc
            segmented_volume = (segmented_volume * 127).astype(np.int8)
            with mrcfile.new(output_tomograms[feature], overwrite=True) as mrc:
                mrc.set_data(segmented_volume)
                # Set header of output tomogram equal to that of input
                mrc.header.cella = tomogram_header.cella
                mrc.header.mx = tomogram_header.mx
                mrc.header.my = tomogram_header.my
                mrc.header.mz = tomogram_header.mz

            # Generate binned mrc images of segmented features
            if feature != easymode_params.mask:
                ispyb_tomograms[feature] = generate_binned_mrc(
                    output_tomograms[feature], easymode_params.display_binning
                )

        # Forward results to images service
        self.log.info(f"Sending to images service {easymode_params.segmentation_apng}")
        full_segmentation_list = (
            [easymode_params.membrain_segmentation]
            if easymode_params.membrain_segmentation
            else []
        )
        full_segmentation_list += [
            str(output_tomograms[feat])
            for feat in output_tomograms.keys()
            if feat != easymode_params.mask
        ]
        rw.send_to(
            "images",
            {
                "image_command": "mrc_to_apng_colour",
                "file_list": full_segmentation_list,
                "outfile": easymode_params.segmentation_apng,
                "mask": str(output_tomograms[easymode_params.mask])
                if easymode_params.mask
                else None,
            },
        )

        # Forward binned tomograms to ispyb
        if easymode_params.membrain_segmentation:
            mini_membrain_mrc = generate_binned_mrc(
                Path(easymode_params.membrain_segmentation),
                easymode_params.display_binning,
            )
            rw.send_to(
                "ispyb",
                {
                    "ispyb_command": "insert_processed_tomogram",
                    "file_path": str(mini_membrain_mrc),
                    "processing_type": "Feature",
                    "feature_type": "membrane",
                },
            )
        for feature, tomogram in ispyb_tomograms.items():
            rw.send_to(
                "ispyb",
                {
                    "ispyb_command": "insert_processed_tomogram",
                    "file_path": str(tomogram),
                    "processing_type": "Feature",
                    "feature_type": feature,
                },
            )

        self.log.info(f"Finished segmenting {easymode_params.tomogram}")
        rw.transport.ack(header)


def generate_binned_mrc(input_path: Path, binning: int) -> Path:
    """Produce binned mrc files for display purposes"""
    with mrcfile.open(input_path) as mrc:
        input_header = mrc.header
        input_data = mrc.data

    # Bin the data and set values to a range of 0-127
    input_size = np.array(input_data.shape)
    output_size = (input_size / binning).astype("int")
    if not np.sum(input_size / output_size) == binning * 3:
        reduction = abs(output_size * binning - input_size)
        input_data = input_data[reduction[0] :, reduction[1] :, reduction[2] :]
    reshaped_data = input_data.reshape(
        output_size[0], binning, output_size[1], binning, output_size[2], binning
    )
    binned_data = reshaped_data.mean(5).mean(3).mean(1)
    binned_data -= binned_data.min()
    binned_data *= 127 / binned_data.max()

    # Edge clip all directions as segmentations often have edge artifacts
    binned_data[-5:] = 0
    binned_data[:5] = 0
    binned_data[:, :5] = 0
    binned_data[:, -5:] = 0
    binned_data[:, :, :5] = 0
    binned_data[:, :, -5:] = 0

    # Save output binned mrc
    mini_mrc_name = str(input_path.with_suffix("")) + f"_bin{binning}.mrc"
    with mrcfile.new(mini_mrc_name, overwrite=True) as mrc:
        mrc.set_data(binned_data.astype("int8"))
        mrc.header.cella = input_header.cella
    return Path(mini_mrc_name)
