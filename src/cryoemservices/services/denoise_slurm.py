from __future__ import annotations

from pathlib import Path
from typing import Optional

import workflows.recipe
from pydantic import BaseModel, Field, ValidationError, validator
from workflows.services.common_service import CommonService

from cryoemservices.util.slurm_submission import slurm_submission


class DenoiseParameters(BaseModel):
    volume: str = Field(..., min_length=1)
    output: Optional[str] = None  # volume directory
    suffix: Optional[str] = None  # ".denoised"
    model: Optional[str] = None  # "unet-3d"
    even_train_path: Optional[str] = None
    odd_train_path: Optional[str] = None
    n_train: Optional[int] = None  # 1000
    n_test: Optional[int] = None  # 200
    crop: Optional[int] = None  # 96
    base_kernel_width: Optional[int] = None  # 11
    optim: Optional[str] = None  # "adagrad"
    lr: Optional[float] = None  # 0.001
    criteria: Optional[str] = None  # "L2"
    momentum: Optional[float] = None  # 0.8
    batch_size: Optional[int] = None  # 10
    num_epochs: Optional[int] = None  # 500
    weight_decay: Optional[int] = None  # 0
    save_interval: Optional[int] = None  # 10
    save_prefix: Optional[str] = None
    num_workers: Optional[int] = None  # 1
    num_threads: Optional[int] = None  # 0
    gaussian: Optional[int] = None  # 0
    patch_size: Optional[int] = None  # 96
    patch_padding: Optional[int] = None  # 48
    device: Optional[int] = None  # -2
    cleanup_output: bool = True

    @validator("model")
    def saved_models(cls, v):
        if v not in ["unet-3d-10a", "unet-3d-20a", "unet-3d"]:
            raise ValueError("Model must be one of unet-3d-10a, unet-3d-20a, unet-3d")
        return v

    @validator("optim")
    def optimizers(cls, v):
        if v not in ["adam", "adagrad", "sgd"]:
            raise ValueError("Optimizer must be one of adam, adagrad, sgd")
        return v

    @validator("criteria")
    def training_criteria(cls, v):
        if v not in ["L1", "L2"]:
            raise ValueError("Optimizer must be one of L1, L2")
        return v


class DenoiseSlurm(CommonService):
    """
    A service for denoising cryoEM tomograms using Topaz
    Submits jobs to a slurm cluster via RestAPI
    """

    # Human readable service name
    _service_name = "Denoise"

    # Logger name
    _logger_name = "cryoemservices.services.denoise_slurm"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Denoise service starting")
        workflows.recipe.wrap_subscribe(
            self._transport,
            "denoise",
            self.denoise,
            acknowledgement=True,
            log_extender=self.extend_log,
            allow_non_recipe_messages=True,
        )

    def denoise(self, rw, header: dict, message: dict):
        class MockRW:
            def dummy(self, *args, **kwargs):
                pass

        if not rw:
            if (
                not isinstance(message, dict)
                or not message.get("parameters")
                or not message.get("content")
            ):
                self.log.error("Rejected invalid simple message")
                self._transport.nack(header)
                return

            # Create a wrapper-like object that can be passed to functions
            # as if a recipe wrapper was present.
            rw = MockRW()
            rw.transport = self._transport
            rw.recipe_step = {"parameters": message["parameters"], "output": None}
            rw.environment = {"has_recipe_wrapper": False}
            rw.set_default_channel = rw.dummy
            rw.send = rw.dummy
            message = message["content"]

        try:
            if isinstance(message, dict):
                denoise_params = DenoiseParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                denoise_params = DenoiseParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"Denoise parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        command = [
            "topaz",
            "denoise3d",
            str(Path(denoise_params.volume).name),
        ]

        denoise_flags = {
            "output": "-o",
            "suffix": "--suffix",
            "model": "-m",
            "even_train_path": "-a",
            "odd_train_path": "-b",
            "n_train": "--N-train",
            "n_test": "--N-test",
            "crop": "-c",
            "base_kernel_width": "--base-kernel-width",
            "optim": "--optim",
            "lr": "--lr",
            "criteria": "--criteria",
            "momentum": "--momentum",
            "batch_size": "--batch-size",
            "num_epochs": "--num-epochs",
            "weight_decay": "-w",
            "save_interval": "--save-interval",
            "num_workers": "--num-workers",
            "save_prefix": "--save-prefix",
            "num_threads": "-j",
            "gaussian": "-g",
            "patch_size": "-s",
            "patch_padding": "-p",
            "device": "-d",
        }
        for k, v in denoise_params.dict().items():
            if v and (k in denoise_flags):
                command.extend((denoise_flags[k], str(v)))

        suffix = str(Path(denoise_params.volume).suffix)
        alignment_output_dir = Path(denoise_params.volume).parent
        denoised_file = str(Path(denoise_params.volume).stem) + ".denoised" + suffix
        denoised_full_path = Path(denoise_params.volume).parent / denoised_file

        self.log.info(f"Input: {denoise_params.volume} Output: {denoised_full_path}")
        self.log.info(f"Running Topaz {command}")

        # Submit the command to slurm
        slurm_outcome = slurm_submission(
            log=self.log,
            job_name="Denoising",
            command=command,
            project_dir=alignment_output_dir,
            output_file=denoised_full_path,
            cpus=1,
            use_gpu=True,
            use_singularity=False,
            script_extras="module load EM/topaz",
        )

        # Stop here if the job failed
        if slurm_outcome.returncode:
            self.log.error("Denoising failed to run")
            rw.transport.nack(header)
            return

        # Clean up the slurm files
        if denoise_params.cleanup_output:
            Path(f"{denoised_full_path}.out").unlink()
            Path(f"{denoised_full_path}.err").unlink()
            Path(f"{denoised_full_path}.json").unlink()

        # Forward results to images service
        self.log.info(f"Sending to images service {denoise_params.volume}")
        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="images",
                message={
                    "image_command": "mrc_central_slice",
                    "file": str(denoised_full_path),
                },
            )
            rw.transport.send(
                destination="movie",
                message={
                    "image_command": "mrc_to_apng",
                    "file": str(denoised_full_path),
                },
            )
        else:
            rw.send_to(
                "images",
                {
                    "image_command": "mrc_central_slice",
                    "file": str(denoised_full_path),
                },
            )
            rw.send_to(
                "movie",
                {
                    "image_command": "mrc_to_apng",
                    "file": str(denoised_full_path),
                },
            )

        # Send to segmentation
        self.log.info(f"Sending {denoised_full_path} for segmentation")
        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="segmentation",
                message={
                    "tomogram": str(denoised_full_path),
                },
            )
        else:
            rw.send_to(
                "segmentation",
                {
                    "tomogram": str(denoised_full_path),
                },
            )

        self.log.info(f"Done denoising for {denoise_params.volume}")
        rw.transport.ack(header)
        return
