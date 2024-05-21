from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import workflows.recipe
from pydantic import BaseModel, Field, ValidationError
from workflows.services.common_service import CommonService

from cryoemservices.util.relion_service_options import (
    RelionServiceOptions,
    update_relion_options,
)


class BFactorParameters(BaseModel):
    bfactor_directory: str = Field(..., min_length=1)
    rescaled_class_reference: str = Field(..., min_length=1)
    class_number: int
    number_of_particles: int
    batch_size: int
    pixel_size: float
    mask: str
    relion_options: RelionServiceOptions


class BFactor(CommonService):
    """
    A service for selecting particles for a b-factor calculation run
    """

    # Human readable service name
    _service_name = "BFactor"

    # Logger name
    _logger_name = "cryoemservices.services.bfactor_setup"

    # Job name
    job_type = "relion.select.split"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("BFactor setup service starting")
        workflows.recipe.wrap_subscribe(
            self._transport,
            "bfactor",
            self.bfactor_setup,
            acknowledgement=True,
            log_extender=self.extend_log,
            allow_non_recipe_messages=True,
        )

    def bfactor_setup(self, rw, header: dict, message: dict):
        class MockRW:
            def dummy(self, *args, **kwargs):
                pass

        if not rw:
            print(
                "Incoming message is not a recipe message. Simple messages can be valid"
            )
            if (
                not isinstance(message, dict)
                or not message.get("parameters")
                or not message.get("content")
            ):
                self.log.error("Rejected invalid simple message")
                self._transport.nack(header)
                return
            self.log.debug("Received a simple message")

            # Create a wrapper-like object that can be passed to functions
            # as if a recipe wrapper was present.
            rw = MockRW()
            rw.transport = self._transport
            rw.recipe_step = {"parameters": message["parameters"]}
            rw.environment = {"has_recipe_wrapper": False}
            rw.set_default_channel = rw.dummy
            rw.send = rw.dummy
            message = message["content"]

        try:
            if isinstance(message, dict):
                bfactor_params = BFactorParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                bfactor_params = BFactorParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"B-factor setup parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        self.log.info(
            f"Setting up b-factor for {bfactor_params.rescaled_class_reference} "
            f"with {bfactor_params.number_of_particles} particles "
            f"from a batch of {bfactor_params.batch_size}."
        )

        # Update the relion options
        bfactor_params.relion_options = update_relion_options(
            bfactor_params.relion_options, dict(bfactor_params)
        )

        # Determine the directory to run in
        project_dir = Path(bfactor_params.bfactor_directory).parent.parent
        bfactor_dir = Path(bfactor_params.bfactor_directory)
        (bfactor_dir / "Import/job001").mkdir(parents=True, exist_ok=True)
        os.chdir(bfactor_dir)

        # Link the required files
        linked_class_particles = bfactor_dir / "Import/job001/particles.star"
        linked_class_particles.unlink(missing_ok=True)
        linked_class_particles.symlink_to(
            project_dir
            / f"Extract/Reextract_class{bfactor_params.class_number}/particles.star"
        )

        (bfactor_dir / "Extract").unlink(missing_ok=True)
        (bfactor_dir / "Extract").symlink_to(project_dir / "Extract")

        linked_class_reference = bfactor_dir / "Import/job001/refinement_ref.mrc"
        linked_class_reference.unlink(missing_ok=True)
        linked_class_reference.symlink_to(bfactor_params.rescaled_class_reference)

        linked_mask_file = bfactor_dir / "Import/job001/mask.mrc"
        linked_mask_file.unlink(missing_ok=True)
        linked_mask_file.symlink_to(bfactor_params.mask)

        # Split the particles file
        split_job_dir = Path("Select/job002")
        split_job_dir.mkdir(parents=True, exist_ok=True)
        split_command = [
            "relion_star_handler",
            "--i",
            str(linked_class_particles),
            "--o",
            f"{split_job_dir}/particles_split1.star",
            "--split",
            "--random_order",
            "--nr_split",
            "1",
            "--size_split",
            str(bfactor_params.number_of_particles),
            "--pipeline_control",
            f"{split_job_dir}/",
        ]

        # Don't run the Relion command, this can be done directly in python
        rng = np.random.default_rng()
        random_particle_ids = rng.choice(
            np.arange(0, bfactor_params.batch_size),
            bfactor_params.number_of_particles,
            replace=False,
        )
        particle_id = 0
        with open(linked_class_particles, "r") as particles_file, open(
            split_job_dir / "particles_split1.star", "w"
        ) as particles_split:
            while True:
                line = particles_file.readline()
                if not line:
                    break
                tidy_line = line.lstrip()
                if tidy_line and tidy_line[0].isnumeric():
                    if particle_id in random_particle_ids:
                        particles_split.write(line)
                    particle_id += 1
                else:
                    particles_split.write(line)

        # Register the Selection job with the node creator
        self.log.info(f"Sending {self.job_type} to node creator")
        node_creator_select = {
            "job_type": self.job_type,
            "input_file": str(linked_class_particles),
            "output_file": f"{bfactor_dir}/{split_job_dir}/particles_split1.star",
            "relion_options": dict(bfactor_params.relion_options),
            "command": " ".join(split_command),
            "stdout": "",
            "stderr": "",
            "success": True,
        }
        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="node_creator",
                message={"parameters": node_creator_select, "content": "dummy"},
            )
        else:
            rw.send_to("node_creator", node_creator_select)

        # Send on to the refinement wrapper
        refine_params = {
            "refine_job_dir": f"{bfactor_dir}/Refine3D/job003",
            "particles_file": f"{bfactor_dir}/{split_job_dir}/particles_split1.star",
            "rescaled_class_reference": str(linked_class_reference),
            "is_first_refinement": False,
            "number_of_particles": bfactor_params.number_of_particles,
            "batch_size": bfactor_params.batch_size,
            "pixel_size": bfactor_params.pixel_size,
            "mask": str(linked_mask_file),
            "class_number": bfactor_params.class_number,
        }
        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="refine_wrapper",
                message={"parameters": refine_params, "content": "dummy"},
            )
        else:
            rw.send_to("refine_wrapper", refine_params)

        self.log.info(f"Set up b-factor run for {bfactor_params.bfactor_directory}")
        rw.transport.ack(header)
