from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import workflows.recipe
from gemmi import cif
from pydantic import BaseModel, Field, ValidationError
from workflows.services.common_service import CommonService

from cryoemservices.util.relion_service_options import (
    RelionServiceOptions,
    update_relion_options,
)


class PostProcessParameters(BaseModel):
    half_map: str = Field(..., min_length=1)
    mask: str = Field(..., min_length=1)
    rescaled_class_reference: str = Field(..., min_length=1)
    job_dir: str = Field(..., min_length=1)
    is_first_refinement: bool
    pixel_size: float
    number_of_particles: int
    batch_size: int
    class_number: int
    postprocess_lowres: float = 10
    symmetry: str = "C1"
    picker_id: int
    refined_grp_uuid: int
    refined_class_uuid: int
    relion_options: RelionServiceOptions


class PostProcess(CommonService):
    """
    A service for running Relion postprocessing
    """

    # Human readable service name
    _service_name = "PostProcess"

    # Logger name
    _logger_name = "cryoemservices.services.postprocess"

    # Job name
    job_type = "relion.postprocess"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Postprocessing service starting")
        workflows.recipe.wrap_subscribe(
            self._transport,
            "postprocess",
            self.postprocess,
            acknowledgement=True,
            log_extender=self.extend_log,
            allow_non_recipe_messages=True,
        )

    def postprocess(self, rw, header: dict, message: dict):
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
                postprocess_params = PostProcessParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                postprocess_params = PostProcessParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"Postprocessing parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        # Run in the project directory
        project_dir = Path(postprocess_params.job_dir).parent.parent
        os.chdir(project_dir)
        self.log.info(
            f"Input: {postprocess_params.half_map}, Output: {postprocess_params.job_dir}"
        )

        # Update the relion options
        postprocess_params.relion_options = update_relion_options(
            postprocess_params.relion_options, dict(postprocess_params)
        )

        # Use the Relion success file to determine if this is a rerun
        if (Path(postprocess_params.job_dir) / "RELION_JOB_EXIT_SUCCESS").exists():
            job_is_rerun = True
        else:
            job_is_rerun = False
            Path(postprocess_params.job_dir).mkdir(parents=True, exist_ok=True)

        # Do the post-processsing
        postprocess_command = [
            "relion_postprocess",
            "--i",
            postprocess_params.half_map,
            "--o",
            f"{postprocess_params.job_dir}/postprocess",
            "--mask",
            postprocess_params.mask,
            "--angpix",
            str(postprocess_params.pixel_size),
            "--auto_bfac",
            "--autob_lowres",
            str(postprocess_params.postprocess_lowres),
            "--pipeline_control",
            f"{postprocess_params.job_dir}/",
        ]
        postprocess_result = subprocess.run(postprocess_command, capture_output=True)
        if not job_is_rerun:
            (Path(postprocess_params.job_dir) / "RELION_JOB_EXIT_SUCCESS").unlink(
                missing_ok=True
            )

        # Register the post-processing job with the node creator
        self.log.info(f"Sending {self.job_type} to node creator")
        node_creator_params = {
            "job_type": self.job_type,
            "input_file": f"{postprocess_params.half_map}:{postprocess_params.mask}",
            "output_file": f"{postprocess_params.job_dir}/postprocess.mrc",
            "relion_options": dict(postprocess_params.relion_options),
            "command": " ".join(postprocess_command),
            "stdout": postprocess_result.stdout.decode("utf8", "replace"),
            "stderr": postprocess_result.stderr.decode("utf8", "replace"),
        }
        if postprocess_result.returncode:
            node_creator_params["success"] = False
        else:
            node_creator_params["success"] = True
        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="node_creator",
                message={"parameters": node_creator_params, "content": "dummy"},
            )
        else:
            rw.send_to("node_creator", node_creator_params)

        # End here if the command failed
        if postprocess_result.returncode:
            self.log.error(
                "Refinement post-process failed with exitcode "
                f"{postprocess_result.returncode}:\n"
                + postprocess_result.stderr.decode("utf8", "replace")
            )
            rw.transport.nack(header)
            return

        # Copy the angular distribution from Refinement
        refine_angdist = (
            Path(postprocess_params.half_map).parent / "run_class001_angdist.jpeg"
        )
        if refine_angdist.is_file():
            shutil.copy(
                refine_angdist, f"{postprocess_params.job_dir}/postprocess_angdist.jpeg"
            )

        # Get the bfactor and resolution from the postprocessing output
        # Should this be interpolated??
        postprocess_lines = postprocess_result.stdout.decode("utf8", "replace").split(
            "\n"
        )
        final_bfactor = None
        final_resolution = None
        for line in postprocess_lines:
            if "+ apply b-factor of:" in line:
                final_bfactor = float(line.split()[-1])
            elif "+ FINAL RESOLUTION:" in line:
                final_resolution = float(line.split()[-1])

        self.log.info(
            f"Final results: bfactor {final_bfactor} and resolution {final_resolution} "
            f"for {postprocess_params.number_of_particles} particles."
        )
        if not final_resolution:
            self.log.error(
                f"Unable to read bfactor and resolution in {postprocess_params.job_dir}"
            )
            rw.transport.nack(header)
            return

        # Send refinement job information to ispyb
        ispyb_parameters = []
        if postprocess_params.is_first_refinement:
            # Construct a bfactor group in the classification group table
            refined_grp_ispyb_parameters = {
                "ispyb_command": "buffer",
                "buffer_command": {
                    "ispyb_command": "insert_particle_classification_group"
                },
                "type": "3D",
                "batch_number": "1",
                "number_of_particles_per_batch": postprocess_params.number_of_particles,
                "number_of_classes_per_batch": "1",
                "symmetry": postprocess_params.symmetry,
                "particle_picker_id": postprocess_params.picker_id,
            }
            if job_is_rerun:
                # If this job overwrites another get the id for it
                refined_grp_ispyb_parameters["buffer_lookup"] = {
                    "particle_classification_group_id": postprocess_params.refined_grp_uuid,
                }
            else:
                refined_grp_ispyb_parameters[
                    "buffer_store"
                ] = postprocess_params.refined_grp_uuid
            ispyb_parameters.append(refined_grp_ispyb_parameters)

            # Send individual classes to ispyb
            try:
                class_star_file = cif.read_file(
                    f"{Path(postprocess_params.half_map).parent}/run_model.star"
                )
                classes_block = class_star_file.find_block("model_classes")
                classes_loop = classes_block.find_loop("_rlnReferenceImage").get_loop()
            except FileNotFoundError:
                self.log.error(
                    f"{Path(postprocess_params.half_map).parent}/run_model.star "
                    f"does not exist"
                )
                rw.transport.nack(header)
                return

            refined_ispyb_parameters = {
                "ispyb_command": "buffer",
                "buffer_lookup": {
                    "particle_classification_group_id": postprocess_params.refined_grp_uuid
                },
                "buffer_command": {"ispyb_command": "insert_particle_classification"},
                "class_number": postprocess_params.class_number,
                "class_image_full_path": f"{postprocess_params.job_dir}/postprocess.mrc",
                "particles_per_class": postprocess_params.number_of_particles,
                "class_distribution": 1,
                "rotation_accuracy": classes_loop[0, 2],
                "translation_accuracy": classes_loop[0, 3],
                "estimated_resolution": final_resolution,
                "selected": "1",
            }
            if job_is_rerun:
                refined_ispyb_parameters["buffer_lookup"].update(
                    {
                        "particle_classification_id": postprocess_params.refined_class_uuid,
                    }
                )
            else:
                refined_ispyb_parameters[
                    "buffer_store"
                ] = postprocess_params.refined_class_uuid

            # Add the resolution and fourier completeness if they are valid numbers
            estimated_resolution = float(classes_loop[0, 4])
            if np.isfinite(estimated_resolution):
                refined_ispyb_parameters["estimated_resolution"] = estimated_resolution
            else:
                refined_ispyb_parameters["estimated_resolution"] = 0.0
            fourier_completeness = float(classes_loop[0, 5])
            if np.isfinite(fourier_completeness):
                refined_ispyb_parameters[
                    "overall_fourier_completeness"
                ] = fourier_completeness
            else:
                refined_ispyb_parameters["overall_fourier_completeness"] = 0.0
            ispyb_parameters.append(refined_ispyb_parameters)

        bfactor_ispyb_parameters = {
            "ispyb_command": "buffer",
            "buffer_lookup": {
                "particle_classification_id": postprocess_params.refined_class_uuid
            },
            "buffer_command": {"ispyb_command": "insert_bfactor_fit"},
            "resolution": final_resolution,
            "number_of_particles": postprocess_params.number_of_particles,
            "particle_batch_size": postprocess_params.batch_size,
        }
        ispyb_parameters.append(bfactor_ispyb_parameters)

        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="ispyb_connector",
                message={
                    "parameters": {
                        "ispyb_command": "multipart_message",
                        "ispyb_command_list": ispyb_parameters,
                    },
                    "content": "dummy",
                },
            )
        else:
            rw.send_to(
                "ispyb_connector",
                {
                    "ispyb_command": "multipart_message",
                    "ispyb_command_list": ispyb_parameters,
                },
            )

        # Tell Murfey the refinement has finished
        if postprocess_params.is_first_refinement:
            murfey_postprocess_params = {
                "register": "done_refinement",
                "project_dir": str(project_dir),
                "resolution": final_resolution,
                "number_of_particles": postprocess_params.number_of_particles,
                "refined_grp_uuid": postprocess_params.refined_grp_uuid,
                "refined_class_uuid": postprocess_params.refined_class_uuid,
                "class_reference": postprocess_params.rescaled_class_reference,
                "class_number": postprocess_params.class_number,
                "mask_file": postprocess_params.mask,
                "pixel_size": postprocess_params.pixel_size,
            }
        else:
            murfey_postprocess_params = {
                "register": "done_bfactor",
                "resolution": final_resolution,
                "number_of_particles": postprocess_params.number_of_particles,
                "refined_class_uuid": postprocess_params.refined_class_uuid,
            }
        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="murfey_feedback",
                message={"parameters": murfey_postprocess_params, "content": "dummy"},
            )
        else:
            rw.send_to("murfey_feedback", murfey_postprocess_params)

        (Path(postprocess_params.job_dir) / "RELION_JOB_EXIT_SUCCESS").touch(
            exist_ok=True
        )
        self.log.info(f"Done {self.job_type} for {postprocess_params.half_map}.")
        rw.transport.ack(header)
