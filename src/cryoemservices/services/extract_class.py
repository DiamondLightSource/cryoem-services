from __future__ import annotations

import math
import os
import re
from pathlib import Path

import workflows.recipe
import yaml
from pydantic import BaseModel, Field, ValidationError, validator
from workflows.services.common_service import CommonService

from cryoemservices.util.spa_relion_service_options import (
    RelionServiceOptions,
    update_relion_options,
)


class ExtractClassParameters(BaseModel):
    mode: str
    micrographs_file: str = Field(..., min_length=1)
    class3d_dir: str = Field(..., min_length=1)
    refine_job_dir: str = Field(..., min_length=1)
    refine_class_nr: int
    boxsize: int
    pixel_size: float
    extracted_pixel_size: float
    nr_iter_3d: int = 20
    bg_radius: int = -1
    downscale_factor: float = 2
    downscale: bool = True
    norm: bool = True
    invert_contrast: bool = True
    reextract_name: str = ""
    number_of_particles: int = 0
    relion_options: RelionServiceOptions

    @validator("mode")
    def is_valid_mode(cls, mode):
        if mode not in ["select", "extract"]:
            raise ValueError("Specify a mode of select or extract.")
        return mode


class ExtractClass(CommonService):
    """
    A service for extracting particles from a class for refinement
    """

    # Human readable service name
    _service_name = "ExtractClass"

    # Logger name
    _logger_name = "cryoemservices.services.extract_class"

    # Job name
    select_job_type = "relion.select.onvalue"
    extract_job_type = "relion.extract.reextract"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Class extraction service starting")
        workflows.recipe.wrap_subscribe(
            self._transport,
            "extract_class",
            self.extract_class,
            acknowledgement=True,
            log_extender=self.extend_log,
            allow_non_recipe_messages=True,
        )

    def extract_class(self, rw, header: dict, message: dict):
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
                extract_params = ExtractClassParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                extract_params = ExtractClassParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"Class extraction parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        self.log.info(
            f"Running extraction of particles for {extract_params.class3d_dir} "
            f"class {extract_params.refine_class_nr}"
        )

        # Run in the project directory
        project_dir = Path(extract_params.refine_job_dir).parent.parent
        os.chdir(project_dir)
        self.log.info(
            f"Input: {extract_params.class3d_dir}, Output: {extract_params.refine_job_dir}"
        )
        job_num_refine = int(
            re.search("/job[0-9]{3}", extract_params.refine_job_dir)[0][4:7]
        )
        original_dir = Path(extract_params.class3d_dir).parent.parent

        # Link the required files
        particles_data = (
            Path(extract_params.class3d_dir)
            / f"run_it{extract_params.nr_iter_3d:03}_data.star"
        )

        # Update the relion options
        extract_params.relion_options = update_relion_options(
            extract_params.relion_options, dict(extract_params)
        )
        if extract_params.downscale:
            exact_scaled_boxsize = (
                extract_params.boxsize / extract_params.downscale_factor
            )
            int_scaled_boxsize = int(math.ceil(exact_scaled_boxsize))
            scaled_boxsize = int_scaled_boxsize + int_scaled_boxsize % 2
            scaled_pixel_size = (
                extract_params.pixel_size * extract_params.downscale_factor
            )
        else:
            scaled_boxsize = extract_params.boxsize
            scaled_pixel_size = extract_params.pixel_size
        extract_params.relion_options.small_boxsize = scaled_boxsize
        extract_params.relion_options.pixel_size_downscaled = scaled_pixel_size

        select_job_dir = project_dir / f"Select/job{job_num_refine - 2:03}"
        extract_job_dir = project_dir / f"Extract/job{job_num_refine - 1:03}"
        if extract_params.mode == "select":
            # Select the particles from the requested class
            Path(select_job_dir).mkdir(parents=True, exist_ok=True)

            refine_selection_link = Path(
                project_dir / f"Select/Refine_class{extract_params.refine_class_nr}"
            )
            refine_selection_link.unlink(missing_ok=True)
            refine_selection_link.symlink_to(f"job{job_num_refine - 2:03}")

            self.log.info(f"Running {self.select_job_type} in {select_job_dir}")
            number_of_particles = 0
            with open(particles_data, "r") as classified_particles, open(
                f"{select_job_dir}/particles.star", "w"
            ) as selected_particles:
                while True:
                    line = classified_particles.readline()
                    if not line:
                        break
                    if line.lstrip() and line.lstrip()[0].isnumeric():
                        split_line = line.split()
                        class_number = int(split_line[19])
                        if class_number != extract_params.refine_class_nr:
                            continue
                        number_of_particles += 1
                    selected_particles.write(line)

            # Register the Selection job with the node creator
            self.log.info(f"Sending {self.select_job_type} to node creator")
            node_creator_select = {
                "job_type": self.select_job_type,
                "input_file": str(particles_data),
                "output_file": f"{select_job_dir}/particles.star",
                "relion_options": dict(extract_params.relion_options),
                "command": "",
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

            # Run re-extraction on the selected particles
            extract_job_dir.mkdir(parents=True, exist_ok=True)
            self.log.info(f"Running {self.extract_job_type} in {extract_job_dir}")

            refine_extraction_link = Path(
                project_dir / f"Extract/Reextract_class{extract_params.refine_class_nr}"
            )
            refine_extraction_link.unlink(missing_ok=True)
            refine_extraction_link.symlink_to(f"job{job_num_refine - 1:03}")

            # Modify the extraction star file to contain reextracted values
            mrcs_dict = {}
            with open(
                f"{select_job_dir}/particles.star", "r"
            ) as selected_particles, open(
                extract_job_dir / "particles.star", "w"
            ) as extracted_particles, open(
                extract_job_dir / "extractpick.star", "w"
            ) as micrograph_list:
                micrograph_list.write(
                    "data_coordinate_files\n\nloop_ \n"
                    "_rlnMicrographName #1 \n_rlnMicrographCoordinates #2 \n"
                )
                while True:
                    line = selected_particles.readline()
                    if not line:
                        break
                    if line.startswith("opticsGroup"):
                        # Optics table change pixel size #7 and image size #8
                        split_line = line.split()
                        split_line[6] = str(scaled_pixel_size)
                        split_line[7] = str(scaled_boxsize)
                        line = " ".join(split_line)
                    elif line.lstrip() and line.lstrip()[0].isnumeric():
                        # Main table change x#1, y#2, name#3, originx#18, originy#19
                        split_line = line.split()

                        coord_x = float(split_line[0])
                        coord_y = float(split_line[1])
                        centre_x = float(split_line[17])
                        centre_y = float(split_line[18])
                        split_line[0] = str(
                            coord_x - int(centre_x / extract_params.pixel_size)
                        )
                        split_line[1] = str(
                            coord_y - int(centre_y / extract_params.pixel_size)
                        )
                        split_line[17] = str(
                            centre_x - int(centre_x / extract_params.pixel_size)
                        )
                        split_line[18] = str(
                            centre_y - int(centre_y / extract_params.pixel_size)
                        )

                        # Create a dictionary of the images and their particles
                        mrcs_name = split_line[2].split("@")[1]
                        reextract_name = re.sub(
                            ".*Extract/job00./",
                            f"{extract_job_dir.relative_to(project_dir)}/",
                            mrcs_name,
                        )
                        if mrcs_dict.get(mrcs_name):
                            mrcs_dict[mrcs_name]["counter"] += 1
                            mrcs_dict[mrcs_name]["x"].append(float(split_line[0]))
                            mrcs_dict[mrcs_name]["y"].append(float(split_line[1]))
                        else:
                            mrcs_dict[mrcs_name] = {
                                "counter": 1,
                                "motioncorr_name": original_dir / split_line[3],
                                "reextract_name": reextract_name,
                                "x": [float(split_line[0])],
                                "y": [float(split_line[1])],
                            }
                            micrograph_list.write(f"{split_line[3]} {reextract_name}\n")

                        split_line[
                            2
                        ] = f"{mrcs_dict[mrcs_name]['counter']:06}@{reextract_name}"
                        line = "  ".join(split_line) + "\n"
                    extracted_particles.write(line)

            # Find the size of the full and downscaled extracted particles
            full_extract_width = round(extract_params.boxsize / 2)
            if extract_params.downscale:
                scaled_extract_width = round(scaled_boxsize / 2)
            else:
                scaled_extract_width = round(extract_params.boxsize / 2)

            with open(extract_job_dir / "done_micrographs.txt", "w") as done_mics:
                for mrcs_name in mrcs_dict.keys():
                    reextract_name = mrcs_dict[mrcs_name]["reextract_name"]
                    done_mics.write(f"{reextract_name}: 0\n")
                    extract_params = {
                        "motioncorr_name": mrcs_dict[mrcs_name]["motioncorr_name"],
                        "reextract_name": reextract_name,
                        "particles_x": mrcs_dict[mrcs_name]["x"],
                        "particles_y": mrcs_dict[mrcs_name]["y"],
                        "scaled_pixel_size": scaled_pixel_size,
                        "scaled_boxsize": scaled_boxsize,
                        "full_extract_width": full_extract_width,
                        "scaled_extract_width": scaled_extract_width,
                        "number_of_particles": number_of_particles,
                    }
                    if isinstance(rw, MockRW):
                        rw.transport.send(
                            destination="reextract",
                            message={"parameters": extract_params, "content": "dummy"},
                        )
                    else:
                        rw.send_to("reextract", extract_params)

        else:
            if (
                not extract_params.reextract_name
                and not Path(extract_params.reextract_name).is_file()
            ):
                self.log.warning(f"Unable to find file {extract_params.reextract_name}")
                rw.transport.nack(header)
                return

            with open(extract_job_dir / "done_micrographs.txt", "r") as done_mics:
                run_mics = yaml.safe_load(done_mics)

            try:
                run_mics[extract_params.reextract_name] = 1
            except KeyError:
                self.log.warning(
                    f"{extract_params.reextract_name} is not in "
                    f"{extract_job_dir}/done_micrographs.txt"
                )
                rw.transport.nack(header)
                return

            with open(extract_job_dir / "done_micrographs.txt", "w") as done_mics:
                yaml.dump(run_mics, done_mics)

            if not all(run_mics.values()):
                self.log.info(f"Added {extract_params.reextract_name}.")
                rw.transport.ack(header)
                return

            # Register the Re-extraction job with the node creator
            self.log.info(f"Sending {self.extract_job_type} to node creator")
            node_creator_extract = {
                "job_type": self.extract_job_type,
                "input_file": f"{select_job_dir}/particles.star:{extract_params.micrographs_file}",
                "output_file": f"{extract_job_dir}/particles.star",
                "relion_options": dict(extract_params.relion_options),
                "command": "",
                "stdout": "",
                "stderr": "",
                "success": True,
            }
            if isinstance(rw, MockRW):
                rw.transport.send(
                    destination="node_creator",
                    message={"parameters": node_creator_extract, "content": "dummy"},
                )
            else:
                rw.send_to("node_creator", node_creator_extract)

            # Create a reference for the refinement
            class_reference = (
                Path(extract_params.class3d_dir)
                / f"run_it{extract_params.nr_iter_3d:03}_class{extract_params.refine_class_nr:03}.mrc"
            )
            rescaled_class_reference = (
                extract_job_dir
                / f"refinement_reference_class{extract_params.refine_class_nr:03}.mrc"
            )

            # Make the scaling command but don't run it here as we don't have Relion
            self.log.info("Running class reference rescaling")
            rescaling_command = [
                "relion_image_handler",
                "--i",
                str(class_reference),
                "--o",
                str(rescaled_class_reference),
                "--angpix",
                str(extract_params.extracted_pixel_size),
                "--rescale_angpix",
                str(scaled_pixel_size),
                "--new_box",
                str(scaled_boxsize),
            ]

            # Send on to the refinement wrapper
            refine_params = {
                "refine_job_dir": extract_params.refine_job_dir,
                "particles_file": f"{extract_job_dir}/particles.star",
                "rescaling_command": rescaling_command,
                "rescaled_class_reference": str(rescaled_class_reference),
                "is_first_refinement": True,
                "number_of_particles": extract_params.number_of_particles,
                "batch_size": extract_params.number_of_particles,
                "pixel_size": str(scaled_pixel_size),
                "class_number": extract_params.refine_class_nr,
            }
            if isinstance(rw, MockRW):
                rw.transport.send(
                    destination="refine_wrapper",
                    message={"parameters": refine_params, "content": "dummy"},
                )
            else:
                rw.send_to("refine_wrapper", refine_params)

        self.log.info(
            f"Done {self.extract_job_type} {extract_params.mode} for {extract_params.class3d_dir}."
        )
        rw.transport.ack(header)
