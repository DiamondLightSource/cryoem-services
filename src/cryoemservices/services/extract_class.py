from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import mrcfile
import numpy as np
import workflows.recipe
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm
from workflows.services.common_service import CommonService

from cryoemservices.util.spa_relion_service_options import (
    RelionServiceOptions,
    update_relion_options,
)


class ExtractClassParameters(BaseModel):
    micrographs_file: str = Field(..., min_length=1)
    class3d_dir: str = Field(..., min_length=1)
    refine_job_dir: str = Field(..., min_length=1)
    refine_class_nr: int
    boxsize: int
    pixel_size: float
    downscaled_pixel_size: float
    nr_iter_3d: int = 20
    bg_radius: int = -1
    norm: bool = True
    invert_contrast: bool = True
    relion_options: RelionServiceOptions


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

        # Select the particles from the requested class
        select_job_dir = project_dir / f"Select/job{job_num_refine - 2:03}"
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
        extract_job_dir = project_dir / f"Extract/job{job_num_refine - 1:03}"
        extract_job_dir.mkdir(parents=True, exist_ok=True)
        self.log.info(f"Running {self.extract_job_type} in {extract_job_dir}")

        refine_extraction_link = Path(
            project_dir / f"Extract/Reextract_class{extract_params.refine_class_nr}"
        )
        refine_extraction_link.unlink(missing_ok=True)
        refine_extraction_link.symlink_to(f"job{job_num_refine - 1:03}")

        # If no background radius set diameter as 75% of box
        if extract_params.bg_radius == -1:
            extract_params.bg_radius = round(0.375 * extract_params.boxsize)

        # Modify the extraction star file to contain reextracted values
        mrcs_dict = {}
        with open(f"{select_job_dir}/particles.star", "r") as selected_particles, open(
            extract_job_dir / "particles.star", "w"
        ) as extracted_particles, open(
            extract_job_dir / "extractpick.star", "w"
        ) as micrograph_list:
            micrograph_list.write(
                "data_coordinate_files\n\nloop_"
                "_rlnMicrographName #1\n_rlnMicrographCoordinates #2\n"
            )
            while True:
                line = selected_particles.readline()
                if not line:
                    break
                if line.startswith("opticsGroup"):
                    # Optics table change pixel size #7 and image size #8
                    split_line = line.split()
                    split_line[6] = str(extract_params.pixel_size)
                    split_line[7] = str(extract_params.boxsize)
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
                        ".+/Extract/job00./",
                        str(extract_job_dir.relative_to(project_dir)),
                        mrcs_name,
                    )
                    if mrcs_dict.get(mrcs_name):
                        mrcs_dict[mrcs_name]["counter"] += 1
                        mrcs_dict[mrcs_name]["x"].append(float(split_line[0]))
                        mrcs_dict[mrcs_name]["y"].append(float(split_line[1]))
                    else:
                        mrcs_dict[mrcs_name]["counter"] = 0
                        mrcs_dict[mrcs_name]["motioncorr_name"] = split_line[3]
                        mrcs_dict[mrcs_name]["reextract_name"] = reextract_name
                        mrcs_dict[mrcs_name]["x"] = [float(split_line[0])]
                        mrcs_dict[mrcs_name]["y"] = [float(split_line[1])]
                        micrograph_list.write(f"{split_line[3]} {reextract_name}\n")

                    split_line[
                        2
                    ] = f"{mrcs_dict[mrcs_name]['counter']:06}@{reextract_name}"
                    line = " ".join(split_line)
                extracted_particles.write(line)

        # Extraction for each micrograph
        for mrcs_name in tqdm(mrcs_dict.keys()):
            motioncorr_name = mrcs_dict[mrcs_name]["motioncorr_name"]
            reextract_name = mrcs_dict[mrcs_name]["reextract_name"]
            with mrcfile.open(original_dir / motioncorr_name) as input_micrograph:
                input_micrograph_image = np.array(
                    input_micrograph.data, dtype=np.float32
                )
            image_size = np.shape(input_micrograph_image)
            output_mrc_stack = []

            for particle in range(len(mrcs_dict[mrcs_name]["x"])):
                # Pixel locations are from bottom left, need to flip the image later
                pixel_location_x = round(float(mrcs_dict[mrcs_name]["x"][particle]))
                pixel_location_y = round(float(mrcs_dict[mrcs_name]["y"][particle]))

                # Extract the particle image and pad the edges if it is not square
                x_left_pad = 0
                x_right_pad = 0
                y_top_pad = 0
                y_bot_pad = 0

                extract_width = round(extract_params.relion_options.boxsize / 2)

                x_left = pixel_location_x - extract_width
                if x_left < 0:
                    x_left_pad = -x_left
                    x_left = 0
                x_right = pixel_location_x + extract_width
                if x_right >= image_size[1]:
                    x_right_pad = x_right - image_size[1]
                    x_right = image_size[1]
                y_top = pixel_location_y - extract_width
                if y_top < 0:
                    y_top_pad = -y_top
                    y_top = 0
                y_bot = pixel_location_y + extract_width
                if y_bot >= image_size[0]:
                    y_bot_pad = y_bot - image_size[0]
                    y_bot = image_size[0]

                particle_subimage = input_micrograph_image[y_top:y_bot, x_left:x_right]
                particle_subimage = np.pad(
                    particle_subimage,
                    ((y_bot_pad, y_top_pad), (x_left_pad, x_right_pad)),
                    mode="edge",
                )

                # Flip all the values on inversion
                if extract_params.invert_contrast:
                    particle_subimage = -1 * particle_subimage

                # Distance of each pixel from the centre, compared to background radius
                grid_indexes = np.meshgrid(
                    np.arange(2 * extract_width),
                    np.arange(2 * extract_width),
                )
                distance_from_centre = np.sqrt(
                    (grid_indexes[0] - extract_width + 0.5) ** 2
                    + (grid_indexes[1] - extract_width + 0.5) ** 2
                )
                bg_region = (
                    distance_from_centre
                    > np.ones(np.shape(particle_subimage)) * extract_params.bg_radius
                )

                # Fit background to a plane and subtract the plane from the image
                positions = [grid_indexes[0][bg_region], grid_indexes[1][bg_region]]
                # needs to create a matrix of the correct shape for  a*x + b*y + c plane fit
                if not len(positions[0]) == len(positions[1]):
                    self.log.warning(
                        f"Particle image {particle} in {extract_params.micrographs_file} is not square"
                    )
                    continue
                data_size = len(positions[0])
                positions_matrix = np.hstack(
                    (
                        np.reshape(positions[0], (data_size, 1)),
                        np.reshape(positions[1], (data_size, 1)),
                    )
                )
                # this ones for c
                positions_matrix = np.hstack(
                    (np.ones((data_size, 1)), positions_matrix)
                )
                values = particle_subimage[bg_region]
                # normal equation
                try:
                    theta = np.dot(
                        np.dot(
                            np.linalg.inv(
                                np.dot(positions_matrix.transpose(), positions_matrix)
                            ),
                            positions_matrix.transpose(),
                        ),
                        values,
                    )
                except np.linalg.LinAlgError:
                    self.log.warning(
                        f"Could not fit image plane for particle {particle} in {extract_params.micrographs_file}"
                    )
                    continue
                # now we need the full grid across the image
                positions_matrix = np.hstack(
                    (
                        np.reshape(grid_indexes[0], (4 * extract_width**2, 1)),
                        np.reshape(grid_indexes[1], (4 * extract_width**2, 1)),
                    )
                )
                positions_matrix = np.hstack(
                    (np.ones((4 * extract_width**2, 1)), positions_matrix)
                )
                plane = np.reshape(
                    np.dot(positions_matrix, theta),
                    (2 * extract_width, 2 * extract_width),
                )

                particle_subimage -= plane

                # Background normalisation
                if extract_params.norm:
                    # Standardise the values using the background
                    bg_mean = np.mean(particle_subimage[bg_region])
                    bg_std = np.std(particle_subimage[bg_region])
                    particle_subimage = (particle_subimage - bg_mean) / bg_std

                # Add to output stack
                if len(output_mrc_stack):
                    output_mrc_stack = np.append(
                        output_mrc_stack, [particle_subimage], axis=0
                    )
                else:
                    output_mrc_stack = np.array([particle_subimage], dtype=np.float32)

            # Produce the mrc file of the extracted particles
            box_len = extract_params.relion_options.boxsize
            pixel_size = extract_params.relion_options.pixel_size

            particle_count = np.shape(output_mrc_stack)[0]
            self.log.info(f"Extracted {particle_count} particles")
            if particle_count > 0:
                with mrcfile.new(str(reextract_name), overwrite=True) as mrc:
                    mrc.set_data(output_mrc_stack.astype(np.float32))
                    mrc.header.mx = box_len
                    mrc.header.my = box_len
                    mrc.header.mz = 1
                    mrc.header.cella.x = pixel_size * box_len
                    mrc.header.cella.y = pixel_size * box_len
                    mrc.header.cella.z = 1

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

        self.log.info("Running class reference rescaling")
        rescale_command = [
            "relion_image_handler",
            "--i",
            str(class_reference),
            "--o",
            str(rescaled_class_reference),
            "--angpix",
            str(extract_params.downscaled_pixel_size),
            "--rescale_angpix",
            str(extract_params.pixel_size),
            "--new_box",
            str(extract_params.boxsize),
        ]
        rescale_result = subprocess.run(
            rescale_command, cwd=str(project_dir), capture_output=True
        )

        # End here if the command failed
        if rescale_result.returncode:
            self.log.error(
                "Refinement reference scaling failed with exitcode "
                f"{rescale_result.returncode}:\n"
                + rescale_result.stderr.decode("utf8", "replace")
            )
            return False

        # Send on to the refinement wrapper
        refine_params = {
            "refine_job_dir": extract_params.refine_job_dir,
            "particles_file": f"{extract_job_dir}/particles.star",
            "rescaled_class_reference": str(rescaled_class_reference),
            "is_first_refinement": True,
            "number_of_particles": number_of_particles,
            "batch_size": number_of_particles,
            "pixel_size": extract_params.pixel_size,
            "class_number": extract_params.refine_class_nr,
        }
        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="refine_wrapper",
                message={"parameters": refine_params, "content": "dummy"},
            )
        else:
            rw.send_to("refine_wrapper", refine_params)

        self.log.info(f"Done {self.extract_job_type} for {extract_params.class3d_dir}.")
        rw.transport.ack(header)
