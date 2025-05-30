from __future__ import annotations

import re
from pathlib import Path

from gemmi import cif
from pydantic import BaseModel, Field, ValidationError
from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW
from cryoemservices.util.relion_service_options import RelionServiceOptions
from cryoemservices.util.spa_output_files import get_optics_table


class SelectParticlesParameters(BaseModel):
    input_file: str = Field(..., min_length=1)
    batch_size: int
    image_size: int
    incomplete_batch_size: int = 10000
    relion_options: RelionServiceOptions


class SelectParticles(CommonService):
    """
    A service for batching particles
    """

    # Logger name
    _logger_name = "cryoemservices.services.select_particles"

    # Job name
    job_type = "relion.select.split"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Select particles service starting")
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "select_particles",
            self.select_particles,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def select_particles(self, rw, header: dict, message: dict):
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
                select_params = SelectParticlesParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                select_params = SelectParticlesParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"Selection parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        self.log.info(f"Inputs: {select_params.input_file}")
        select_params.relion_options.batch_size = select_params.batch_size

        job_dir_search = re.search(".+/job[0-9]+/", select_params.input_file)
        job_num_search = re.search("/job[0-9]+/", select_params.input_file)
        if job_dir_search and job_num_search:
            extract_job_dir = Path(job_dir_search[0])
            select_job_num = int(job_num_search[0][4:7]) + 1
        else:
            self.log.warning(f"Invalid job directory in {select_params.input_file}")
            rw.transport.nack(header)
            return
        project_dir = extract_job_dir.parent.parent
        select_dir = project_dir / f"Select/job{select_job_num:03}"
        select_dir.mkdir(parents=True, exist_ok=True)

        extracted_parts_file = cif.read_file(select_params.input_file)
        extracted_parts_block = extracted_parts_file.sole_block()
        extracted_parts_loop = extracted_parts_block.find_loop(
            "_rlnCoordinateX"
        ).get_loop()

        current_splits = list(select_dir.glob("particles_split*.star"))
        try:
            num_new_parts = extracted_parts_loop.length()
            num_remaining_parts = extracted_parts_loop.length()
        except AttributeError:
            self.log.info("No particles found for selection")
            num_new_parts = 0
            num_remaining_parts = 0
        if current_splits:
            # If this is a continuation, find the previous split files
            last_split = 1
            for split_file in current_splits:
                split_number_search = re.search("split[0-9]+", str(split_file))
                if split_number_search is not None:
                    split_number = int(split_number_search[0][5:])
                    if split_number > last_split:
                        last_split = split_number
            select_output_file = f"{select_dir}/particles_split{last_split}.star"

            particles_cif = cif.read_file(select_output_file)
            prev_parts_block = particles_cif.find_block("particles")
            prev_parts_loop = prev_parts_block.find_loop("_rlnCoordinateX").get_loop()

            previous_batch_count = prev_parts_loop.length()
            num_prev_parts = prev_parts_loop.length()
            # While we have particles to add and the file is not full
            while num_prev_parts < select_params.batch_size and num_remaining_parts > 0:
                new_row = []
                for col in range(extracted_parts_loop.width()):
                    new_row.append(
                        extracted_parts_loop[num_new_parts - num_remaining_parts, col]
                    )
                prev_parts_loop.add_row(new_row)

                num_prev_parts += 1
                num_remaining_parts -= 1

            particles_cif.write_file(f"{select_output_file}.tmp")
            Path(f"{select_output_file}.tmp").rename(select_output_file)
        else:
            # If this is the first time we ran the job create a new particle split
            # Set this to be split zero so the while loop starts from one
            previous_batch_count = 0
            select_output_file = str(select_dir / "particles_split0.star")

        new_finished_files = []
        # If we filled the last file we need a new one for the remaining particles
        while num_remaining_parts > 0:
            new_split_search = re.search("split[0-9]+", str(select_output_file))
            if new_split_search is not None:
                new_split = int(new_split_search[0][5:]) + 1
            else:
                new_split = 1
            if new_split != 1:
                new_finished_files.append(new_split - 1)
            select_output_file = f"{select_dir}/particles_split{new_split}.star"
            new_particles_cif = get_optics_table(
                select_params.relion_options,
                particle=True,
                im_size=select_params.image_size,
            )

            new_split_block = new_particles_cif.add_new_block("particles")
            new_split_loop = new_split_block.init_loop(
                "_rln",
                [
                    "CoordinateX",
                    "CoordinateY",
                    "ImageName",
                    "MicrographName",
                    "OpticsGroup",
                    "CtfMaxResolution",
                    "CtfFigureOfMerit",
                    "DefocusU",
                    "DefocusV",
                    "DefocusAngle",
                    "CtfBfactor",
                    "CtfScalefactor",
                    "PhaseShift",
                ],
            )

            num_prev_parts = 0
            # While we have particles to add and the file is not full
            while num_prev_parts < select_params.batch_size and num_remaining_parts > 0:
                new_row = []
                for col in range(extracted_parts_loop.width()):
                    new_row.append(
                        extracted_parts_loop[num_new_parts - num_remaining_parts, col]
                    )
                new_split_loop.add_row(new_row)

                num_prev_parts += 1
                num_remaining_parts -= 1

            new_particles_cif.write_file(f"{select_output_file}.tmp")
            Path(f"{select_output_file}.tmp").rename(select_output_file)

        # Send to node creator if a new file was made or there isn't a complete batch
        if (
            select_output_file == f"{select_dir}/particles_split1.star"
            or new_finished_files
        ):
            self.log.info(f"Sending {self.job_type} to node creator")
            node_creator_params = {
                "job_type": self.job_type,
                "input_file": select_params.input_file,
                "output_file": select_output_file,
                "relion_options": dict(select_params.relion_options),
                "command": "",
                "stdout": "",
                "stderr": "",
            }
            rw.send_to("node_creator", node_creator_params)

        if select_output_file == f"{select_dir}/particles_split1.star":
            # If still on the first file then register it with murfey
            send_to_2d_classification = False
            if previous_batch_count == 0:
                # First run of this job, do it with any number of particles
                send_to_2d_classification = True
            else:
                # Re-run only if a multiple of the run size is passed
                previous_batch_multiple = (
                    previous_batch_count // select_params.incomplete_batch_size
                )
                new_batch_multiple = (
                    previous_batch_count + num_new_parts
                ) // select_params.incomplete_batch_size
                if new_batch_multiple > previous_batch_multiple:
                    send_to_2d_classification = True

            if send_to_2d_classification:
                self.log.info(
                    f"Sending incomplete batch {select_output_file} to Murfey"
                )
                class2d_params = {
                    "class2d_dir": f"{project_dir}/Class2D/job",
                    "batch_size": select_params.batch_size,
                    "particles_file": f"{select_dir}/particles_split1.star",
                }
                murfey_params = {
                    "register": "incomplete_particles_file",
                    "class2d_message": class2d_params,
                }
                rw.send_to("murfey_feedback", murfey_params)

        if new_finished_files:
            for new_split in new_finished_files:
                # Set up Class2D job parameters
                class2d_params = {
                    "class2d_dir": f"{project_dir}/Class2D/job",
                    "batch_size": select_params.batch_size,
                    "particles_file": f"{select_dir}/particles_split{new_split}.star",
                }

                # Send all newly completed files to murfey
                self.log.info(
                    f"Sending complete batch {class2d_params['particles_file']} to Murfey"
                )
                murfey_params = {
                    "register": "complete_particles_file",
                    "class2d_message": class2d_params,
                }
                rw.send_to("murfey_feedback", murfey_params)

        murfey_confirmation = {
            "register": "done_particle_selection",
        }
        rw.send_to("murfey_feedback", murfey_confirmation)

        self.log.info(f"Done {self.job_type} for {select_params.input_file}.")
        rw.transport.ack(header)
