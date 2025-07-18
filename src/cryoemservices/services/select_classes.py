from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
import starfile
from gemmi import cif
from pydantic import BaseModel, Field, ValidationError
from workflows.recipe import wrap_subscribe

from cryoemservices.pipeliner_plugins.combine_star_files import (
    combine_star_files,
    split_star_file,
)
from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW
from cryoemservices.util.relion_service_options import RelionServiceOptions


class SelectClassesParameters(BaseModel):
    input_file: str = Field(..., min_length=1)
    combine_star_job_number: int
    particle_diameter: float
    class2d_fraction_of_classes_to_remove: float = 0.9
    particles_file: str = "particles.star"
    classes_file: str = "class_averages.star"
    python_exe: str = "python"
    autoselect_min_score: float = 0
    min_particles: int = 500
    class3d_batch_size: int = 50000
    class3d_max_size: int = 200000
    class_uuids: str
    relion_options: RelionServiceOptions


class SelectClasses(CommonService):
    """
    A service for running Relion autoselection on 2D classes
    """

    # Logger name
    _logger_name = "cryoemservices.services.select_classes"

    # Job name
    job_type = "relion.select.class2dauto"

    # Values to extract for ISPyB
    previous_total_count = 0
    total_count = 0

    # Values for ISPyB lookups
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_uuids_dict = {}
        self.class_uuids_keys = []

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Select classes service starting")
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "select_classes",
            self.select_classes,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def parse_combiner_output(self, combiner_stdout: str):
        """
        Read the output logs of the star file combination
        """
        for line in combiner_stdout.split("\n"):
            if line.startswith("Adding") and "particles_all.star" in line:
                line_split = line.split()
                self.previous_total_count = int(line_split[3])

            if line.startswith("Combined"):
                line_split = line.split()
                self.total_count = int(line_split[6])

    def select_classes(self, rw, header: dict, message: dict):
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
                autoselect_params = SelectClassesParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                autoselect_params = SelectClassesParameters(
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

        # Update the relion options
        autoselect_params.relion_options.class2d_fraction_of_classes_to_remove = (
            autoselect_params.class2d_fraction_of_classes_to_remove
        )
        autoselect_params.relion_options.autoselect_min_score = (
            autoselect_params.autoselect_min_score
        )
        autoselect_params.relion_options.particle_diameter = (
            autoselect_params.particle_diameter
        )

        self.log.info(f"Inputs: {autoselect_params.input_file}")
        job_dir_search = re.search(".+/job[0-9]+", autoselect_params.input_file)
        job_num_search = re.search("/job[0-9]+", autoselect_params.input_file)
        if job_dir_search and job_num_search:
            class2d_job_dir = Path(job_dir_search[0])
            select_job_num = int(job_num_search[0][4:]) + 2
        else:
            self.log.warning(f"Invalid job directory in {autoselect_params.input_file}")
            rw.transport.nack(header)
            return
        project_dir = class2d_job_dir.parent.parent
        select_dir = project_dir / f"Select/job{select_job_num:03}"
        select_dir.mkdir(parents=True, exist_ok=True)

        autoselect_flags = {
            "particles_file": "--fn_sel_parts",
            "classes_file": "--fn_sel_classavgs",
            "python_exe": "--python",
            "min_particles": "--select_min_nr_particles",
        }
        # Create the class selection command
        autoselect_command = [
            "relion_class_ranker",
            "--opt",
            autoselect_params.input_file,
            "--o",
            f"{select_dir.relative_to(project_dir)}/",
            "--auto_select",
            "--fn_root",
            "rank",
            "--do_granularity_features",
        ]
        for k, v in autoselect_params.model_dump().items():
            if (v not in [None, ""]) and (k in autoselect_flags):
                autoselect_command.extend((autoselect_flags[k], str(v)))
        autoselect_command.extend(
            ("--pipeline_control", f"{select_dir.relative_to(project_dir)}/")
        )

        if not autoselect_params.autoselect_min_score:
            autoselect_command.extend(("--min_score", "0.0"))
        else:
            autoselect_command.extend(
                ("--min_score", str(autoselect_params.autoselect_min_score))
            )

        # Run the class selection
        autoselect_result = subprocess.run(
            autoselect_command, cwd=str(project_dir), capture_output=True
        )

        particle_data_starfile = autoselect_params.input_file.replace(
            "_optimiser", "_data"
        )
        input_particle_data = None
        if Path(particle_data_starfile).is_file():
            input_particle_data = starfile.read(particle_data_starfile)
        if not autoselect_result.returncode:
            quantile_threshold = autoselect_params.autoselect_min_score
            if not autoselect_params.autoselect_min_score:
                # If a minimum score isn't given, then work it out
                star_doc = cif.read_file(str(select_dir / "rank_model.star"))
                star_block = star_doc["model_classes"]
                class_scores = np.array(
                    star_block.find_loop("_rlnClassScore"), dtype=float
                )
                quantile_threshold = np.quantile(
                    class_scores,
                    float(autoselect_params.class2d_fraction_of_classes_to_remove),
                )

                self.log.info(f"Sending new threshold {quantile_threshold} to Murfey")
                murfey_params = {
                    "register": "save_class_selection_score",
                    "class_selection_score": quantile_threshold,
                }
                rw.send_to("murfey_feedback", murfey_params)

            if (
                input_particle_data
                and "rlnCryodannScore" in input_particle_data["particles"].columns
            ):
                model_score_data = starfile.read(select_dir / "rank_model.star")
                class_scores = list(model_score_data["rlnClassScore"])
                input_particle_data["particles"][
                    "rlnParticleScore"
                ] = input_particle_data["particles"].apply(
                    lambda r: r["rlnCryodannScore"]
                    * class_scores[r["rlnClassNumber"] - 1],
                    axis=1,
                )
                input_particle_data["particles"] = (
                    input_particle_data["particles"]
                    .sort_values("rlnParticleScore", ascending=False)
                    .head(len(input_particle_data["particles"]) // 2)
                )
                starfile.write(
                    input_particle_data,
                    select_dir / autoselect_params.particles_file,
                    overwrite=True,
                )
            elif not autoselect_params.autoselect_min_score:
                # Re-run the class selection if a score was not given
                self.log.info(
                    f"Re-running class selection with new threshold {quantile_threshold}"
                )
                autoselect_command[-1] = str(quantile_threshold)
                autoselect_result = subprocess.run(
                    autoselect_command, cwd=str(project_dir), capture_output=True
                )

        # Send class selection job to node creator
        self.log.info(f"Sending {self.job_type} to node creator")
        autoselect_node_creator_params: dict[str, Any] = {
            "job_type": self.job_type,
            "input_file": autoselect_params.input_file,
            "output_file": str(select_dir / autoselect_params.particles_file),
            "relion_options": dict(autoselect_params.relion_options),
            "command": " ".join(autoselect_command),
            "stdout": autoselect_result.stdout.decode("utf8", "replace"),
            "stderr": autoselect_result.stderr.decode("utf8", "replace"),
        }
        if autoselect_result.returncode:
            autoselect_node_creator_params["success"] = False
        else:
            autoselect_node_creator_params["success"] = True
        rw.send_to("node_creator", autoselect_node_creator_params)

        # End here if the command failed
        if autoselect_result.returncode:
            self.log.error(
                f"2D autoselection failed with exitcode {autoselect_result.returncode}:\n"
                + autoselect_result.stderr.decode("utf8", "replace")
            )
            rw.transport.nack(header)
            return

        # Find which classes were picked
        classes_block = cif.read_file(
            f"{select_dir}/{autoselect_params.classes_file}"
        ).sole_block()
        selected_classes = list(classes_block.find_loop("_rlnReferenceImage"))

        # Class ids get fed in as a string, need to convert these to a dictionary
        self.class_uuids_dict = json.loads(
            autoselect_params.class_uuids.replace("'", '"')
        )
        self.class_uuids_keys = list(self.class_uuids_dict.keys())

        # Send the picked classes to ispyb
        ispyb_command_list = []
        for picked_class in selected_classes:
            class_id = int(picked_class.split("@")[0])
            ispyb_command_list.append(
                {
                    "ispyb_command": "buffer",
                    "buffer_lookup": {
                        "particle_classification_id": self.class_uuids_dict[
                            self.class_uuids_keys[class_id - 1]
                        ],
                    },
                    "buffer_command": {
                        "ispyb_command": "insert_particle_classification"
                    },
                    "selected": 1,
                }
            )
        ispyb_parameters = {
            "ispyb_command": "multipart_message",
            "ispyb_command_list": ispyb_command_list,
        }
        self.log.info(f"Sending to ispyb {ispyb_parameters}")
        rw.send_to("ispyb_connector", ispyb_parameters)

        # Run the combine star files job to combine the files into particles_all.star
        self.log.info("Running star file combination and splitting")
        combine_star_dir = Path(
            project_dir / f"Select/job{autoselect_params.combine_star_job_number:03}"
        )
        files_to_combine = [select_dir / autoselect_params.particles_file]
        if (combine_star_dir / "particles_all.star").exists():
            files_to_combine.append(combine_star_dir / "particles_all.star")
        else:
            combine_star_dir.mkdir(parents=True, exist_ok=True)
            Path(project_dir / "Select/Best_particles").symlink_to(combine_star_dir)
            self.previous_total_count = 0

        if not (
            combine_star_dir / f".done_{autoselect_params.particles_file}"
        ).is_file():
            # Only run this if the particles file has not been added before
            combine_node_creator_params: dict[str, Any] = {
                "job_type": "combine_star_files_job",
                "input_file": f"{select_dir}/{autoselect_params.particles_file}",
                "output_file": f"{combine_star_dir}/particles_all.star",
                "relion_options": dict(autoselect_params.relion_options),
                "command": (
                    "combine_star_files "
                    + " ".join([str(i) for i in files_to_combine])
                    + f" --output_dir {combine_star_dir}"
                ),
                "stdout": "",
                "stderr": "",
                "alias": "Best_particles",
            }

            # Call the combining function and redirect prints to an io object
            combine_result = io.StringIO()
            with redirect_stdout(combine_result):
                try:
                    combine_star_files(files_to_combine, combine_star_dir)
                    (
                        combine_star_dir / f".done_{autoselect_params.particles_file}"
                    ).touch()
                    combine_node_creator_params["success"] = True
                except (IndexError, KeyError):
                    combine_node_creator_params["success"] = False
            self.parse_combiner_output(combine_result.getvalue())

            # Send combination job to node creator
            self.log.info("Sending combine_star_files_job (combine) to node creator")
            rw.send_to("node_creator", combine_node_creator_params)

            # End here if the command failed
            if not combine_node_creator_params["success"]:
                self.log.error("Star file combination failed")
                rw.transport.nack(header)
                return
        else:
            # If combination isn't run the number of particles needs to be found
            self.total_count = 0
            self.previous_total_count = 0
            with open(combine_star_dir / "particles_all.star", "r") as particles_file:
                while True:
                    particle_line = particles_file.readline()
                    if not particle_line:
                        break
                    particle_split_line = particle_line.split()
                    if (
                        len(particle_split_line) > 0
                        and particle_split_line[0][0].isdigit()
                    ):
                        self.total_count += 1
                        self.previous_total_count += 1

        # Create a file containing all selected classes
        if not (combine_star_dir / autoselect_params.classes_file).is_file():
            add_header = True
        else:
            add_header = False
        with (
            open(select_dir / autoselect_params.classes_file, "r") as class_file,
            open(
                combine_star_dir / autoselect_params.classes_file, "a"
            ) as combined_classes,
        ):
            while True:
                line = class_file.readline()
                if not line:
                    break
                if line[0].isdigit():
                    # Always add lines which list a class
                    add_header = False
                    combined_classes.write(line)
                elif add_header:
                    # Only add other lines if this is a new file and not on classes yet
                    combined_classes.write(line)

        # Determine the next split size to use and whether to run 3D classification
        send_to_3d_classification = False
        if self.previous_total_count == 0:
            # First run of this job, use class3d_max_size
            next_batch_size = autoselect_params.class3d_batch_size
            if self.total_count > autoselect_params.class3d_batch_size:
                # Do 3D classification if there are more particles than the batch size
                send_to_3d_classification = True
        elif self.previous_total_count >= autoselect_params.class3d_max_size:
            # Iterations beyond those where 3D classification is run
            next_batch_size = autoselect_params.class3d_max_size
        else:
            # Re-runs with fewer particles than the maximum
            previous_batch_multiple = (
                self.previous_total_count // autoselect_params.class3d_batch_size
            )
            new_batch_multiple = (
                self.total_count // autoselect_params.class3d_batch_size
            )
            if new_batch_multiple > previous_batch_multiple:
                # Do 3D classification if a batch threshold has been crossed
                send_to_3d_classification = True
                # Set the batch size from the total count, but do not exceed the maximum
                next_batch_size = (
                    new_batch_multiple * autoselect_params.class3d_batch_size
                )
                if next_batch_size > autoselect_params.class3d_max_size:
                    next_batch_size = autoselect_params.class3d_max_size
            else:
                # Otherwise just get the next threshold
                next_batch_size = (
                    previous_batch_multiple + 1
                ) * autoselect_params.class3d_batch_size

        # Run the combine star files job to split particles_all.star into batches
        split_node_creator_params: dict[str, Any] = {
            "job_type": "combine_star_files_job",
            "input_file": f"{select_dir}/{autoselect_params.particles_file}",
            "output_file": f"{combine_star_dir}/particles_all.star",
            "relion_options": dict(autoselect_params.relion_options),
            "command": (
                f"combine_star_files {combine_star_dir}/particles_all.star "
                f"--output_dir {combine_star_dir} "
                f"--split --split_size {next_batch_size}"
            ),
            "stdout": "",
            "stderr": "",
            "alias": "Best_particles",
        }

        # Call the combining function and redirect prints to an io object
        split_result = io.StringIO()
        with redirect_stdout(split_result):
            try:
                split_star_file(
                    file_to_process=combine_star_dir / "particles_all.star",
                    output_dir=combine_star_dir,
                    split_size=next_batch_size,
                )
                split_node_creator_params["success"] = True
            except (IndexError, KeyError):
                split_node_creator_params["success"] = False
        self.parse_combiner_output(split_result.getvalue())

        # Send splitting job to node creator
        self.log.info("Sending combine_star_files_job (split) to node creator")
        rw.send_to("node_creator", split_node_creator_params)

        # End here if the command failed
        if not split_node_creator_params["success"]:
            self.log.error("Star file splitting failed")
            rw.transport.nack(header)
            return

        # Request selected particles image from images service
        self.log.info("Sending to images service")
        files_selected_from = []
        (combine_star_dir / "Movies").mkdir(exist_ok=True)
        with open(
            select_dir / autoselect_params.particles_file, "r"
        ) as selected_particles:
            while True:
                line = selected_particles.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                if line.strip()[0].isnumeric():
                    # Second entry is particle files in the form 001@Extract/file.star
                    particle_x = line.split()[0]
                    particle_y = line.split()[1]
                    extracted_file = Path(line.split()[2].split("@")[1])
                    motioncorr_file = Path(line.split()[3])
                    if [extracted_file, motioncorr_file] not in files_selected_from:
                        # Make a list of all files
                        files_selected_from.append([extracted_file, motioncorr_file])
                    # Append any newly selected particles to a file
                    with open(
                        combine_star_dir / f"Movies/{extracted_file.stem}.star",
                        "a",
                    ) as selected_file:
                        selected_file.write(f"{particle_x} {particle_y}\n")

        original_pixel_size = None
        for extracted_file, motioncorr_file in files_selected_from:
            # Get the selected picks for each file
            extract_job_number = int(str(extracted_file).split("job")[1][:3])
            try:
                with open(
                    combine_star_dir / f"Movies/{extracted_file.stem}.star", "r"
                ) as selected_file:
                    selected_coords = [line.split() for line in selected_file]
            except FileNotFoundError:
                selected_coords = []

            if not motioncorr_file.is_relative_to(project_dir):
                motioncorr_file = project_dir / motioncorr_file

            if not original_pixel_size:
                with mrcfile.open(motioncorr_file) as mrc:
                    original_pixel_size = float(mrc.header.cella.x) / float(
                        mrc.header.mx
                    )

            # Get the name of the  picking image file
            cryolo_output_path = (
                Path(
                    re.sub(
                        "MotionCorr/job002/.+",
                        f"AutoPick/job{extract_job_number - 1:03}/STAR/",
                        str(motioncorr_file),
                    )
                )
                / motioncorr_file.with_suffix(".star").name
            )

            # Get all the picks
            try:
                with open(cryolo_output_path, "r") as coords_file:
                    coords = [line.split() for line in coords_file][6:]
            except FileNotFoundError:
                coords = []

            # Generate image of selected and non-selected picks
            rw.send_to(
                "images",
                {
                    "image_command": "picked_particles",
                    "file": str(motioncorr_file),
                    "coordinates": coords,
                    "selected_coordinates": selected_coords,
                    "pixel_size": original_pixel_size,
                    "diameter": autoselect_params.particle_diameter,
                    "outfile": str(Path(cryolo_output_path).with_suffix(".jpeg")),
                    "remove_input": False,
                    "flatten_image": True,
                },
            )

        # Create 3D classification jobs
        if send_to_3d_classification:
            # Only send to 3D if a new multiple of the batch threshold is crossed
            # and the count has not passed the maximum
            self.log.info("Sending to Murfey for Class3D")

            # Copy the particle batch file
            shutil.copy2(
                combine_star_dir / "particles_split1.star",
                combine_star_dir / f"particles_batch_{next_batch_size}.star",
            )

            # Tell Murfey to do Class3D
            class3d_params = {
                "particles_file": f"{combine_star_dir}/particles_batch_{next_batch_size}.star",
                "class3d_dir": f"{project_dir}/Class3D/job",
                "batch_size": next_batch_size,
            }
            murfey_3d_params = {
                "register": "run_class3d",
                "class3d_message": class3d_params,
            }
            rw.send_to("murfey_feedback", murfey_3d_params)

        murfey_confirmation = {
            "register": "done_class_selection",
        }
        rw.send_to("murfey_feedback", murfey_confirmation)

        # Remove the temporary hold file
        (combine_star_dir / f".done_{autoselect_params.particles_file}").unlink(
            missing_ok=True
        )

        self.log.info(f"Done {self.job_type} for {autoselect_params.input_file}.")
        rw.transport.ack(header)
