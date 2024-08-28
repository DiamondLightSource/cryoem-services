from __future__ import annotations

import datetime
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import workflows.recipe
from pipeliner.api.api_utils import (
    edit_jobstar,
    job_default_parameters_dict,
    write_default_jobstar,
)
from pipeliner.api.manage_project import PipelinerProject
from pipeliner.data_structure import FAIL_FILE, SUCCESS_FILE
from pipeliner.job_factory import read_job
from pipeliner.project_graph import ProjectGraph
from pipeliner.utils import DirectoryBasedLock
from pydantic import BaseModel, Field, ValidationError, validator
from workflows.services.common_service import CommonService

from cryoemservices.util.relion_service_options import (
    RelionServiceOptions,
    generate_service_options,
)
from cryoemservices.util.spa_output_files import create_spa_output_files
from cryoemservices.util.tomo_output_files import create_tomo_output_files


@lru_cache(maxsize=2)
class CachedProjectGraph(ProjectGraph):
    def __enter__(self):
        if not self._lock:
            lock = DirectoryBasedLock(self.pipeline_dir / ".relion_lock")
            acquired = lock.acquire()
            if acquired:
                self._lock = lock
            else:
                raise RuntimeError("Cannot acquire lock")
        return self


# A dictionary of all the available jobs,
# the folder name they run in, and the names of their inputs in the job star
pipeline_jobs: dict[str, dict] = {
    "relion.import.movies": {"folder": "Import", "spa_input": {"fn_in_raw": "*.tiff"}},
    "relion.import.tilt_series": {
        "folder": "Import",
        "tomography_input": {"movie_files": "*.tiff", "mdoc_files": "*.mdoc"},
    },
    "relion.motioncorr.own": {
        "folder": "MotionCorr",
        "spa_input": {"input_star_mics": "movies.star"},
        "tomography_input": {"input_star_mics": "tilt_series.star"},
    },
    "relion.motioncorr.motioncor2": {
        "folder": "MotionCorr",
        "spa_input": {"input_star_mics": "movies.star"},
        "tomography_input": {"input_star_mics": "tilt_series.star"},
    },
    "icebreaker.micrograph_analysis.micrographs": {
        "folder": "IceBreaker",
        "spa_input": {"in_mics": "corrected_micrographs.star"},
    },
    "icebreaker.micrograph_analysis.enhancecontrast": {
        "folder": "IceBreaker",
        "spa_input": {"in_mics": "corrected_micrographs.star"},
    },
    "icebreaker.micrograph_analysis.summary": {
        "folder": "IceBreaker",
        "spa_input": {"in_mics": "grouped_micrographs.star"},
    },
    "relion.ctffind.ctffind4": {
        "folder": "CtfFind",
        "spa_input": {"input_star_mics": "corrected_micrographs.star"},
        "tomography_input": {"input_star_mics": "corrected_tilt_series.star"},
    },
    "cryolo.autopick": {
        "folder": "AutoPick",
        "spa_input": {"input_file": "corrected_micrographs.star"},
    },
    "relion.extract": {
        "folder": "Extract",
        "spa_input": {
            "coords_suffix": "autopick.star",
            "star_mics": "micrographs_ctf.star",
        },
    },
    "relion.select.split": {
        "folder": "Select",
        "spa_input": {"fn_data": "particles.star"},
    },
    "icebreaker.micrograph_analysis.particles": {
        "folder": "IceBreaker",
        "spa_input": {
            "in_mics": "grouped_micrographs.star",
            "in_parts": "particles_split1.star",
        },
    },
    "relion.class2d.em": {
        "folder": "Class2D",
        "spa_input": {"fn_img": "particles_split1.star"},
    },
    "relion.class2d.vdam": {
        "folder": "Class2D",
        "spa_input": {"fn_img": "particles_split1.star"},
    },
    "relion.select.class2dauto": {
        "folder": "Select",
        "spa_input": {"fn_model": "run_it020_optimiser.star"},
    },
    "combine_star_files_job": {
        "folder": "Select",
        "spa_input": {"files_to_process": "particles.star"},
    },
    "relion.initialmodel": {
        "folder": "InitialModel",
        "spa_input": {"fn_img": "particles_split1.star"},
    },
    "relion.class3d": {
        "folder": "Class3D",
        "spa_input": {
            "fn_img": "particles_split1.star",
            "fn_ref": "initial_model.mrc",
        },
    },
    "relion.select.onvalue": {
        "folder": "Select",
        "spa_input": {"fn_data": "run_it025_data.star"},
    },
    "relion.refine3d": {
        "folder": "Refine3D",
        "spa_input": {
            "fn_img": "particles_split1.star",
            "fn_ref": "run_it025_class.mrc",
        },
    },
    "relion.maskcreate": {
        "folder": "MaskCreate",
        "spa_input": {"fn_in": "run_class001.star"},
    },
    "relion.postprocess": {
        "folder": "PostProcess",
        "spa_input": {
            "fn_in": "run_half1_class001_unfil.mrc",
            "fn_mask": "mask.mrc",
        },
    },
    "relion.excludetilts": {
        "folder": "ExcludeTiltImages",
        "tomography_input": {"in_tiltseries": "tilt_series_ctf.star"},
    },
    "relion.aligntiltseries": {
        "folder": "AlignTiltSeries",
        "tomography_input": {"in_tiltseries": "selected_tilt_series.star"},
    },
    "relion.reconstructtomograms": {
        "folder": "Tomograms",
        "tomography_input": {"in_tiltseries": "aligned_tilt_series.star"},
    },
    "relion.denoisetomo": {
        "folder": "Denoise",
        "tomography_input": {"in_tomoset": "tomograms.star"},
    },
}


class NodeCreatorParameters(BaseModel):
    job_type: str
    input_file: str = Field(..., min_length=1)
    output_file: str = Field(..., min_length=1)
    relion_options: RelionServiceOptions
    command: str
    stdout: str
    stderr: str
    experiment_type: str = "spa"
    success: bool = True
    results: dict = {}
    alias: Optional[str] = None

    @validator("experiment_type")
    def is_spa_or_tomo(cls, experiment):
        if experiment not in ["spa", "tomography"]:
            raise ValueError("Specify an experiment type of spa or tomography.")
        return experiment


class NodeCreator(CommonService):
    """
    A service for setting up pipeliner jobs
    """

    # Human readable service name
    _service_name = "NodeCreator"

    # Logger name
    _logger_name = "cryoemservices.services.node_creator"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        try:
            queue_name = os.environ["NODE_CREATOR_QUEUE"]
        except KeyError:
            queue_name = "node_creator"
        self.log.info(f"Relion node creator service starting for queue {queue_name}")
        workflows.recipe.wrap_subscribe(
            self._transport,
            queue_name,
            self.node_creator,
            acknowledgement=True,
            log_extender=self.extend_log,
            allow_non_recipe_messages=True,
        )

    def node_creator(self, rw, header: dict, message: dict):
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
            rw.recipe_step = {"parameters": message["parameters"]}
            rw.environment = {"has_recipe_wrapper": False}
            rw.set_default_channel = rw.dummy
            rw.send = rw.dummy
            message = message["content"]

        # Read in and validate the parameters
        try:
            if isinstance(message, dict):
                job_info = NodeCreatorParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                job_info = NodeCreatorParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"Node creator parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        self.log.info(
            f"Received job {job_info.job_type} with output file {job_info.output_file}"
        )
        start_time = datetime.datetime.now()

        # Find the job directory and make sure we are in the processing directory
        job_dir_search = re.search(".+/job[0-9]+", job_info.output_file)
        job_num_search = re.search("/job[0-9]+", job_info.output_file)
        if job_dir_search and job_num_search:
            job_dir = Path(job_dir_search[0])
            job_number = int(job_num_search[0][4:])
        else:
            self.log.warning(f"Cannot determine job dir for {job_info.output_file}")
            rw.transport.nack(header)
            return
        project_dir = job_dir.parent.parent
        os.chdir(project_dir)

        if not (project_dir / "default_pipeline.star").exists():
            self.log.info("No existing project found, so creating one")
            PipelinerProject(make_new_project=True)

        if not pipeline_jobs.get(job_info.job_type) or not pipeline_jobs[
            job_info.job_type
        ].get(job_info.experiment_type + "_input"):
            self.log.error(
                f"Unknown node creator job type {job_info.job_type} "
                f"in {job_info.experiment_type} collection"
            )
            rw.transport.nack(header)
            return

        # Get the options for this job out of the RelionServiceOptions
        pipeline_options = generate_service_options(
            job_info.relion_options,
            job_info.job_type,
        )
        if not pipeline_options:
            self.log.error(f"Cannot generate pipeline options for {job_info.job_type}")
            rw.transport.nack(header)
            return

        # Work out the name of the input star file and add this to the job.star
        if job_dir.parent.name != "Import":
            ii = 0
            for label, star in pipeline_jobs[job_info.job_type][
                job_info.experiment_type + "_input"
            ].items():
                added_file = job_info.input_file.split(":")[ii]
                input_job_in_project = re.search(".+/job[0-9]+", added_file)
                if input_job_in_project:
                    input_job_dir = Path(input_job_in_project[0])
                    try:
                        pipeline_options[label] = (
                            input_job_dir.relative_to(project_dir) / star
                        )
                    except ValueError:
                        self.log.warning(
                            f"WARNING: {input_job_dir} is not relative to {project_dir}"
                        )
                        pipeline_options[label] = input_job_dir / star
                else:
                    self.log.warning(f"WARNING: {added_file} is not in a job")
                    pipeline_options[label] = Path(added_file)
                ii += 1
        elif job_info.job_type == "relion.import.movies":
            pipeline_options["fn_in_raw"] = job_info.input_file
        elif job_info.job_type == "relion.import.tilt_series":
            pipeline_options["movie_files"] = job_info.input_file.split(":")[0]
            pipeline_options["mdoc_files"] = job_info.input_file.split(":")[1]

        try:
            # If this is a new job we need a job.star
            if not Path(f"{job_info.job_type.replace('.', '_')}_job.star").is_file():
                self.log.info(f"Generating options for new job: {job_info.job_type}")
                write_default_jobstar(job_info.job_type)
                params = job_default_parameters_dict(job_info.job_type)
                params.update(pipeline_options)
                _params = {
                    k: str(v) for k, v in params.items() if not isinstance(v, bool)
                }
                _params.update(
                    {
                        k: ("Yes" if v else "No")
                        for k, v in params.items()
                        if isinstance(v, bool)
                    }
                )
                params = _params
                edit_jobstar(
                    f"{job_info.job_type.replace('.', '_')}_job.star",
                    params,
                    f"{job_info.job_type.replace('.', '_')}_job.star",
                )
        except (IndexError, ValueError) as e:
            self.log.error(f"Pipeliner failed for {job_info.job_type}, error {e}")
            rw.transport.nack(header)
            return

        # Copy the job.star file
        (job_dir / "job.star").write_bytes(
            Path(f"{job_info.job_type.replace('.', '_')}_job.star").read_bytes()
        )

        # Mark the job completion status
        job_is_continue = False
        for exit_file in job_dir.glob("PIPELINER_JOB_EXIT_*"):
            if exit_file.name == "PIPELINER_JOB_EXIT_SUCCESS" and job_info.success:
                job_is_continue = True
            exit_file.unlink()
        if job_info.success:
            (job_dir / SUCCESS_FILE).touch()
        else:
            (job_dir / FAIL_FILE).touch()

        # Get the files and directories relative to the project if possible
        relative_job_dir = (
            job_dir.relative_to(project_dir)
            if job_dir.is_relative_to(project_dir)
            else job_dir
        )
        first_input_file = job_info.input_file.split(":")[0]
        relative_input_file = (
            Path(first_input_file).relative_to(project_dir)
            if Path(first_input_file).is_relative_to(project_dir)
            else Path(first_input_file)
        )
        relative_output_file = (
            Path(job_info.output_file).relative_to(project_dir)
            if Path(job_info.output_file).is_relative_to(project_dir)
            else Path(job_info.output_file)
        )

        # Load this job as a pipeliner job to create the nodes
        pipeliner_job = read_job(f"{job_dir}/job.star")
        pipeliner_job.output_dir = str(relative_job_dir) + "/"
        relion_commands = [[], pipeliner_job.get_final_commands()]

        # These parts would normally happen in pipeliner_job.prepare_to_run
        pipeliner_job.create_input_nodes()
        pipeliner_job.create_output_nodes()

        if not job_is_continue:
            try:
                pipeliner_job.handle_doppio_uploads()
            except ValueError:
                self.log.info("Cannot copy Doppio input file that already exists")
            pipeliner_job.write_runjob(pipeliner_job.output_dir)
            pipeliner_job.write_jobstar(pipeliner_job.output_dir)
            pipeliner_job.write_jobstar(".gui_" + job_info.job_type.replace(".", "_"))
            pipeliner_job.write_jobstar(
                f"{pipeliner_job.output_dir}/continue_", is_continue=True
            )

        # Write the log files
        with open(job_dir / "run.out", "w") as f:
            f.write(job_info.stdout)
        with open(job_dir / "run.err", "a") as f:
            if job_info.stderr:
                f.write(f"{job_info.stderr}\n")
        with open(job_dir / "note.txt", "a") as f:
            f.write(f"{job_info.command}\n")

        extra_output_nodes = None
        if job_info.success:
            # Write the output files which Relion produces
            if job_info.experiment_type == "spa":
                extra_output_nodes = create_spa_output_files(
                    job_type=job_info.job_type,
                    job_dir=relative_job_dir,
                    input_file=relative_input_file,
                    output_file=relative_output_file,
                    relion_options=job_info.relion_options,
                    results=job_info.results,
                )
            else:
                extra_output_nodes = create_tomo_output_files(
                    job_type=job_info.job_type,
                    job_dir=relative_job_dir,
                    input_file=relative_input_file,
                    output_file=relative_output_file,
                    relion_options=job_info.relion_options,
                    results=job_info.results,
                )
            if extra_output_nodes:
                # Add any extra nodes if they are not already present
                existing_nodes = []
                for node in pipeliner_job.output_nodes:
                    existing_nodes.append(node.name)
                for node in extra_output_nodes.keys():
                    if f"{relative_job_dir}/{node}" not in existing_nodes:
                        pipeliner_job.add_output_node(
                            node,
                            extra_output_nodes[node][0],
                            extra_output_nodes[node][1],
                        )

            # Save the metadata file
            # try:
            #     metadata_dict = pipeliner_job.gather_metadata()
            #     with open(job_dir / "job_metadata.json", "w") as metadata_file:
            #         metadata_file.write(json.dumps(metadata_dict))
            # except FileNotFoundError as e:
            #     self.log.info(f"Cannot open expected metadata file: {e}")

            # Create the results display for the non-pipeliner job
            if job_info.job_type == "combine_star_files_job":
                results_files = job_dir.glob(".results_display*")
                for results_obj in results_files:
                    results_obj.unlink()

                results_displays = pipeliner_job.create_results_display()
                for results_obj in results_displays:
                    results_obj.write_displayobj_file(outdir=str(job_dir))  # type: ignore

        # If there are no new jobs or new nodes then stop here
        if job_is_continue and not extra_output_nodes:
            end_time = datetime.datetime.now()
            self.log.info(
                f"Skipping graph update for job {job_info.job_type}, "
                f"in {(end_time - start_time).total_seconds()} seconds."
            )
            rw.transport.ack(header)
            return

        # Check the lock status
        if (
            Path(project_dir / ".relion_lock").is_dir()
            or Path(job_dir / ".relion_lock").is_dir()
        ):
            self.log.warning("WARNING: Relion lock found")
            time.sleep(5)
            try:
                Path(project_dir / ".relion_lock").rmdir()
                self.log.warning("Relion project lock has been removed")
            except FileNotFoundError:
                self.log.warning("No project lock found to remove")
            try:
                Path(job_dir / ".relion_lock").rmdir()
                self.log.warning("Relion job lock has been removed")
            except FileNotFoundError:
                self.log.warning("No job lock found to remove")

        if job_info.alias:
            (job_dir.parent / job_info.alias).unlink(missing_ok=True)

        # Create the node and default_pipeline.star files in the project directory
        with CachedProjectGraph(
            read_only=False, pipeline_dir=str(project_dir)
        ) as project:
            process = project.add_job(
                pipeliner_job,
                as_status=("Succeeded" if job_info.success else "Failed"),
                do_overwrite=True,
                alias=job_info.alias,
            )
            # Add the job commands to the process .CCPEM_pipeliner_jobinfo file
            if not (job_dir / ".CCPEM_pipeliner_jobinfo").exists():
                process.update_jobinfo_file(action="Run", command_list=relion_commands)
            # Generate the default_pipeline.star file
            project.check_process_completion()
            # Check the job count in the default_pipeline.star
            with open("default_pipeline.star", "r") as pipeline_file:
                while True:
                    line = pipeline_file.readline()
                    if not line:
                        break
                    if line.startswith("_rlnPipeLineJobCounter"):
                        job_count = int(line.split()[1])
                        break
            if job_count <= job_number:
                project.job_counter = job_number + 1
                with open("default_pipeline.star", "r") as pipeline_file, open(
                    "default_pipeline.star.tmp", "w"
                ) as new_pipeline:
                    while True:
                        line = pipeline_file.readline()
                        if not line:
                            break
                        if line.startswith("_rlnPipeLineJobCounter"):
                            split_line = line.split()
                            split_line[1] = str(project.job_counter)
                            line = " ".join(split_line)
                        new_pipeline.write(line)
                Path("default_pipeline.star").unlink()
                Path("default_pipeline.star.tmp").rename("default_pipeline.star")
            # Copy the default_pipeline.star file
            (job_dir / "default_pipeline.star").write_bytes(
                Path("default_pipeline.star").read_bytes()
            )

        end_time = datetime.datetime.now()
        self.log.info(
            f"Processed outputs from job {job_info.job_type}, "
            f"in {(end_time - start_time).total_seconds()} seconds."
        )
        rw.transport.ack(header)
