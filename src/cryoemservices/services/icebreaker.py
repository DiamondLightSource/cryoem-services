from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Literal, Optional

import workflows.recipe
from gemmi import cif
from icebreaker import ice_groups, icebreaker_equalize_multi, icebreaker_icegroups_multi
from icebreaker.five_figures import single_mic_5fig
from pydantic import BaseModel, Field, ValidationError
from workflows.services.common_service import CommonService

from cryoemservices.util.relion_service_options import RelionServiceOptions
from cryoemservices.util.slurm_submission import slurm_submission


class IceBreakerParameters(BaseModel):
    input_micrographs: str = Field(..., min_length=1)
    input_particles: Optional[str] = None
    output_path: str = Field(..., min_length=1)
    icebreaker_type: str = Literal[
        "micrographs", "enhancecontrast", "summary", "particles"
    ]
    cpus: int = 10
    total_motion: float = 0
    early_motion: float = 0
    late_motion: float = 0
    mc_uuid: int
    submit_to_slurm: bool = True
    relion_options: RelionServiceOptions


class IceBreaker(CommonService):
    """
    A service that runs the IceBreaker micrographs job
    """

    # Human readable service name
    _service_name = "IceBreaker"

    # Logger name
    _logger_name = "cryoemservices.services.icebreaker"

    # Job name
    job_type = "icebreaker.micrograph_analysis"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("IceBreaker service starting")
        workflows.recipe.wrap_subscribe(
            self._transport,
            "icebreaker",
            self.icebreaker,
            acknowledgement=True,
            log_extender=self.extend_log,
            allow_non_recipe_messages=True,
        )

    def icebreaker(self, rw, header: dict, message: dict):
        """
        Main function which interprets received messages, runs icebreaker
        and sends messages to the ispyb and image services
        """

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

        try:
            if isinstance(message, dict):
                icebreaker_params = IceBreakerParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                icebreaker_params = IceBreakerParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"IceBreaker parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        # IceBreaker requires running in the job directory
        project_dir = Path(icebreaker_params.output_path).parent.parent
        if not Path(icebreaker_params.output_path).exists():
            Path(icebreaker_params.output_path).mkdir(parents=True)
        os.chdir(icebreaker_params.output_path)
        mic_from_project = Path(icebreaker_params.input_micrographs).relative_to(
            project_dir
        )
        micrograph_name = Path(mic_from_project.name)

        self.log.info(
            f"Type: {icebreaker_params.icebreaker_type} "
            f"Input: {icebreaker_params.input_micrographs} "
            f"Output: {icebreaker_params.output_path}"
        )
        this_job_type = f"{self.job_type}.{icebreaker_params.icebreaker_type}"
        summary_results = []

        icebreaker_success = True
        if icebreaker_params.icebreaker_type in ["micrographs", "enhancecontrast"]:
            # Create the temporary working directory
            icebreaker_tmp_dir = Path(f"IB_tmp_{micrograph_name.stem}")
            if icebreaker_tmp_dir.is_dir():
                self.log.warning(
                    f"Directory {icebreaker_tmp_dir} already exists - now removing it"
                )
                shutil.rmtree(icebreaker_tmp_dir)
            icebreaker_tmp_dir.mkdir()
            (icebreaker_tmp_dir / mic_from_project.name).symlink_to(
                icebreaker_params.input_micrographs
            )

            mic_path_parts = list(mic_from_project.parts)
            mic_dir_from_job = "/".join(mic_path_parts[2:-1])
            Path(mic_dir_from_job).mkdir(parents=True, exist_ok=True)

            # Run the icebreaker flattening or grouping functions
            if icebreaker_params.icebreaker_type == "micrographs":
                (icebreaker_tmp_dir / "grouped").mkdir()
                icebreaker_icegroups_multi.multigroup(
                    icebreaker_tmp_dir / mic_from_project.name
                )
                (
                    icebreaker_tmp_dir
                    / "grouped"
                    / f"{micrograph_name.stem}_grouped.mrc"
                ).rename(f"{mic_dir_from_job}/{micrograph_name.stem}_grouped.mrc")
            else:
                (icebreaker_tmp_dir / "flattened").mkdir()
                icebreaker_equalize_multi.multigroup(
                    icebreaker_tmp_dir / mic_from_project.name
                )
                (
                    icebreaker_tmp_dir
                    / "flattened"
                    / f"{micrograph_name.stem}_flattened.mrc"
                ).rename(f"{mic_dir_from_job}/{micrograph_name.stem}_flattened.mrc")
            shutil.rmtree(icebreaker_tmp_dir)

            # Create the command this replicates
            command = [
                "ib_job",
                "--j",
                str(icebreaker_params.cpus),
                "--single_mic",
                str(mic_from_project),
                "--o",
                icebreaker_params.output_path,
            ]
            if icebreaker_params.icebreaker_type == "micrographs":
                command.extend(["--mode", "group"])
            else:
                command.extend(["--mode", "flatten"])
        elif icebreaker_params.icebreaker_type == "summary":
            # Run the icebreaker five-figure function
            Path("IB_input").mkdir(exist_ok=True)
            (Path("IB_input") / mic_from_project.name).unlink(missing_ok=True)
            (Path("IB_input") / mic_from_project.name).symlink_to(
                icebreaker_params.input_micrographs
            )
            five_fig_csv = single_mic_5fig(
                str(Path(icebreaker_params.output_path) / "IB_input" / micrograph_name)
            )
            summary_results = five_fig_csv.split(",")
            if len(summary_results) != 6:
                summary_results = ["0", "0", "0", "0", "0", "0"]

            # Create the command this replicates
            command = [
                "ib_5fig",
                "--j",
                str(icebreaker_params.cpus),
                "--single_mic",
                str(mic_from_project),
                "--o",
                icebreaker_params.output_path,
            ]
        elif icebreaker_params.icebreaker_type == "particles":
            # Create the command this replicates
            command = [
                "ib_group",
                "--j",
                str(icebreaker_params.cpus),
                "--in_mics",
                str(mic_from_project),
                "--in_parts",
                str(Path(icebreaker_params.input_particles).relative_to(project_dir)),
                "--o",
                icebreaker_params.output_path,
            ]

            if not icebreaker_params.submit_to_slurm:
                # Run the icebreaker particle batch function
                try:
                    ice_groups.main(
                        icebreaker_params.input_particles,
                        icebreaker_params.input_micrographs,
                    )
                except FileNotFoundError as e:
                    self.log.warning(f"IceBreaker failed to find file: {e}")
                    rw.transport.nack(header)
                    return

                # Create a star file with the input data
                icegroups_doc = cif.Document()
                icegroups_block = icegroups_doc.add_new_block("input_files")
                loop = icegroups_block.init_loop(
                    "", ["_rlnParticles", "_rlnMicrographs"]
                )
                loop.add_row(
                    [
                        icebreaker_params.input_particles,
                        icebreaker_params.input_micrographs,
                    ]
                )
                icegroups_doc.write_file("ib_icegroups.star")
            else:
                # Run the icebreaker command and confirm it ran successfully
                slurm_outcome = slurm_submission(
                    log=self.log,
                    job_name="IceBreaker",
                    command=command,
                    project_dir=project_dir,
                    output_file=Path(icebreaker_params.output_path) / "slurm",
                    cpus=icebreaker_params.cpus,
                    use_gpu=False,
                    use_singularity=False,
                    script_extras="module load EM/icebreaker/0.3.9",
                )
                if slurm_outcome.returncode:
                    # Mark failures only as success is True by default above
                    icebreaker_success = False
        else:
            self.log.warning(
                f"Unknown IceBreaker job type {icebreaker_params.icebreaker_type}"
            )
            rw.transport.nack(header)
            return

        # Register the icebreaker job with the node creator
        self.log.info(f"Sending {this_job_type} to node creator")
        node_creator_parameters = {
            "job_type": this_job_type,
            "input_file": icebreaker_params.input_micrographs,
            "output_file": icebreaker_params.output_path,
            "relion_options": dict(icebreaker_params.relion_options),
            "command": " ".join(command),
            "stdout": "",
            "stderr": "",
            "results": {
                "icebreaker_type": icebreaker_params.icebreaker_type,
                "total_motion": icebreaker_params.total_motion,
                "early_motion": icebreaker_params.early_motion,
                "late_motion": icebreaker_params.late_motion,
            },
        }
        if not icebreaker_success:
            node_creator_parameters["success"] = False
        else:
            node_creator_parameters["success"] = True
        if icebreaker_params.icebreaker_type == "summary":
            # Summary jobs need to send results to node creation
            node_creator_parameters["results"]["summary"] = summary_results[1:]
        if icebreaker_params.icebreaker_type == "particles":
            node_creator_parameters[
                "input_file"
            ] += f":{icebreaker_params.input_particles}"
        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="node_creator",
                message={"parameters": node_creator_parameters, "content": "dummy"},
            )
        else:
            rw.send_to("node_creator", node_creator_parameters)

        # End here if the command failed
        if not icebreaker_success:
            self.log.error("IceBreaker failed")
            rw.transport.nack(header)
            return

        # Forward results to next IceBreaker job
        if icebreaker_params.icebreaker_type == "micrographs":
            self.log.info("Sending to IceBreaker summary")
            next_icebreaker_params = {
                "icebreaker_type": "summary",
                "input_micrographs": str(
                    Path(
                        re.sub(
                            ".+/job[0-9]+/",
                            icebreaker_params.output_path,
                            str(mic_from_project),
                        )
                    ).parent
                    / mic_from_project.stem
                )
                + "_grouped.mrc",
                "relion_options": dict(icebreaker_params.relion_options),
                "mc_uuid": icebreaker_params.mc_uuid,
            }
            job_number = int(
                re.search("/job[0-9]+", icebreaker_params.output_path)[0][4:]
            )
            next_icebreaker_params["output_path"] = re.sub(
                f"IceBreaker/job{job_number:03}/",
                f"IceBreaker/job{job_number + 2:03}/",
                icebreaker_params.output_path,
            )
            if isinstance(rw, MockRW):
                rw.transport.send(
                    destination="icebreaker",
                    message={"parameters": next_icebreaker_params, "content": "dummy"},
                )
            else:
                rw.send_to("icebreaker", next_icebreaker_params)

        # Send results to ispyb
        if icebreaker_params.icebreaker_type == "summary":
            ispyb_parameters = {
                "ispyb_command": "buffer",
                "buffer_lookup": {"motion_correction_id": icebreaker_params.mc_uuid},
                "buffer_command": {"ispyb_command": "insert_relative_ice_thickness"},
                "minimum": summary_results[1],
                "q1": summary_results[2],
                "median": summary_results[3],
                "q3": summary_results[4],
                "maximum": summary_results[5],
            }
            self.log.info(f"Sending to ispyb: {ispyb_parameters}")
            if isinstance(rw, MockRW):
                rw.transport.send(
                    destination="ispyb_connector",
                    message={"parameters": ispyb_parameters, "content": "dummy"},
                )
            else:
                rw.send_to("ispyb_connector", ispyb_parameters)

        # Create symlink for the particle grouping jobs
        if (
            icebreaker_params.icebreaker_type == "particles"
            and Path(icebreaker_params.input_particles).name == "particles_split1.star"
        ):
            Path(project_dir / "IceBreaker/Icebreaker_group_batch_1").unlink(
                missing_ok=True
            )
            Path(project_dir / "IceBreaker/Icebreaker_group_batch_1").symlink_to(
                icebreaker_params.output_path
            )

        self.log.info(
            f"Done {this_job_type} for {icebreaker_params.input_micrographs}."
        )
        rw.transport.ack(header)
