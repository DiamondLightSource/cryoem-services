from __future__ import annotations

import json
import os
import re
import subprocess
from math import hypot
from pathlib import Path
from typing import Any, List, Optional

import plotly.express as px
from gemmi import cif
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW
from cryoemservices.util.relion_service_options import (
    RelionServiceOptions,
    update_relion_options,
)
from cryoemservices.util.slurm_submission import slurm_submission_for_services


class MotionCorrParameters(BaseModel):
    movie: str = Field(..., min_length=1)
    mrc_out: str = Field(..., min_length=1)
    experiment_type: str
    pixel_size: float
    dose_per_frame: float
    use_motioncor2: bool = False
    submit_to_slurm: bool = False
    patch_sizes: dict = {"x": 5, "y": 5}
    gpu: int = 0
    threads: int = 1
    gain_ref: Optional[str] = None
    rot_gain: Optional[int] = None
    flip_gain: Optional[int] = None
    dark: Optional[str] = None
    use_gpus: Optional[int] = None
    sum_range: Optional[dict] = None
    iter: Optional[int] = None
    tol: Optional[float] = None
    throw: Optional[int] = None
    trunc: Optional[int] = None
    fm_ref: int = 0
    voltage: Optional[int] = None
    fm_int_file: Optional[str] = None
    init_dose: Optional[float] = None
    mag: Optional[dict] = None
    motion_corr_binning: Optional[float] = None
    serial: Optional[int] = None
    in_suffix: Optional[str] = None
    eer_sampling: Optional[int] = None
    out_stack: Optional[int] = None
    bft: Optional[dict] = None
    group: Optional[int] = None
    defect_file: Optional[str] = None
    arc_dir: Optional[str] = None
    in_fm_motion: Optional[int] = None
    frame_count: Optional[int] = None
    split_sum: Optional[int] = None
    dose_motionstats_cutoff: float = 4.0
    do_icebreaker_jobs: bool = True
    mc_uuid: int
    picker_uuid: int
    relion_options: RelionServiceOptions
    ctf: dict = {}

    @field_validator("experiment_type")
    @classmethod
    def is_spa_or_tomo(cls, experiment):
        if experiment not in ["spa", "tomography"]:
            raise ValueError("Specify an experiment type of spa or tomography.")
        return experiment

    model_config = ConfigDict(extra="allow")


class MotionCorr(CommonService):
    """
    A service for motion correcting cryoEM movies using MotionCor2
    """

    # Logger name
    _logger_name = "cryoemservices.services.motioncorr"

    # Job name
    job_type = "relion.motioncorr"

    # Values to extract for ISPyB
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_shift_list = []
        self.y_shift_list = []

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Motion correction service starting")
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "motioncorr",
            self.motion_correction,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def parse_mc2_stdout(self, mc_stdout: str):
        """
        Read the output logs of MotionCor2 to determine
        the movement of each frame
        """
        frames_line = False
        for line in mc_stdout.split("\n"):
            # Frame reading in MotionCorr 1.4.0
            if line.startswith("...... Frame"):
                line_split = line.split()
                self.x_shift_list.append(float(line_split[-2]))
                self.y_shift_list.append(float(line_split[-1]))

            # Alternative frame reading for MotionCorr 1.6.3
            if not line:
                frames_line = False
            if frames_line:
                line_split = line.split()
                self.x_shift_list.append(float(line_split[1]))
                self.y_shift_list.append(float(line_split[2]))
            if "x Shift" in line:
                frames_line = True

    def parse_mc2_slurm_output(self, mc_output_file):
        """
        Read the output files produced by MotionCor2 running via Slurm to determine
        the movement of each frame
        """
        with open(mc_output_file, "r") as mc_file:
            lines = mc_file.readlines()
            frames_line = False
            for line in lines:
                # Frame reading in MotionCorr 1.4.0
                if line.startswith("...... Frame"):
                    line_split = line.split()
                    self.x_shift_list.append(float(line_split[-2]))
                    self.y_shift_list.append(float(line_split[-1]))

                # Alternative frame reading for MotionCorr 1.6.3
                if not line:
                    frames_line = False
                if frames_line:
                    line_split = line.split()
                    self.x_shift_list.append(float(line_split[1]))
                    self.y_shift_list.append(float(line_split[2]))
                if "x Shift" in line:
                    frames_line = True

    def parse_relion_mc_output(self, stdout_file):
        """
        Read the output star file made by Relion to determine
        the movement of each frame
        """
        shift_cif = cif.read_file(str(stdout_file))
        shift_block = shift_cif.find_block("global_shift")
        x_shifts_str = list(shift_block.find_loop("_rlnMicrographShiftX"))
        y_shifts_str = list(shift_block.find_loop("_rlnMicrographShiftY"))
        for frame in range(len(x_shifts_str)):
            self.x_shift_list.append(float(x_shifts_str[frame]))
            self.y_shift_list.append(float(y_shifts_str[frame]))

    def motioncor2(self, command: List[str], mrc_out: Path):
        """Run the MotionCor2 command"""
        result = subprocess.run(command, capture_output=True)
        self.parse_mc2_stdout(result.stdout.decode("utf8", "replace"))
        return result

    def motioncor2_slurm(self, command: List[str], mrc_out: Path):
        """Submit MotionCor2 jobs to a slurm cluster via the RestAPI"""
        slurm_outcome = slurm_submission_for_services(
            log=self.log,
            service_config_file=self._environment["config"],
            slurm_cluster=self._environment["slurm_cluster"],
            job_name="MotionCor2",
            command=command,
            project_dir=mrc_out.parent,
            output_file=mrc_out,
            cpus=1,
            use_gpu=True,
            use_singularity=True,
            cif_name=os.environ["MOTIONCOR2_SIF"],
            extra_singularity_directories=["/lib64"],
        )

        if not slurm_outcome.returncode:
            # Read in the output logs
            slurm_output_file = mrc_out.with_suffix(".out")
            slurm_error_file = mrc_out.with_suffix(".err")
            if Path(slurm_output_file).is_file():
                self.parse_mc2_slurm_output(slurm_output_file)

            # Clean up if everything succeeded
            if self.x_shift_list and self.y_shift_list:
                Path(slurm_output_file).unlink()
                Path(slurm_error_file).unlink()
            else:
                self.log.error(f"Reading shifts from {slurm_output_file} failed")
                slurm_outcome.returncode = 1
        return slurm_outcome

    def relion_motioncorr(self, command: List[str], mrc_out: Path):
        """Run Relion's own motion correction"""
        result = subprocess.run(command, capture_output=True)
        if Path(mrc_out).with_suffix(".star").exists():
            self.parse_relion_mc_output(Path(mrc_out).with_suffix(".star"))
        else:
            self.log.error(
                f"Relion output log {Path(mrc_out).with_suffix('.star')} not found"
            )
            result.returncode = 1
        return result

    def relion_motioncorr_slurm(self, command: List[str], mrc_out: Path):
        """Submit Relion's own motion correction to a slurm cluster via the RestAPI"""
        result = slurm_submission_for_services(
            log=self.log,
            service_config_file=self._environment["config"],
            slurm_cluster=self._environment["slurm_cluster"],
            job_name="RelionMotionCorr",
            command=command,
            project_dir=mrc_out.parent,
            output_file=mrc_out,
            cpus=4,
            use_gpu=False,
            use_singularity=False,
            script_extras="module load EM/relion/motioncorr",
        )
        if Path(mrc_out).with_suffix(".star").exists():
            self.parse_relion_mc_output(Path(mrc_out).with_suffix(".star"))
        else:
            self.log.error(
                f"Relion output log {Path(mrc_out).with_suffix('.star')} not found"
            )
        return result

    def motion_correction(self, rw, header: dict, message: dict):
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
                mc_params = MotionCorrParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                mc_params = MotionCorrParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"Motion correction parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        # Catch any cases where the movie does not exist
        if not Path(mc_params.movie).is_file():
            self.log.warning(f"Movie {mc_params.movie} does not exist")
            rw.transport.nack(header)
            return

        # Check if the gain file exists:
        if mc_params.gain_ref and not Path(mc_params.gain_ref).is_file():
            self.log.warning(f"Gain reference {mc_params.gain_ref} does not exist")
            mc_params.gain_ref = ""

        # Generate the plotting paths
        drift_plot_name = str(Path(mc_params.movie).stem) + "_drift_plot.json"
        plot_path = Path(mc_params.mrc_out).parent / drift_plot_name

        # Check if this file has been run before
        if Path(mc_params.mrc_out).is_file():
            job_is_rerun = True
        else:
            job_is_rerun = False

        # Get the eer grouping out of the fractionation file
        eer_grouping = 0
        if (
            mc_params.movie.endswith(".eer")
            and mc_params.fm_int_file
            and Path(mc_params.fm_int_file).is_file()
        ):
            with open(mc_params.fm_int_file, "r") as eer_file:
                eer_values = eer_file.readline()
                try:
                    eer_grouping = int(eer_values.split()[1])
                except ValueError:
                    self.log.warning("Cannot read eer grouping")

        # Submit all super-resolution jobs to slurm using MotionCor2
        if mc_params.motion_corr_binning == 2:
            mc_params.use_motioncor2 = True
            mc_params.submit_to_slurm = True

        # Update the relion options
        mc_params.relion_options = update_relion_options(
            mc_params.relion_options, dict(mc_params)
        )
        mc_params.relion_options.eer_grouping = eer_grouping

        # Determine the input and output files
        self.log.info(f"Input: {mc_params.movie} Output: {mc_params.mrc_out}")
        if not Path(mc_params.mrc_out).parent.exists():
            Path(mc_params.mrc_out).parent.mkdir(parents=True)
        if mc_params.movie.endswith(".mrc"):
            input_flag = "-InMrc"
        elif mc_params.movie.endswith((".tif", ".tiff")):
            input_flag = "-InTiff"
        elif mc_params.movie.endswith(".eer"):
            input_flag = "-InEer"
        else:
            self.log.error(f"No input flag found for movie {mc_params.movie}")
            rw.transport.nack(header)
            return

        # Run motion correction
        if mc_params.use_motioncor2:
            # Construct the command for MotionCor2
            self.job_type = "relion.motioncorr.motioncor2"
            self.log.info("Using MotionCor2")
            command = ["MotionCor2", input_flag, mc_params.movie]
            mc2_flags = {
                "mrc_out": "-OutMrc",
                "patch_sizes": "-Patch",
                "pixel_size": "-PixSize",
                "gain_ref": "-Gain",
                "rot_gain": "-RotGain",
                "flip_gain": "-FlipGain",
                "dark": "-Dark",
                "gpu": "-Gpu",
                "use_gpus": "-UseGpus",
                "sum_range": "-SumRange",
                "iter": "-Iter",
                "tol": "-Tol",
                "throw": "-Throw",
                "trunc": "-Trunc",
                "fm_ref": "-FmRef",
                "voltage": "-Kv",
                "dose_per_frame": "-FmDose",
                "init_dose": "-InitDose",
                "mag": "-Mag",
                "motion_corr_binning": "-FtBin",
                "serial": "-Serial",
                "in_suffix": "-InSuffix",
                "eer_sampling": "-EerSampling",
                "out_stack": "-OutStack",
                "bft": "-Bft",
                "group": "-Group",
                "defect_file": "-DefectFile",
                "arc_dir": "-ArcDir",
                "in_fm_motion": "-InFmMotion",
                "split_sum": "-SplitSum",
            }
            if mc_params.movie.endswith(".eer"):
                mc2_flags["fm_int_file"] = "-FmIntFile"

            # Add values from input parameters with flags
            for k, v in mc_params.model_dump().items():
                if (v not in [None, ""]) and (k in mc2_flags):
                    if type(v) is dict:
                        command.extend(
                            (mc2_flags[k], " ".join(str(_) for _ in v.values()))
                        )
                    else:
                        command.extend((mc2_flags[k], str(v)))
            # Run MotionCor2
            if mc_params.submit_to_slurm:
                result = self.motioncor2_slurm(command, Path(mc_params.mrc_out))
            else:
                result = self.motioncor2(command, Path(mc_params.mrc_out))

            dose_weighted = Path(mc_params.mrc_out).parent / (
                Path(mc_params.mrc_out).stem + "_DW.mrc"
            )
            if dose_weighted.is_file():
                mc_params.mrc_out = str(dose_weighted)
        else:
            # Construct the command for Relion motion correction
            self.job_type = "relion.motioncorr.own"
            self.log.info("Using Relion's own motion correction")
            os.environ["FI_PROVIDER"] = "tcp"
            command = ["relion_motion_correction", "--use_own"]
            relion_mc_flags: dict[str, Any] = {
                "threads": "--j",
                "movie": "--in_movie",
                "mrc_out": "--out_mic",
                "pixel_size": "--angpix",
                "voltage": "--voltage",
                "dose_per_frame": "--dose_per_frame",
                "gain_ref": "--gainref",
                "defect_file": "--defect_file",
                "rot_gain": "--gain_rot",
                "flip_gain": "--gain_flip",
                "motion_corr_binning": "--bin_factor",
                "eer_sampling": "--eer_upsampling",
                "init_dose": "--preexposure",
                "patch_sizes": {"--patch_x": "x", "--patch_y": "y"},
                "bft": {"--bfactor": "local_motion"},
            }
            # Add values from input parameters with flags
            for param_k, param_v in mc_params.model_dump().items():
                if (param_v not in [None, ""]) and (param_k in relion_mc_flags):
                    if type(param_v) is dict:
                        for flag_k, flag_v in relion_mc_flags[param_k].items():
                            command.extend(
                                (flag_k, str(mc_params.model_dump()[param_k][flag_v]))
                            )
                    else:
                        command.extend((relion_mc_flags[param_k], str(param_v)))

            # Add eer grouping if file is eer
            if eer_grouping:
                command.extend(("--eer_grouping", str(eer_grouping)))

            # Add some standard flags
            command.extend(("--dose_weighting", "--i", "dummy"))
            # Run Relion motion correction
            if mc_params.submit_to_slurm:
                result = self.relion_motioncorr_slurm(command, Path(mc_params.mrc_out))
            else:
                result = self.relion_motioncorr(command, Path(mc_params.mrc_out))

        # Adjust the pixel size based on the binning
        if mc_params.motion_corr_binning:
            mc_params.pixel_size *= mc_params.motion_corr_binning
            if mc_params.relion_options:
                mc_params.relion_options.pixel_size *= mc_params.motion_corr_binning

        # Confirm the command ran successfully
        if result.returncode:
            self.log.error(
                f"Motion correction of {mc_params.movie} "
                f"failed with exitcode {result.returncode}:\n"
                + result.stderr.decode("utf8", "replace")
            )
            # On failure send the outputs to the node creator
            node_creator_parameters = {
                "experiment_type": mc_params.experiment_type,
                "job_type": self.job_type,
                "input_file": mc_params.mrc_out,
                "output_file": mc_params.mrc_out,
                "relion_options": dict(mc_params.relion_options),
                "command": " ".join(command),
                "stdout": result.stdout.decode("utf8", "replace"),
                "stderr": result.stderr.decode("utf8", "replace"),
                "success": False,
            }
            rw.send_to("node_creator", node_creator_parameters)
            rw.transport.nack(header)
            return

        # Extract results for ispyb
        total_motion = 0.0
        early_motion = 0.0
        late_motion = 0.0
        cutoff_frame = round(
            mc_params.dose_motionstats_cutoff / mc_params.dose_per_frame
        )
        for i in range(1, len(self.x_shift_list)):
            total_motion += hypot(
                self.x_shift_list[i] - self.x_shift_list[i - 1],
                self.y_shift_list[i] - self.y_shift_list[i - 1],
            )
            if i < cutoff_frame:
                early_motion += hypot(
                    self.x_shift_list[i] - self.x_shift_list[i - 1],
                    self.y_shift_list[i] - self.y_shift_list[i - 1],
                )
            else:
                late_motion += hypot(
                    self.x_shift_list[i] - self.x_shift_list[i - 1],
                    self.y_shift_list[i] - self.y_shift_list[i - 1],
                )
        average_motion_per_frame = total_motion / len(self.x_shift_list)

        # Extract results for ispyb
        fig = px.scatter(x=self.x_shift_list, y=self.y_shift_list)
        fig_as_json = {
            "data": [json.loads(fig["data"][0].to_json())],
            "layout": json.loads(fig["layout"].to_json()),
        }
        with open(plot_path, "w") as plot_json:
            json.dump(fig_as_json, plot_json)
        snapshot_path = Path(mc_params.mrc_out).with_suffix(".jpeg")

        # Forward results to ISPyB
        ispyb_parameters = {
            "ispyb_command": "buffer",
            "buffer_command": {"ispyb_command": "insert_motion_correction"},
            "buffer_store": mc_params.mc_uuid,
            "first_frame": 1,
            "last_frame": len(self.x_shift_list),
            "total_motion": total_motion,
            "average_motion_per_frame": average_motion_per_frame,
            "drift_plot_full_path": str(plot_path),
            "micrograph_snapshot_full_path": str(snapshot_path),
            "micrograph_full_path": str(mc_params.mrc_out),
            "patches_used_x": mc_params.patch_sizes["x"],
            "patches_used_y": mc_params.patch_sizes["y"],
            "dose_per_frame": mc_params.dose_per_frame,
        }
        self.log.info(f"Sending to ispyb {ispyb_parameters}")
        rw.send_to("ispyb_connector", ispyb_parameters)

        # Determine and set up the next jobs
        if mc_params.experiment_type == "spa":
            # Set up icebreaker if requested, then ctffind
            icebreaker_output = Path(
                re.sub(
                    "MotionCorr/job002/",
                    "IceBreaker/job003/",
                    mc_params.mrc_out,
                )
            )
            if mc_params.do_icebreaker_jobs and not icebreaker_output.is_file():
                # Three IceBreaker jobs: CtfFind job is MC+4
                ctf_job_number = 6

                # Both IceBreaker micrographs and flattening inherit from motioncorr
                self.log.info(
                    f"Sending to IceBreaker micrograph analysis: {mc_params.mrc_out}"
                )
                icebreaker_job003_params = {
                    "icebreaker_type": "micrographs",
                    "input_micrographs": mc_params.mrc_out,
                    "output_path": re.sub(
                        "MotionCorr/job002/.+",
                        "IceBreaker/job003/",
                        mc_params.mrc_out,
                    ),
                    "mc_uuid": mc_params.mc_uuid,
                    "relion_options": dict(mc_params.relion_options),
                    "total_motion": total_motion,
                    "early_motion": early_motion,
                    "late_motion": late_motion,
                }
                rw.send_to("icebreaker", icebreaker_job003_params)

                self.log.info(
                    f"Sending to IceBreaker contrast enhancement: {mc_params.mrc_out}"
                )
                icebreaker_job004_params = {
                    "icebreaker_type": "enhancecontrast",
                    "input_micrographs": mc_params.mrc_out,
                    "output_path": re.sub(
                        "MotionCorr/job002/.+",
                        "IceBreaker/job004/",
                        mc_params.mrc_out,
                    ),
                    "mc_uuid": mc_params.mc_uuid,
                    "relion_options": dict(mc_params.relion_options),
                    "total_motion": total_motion,
                    "early_motion": early_motion,
                    "late_motion": late_motion,
                }
                rw.send_to("icebreaker", icebreaker_job004_params)
            elif mc_params.do_icebreaker_jobs and icebreaker_output.is_file():
                # On a rerun, skip IceBreaker jobs but mark the CtfFind job as MC+4
                ctf_job_number = 6
            else:
                # No IceBreaker jobs: CtfFind job is MC+1
                ctf_job_number = 3
        else:
            # Tomography: CtfFind job is MC+1
            ctf_job_number = 3

        # Forward results to ctffind (in both SPA and tomography)
        self.log.info(f"Sending to ctf: {mc_params.mrc_out}")
        mc_params.ctf["experiment_type"] = mc_params.experiment_type
        mc_params.ctf["input_image"] = mc_params.mrc_out
        mc_params.ctf["mc_uuid"] = mc_params.mc_uuid
        mc_params.ctf["picker_uuid"] = mc_params.picker_uuid
        mc_params.ctf["pixel_size"] = mc_params.pixel_size
        mc_params.ctf["relion_options"] = dict(mc_params.relion_options)
        mc_params.ctf["amplitude_contrast"] = mc_params.relion_options.ampl_contrast
        mc_params.ctf["output_image"] = str(
            Path(
                mc_params.mrc_out.replace(
                    "MotionCorr/job002", f"CtfFind/job{ctf_job_number:03}"
                )
            ).with_suffix(".ctf")
        )
        rw.send_to("ctffind", mc_params.ctf)

        # Forward results to images service
        self.log.info(f"Sending to images service {mc_params.mrc_out}")
        rw.send_to(
            "images",
            {
                "image_command": "mrc_to_jpeg",
                "file": mc_params.mrc_out,
            },
        )

        # If this is a new run, send the results to be processed by the node creator
        if not job_is_rerun:
            # As this is the entry point we need to import the file to the project
            self.log.info("Sending relion.import to node creator")
            project_dir_search = re.search(".+/job[0-9]+/", mc_params.mrc_out)
            if project_dir_search:
                project_dir = Path(project_dir_search[0]).parent.parent
            else:
                self.log.error(f"Cannot find project dir for {mc_params.mrc_out}")
                rw.transport.nack(header)
                return
            import_movie = (
                project_dir
                / "Import/job001"
                / Path(mc_params.mrc_out)
                .relative_to(project_dir / "MotionCorr/job002")
                .parent
                / Path(mc_params.movie).name
            )
            if not import_movie.parent.is_dir():
                import_movie.parent.mkdir(parents=True)
            import_movie.unlink(missing_ok=True)
            import_movie.symlink_to(mc_params.movie)
            if mc_params.experiment_type == "spa":
                import_parameters = {
                    "experiment_type": mc_params.experiment_type,
                    "job_type": "relion.import.movies",
                    "input_file": str(mc_params.movie),
                    "output_file": str(import_movie),
                    "relion_options": dict(mc_params.relion_options),
                    "command": "",
                    "stdout": "",
                    "stderr": "",
                }
            else:
                import_parameters = {
                    "experiment_type": mc_params.experiment_type,
                    "job_type": "relion.importtomo",
                    "input_file": f"{mc_params.movie}:{Path(mc_params.movie).parent}/*.mdoc",
                    "output_file": str(import_movie),
                    "relion_options": dict(mc_params.relion_options),
                    "command": "",
                    "stdout": "",
                    "stderr": "",
                }
            rw.send_to("node_creator", import_parameters)

            # Then register the motion correction job with the node creator
            self.log.info(f"Sending {self.job_type} to node creator")
            node_creator_parameters = {
                "experiment_type": mc_params.experiment_type,
                "job_type": self.job_type,
                "input_file": str(import_movie),
                "output_file": mc_params.mrc_out,
                "relion_options": dict(mc_params.relion_options),
                "command": " ".join(command),
                "stdout": result.stdout.decode("utf8", "replace"),
                "stderr": result.stderr.decode("utf8", "replace"),
                "results": {
                    "total_motion": total_motion,
                    "early_motion": early_motion,
                    "late_motion": late_motion,
                },
            }
            rw.send_to("node_creator", node_creator_parameters)

        # Register completion with Murfey if this is tomography
        if mc_params.experiment_type == "tomography":
            self.log.info("Sending to Murfey")
            rw.send_to(
                "murfey_feedback",
                {
                    "register": "motion_corrected",
                    "movie": mc_params.movie,
                    "mrc_out": mc_params.mrc_out,
                },
            )

        self.log.info(f"Done {self.job_type} for {mc_params.movie}.")
        rw.transport.ack(header)
        self.x_shift_list = []
        self.y_shift_list = []
