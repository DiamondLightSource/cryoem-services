from __future__ import annotations

import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

import yaml
from workflows.services.common_service import CommonService

from cryoemservices.services.tomo_align import TomoAlign


def retrieve_files(*args):
    pass


def file_transfer(*args):
    pass


# To get the required image run
# singularity pull docker://gcr.io/diamond-pubreg/em/motioncorr:version
slurm_json_template = {
    "job": {
        "name": "TomoAlign",
        "nodes": 1,
        "tasks": 1,
        "cpus_per_task": 1,
        "gpus": 1,
        "memory_per_gpu": 7000,
        "time_limit": "1:00:00",
    },
    "script": (
        "#!/bin/bash\n"
        "echo \"$(date '+%Y-%m-%d %H:%M:%S.%3N'): running AreTomo\"\n"
        "mkdir /tmp/tmp_$SLURM_JOB_ID\n"
        "export SINGULARITY_CACHEDIR=/tmp/tmp_$SLURM_JOB_ID\n"
        "export SINGULARITY_TMPDIR=/tmp/tmp_$SLURM_JOB_ID\n"
        "singularity exec --nv --bind /tmp/tmp_$SLURM_JOB_ID:/tmp"
    ),
}
slurm_tmp_cleanup = "\nrm -rf /tmp/tmp_$SLURM_JOB_ID"


class TomoAlignSlurm(TomoAlign, CommonService):
    """
    A service for submitting AreTomo jobs to a slurm cluster via RestAPI
    """

    # Logger name
    _logger_name = "cryoemservices.services.tomo_align_slurm"

    def parse_tomo_output(self, tomo_output_file):
        tomo_file = open(tomo_output_file, "r")
        lines = tomo_file.readlines()
        for line in lines:
            if line.startswith("Rot center Z"):
                self.rot_centre_z_list.append(line.split()[5])
            if line.startswith("Tilt offset"):
                self.tilt_offset = float(line.split()[2].strip(","))
            if line.startswith("Best tilt axis"):
                self.alignment_quality = float(line.split()[5])
        tomo_file.close()

    def aretomo(self, tomo_parameters):
        """Submit AreTomo jobs to the slurm cluster via the RestAPI"""
        try:
            # Get the configuration and token for the restAPI
            with open(os.environ["SLURM_RESTAPI_CONFIG"], "r") as f:
                slurm_rest = yaml.safe_load(f)
            user = slurm_rest["user"]
            user_home = slurm_rest["user_home"]
            with open(slurm_rest["user_token"], "r") as f:
                slurm_token = f.read().strip()
        except (KeyError, FileNotFoundError):
            self.log.error("Unable to load slurm restAPI config file and token")
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
                stdout="".encode("utf8"),
                stderr="No restAPI config or token".encode("utf8"),
            )

        # Construct the json for submission
        slurm_output_file = f"{tomo_parameters.aretomo_output_file}.out"
        slurm_error_file = f"{tomo_parameters.aretomo_output_file}.err"
        submission_file = f"{tomo_parameters.aretomo_output_file}.json"
        slurm_config = {
            "environment": {"USER": user, "HOME": user_home},
            "standard_output": slurm_output_file,
            "standard_error": slurm_error_file,
            "current_working_directory": str(
                Path(tomo_parameters.aretomo_output_file).parent
            ),
        }

        # Add slurm partition and cluster preferences if given
        if slurm_rest.get("partition"):
            slurm_config["partition"] = slurm_rest["partition"]
        if slurm_rest.get("partition_preference"):
            slurm_config["prefer"] = slurm_rest["partition_preference"]
        if slurm_rest.get("cluster"):
            slurm_config["cluster"] = slurm_rest["cluster"]
        slurm_json_job = dict(slurm_json_template["job"], **slurm_config)

        # Assemble the command to run AreTomo
        command = [
            os.environ["ARETOMO_SIF"],
            "-OutMrc",
            tomo_parameters.aretomo_output_file,
            "-InMrc",
            str(Path(tomo_parameters.stack_file).name),
        ]

        if tomo_parameters.angle_file:
            command.extend(("-AngFile", tomo_parameters.angle_file))
        else:
            command.extend(
                (
                    "-TiltRange",
                    tomo_parameters.input_file_list[0][1],  # lowest tilt
                    tomo_parameters.input_file_list[-1][1],
                )
            )  # highest tilt

        if tomo_parameters.manual_tilt_offset:
            command.extend(
                (
                    "-TiltCor",
                    str(tomo_parameters.tilt_cor),
                    str(tomo_parameters.manual_tilt_offset),
                )
            )
        elif tomo_parameters.tilt_cor:
            command.extend(("-TiltCor", str(tomo_parameters.tilt_cor)))

        aretomo_flags = {
            "vol_z": "-VolZ",
            "out_bin": "-OutBin",
            "tilt_axis": "-TiltAxis",
            "flip_int": "-FlipInt",
            "flip_vol": "-FlipVol",
            "wbp": "-Wbp",
            "align": "-Align",
            "roi_file": "-RoiFile",
            "patch": "-Patch",
            "kv": "-Kv",
            "align_file": "-AlnFile",
            "align_z": "-AlignZ",
            "pix_size": "-PixSize",
            "init_val": "initVal",
            "refine_flag": "refineFlag",
            "out_imod": "-OutImod",
            "out_imod_xf": "-OutXf",
            "dark_tol": "-DarkTol",
        }

        for k, v in tomo_parameters.dict().items():
            if v and (k in aretomo_flags):
                command.extend((aretomo_flags[k], str(v)))

        # Construct the job command and save the job script
        if slurm_rest.get("required_directories"):
            binding_dirs = "," + ",".join(slurm_rest["required_directories"])
        else:
            binding_dirs = ""
        job_command = (
            slurm_json_template["script"]
            + f"{binding_dirs} --home {user_home} "
            + " ".join(command)
            + slurm_tmp_cleanup
        )
        slurm_json = {"job": slurm_json_job, "script": job_command}
        with open(submission_file, "w") as f:
            json.dump(slurm_json, f)

        self.log.info(f"Running AreTomo with command: {command}")
        self.log.info(
            f"Input stack: {tomo_parameters.stack_file} \n"
            f"Output file: {tomo_parameters.aretomo_output_file}"
        )

        # Transfer the required files
        file_transfer(
            "$(stack_file), /dls/ebic/data/staff-scratch/murfey/AreTomo_1.3.0_Cuda112_09292022"
        )

        # Command to submit jobs to the restAPI
        slurm_submit_command = (
            f'curl -H "X-SLURM-USER-NAME:{user}" -H "X-SLURM-USER-TOKEN:{slurm_token}" '
            '-H "Content-Type: application/json" -X POST '
            f'{slurm_rest["url"]}/slurm/{slurm_rest["api_version"]}/job/submit '
            f"-d @{submission_file}"
        )
        slurm_submission_json = subprocess.run(
            slurm_submit_command, capture_output=True, shell=True
        )
        try:
            # Extract the job id from the submission response to use in the next query
            slurm_response = slurm_submission_json.stdout.decode("utf8", "replace")
            job_id = json.loads(slurm_response)["job_id"]
        except (json.JSONDecodeError, KeyError):
            self.log.error(
                f"Unable to submit job to {slurm_rest['url']}. The restAPI returned "
                f"{slurm_submission_json.stdout.decode('utf8', 'replace')}"
            )
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
                stdout=slurm_submission_json.stdout,
                stderr=slurm_submission_json.stderr,
            )
        self.log.info(f"Submitted job {job_id} to Wilson. Waiting...")

        # Command to get the status of the submitted job from the restAPI
        slurm_status_command = (
            f'curl -H "X-SLURM-USER-NAME:{user}" -H "X-SLURM-USER-TOKEN:{slurm_token}" '
            '-H "Content-Type: application/json" -X GET '
            f'{slurm_rest["url"]}/slurm/{slurm_rest["api_version"]}/job/{job_id}'
        )
        slurm_job_state = "PENDING"

        # Wait until the job has a status indicating it has finished
        loop_counter = 0
        while slurm_job_state in (
            "PENDING",
            "CONFIGURING",
            "RUNNING",
            "COMPLETING",
        ):
            if loop_counter < 5:
                time.sleep(5)
            else:
                time.sleep(30)
            loop_counter += 1

            # Call the restAPI to find out the job state
            slurm_status_json = subprocess.run(
                slurm_status_command, capture_output=True, shell=True
            )
            try:
                slurm_response = slurm_status_json.stdout.decode("utf8", "replace")
                slurm_job_state = json.loads(slurm_response)["jobs"][0]["job_state"]
            except (json.JSONDecodeError, KeyError):
                print(slurm_status_command)
                self.log.error(
                    f"Unable to get status for job {job_id}. The restAPI returned "
                    f"{slurm_status_json.stdout.decode('utf8', 'replace')}"
                )
                return subprocess.CompletedProcess(
                    args="",
                    returncode=1,
                    stdout=slurm_status_json.stdout,
                    stderr=slurm_status_json.stderr,
                )

            if loop_counter >= 60:
                slurm_cancel_command = (
                    f'curl -H "X-SLURM-USER-NAME:{user}" '
                    f'-H "X-SLURM-USER-TOKEN:{slurm_token}" '
                    '-H "Content-Type: application/json" -X DELETE '
                    f'{slurm_rest["url"]}/slurm/{slurm_rest["api_version"]}/job/{job_id}'
                )
                subprocess.run(slurm_cancel_command, capture_output=True, shell=True)
                self.log.error("Timeout running motion correction")
                return subprocess.CompletedProcess(
                    args="",
                    returncode=1,
                    stdout="".encode("utf8"),
                    stderr="Timeout running motion correction".encode("utf8"),
                )

        # Get back the output files
        retrieve_files(slurm_output_file, slurm_error_file, "tar_imod")
        tar_imod_dir = str(Path(self.imod_directory).with_suffix(".tar.gz"))
        file = tarfile.open(tar_imod_dir)
        file.extractall(self.alignment_output_dir)
        file.close()

        # Read in the output
        self.log.info(f"Job {job_id} has finished!")
        try:
            if tomo_parameters.tilt_cor:
                self.parse_tomo_output(slurm_output_file)
            with open(slurm_output_file, "r") as mc_stdout:
                stdout = mc_stdout.read()
            with open(slurm_error_file, "r") as mc_stderr:
                stderr = mc_stderr.read()
        except FileNotFoundError:
            self.log.error(f"Output file {slurm_output_file} not found")
            stdout = ""
            stderr = f"Reading output file {slurm_error_file} failed"
            slurm_job_state = "FAILED"

        # Read in the output then clean up the files
        self.log.info(f"Job {job_id} has finished!")
        if slurm_job_state == "COMPLETED":
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
                stdout=stdout.encode("utf8"),
                stderr=stderr.encode("utf8"),
            )
        else:
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
                stdout=stdout.encode("utf8"),
                stderr=stderr.encode("utf8"),
            )
