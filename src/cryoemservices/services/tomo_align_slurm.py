from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import List

import requests

from cryoemservices.services.tomo_align import TomoAlign, TomoParameters
from cryoemservices.util.slurm_submission import slurm_submission_for_services


def retrieve_files(
    job_directory: Path,
    files_to_skip: List[Path],
    basepath: str,
    local_base: str = "/dls",
    remote_base: str = "/iris",
):
    """Copy files back from remote filesystem to local filesystem"""
    remote_directory = Path(remote_base) / job_directory.relative_to(local_base)
    for remote_item in remote_directory.glob(f"{basepath}*"):
        # Find all files in the job directory
        local_item = job_directory / remote_item.relative_to(remote_directory)
        local_item.parent.mkdir(parents=True, exist_ok=True)
        if remote_item.is_dir():
            # Transfer imod directory files (assumes only one layer of subdirectories)
            local_item.mkdir(exist_ok=True)
            for remote_imod in remote_item.glob("*"):
                local_imod = job_directory / remote_imod.relative_to(remote_directory)
                shutil.copy(remote_imod, local_imod)
                remote_imod.unlink()
            remote_item.rmdir()
        else:
            # Transfer and remove all other files, but skip copying input files
            if local_item not in files_to_skip:
                shutil.copy(remote_item, local_item)
            remote_item.unlink()
    for extra_local_item in files_to_skip:
        extra_remote_item = Path(remote_base) / extra_local_item.relative_to(local_base)
        if extra_remote_item.is_file():
            extra_remote_item.unlink()


def transfer_files(
    file_list: List[Path], local_base: str = "/dls", remote_base: str = "/iris"
):
    """Transfer files from local filesystem to remote filesystem"""
    transferred_items: List[Path] = []
    for local_item in file_list:
        if not local_item.is_file():
            continue
        remote_item = Path(remote_base) / local_item.relative_to(local_base)
        try:
            remote_item.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(local_item, remote_item)
            transferred_items.append(local_item)
        except Exception:
            continue
    return transferred_items


def get_iris_state(logger, wait=True) -> str:
    logger.info("Checking IRIS status...")
    iris_status = requests.get("https://iristrafficlights.diamond.ac.uk/status")
    if iris_status.status_code == 200:
        iris_colour = iris_status.json()["status"]
        if iris_colour == "green":
            logger.info("IRIS state is green")
        elif iris_colour == "red":
            logger.warning("IRIS state is red, service will wait 30 minutes")
            if wait:
                time.sleep(30 * 60)
            else:
                return ""
            iris_colour = get_iris_state(logger, wait=False)
        else:
            logger.warning(f"IRIS state is {iris_colour}")
        return iris_colour
    logger.warning("Could not get IRIS state")
    return "unknown"


class TomoAlignSlurm(TomoAlign):
    """
    A service for submitting AreTomo2 jobs to a slurm cluster via RestAPI
    """

    # Logger name
    _logger_name = "cryoemservices.services.tomo_align_slurm"

    def initializing(self):
        if not get_iris_state(self.log):
            exit()
        super().initializing()

    @staticmethod
    def check_visit(tomo_params: TomoParameters):
        # Requeue visits that should not be sent via slurm
        visit_search = re.search(
            "/[a-z]{2}[0-9]{5}-[0-9]{1,3}/", tomo_params.stack_file
        )
        if visit_search:
            visit_name = visit_search[0][1:-1]
            visit_code = visit_name[:2]
            if (
                not tomo_params.visits_for_slurm
                or visit_code in tomo_params.visits_for_slurm
            ):
                return True
        return False

    def parse_tomo_output_file(self, tomo_output_file: Path):
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

    def aretomo(
        self,
        tomo_parameters: TomoParameters,
        aretomo_output_path: Path,
        angle_file: Path,
    ):
        """Submit AreTomo2 or 3 jobs to the slurm cluster via the RestAPI"""
        self.log.info(f"Input stack: {tomo_parameters.stack_file}")
        if tomo_parameters.aretomo_version == 3:
            command = self.assemble_aretomo3_command(
                aretomo_executable=os.environ["ARETOMO3_EXECUTABLE"],
                input_file=str(Path(tomo_parameters.stack_file).name),
                tomo_parameters=tomo_parameters,
            )
            aretomo_version = "AreTomo3"
        else:
            command = self.assemble_aretomo2_command(
                aretomo_executable=os.environ["ARETOMO2_EXECUTABLE"],
                input_file=str(Path(tomo_parameters.stack_file).name),
                tomo_parameters=tomo_parameters,
                aretomo_output_path=aretomo_output_path,
                angle_file=angle_file,
            )
            aretomo_version = "AreTomo2"

        # Transfer the required files
        self.log.info("Transferring files...")
        items_to_transfer = [Path(tomo_parameters.stack_file), angle_file]
        transfer_status = transfer_files(items_to_transfer)
        if len(transfer_status) != len(items_to_transfer):
            self.log.error(
                "Unable to transfer files: "
                f"desired {items_to_transfer}, done {transfer_status}"
            )
            return (
                subprocess.CompletedProcess(
                    args="",
                    returncode=1,
                    stdout="".encode("utf8"),
                    stderr="Failed transfer".encode("utf8"),
                ),
                command,
            )
        self.log.info("All files transferred")

        self.log.info(f"Running {aretomo_version} with command: {command}")
        slurm_outcome = slurm_submission_for_services(
            log=self.log,
            service_config_file=self._environment["config"],
            slurm_cluster=self._environment["slurm_cluster"],
            job_name=aretomo_version,
            command=command,
            project_dir=aretomo_output_path.parent,
            output_file=aretomo_output_path,
            cpus=1,
            use_gpu=True,
            use_singularity=False,
            script_extras=(
                f"export LD_LIBRARY_PATH={os.environ['EXTRA_LIBRARIES']}:$LD_LIBRARY_PATH"
            ),
            external_filesystem=True,
            memory_request=20000,
        )

        # Get back any output files and clean up
        self.log.info("Retrieving output files...")
        retrieve_files(
            job_directory=aretomo_output_path.parent,
            files_to_skip=[Path(tomo_parameters.stack_file), angle_file],
            basepath=str(Path(tomo_parameters.stack_file).stem),
        )
        self.log.info("All output files retrieved")

        if not aretomo_output_path.is_file():
            return (
                subprocess.CompletedProcess(
                    args="",
                    returncode=1,
                    stdout="".encode("utf8"),
                    stderr=f"Output {aretomo_output_path} not found".encode("utf8"),
                ),
                command,
            )

        slurm_output_file = aretomo_output_path.with_suffix(".out")
        slurm_error_file = aretomo_output_path.with_suffix(".err")
        if tomo_parameters.tilt_cor and slurm_output_file.is_file():
            self.parse_tomo_output_file(slurm_output_file)

        try:
            with open(slurm_output_file, "r") as slurm_stdout:
                slurm_outcome.stdout = slurm_stdout.read()
            with open(slurm_error_file, "r") as slurm_stderr:
                slurm_outcome.stderr = slurm_stderr.read()
        except FileNotFoundError:
            self.log.error(f"Output file {slurm_output_file} not found")
            slurm_outcome.stdout = ""
            slurm_outcome.stderr = f"Reading output file {slurm_output_file} failed"
        slurm_outcome.stdout = slurm_outcome.stdout.encode("utf8")
        slurm_outcome.stderr = slurm_outcome.stderr.encode("utf8")
        return slurm_outcome, command
