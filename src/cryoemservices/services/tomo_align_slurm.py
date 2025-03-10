from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List

from workflows.services.common_service import CommonService

from cryoemservices.services.tomo_align import TomoAlign, TomoParameters
from cryoemservices.util.slurm_submission import slurm_submission


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


class TomoAlignSlurm(TomoAlign, CommonService):
    """
    A service for submitting AreTomo2 jobs to a slurm cluster via RestAPI
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

    def aretomo(
        self,
        tomo_parameters: TomoParameters,
        aretomo_output_path: Path,
        angle_file: Path,
    ):
        """Submit AreTomo2 jobs to the slurm cluster via the RestAPI"""
        self.log.info(
            f"Input stack: {tomo_parameters.stack_file} \n"
            f"Output file: {aretomo_output_path}"
        )
        command = self.assemble_aretomo_command(
            aretomo_executable=os.environ["ARETOMO2_EXECUTABLE"],
            input_file=str(Path(tomo_parameters.stack_file).name),
            tomo_parameters=tomo_parameters,
            aretomo_output_path=aretomo_output_path,
            angle_file=angle_file,
        )

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

        self.log.info(f"Running AreTomo2 with command: {command}")
        slurm_outcome = slurm_submission(
            log=self.log,
            service_config_file=self._environment["config"],
            slurm_cluster=self._environment["slurm_cluster"],
            job_name="AreTomo2",
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

        slurm_output_file = f"{aretomo_output_path}.out"
        slurm_error_file = f"{aretomo_output_path}.out"
        if tomo_parameters.tilt_cor and Path(slurm_output_file).is_file():
            self.parse_tomo_output(slurm_output_file)

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
