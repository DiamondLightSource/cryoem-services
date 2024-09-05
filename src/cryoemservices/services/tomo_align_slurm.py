from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List

import datasyncer
from requests import HTTPError
from workflows.services.common_service import CommonService

from cryoemservices.services.tomo_align import TomoAlign, TomoParameters
from cryoemservices.util.slurm_submission import slurm_submission


def retrieve_files(job_directory: Path, files_to_skip: List[Path], basepath: str):
    """Copy files back from the Iris cluster"""
    iris_directory = Path("/iris") / Path(job_directory).relative_to("/dls")
    for iris_item in iris_directory.glob(f"{basepath}*"):
        # Find all files in the job directory
        dls_item = job_directory / iris_item.relative_to(iris_directory)
        dls_item.parent.mkdir(parents=True, exist_ok=True)
        if iris_item.is_dir():
            # Transfer imod directory files (assumes only one layer of subdirectories)
            dls_item.mkdir(exist_ok=True)
            for iris_imod in iris_item.glob("*"):
                dls_imod = job_directory / iris_imod.relative_to(iris_directory)
                shutil.copy(iris_imod, dls_imod)
                iris_imod.unlink()
            iris_item.rmdir()
        else:
            # Transfer and remove all other files, but skip copying input files
            if dls_item not in files_to_skip:
                shutil.copy(iris_item, dls_item)
            iris_item.unlink()


def transfer_files(file_list: List[str]):
    """Transfer files to the Iris cluster"""
    try:
        transfer_id = datasyncer.transfer(file_list)
        status = "active"
        while status == "active":
            time.sleep(5)
            status = datasyncer.status(transfer_id)
    except HTTPError as e:
        return f"Unable to transfer data: {e}"

    if status == "succeeded":
        return 0
    elif status == "failed":
        return "Data syncer reported a failure"
    else:
        return f"Unknown status {status}"


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

    def aretomo(self, tomo_parameters: TomoParameters, aretomo_output_path: Path):
        """Submit AreTomo2 jobs to the slurm cluster via the RestAPI"""
        self.log.info(
            f"Input stack: {tomo_parameters.stack_file} \n"
            f"Output file: {aretomo_output_path}"
        )

        # Assemble the command to run AreTomo2
        command = [
            os.environ["ARETOMO2_EXECUTABLE"],
            "-OutMrc",
            str(aretomo_output_path),
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
            "pixel_size": "-PixSize",
            "init_val": "initVal",
            "refine_flag": "refineFlag",
            "out_imod": "-OutImod",
            "out_imod_xf": "-OutXf",
            "dark_tol": "-DarkTol",
        }
        for k, v in tomo_parameters.dict().items():
            if v and (k in aretomo_flags):
                command.extend((aretomo_flags[k], str(v)))

        # Transfer the required files
        self.log.info("Transferring files...")
        transfer_status = transfer_files([tomo_parameters.stack_file])
        if transfer_status:
            self.log.error(f"Unable to transfer files: {transfer_status}")
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
                stdout="".encode("utf8"),
                stderr=transfer_status.encode("utf8"),
            )
        self.log.info("All files transferred")
        self.log.info(f"Running AreTomo2 with command: {command}")

        slurm_outcome = slurm_submission(
            log=self.log,
            job_name="AreTomo2",
            command=command,
            project_dir=Path(self.alignment_output_dir),
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
            job_directory=Path(self.alignment_output_dir),
            files_to_skip=[Path(tomo_parameters.stack_file)],
            basepath=str(Path(tomo_parameters.stack_file).stem),
        )
        self.log.info("All output files retrieved")

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
