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

from cryoemservices.services.tomo_align import TomoAlign
from cryoemservices.util.slurm_submission import slurm_submission


def retrieve_files(job_directory: Path, files_to_skip: List[Path]):
    """Copy files back from the Iris cluster"""
    iris_directory = Path("/iris") / Path(job_directory).relative_to("/dls")
    for iris_file in iris_directory.glob("*"):
        dls_file = job_directory / iris_file.relative_to(iris_directory)
        if dls_file not in files_to_skip:
            dls_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(iris_file, dls_file)
        iris_file.unlink()


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

        # Copy the AreTomo executable into the visit directory
        aretomo_executable = (
            str(Path(self.aretomo_output_path).with_suffix("")) + "_AreTomo"
        )
        shutil.copy(os.environ["ARETOMO_EXECUTABLE"], aretomo_executable)

        # Assemble the command to run AreTomo
        command = [
            aretomo_executable,
            "-OutMrc",
            self.aretomo_output_path,
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

        self.log.info(f"Running AreTomo with command: {command}")
        self.log.info(
            f"Input stack: {tomo_parameters.stack_file} \n"
            f"Output file: {self.aretomo_output_path}"
        )

        # Transfer the required files
        transfer_status = transfer_files(
            [tomo_parameters.stack_file, aretomo_executable]
        )
        if transfer_status:
            self.log.error(f"Unable to transfer files: {transfer_status}")
            return subprocess.CompletedProcess(
                args="",
                returncode=1,
                stdout="",
                stderr=transfer_status,
            )

        slurm_outcome = slurm_submission(
            log=self.log,
            job_name="AreTomo",
            command=command,
            project_dir=Path(self.alignment_output_dir),
            output_file=Path(self.aretomo_output_path),
            cpus=1,
            use_singularity=True,
        )

        # Get back any output files and clean up
        retrieve_files(
            job_directory=Path(self.alignment_output_dir),
            files_to_skip=[tomo_parameters.stack_file, aretomo_executable],
        )
        Path(aretomo_executable).unlink()

        return slurm_outcome
