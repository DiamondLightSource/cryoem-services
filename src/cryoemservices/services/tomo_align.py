from __future__ import annotations

import ast
import json
import os.path
import re
import subprocess
import time
from pathlib import Path
from typing import Any, List, Optional

import mrcfile
import numpy as np
import plotly.express as px
from gemmi import cif
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW
from cryoemservices.util.relion_service_options import (
    RelionServiceOptions,
    update_relion_options,
)
from cryoemservices.util.tomo_output_files import _get_tilt_number_v5_12


class TomoParameters(BaseModel):
    stack_file: str = Field(..., min_length=1)
    pixel_size: float
    path_pattern: Optional[str] = None
    input_file_list: Optional[str] = None
    vol_z: int = 1200
    align: Optional[int] = None
    out_bin: int = 4
    tilt_axis: float = 85
    tilt_cor: int = 1
    flip_int: Optional[int] = None
    flip_vol: int = 0
    flip_vol_post_reconstruction: bool = True
    wbp: Optional[int] = None
    roi_file: Optional[list] = None
    patch: Optional[int] = None
    kv: Optional[int] = None
    dose_per_frame: Optional[float] = None
    frame_count: Optional[int] = None
    align_file: Optional[str] = None
    align_z: Optional[int] = None
    refine_flag: int = 1
    make_angle_file: bool = True
    out_imod: int = 1
    out_imod_xf: Optional[int] = None
    dark_tol: Optional[float] = None
    manual_tilt_offset: Optional[float] = None
    relion_options: RelionServiceOptions

    @model_validator(mode="before")
    @classmethod
    def check_only_one_is_provided(cls, values):
        input_file_list = values.get("input_file_list")
        path_pattern = values.get("path_pattern")
        if not input_file_list and not path_pattern:
            raise ValueError("input_file_list or path_pattern must be provided")
        if input_file_list and path_pattern:
            raise ValueError(
                "Message must only include one of 'path_pattern' and 'input_file_list'."
                " Both are set or one has been set by the recipe."
            )
        return values

    @field_validator("input_file_list")
    @classmethod
    def check_list_of_lists_is_not_empty(cls, v):
        try:
            file_list = ast.literal_eval(
                v
            )  # if input_file_list is '' it will break here
        except Exception:
            return v
        if isinstance(file_list, list) and isinstance(file_list[0], list):
            for item in file_list:
                if not len(item) == 2 or not item[0] or not item[1]:
                    raise ValueError("Empty list found")
            return v
        else:
            raise ValueError("input_file_list is not a list of lists")


class TomoAlign(CommonService):
    """
    A service for grouping and aligning tomography tilt-series
    with Newstack and AreTomo2
    """

    # Logger name
    _logger_name = "cryoemservices.services.tomo_align"

    # Job name
    job_type = "relion.reconstructtomograms"

    # Values to extract for ISPyB
    input_file_list_of_lists: List[Any]
    refined_tilts: List[float]
    x_shift: List[float]
    y_shift: List[float]
    rot_centre_z_list: List[str]
    tilt_offset: Optional[float] = None
    rot: float | None = None
    mag: float | None = None
    alignment_quality: Optional[float] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refined_tilts = []
        self.rot_centre_z_list = []

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("TomoAlign service starting")
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "tomo_align",
            self.tomo_align,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def parse_tomo_output(self, tomo_stdout: str):
        self.rot_centre_z_list = []
        self.tilt_offset = None
        self.alignment_quality = None
        for line in tomo_stdout.split("\n"):
            if line.startswith("Rot center Z"):
                self.rot_centre_z_list.append(line.split()[5])
            if line.startswith("Tilt offset"):
                self.tilt_offset = float(line.split()[2].strip(","))
            if line.startswith("Best tilt axis"):
                self.alignment_quality = float(line.split()[5])

    def extract_from_aln(self, tomo_parameters, alignment_output_dir, plot_path):
        tomo_aln_file = None
        self.rot = None
        self.mag = None
        self.x_shift = []
        self.y_shift = []
        self.refined_tilts = []
        aln_files = list(alignment_output_dir.glob("*.aln"))

        file_name = Path(tomo_parameters.stack_file).stem
        for aln_file in aln_files:
            if file_name in str(aln_file):
                tomo_aln_file = aln_file

        if not tomo_aln_file:
            return None

        with open(tomo_aln_file) as f:
            lines = f.readlines()
            for line in lines:
                if not line.startswith("#"):
                    line_split = line.split()
                    self.rot = float(line_split[1])
                    self.mag = float(line_split[2])
                    self.x_shift.append(float(line_split[3]))
                    self.y_shift.append(float(line_split[4]))
                    self.refined_tilts.append(float(line_split[9]))
        if self.x_shift and self.y_shift:
            fig = px.scatter(x=self.x_shift, y=self.y_shift)
            fig_as_json = {
                "data": [json.loads(fig["data"][0].to_json())],
                "layout": json.loads(fig["layout"].to_json()),
            }
            with open(plot_path, "w") as plot_json:
                json.dump(fig_as_json, plot_json)
        return tomo_aln_file  # not needed anywhere atm

    def tomo_align(self, rw, header: dict, message: dict):
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
                tomo_params = TomoParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                tomo_params = TomoParameters(**{**rw.recipe_step.get("parameters", {})})
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"TomoAlign parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        def _tilt(file_list_for_tilts):
            return float(file_list_for_tilts[1])

        if tomo_params.manual_tilt_offset is not None:
            # Stretch the volume for tilted collection
            tomo_params.vol_z = int(tomo_params.vol_z * 4 / 3)

        # Update the relion options
        tomo_params.relion_options = update_relion_options(
            tomo_params.relion_options, dict(tomo_params)
        )
        tomo_params.relion_options.pixel_size_downscaled = (
            tomo_params.pixel_size * tomo_params.out_bin
        )

        # Convert a path pattern into a file list
        if tomo_params.path_pattern:
            directory = Path(tomo_params.path_pattern).parent

            input_file_list = []
            for item in directory.glob(Path(tomo_params.path_pattern).name):
                parts = str(Path(item).with_suffix("").name).split("_")
                for part in parts:
                    if "." in part:
                        input_file_list.append([str(item), part])
            self.input_file_list_of_lists = input_file_list
        elif tomo_params.input_file_list:
            try:
                file_list = ast.literal_eval(
                    tomo_params.input_file_list
                )  # if input_file_list is '' it will break here
            except Exception as e:
                self.log.warning(f"Input file list conversion failed with: {e}")
                rw.transport.nack(header)
                return
            if isinstance(file_list, list) and isinstance(file_list[0], list):
                self.input_file_list_of_lists = file_list
            else:
                self.log.warning("input_file_list is not a list of lists")
                rw.transport.nack(header)
                return
        else:
            self.log.error("Model validation failed")
            rw.transport.nack(header)
            return

        self.log.info(f"Input list {self.input_file_list_of_lists}")
        self.input_file_list_of_lists.sort(key=_tilt)

        # Find all the tilt angles and remove duplicates
        tilt_dict: dict = {}
        for tilt in self.input_file_list_of_lists:
            if not Path(tilt[0]).is_file():
                self.log.warning(f"File not found {tilt[0]}")
                rw.transport.nack(header)
            if tilt[1] not in tilt_dict:
                tilt_dict[tilt[1]] = []
            tilt_dict[tilt[1]].append(tilt[0])

        values_to_remove = []
        for item in tilt_dict:
            values = tilt_dict[item]
            if len(values) > 1:
                # sort by age and remove oldest ones
                values.sort(key=os.path.getctime, reverse=True)
                values_to_remove.extend(values[1:])

        indexes_to_remove = []
        for tilt in self.input_file_list_of_lists:
            if tilt[0] in values_to_remove:
                self.log.warning(f"Removing: {tilt[0]}")
                indexes_to_remove.append(self.input_file_list_of_lists.index(tilt))
        for index in sorted(indexes_to_remove, reverse=True):
            self.input_file_list_of_lists.remove(self.input_file_list_of_lists[index])

        # Remove any images with very bad motion correction results
        tilts_to_remove = []
        for tilt in self.input_file_list_of_lists:
            if Path(tilt[0]).with_suffix(".star").is_file():
                star_doc = cif.read_file(str(Path(tilt[0]).with_suffix(".star")))
                general_block = star_doc.find_block("general")
                if int(general_block.find_value("_rlnMotionModelVersion")):
                    motion_model_block = star_doc.find_block("local_motion_model")
                    motion_model_coeffs = list(
                        motion_model_block.find_loop("_rlnMotionModelCoeff")
                    )
                    if any(np.array(motion_model_coeffs, dtype=float) > 1000):
                        self.log.info(f"Bad motion correction found for {tilt[0]}")
                        tilts_to_remove.append(
                            self.input_file_list_of_lists.index(tilt)
                        )
        removed_tilt_numbers = np.array(
            [
                _get_tilt_number_v5_12(Path(self.input_file_list_of_lists[index][0]))
                for index in tilts_to_remove
            ]
        )
        for index in sorted(tilts_to_remove, reverse=True):
            self.input_file_list_of_lists.remove(self.input_file_list_of_lists[index])

        # Find the input image dimensions
        with mrcfile.open(self.input_file_list_of_lists[0][0]) as mrc:
            mrc_header = mrc.header
        # x and y get flipped on tomogram creation
        tomo_params.relion_options.tomo_size_x = int(mrc_header["nx"])
        tomo_params.relion_options.tomo_size_y = int(mrc_header["ny"])
        scaled_x_size = tomo_params.relion_options.tomo_size_x / int(
            tomo_params.out_bin
        )
        scaled_y_size = tomo_params.relion_options.tomo_size_y / int(
            tomo_params.out_bin
        )
        scaled_z_size = tomo_params.vol_z / int(tomo_params.out_bin)

        # Get the names of the output files expected
        alignment_output_dir = Path(tomo_params.stack_file).parent
        Path(tomo_params.stack_file).parent.mkdir(parents=True, exist_ok=True)
        stack_name = str(Path(tomo_params.stack_file).stem)

        project_dir_search = re.search(".+/job[0-9]+/", tomo_params.stack_file)
        job_num_search = re.search("/job[0-9]+", tomo_params.stack_file)
        if project_dir_search and job_num_search:
            project_dir = Path(project_dir_search[0]).parent.parent
            job_number = int(job_num_search[0][4:])
        else:
            self.log.warning(f"Invalid project directory in {tomo_params.stack_file}")
            rw.transport.nack(header)
            return

        # Stack the tilts with newstack
        newstack_path = alignment_output_dir / f"{stack_name}_newstack.txt"
        newstack_result = self.newstack(tomo_params, newstack_path)
        if newstack_result.returncode:
            self.log.error(
                f"Newstack failed with exitcode {newstack_result.returncode}:\n"
                + newstack_result.stderr.decode("utf8", "replace")
            )
            rw.transport.nack(header)
            return

        # Set up the angle file needed for dose weighting
        angle_file = (
            Path(tomo_params.stack_file).parent
            / f"{Path(tomo_params.stack_file).stem}_tilt_angles.txt"
        )
        tilt_angles = {}
        for i in range(len(self.input_file_list_of_lists)):
            tilt_index = _get_tilt_number_v5_12(
                Path(self.input_file_list_of_lists[i][0])
            )
            tilt_index -= len(np.where(removed_tilt_numbers < tilt_index)[0])
            tilt_angles[tilt_index] = self.input_file_list_of_lists[i][1]
        with open(angle_file, "w") as angfile:
            for tilt_id in tilt_angles.keys():
                angfile.write(f"{tilt_angles[tilt_id]}  {int(tilt_id)}\n")

        # Do alignment with AreTomo
        aretomo_output_path = alignment_output_dir / f"{stack_name}_aretomo.mrc"
        aretomo_result, aretomo_command = self.aretomo(
            tomo_params, aretomo_output_path, angle_file
        )

        # Send to node creator
        self.log.info("Sending tomo align to node creator")
        node_creator_parameters = {
            "experiment_type": "tomography",
            "job_type": self.job_type,
            "input_file": self.input_file_list_of_lists[0][0],
            "output_file": str(aretomo_output_path),
            "relion_options": dict(tomo_params.relion_options),
            "command": " ".join(aretomo_command),
            "stdout": aretomo_result.stdout.decode("utf8", "replace"),
            "stderr": aretomo_result.stderr.decode("utf8", "replace"),
        }
        if aretomo_result.returncode:
            node_creator_parameters["success"] = False
        else:
            node_creator_parameters["success"] = True
        rw.send_to("node_creator", node_creator_parameters)

        # Stop here if the job failed
        if aretomo_result.returncode or not aretomo_output_path.is_file():
            self.log.error(
                f"AreTomo2 failed with exitcode {aretomo_result.returncode}:\n"
                + aretomo_result.stderr.decode("utf8", "replace")
            )
            # Update failure processing status
            rw.send_to("failure", {})
            rw.transport.nack(header)
            return

        imod_directory_option1 = alignment_output_dir / f"{stack_name}_aretomo_Imod"
        imod_directory_option2 = alignment_output_dir / f"{stack_name}_Imod"
        if tomo_params.out_imod:
            start_time = time.time()
            while (
                not imod_directory_option1.is_dir()
                and not imod_directory_option2.is_dir()
            ):
                time.sleep(30)
                elapsed = time.time() - start_time
                if elapsed > 600:
                    self.log.warning("Timeout waiting for Imod directory")
                    break
            else:
                if imod_directory_option1.is_dir():
                    _f = imod_directory_option1
                else:
                    _f = imod_directory_option2
                _f.chmod(0o750)
                for file in _f.iterdir():
                    file.chmod(0o740)

        # Flip the volume if AreTomo has not done this
        if tomo_params.flip_vol_post_reconstruction and not tomo_params.flip_vol:
            if tomo_params.tilt_axis is not None and -45 < tomo_params.tilt_axis < 45:
                # If given tilt axis of around 0, don't do rotations
                angles_to_flip = "0,0,-90"
            else:
                # Otherwise assume we need to flip (which is the behaviour for axis 90)
                angles_to_flip = "90,-90,0"
            rotate_result = subprocess.run(
                [
                    "rotatevol",
                    "-i",
                    str(aretomo_output_path),
                    "-ou",
                    str(aretomo_output_path),
                    "-size",
                    f"{int(scaled_x_size)},{int(scaled_y_size)},{int(scaled_z_size)}",
                    "-a",
                    angles_to_flip,
                ],
                capture_output=True,
            )
            if rotate_result.returncode:
                self.log.error(
                    f"rotatevol failed with exitcode {rotate_result.returncode}:\n"
                    + rotate_result.stderr.decode("utf8", "replace")
                )
                rw.transport.nack(header)
                return

        # Names of the files made for ispyb images
        plot_file = stack_name + "_xy_shift_plot.json"
        plot_path = alignment_output_dir / plot_file
        xy_proj_file = stack_name + "_aretomo_projXY.jpeg"
        xz_proj_file = stack_name + "_aretomo_projXZ.jpeg"
        central_slice_file = stack_name + "_aretomo_thumbnail.jpeg"
        tomogram_movie_file = stack_name + "_aretomo_movie.png"

        # Extract results for ispyb
        aln_file = self.extract_from_aln(tomo_params, alignment_output_dir, plot_path)
        if not aln_file:
            self.log.error("Failed to read alignment file")
            rw.transport.nack(header)
            return
        rot_centre_z = None
        if tomo_params.tilt_cor:
            try:
                rot_centre_z = self.rot_centre_z_list[-1]
            except IndexError:
                self.log.warning(f"No rot Z {self.rot_centre_z_list}")

        pixel_spacing: str = str(tomo_params.pixel_size * tomo_params.out_bin)
        # Forward results to ispyb

        # Tomogram (one per-tilt-series)
        ispyb_command_list = [
            {
                "ispyb_command": "insert_tomogram",
                "volume_file": str(
                    aretomo_output_path.relative_to(alignment_output_dir)
                ),
                "stack_file": tomo_params.stack_file,
                "size_x": scaled_x_size,  # volume image size, pix
                "size_y": scaled_y_size,
                "size_z": scaled_z_size,
                "pixel_spacing": pixel_spacing,
                "tilt_angle_offset": str(self.tilt_offset),
                "z_shift": rot_centre_z,
                "file_directory": str(alignment_output_dir),
                "central_slice_image": central_slice_file,
                "tomogram_movie": tomogram_movie_file,
                "xy_shift_plot": plot_file,
                "proj_xy": xy_proj_file,
                "proj_xz": xz_proj_file,
                "alignment_quality": str(self.alignment_quality),
            }
        ]

        # Find the indexes of the dark images removed by AreTomo
        missing_indices = []
        dark_images_file = Path(stack_name + "_DarkImgs.txt")
        if dark_images_file.is_file():
            with open(dark_images_file) as f:
                missing_indices = [int(i) for i in f.readlines()[2:]]
        elif imod_directory_option1.is_dir() or imod_directory_option2.is_dir():
            if imod_directory_option1.is_dir():
                dark_images_file = imod_directory_option1 / "tilt.com"
            else:
                dark_images_file = imod_directory_option2 / "tilt.com"
            with open(dark_images_file) as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("EXCLUDELIST"):
                        numbers = "".join(line.split(" ")[1:])
                        if numbers:
                            numbers_list = numbers.split(",")
                            missing_indices = [
                                int(item.strip()) for item in numbers_list
                            ]

        im_diff = 0
        # TiltImageAlignment (one per movie)
        node_creator_params_list = []
        (project_dir / f"ExcludeTiltImages/job{job_number - 2:03}").mkdir(
            parents=True, exist_ok=True
        )
        (project_dir / f"AlignTiltSeries/job{job_number - 1:03}").mkdir(
            parents=True, exist_ok=True
        )
        for im, movie in enumerate(self.input_file_list_of_lists):
            if im + 1 in missing_indices:
                im_diff += 1
            else:
                try:
                    ispyb_command_list.append(
                        {
                            "ispyb_command": "insert_tilt_image_alignment",
                            "psd_file": None,  # should be in ctf table but useful, so we will insert
                            "refined_magnification": str(self.mag),
                            "refined_tilt_angle": (
                                str(self.refined_tilts[im - im_diff])
                                if self.refined_tilts
                                else None
                            ),
                            "refined_tilt_axis": str(self.rot),
                            "path": movie[0],
                        }
                    )
                    node_creator_params_list.append(
                        {
                            "job_type": "relion.excludetilts",
                            "experiment_type": "tomography",
                            "input_file": str(movie[0]),
                            "output_file": str(
                                project_dir
                                / f"ExcludeTiltImages/job{job_number - 2:03}/tilts"
                                / Path(movie[0]).name
                            ),
                            "relion_options": dict(tomo_params.relion_options),
                            "command": "",
                            "stdout": "",
                            "stderr": "",
                            "success": True,
                        }
                    )
                    node_creator_params_list.append(
                        {
                            "job_type": "relion.aligntiltseries.aretomo",
                            "experiment_type": "tomography",
                            "input_file": str(
                                project_dir
                                / f"ExcludeTiltImages/job{job_number - 2:03}/tilts"
                                / Path(movie[0]).name
                            ),
                            "output_file": str(
                                project_dir
                                / f"AlignTiltSeries/job{job_number - 1:03}/tilts"
                                / Path(movie[0]).name
                            ),
                            "relion_options": dict(tomo_params.relion_options),
                            "command": "",
                            "stdout": "",
                            "stderr": "",
                            "results": {
                                "TomoXTilt": "0.00",
                                "TomoYTilt": str(self.refined_tilts[im - im_diff]),
                                "TomoZRot": str(self.rot),
                                "TomoXShiftAngst": str(self.x_shift[im - im_diff]),
                                "TomoYShiftAngst": str(self.y_shift[im - im_diff]),
                            },
                            "success": True,
                        }
                    )
                except IndexError as e:
                    self.log.error(
                        f"{e} - Dark images haven't been accounted for properly"
                    )
                    rw.transport.nack(header)
                    return

        for tilt_params in node_creator_params_list:
            rw.send_to("node_creator", tilt_params)

        ispyb_parameters = {
            "ispyb_command": "multipart_message",
            "ispyb_command_list": ispyb_command_list,
        }
        self.log.info(f"Sending to ispyb {ispyb_parameters}")
        rw.send_to("ispyb_connector", ispyb_parameters)

        # Forward results to images service
        self.log.info(f"Sending to images service {tomo_params.stack_file}")
        rw.send_to(
            "images",
            {
                "image_command": "mrc_central_slice",
                "file": tomo_params.stack_file,
            },
        )
        rw.send_to(
            "images",
            {
                "image_command": "mrc_to_apng",
                "file": tomo_params.stack_file,
            },
        )
        self.log.info(f"Sending to images service {aretomo_output_path}")
        rw.send_to(
            "images",
            {
                "image_command": "mrc_central_slice",
                "file": str(aretomo_output_path),
            },
        )
        rw.send_to(
            "images",
            {
                "image_command": "mrc_to_apng",
                "file": str(aretomo_output_path),
            },
        )

        xy_input = alignment_output_dir / Path(xy_proj_file).with_suffix(".mrc")
        xz_input = alignment_output_dir / Path(xz_proj_file).with_suffix(".mrc")
        self.log.info(f"Sending to images service {xy_input}, {xz_input}")
        for projection_mrc in [xy_input, xz_input]:
            rw.send_to(
                "images",
                {
                    "image_command": "mrc_to_jpeg",
                    "file": str(projection_mrc),
                    "pixel_spacing": pixel_spacing,
                },
            )

        # Forward results to denoise service
        self.log.info(f"Sending to denoise service {aretomo_output_path}")
        rw.send_to(
            "denoise",
            {
                "volume": str(aretomo_output_path),
                "output_dir": str(
                    project_dir / f"Denoise/job{job_number+1:03}/tomograms"
                ),
                "relion_options": dict(tomo_params.relion_options),
            },
        )

        # Update success processing status
        rw.send_to("success", {})
        self.log.info(f"Done tomogram alignment for {tomo_params.stack_file}")
        rw.transport.ack(header)

    def newstack(self, tomo_parameters: TomoParameters, newstack_path: Path):
        """
        Construct file containing a list of files
        Run newstack
        """

        # Write a file with a list of .mrcs for input to Newstack
        with open(newstack_path, "w") as f:
            f.write(f"{len(self.input_file_list_of_lists)}\n")
            f.write("\n0\n".join(i[0] for i in self.input_file_list_of_lists))
            f.write("\n0\n")

        newstack_cmd = [
            "newstack",
            "-fileinlist",
            str(newstack_path),
            "-output",
            tomo_parameters.stack_file,
            "-quiet",
        ]
        self.log.info("Running Newstack")
        result = subprocess.run(newstack_cmd, capture_output=True)
        return result

    def assemble_aretomo_command(
        self,
        aretomo_executable: str,
        input_file: str,
        tomo_parameters: TomoParameters,
        aretomo_output_path: Path,
        angle_file: Path,
    ):
        """
        Assemble the command to run AreTomo2, using a base command with
        <executable> -InMrc <input file>
        """
        command = [
            aretomo_executable,
            "-InMrc",
            input_file,
            "-OutMrc",
            str(aretomo_output_path),
        ]

        if tomo_parameters.make_angle_file:
            command.extend(("-AngFile", str(angle_file)))
        else:
            command.extend(
                (
                    "-TiltRange",
                    str(self.input_file_list_of_lists[0][1]),  # lowest tilt
                    str(self.input_file_list_of_lists[-1][1]),
                )
            )  # highest tilt

        if tomo_parameters.manual_tilt_offset is None:
            command.extend(
                (
                    "-TiltCor",
                    str(tomo_parameters.tilt_cor),
                    "-VolZ",
                    str(tomo_parameters.vol_z),
                )
            )
        else:
            command.extend(
                (
                    "-TiltCor",
                    str(tomo_parameters.tilt_cor),
                    str(tomo_parameters.manual_tilt_offset),
                    "-VolZ",
                    str(tomo_parameters.vol_z),
                )
            )

        command.extend(
            (
                "-TiltAxis",
                str(tomo_parameters.tilt_axis),
                str(tomo_parameters.refine_flag),
            )
        )

        if tomo_parameters.frame_count and tomo_parameters.dose_per_frame:
            command.extend(
                (
                    "-ImgDose",
                    str(tomo_parameters.frame_count * tomo_parameters.dose_per_frame),
                )
            )

        aretomo_flags = {
            "out_bin": "-OutBin",
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
            "out_imod": "-OutImod",
            "out_imod_xf": "-OutXf",
            "dark_tol": "-DarkTol",
        }

        for k, v in tomo_parameters.model_dump().items():
            if (v not in [None, ""]) and (k in aretomo_flags):
                command.extend((aretomo_flags[k], str(v)))

        return command

    def aretomo(
        self,
        tomo_parameters: TomoParameters,
        aretomo_output_path: Path,
        angle_file: Path,
    ):
        """
        Run AreTomo2 on output of Newstack
        """
        command = self.assemble_aretomo_command(
            aretomo_executable="AreTomo2",
            input_file=tomo_parameters.stack_file,
            tomo_parameters=tomo_parameters,
            aretomo_output_path=aretomo_output_path,
            angle_file=angle_file,
        )

        self.log.info(f"Running AreTomo2 {command}")
        self.log.info(
            f"Input stack: {tomo_parameters.stack_file} \n"
            f"Output file: {aretomo_output_path}"
        )

        # Save the AreTomo2 command then run it
        with open(aretomo_output_path.with_suffix(".com"), "w") as f:
            f.write(" ".join(command))
        result = subprocess.run(command, capture_output=True)
        if tomo_parameters.tilt_cor:
            self.parse_tomo_output(result.stdout.decode("utf8", "replace"))
        return result, command
