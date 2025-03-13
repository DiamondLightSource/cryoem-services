from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np
import workflows.recipe
from gemmi import cif
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from workflows.services.common_service import CommonService

from cryoemservices.util.models import MockRW
from cryoemservices.util.relion_service_options import RelionServiceOptions


class CryoloParameters(BaseModel):
    input_path: str = Field(..., min_length=1)
    output_path: str = Field(..., min_length=1)
    experiment_type: str
    pixel_size: Optional[float] = None
    cryolo_box_size: int = 160
    cryolo_model_weights: str = "gmodel_phosnet_202005_N63_c17.h5"
    cryolo_threshold: float = 0.3
    retained_fraction: float = 1
    min_particles: int = 30
    cryolo_command: str = "cryolo_predict.py"
    particle_diameter: Optional[float] = None
    min_distance: float = 0
    normalization_margin: float = 0
    tomo_tracing_min_frames: int = 5
    tomo_tracing_missing_frames: int = 0
    tomo_tracing_search_range: int = -1
    on_the_fly: bool = True
    mc_uuid: Optional[int] = None
    picker_uuid: Optional[int] = None
    relion_options: RelionServiceOptions
    ctf_values: dict = {}

    @field_validator("experiment_type")
    @classmethod
    def is_spa_or_tomo(cls, experiment):
        if experiment not in ["spa", "tomography"]:
            raise ValueError("Specify an experiment type of spa or tomography.")
        return experiment

    @model_validator(mode="after")
    def check_spa_has_uuids_and_pixel_size(self):
        if self.experiment_type == "spa" and (
            self.mc_uuid is None or self.picker_uuid is None or not self.pixel_size
        ):
            raise ValueError(
                "In SPA mode the following must be provided: "
                f"mc_uuid (given {self.mc_uuid}), "
                f"picker_uuid (given {self.picker_uuid}), "
                f"pixel_size (given {self.pixel_size})"
            )
        return self


class CrYOLO(CommonService):
    """
    A service that runs crYOLO particle picking
    """

    # Human readable service name
    _service_name = "CrYOLO"

    # Logger name
    _logger_name = "cryoemservices.services.cryolo"

    # Job name
    job_type = "cryolo.autopick"

    # Values to extract for ISPyB
    number_of_particles: int

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("crYOLO service starting")
        workflows.recipe.wrap_subscribe(
            self._transport,
            self._environment["queue"] or "cryolo",
            self.cryolo,
            acknowledgement=True,
            log_extender=self.extend_log,
            allow_non_recipe_messages=True,
        )

    def parse_cryolo_output(self, cryolo_stdout: str):
        """
        Read the output logs of cryolo to determine
        the number of particles that are picked
        """
        for line in cryolo_stdout.split("\n"):
            if "particles in total has been found" in line:
                line_split = line.split()
                self.number_of_particles += int(line_split[0])

            if line.startswith("Deleted"):
                line_split = line.split()
                self.number_of_particles -= int(line_split[1])

    def cryolo(self, rw, header: dict, message: dict):
        """
        Main function which interprets received messages, runs cryolo
        and sends messages to the ispyb and image services
        """
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

        # Reset number of particles
        self.number_of_particles = 0

        try:
            if isinstance(message, dict):
                cryolo_params = CryoloParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                cryolo_params = CryoloParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"crYOLO parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        # Check if this file has been run before
        if Path(cryolo_params.output_path).is_file():
            job_is_rerun = True
            Path(cryolo_params.output_path).unlink()
        else:
            job_is_rerun = False

        # CrYOLO requires running in the project directory or job directory
        job_dir_search = re.search(".+/job[0-9]+/", cryolo_params.output_path)
        job_num_search = re.search("/job[0-9]+/", cryolo_params.output_path)
        if job_dir_search and job_num_search:
            job_dir = Path(job_dir_search[0])
            job_number = int(job_num_search[0][4:7])
        else:
            self.log.warning(f"Invalid job directory in {cryolo_params.output_path}")
            rw.transport.nack(header)
            return
        job_dir.mkdir(parents=True, exist_ok=True)

        # Construct a command to run cryolo with the given parameters
        command = cryolo_params.cryolo_command.split()
        command.extend((["--conf", str(job_dir / "cryolo_config.json")]))
        command.extend((["-o", str(job_dir)]))
        if cryolo_params.on_the_fly and cryolo_params.experiment_type == "spa":
            command.extend((["--otf"]))
        if cryolo_params.experiment_type == "tomography":
            command.extend(
                (
                    [
                        "--tomogram",
                        "-tsr",
                        str(cryolo_params.tomo_tracing_search_range),
                        "-tmem",
                        str(cryolo_params.tomo_tracing_missing_frames),
                        "-tmin",
                        str(cryolo_params.tomo_tracing_min_frames),
                        "--gpus",
                        "0",
                    ]
                )
            )

        cryolo_flags = {
            "cryolo_model_weights": "--weights",
            "input_path": "-i",
            "cryolo_threshold": "--threshold",
            "cryolo_gpus": "--gpu",
            "min_distance": "--distance",
            "normalization_margin": "--norm_margin",
        }

        for k, v in cryolo_params.model_dump().items():
            if (v not in [None, ""]) and (k in cryolo_flags):
                command.extend((cryolo_flags[k], str(v)))

        self.log.info(
            f"Input: {cryolo_params.input_path} "
            + f"Output: {cryolo_params.output_path}"
        )

        # Create the config file
        cryolo_params.relion_options.cryolo_config_file = str(
            job_dir / "cryolo_config.json"
        )
        config: dict = {
            "model": {
                "architecture": "PhosaurusNet",
                "input_size": 1024,
                "max_box_per_image": 600,
                "norm": "STANDARD",
                "num_patches": 1,
            },
            "other": {"log_path": "logs/"},
        }
        if cryolo_params.experiment_type == "spa":
            config["model"]["filter"] = [0.1, "filtered"]
        config["model"]["anchors"] = [
            cryolo_params.cryolo_box_size,
            cryolo_params.cryolo_box_size,
        ]
        if not (job_dir / "cryolo_config.json").is_file():
            with open(job_dir / "cryolo_config.json", "w") as config_file:
                json.dump(config, config_file)

        # Run cryolo prediction
        result = subprocess.run(command, cwd=job_dir, capture_output=True)

        # Remove cryosparc file to minimise crashes
        (job_dir / "CRYOSPARC/cryosparc.star").unlink(missing_ok=True)

        # Read in the stdout from cryolo
        self.parse_cryolo_output(result.stdout.decode("utf8", "replace"))
        # Register the cryolo job with the node creator
        self.log.info(f"Sending {self.job_type} to node creator")
        node_creator_parameters: dict[str, Any] = {
            "job_type": self.job_type,
            "input_file": cryolo_params.input_path,
            "output_file": cryolo_params.output_path,
            "relion_options": dict(cryolo_params.relion_options),
            "command": " ".join(command),
            "stdout": result.stdout.decode("utf8", "replace"),
            "stderr": result.stderr.decode("utf8", "replace"),
            "experiment_type": cryolo_params.experiment_type,
        }
        if (
            result.returncode
            and result.stderr.decode("utf8", "replace").split("\n")[-1]
            == "IndexError: list index out of range"
        ):
            # If Cryolo failed because there are no picks then consider it a success
            result.returncode = 0

        if result.returncode:
            node_creator_parameters["success"] = False
        else:
            node_creator_parameters["success"] = True
        if not job_is_rerun:
            # Only do the node creator inserts for new files
            rw.send_to("node_creator", node_creator_parameters)

        # End here if the command failed
        if result.returncode:
            self.log.error(
                f"crYOLO failed with exitcode {result.returncode}:\n"
                + result.stderr.decode("utf8", "replace")
            )
            rw.transport.nack(header)
            return

        # If this is tomo then make an image and stop here
        if cryolo_params.experiment_type == "tomography":
            # Insert the picked tomogram into ISPyB
            ispyb_parameters_tomo = {
                "ispyb_command": "insert_processed_tomogram",
                "file_path": cryolo_params.output_path,
                "processing_type": "Picked",
            }
            rw.send_to("ispyb_connector", ispyb_parameters_tomo)

            # Forward results to images service
            self.log.info("Sending to images service")
            movie_parameters = {
                "image_command": "picked_particles_3d_apng",
                "file": cryolo_params.input_path,
                "coordinates_file": cryolo_params.output_path,
                "box_size": cryolo_params.cryolo_box_size,
            }
            central_slice_parameters = {
                "image_command": "picked_particles_3d_central_slice",
                "file": cryolo_params.input_path,
                "coordinates_file": cryolo_params.output_path,
                "box_size": cryolo_params.cryolo_box_size,
            }
            rw.send_to("images", movie_parameters)
            rw.send_to("images", central_slice_parameters)

            # Remove unnecessary files
            eman_file = (
                job_dir
                / f"EMAN_3D/{Path(cryolo_params.output_path).with_suffix('.box').name}"
            )
            eman_file.unlink(missing_ok=True)

            self.log.info(
                f"Done {self.job_type} {cryolo_params.experiment_type} "
                f"for {cryolo_params.input_path}."
            )
            rw.transport.ack(header)
            return

        # Read in the cbox file for particle selection and finding sizes
        try:
            cbox_file = cif.read_file(
                str(
                    job_dir
                    / f"CBOX/{Path(cryolo_params.output_path).with_suffix('.cbox').name}"
                )
            )
            cbox_block = cbox_file.find_block("cryolo")
            cbox_sizes = np.append(
                np.array(cbox_block.find_loop("_EstWidth"), dtype=float),
                np.array(cbox_block.find_loop("_EstHeight"), dtype=float),
            )
            cbox_confidence = np.append(
                np.array(cbox_block.find_loop("_Confidence"), dtype=float),
                np.array(cbox_block.find_loop("_Confidence"), dtype=float),
            )

            # Select only a fraction of particles based on confidence if requested
            if (
                cryolo_params.retained_fraction < 1
                and cryolo_params.retained_fraction > 0
            ):
                particles_confidence = np.array(
                    cbox_block.find_loop("_Confidence"), dtype=float
                )
                if len(particles_confidence) <= cryolo_params.min_particles:
                    cryolo_threshold = 0.1
                elif (
                    len(particles_confidence) * cryolo_params.retained_fraction
                    < cryolo_params.min_particles
                ):
                    cryolo_threshold = sorted(particles_confidence, reverse=True)[
                        cryolo_params.min_particles
                    ]
                else:
                    cryolo_threshold = np.quantile(
                        particles_confidence, 1 - cryolo_params.retained_fraction
                    )

                self.log.info(
                    f"Selecting particles with confidence above {cryolo_threshold}"
                )

                particles_x_all = np.array(
                    cbox_block.find_loop("_CoordinateX"), dtype=float
                )
                particles_y_all = np.array(
                    cbox_block.find_loop("_CoordinateY"), dtype=float
                )
                box_size_x_all = np.array(cbox_block.find_loop("_Width"), dtype=float)
                box_size_y_all = np.array(cbox_block.find_loop("_Height"), dtype=float)

                thresholded_x = (particles_x_all + box_size_x_all / 2)[
                    particles_confidence > cryolo_threshold
                ]
                thresholded_y = (particles_y_all + box_size_y_all / 2)[
                    particles_confidence > cryolo_threshold
                ]

                # Update the number of particles
                self.number_of_particles = len(thresholded_x)

                # Rewrite the star file with only the selected particles
                with open(job_dir / cryolo_params.output_path, "w") as particles_star:
                    particles_star.write(
                        "\ndata_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n"
                    )
                    for particle in range(len(thresholded_x)):
                        particles_star.write(
                            f"{thresholded_x[particle]} {thresholded_y[particle]}\n"
                        )
            else:
                cryolo_threshold = cryolo_params.cryolo_threshold

            # Update the relion options, but this value is currently unusable
            cryolo_params.relion_options.cryolo_threshold = cryolo_threshold

            # Get the diameters of the particles in Angstroms for Murfey
            cryolo_particle_sizes = (
                cbox_sizes[cbox_confidence > cryolo_threshold]
                * cryolo_params.pixel_size
            )
        except (FileNotFoundError, OSError, AttributeError):
            self.number_of_particles = 0
            cryolo_particle_sizes = []

        # Forward results to ISPyB
        ispyb_parameters_spa: dict = {
            "ispyb_command": "buffer",
            "buffer_lookup": {"motion_correction_id": cryolo_params.mc_uuid},
            "buffer_command": {"ispyb_command": "insert_particle_picker"},
            "buffer_store": cryolo_params.picker_uuid,
            "particle_picking_template": cryolo_params.cryolo_model_weights,
            "number_of_particles": self.number_of_particles,
            "summary_image_full_path": str(
                Path(cryolo_params.output_path).with_suffix(".jpeg")
            ),
        }
        if cryolo_params.particle_diameter:
            ispyb_parameters_spa["particle_diameter"] = cryolo_params.particle_diameter
        self.log.info(f"Sending to ispyb {ispyb_parameters_spa}")
        rw.send_to("ispyb_connector", ispyb_parameters_spa)

        # Extract results for images service
        try:
            with open(cryolo_params.output_path, "r") as coords_file:
                coords = [line.split() for line in coords_file][6:]
        except FileNotFoundError:
            coords = []

        # Forward results to images service
        self.log.info("Sending to images service")
        images_parameters = {
            "image_command": "picked_particles",
            "file": cryolo_params.input_path,
            "coordinates": coords,
            "pixel_size": cryolo_params.pixel_size,
            "diameter": (
                cryolo_params.particle_diameter
                if cryolo_params.particle_diameter
                else cryolo_params.cryolo_box_size
            ),
            "outfile": str(Path(cryolo_params.output_path).with_suffix(".jpeg")),
        }
        rw.send_to("images", images_parameters)

        # Gather results needed for particle extraction
        extraction_params: dict[str, Any] = {
            "ctf_values": cryolo_params.ctf_values,
            "micrographs_file": cryolo_params.input_path,
            "coord_list_file": cryolo_params.output_path,
        }
        extraction_params["extract_file"] = str(
            Path(
                re.sub(
                    "MotionCorr/job002/.+",
                    f"Extract/job{job_number + 1:03}/Movies/",
                    cryolo_params.input_path,
                )
            )
            / (Path(cryolo_params.input_path).stem + "_extract.star")
        )

        # Forward results to murfey
        self.log.info("Sending to Murfey for particle extraction")
        rw.send_to(
            "murfey_feedback",
            {
                "register": "picked_particles",
                "motion_correction_id": cryolo_params.mc_uuid,
                "micrograph": cryolo_params.input_path,
                "particle_diameters": list(cryolo_particle_sizes),
                "particle_count": len(cryolo_particle_sizes),
                "resolution": cryolo_params.ctf_values["CtfMaxResolution"],
                "astigmatism": cryolo_params.ctf_values["DefocusV"]
                - cryolo_params.ctf_values["DefocusU"],
                "defocus": (
                    cryolo_params.ctf_values["DefocusU"]
                    + cryolo_params.ctf_values["DefocusV"]
                )
                / 2,
                "extraction_parameters": extraction_params,
            },
        )

        # Remove unnecessary files
        eman_file = (
            job_dir / f"EMAN/{Path(cryolo_params.output_path).with_suffix('.box').name}"
        )
        eman_file.unlink(missing_ok=True)

        self.log.info(
            f"Done {self.job_type} {cryolo_params.experiment_type} "
            f"for {cryolo_params.input_path}."
        )
        rw.transport.ack(header)
