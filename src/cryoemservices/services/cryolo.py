from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np
import workflows.recipe
from gemmi import cif
from pydantic import BaseModel, Field, ValidationError
from workflows.services.common_service import CommonService

from cryoemservices.util.models import MockRW
from cryoemservices.util.relion_service_options import RelionServiceOptions


class CryoloParameters(BaseModel):
    input_path: str = Field(..., min_length=1)
    output_path: str = Field(..., min_length=1)
    pixel_size: float
    cryolo_config_file: str = "/dls_sw/apps/EM/crYOLO/phosaurus_models/config.json"
    cryolo_model_weights: str = (
        "/dls_sw/apps/EM/crYOLO/phosaurus_models/gmodel_phosnet_202005_N63_c17.h5"
    )
    cryolo_threshold: float = 0.3
    retained_fraction: float = 1
    min_particles: int = 30
    cryolo_command: str = "cryolo_predict.py"
    particle_diameter: Optional[float] = None
    on_the_fly: bool = True
    mc_uuid: int
    picker_uuid: int
    relion_options: RelionServiceOptions
    ctf_values: dict = {}


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
            "cryolo",
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
            rw = MockRW(self._transport)
            rw.recipe_step = {"parameters": message["parameters"]}
            message = message["content"]

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
        command.extend((["--conf", cryolo_params.cryolo_config_file]))
        command.extend((["-o", str(job_dir)]))
        if cryolo_params.on_the_fly:
            command.extend((["--otf"]))

        cryolo_flags = {
            "cryolo_model_weights": "--weights",
            "input_path": "-i",
            "cryolo_threshold": "--threshold",
            "cryolo_gpus": "--gpu",
        }

        for k, v in cryolo_params.model_dump().items():
            if (v not in [None, ""]) and (k in cryolo_flags):
                command.extend((cryolo_flags[k], str(v)))

        self.log.info(
            f"Input: {cryolo_params.input_path} "
            + f"Output: {cryolo_params.output_path}"
        )

        # Run cryolo
        result = subprocess.run(command, cwd=job_dir, capture_output=True)

        # Remove cryosparc file to minimise crashes
        (job_dir / "CRYOSPARC/cryosparc.star").unlink(missing_ok=True)

        # Read in the stdout from cryolo
        self.parse_cryolo_output(result.stdout.decode("utf8", "replace"))

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
            cryolo_params.relion_options.cryolo_threshold = cryolo_threshold

            # Get the diameters of the particles in Angstroms for Murfey
            cryolo_particle_sizes = (
                cbox_sizes[cbox_confidence > cryolo_threshold]
                * cryolo_params.pixel_size
            )
        except (FileNotFoundError, OSError, AttributeError):
            cryolo_particle_sizes = []

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
            if isinstance(rw, MockRW):
                rw.transport.send(
                    destination="node_creator",
                    message={"parameters": node_creator_parameters, "content": "dummy"},
                )
            else:
                rw.send_to("node_creator", node_creator_parameters)

        # End here if the command failed
        if result.returncode:
            self.log.error(
                f"crYOLO failed with exitcode {result.returncode}:\n"
                + result.stderr.decode("utf8", "replace")
            )
            rw.transport.nack(header)
            return

        # Forward results to ISPyB
        ispyb_parameters = {
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
            ispyb_parameters["particle_diameter"] = cryolo_params.particle_diameter
        self.log.info(f"Sending to ispyb {ispyb_parameters}")
        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="ispyb_connector",
                message={
                    "parameters": ispyb_parameters,
                    "content": {"dummy": "dummy"},
                },
            )
        else:
            rw.send_to("ispyb_connector", ispyb_parameters)

        # Extract results for images service
        try:
            with open(cryolo_params.output_path, "r") as coords_file:
                coords = [line.split() for line in coords_file][6:]
        except FileNotFoundError:
            coords = []

        # Forward results to images service
        self.log.info("Sending to images service")
        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="images",
                message={
                    "image_command": "picked_particles",
                    "file": cryolo_params.input_path,
                    "coordinates": coords,
                    "pixel_size": cryolo_params.pixel_size,
                    "diameter": (
                        cryolo_params.particle_diameter
                        if cryolo_params.particle_diameter
                        else 160
                    ),
                    "outfile": str(
                        Path(cryolo_params.output_path).with_suffix(".jpeg")
                    ),
                },
            )
        else:
            rw.send_to(
                "images",
                {
                    "image_command": "picked_particles",
                    "file": cryolo_params.input_path,
                    "coordinates": coords,
                    "pixel_size": cryolo_params.pixel_size,
                    "diameter": (
                        cryolo_params.particle_diameter
                        if cryolo_params.particle_diameter
                        else 160
                    ),
                    "outfile": str(
                        Path(cryolo_params.output_path).with_suffix(".jpeg")
                    ),
                },
            )

        # Gather results needed for particle extraction
        extraction_params = {
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
        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="murfey_feedback",
                message={
                    "register": "picked_particles",
                    "motion_correction_id": cryolo_params.mc_uuid,
                    "micrograph": cryolo_params.input_path,
                    "particle_diameters": list(cryolo_particle_sizes),
                    "extraction_parameters": extraction_params,
                },
            )
        else:
            rw.send_to(
                "murfey_feedback",
                {
                    "register": "picked_particles",
                    "motion_correction_id": cryolo_params.mc_uuid,
                    "micrograph": cryolo_params.input_path,
                    "particle_diameters": list(cryolo_particle_sizes),
                    "extraction_parameters": extraction_params,
                },
            )

        self.log.info(f"Done {self.job_type} for {cryolo_params.input_path}.")
        rw.transport.ack(header)
