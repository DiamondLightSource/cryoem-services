from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, ValidationError
from topaz.algorithms import non_maximum_suppression
from topaz.extract import score_images
from topaz.stats import normalize_images
from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.util.models import MockRW
from cryoemservices.util.relion_service_options import RelionServiceOptions


class TopazPickParameters(BaseModel):
    input_path: str = Field(..., min_length=1)
    output_path: str = Field(..., min_length=1)
    pixel_size: float
    particle_diameter: float
    scale: int = 8
    log_threshold: float = -6
    topaz_model: str = "resnet16"
    use_gpu: bool = False
    mc_uuid: int
    picker_uuid: int
    relion_options: RelionServiceOptions
    ctf_values: dict = {}


class TopazPick(CommonService):
    """
    A service that runs topaz particle picking
    """

    # Logger name
    _logger_name = "cryoemservices.services.topaz_pick"

    # Job name
    job_type = "relion.autopick.topaz.pick"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Topaz service starting")
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "topaz",
            self.topaz,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def topaz(self, rw, header: dict, message: dict):
        """
        Main function which interprets received messages, runs topaz
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

        try:
            if isinstance(message, dict):
                topaz_params = TopazPickParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                topaz_params = TopazPickParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"topaz parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        # Check if this file has been run before
        if (
            Path(topaz_params.output_path).is_file()
            and not Path(topaz_params.output_path).with_suffix(".tmp").is_file()
        ):
            job_is_rerun = True
            Path(topaz_params.output_path).unlink()
        else:
            job_is_rerun = False
            Path(topaz_params.output_path).with_suffix(".tmp").touch(exist_ok=True)

        # Determine the project directory or job directory
        job_num_search = re.search("/job[0-9]+/", topaz_params.output_path)
        if job_num_search:
            job_number = int(job_num_search[0][4:7])
        else:
            self.log.warning(f"Invalid job directory in {topaz_params.output_path}")
            rw.transport.nack(header)
            return
        Path(topaz_params.output_path).parent.mkdir(parents=True, exist_ok=True)

        # Construct a command to run topaz with the given parameters
        self.log.info(
            f"Input: {topaz_params.input_path} Output: {topaz_params.output_path}"
        )

        normalize_images(
            [topaz_params.input_path],
            str(Path(topaz_params.output_path).parent / "scaled"),
            num_workers=0,
            scale=topaz_params.scale,
            affine=False,
            niters=100,
            alpha=900,
            beta=1,
            sample=10,
            metadata=False,
            formats=["mrc"],
            use_cuda=topaz_params.use_gpu,
            verbose=False,
        )
        stream_from_scoring = score_images(
            topaz_params.topaz_model,
            [
                f"{Path(topaz_params.output_path).parent}"
                f"/scaled/{Path(topaz_params.input_path).name}"
            ],
            device=0 if topaz_params.use_gpu else -1,
            batch_size=1,
        )
        scores_for_picking = [v for k, v in stream_from_scoring][0]

        scaled_radius_pixels = int(
            topaz_params.particle_diameter
            / topaz_params.pixel_size
            / topaz_params.scale
            / 2
        )
        n_particles = topaz_extract_particles(
            scores_for_picking, topaz_params, radius=scaled_radius_pixels
        )
        topaz_params.relion_options.particle_diameter = topaz_params.particle_diameter
        self.log.info(
            f"Extracted {n_particles} particles from {topaz_params.input_path}"
        )

        # Remove the scaled image topaz makes
        (
            Path(topaz_params.output_path).parent
            / "scaled"
            / Path(topaz_params.input_path).name
        ).unlink(missing_ok=True)

        # Construct the command we have replicated
        command = [
            "topaz",
            "extract",
            "-m",
            topaz_params.topaz_model,
            "-r",
            str(scaled_radius_pixels),
            "-s",
            str(topaz_params.scale),
            "-t",
            str(topaz_params.log_threshold),
            "--per-micrograph",
            "-o",
            topaz_params.output_path,
            topaz_params.input_path,
        ]

        # Register the topaz job with the node creator
        self.log.info(f"Sending {self.job_type} to node creator")
        node_creator_parameters: dict[str, Any] = {
            "job_type": self.job_type,
            "input_file": topaz_params.input_path,
            "output_file": topaz_params.output_path,
            "relion_options": dict(topaz_params.relion_options),
            "command": " ".join(command),
            "stdout": "",
            "stderr": "",
            "success": True,
        }
        if not job_is_rerun:
            # Only do the node creator inserts for new files
            rw.send_to("node_creator", node_creator_parameters)
            # Remove tmp file after requesting node creation
            Path(topaz_params.output_path).with_suffix(".tmp").unlink(missing_ok=True)

        # Read picks for images service
        try:
            with open(topaz_params.output_path, "r") as coords_file:
                coords = [line.split()[:2] for line in coords_file][6:]
        except FileNotFoundError:
            coords = []

        # Forward results to images service
        self.log.info("Sending to images service")
        images_parameters = {
            "image_command": "picked_particles",
            "file": topaz_params.input_path,
            "coordinates": coords,
            "pixel_size": topaz_params.pixel_size,
            "diameter": topaz_params.particle_diameter,
            "outfile": str(Path(topaz_params.output_path).with_suffix(".jpeg")),
        }
        rw.send_to("images", images_parameters)

        # Gather results needed for particle extraction
        extraction_params = {
            "ctf_values": topaz_params.ctf_values,
            "micrographs_file": topaz_params.input_path,
            "coord_list_file": topaz_params.output_path,
            "extract_file": str(
                Path(
                    re.sub(
                        "MotionCorr/job002/.+",
                        f"Extract/job{job_number + 1:03}/Movies/",
                        topaz_params.input_path,
                    )
                )
                / (Path(topaz_params.input_path).stem + "_extract.star")
            ),
        }

        # Forward results to murfey
        self.log.info("Sending to Murfey for particle extraction")
        rw.send_to(
            "murfey_feedback",
            {
                "register": "picked_particles",
                "motion_correction_id": topaz_params.mc_uuid,
                "micrograph": topaz_params.input_path,
                "particle_diameters": [topaz_params.particle_diameter] * n_particles,
                "extraction_parameters": extraction_params,
            },
        )

        # Forward results to ISPyB
        ispyb_parameters_spa: dict = {
            "ispyb_command": "buffer",
            "buffer_lookup": {"motion_correction_id": topaz_params.mc_uuid},
            "buffer_command": {"ispyb_command": "insert_particle_picker"},
            "buffer_store": topaz_params.picker_uuid,
            "particle_picking_template": topaz_params.topaz_model,
            "number_of_particles": n_particles,
            "summary_image_full_path": str(
                Path(topaz_params.output_path).with_suffix(".jpeg")
            ),
            "particle_diameter": topaz_params.particle_diameter,
        }
        self.log.info(f"Sending to ispyb {ispyb_parameters_spa}")
        rw.send_to("ispyb_connector", ispyb_parameters_spa)

        self.log.info(f"Done {self.job_type} for {topaz_params.input_path}.")
        rw.transport.ack(header)


def topaz_extract_particles(
    scores_for_picking, topaz_params: TopazPickParameters, radius: int
) -> int:
    # Extract coordinates using radius
    particle_scores, coords = non_maximum_suppression(
        scores_for_picking,
        radius,
        threshold=topaz_params.log_threshold,
    )
    # Scale the coordinates
    scaled_coords = np.round(coords * topaz_params.scale).astype(int)
    # Save the coordinates into a Relion-type star file
    with open(topaz_params.output_path, "w") as outfile:
        outfile.write(
            "data_\n\nloop_\n"
            "_rlnCoordinateX\n_rlnCoordinateY\n_rlnAutopickFigureOfMerit\n"
        )
        for particle in range(len(particle_scores)):
            outfile.write(
                f"{scaled_coords[particle, 0]}  {scaled_coords[particle, 1]}  "
                f"{particle_scores[particle]}\n"
            )
    return len(particle_scores)
