from pathlib import Path

import mrcfile
import numpy as np
from gemmi import cif
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm
from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.services.extract import enhance_single_particle
from cryoemservices.util.models import MockRW
from cryoemservices.util.relion_service_options import (
    RelionServiceOptions,
    update_relion_options,
)
from cryoemservices.util.tomo_output_files import _get_tilt_name_v5_12


class ExtractSubTomoParameters2D(BaseModel):
    cbox_3d_file: str = Field(..., min_length=1)
    tomogram: str = Field(..., min_length=1)
    output_star: str = Field(..., min_length=1)
    pixel_size: float
    particle_diameter: float = 0
    boxsize: int = 256
    batch_size: int = 5000
    relion_options: RelionServiceOptions


class ExtractSubTomoFor2D(CommonService):
    """
    A service for extracting 2D particles from cryolo autopicking for tomograms
    This extracts the particle in 3D, projects it to 2D
    and then processes it ready for SPA-like 2D classification
    """

    # Human readable service name
    _service_name = "ExtractSubTomo"

    # Logger name
    _logger_name = "cryoemservices.services.extract_subtomo"

    # Job name
    job_type = "relion.pseudosubtomo"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Sub-tomogram extraction service starting")
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "extract_subtomo",
            self.extract_subtomo_for_2d,
            acknowledgement=True,
            allow_non_recipe_messages=True,
        )

    def extract_subtomo_for_2d(self, rw, header: dict, message: dict):
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
                extract_subtomo_params = ExtractSubTomoParameters2D(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                extract_subtomo_params = ExtractSubTomoParameters2D(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"Extraction parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        self.log.info(
            f"Inputs: {extract_subtomo_params.tomogram}, "
            f"{extract_subtomo_params.cbox_3d_file} "
            f"Output: {extract_subtomo_params.output_star}"
        )

        # Update the relion options and get box sizes
        extract_subtomo_params.relion_options = update_relion_options(
            extract_subtomo_params.relion_options, dict(extract_subtomo_params)
        )
        if extract_subtomo_params.particle_diameter:
            extract_subtomo_params.boxsize = (
                extract_subtomo_params.relion_options.boxsize
            )

        # Make sure the output directory exists
        if not Path(extract_subtomo_params.output_star).parent.exists():
            Path(extract_subtomo_params.output_star).parent.mkdir(parents=True)

        # Find the locations of the particles
        coords_file = cif.read(extract_subtomo_params.cbox_3d_file)
        coords_block = coords_file.find_block("cryolo")
        pick_radius = float(coords_block.find_loop("_Width")[0]) / 2
        particles_x = (
            np.array(coords_block.find_loop("_CoordinateX"), dtype=float) + pick_radius
        )
        particles_y = (
            np.array(coords_block.find_loop("_CoordinateY"), dtype=float) + pick_radius
        )
        particles_z = np.array(coords_block.find_loop("_CoordinateZ"), dtype=float)

        # Read tomogram
        with mrcfile.open(extract_subtomo_params.tomogram) as mrc:
            tomogram_data = mrc.data

        # Extract at the same downscaling as during tomogram reconstruction
        extract_width = round(extract_subtomo_params.relion_options.boxsize / 2)

        output_mrc_stack = np.array([])
        for particle in tqdm(range(len(particles_x))):
            if (
                particles_y[particle] - extract_width < 0
                or particles_y[particle] + extract_width > tomogram_data.shape[1]
                or particles_x[particle] - extract_width < 0
                or particles_x[particle] + extract_width > tomogram_data.shape[2]
            ):
                self.log.info(
                    f"Skipping particle {particle} runs over the edge of the volume"
                )
                continue

            min_z = particles_z[particle] - extract_width
            max_z = particles_z[particle] + extract_width
            if min_z < 0:
                min_z = 0
            if max_z >= tomogram_data.shape[0]:
                max_z = tomogram_data.shape[0] - 1
            extract_vol = tomogram_data[
                round(float(min_z)) : round(float(max_z)),
                round(float(particles_y[particle] - extract_width)) : round(
                    float(particles_y[particle] + extract_width)
                ),
                round(float(particles_x[particle] - extract_width)) : round(
                    float(particles_x[particle] + extract_width)
                ),
            ]

            # Run projection along x axis
            flat_particle = extract_vol.mean(axis=0)
            particle_subimage, failure_reason = enhance_single_particle(
                particle_subimage=flat_particle,
                extract_width=extract_width,
                small_boxsize=extract_subtomo_params.boxsize,
                bg_radius=round(0.375 * extract_subtomo_params.boxsize),
                invert_contrast=True,
                downscale=False,
                norm=True,
                plane_fit=True,
            )
            if failure_reason:
                self.log.warning(
                    f"Extraction failed for {particle}. Reason was {failure_reason}."
                )
                continue

            # Add to output stack
            if len(output_mrc_stack):
                output_mrc_stack = np.append(
                    output_mrc_stack, [particle_subimage], axis=0
                )
            else:
                output_mrc_stack = np.array([particle_subimage], dtype=np.float32)

        # Produce the mrc file of the extracted particles
        output_mrc_file = (
            Path(extract_subtomo_params.output_star).parent
            / Path(extract_subtomo_params.tomogram).with_suffix(".mrcs").name
        )
        particle_count = np.shape(output_mrc_stack)[0]
        self.log.info(f"Extracted {particle_count} particles")
        with mrcfile.new(str(output_mrc_file), overwrite=True) as mrc:
            mrc.set_data(output_mrc_stack.astype(np.float32))
            mrc.header.mx = extract_subtomo_params.relion_options.boxsize
            mrc.header.my = extract_subtomo_params.relion_options.boxsize
            mrc.header.mz = 1
            mrc.header.cella.x = (
                extract_subtomo_params.pixel_size
                * extract_subtomo_params.relion_options.boxsize
            )
            mrc.header.cella.y = (
                extract_subtomo_params.pixel_size
                * extract_subtomo_params.relion_options.boxsize
            )
            mrc.header.cella.z = 1

        # Construct the output star file
        if not Path(extract_subtomo_params.output_star).is_file():
            extracted_parts_doc = cif.Document()
            optics_block = extracted_parts_doc.add_new_block("optics")
            optics_loop = optics_block.init_loop(
                "_rln",
                [
                    "Voltage",
                    "SphericalAberration",
                    "AmplitudeContrast",
                    "OpticsGroup",
                    "OpticsGroupName",
                    "CtfDataAreCtfPremultiplied",
                    "ImageDimensionality",
                    "ImagePixelSize",
                    "ImageSize",
                ],
            )
            optics_loop.add_row(
                [
                    "300.00",
                    "2.70",
                    "0.10",
                    "1",
                    "opticsGroup1",
                    "1",
                    "2",
                    str(extract_subtomo_params.pixel_size),
                    str(extract_subtomo_params.boxsize),
                ]
            )
            extracted_parts_block = extracted_parts_doc.add_new_block("particles")
            extracted_parts_loop = extracted_parts_block.init_loop(
                "_rln",
                [
                    "TomoName",
                    "OpticsGroup",
                    "TomoParticleName",
                    "ImageName",
                ],
            )
        else:
            extracted_parts_doc = cif.read_file(extract_subtomo_params.output_star)
            extracted_parts_block = extracted_parts_doc.find_block("particles")
            extracted_parts_loop = extracted_parts_block.find_loop(
                "_rlnTomoName"
            ).get_loop()
        for particle in range(particle_count):
            extracted_parts_loop.add_row(
                [
                    _get_tilt_name_v5_12(Path(extract_subtomo_params.tomogram)),
                    "1",
                    f"{particle}@{output_mrc_file}",
                    f"{particle}@{output_mrc_file}",
                ]
            )
        extracted_parts_doc.write_file(
            extract_subtomo_params.output_star, style=cif.Style.Simple
        )

        # Register the extract job with the node creator
        self.log.info(f"Sending {self.job_type} to node creator")
        node_creator_parameters = {
            "job_type": self.job_type,
            "input_file": extract_subtomo_params.cbox_3d_file,
            "output_file": extract_subtomo_params.output_star,
            "relion_options": dict(extract_subtomo_params.relion_options),
            "command": "",
            "stdout": "",
            "stderr": "",
        }
        rw.send_to("node_creator", node_creator_parameters)

        # Register the files needed for selection and batching
        self.log.info("Sending to particle selection")
        select_params = {
            "input_file": extract_subtomo_params.output_star,
            "batch_size": extract_subtomo_params.batch_size,
            "image_size": extract_subtomo_params.boxsize,
            "relion_options": dict(extract_subtomo_params.relion_options),
        }
        rw.send_to("select_particles", select_params)

        self.log.info(f"Done {self.job_type} for {extract_subtomo_params.cbox_3d_file}")
        rw.transport.ack(header)
