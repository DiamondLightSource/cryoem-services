from __future__ import annotations

import ast
from pathlib import Path

import mrcfile
import numpy as np
import workflows.recipe
from gemmi import cif
from pydantic import BaseModel, Field, ValidationError, field_validator
from workflows.services.common_service import CommonService

from cryoemservices.util.models import MockRW
from cryoemservices.util.relion_service_options import (
    RelionServiceOptions,
    update_relion_options,
)
from cryoemservices.util.tomo_output_files import _get_tilt_name_v5_12


class ExtractSubTomoParameters(BaseModel):
    cbox_3d_file: str = Field(..., min_length=1)
    tilt_alignment_file: str = Field(..., min_length=1)
    newstack_file: str = Field(..., min_length=1)
    output_star: str = Field(..., min_length=1)
    scaled_tomogram_shape: list[int] | str
    pixel_size: float
    particle_diameter: float = 0
    boxsize: int = 256
    small_boxsize: int = 64
    binning: int = 8
    min_frames: int = 1
    maximum_dose: int = -1
    tomogram_binning: int = 4
    invert_contrast: bool = True
    bg_radius: int = -1
    relion_options: RelionServiceOptions

    @field_validator("scaled_tomogram_shape")
    @classmethod
    def check_shape_is_3d(cls, v):
        if not len(v):
            raise ValueError("Tomogram shape not given")
        if type(v) is str:
            shape_list = ast.literal_eval(v)
        else:
            shape_list = v
        if len(shape_list) != 3:
            raise ValueError("Tomogram shape must be 3D")
        return shape_list


class ExtractSubTomo(CommonService):
    """
    A service for extracting particles from cryolo autopicking for tomograms
    """

    # Human readable service name
    _service_name = "ExtractSubTomo"

    # Logger name
    _logger_name = "cryoemservices.services.extract_subtomo"

    # Job name
    job_type = "relion.pseudosubtomo"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Extract service starting")
        workflows.recipe.wrap_subscribe(
            self._transport,
            self._environment["queue"] or "extract_subtomo",
            self.extract_subtomo,
            acknowledgement=True,
            log_extender=self.extend_log,
            allow_non_recipe_messages=True,
        )

    def extract_subtomo(self, rw, header: dict, message: dict):
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
                extract_subtomo_params = ExtractSubTomoParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                extract_subtomo_params = ExtractSubTomoParameters(
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
            f"Inputs: {extract_subtomo_params.tilt_alignment_file}, "
            f"{extract_subtomo_params.cbox_3d_file} "
            f"Output: {extract_subtomo_params.output_star}"
        )

        # Update the relion options and get box sizes
        extract_subtomo_params.relion_options = update_relion_options(
            extract_subtomo_params.relion_options, dict(extract_subtomo_params)
        )

        # Make sure the output directory exists
        if not Path(extract_subtomo_params.output_star).parent.exists():
            Path(extract_subtomo_params.output_star).parent.mkdir(parents=True)

        # If no background radius set diameter as 75% of box
        if extract_subtomo_params.bg_radius == -1:
            extract_subtomo_params.bg_radius = round(
                0.375 * extract_subtomo_params.relion_options.small_boxsize
            )

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
        particles_z = (
            np.array(coords_block.find_loop("_CoordinateZ"), dtype=float) + pick_radius
        )

        # Get the shifts between tilts
        shift_data = np.genfromtxt(extract_subtomo_params.tilt_alignment_file)
        # tilt_ids = shift_data[:, 0].astype(int)
        refined_tilt_axis = float(shift_data[0, 1])
        x_shifts = shift_data[:, 3].astype(float)
        y_shifts = shift_data[:, 4].astype(float)
        tilt_count = len(x_shifts)

        # Rotation around the tilt axis is about (0, height/2)
        # Or possibly not, sometimes seems to be (width/2, height/2), needs exploration
        centre_x = 0
        centre_y = float(extract_subtomo_params.scaled_tomogram_shape[1]) / 2
        tilt_axis_radians = (90 - refined_tilt_axis) * np.pi / 180

        x_coords_in_tilts = centre_x + (
            (particles_x - centre_x) * np.cos(tilt_axis_radians)
            - (particles_y - centre_y) * np.sin(tilt_axis_radians)
        )
        y_coords_in_tilts = centre_y + (
            (particles_x - centre_x) * np.sin(tilt_axis_radians)
            + (particles_y - centre_y) * np.cos(tilt_axis_radians)
        )
        x_coords_in_tilts *= extract_subtomo_params.tomogram_binning
        y_coords_in_tilts *= extract_subtomo_params.tomogram_binning

        # Downscaling dimensions
        extract_subtomo_params.relion_options.pixel_size_downscaled = (
            extract_subtomo_params.pixel_size
            * extract_subtomo_params.relion_options.boxsize
            / extract_subtomo_params.relion_options.small_boxsize
        )
        extract_width = round(extract_subtomo_params.relion_options.boxsize / 2)
        scaled_extract_width = round(
            extract_subtomo_params.relion_options.small_boxsize / 2
        )
        box_len = extract_subtomo_params.relion_options.small_boxsize
        pixel_size = extract_subtomo_params.relion_options.pixel_size_downscaled

        # Distance of each pixel from the centre for background normalization
        grid_indexes = np.meshgrid(
            np.arange(2 * scaled_extract_width),
            np.arange(2 * scaled_extract_width),
        )
        distance_from_centre = np.sqrt(
            (grid_indexes[0] - scaled_extract_width + 0.5) ** 2
            + (grid_indexes[1] - scaled_extract_width + 0.5) ** 2
        )

        # Read in tilt images
        self.log.info("Reading tilt images")
        tilt_images = []
        with open(extract_subtomo_params.newstack_file) as ns_file:
            while True:
                line = ns_file.readline()
                if not line:
                    break
                elif line.startswith("/"):
                    tilt_name = line.strip()
                    with mrcfile.open(tilt_name) as mrc:
                        tilt_images.append(mrc.data)

        for particle in range(len(particles_x)):
            output_mrc_stack = np.array([])
            for tilt in range(tilt_count):
                # Extract the particle image and pad the edges if it is not square
                x_left_pad = 0
                x_right_pad = 0
                y_top_pad = 0
                y_bot_pad = 0

                x_left = round(
                    x_coords_in_tilts[particle] - extract_width - x_shifts[tilt]
                )
                if x_left < 0:
                    x_left_pad = -x_left
                    x_left = 0
                x_right = round(
                    x_coords_in_tilts[particle] + extract_width - x_shifts[tilt]
                )
                if (
                    x_right
                    >= extract_subtomo_params.scaled_tomogram_shape[0]
                    * extract_subtomo_params.tomogram_binning
                ):
                    x_right_pad = (
                        x_right - extract_subtomo_params.scaled_tomogram_shape[0]
                    )
                    x_right = (
                        extract_subtomo_params.scaled_tomogram_shape[0]
                        * extract_subtomo_params.tomogram_binning
                    )
                y_top = round(
                    y_coords_in_tilts[particle] - extract_width - y_shifts[tilt]
                )
                if y_top < 0:
                    y_top_pad = -y_top
                    y_top = 0
                y_bot = round(
                    y_coords_in_tilts[particle] + extract_width - y_shifts[tilt]
                )
                if (
                    y_bot
                    >= extract_subtomo_params.scaled_tomogram_shape[1]
                    * extract_subtomo_params.tomogram_binning
                ):
                    y_bot_pad = y_bot - extract_subtomo_params.scaled_tomogram_shape[1]
                    y_bot = (
                        extract_subtomo_params.scaled_tomogram_shape[1]
                        * extract_subtomo_params.tomogram_binning
                    )

                if y_bot <= y_top or x_left >= x_right:
                    self.log.warning(f"Invalid {tilt} for particle {particle}")
                    continue

                particle_subimage = tilt_images[tilt][y_top:y_bot, x_left:x_right]
                particle_subimage = np.pad(
                    particle_subimage,
                    ((y_bot_pad, y_top_pad), (x_left_pad, x_right_pad)),
                    mode="edge",
                )

                # Flip all the values on inversion
                if extract_subtomo_params.invert_contrast:
                    particle_subimage = -1 * particle_subimage

                # Downscale the image size
                subimage_ft = np.fft.fftshift(np.fft.fft2(particle_subimage))
                deltax = (
                    subimage_ft.shape[0]
                    - extract_subtomo_params.relion_options.small_boxsize
                )
                deltay = (
                    subimage_ft.shape[1]
                    - extract_subtomo_params.relion_options.small_boxsize
                )
                particle_subimage = np.real(
                    np.fft.ifft2(
                        np.fft.ifftshift(
                            subimage_ft[
                                deltax // 2 : subimage_ft.shape[0] - deltax // 2,
                                deltay // 2 : subimage_ft.shape[1] - deltay // 2,
                            ]
                        )
                    )
                )

                # Background normalisation
                bg_region = (
                    distance_from_centre
                    > np.ones(np.shape(particle_subimage))
                    * extract_subtomo_params.bg_radius
                )
                bg_mean = np.mean(particle_subimage[bg_region])
                bg_std = np.std(particle_subimage[bg_region])
                particle_subimage = (particle_subimage - bg_mean) / bg_std

                # Add to output stack
                if len(output_mrc_stack):
                    output_mrc_stack = np.append(
                        output_mrc_stack, [particle_subimage], axis=0
                    )
                else:
                    output_mrc_stack = np.array([particle_subimage], dtype=np.float32)

            if not len(output_mrc_stack):
                self.log.warning(f"Could not extract particle {particle}")
                continue

            # Produce the mrc file of the extracted particles
            output_mrc_file = (
                Path(extract_subtomo_params.output_star).parent
                / f"{particle}_stack2d.mrcs"
            )
            self.log.info(f"Extracted particle {particle} of {len(particles_x)}")
            with mrcfile.new(str(output_mrc_file), overwrite=True) as mrc:
                mrc.set_data(output_mrc_stack.astype(np.float32))
                mrc.header.mx = box_len
                mrc.header.my = box_len
                mrc.header.mz = 1
                mrc.header.cella.x = pixel_size * box_len
                mrc.header.cella.y = pixel_size * box_len
                mrc.header.cella.z = 1

        # Construct the output star file
        extracted_parts_doc = cif.Document()
        extracted_parts_block = extracted_parts_doc.add_new_block("particles")
        extracted_parts_loop = extracted_parts_block.init_loop(
            "_rln",
            [
                "CenteredCoordinateXAngst",
                "CenteredCoordinateYAngst",
                "CenteredCoordinateZAngst",
                "OpticsGroup",
                "TomoParticleName",
                "TomoVisibleFrames",
                "ImageName",
                "OriginXAngst",
                "OriginYAngst",
                "OriginZAngst",
            ],
        )
        frames = "?????"
        for particle in range(len(particles_x)):
            extracted_parts_loop.add_row(
                [
                    str(
                        float(particles_x[particle])
                        - float(extract_subtomo_params.scaled_tomogram_shape[2])
                        / 2
                        * extract_subtomo_params.tomogram_binning
                    ),
                    str(
                        float(particles_y[particle])
                        - float(extract_subtomo_params.scaled_tomogram_shape[1])
                        / 2
                        * extract_subtomo_params.tomogram_binning
                    ),
                    str(
                        float(particles_z[particle])
                        - float(extract_subtomo_params.scaled_tomogram_shape[0])
                        / 2
                        * extract_subtomo_params.tomogram_binning
                    ),
                    "1",
                    f"{_get_tilt_name_v5_12(Path(extract_subtomo_params.tilt_alignment_file))}/{particle}",
                    f"[{frames}]",
                    f"{Path(extract_subtomo_params.output_star).parent}/{particle}_stack2d.mrcs",
                    "0.0",
                    "0.0",
                    "0.0",
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
            "results": {"box_size": box_len},
        }
        rw.send_to("node_creator", node_creator_parameters)

        self.log.info(
            f"Done {self.job_type} for {extract_subtomo_params.cbox_3d_file}."
        )
        rw.transport.ack(header)
