from __future__ import annotations

from pathlib import Path
from typing import List

import mrcfile
import numpy as np
import workflows.recipe
from pydantic import BaseModel, Field, ValidationError
from workflows.services.common_service import CommonService


class ReExtractParameters(BaseModel):
    motioncorr_name: str = Field(..., min_length=1)
    reextract_name: str = Field(..., min_length=1)
    particles_x: List[float]
    particles_y: List[float]
    number_of_particles: int
    scaled_pixel_size: float
    scaled_boxsize: int
    full_extract_width: int
    scaled_extract_width: int
    bg_radius: int = -1
    downscale: bool = True
    norm: bool = True
    invert_contrast: bool = True


class ReExtract(CommonService):
    """
    A service for extracting particles from a class for refinement
    """

    # Human readable service name
    _service_name = "ReExtract"

    # Logger name
    _logger_name = "cryoemservices.services.reextract"

    def initializing(self):
        """Subscribe to a queue. Received messages must be acknowledged."""
        self.log.info("Class extraction service starting")
        workflows.recipe.wrap_subscribe(
            self._transport,
            "extract_class",
            self.extract_class,
            acknowledgement=True,
            log_extender=self.extend_log,
            allow_non_recipe_messages=True,
        )

    def extract_class(self, rw, header: dict, message: dict):
        class MockRW:
            def dummy(self, *args, **kwargs):
                pass

        if not rw:
            print(
                "Incoming message is not a recipe message. Simple messages can be valid"
            )
            if (
                not isinstance(message, dict)
                or not message.get("parameters")
                or not message.get("content")
            ):
                self.log.error("Rejected invalid simple message")
                self._transport.nack(header)
                return
            self.log.debug("Received a simple message")

            # Create a wrapper-like object that can be passed to functions
            # as if a recipe wrapper was present.
            rw = MockRW()
            rw.transport = self._transport
            rw.recipe_step = {"parameters": message["parameters"]}
            rw.environment = {"has_recipe_wrapper": False}
            rw.set_default_channel = rw.dummy
            rw.send = rw.dummy
            message = message["content"]

        try:
            if isinstance(message, dict):
                extract_params = ReExtractParameters(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                extract_params = ReExtractParameters(
                    **{**rw.recipe_step.get("parameters", {})}
                )
        except (ValidationError, TypeError) as e:
            self.log.warning(
                f"Class extraction parameter validation failed for message: {message} "
                f"and recipe parameters: {rw.recipe_step.get('parameters', {})} "
                f"with exception: {e}"
            )
            rw.transport.nack(header)
            return

        self.log.info(
            f"Running extraction of particles for {extract_params.reextract_name}."
        )

        # If no background radius set diameter as 75% of box
        if extract_params.bg_radius == -1:
            extract_params.bg_radius = round(0.375 * extract_params.scaled_boxsize)

        # Distance of each pixel from the centre, compared to background radius
        grid_indexes = np.meshgrid(
            np.arange(2 * extract_params.scaled_extract_width),
            np.arange(2 * extract_params.scaled_extract_width),
        )
        distance_from_centre = np.sqrt(
            (grid_indexes[0] - extract_params.scaled_extract_width + 0.5) ** 2
            + (grid_indexes[1] - extract_params.scaled_extract_width + 0.5) ** 2
        )
        bg_region = (
            distance_from_centre
            > np.ones(np.shape(distance_from_centre)) * extract_params.bg_radius
        )
        # Fit background to a plane and subtract the plane from the image
        positions = [grid_indexes[0][bg_region], grid_indexes[1][bg_region]]
        # needs to create a matrix of the correct shape for  a*x + b*y + c plane fit
        data_size = len(positions[0])
        positions_matrix = np.hstack(
            (
                np.reshape(positions[0], (data_size, 1)),
                np.reshape(positions[1], (data_size, 1)),
            )
        )
        # this ones for c
        positions_matrix = np.hstack((np.ones((data_size, 1)), positions_matrix))
        try:
            flat_positions_matrix = np.dot(
                np.linalg.inv(np.dot(positions_matrix.transpose(), positions_matrix)),
                positions_matrix.transpose(),
            )
        except np.linalg.LinAlgError:
            print(f"Could not fit image plane for {extract_params.reextract_name}")
            rw.transport.nack(header)
            return

        # now we need the full grid across the image
        grid_matrix = np.hstack(
            (
                np.reshape(
                    grid_indexes[0], (4 * extract_params.scaled_extract_width**2, 1)
                ),
                np.reshape(
                    grid_indexes[1], (4 * extract_params.scaled_extract_width**2, 1)
                ),
            )
        )
        grid_matrix = np.hstack(
            (np.ones((4 * extract_params.scaled_extract_width**2, 1)), grid_matrix)
        )

        # Extraction for each micrograph
        with mrcfile.open(extract_params.motioncorr_name) as input_micrograph:
            input_micrograph_image = np.array(input_micrograph.data, dtype=np.float32)
        image_size = np.shape(input_micrograph_image)
        output_mrc_stack = []

        for particle in range(len(extract_params.particles_x)):
            # Pixel locations are from bottom left, need to flip the image later
            pixel_location_x = round(extract_params.particles_x[particle])
            pixel_location_y = round(extract_params.particles_y[particle])

            # Extract the particle image and pad the edges if it is not square
            x_left_pad = 0
            x_right_pad = 0
            y_top_pad = 0
            y_bot_pad = 0

            x_left = pixel_location_x - extract_params.full_extract_width
            if x_left < 0:
                x_left_pad = -x_left
                x_left = 0
            x_right = pixel_location_x + extract_params.full_extract_width
            if x_right >= image_size[1]:
                x_right_pad = x_right - image_size[1]
                x_right = image_size[1]
            y_top = pixel_location_y - extract_params.full_extract_width
            if y_top < 0:
                y_top_pad = -y_top
                y_top = 0
            y_bot = pixel_location_y + extract_params.full_extract_width
            if y_bot >= image_size[0]:
                y_bot_pad = y_bot - image_size[0]
                y_bot = image_size[0]

            particle_subimage = input_micrograph_image[y_top:y_bot, x_left:x_right]
            particle_subimage = np.pad(
                particle_subimage,
                ((y_bot_pad, y_top_pad), (x_left_pad, x_right_pad)),
                mode="edge",
            )

            # Flip all the values on inversion
            if extract_params.invert_contrast:
                particle_subimage = -1 * particle_subimage

            # Downscale the image size
            if extract_params.downscale:
                subimage_ft = np.fft.fftshift(np.fft.fft2(particle_subimage))
                deltax = subimage_ft.shape[0] - extract_params.scaled_boxsize
                deltay = subimage_ft.shape[1] - extract_params.scaled_boxsize
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

            # Plane fitting
            values = particle_subimage[bg_region]
            # normal equation
            theta = np.dot(flat_positions_matrix, values)
            plane = np.reshape(
                np.dot(grid_matrix, theta),
                (
                    2 * extract_params.scaled_extract_width,
                    2 * extract_params.scaled_extract_width,
                ),
            )
            particle_subimage -= plane

            # Background normalisation
            if extract_params.norm:
                # Standardise the values using the background
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

        # Produce the mrc file of the extracted particles
        Path(extract_params.reextract_name).parent.mkdir(exist_ok=True)
        particle_count = np.shape(output_mrc_stack)[0]
        if particle_count > 0:
            with mrcfile.new(str(extract_params.reextract_name), overwrite=True) as mrc:
                mrc.set_data(output_mrc_stack.astype(np.float32))
                mrc.header.mx = extract_params.scaled_boxsize
                mrc.header.my = extract_params.scaled_boxsize
                mrc.header.mz = 1
                mrc.header.cella.x = (
                    extract_params.scaled_pixel_size * extract_params.scaled_boxsize
                )
                mrc.header.cella.y = (
                    extract_params.scaled_pixel_size * extract_params.scaled_boxsize
                )
                mrc.header.cella.z = 1

        # Send on to the refinement wrapper
        refine_params = {
            "reextract_name": extract_params.reextract_name,
            "number_of_particles": extract_params.number_of_particles,
        }
        if isinstance(rw, MockRW):
            rw.transport.send(
                destination="refine_wrapper",
                message={"parameters": refine_params, "content": "dummy"},
            )
        else:
            rw.send_to("refine_wrapper", refine_params)

        self.log.info(f"Done extraction for {extract_params.reextract_name}.")
        rw.transport.ack(header)
