from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from gemmi import cif
from pydantic import BaseModel, Field, ValidationError, field_validator
from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.services.extract import extract_single_particle
from cryoemservices.util.models import MockRW
from cryoemservices.util.relion_service_options import (
    RelionServiceOptions,
    update_relion_options,
)
from cryoemservices.util.tomo_output_files import (
    _get_tilt_angle_v5_12,
    _get_tilt_name_v5_12,
    _get_tilt_number_v5_12,
)


class ExtractSubTomoParameters(BaseModel):
    cbox_3d_file: str = Field(..., min_length=1)
    tilt_alignment_file: str = Field(..., min_length=1)
    newstack_file: str = Field(..., min_length=1)
    output_star: str = Field(..., min_length=1)
    scaled_tomogram_shape: list[int] | str
    pixel_size: float
    dose_per_tilt: float
    tilt_offset: float
    particle_diameter: float = 0
    boxsize: int = 256
    small_boxsize: int = 64
    min_frames: int = 1
    maximum_dose: int = -1
    tomogram_binning: int = 4
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
        self.log.info("Sub-tomogram extraction service starting")
        wrap_subscribe(
            self._transport,
            self._environment["queue"] or "extract_subtomo",
            self.extract_subtomo,
            acknowledgement=True,
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
        centre_x = float(extract_subtomo_params.scaled_tomogram_shape[0]) / 2
        centre_y = float(extract_subtomo_params.scaled_tomogram_shape[1]) / 2
        centre_z = float(extract_subtomo_params.scaled_tomogram_shape[2]) / 2
        tilt_axis_radians = (refined_tilt_axis - 90) * np.pi / 180

        # Downscaling dimensions
        extract_subtomo_params.relion_options.pixel_size_downscaled = (
            extract_subtomo_params.pixel_size
            * extract_subtomo_params.relion_options.boxsize
            / extract_subtomo_params.relion_options.small_boxsize
        )
        self.log.info(
            f"Downscaling to {extract_subtomo_params.relion_options.pixel_size_downscaled}"
        )
        extract_width = round(extract_subtomo_params.relion_options.boxsize / 2)

        pixel_size = extract_subtomo_params.relion_options.pixel_size_downscaled

        # Read in tilt images
        self.log.info("Reading tilt images")
        tilt_png_names = []
        tilt_images = []
        tilt_numbers = []
        tilt_angles_radians = []
        with open(extract_subtomo_params.newstack_file) as ns_file:
            while True:
                line = ns_file.readline()
                if not line:
                    break
                elif line.startswith("/"):
                    tilt_name = line.strip()
                    tilt_png = Path(tilt_name).with_suffix(".png")
                    tilt_png_names.append(tilt_png)
                    tilt_numbers.append(_get_tilt_number_v5_12(Path(tilt_name)))
                    tilt_axis_from_file = float(_get_tilt_angle_v5_12(Path(tilt_name)))
                    tilt_angles_radians.append(
                        (tilt_axis_from_file + extract_subtomo_params.tilt_offset)
                        * np.pi
                        / 180
                    )
                    with mrcfile.open(tilt_name) as mrc:
                        tilt_images.append(mrc.data)

                    plt.imshow(tilt_images[-1])
                    plt.savefig(tilt_png)
                    plt.close()

        frames = np.zeros((len(particles_x), tilt_count), dtype=int)
        tilt_coords: list = [[] for tilt in range(tilt_count)]
        for particle in range(len(particles_x)):
            output_mrc_stack = np.array([])
            for tilt in range(tilt_count):
                if extract_subtomo_params.maximum_dose > 0 and (
                    extract_subtomo_params.dose_per_tilt * tilt_numbers[tilt]
                    > extract_subtomo_params.maximum_dose
                ):
                    self.log.info(f"Skipping {tilt} due to dose limit")
                    continue

                x_in_tilt, y_in_tilt = get_coord_in_tilt(
                    x=particles_x[particle],
                    y=particles_y[particle],
                    z=particles_z[particle],
                    cen_x=centre_x,
                    cen_y=centre_y,
                    cen_z=centre_z,
                    theta_y=tilt_angles_radians[tilt],
                    theta_z=tilt_axis_radians,
                    delta_x=x_shifts[tilt],
                    delta_y=y_shifts[tilt],
                    binning=extract_subtomo_params.tomogram_binning,
                )
                if tilt_angles_radians[tilt] == 0:
                    with open(
                        Path(extract_subtomo_params.output_star).parent
                        / "coords_in_tilts.txt",
                        "a",
                    ) as f:
                        f.write(
                            f"{particles_x[particle]} {particles_y[particle]} {x_in_tilt} {y_in_tilt}\n"
                        )
                tilt_coords[tilt].append([x_in_tilt, y_in_tilt])

                particle_subimage, failure_reason = extract_single_particle(
                    input_image=tilt_images[tilt],
                    x_coord=x_in_tilt,
                    y_coord=y_in_tilt,
                    extract_width=extract_width,
                    shape=[
                        int(i * extract_subtomo_params.tomogram_binning)
                        for i in extract_subtomo_params.scaled_tomogram_shape
                    ],
                    small_boxsize=extract_subtomo_params.small_boxsize,
                    bg_radius=round(0.375 * extract_subtomo_params.small_boxsize),
                    invert_contrast=True,
                    downscale=True,
                    norm=True,
                    plane_fit=True,
                )

                if failure_reason:
                    self.log.warning(
                        f"Extraction failed for {particle} in {tilt}. "
                        f"Reason was {failure_reason}."
                    )
                    particle_subimage = np.zeros(
                        (
                            extract_subtomo_params.small_boxsize,
                            extract_subtomo_params.small_boxsize,
                        )
                    )

                # Add to output stack
                if len(output_mrc_stack):
                    output_mrc_stack = np.append(
                        output_mrc_stack, [particle_subimage], axis=0
                    )
                else:
                    output_mrc_stack = np.array([particle_subimage], dtype=np.float32)
                frames[particle, tilt] = 1

            if not len(output_mrc_stack):
                self.log.warning(f"Could not extract particle {particle}")
                continue

            # Produce the mrc file of the extracted particles
            output_mrc_file = (
                Path(extract_subtomo_params.output_star).parent
                / f"{particle}_stack2d.mrcs"
            )
            self.log.info(f"Extracted particle {particle+1} of {len(particles_x)}")
            with mrcfile.new(str(output_mrc_file), overwrite=True) as mrc:
                mrc.set_data(output_mrc_stack.astype(np.float32))
                mrc.header.mx = extract_subtomo_params.relion_options.small_boxsize
                mrc.header.my = extract_subtomo_params.relion_options.small_boxsize
                mrc.header.mz = 1
                mrc.header.cella.x = (
                    pixel_size * extract_subtomo_params.relion_options.small_boxsize
                )
                mrc.header.cella.y = (
                    pixel_size * extract_subtomo_params.relion_options.small_boxsize
                )
                mrc.header.cella.z = 1

        for tilt in range(tilt_count):
            plt.imshow(tilt_images[tilt])
            for loc in tilt_coords[tilt]:
                plt.scatter(loc[0], loc[1], s=2, color="red")
            plt.savefig(tilt_png_names[tilt])
            plt.close()

        # Construct the output star file
        extracted_parts_doc = cif.Document()
        extracted_parts_block = extracted_parts_doc.add_new_block("particles")
        extracted_parts_loop = extracted_parts_block.init_loop(
            "_rln",
            [
                "TomoName",
                "OpticsGroup",
                "TomoParticleName",
                "TomoVisibleFrames",
                "ImageName",
                "OriginXAngst",
                "OriginYAngst",
                "OriginZAngst",
                "CenteredCoordinateXAngst",
                "CenteredCoordinateYAngst",
                "CenteredCoordinateZAngst",
            ],
        )
        for particle in range(len(particles_x)):
            extracted_parts_loop.add_row(
                [
                    _get_tilt_name_v5_12(
                        Path(extract_subtomo_params.tilt_alignment_file)
                    ),
                    "1",
                    f"{_get_tilt_name_v5_12(Path(extract_subtomo_params.tilt_alignment_file))}/{particle}",
                    f"[{','.join([str(frm) for frm in frames[particle]])}]",
                    f"{Path(extract_subtomo_params.output_star).parent}/{particle}_stack2d.mrcs",
                    str(centre_x * extract_subtomo_params.tomogram_binning),
                    str(centre_y * extract_subtomo_params.tomogram_binning),
                    str(centre_z * extract_subtomo_params.tomogram_binning),
                    str(
                        float(particles_x[particle])
                        - centre_x * extract_subtomo_params.tomogram_binning
                    ),
                    str(
                        float(particles_y[particle])
                        - centre_y * extract_subtomo_params.tomogram_binning
                    ),
                    str(
                        float(particles_z[particle])
                        - centre_z * extract_subtomo_params.tomogram_binning
                    ),
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
            "results": {
                "box_size": extract_subtomo_params.relion_options.small_boxsize
            },
        }
        rw.send_to("node_creator", node_creator_parameters)

        self.log.info(f"Done {self.job_type} for {extract_subtomo_params.cbox_3d_file}")
        rw.transport.ack(header)


def get_coord_in_tilt(
    x: float,
    y: float,
    z: float,
    cen_x: float,
    cen_y: float,
    cen_z: float,
    theta_y: float,
    theta_z: float,
    delta_x: float,
    delta_y: float,
    binning: int,
):
    # In binned coordinates here
    x_centred = x - cen_x
    y_centred = y - cen_y + cen_x * np.tan(theta_z)  # TODO: last factor depends on rot
    z_centred = z - cen_z
    x_2d = (
        x_centred * np.cos(theta_z) * np.cos(theta_y)
        - y_centred * np.sin(theta_z)
        + z_centred * np.cos(theta_z) * np.sin(theta_y)
    )
    y_2d = (
        x_centred * np.sin(theta_z) * np.cos(theta_y)
        + y_centred * np.cos(theta_z)
        + z_centred * np.sin(theta_z) * np.sin(theta_y)
    )
    # Un-bin and apply shifts
    x_tilt = (cen_x + x_2d) * binning - delta_x
    y_tilt = (cen_y + y_2d) * binning - delta_y
    return x_tilt, y_tilt
