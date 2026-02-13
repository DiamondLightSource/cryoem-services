from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import workflows.transport.pika_transport as pt
from gemmi import cif
from pydantic import BaseModel, Field, ValidationError, field_validator
from tqdm import tqdm
from workflows.recipe import wrap_subscribe

from cryoemservices.services.common_service import CommonService
from cryoemservices.services.extract import (
    enhance_single_particle,
    extract_single_particle,
)
from cryoemservices.util.models import MockRW
from cryoemservices.util.relion_service_options import (
    RelionServiceOptions,
    update_relion_options,
)
from cryoemservices.util.tomo_output_files import (
    _get_tilt_name_v5_12,
    _get_tilt_number_v5_12,
)

transport = pt.PikaTransport()
transport.load_configuration_file(
    "/dls_sw/apps/murfey/config/rmq-connection-creds-pollux.yml"
)
transport.connect()

root_dir = Path("/dls/m07/data/2025/cm40593-13/spool/ctfcorrect")
for tomo in root_dir.glob("Tomograms/job006/tomograms/*_aretomo.mrc"):
    transport.send(
        "segmentation",
        {
            "cbox_3d_file": f"{root_dir}/AutoPick/job009/CBOX_3D/{tomo.stem}.denoised.cbox",
            "tomogram": str(tomo),
            "output_star": f"{root_dir}/Extract/job010/{tomo.stem}.star",
            "pixel_size": 5.36,
            "particle_diameter": 250,
            "relion_options": {},
        },
    )

inptus = {
    "cbox_3d_file": "/scratch/yxd92326/data/tomo-extract/2_2_Ribosome_Pos_1_stack_aretomo.denoised.cbox",
    "tilt_alignment_file": "/scratch/yxd92326/data/tomo-extract/2_2_Ribosome_Pos_1_stack.aln",
    "newstack_file": "/scratch/yxd92326/data/tomo-extract/2_2_Ribosome_Pos_1_stack_newstack.txt",
    "output_star": "/scratch/yxd92326/data/tomo-extract/extracted/extract.star",
    "pixel_size": 1.34,
    "dose_per_tilt": 4,
    "tilt_offset": 0,
    "scaled_tomogram_shape": [1440, 1023, 400],
    "relion_options": {},
}
in2 = {
    "cbox_3d_file": "/scratch/yxd92326/data/tomo-extract/2_1_ApoF_Pos_13_9_test.cbox",
    "tilt_alignment_file": "/scratch/yxd92326/data/tomo-extract/2_1_ApoF_Pos_13_9_stack_aretomo.aln",
    "newstack_file": "/scratch/yxd92326/data/tomo-extract/2_1_ApoF_Pos_13_9_stack_newstack.txt",
    "output_star": "/scratch/yxd92326/data/tomo-extract/extracted_apof/extract.star",
    "pixel_size": 1.34,
    "dose_per_tilt": 4,
    "tilt_offset": 0,
    "scaled_tomogram_shape": [1440, 1023, 400],
    "relion_options": {},
    "particle_diameter": 500,
}

in3 = {
    "cbox_3d_file": "/dls/m06/data/2025/bi37708-55/tmp/extract-test/AutoPick/job009/CBOX_3D/Tomo_position3_stack_Vol.denoised.cbox",
    "tomogram": "/dls/m06/data/2025/bi37708-55/tmp/extract-test/Tomograms/job006/tomograms/Tomo_position3_stack_Vol.mrc",
    "output_star": "/dls/m06/data/2025/bi37708-55/tmp/extract-test/Extract/class2d/Tomo_position3_stack_Vol.star",
    "pixel_size": 7.76,
    "particle_diameter": 225,
    "relion_options": {},
}

transport.send(
    "segmentation",
    {
        "cbox_3d_file": "/dls/m06/data/2025/bi38637-22/spool/extract-test/AutoPick/job009/CBOX_3D/Position_002_stack_Vol.denoised.cbox",
        "tomogram": "/dls/m06/data/2025/bi38637-22/spool/extract-test/CtfFind/corrected/aretomo2_compare/Position_002_stack_aretomo2_raw.mrc",
        "output_star": "/dls/m06/data/2025/bi38637-22/spool/extract-test/Extract/ctftest/Position_002_stack_raw.star",
        "pixel_size": 7.76,
        "particle_diameter": 250,
        "relion_options": {},
    },
)


class ExtractSubTomoParameters3D(BaseModel):
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


class ExtractSubTomoFor3D(CommonService):
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
                extract_subtomo_params = ExtractSubTomoParameters3D(
                    **{**rw.recipe_step.get("parameters", {}), **message}
                )
            else:
                extract_subtomo_params = ExtractSubTomoParameters3D(
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
        particles_z = np.array(coords_block.find_loop("_CoordinateZ"), dtype=float)

        # Get the shifts between tilts
        shift_data = np.genfromtxt(extract_subtomo_params.tilt_alignment_file)
        refined_tilt_axis = float(shift_data[0, 1])
        x_shifts = shift_data[:, 3].astype(float)
        y_shifts = shift_data[:, 4].astype(float)
        tilt_angles = shift_data[:, 9].astype(float) * np.pi / 180
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
        tid = 0
        with open(extract_subtomo_params.newstack_file) as ns_file:
            while True:
                tid += 1
                line = ns_file.readline()
                if not line:
                    break
                elif line.startswith("/"):
                    tilt_name = line.strip()
                    tilt_png = Path("/home/yxd92326/Pictures/picking/tomo/") / (
                        f"{tid}T_" + Path(tilt_name).with_suffix(".png").name
                    )
                    tilt_png_names.append(tilt_png)
                    tilt_png.unlink(missing_ok=True)

                    tilt_numbers.append(_get_tilt_number_v5_12(Path(tilt_name)))
                    with mrcfile.open(tilt_name) as mrc:
                        tilt_images.append(mrc.data)

        for tilt in range(tilt_count):
            if extract_subtomo_params.maximum_dose > 0 and (
                extract_subtomo_params.dose_per_tilt * tilt_numbers[tilt]
                > extract_subtomo_params.maximum_dose
            ):
                self.log.info(f"Skipping tilt {tilt} due to dose limit")

        frames = np.zeros((len(particles_x), tilt_count), dtype=int)
        tilt_coords: list = [[] for tilt in range(tilt_count)]
        for particle in tqdm(range(len(particles_x))):
            output_mrc_stack = np.array([])
            for tilt in range(tilt_count):
                if extract_subtomo_params.maximum_dose > 0 and (
                    extract_subtomo_params.dose_per_tilt * tilt_numbers[tilt]
                    > extract_subtomo_params.maximum_dose
                ):
                    continue

                x_in_tilt, y_in_tilt = get_coord_in_tilt(
                    x=particles_x[particle],
                    y=particles_y[particle],
                    z=particles_z[particle],
                    cen_x=centre_x,
                    cen_y=centre_y,
                    cen_z=centre_z,
                    theta_y=tilt_angles[tilt],
                    theta_z=tilt_axis_radians,
                    delta_x=x_shifts[tilt],
                    delta_y=y_shifts[tilt],
                    binning=extract_subtomo_params.tomogram_binning,
                )
                tilt_coords[tilt].append([x_in_tilt, y_in_tilt])
                # print(x_in_tilt, y_in_tilt, tilt_angles[tilt])

                particle_subimage, failure_reason1 = extract_single_particle(
                    input_image=tilt_images[tilt],
                    x_coord=x_in_tilt,
                    y_coord=y_in_tilt,
                    extract_width=extract_width,
                    shape=[
                        int(i * extract_subtomo_params.tomogram_binning)
                        for i in extract_subtomo_params.scaled_tomogram_shape
                    ],
                )
                particle_subimage, failure_reason2 = enhance_single_particle(
                    particle_subimage=particle_subimage,
                    extract_width=extract_width,
                    small_boxsize=extract_subtomo_params.small_boxsize,
                    bg_radius=round(0.375 * extract_subtomo_params.small_boxsize),
                    invert_contrast=True,
                    downscale=True,
                    norm=True,
                    plane_fit=True,
                )

                if failure_reason1 or failure_reason2:
                    self.log.warning(
                        f"Extraction failed for {particle} in {tilt}. "
                        f"Reason was {failure_reason1} {failure_reason2}."
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

        for tilt in tqdm(range(tilt_count)):
            plt.imshow(tilt_images[tilt], vmin=0.5, vmax=1.7)
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
    # Translation raw to aligned tilt is subtract shift then rotate around centre
    # In binned coordinates here
    x_centred = x - cen_x
    y_centred = y - cen_y  # + cen_x * np.tan(theta_z) TODO: last factor depends on rot
    x_2d = x_centred * np.cos(theta_z) - y_centred * np.sin(theta_z)
    y_2d = x_centred * np.sin(theta_z) + y_centred * np.cos(theta_z)
    # Un-bin and apply shifts
    x_tilt = (cen_x + x_2d) * binning + delta_x
    y_flat = (cen_y + y_2d) * binning + delta_y
    y_tilt = (y_flat - cen_y * binning) * np.cos(theta_y) + cen_y * binning
    # print(cen_x, x, x_2d, delta_x, cen_y, y, y_2d, delta_y)

    z_centred = z - cen_z
    y_tilt += z_centred * np.sin(theta_y) * binning
    # print(z_centred * np.sin(theta_y), x_tilt, y_tilt)
    return x_tilt, y_tilt


"""
with open("Extract_maxdose/particles.star", "a") as partstar:
    for tomo in Path("Extract_maxdose").glob("2_1*"):
        with open(tomo / "extract.star") as tomostar:
            while True:
                line=tomostar.readline()
                if not line:
                    break
                if line.startswith("2_1"):
                    partstar.write(line)
"""


def cbox_to_star(name, max_subsample):
    import pandas as pd
    import starfile

    # make data_particles
    cbox = starfile.read(f"AutoPick/job009/CBOX_3D/{name}_stack_aretomo.denoised.cbox")
    all_particles = cbox["cryolo"]
    new_particles = pd.DataFrame()
    new_particles["rlnTomoName"] = [f"{name}" for i in range(len(all_particles))]
    new_particles["rlnCenteredCoordinateXAngst"] = (
        all_particles["CoordinateY"] * 4 + 80 - 4092 / 2
    ) * 1.34
    new_particles["rlnCenteredCoordinateYAngst"] = (
        5760 - all_particles["CoordinateX"] * 4 - 80 - 5760 / 2
    ) * 1.34
    new_particles["rlnCenteredCoordinateZAngst"] = (
        all_particles["CoordinateZ"] * 4 - 1600 / 2
    ) * 1.34
    for subsamp in range(2, max_subsample + 1):
        if Path(
            f"AutoPick/job009/CBOX_3D/{name}_{subsamp}_stack_aretomo.denoised.cbox"
        ).is_file():
            cbox = starfile.read(
                f"AutoPick/job009/CBOX_3D/{name}_{subsamp}_stack_aretomo.denoised.cbox"
            )
            particles = cbox["cryolo"]
            add_particles = pd.DataFrame()
            add_particles["rlnTomoName"] = [
                f"{name}_{subsamp}" for i in range(len(particles))
            ]
            add_particles["rlnCenteredCoordinateXAngst"] = (
                particles["CoordinateY"] * 4 + 80 - 4092 / 2
            ) * 1.34
            add_particles["rlnCenteredCoordinateYAngst"] = (
                5760 - particles["CoordinateX"] * 4 - 80 - 5760 / 2
            ) * 1.34
            add_particles["rlnCenteredCoordinateZAngst"] = (
                particles["CoordinateZ"] * 4 - 1600 / 2
            ) * 1.34
            new_particles = pd.concat((new_particles, add_particles))
    starfile.write(new_particles, f"AutoPick/job009/{name}_all_particles_centered.star")


def cbox_to_star_whole_dir(pixel_size=1.63, xdim=5760, ydim=4092, zdim=1600):
    import pandas as pd
    import starfile

    # make data_particles
    new_particles = None

    for cbox in Path("AutoPick/job009/CBOX_3D").glob("*.cbox"):
        all_particles = starfile.read(cbox)["cryolo"]

        particles_to_drop = []
        for pindex, particle in all_particles.iterrows():
            if (
                particle["CoordinateZ"] < particle["Depth"] / 2
                or zdim / 4 - particle["CoordinateZ"] < particle["Depth"] / 2
            ):
                particles_to_drop.append(pindex)
        print(cbox, particles_to_drop)
        all_particles.drop(labels=particles_to_drop, axis=0, inplace=True)

        add_particles = pd.DataFrame()
        add_particles["rlnCenteredCoordinateXAngst"] = (
            all_particles["CoordinateY"] * 4 + 80 - ydim / 2
        ) * pixel_size
        add_particles["rlnCenteredCoordinateYAngst"] = (
            xdim / 2 - all_particles["CoordinateX"] * 4 - 80
        ) * pixel_size
        add_particles["rlnCenteredCoordinateZAngst"] = (
            all_particles["CoordinateZ"] * 4 - zdim / 2
        ) * pixel_size
        add_particles["rlnTomoName"] = [
            cbox.name.split("_stack_")[0] for i in range(len(all_particles))
        ]
        if new_particles is None:
            new_particles = add_particles
        else:
            new_particles = pd.concat((new_particles, add_particles))
    starfile.write(new_particles, "AutoPick/job009/all_particles_centered.star")


def cbox_to_star_whole_dir_noflip():
    import pandas as pd
    import starfile

    # make data_particles
    new_particles = None

    for cbox in Path("AutoPick/job009/CBOX_3D").glob("*.cbox"):
        all_particles = starfile.read(cbox)["cryolo"]
        add_particles = pd.DataFrame()
        add_particles["rlnCenteredCoordinateXAngst"] = (
            all_particles["CoordinateX"] * 4 + 80 - 5760 / 2
        ) * 1.63
        add_particles["rlnCenteredCoordinateYAngst"] = (
            all_particles["CoordinateY"] * 4 + 80 - 4092 / 2
        ) * 1.63
        add_particles["rlnCenteredCoordinateZAngst"] = (
            all_particles["CoordinateZ"] * 4 - 1600 / 2
        ) * 1.63
        add_particles["rlnTomoName"] = [
            cbox.name.split("_stack_")[0] for i in range(len(all_particles))
        ]
        if new_particles is None:
            new_particles = add_particles
        else:
            new_particles = pd.concat((new_particles, add_particles))
    starfile.write(new_particles, "AutoPick/job009/all_particles_centered_noflip.star")
