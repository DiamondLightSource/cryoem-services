from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

import mrcfile
import numpy as np


def reextract_single_micrograph(
    motioncorr_name: Path,
    reextract_name: Path,
    particles_x: List[float],
    particles_y: List[float],
    full_extract_width: int,
    scaled_extract_width: int,
    scaled_boxsize: int,
    scaled_pixel_size: float,
    grid_matrix,
    flat_positions_matrix,
    bg_region,
    invert_contrast: bool = True,
    normalise: bool = True,
    downscale: bool = True,
):
    """Function to run reextraction of particles from a single micrograph image"""
    # Extraction for each micrograph
    with mrcfile.open(motioncorr_name) as input_micrograph:
        input_micrograph_image = np.array(input_micrograph.data, dtype=np.float32)
    image_size = np.shape(input_micrograph_image)
    output_mrc_stack = []

    for particle in range(len(particles_x)):
        # Pixel locations are from bottom left, need to flip the image later
        pixel_location_x = round(particles_x[particle])
        pixel_location_y = round(particles_y[particle])

        # Extract the particle image and pad the edges if it is not square
        x_left_pad = 0
        x_right_pad = 0
        y_top_pad = 0
        y_bot_pad = 0

        x_left = pixel_location_x - full_extract_width
        if x_left < 0:
            x_left_pad = -x_left
            x_left = 0
        x_right = pixel_location_x + full_extract_width
        if x_right >= image_size[1]:
            x_right_pad = x_right - image_size[1]
            x_right = image_size[1]
        y_top = pixel_location_y - full_extract_width
        if y_top < 0:
            y_top_pad = -y_top
            y_top = 0
        y_bot = pixel_location_y + full_extract_width
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
        if invert_contrast:
            particle_subimage = -1 * particle_subimage

        # Downscale the image size
        if downscale:
            subimage_ft = np.fft.fftshift(np.fft.fft2(particle_subimage))
            deltax = subimage_ft.shape[0] - scaled_boxsize
            deltay = subimage_ft.shape[1] - scaled_boxsize
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
                2 * scaled_extract_width,
                2 * scaled_extract_width,
            ),
        )
        particle_subimage -= plane

        # Background normalisation
        if normalise:
            # Standardise the values using the background
            bg_mean = np.mean(particle_subimage[bg_region])
            bg_std = np.std(particle_subimage[bg_region])
            particle_subimage = (particle_subimage - bg_mean) / bg_std

        # Add to output stack
        if len(output_mrc_stack):
            output_mrc_stack = np.append(output_mrc_stack, [particle_subimage], axis=0)
        else:
            output_mrc_stack = np.array([particle_subimage], dtype=np.float32)

    # Produce the mrc file of the extracted particles
    Path(reextract_name).parent.mkdir(exist_ok=True, parents=True)
    particle_count = np.shape(output_mrc_stack)[0]
    if particle_count > 0:
        with mrcfile.new(str(reextract_name), overwrite=True) as mrc:
            mrc.set_data(output_mrc_stack.astype(np.float32))
            mrc.header.mx = scaled_boxsize
            mrc.header.my = scaled_boxsize
            mrc.header.mz = 1
            mrc.header.cella.x = scaled_pixel_size * scaled_boxsize
            mrc.header.cella.y = scaled_pixel_size * scaled_boxsize
            mrc.header.cella.z = 1


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extract_job_dir",
        help="Job directory for reextraction",
        dest="extract_job_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--select_job_dir",
        help="Job directory of the selected particles",
        dest="select_job_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--original_dir",
        help="Original project directory",
        dest="original_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--full_boxsize",
        help="Box size in pixels before scaling",
        dest="full_boxsize",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--scaled_boxsize",
        help="Scaled box size in pixels",
        dest="scaled_boxsize",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--full_pixel_size",
        help="Size of pixels in angstroms before scaling",
        dest="full_pixel_size",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--scaled_pixel_size",
        help="Size of pixels in angstroms after scaling",
        dest="scaled_pixel_size",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--bg_radius",
        help="Background radius in pixels",
        dest="bg_radius",
        default=-1,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--invert_contrast",
        dest="invert_contrast",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--normalise",
        dest="normalise",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--downscale",
        dest="downscale",
        action="store_true",
        default=False,
    )
    return parser


def run():
    """Function to run re-extraction of particles from a selection job"""
    parser = create_parser()
    args = parser.parse_args()

    # Run re-extraction on the selected particles
    print(f"Reextraction requested for {args.extract_job_dir}")
    extract_job_dir = Path(args.extract_job_dir)
    extract_job_dir.mkdir(parents=True, exist_ok=True)
    project_dir = Path(extract_job_dir).parent.parent

    # Modify the extraction star file to contain reextracted values
    mrcs_dict = {}
    with open(f"{args.select_job_dir}/particles.star", "r") as selected_particles, open(
        extract_job_dir / "particles.star", "w"
    ) as extracted_particles, open(
        extract_job_dir / "extractpick.star", "w"
    ) as micrograph_list:
        micrograph_list.write(
            "data_coordinate_files\n\nloop_ \n"
            "_rlnMicrographName #1 \n_rlnMicrographCoordinates #2 \n"
        )
        while True:
            line = selected_particles.readline()
            if not line:
                break
            if line.startswith("opticsGroup"):
                # Optics table change pixel size #7 and image size #8
                split_line = line.split()
                split_line[6] = str(args.scaled_pixel_size)
                split_line[7] = str(args.scaled_boxsize)
                line = " ".join(split_line)
            elif line.lstrip() and line.lstrip()[0].isnumeric():
                # Main table change x#1, y#2, name#3, originx#18, originy#19
                split_line = line.split()

                coord_x = float(split_line[0])
                coord_y = float(split_line[1])
                centre_x = float(split_line[17])
                centre_y = float(split_line[18])
                split_line[0] = str(coord_x - int(centre_x / args.full_pixel_size))
                split_line[1] = str(coord_y - int(centre_y / args.full_pixel_size))
                split_line[17] = str(centre_x - int(centre_x / args.full_pixel_size))
                split_line[18] = str(centre_y - int(centre_y / args.full_pixel_size))

                # Create a dictionary of the images and their particles
                mrcs_name = split_line[2].split("@")[1]
                reextract_name = re.sub(
                    ".*Extract/job[0-9]+/",
                    f"{extract_job_dir.relative_to(project_dir)}/",
                    mrcs_name,
                )
                if mrcs_dict.get(mrcs_name):
                    mrcs_dict[mrcs_name]["counter"] += 1
                    mrcs_dict[mrcs_name]["x"].append(float(split_line[0]))
                    mrcs_dict[mrcs_name]["y"].append(float(split_line[1]))
                else:
                    mrcs_dict[mrcs_name] = {
                        "counter": 1,
                        "motioncorr_name": Path(args.original_dir) / split_line[3],
                        "reextract_name": project_dir / reextract_name,
                        "x": [float(split_line[0])],
                        "y": [float(split_line[1])],
                    }
                    micrograph_list.write(f"{split_line[3]} {reextract_name}\n")

                split_line[2] = f"{mrcs_dict[mrcs_name]['counter']:06}@{reextract_name}"
                line = "  ".join(split_line) + "\n"
            extracted_particles.write(line)

    full_extract_width = round(args.full_boxsize / 2)
    scaled_extract_width = round(args.scaled_boxsize / 2)

    # If no background radius set diameter as 75% of box
    if args.bg_radius == -1:
        args.bg_radius = round(0.375 * args.scaled_boxsize)

    # Distance of each pixel from the centre, compared to background radius
    grid_indexes = np.meshgrid(
        np.arange(2 * scaled_extract_width),
        np.arange(2 * scaled_extract_width),
    )
    distance_from_centre = np.sqrt(
        (grid_indexes[0] - scaled_extract_width + 0.5) ** 2
        + (grid_indexes[1] - scaled_extract_width + 0.5) ** 2
    )
    bg_region = (
        distance_from_centre > np.ones(np.shape(distance_from_centre)) * args.bg_radius
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
    flat_positions_matrix = np.dot(
        np.linalg.inv(np.dot(positions_matrix.transpose(), positions_matrix)),
        positions_matrix.transpose(),
    )

    # now we need the full grid across the image
    grid_matrix = np.hstack(
        (
            np.reshape(grid_indexes[0], (4 * scaled_extract_width**2, 1)),
            np.reshape(grid_indexes[1], (4 * scaled_extract_width**2, 1)),
        )
    )
    grid_matrix = np.hstack((np.ones((4 * scaled_extract_width**2, 1)), grid_matrix))

    print(f"Extracting particles from {len(mrcs_dict.keys())} micrographs")
    with ProcessPoolExecutor() as executor:
        for mrcs_name in mrcs_dict.keys():
            executor.submit(
                reextract_single_micrograph,
                motioncorr_name=mrcs_dict[mrcs_name]["motioncorr_name"],
                reextract_name=mrcs_dict[mrcs_name]["reextract_name"],
                particles_x=mrcs_dict[mrcs_name]["x"],
                particles_y=mrcs_dict[mrcs_name]["y"],
                full_extract_width=full_extract_width,
                scaled_extract_width=scaled_extract_width,
                scaled_boxsize=args.scaled_boxsize,
                scaled_pixel_size=args.scaled_pixel_size,
                grid_matrix=grid_matrix,
                flat_positions_matrix=flat_positions_matrix,
                bg_region=bg_region,
                invert_contrast=args.invert_contrast,
                normalise=args.normalise,
                downscale=args.downscale,
            )
    print(f"Done reextraction job {extract_job_dir}")
