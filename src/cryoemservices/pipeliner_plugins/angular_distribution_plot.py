from __future__ import annotations

import argparse
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from gemmi import cif


def angular_distribution_plot(
    theta_degrees: np.ndarray,
    phi_degrees: np.ndarray,
    healpix_order: int,
    output_jpeg: Path,
    class_label: str = "",
):
    """Generate healpix image of the particle distribution"""
    # Extract counts of particles in each healpix bin
    angle_pixel_bins = hp.pixelfunc.ang2pix(
        np.power(2, healpix_order + 1),
        theta_degrees * np.pi / 180,
        phi_degrees * np.pi / 180,
    )
    bin_ids, pixel_counts = np.unique(angle_pixel_bins, return_counts=True)
    all_pixel_bins = np.zeros(hp.nside2npix(np.power(2, healpix_order + 1)))
    all_pixel_bins[bin_ids] = pixel_counts

    # Create and save the healpix image
    hp.mollview(
        all_pixel_bins,
        title=f"Angular distribution of particles for class: {class_label}",
        unit="Number of particles",
        flip="geo",
    )
    hp.graticule()
    plt.savefig(output_jpeg)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="Particles star file to process",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="File name for the angular distribution image",
    )
    parser.add_argument(
        "-c",
        "--class_id",
        type=int,
        default=-1,
        help="Class to show distribution for. Default (-1) does all particles",
    )
    parser.add_argument(
        "--healpix_order",
        type=int,
        default=2,
        help="Healpix order for image binning",
    )
    args = parser.parse_args()

    data = cif.read_file(args.file)
    particles_block = data.find_block("particles")
    if particles_block:
        angles_rot = np.array(particles_block.find_loop("_rlnAngleRot"), dtype=float)
        angles_tilt = np.array(particles_block.find_loop("_rlnAngleTilt"), dtype=float)
        all_class_numbers = np.array(
            particles_block.find_loop("_rlnClassNumber"), dtype=int
        )
        if args.class_id != -1:
            angles_rot = angles_rot[all_class_numbers == args.class_id]
            angles_tilt = angles_tilt[all_class_numbers == args.class_id]

        if not len(angles_tilt):
            # Skip any classes with no particles
            print(f"No particles present in class {args.class_id}")
            return

        angular_distribution_plot(
            theta_degrees=angles_tilt,
            phi_degrees=angles_rot,
            healpix_order=args.healpix_order,
            output_jpeg=args.output,
            class_label=str(args.class_id) if args.class_id != -1 else "",
        )
