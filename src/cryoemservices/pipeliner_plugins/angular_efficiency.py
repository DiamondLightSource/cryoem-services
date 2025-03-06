from __future__ import annotations

import argparse

import healpy as hp
import numpy as np
import starfile


def efficiency_from_map(psf_map: np.ndarray, boxsize: int) -> float:
    # Discard all entries below the threshold
    thr = np.max(psf_map) / np.exp(2)
    psf_map[psf_map < thr] = 0

    # Construct binary map of 1 inside boundary, 0 outside
    volume_map = np.zeros((boxsize, boxsize, boxsize))
    volume_map[psf_map != 0] = 1

    # Find the average surface radius
    area = 0.0
    average = 0.0
    mean_sq = 0.0
    for i in range(1, boxsize):
        for j in range(1, boxsize):
            for k in range(1, boxsize):
                res = np.sqrt(
                    (i - boxsize / 2) ** 2
                    + (j - boxsize / 2) ** 2
                    + (k - boxsize / 2) ** 2
                )
                if res != 0 and np.sum(
                    volume_map[i - 1 : i + 1, j - 1 : j + 1, k - 1 : k + 1]
                ) not in [0, 8]:
                    # Mixture of 0 and 1 volume values means on surface
                    area += 1
                    average += res
                    mean_sq += res**2
    average /= area
    mean_sq /= area
    stdev = np.sqrt(mean_sq - average**2)
    return 1 - 2 * stdev / average


def map_from_angles(
    theta,
    phi,
    boxsize: int,
    pixel_size: float = 0.5,
    bfactor: float = 160,
    hp_nside: int = 64,
):
    # Convert angles to healpix bins
    pixel_bins = hp.pixelfunc.ang2pix(hp_nside, theta, phi)
    bin_ids, pixel_counts = np.unique(pixel_bins, return_counts=True)
    all_bins = np.zeros(hp.nside2npix(hp_nside))
    all_bins[bin_ids] = pixel_counts

    # Find all unique occupied bins
    unique_coords, unique_indexes, all_bins = np.unique(
        theta * 1000 + phi, return_index=True, return_counts=True
    )
    theta = theta[unique_indexes]
    phi = phi[unique_indexes]

    # Calculate plane for each point, which passes through the origin
    plane_x = np.sin(theta) * np.cos(phi)
    plane_y = np.sin(theta) * np.sin(phi)
    plane_z = np.cos(theta)
    plane_fits = np.transpose([plane_x, plane_y, plane_z])
    plane_coeff_squares = (
        plane_fits[:, 0] ** 2 + plane_fits[:, 1] ** 2 + plane_fits[:, 2] ** 2
    )
    unscaled_dist_x = plane_fits[:, 0] / plane_coeff_squares
    unscaled_dist_y = plane_fits[:, 1] / plane_coeff_squares
    unscaled_dist_z = plane_fits[:, 2] / plane_coeff_squares

    # Find the number of crossing planes in each occupied bin
    counts = np.ones(len(theta))
    for i in range(boxsize):
        for j in range(boxsize):
            for k in range(boxsize):
                ijk = np.array([i - boxsize / 2, j - boxsize / 2, k - boxsize / 2])
                if np.linalg.norm(ijk) < boxsize / 2 - 2:
                    plane_distances = np.dot(plane_fits, ijk)
                    dist_x = unscaled_dist_x * plane_distances
                    dist_y = unscaled_dist_y * plane_distances
                    dist_z = unscaled_dist_z * plane_distances
                    plane_matches = (
                        (np.abs(dist_x) < pixel_size)
                        * (np.abs(dist_y) < pixel_size)
                        * (np.abs(dist_z) < pixel_size)
                    )
                    counts[plane_matches] += np.ones(len(np.where(plane_matches)[0]))

    # Calculate the map
    k_array = np.zeros((boxsize, boxsize, boxsize))
    for i in range(boxsize):
        for j in range(boxsize):
            for k in range(boxsize):
                ijk = np.array([i - boxsize / 2, j - boxsize / 2, k - boxsize / 2])
                if np.linalg.norm(ijk) < boxsize / 2 - 2:
                    plane_distances = np.dot(plane_fits, ijk)
                    dist_x = unscaled_dist_x * plane_distances
                    dist_y = unscaled_dist_y * plane_distances
                    dist_z = unscaled_dist_z * plane_distances
                    plane_matches = (
                        (np.abs(dist_x) <= pixel_size)
                        * (np.abs(dist_y) <= pixel_size)
                        * (np.abs(dist_z) <= pixel_size)
                    )
                    k_array[i, j, k] = np.sum(
                        (
                            all_bins[plane_matches]
                            * np.exp(
                                -bfactor / boxsize**2 * np.linalg.norm(ijk) ** 2 / 2
                            )
                        )
                        / counts[plane_matches]
                    )
    return k_array


def find_efficiency(
    theta_degrees, phi_degrees, boxsize: int = 64, bfactor: float = 160
):
    fourier_map = map_from_angles(
        theta=theta_degrees * np.pi / 180,
        phi=phi_degrees * np.pi / 180,
        boxsize=boxsize,
        bfactor=bfactor,
    )
    normalised_fourier_map = np.sqrt(fourier_map / np.sum(fourier_map))

    shifted_fourier_map = np.zeros((boxsize * 2, boxsize * 2, boxsize * 2))
    for i in range(int(boxsize / 2), int(boxsize + boxsize / 2)):
        for j in range(int(boxsize / 2), int(boxsize + boxsize / 2)):
            for k in range(int(boxsize / 2), int(boxsize + boxsize / 2)):
                shifted_fourier_map[i, j, k] = normalised_fourier_map[
                    i - int(boxsize / 2), j - int(boxsize / 2), k - int(boxsize / 2)
                ]

    # real space psf
    map_ifft = np.abs(np.fft.ifftn(np.fft.ifftshift(shifted_fourier_map)))
    psf_map = np.fft.fftshift(map_ifft)[
        int(boxsize / 2) : int(boxsize + boxsize / 2),
        int(boxsize / 2) : int(boxsize + boxsize / 2),
        int(boxsize / 2) : int(boxsize + boxsize / 2),
    ]
    normalised_psf = psf_map / np.sqrt(np.sum(psf_map**2))

    eff = efficiency_from_map(normalised_psf, boxsize)
    print("Efficiency: ", eff)
    return eff


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="Star file to process",
    )
    parser.add_argument(
        "-c",
        "--class_id",
        default=-1,
        help="Class to calculate efficiency for. Default (-1) does all particles",
    )
    parser.add_argument(
        "--boxsize",
        default=64,
        help="Box size for efficiency calculations",
    )
    parser.add_argument(
        "--bfactor",
        default=160,
        help="B-Factor for efficiency_calculations",
    )
    args = parser.parse_args()

    star_file_data = starfile.read(args.file)
    theta_array = star_file_data["particles"]["rlnAngleTilt"]
    phi_array = star_file_data["particles"]["rlnAngleRot"]
    if args.class_id != -1:
        theta_array = theta_array[
            star_file_data["particles"]["rlnClassNumber"] == int(args.class_id)
        ]
        phi_array = phi_array[
            star_file_data["particles"]["rlnClassNumber"] == int(args.class_id)
        ]
    print(f"Processing {len(theta_array)} particles")
    find_efficiency(
        theta_degrees=np.array(theta_array),
        phi_degrees=np.array(phi_array),
        boxsize=int(args.boxsize),
        bfactor=float(args.bfactor),
    )
