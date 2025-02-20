from __future__ import annotations

from datetime import datetime

import healpy as hp
import numpy as np


def efficiency_from_map(Rmap: np.ndarray, boxsize: int) -> float:
    # Discard all entries below the threshold
    thr = np.max(Rmap) / np.exp(2)
    Rmap[Rmap < thr] = 0

    # Construct binary map of 1 inside boundary, 0 outside
    volume_map = np.zeros((boxsize, boxsize, boxsize))
    volume_map[Rmap != 0] = 1

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


def map_from_angles(theta, phi, boxsize, pixel_size=0.5, bfactor=160):
    hp_nside = 64

    pixel_bins = hp.pixelfunc.ang2pix(hp_nside, theta, phi)
    bin_ids, pixel_counts = np.unique(pixel_bins, return_counts=True)
    all_bins = np.zeros(hp.nside2npix(hp_nside))
    all_bins[bin_ids] = pixel_counts

    unique_coords, unique_indexes, all_bins = np.unique(
        theta * 1000 + phi, return_index=True, return_counts=True
    )
    theta = theta[unique_indexes]
    phi = phi[unique_indexes]

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


def find_efficiency(theta_degrees, phi_degrees, boxsize, bfactor):
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
    return normalised_psf


def Euler2Matrix(phi, theta, psi):
    sinphi = np.sin(phi * np.pi / 180)
    cosphi = np.cos(phi * np.pi / 180)
    sintheta = np.sin(theta * np.pi / 180)
    costheta = np.cos(theta * np.pi / 180)
    sinpsi = np.sin(psi * np.pi / 180)
    cospsi = np.cos(phi * np.pi / 180)

    R = np.zeros((3, 3))
    R[0, 0] = cospsi * costheta * cosphi - sinpsi * sinphi
    R[0, 1] = cospsi * costheta * sinphi + sinpsi * cosphi
    R[0, 2] = -cospsi * sintheta
    R[1, 0] = -sinpsi * costheta * cosphi - cospsi * sinphi
    R[1, 1] = cospsi * cosphi - sinpsi * costheta * sinphi
    R[1, 2] = sinpsi * sintheta
    R[2, 0] = sintheta * cosphi
    R[2, 1] = sintheta * sinphi
    R[2, 2] = costheta
    return R


def suggest_tilt(Rmap, boxsize, tiltmax=45):
    # Discard all entries below the threshold
    thr = np.max(Rmap) / np.exp(2)
    Rmap[Rmap < thr] = 0

    # Construct binary map of 1 inside boundary, 0 outside
    volume_map = np.zeros((boxsize, boxsize, boxsize))
    volume_map[Rmap != 0] = 1

    res_worst = 0
    theta_worst = 0
    phi_worst = 0
    for i in range(1, boxsize):
        for j in range(1, boxsize):
            for k in range(1, boxsize):
                res = np.sqrt(
                    (i - boxsize / 2) ** 2
                    + (j - boxsize / 2) ** 2
                    + (k - boxsize / 2) ** 2
                )
                if res > res_worst and np.sum(
                    volume_map[i - 1 : i + 1, j - 1 : j + 1, k - 1 : k + 1]
                ) not in [0, 8]:
                    # Mixture of 0 and 1 volume values means on surface
                    res_worst = res
                    theta_worst = np.arccos(
                        (k - boxsize / 2)
                        / np.sqrt(
                            (i - boxsize / 2) ** 2
                            + (j - boxsize / 2) ** 2
                            + (k - boxsize / 2) ** 2
                        )
                    )
                    phi_worst = np.arctan((j - boxsize / 2) / (i - boxsize / 2 + 1e-10))

    if phi_worst < 0:
        phi_worst = phi_worst + np.pi
        theta_worst = np.pi - theta_worst

    print(phi_worst * 180 / np.pi, theta_worst * 180 / np.pi, res_worst)

    # Find the surface radius
    resmap = np.zeros((180, 180))
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
                    theta = np.arccos((k - dim / 2) / res) * 180 / np.pi
                    phi = np.arctan((j - dim / 2) / (i - dim / 2 + 1e-10)) * 180 / np.pi
                    if phi < 0:
                        phi = phi + 180
                        theta = 180 - theta
                    thetaindex = int(theta)
                    phiindex = int(phi)
                    if phiindex == 180:
                        phiindex = 0
                    if thetaindex == 180:
                        thetaindex = 0
                    resmap[phiindex, thetaindex] = res

    # Interpolation of the resolution map
    for thetaindex in range(180):
        for phiindex in range(180):
            pickout_region = resmap[
                max(phiindex - 1, 0) : min(phiindex + 2, 179),
                max(thetaindex - 1, 0) : min(thetaindex + 2, 179),
            ]
            resmap[phiindex, thetaindex] = np.mean(pickout_region[pickout_region != 0])

    # pass in theta, phi in radians
    coverage = np.zeros((180, 180))
    x1 = np.sin(theta_worst) * np.cos(phi_worst)
    y1 = np.sin(theta_worst) * np.sin(phi_worst)
    z1 = np.cos(theta_worst)

    for theta in range(1, tiltmax):
        for phi in range(180):
            Rtilt = Euler2Matrix(phi + 90, theta, -90 - phi)
            x2, y2, z2 = np.dot(Rtilt, [x1, y1, z1])
            theta2 = np.arccos(z2) * 180 / np.pi
            phi2 = np.arctan(y2 / x2) * 180 / np.pi
            if phi2 < 0:
                phi2 = phi2 + 180
                theta2 = 180 - theta2
            thetaindex = int(theta2)
            phiindex = int(phi2)
            if phiindex == 180:
                phiindex = 0
            if thetaindex == 180:
                thetaindex = 0
            coverage[phiindex, thetaindex] = theta * np.pi / 180

    print(np.max(resmap), np.min(resmap[(resmap != 0) * (coverage != 0)]))

    predict_res = resmap / np.cos(coverage)
    expected_res = min(predict_res[(resmap != 0) * (coverage != 0)])
    expected_tilt = coverage[predict_res == expected_res]
    if expected_res < res_worst:
        print(
            "Found tilt", expected_tilt * 180 / np.pi, "with resolution", expected_res
        )
    else:
        print("No tilt found")
    return resmap, coverage


if __name__ == "__main__":
    start_time = datetime.now()
    dim = 64
    B = 160
    inFile = "/dls/tmp/yxd92326/data_cef/phi27.dat"
    angles_data = np.genfromtxt(inFile)
    norm_psf = find_efficiency(
        theta_degrees=angles_data[:, 1],
        phi_degrees=angles_data[:, 0],
        boxsize=dim,
        bfactor=B,
    )

    resmap, coverage = suggest_tilt(norm_psf, boxsize=dim)
    end_time = datetime.now()
    print("Time", (end_time - start_time).total_seconds() / 3600)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3)
    # resmap[coverage == 0] = 0
    ax[0].imshow(resmap)
    ax[1].imshow(coverage)
    # plt.imshow(resmap)
    pred_res = resmap / np.cos(coverage)
    pred_res[coverage == 0] = 0
    ax[2].imshow(pred_res)
    plt.show()

    # import mrcfile
    # with mrcfile.new("/dls/tmp/yxd92326/cryoEF_v1.1.0/norm_psf.mrc", overwrite=True) as mrc:
    #    mrc.set_data(norm_psf.astype(np.float32))
