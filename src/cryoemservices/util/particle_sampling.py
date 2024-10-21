from __future__ import annotations

from pathlib import Path

import numpy as np
from gemmi import cif
from scipy.cluster.hierarchy import fcluster, linkage

from cryoemservices.pipeliner_plugins.combine_star_files import (
    combine_star_files,
    compare_optics_tables,
)

# take particles from various 2D batches
# sample based on the classes
# pick best particles using ctf max res
# rerun 3d classification


def combine_batches(
    job_dir: Path,
    filter_field: str,
    batches: list[Path],
    batch_sizes: int = 50000,
    max_particles: int = 200000,
):
    particles_per_batch = int(max_particles / len(batches))
    if particles_per_batch > batch_sizes:
        raise ValueError("Not enough batches have been supplied")

    reference_optics = None
    with open(batches[0], "r") as full_starfile:
        for line_counter in range(50):
            line = full_starfile.readline()
            if line.startswith("opticsGroup"):
                reference_optics = line.split()
            if not line:
                break
    if not reference_optics:
        raise IndexError(f"Cannot find optics group in {batches[0]}")

    new_particles_file = cif.Document()
    batch_star = cif.read_file(str(batches[0]))
    optics_block = batch_star.find_block("optics")
    new_particles_file.add_copied_block(optics_block)
    new_particles_file.write_file(str(job_dir / "particles_rebatch.star"))

    for batch_2d in batches:
        compare_optics_tables(batch_2d, reference_optics)

        tmp_particles_file = cif.Document()
        batch_star = cif.read_file(str(batch_2d))
        optics_block = batch_star.find_block("optics")
        tmp_particles_file.add_copied_block(optics_block)

        particles_block = batch_star.find_block("particles")
        ctf_loop = particles_block.find_loop(filter_field)
        selection_limit = sorted(ctf_loop)[particles_per_batch]
        selected_particles = ctf_loop.get_loop()(list(ctf_loop) < selection_limit)
        tmp_particles_file.add_copied_block(selected_particles)

        tmp_particles_file.write_file(str(job_dir / ".tmp_batch.star"))
        combine_star_files(
            [job_dir / "particles_rebatch.star", job_dir / ".tmp_batch.star"],
            output_dir=job_dir,
            output_name="particles_rebatch.star",
        )
        (job_dir / ".tmp_particles.star").unlink()

    return job_dir / "particles_rebatch.star"


def sample_orientations(particles_file: Path, threshold: float = 0.1):
    particle_ctf_max_res = np.array([])
    particle_angles = []
    particle_norm_angles = np.array([])
    with open(particles_file, "r") as particles_data:
        while True:
            line = particles_data.readline()
            if not line:
                break
            if line.strip()[0].isnumeric():
                particle_ctf_max_res = np.append(
                    particle_ctf_max_res, float(line.split()[10])
                )  # wherever ctf max res is
                particle_angles.append(
                    (
                        float(line.split()[11]),
                        float(line.split()[12]),  # wherever theta, phi are in the line
                    )
                )
                particle_norm_angles = np.append(
                    particle_norm_angles,
                    [
                        np.sin(float(line.split()[11]) / 2),  # if theta is 0 to 2pi
                        np.sin(float(line.split()[12]) / 2),  # if phi is -pi to pi
                    ],
                )

    clusters = linkage(particle_norm_angles, method="single", metric="euclidean")
    clustered_particles = fcluster(clusters, threshold, criterion="distance")

    selected_particles = np.zeros(len(particle_ctf_max_res), dtype=bool)

    number_of_clusters = max(clustered_particles)
    for cluster_id in range(number_of_clusters):
        cluster_ctfs = particle_ctf_max_res[clustered_particles == cluster_id]
        if min(cluster_ctfs) > 5:
            # some threshold on bad clusters (possibly from class2d res as well?)
            continue

        particles_to_use = len(clustered_particles) / number_of_clusters / 10
        if len(cluster_ctfs) < particles_to_use:
            # some threshold on having very few particles
            selected_particles[clustered_particles == cluster_id] = True
        else:
            ctf_threshold = sorted(cluster_ctfs)[particles_to_use]
            selected_particles[clustered_particles == cluster_id][
                cluster_ctfs < ctf_threshold
            ] = True

    return selected_particles
