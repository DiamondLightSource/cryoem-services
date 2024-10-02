from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import mrcfile
import numpy as np


def align_symmetry(volume_file: Path, symmetry: str):
    """Use Relion to align a volume with the given symmetry"""
    aligned_file = volume_file.parent / f"{volume_file.stem}_aligned_{symmetry}.mrc"
    align_command = [
        "relion_align_symmetry",
        "--i",
        str(volume_file),
        "--o",
        str(aligned_file),
        "--sym",
        symmetry,
    ]
    result = subprocess.run(align_command, capture_output=True)
    return result.returncode, aligned_file


def force_symmetry(aligned_file: Path, symmetry: str):
    """Run Relion's image handler to force a volume to the given symmetry"""
    symmetrised_file = aligned_file.parent / f"{aligned_file.stem}_symmetrised.mrc"
    image_handler_command = [
        "relion_image_handler",
        "--i",
        str(aligned_file),
        "--o",
        str(symmetrised_file),
        "--sym",
        symmetry,
    ]
    result = subprocess.run(image_handler_command, capture_output=True)
    return result.returncode, symmetrised_file


def find_difference(aligned_file: Path, symmetrised_file: Path):
    """Compare two volumes using sum-of-squares"""
    with mrcfile.open(aligned_file) as mrc:
        aligned_data = np.copy(mrc.data)

    with mrcfile.open(symmetrised_file) as mrc:
        sym_data = np.copy(mrc.data)

    return np.sum((aligned_data - sym_data) ** 2) / np.size(aligned_data)


def determine_symmetry(volume: Path):
    # List of the symmetries to test
    symmetry_list = np.array(["C2", "C3", "C4", "C5", "C6", "C7", "C8", "T", "O", "I"])

    # Find the difference between aligned and symmetrised cases for each symmetry
    difference_scores = np.zeros(len(symmetry_list), dtype=float)
    all_symmetrised_files = np.array([], dtype=str)
    for i, sym in enumerate(symmetry_list):
        align_result, aligned_file = align_symmetry(volume, sym)
        if align_result:
            difference_scores[i] = 100
            print(f"Failed to align {volume} to {sym}")
            continue

        symmetrised_result, symmetrised_file = force_symmetry(aligned_file, sym)
        if symmetrised_result:
            difference_scores[i] = 100
            print(f"Failed to make {volume} into {sym}")
            continue

        difference_scores[i] = find_difference(aligned_file, symmetrised_file)
        all_symmetrised_files = np.append(all_symmetrised_files, str(symmetrised_file))

        print(f"{sym}: {difference_scores[i]}")

    # Empirically determined reference scores for each symmetry compared to C2
    # found by running alignment on random noise
    assessment_scores = np.array(
        [1, 1.166, 1.507, 1.489, 1.589, 1.634, 1.709, 1.772, 1.927, 1.964]
    )

    # Assess the symmetry by comparing all scores to the C2 score and the references
    sym_tests = difference_scores / difference_scores[0] / assessment_scores
    expected_sym = symmetry_list[sym_tests == min(sym_tests)][0]

    # Examine a few specific more challenging symmetries
    if difference_scores[-3] < difference_scores[-4]:
        # T better than C8
        if difference_scores[-2] < difference_scores[-4]:
            # O also better than C8
            expected_sym = "O"
        else:
            # Otherwise T
            expected_sym = "T"
    elif difference_scores[-2] < difference_scores[-3]:
        # O better than T
        expected_sym = "O"
    elif difference_scores[-1] < difference_scores[-2]:
        # I better than O
        expected_sym = "I"

    print(f"{volume} is predicted to be {expected_sym}")
    return expected_sym, all_symmetrised_files[symmetry_list == expected_sym]


def run():
    parser = argparse.ArgumentParser(description="Run symmetry alignment for file")
    parser.add_argument(
        "--volume",
        help="File to symmetrise",
        required=True,
    )
    args = parser.parse_args()

    if Path(args.volume).is_file():
        determine_symmetry(Path(args.volume))
    else:
        raise FileNotFoundError(f"Cannot find {args.volume}")


if __name__ == "__main__":
    run()
