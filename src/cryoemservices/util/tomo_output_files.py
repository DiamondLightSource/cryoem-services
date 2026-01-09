from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from gemmi import cif

from cryoemservices.util.relion_service_options import RelionServiceOptions
from cryoemservices.util.spa_output_files import get_ice_ring_density


def _find_angle_index(split_name: List[str]) -> int:
    # Naming structure {tilt_series_name}_{tilt_number}_{tilt_angle}_{timestamp}.suffix
    for i, part in enumerate(split_name):
        if "." in part and part[-1].isnumeric():
            return i
    return -1


def _get_tilt_angle_v5_12(p: Path) -> str:
    split_name = p.stem.split("_")
    angle_idx = _find_angle_index(split_name)
    if angle_idx == -1:
        return "0.0"
    return split_name[angle_idx]


def _get_tilt_number_v5_12(p: Path) -> int:
    split_name = p.stem.split("_")
    angle_idx = _find_angle_index(split_name)
    try:
        int(split_name[angle_idx - 1])
    except ValueError:
        return 0
    return int(split_name[angle_idx - 1])


def _get_tilt_name_v5_12(p: Path) -> str:
    split_name = p.stem.split("_")
    angle_idx = _find_angle_index(split_name)
    if angle_idx == -1:
        # If no angle, strip off anything non-numeric at the end
        last_numeric = 0
        for i, part in enumerate(p.stem):
            if part.isnumeric():
                last_numeric = i
        if last_numeric == 0:
            # Can't find any numbers
            return p.stem
        return p.stem[: last_numeric + 1]
    return "_".join(split_name[: angle_idx - 1])


def _global_tilt_series_file(
    global_tilt_star: Path,
    tilt_series_name: str,
    tilt_series_star: str,
    relion_options: RelionServiceOptions,
):
    """Construction of files which list all tilt series"""
    tilt_series_line = [
        tilt_series_name,
        tilt_series_star,
        str(relion_options.voltage),
        str(relion_options.spher_aber),
        str(relion_options.ampl_contrast),
        str(relion_options.pixel_size),
        str(relion_options.invert_hand),
        "opticsGroup1",
        str(relion_options.pixel_size),
    ]

    # Create or append to the overall tilt series star file
    if not global_tilt_star.exists():
        output_cif = cif.Document()
        data_global = output_cif.add_new_block("global")

        global_loop = data_global.init_loop(
            "_rln",
            [
                "TomoName",
                "TomoTiltSeriesStarFile",
                "Voltage",
                "SphericalAberration",
                "AmplitudeContrast",
                "MicrographOriginalPixelSize",
                "TomoHand",
                "OpticsGroupName",
                "TomoTiltSeriesPixelSize",
            ],
        )
        global_loop.add_row(tilt_series_line)
        output_cif.write_file(str(global_tilt_star), style=cif.Style.Simple)
    else:
        tilts_cif = cif.read_file(str(global_tilt_star))
        tilts_loop = list(tilts_cif.find_block("global").find_loop("_rlnTomoName"))

        if tilt_series_name not in tilts_loop:
            with open(global_tilt_star, "a") as output_cif:
                output_cif.write(" ".join(tilt_series_line) + "\n")


def _import_output_files(
    job_dir: Path,
    input_file: Path,
    output_file: Path,
    relion_options: RelionServiceOptions,
    results: dict,
):
    """Import jobs save a list of all micrographs"""
    tilt_series_name = _get_tilt_name_v5_12(output_file)
    tilt_number = _get_tilt_number_v5_12(output_file)
    stage_tilt_angle = _get_tilt_angle_v5_12(output_file)

    # Construct the global file for all tilt series
    _global_tilt_series_file(
        job_dir / "tilt_series.star",
        tilt_series_name,
        f"Import/job001/tilt_series/{tilt_series_name}.star",
        relion_options,
    )

    # Prepare the file for this tilt series
    movies_file = job_dir / f"tilt_series/{tilt_series_name}.star"
    if not (job_dir / "tilt_series").is_dir():
        (job_dir / "tilt_series").mkdir()

    added_line = [
        str(output_file),
        str(relion_options.frame_count),
        str(stage_tilt_angle),
        str(relion_options.tilt_axis_angle),
        str(tilt_number * relion_options.frame_count * relion_options.dose_per_frame),
        str(relion_options.defocus),
    ]

    # Create or append to the star file for the individual tilt series
    if not Path(movies_file).exists():
        output_cif = cif.Document()
        data_movies = output_cif.add_new_block(tilt_series_name)

        movies_loop = data_movies.init_loop(
            "_rln",
            [
                "MicrographMovieName",
                "TomoTiltMovieFrameCount",
                "TomoNominalStageTiltAngle",
                "TomoNominalTiltAxisAngle",
                "MicrographPreExposure",
                "TomoNominalDefocus",
            ],
        )
        movies_loop.add_row(added_line)
        output_cif.write_file(str(movies_file), style=cif.Style.Simple)
    else:
        with open(movies_file, "a") as output_cif:
            output_cif.write(" ".join(added_line) + "\n")

    return {f"{job_dir}/tilt_series.star": ["TomogramGroupMetadata", ["relion"]]}


def _motioncorr_output_files(
    job_dir: Path,
    input_file: Path,
    output_file: Path,
    relion_options: RelionServiceOptions,
    results: dict,
):
    """Motion correction saves a list of micrographs and their motion"""
    tilt_series_name = _get_tilt_name_v5_12(output_file)
    tilt_number = _get_tilt_number_v5_12(output_file)
    stage_tilt_angle = _get_tilt_angle_v5_12(output_file)

    # Construct the global file for all tilt series
    _global_tilt_series_file(
        job_dir / "corrected_tilt_series.star",
        tilt_series_name,
        f"MotionCorr/job002/tilt_series/{tilt_series_name}.star",
        relion_options,
    )

    # Prepare the file for this tilt series
    movies_file = job_dir / f"tilt_series/{tilt_series_name}.star"
    if not (job_dir / "tilt_series").is_dir():
        (job_dir / "tilt_series").mkdir()

    added_line = [
        str(input_file),
        str(relion_options.frame_count),
        str(stage_tilt_angle),
        str(relion_options.tilt_axis_angle),
        str(tilt_number * relion_options.frame_count * relion_options.dose_per_frame),
        str(relion_options.defocus),
        str(output_file),
        str(output_file.with_suffix(".star")),
        str(results["total_motion"]),
        str(results["early_motion"]),
        str(results["late_motion"]),
    ]

    # TODO: Metadata file?

    # Create or append to the star file for the individual tilt series
    if not Path(movies_file).exists():
        output_cif = cif.Document()
        data_movies = output_cif.add_new_block(tilt_series_name)

        movies_loop = data_movies.init_loop(
            "_rln",
            [
                "MicrographMovieName",
                "TomoTiltMovieFrameCount",
                "TomoNominalStageTiltAngle",
                "TomoNominalTiltAxisAngle",
                "MicrographPreExposure",
                "TomoNominalDefocus",
                "MicrographName",
                "MicrographMetadata",
                "AccumMotionTotal",
                "AccumMotionEarly",
                "AccumMotionLate",
            ],
        )
        movies_loop.add_row(added_line)
        output_cif.write_file(str(movies_file), style=cif.Style.Simple)
    else:
        with open(movies_file, "a") as output_cif:
            output_cif.write(" ".join(added_line) + "\n")

    return {
        f"{job_dir}/corrected_tilt_series.star": ["TomogramGroupMetadata", ["relion"]]
    }


def _ctffind_output_files(
    job_dir: Path,
    input_file: Path,
    output_file: Path,
    relion_options: RelionServiceOptions,
    results: dict,
):
    """Ctf estimation saves a list of micrographs and their ctf parameters"""
    tilt_series_name = _get_tilt_name_v5_12(output_file)
    tilt_number = _get_tilt_number_v5_12(output_file)
    stage_tilt_angle = _get_tilt_angle_v5_12(output_file)

    # Construct the global file for all tilt series
    _global_tilt_series_file(
        job_dir / "tilt_series_ctf.star",
        tilt_series_name,
        f"CtfFind/job003/tilt_series/{tilt_series_name}.star",
        relion_options,
    )

    # Prepare the file for this tilt series
    movies_file = job_dir / f"tilt_series/{tilt_series_name}.star"
    if not (job_dir / "tilt_series").is_dir():
        (job_dir / "tilt_series").mkdir()

    # Results needed in the star file are stored in a txt file with the output
    with open(output_file.with_suffix(".txt"), "r") as f:
        ctf_results = f.readlines()[-1].split()
    ice_ring_density = get_ice_ring_density(output_file)

    added_line = [
        str(relion_options.frame_count),
        str(stage_tilt_angle),
        str(relion_options.tilt_axis_angle),
        str(tilt_number * relion_options.frame_count * relion_options.dose_per_frame),
        str(relion_options.defocus),
        str(input_file),
        str(output_file.with_suffix(".ctf")) + ":mrc",
        ctf_results[1],
        ctf_results[2],
        str(abs(float(ctf_results[1]) - float(ctf_results[2]))),
        ctf_results[3],
        ctf_results[5],
        ctf_results[6],
        str(ice_ring_density),
    ]

    # Create or append to the star file for the individual tilt series
    if not Path(movies_file).exists():
        output_cif = cif.Document()
        data_movies = output_cif.add_new_block(tilt_series_name)

        movies_loop = data_movies.init_loop(
            "_rln",
            [
                "TomoTiltMovieFrameCount",
                "TomoNominalStageTiltAngle",
                "TomoNominalTiltAxisAngle",
                "MicrographPreExposure",
                "TomoNominalDefocus",
                "MicrographName",
                "CtfImage",
                "DefocusU",
                "DefocusV",
                "CtfAstigmatism",
                "DefocusAngle",
                "CtfFigureOfMerit",
                "CtfMaxResolution",
                "CtfIceRingDensity",
            ],
        )
        movies_loop.add_row(added_line)
        output_cif.write_file(str(movies_file), style=cif.Style.Simple)
    else:
        with open(movies_file, "a") as output_cif:
            output_cif.write(" ".join(added_line) + "\n")

    return {f"{job_dir}/tilt_series_ctf.star": ["TomogramGroupMetadata", ["relion"]]}


def _exclude_tilt_output_files(
    job_dir: Path,
    input_file: Path,
    output_file: Path,
    relion_options: RelionServiceOptions,
    results: dict,
):
    """Tilt exclusion lists all tilts which have not been removed"""
    tilt_series_name = _get_tilt_name_v5_12(output_file)
    tilt_number = _get_tilt_number_v5_12(output_file)
    stage_tilt_angle = _get_tilt_angle_v5_12(output_file)

    # Construct the global file for all tilt series
    _global_tilt_series_file(
        job_dir / "selected_tilt_series.star",
        tilt_series_name,
        f"ExcludeTiltImages/job004/tilt_series/{tilt_series_name}.star",
        relion_options,
    )

    # Prepare the file for this tilt series
    movies_file = job_dir / f"tilt_series/{tilt_series_name}.star"
    if not (job_dir / "tilt_series").is_dir():
        (job_dir / "tilt_series").mkdir()

    # Try and figure out where the CTF output files will be
    mc_job_num_search = re.search("/job[0-9]+", str(input_file))
    if "MotionCorr" in input_file.parts and mc_job_num_search:
        job_number = int(mc_job_num_search[0][4:])
        relative_tilt = input_file.relative_to(
            f"{job_dir.parent.parent}/MotionCorr/job{job_number:03}"
        )
        ctf_txt_file = (
            job_dir.parent.parent
            / f"CtfFind/job{job_number + 1:03}"
            / relative_tilt.with_suffix(".txt")
        )
        micrograph_name = str(input_file)
    else:
        ctf_txt_file = input_file.with_suffix(".txt")
        micrograph_name = str(input_file.with_suffix(".mrc")).replace(
            "CtfFind/job003", "MotionCorr/job002"
        )

    # Later extraction jobs require some ctf parameters for the tilts
    if not ctf_txt_file.is_file():
        ctf_txt_file = ctf_txt_file.parent / (ctf_txt_file.stem + "_DW.txt")
    with open(ctf_txt_file, "r") as f:
        ctf_results = f.readlines()[-1].split()

    added_line = [
        str(relion_options.frame_count),
        str(stage_tilt_angle),
        str(relion_options.tilt_axis_angle),
        str(tilt_number * relion_options.frame_count * relion_options.dose_per_frame),
        str(relion_options.defocus),
        micrograph_name,
        ctf_results[1],
        ctf_results[2],
        ctf_results[3],
    ]

    # Create or append to the star file for the individual tilt series
    if not Path(movies_file).exists():
        output_cif = cif.Document()
        data_movies = output_cif.add_new_block(tilt_series_name)

        movies_loop = data_movies.init_loop(
            "_rln",
            [
                "TomoTiltMovieFrameCount",
                "TomoNominalStageTiltAngle",
                "TomoNominalTiltAxisAngle",
                "MicrographPreExposure",
                "TomoNominalDefocus",
                "MicrographName",
                "DefocusU",
                "DefocusV",
                "DefocusAngle",
            ],
        )
        movies_loop.add_row(added_line)
        output_cif.write_file(str(movies_file), style=cif.Style.Simple)
    else:
        tilt_cif_doc = cif.read_file(str(movies_file))
        tilt_loop = list(tilt_cif_doc.sole_block().find_loop("_rlnMicrographName"))

        if micrograph_name not in tilt_loop:
            with open(movies_file, "a") as output_cif:
                output_cif.write(" ".join(added_line) + "\n")
        else:
            index_to_replace = tilt_loop.index(micrograph_name)
            tilt_cif_item = tilt_cif_doc.sole_block()[0]
            tilt_cif_table = tilt_cif_doc.sole_block().item_as_table(tilt_cif_item)
            tilt_cif_table.remove_row(index_to_replace)
            tilt_cif_table.append_row(added_line)
            movies_file.unlink()
            tilt_cif_doc.write_file(str(movies_file))

    return {
        f"{job_dir}/selected_tilt_series.star": ["TomogramGroupMetadata", ["relion"]]
    }


def _align_tilt_output_files(
    job_dir: Path,
    input_file: Path,
    output_file: Path,
    relion_options: RelionServiceOptions,
    results: dict,
):
    """Alignment lists all the tilts and their aligned positions"""
    tilt_series_name = _get_tilt_name_v5_12(output_file)
    tilt_number = _get_tilt_number_v5_12(output_file)
    stage_tilt_angle = _get_tilt_angle_v5_12(output_file)

    # Construct the global file for all tilt series
    _global_tilt_series_file(
        job_dir / "aligned_tilt_series.star",
        tilt_series_name,
        f"AlignTiltSeries/job005/tilt_series/{tilt_series_name}.star",
        relion_options,
    )

    # Prepare the file for this tilt series
    movies_file = job_dir / f"tilt_series/{tilt_series_name}.star"
    if not (job_dir / "tilt_series").is_dir():
        (job_dir / "tilt_series").mkdir()

    # Try and figure out where the CTF output files will be
    job_num_search = re.search("/job[0-9]+", str(input_file))
    if "MotionCorr" in input_file.parts and job_num_search:
        job_number = int(job_num_search[0][4:])
        relative_tilt = input_file.relative_to(
            f"{job_dir.parent.parent}/MotionCorr/job{job_number:03}"
        )
        ctf_txt_file = (
            job_dir.parent.parent
            / f"CtfFind/job{job_number + 1:03}"
            / relative_tilt.with_suffix(".txt")
        )
        micrograph_name = str(input_file)
    elif "ExcludeTiltImages" in input_file.parts and job_num_search:
        job_number = int(job_num_search[0][4:])
        relative_tilt = input_file.relative_to(
            f"{job_dir.parent.parent}/ExcludeTiltImages/job{job_number:03}/tilts"
        )
        ctf_txt_file = (
            job_dir.parent.parent
            / f"CtfFind/job{job_number - 1:03}/Movies"
            / relative_tilt.with_suffix(".txt")
        )
        micrograph_name = str(input_file)
    else:
        ctf_txt_file = input_file.with_suffix(".txt")
        micrograph_name = str(input_file.with_suffix(".mrc")).replace(
            "CtfFind/job003", "MotionCorr/job002"
        )

    # Later extraction jobs require some ctf parameters for the tilts
    if not ctf_txt_file.is_file():
        ctf_txt_file = ctf_txt_file.parent / (ctf_txt_file.stem + "_DW.txt")
    with open(ctf_txt_file, "r") as f:
        ctf_results = f.readlines()[-1].split()

    added_line = [
        str(relion_options.frame_count),
        str(stage_tilt_angle),
        str(relion_options.tilt_axis_angle),
        str(tilt_number * relion_options.frame_count * relion_options.dose_per_frame),
        str(relion_options.defocus),
        micrograph_name,
        ctf_results[1],
        ctf_results[2],
        ctf_results[3],
        results["TomoXTilt"],
        results["TomoYTilt"],
        results["TomoZRot"],
        results["TomoXShiftAngst"],
        results["TomoYShiftAngst"],
    ]

    # Create or append to the star file for the individual tilt series
    if not Path(movies_file).exists():
        output_cif = cif.Document()
        data_movies = output_cif.add_new_block(tilt_series_name)

        movies_loop = data_movies.init_loop(
            "_rln",
            [
                "TomoTiltMovieFrameCount",
                "TomoNominalStageTiltAngle",
                "TomoNominalTiltAxisAngle",
                "MicrographPreExposure",
                "TomoNominalDefocus",
                "MicrographName",
                "DefocusU",
                "DefocusV",
                "DefocusAngle",
                "TomoXTilt",
                "TomoYTilt",
                "TomoZRot",
                "TomoXShiftAngst",
                "TomoYShiftAngst",
            ],
        )
        movies_loop.add_row(added_line)
        output_cif.write_file(str(movies_file), style=cif.Style.Simple)
    else:
        tilt_cif_doc = cif.read_file(str(movies_file))
        tilt_loop = list(tilt_cif_doc.sole_block().find_loop("_rlnMicrographName"))

        if micrograph_name not in tilt_loop:
            with open(movies_file, "a") as output_cif:
                output_cif.write(" ".join(added_line) + "\n")
        else:
            index_to_replace = tilt_loop.index(micrograph_name)
            tilt_cif_item = tilt_cif_doc.sole_block()[0]
            tilt_cif_table = tilt_cif_doc.sole_block().item_as_table(tilt_cif_item)
            tilt_cif_table.remove_row(index_to_replace)
            tilt_cif_table.append_row(added_line)
            movies_file.unlink()
            tilt_cif_doc.write_file(str(movies_file))

    return {
        f"{job_dir}/aligned_tilt_series.star": ["TomogramGroupMetadata", ["relion"]]
    }


def _tomogram_output_files(
    job_dir: Path,
    input_file: Path,
    output_file: Path,
    relion_options: RelionServiceOptions,
    results: dict,
):
    """Tomogram reconstruction lists the details of each tomogram"""
    tilt_series_name = _get_tilt_name_v5_12(output_file)
    tomograms_file = job_dir / "tomograms.star"

    added_line = [
        tilt_series_name,
        str(relion_options.voltage),
        str(relion_options.spher_aber),
        str(relion_options.ampl_contrast),
        str(relion_options.pixel_size),
        str(relion_options.invert_hand),
        "opticsGroup1",
        str(relion_options.pixel_size),
        f"AlignTiltSeries/job005/tilt_series/{tilt_series_name}.star",
        str(relion_options.pixel_size_downscaled / relion_options.pixel_size),
        str(relion_options.tomo_size_x),
        str(relion_options.tomo_size_y),
        str(relion_options.vol_z),
        str(output_file),
    ]

    # Create or append to the star file for the individual tilt series
    if not Path(tomograms_file).exists():
        output_cif = cif.Document()
        data_movies = output_cif.add_new_block("global")

        movies_loop = data_movies.init_loop(
            "_rln",
            [
                "TomoName",
                "Voltage",
                "SphericalAberration",
                "AmplitudeContrast",
                "MicrographOriginalPixelSize",
                "TomoHand",
                "OpticsGroupName",
                "TomoTiltSeriesPixelSize",
                "TomoTiltSeriesStarFile",
                "TomoTomogramBinning",
                "TomoSizeX",
                "TomoSizeY",
                "TomoSizeZ",
                "TomoReconstructedTomogram",
            ],
        )
        movies_loop.add_row(added_line)
        output_cif.write_file(str(tomograms_file), style=cif.Style.Simple)
    else:
        with open(tomograms_file, "a") as output_cif:
            output_cif.write(" ".join(added_line) + "\n")

    return {f"{job_dir}/tomograms.star": ["TomogramGroupMetadata", ["relion"]]}


def _denoising_output_files(
    job_dir: Path,
    input_file: Path,
    output_file: Path,
    relion_options: RelionServiceOptions,
    results: dict,
):
    """Denoising lists the details of each tomogram"""
    tilt_series_name = _get_tilt_name_v5_12(output_file)
    tomograms_file = job_dir / "tomograms.star"

    added_line = [
        tilt_series_name,
        str(relion_options.voltage),
        str(relion_options.spher_aber),
        str(relion_options.ampl_contrast),
        str(relion_options.pixel_size),
        str(relion_options.invert_hand),
        "opticsGroup1",
        str(relion_options.pixel_size),
        f"AlignTiltSeries/job005/tilt_series/{tilt_series_name}.star",
        str(relion_options.pixel_size_downscaled / relion_options.pixel_size),
        str(relion_options.tomo_size_x),
        str(relion_options.tomo_size_y),
        str(relion_options.vol_z),
        str(input_file),
        str(output_file),
    ]

    # Create or append to the star file for the individual tilt series
    if not Path(tomograms_file).exists():
        output_cif = cif.Document()
        data_movies = output_cif.add_new_block("global")

        movies_loop = data_movies.init_loop(
            "_rln",
            [
                "TomoName",
                "Voltage",
                "SphericalAberration",
                "AmplitudeContrast",
                "MicrographOriginalPixelSize",
                "TomoHand",
                "OpticsGroupName",
                "TomoTiltSeriesPixelSize",
                "TomoTiltSeriesStarFile",
                "TomoTomogramBinning",
                "TomoSizeX",
                "TomoSizeY",
                "TomoSizeZ",
                "TomoReconstructedTomogram",
                "TomoReconstructedTomogramDenoised",
            ],
        )
        movies_loop.add_row(added_line)
        output_cif.write_file(str(tomograms_file), style=cif.Style.Simple)
    else:
        with open(tomograms_file, "a") as output_cif:
            output_cif.write(" ".join(added_line) + "\n")

    return {f"{job_dir}/tomograms.star": ["TomogramGroupMetadata", ["relion"]]}


def _membrain_output_files(
    job_dir: Path,
    input_file: Path,
    output_file: Path,
    relion_options: RelionServiceOptions,
    results: dict,
):
    """Segmentation can list the details of each tomogram"""
    tilt_series_name = _get_tilt_name_v5_12(output_file)
    tomograms_file = job_dir / "tomograms.star"

    added_line = [
        tilt_series_name,
        str(relion_options.voltage),
        str(relion_options.spher_aber),
        str(relion_options.ampl_contrast),
        str(relion_options.pixel_size),
        str(relion_options.invert_hand),
        "opticsGroup1",
        str(relion_options.pixel_size),
        f"AlignTiltSeries/job005/tilt_series/{tilt_series_name}.star",
        str(relion_options.pixel_size_downscaled / relion_options.pixel_size),
        str(relion_options.tomo_size_x),
        str(relion_options.tomo_size_y),
        str(relion_options.vol_z),
        str(input_file),
        str(output_file),
    ]

    # Create or append to the star file for the individual tilt series
    if not Path(tomograms_file).exists():
        output_cif = cif.Document()
        data_movies = output_cif.add_new_block("global")

        movies_loop = data_movies.init_loop(
            "_rln",
            [
                "TomoName",
                "Voltage",
                "SphericalAberration",
                "AmplitudeContrast",
                "MicrographOriginalPixelSize",
                "TomoHand",
                "OpticsGroupName",
                "TomoTiltSeriesPixelSize",
                "TomoTiltSeriesStarFile",
                "TomoTomogramBinning",
                "TomoSizeX",
                "TomoSizeY",
                "TomoSizeZ",
                "TomoReconstructedTomogram",
                "TomoReconstructedTomogramSegmented",
            ],
        )
        movies_loop.add_row(added_line)
        output_cif.write_file(str(tomograms_file), style=cif.Style.Simple)
    else:
        with open(tomograms_file, "a") as output_cif:
            output_cif.write(" ".join(added_line) + "\n")

    return {f"{job_dir}/tomograms.star": ["TomogramGroupMetadata", ["relion"]]}


def _cryolo_output_files(
    job_dir: Path,
    input_file: Path,
    output_file: Path,
    relion_options: RelionServiceOptions,
    results: dict,
):
    """Cryolo picking lists the picks from each tomogram"""
    tilt_series_name = _get_tilt_name_v5_12(output_file)
    particles_file = job_dir / "particles.star"

    # Create the optimisation set if it does not exist
    if not (job_dir / "optimisation_set.star").is_file():
        job_dir_search = re.search(".+/job[0-9]+/", str(input_file))
        if job_dir_search:
            input_job_dir = job_dir_search[0]
        else:
            input_job_dir = "Denoise/job007/"
        with open(job_dir / "optimisation_set.star", "w") as opt_file:
            opt_file.write(
                "data_optimisation_set\n\nloop_\n"
                "_rlnTomoParticlesFile\n_rlnTomoTomogramsFile\n"
                f"{particles_file} {input_job_dir}tomograms.star"
            )

    # Create a particles star file if it does not exist
    if not particles_file.exists():
        with open(particles_file, "w") as pf:
            pf.write(
                "data_particles\n\nloop_\n"
                "_rlnTomoName\n_rlnCenteredCoordinateXAngst\n"
                "_rlnCenteredCoordinateYAngst\n_rlnCenteredCoordinateZAngst\n"
            )
    else:
        # Clean out any existing particles from this tomogram
        particles_doc = cif.read_file(str(particles_file))
        particles_block = particles_doc.sole_block()
        if tilt_series_name in particles_block.find_loop("_rlnTomoName"):
            indices_to_remove = np.where(
                np.array(particles_block.find_loop("_rlnTomoName")) == tilt_series_name
            )[0]
            particles_table = particles_block.item_as_table(particles_block[0])
            for i in range(len(indices_to_remove) - 1, -1, -1):
                particles_table.remove_row(indices_to_remove[i])
            particles_file.unlink()
            particles_doc.write_file(str(particles_file))

    # Read in the output particles
    particles_data = cif.read_file(str(output_file))
    try:
        cryolo_block = particles_data["cryolo"]
    except KeyError:
        return {str(particles_file): ["ParticleGroupMetadata", ["relion"]]}
    loop_x = cryolo_block.find_loop("_CoordinateX")
    loop_y = cryolo_block.find_loop("_CoordinateY")
    loop_z = cryolo_block.find_loop("_CoordinateZ")
    loop_width = cryolo_block.find_loop("_Width")
    loop_height = cryolo_block.find_loop("_Height")

    # Scale coordinates back to original tilt size
    scaling_factor = relion_options.pixel_size_downscaled / relion_options.pixel_size

    # Append all the particles to the particles file
    with open(particles_file, "a") as output_cif:
        for particle in range(len(loop_x)):
            # x and y coordinates are a corner, so need shifting by half the box size
            x_particle_center = (
                float(loop_x[particle]) + float(loop_width[particle]) / 2
            ) * scaling_factor
            y_particle_center = (
                float(loop_y[particle]) + float(loop_height[particle]) / 2
            ) * scaling_factor
            if -45 < relion_options.tilt_axis_angle < 45:
                # If given tilt axis of around 0, x and y are unchanged
                x_tomo_centered = x_particle_center - relion_options.tomo_size_x / 2
                y_tomo_centered = y_particle_center - relion_options.tomo_size_y / 2
            else:
                # Otherwise we need to flip x and y
                x_tomo_centered = y_particle_center - relion_options.tomo_size_y / 2
                y_tomo_centered = relion_options.tomo_size_x / 2 - x_particle_center

            # z coordinate is the mid-point so just needs scaling and centering
            z_particle_center = float(loop_z[particle]) * scaling_factor
            z_tomo_centered = z_particle_center - relion_options.vol_z / 2

            added_line = [
                tilt_series_name,
                str(x_tomo_centered * relion_options.pixel_size),
                str(y_tomo_centered * relion_options.pixel_size),
                str(z_tomo_centered * relion_options.pixel_size),
            ]
            output_cif.write(" ".join(added_line) + "\n")

    return {str(particles_file): ["ParticleGroupMetadata", ["relion"]]}


_output_files: Dict[str, Callable] = {
    "relion.importtomo": _import_output_files,
    "relion.motioncorr.own": _motioncorr_output_files,
    "relion.motioncorr.motioncor2": _motioncorr_output_files,
    "relion.ctffind.ctffind4": _ctffind_output_files,
    "relion.excludetilts": _exclude_tilt_output_files,
    "relion.aligntiltseries.aretomo": _align_tilt_output_files,
    "relion.reconstructtomograms": _tomogram_output_files,
    "relion.denoisetomo": _denoising_output_files,
    "membrain.segment": _membrain_output_files,
    "cryolo.autopick.tomo": _cryolo_output_files,
}


def create_tomo_output_files(
    job_type: str,
    job_dir: Path,
    input_file: Path,
    output_file: Path,
    relion_options: RelionServiceOptions,
    results: dict,
):
    return _output_files[job_type](
        job_dir, input_file, output_file, relion_options, results
    )
