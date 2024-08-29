from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List

from gemmi import cif

from cryoemservices.util.relion_service_options import RelionServiceOptions
from cryoemservices.util.spa_output_files import get_ice_ring_density


def _find_angle_index(split_name: List[str]) -> int:
    # Naming structure {tilt_series_name}_{tilt_number}_{tilt_angle}_{timestamp}.suffix
    for i, part in enumerate(split_name):
        if "." in part:
            return i
    return -1


def _get_tilt_angle_v5_12(p: Path) -> str:
    split_name = p.name.split("_")
    angle_idx = _find_angle_index(split_name)
    if angle_idx == -1:
        return "0.0"
    return split_name[angle_idx]


def _get_tilt_number_v5_12(p: Path) -> str:
    split_name = p.name.split("_")
    angle_idx = _find_angle_index(split_name)
    try:
        int(split_name[angle_idx - 1])
    except ValueError:
        return "0"
    return split_name[angle_idx - 1]


def _get_tilt_name_v5_12(p: Path) -> str:
    split_name = p.name.split("_")
    angle_idx = _find_angle_index(split_name)
    if angle_idx == -1:
        return p.name
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
        "optics1",
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
        str(
            int(tilt_number)
            * relion_options.frame_count
            * relion_options.dose_per_frame
        ),
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

    return {f"{job_dir}/tilt_series.star": ["TomogramGroupMetadata", "relion"]}


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
        str(
            int(tilt_number)
            * relion_options.frame_count
            * relion_options.dose_per_frame
        ),
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
        f"{job_dir}/corrected_tilt_series.star": ["TomogramGroupMetadata", "relion"]
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
        str(
            int(tilt_number)
            * relion_options.frame_count
            * relion_options.dose_per_frame
        ),
        str(relion_options.defocus),
        str(input_file),
        str(stage_tilt_angle),
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
                "TomoNominalTiltAxisAngle",
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

    return {f"{job_dir}/tilt_series_ctf.star": ["TomogramGroupMetadata", "relion"]}


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

    added_line = [
        str(relion_options.frame_count),
        str(stage_tilt_angle),
        str(relion_options.tilt_axis_angle),
        str(
            int(tilt_number)
            * relion_options.frame_count
            * relion_options.dose_per_frame
        ),
        str(relion_options.defocus),
        str(input_file),
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
            ],
        )
        movies_loop.add_row(added_line)
        output_cif.write_file(str(movies_file), style=cif.Style.Simple)
    else:
        with open(movies_file, "a") as output_cif:
            output_cif.write(" ".join(added_line) + "\n")

    return {f"{job_dir}/selected_tilt_series.star": ["TomogramGroupMetadata", "relion"]}


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
    mc_job_num_search = re.search("/job[0-9]+", str(input_file))
    if input_file.is_relative_to("MotionCorr") and mc_job_num_search:
        job_number = int(mc_job_num_search[0][4:])
        relative_tilt = input_file.relative_to(f"MotionCorr/job{job_number:03}")
        ctf_txt_file = Path(
            f"CtfFind/job{job_number + 1:03}"
        ) / relative_tilt.with_suffix(".txt")
    else:
        ctf_txt_file = input_file.with_suffix(".txt")

    # Later extraction jobs require some ctf parameters for the tilts
    if ctf_txt_file.is_file():
        with open(ctf_txt_file, "r") as f:
            ctf_results = f.readlines()[-1].split()
    else:
        ctf_results = ["error", "ctf", "not", "found"]

    added_line = [
        str(relion_options.frame_count),
        str(stage_tilt_angle),
        str(relion_options.tilt_axis_angle),
        str(
            int(tilt_number)
            * relion_options.frame_count
            * relion_options.dose_per_frame
        ),
        str(relion_options.defocus),
        str(input_file),
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
        with open(movies_file, "a") as output_cif:
            output_cif.write(" ".join(added_line) + "\n")

    return {f"{job_dir}/aligned_tilt_series.star": ["TomogramGroupMetadata", "relion"]}


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
        "optics1",
        str(relion_options.pixel_size_downscaled),
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

    return {f"{job_dir}/tomograms.star": ["TomogramGroupMetadata", "relion"]}


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
        "optics1",
        str(relion_options.pixel_size_downscaled),
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

    return {f"{job_dir}/tomograms.star": ["TomogramGroupMetadata", "relion"]}


_output_files: Dict[str, Callable] = {
    "relion.import.tilt_series": _import_output_files,
    "relion.motioncorr.own": _motioncorr_output_files,
    "relion.motioncorr.motioncor2": _motioncorr_output_files,
    "relion.ctffind.ctffind4": _ctffind_output_files,
    "relion.excludetilts": _exclude_tilt_output_files,
    "relion.aligntiltseries": _align_tilt_output_files,
    "relion.reconstructtomograms": _tomogram_output_files,
    "relion.denoisetomo": _denoising_output_files,
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
