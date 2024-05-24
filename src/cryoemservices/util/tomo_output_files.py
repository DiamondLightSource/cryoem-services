from __future__ import annotations

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
    return 0


def _get_tilt_angle_v5_12(p: Path) -> str:
    split_name = p.name.split("_")
    angle_idx = _find_angle_index(split_name)
    return split_name[angle_idx]


def _get_tilt_number_v5_12(p: Path) -> str:
    split_name = p.name.split("_")
    angle_idx = _find_angle_index(split_name)
    return split_name[angle_idx - 1]


def _get_tilt_tag_v5_12(p: Path) -> str:
    split_name = p.name.split("_")
    angle_idx = _find_angle_index(split_name)
    if split_name[angle_idx - 2].isnumeric():
        return "_".join(split_name[: angle_idx - 2])
    return "_".join(split_name[: angle_idx - 1])


def _global_tilt_series_file(
    global_tilt_star: Path,
    tilt_series_tag: str,
    tilt_series_star: str,
    relion_options: RelionServiceOptions,
):
    """Construction of files which list all tilt series"""
    tilt_series_line = [
        tilt_series_tag,
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

        if tilt_series_tag not in tilts_loop:
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
    tilt_series_tag = _get_tilt_tag_v5_12(output_file)
    tilt_number = _get_tilt_number_v5_12(output_file)
    stage_tilt_angle = _get_tilt_angle_v5_12(output_file)

    # Construct the global file for all tilt series
    _global_tilt_series_file(
        job_dir / "tilt_series.star",
        tilt_series_tag,
        f"Import/job001/tilt_series/{tilt_series_tag}.star",
        relion_options,
    )

    # Prepare the file for this tilt series
    movies_file = job_dir / f"tilt_series/{tilt_series_tag}.star"
    if not (job_dir / "tilt_series").is_dir():
        (job_dir / "tilt_series").mkdir()

    added_line = [
        str(output_file),
        relion_options.frame_count,
        stage_tilt_angle,
        relion_options.tilt_axis,
        int(tilt_number) * relion_options.frame_count * relion_options.dose_per_frame,
        relion_options.defocus,
    ]

    # Create or append to the star file for the individual tilt series
    if not Path(movies_file).exists():
        output_cif = cif.Document()
        data_movies = output_cif.add_new_block(tilt_series_tag)

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


def _motioncorr_output_files(
    job_dir: Path,
    input_file: Path,
    output_file: Path,
    relion_options: RelionServiceOptions,
    results: dict,
):
    """Motion correction saves a list of micrographs and their motion"""
    tilt_series_tag = _get_tilt_tag_v5_12(output_file)

    # Construct the global file for all tilt series
    _global_tilt_series_file(
        job_dir / "corrected_tilt_series.star",
        tilt_series_tag,
        f"MotionCorr/job002/tilt_series/{tilt_series_tag}.star",
        relion_options,
    )

    # Prepare the file for this tilt series
    movies_file = job_dir / f"tilt_series/{tilt_series_tag}.star"
    if not (job_dir / "tilt_series").is_dir():
        (job_dir / "tilt_series").mkdir()

    added_line = [
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
        data_movies = output_cif.add_new_block(tilt_series_tag)

        movies_loop = data_movies.init_loop(
            "_rln",
            [
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


def _ctffind_output_files(
    job_dir: Path,
    input_file: Path,
    output_file: Path,
    relion_options: RelionServiceOptions,
    results: dict,
):
    """Ctf estimation saves a list of micrographs and their ctf parameters"""
    tilt_series_tag = _get_tilt_tag_v5_12(output_file)

    # Construct the global file for all tilt series
    _global_tilt_series_file(
        job_dir / "tilt_series_ctf.star",
        tilt_series_tag,
        f"CtfFind/job003/tilt_series/{tilt_series_tag}.star",
        relion_options,
    )

    # Prepare the file for this tilt series
    movies_file = job_dir / f"tilt_series/{tilt_series_tag}.star"
    if not (job_dir / "tilt_series").is_dir():
        (job_dir / "tilt_series").mkdir()

    # Results needed in the star file are stored in a txt file with the output
    with open(output_file.with_suffix(".txt"), "r") as f:
        ctf_results = f.readlines()[-1].split()
    ice_ring_density = get_ice_ring_density(output_file)

    added_line = [
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
        data_movies = output_cif.add_new_block(tilt_series_tag)

        movies_loop = data_movies.init_loop(
            "_rln",
            [
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


_output_files: Dict[str, Callable] = {
    "relion.import.tilt_series": _import_output_files,
    "relion.motioncorr.own": _motioncorr_output_files,
    "relion.motioncorr.motioncor2": _motioncorr_output_files,
    "relion.ctffind.ctffind4": _ctffind_output_files,
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
