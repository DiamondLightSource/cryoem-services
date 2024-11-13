from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import motioncorr
from cryoemservices.util.relion_service_options import RelionServiceOptions


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.motioncorr.subprocess.run")
def test_motioncor2_service_spa(mock_subprocess, offline_transport, tmp_path):
    """
    Send a test message to MotionCorr for SPA using MotionCor2
    This should call the mock subprocess then send messages on to
    the ispyb_connector and images services.
    It also creates the next jobs (ctffind and two icebreaker jobs)
    and the node_creator is called for both import and motion correction.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    (tmp_path / "gain.mrc").touch()
    movie = Path(f"{tmp_path}/Movies/sample.tiff")
    movie.parent.mkdir(parents=True)
    movie.touch()

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    motioncorr_test_message = {
        "parameters": {
            "experiment_type": "spa",
            "pixel_size": 0.1,
            "autopick": {"autopick": "autopick"},
            "ctf": {"ctf": "ctf"},
            "movie": str(movie),
            "mrc_out": f"{tmp_path}/MotionCorr/job002/Movies/sample.mrc",
            "patch_sizes": {"x": 5, "y": 5},
            "gpu": 0,
            "threads": 1,
            "gain_ref": f"{tmp_path}/gain.mrc",
            "mc_uuid": 0,
            "picker_uuid": 0,
            "rot_gain": 1,
            "flip_gain": 1,
            "dark": "dark",
            "use_gpus": 1,
            "sum_range": {"sum1": "sum1", "sum2": "sum2"},
            "iter": 1,
            "tol": 1.1,
            "throw": 1,
            "trunc": 1,
            "fm_ref": 1,
            "voltage": 300,
            "dose_per_frame": 1,
            "init_dose": 1,
            "use_motioncor2": True,
            "fm_int_file": "fm_int_file",
            "mag": {"mag1": "mag1", "mag2": "mag2"},
            "motion_corr_binning": 2,
            "serial": 1,
            "in_suffix": "mrc",
            "eer_sampling": 1,
            "out_stack": 1,
            "bft": {"global_motion": 500, "local_motion": 150},
            "group": 1,
            "defect_file": "file",
            "arc_dir": "arc_dir",
            "in_fm_motion": 1,
            "split_sum": 1,
            "movie_id": 1,
            "do_icebreaker_jobs": True,
            "relion_options": {
                "pixel_size": 0.1,
                "do_icebreaker_jobs": True,
                "cryolo_threshold": 0.3,
                "ampl_contrast": 0.2,
            },
        },
        "content": "dummy",
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["voltage"] = motioncorr_test_message["parameters"]["voltage"]
    output_relion_options["pixel_size"] = motioncorr_test_message["parameters"][
        "pixel_size"
    ]
    output_relion_options["dose_per_frame"] = motioncorr_test_message["parameters"][
        "dose_per_frame"
    ]
    output_relion_options["gain_ref"] = motioncorr_test_message["parameters"][
        "gain_ref"
    ]
    output_relion_options["motion_corr_binning"] = motioncorr_test_message[
        "parameters"
    ]["motion_corr_binning"]
    output_relion_options.update(
        motioncorr_test_message["parameters"]["relion_options"]
    )
    output_relion_options["pixel_size"] = 0.2
    output_relion_options["eer_grouping"] = 0

    # Set up the mock service
    service = motioncorr.MotionCorr()
    service.transport = offline_transport
    service.start()

    # Work out the expected shifts
    service.x_shift_list = [-3.0, 3.0]
    service.y_shift_list = [4.0, -4.0]
    service.each_total_motion = [5.0, 5.0]
    total_motion = 10.0
    early_motion = 10.0
    late_motion = 0.0
    average_motion_per_frame = 5

    # Send a message to the service
    service.motion_correction(None, header=header, message=motioncorr_test_message)

    mc_command = [
        "MotionCor2",
        "-InTiff",
        str(movie),
        "-OutMrc",
        motioncorr_test_message["parameters"]["mrc_out"],
        "-PixSize",
        str(motioncorr_test_message["parameters"]["pixel_size"]),
        "-FmDose",
        "1.0",
        "-Patch",
        "5 5",
        "-Gpu",
        "0",
        "-Gain",
        motioncorr_test_message["parameters"]["gain_ref"],
        "-RotGain",
        "1",
        "-FlipGain",
        "1",
        "-Dark",
        "dark",
        "-UseGpus",
        "1",
        "-SumRange",
        "sum1 sum2",
        "-Iter",
        "1",
        "-Tol",
        "1.1",
        "-Throw",
        "1",
        "-Trunc",
        "1",
        "-FmRef",
        "1",
        "-Kv",
        "300",
        "-InitDose",
        "1.0",
        "-Mag",
        "mag1 mag2",
        "-FtBin",
        "2.0",
        "-Serial",
        "1",
        "-InSuffix",
        "mrc",
        "-EerSampling",
        "1",
        "-OutStack",
        "1",
        "-Bft",
        "500 150",
        "-Group",
        "1",
        "-DefectFile",
        "file",
        "-ArcDir",
        "arc_dir",
        "-InFmMotion",
        "1",
        "-SplitSum",
        "1",
    ]

    assert mock_subprocess.call_count == 4
    mock_subprocess.assert_called_with(mc_command, capture_output=True)

    # Check that the correct messages were sent
    offline_transport.send.assert_any_call(
        destination="icebreaker",
        message={
            "parameters": {
                "icebreaker_type": "micrographs",
                "input_micrographs": motioncorr_test_message["parameters"]["mrc_out"],
                "output_path": f"{tmp_path}/IceBreaker/job003/",
                "mc_uuid": motioncorr_test_message["parameters"]["mc_uuid"],
                "relion_options": output_relion_options,
                "total_motion": total_motion,
                "early_motion": early_motion,
                "late_motion": late_motion,
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="icebreaker",
        message={
            "parameters": {
                "icebreaker_type": "enhancecontrast",
                "input_micrographs": motioncorr_test_message["parameters"]["mrc_out"],
                "output_path": f"{tmp_path}/IceBreaker/job004/",
                "mc_uuid": motioncorr_test_message["parameters"]["mc_uuid"],
                "relion_options": output_relion_options,
                "total_motion": total_motion,
                "early_motion": early_motion,
                "late_motion": late_motion,
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="ctffind",
        message={
            "parameters": {
                "ctf": "ctf",
                "input_image": motioncorr_test_message["parameters"]["mrc_out"],
                "mc_uuid": motioncorr_test_message["parameters"]["mc_uuid"],
                "picker_uuid": motioncorr_test_message["parameters"]["picker_uuid"],
                "relion_options": output_relion_options,
                "amplitude_contrast": output_relion_options["ampl_contrast"],
                "experiment_type": "spa",
                "output_image": f"{tmp_path}/CtfFind/job006/Movies/sample.ctf",
                "pixel_size": motioncorr_test_message["parameters"]["pixel_size"] * 2,
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="ispyb_connector",
        message={
            "parameters": {
                "first_frame": 1,
                "last_frame": 2,
                "total_motion": total_motion,
                "average_motion_per_frame": average_motion_per_frame,
                "drift_plot_full_path": f"{tmp_path}/MotionCorr/job002/Movies/sample_drift_plot.json",
                "micrograph_snapshot_full_path": f"{tmp_path}/MotionCorr/job002/Movies/sample.jpeg",
                "micrograph_full_path": motioncorr_test_message["parameters"][
                    "mrc_out"
                ],
                "patches_used_x": motioncorr_test_message["parameters"]["patch_sizes"][
                    "x"
                ],
                "patches_used_y": motioncorr_test_message["parameters"]["patch_sizes"][
                    "y"
                ],
                "buffer_store": motioncorr_test_message["parameters"]["mc_uuid"],
                "dose_per_frame": motioncorr_test_message["parameters"][
                    "dose_per_frame"
                ],
                "ispyb_command": "buffer",
                "buffer_command": {"ispyb_command": "insert_motion_correction"},
            },
            "content": {"dummy": "dummy"},
        },
    )
    offline_transport.send.assert_any_call(
        destination="images",
        message={
            "image_command": "mrc_to_jpeg",
            "file": motioncorr_test_message["parameters"]["mrc_out"],
        },
    )
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "experiment_type": "spa",
                "job_type": "relion.import.movies",
                "input_file": str(movie),
                "output_file": f"{tmp_path}/Import/job001/Movies/sample.tiff",
                "relion_options": output_relion_options,
                "command": "",
                "stdout": "",
                "stderr": "",
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "experiment_type": "spa",
                "job_type": "relion.motioncorr.motioncor2",
                "input_file": f"{tmp_path}/Import/job001/Movies/sample.tiff",
                "output_file": motioncorr_test_message["parameters"]["mrc_out"],
                "relion_options": output_relion_options,
                "command": " ".join(mc_command),
                "stdout": "stdout",
                "stderr": "stderr",
                "results": {
                    "total_motion": total_motion,
                    "early_motion": early_motion,
                    "late_motion": late_motion,
                },
            },
            "content": "dummy",
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.motioncorr.subprocess.run")
def test_motioncor_relion_service_spa(mock_subprocess, offline_transport, tmp_path):
    """
    Send a test message to MotionCorr for SPA using Relion's own version
    This should call the mock subprocess then send messages on to
    the ispyb_connector and images services.
    It also creates the next jobs (ctffind and two icebreaker jobs)
    and the node_creator is called for both import and motion correction.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    (tmp_path / "gain.mrc").touch()
    movie = Path(f"{tmp_path}/Movies/sample.eer")
    movie.parent.mkdir(parents=True)
    movie.touch()

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    motioncorr_test_message = {
        "parameters": {
            "experiment_type": "spa",
            "pixel_size": 0.1,
            "autopick": {"autopick": "autopick"},
            "ctf": {"ctf": "ctf"},
            "movie": str(movie),
            "mrc_out": f"{tmp_path}/MotionCorr/job002/Movies/sample.mrc",
            "patch_sizes": {"x": 5, "y": 5},
            "gpu": 0,
            "threads": 2,
            "gain_ref": f"{tmp_path}/gain.mrc",
            "mc_uuid": 0,
            "picker_uuid": 0,
            "rot_gain": 1,
            "flip_gain": 1,
            "dark": "dark",
            "use_gpus": 1,
            "sum_range": {"sum1": "sum1", "sum2": "sum2"},
            "iter": 1,
            "tol": 1.1,
            "throw": 1,
            "trunc": 1,
            "fm_ref": 1,
            "voltage": 300,
            "dose_per_frame": 1,
            "init_dose": 1,
            "use_motioncor2": False,
            "fm_int_file": f"{tmp_path}/fm_int_file.txt",
            "mag": {"mag1": "mag1", "mag2": "mag2"},
            "motion_corr_binning": 2,
            "serial": 1,
            "in_suffix": "mrc",
            "eer_sampling": 1,
            "out_stack": 1,
            "bft": {"global_motion": 500, "local_motion": 150},
            "group": 1,
            "defect_file": "file",
            "arc_dir": "arc_dir",
            "in_fm_motion": 1,
            "split_sum": 1,
            "movie_id": 1,
            "do_icebreaker_jobs": True,
            "relion_options": {
                "pixel_size": 0.1,
                "do_icebreaker_jobs": True,
                "cryolo_threshold": 0.3,
                "ampl_contrast": 0.2,
            },
        },
        "content": "dummy",
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["voltage"] = motioncorr_test_message["parameters"]["voltage"]
    output_relion_options["pixel_size"] = motioncorr_test_message["parameters"][
        "pixel_size"
    ]
    output_relion_options["dose_per_frame"] = motioncorr_test_message["parameters"][
        "dose_per_frame"
    ]
    output_relion_options["gain_ref"] = motioncorr_test_message["parameters"][
        "gain_ref"
    ]
    output_relion_options["motion_corr_binning"] = motioncorr_test_message[
        "parameters"
    ]["motion_corr_binning"]
    output_relion_options.update(
        motioncorr_test_message["parameters"]["relion_options"]
    )
    output_relion_options["pixel_size"] = 0.2
    output_relion_options["eer_grouping"] = 10

    # Write sample eer frame file
    with open(f"{tmp_path}/fm_int_file.txt", "w") as fm_int_file:
        fm_int_file.write("100 10 1\n")

    # Touch expected output file
    (tmp_path / "MotionCorr/job002/Movies").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "MotionCorr/job002/Movies/sample.star", "w") as relion_output:
        relion_output.write(
            "data_global_shift\nloop_\n_rlnMicrographShiftX\n_rlnMicrographShiftY\n"
        )

    # Set up the mock service
    service = motioncorr.MotionCorr()
    service.transport = offline_transport
    service.start()

    # Work out the expected shifts
    service.x_shift_list = [-3.0, 3.0]
    service.y_shift_list = [4.0, -4.0]
    service.each_total_motion = [5.0, 5.0]
    total_motion = 10.0
    early_motion = 10.0
    late_motion = 0.0
    average_motion_per_frame = 5

    # Send a message to the service
    service.motion_correction(None, header=header, message=motioncorr_test_message)

    mc_command = [
        "relion_motion_correction",
        "--use_own",
        "--in_movie",
        str(movie),
        "--out_mic",
        motioncorr_test_message["parameters"]["mrc_out"],
        "--angpix",
        str(motioncorr_test_message["parameters"]["pixel_size"]),
        "--dose_per_frame",
        "1.0",
        "--patch_x",
        "5",
        "--patch_y",
        "5",
        "--j",
        "2",
        "--gainref",
        motioncorr_test_message["parameters"]["gain_ref"],
        "--gain_rot",
        "1",
        "--gain_flip",
        "1",
        "--voltage",
        "300",
        "--preexposure",
        "1.0",
        "--bin_factor",
        "2.0",
        "--eer_upsampling",
        "1",
        "--bfactor",
        "150",
        "--defect_file",
        "file",
        "--eer_grouping",
        "10",
        "--dose_weighting",
        "--i",
        "dummy",
    ]

    assert mock_subprocess.call_count == 4
    mock_subprocess.assert_called_with(mc_command, capture_output=True)

    # Check that the correct messages were sent
    offline_transport.send.assert_any_call(
        destination="icebreaker",
        message={
            "parameters": {
                "icebreaker_type": "micrographs",
                "input_micrographs": motioncorr_test_message["parameters"]["mrc_out"],
                "output_path": f"{tmp_path}/IceBreaker/job003/",
                "mc_uuid": motioncorr_test_message["parameters"]["mc_uuid"],
                "relion_options": output_relion_options,
                "total_motion": total_motion,
                "early_motion": early_motion,
                "late_motion": late_motion,
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="icebreaker",
        message={
            "parameters": {
                "icebreaker_type": "enhancecontrast",
                "input_micrographs": motioncorr_test_message["parameters"]["mrc_out"],
                "output_path": f"{tmp_path}/IceBreaker/job004/",
                "mc_uuid": motioncorr_test_message["parameters"]["mc_uuid"],
                "relion_options": output_relion_options,
                "total_motion": total_motion,
                "early_motion": early_motion,
                "late_motion": late_motion,
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="ctffind",
        message={
            "parameters": {
                "ctf": "ctf",
                "input_image": motioncorr_test_message["parameters"]["mrc_out"],
                "mc_uuid": motioncorr_test_message["parameters"]["mc_uuid"],
                "picker_uuid": motioncorr_test_message["parameters"]["picker_uuid"],
                "relion_options": output_relion_options,
                "amplitude_contrast": output_relion_options["ampl_contrast"],
                "experiment_type": "spa",
                "output_image": f"{tmp_path}/CtfFind/job006/Movies/sample.ctf",
                "pixel_size": motioncorr_test_message["parameters"]["pixel_size"] * 2,
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="ispyb_connector",
        message={
            "parameters": {
                "first_frame": 1,
                "last_frame": 2,
                "total_motion": total_motion,
                "average_motion_per_frame": average_motion_per_frame,
                "drift_plot_full_path": f"{tmp_path}/MotionCorr/job002/Movies/sample_drift_plot.json",
                "micrograph_snapshot_full_path": f"{tmp_path}/MotionCorr/job002/Movies/sample.jpeg",
                "micrograph_full_path": motioncorr_test_message["parameters"][
                    "mrc_out"
                ],
                "patches_used_x": motioncorr_test_message["parameters"]["patch_sizes"][
                    "x"
                ],
                "patches_used_y": motioncorr_test_message["parameters"]["patch_sizes"][
                    "y"
                ],
                "buffer_store": motioncorr_test_message["parameters"]["mc_uuid"],
                "dose_per_frame": motioncorr_test_message["parameters"][
                    "dose_per_frame"
                ],
                "ispyb_command": "buffer",
                "buffer_command": {"ispyb_command": "insert_motion_correction"},
            },
            "content": {"dummy": "dummy"},
        },
    )
    offline_transport.send.assert_any_call(
        destination="images",
        message={
            "image_command": "mrc_to_jpeg",
            "file": motioncorr_test_message["parameters"]["mrc_out"],
        },
    )
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "experiment_type": "spa",
                "job_type": "relion.import.movies",
                "input_file": str(movie),
                "output_file": f"{tmp_path}/Import/job001/Movies/sample.eer",
                "relion_options": output_relion_options,
                "command": "",
                "stdout": "",
                "stderr": "",
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "experiment_type": "spa",
                "job_type": "relion.motioncorr.own",
                "input_file": f"{tmp_path}/Import/job001/Movies/sample.eer",
                "output_file": motioncorr_test_message["parameters"]["mrc_out"],
                "relion_options": output_relion_options,
                "command": " ".join(mc_command),
                "stdout": "stdout",
                "stderr": "stderr",
                "results": {
                    "total_motion": total_motion,
                    "early_motion": early_motion,
                    "late_motion": late_motion,
                },
            },
            "content": "dummy",
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.motioncorr.subprocess.run")
def test_motioncor2_service_tomo(mock_subprocess, offline_transport, tmp_path):
    """
    Send a test message to MotionCorr for tomography using MotionCor2
    This should call the mock subprocess then send messages on to
    the murfey_feedback, ispyb_connector and images services.
    It also creates the ctffind job.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    (tmp_path / "gain.mrc").touch()
    movie = Path(f"{tmp_path}/Movies/sample.tiff")
    movie.parent.mkdir(parents=True)
    movie.touch()

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    motioncorr_test_message = {
        "parameters": {
            "experiment_type": "tomography",
            "pixel_size": 0.1,
            "autopick": {"autopick": "autopick"},
            "ctf": {"ctf": "ctf"},
            "movie": str(movie),
            "mrc_out": f"{tmp_path}/MotionCorr/job002/Movies/sample_motion_corrected.mrc",
            "patch_sizes": {"x": 5, "y": 5},
            "gpu": 0,
            "threads": 1,
            "gain_ref": f"{tmp_path}/gain.mrc",
            "mc_uuid": 0,
            "picker_uuid": 0,
            "rot_gain": None,
            "flip_gain": None,
            "dark": None,
            "use_gpus": None,
            "sum_range": None,
            "iter": None,
            "tol": None,
            "throw": None,
            "trunc": None,
            "fm_ref": 1,
            "voltage": None,
            "dose_per_frame": 1,
            "use_motioncor2": True,
            "fm_int_file": None,
            "init_dose": None,
            "mag": None,
            "motion_corr_binning": None,
            "serial": None,
            "in_suffix": None,
            "eer_sampling": None,
            "out_stack": None,
            "bft": None,
            "group": None,
            "detect_file": None,
            "arc_dir": None,
            "in_fm_motion": None,
            "split_sum": None,
            "movie_id": 1,
            "relion_options": {
                "frame_count": 5,
                "tilt_axis_angle": 83.0,
                "defocus": -2.0,
                "invert_hand": 1,
            },
        },
        "content": "dummy",
    }

    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["pixel_size"] = motioncorr_test_message["parameters"][
        "pixel_size"
    ]
    output_relion_options["dose_per_frame"] = motioncorr_test_message["parameters"][
        "dose_per_frame"
    ]
    output_relion_options["gain_ref"] = motioncorr_test_message["parameters"][
        "gain_ref"
    ]
    output_relion_options.update(
        motioncorr_test_message["parameters"]["relion_options"]
    )
    output_relion_options["eer_grouping"] = 0

    # Set up the mock service
    service = motioncorr.MotionCorr()
    service.transport = offline_transport
    service.start()

    # Work out the expected shifts
    service.x_shift_list = [-3.0, 3.0]
    service.y_shift_list = [4.0, -4.0]
    service.each_total_motion = [5.0, 5.0]
    total_motion = 10.0
    early_motion = 10.0
    late_motion = 0.0
    average_motion_per_frame = 5

    # Send a message to the service
    service.motion_correction(None, header=header, message=motioncorr_test_message)

    mc_command = [
        "MotionCor2",
        "-InTiff",
        str(movie),
        "-OutMrc",
        motioncorr_test_message["parameters"]["mrc_out"],
        "-PixSize",
        str(motioncorr_test_message["parameters"]["pixel_size"]),
        "-FmDose",
        "1.0",
        "-Patch",
        "5 5",
        "-Gpu",
        "0",
        "-Gain",
        motioncorr_test_message["parameters"]["gain_ref"],
        "-FmRef",
        "1",
    ]

    assert mock_subprocess.call_count == 4
    mock_subprocess.assert_called_with(
        mc_command,
        capture_output=True,
    )

    # Check that the correct messages were sent
    offline_transport.send.assert_any_call(
        destination="ctffind",
        message={
            "parameters": {
                "ctf": "ctf",
                "input_image": motioncorr_test_message["parameters"]["mrc_out"],
                "output_image": f"{tmp_path}/CtfFind/job003/Movies/sample_motion_corrected.ctf",
                "mc_uuid": motioncorr_test_message["parameters"]["mc_uuid"],
                "picker_uuid": motioncorr_test_message["parameters"]["picker_uuid"],
                "relion_options": output_relion_options,
                "amplitude_contrast": output_relion_options["ampl_contrast"],
                "experiment_type": "tomography",
                "pixel_size": motioncorr_test_message["parameters"]["pixel_size"],
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="ispyb_connector",
        message={
            "parameters": {
                "first_frame": 1,
                "last_frame": 2,
                "total_motion": total_motion,
                "average_motion_per_frame": average_motion_per_frame,
                "drift_plot_full_path": f"{tmp_path}/MotionCorr/job002/Movies/sample_drift_plot.json",
                "micrograph_snapshot_full_path": f"{tmp_path}/MotionCorr/job002/Movies/sample_motion_corrected.jpeg",
                "micrograph_full_path": motioncorr_test_message["parameters"][
                    "mrc_out"
                ],
                "patches_used_x": motioncorr_test_message["parameters"]["patch_sizes"][
                    "x"
                ],
                "patches_used_y": motioncorr_test_message["parameters"]["patch_sizes"][
                    "y"
                ],
                "buffer_store": motioncorr_test_message["parameters"]["mc_uuid"],
                "dose_per_frame": motioncorr_test_message["parameters"][
                    "dose_per_frame"
                ],
                "ispyb_command": "buffer",
                "buffer_command": {"ispyb_command": "insert_motion_correction"},
            },
            "content": {"dummy": "dummy"},
        },
    )
    offline_transport.send.assert_any_call(
        destination="murfey_feedback",
        message={
            "register": "motion_corrected",
            "movie": str(movie),
            "mrc_out": motioncorr_test_message["parameters"]["mrc_out"],
            "movie_id": motioncorr_test_message["parameters"]["movie_id"],
        },
    )
    offline_transport.send.assert_any_call(
        destination="images",
        message={
            "image_command": "mrc_to_jpeg",
            "file": motioncorr_test_message["parameters"]["mrc_out"],
        },
    )
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "experiment_type": "tomography",
                "job_type": "relion.importtomo",
                "input_file": f"{movie}:{tmp_path}/Movies/*.mdoc",
                "output_file": f"{tmp_path}/Import/job001/Movies/sample.tiff",
                "relion_options": output_relion_options,
                "command": "",
                "stdout": "",
                "stderr": "",
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "experiment_type": "tomography",
                "job_type": "relion.motioncorr.motioncor2",
                "input_file": f"{tmp_path}/Import/job001/Movies/sample.tiff",
                "output_file": motioncorr_test_message["parameters"]["mrc_out"],
                "relion_options": output_relion_options,
                "command": " ".join(mc_command),
                "stdout": "stdout",
                "stderr": "stderr",
                "results": {
                    "total_motion": total_motion,
                    "early_motion": early_motion,
                    "late_motion": late_motion,
                },
            },
            "content": "dummy",
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.motioncorr.subprocess.run")
def test_motioncor_relion_service_tomo(mock_subprocess, offline_transport, tmp_path):
    """
    Send a test message to MotionCorr for tomography using Relion's own version
    This should call the mock subprocess then send messages on to
    the murfey_feedback, ispyb_connector and images services.
    It also creates the ctffind job.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    (tmp_path / "gain.mrc").touch()
    movie = Path(f"{tmp_path}/Movies/sample.tiff")
    movie.parent.mkdir(parents=True)
    movie.touch()

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    motioncorr_test_message = {
        "parameters": {
            "experiment_type": "tomography",
            "pixel_size": 0.1,
            "autopick": {"autopick": "autopick"},
            "ctf": {"ctf": "ctf"},
            "movie": str(movie),
            "mrc_out": f"{tmp_path}/MotionCorr/job002/Movies/sample_motion_corrected.mrc",
            "patch_sizes": {"x": 5, "y": 5},
            "gpu": 0,
            "threads": 1,
            "gain_ref": f"{tmp_path}/gain.mrc",
            "mc_uuid": 0,
            "picker_uuid": 0,
            "rot_gain": None,
            "flip_gain": None,
            "dark": None,
            "use_gpus": None,
            "sum_range": None,
            "iter": None,
            "tol": None,
            "throw": None,
            "trunc": None,
            "fm_ref": 1,
            "voltage": None,
            "dose_per_frame": 1,
            "use_motioncor2": False,
            "fm_int_file": None,
            "init_dose": None,
            "mag": None,
            "motion_corr_binning": None,
            "serial": None,
            "in_suffix": None,
            "eer_sampling": None,
            "out_stack": None,
            "bft": None,
            "group": None,
            "detect_file": None,
            "arc_dir": None,
            "in_fm_motion": None,
            "split_sum": None,
            "movie_id": 1,
            "relion_options": {
                "frame_count": 5,
                "tilt_axis_angle": 83.0,
                "defocus": -2.0,
                "invert_hand": 1,
            },
        },
        "content": "dummy",
    }

    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["pixel_size"] = motioncorr_test_message["parameters"][
        "pixel_size"
    ]
    output_relion_options["dose_per_frame"] = motioncorr_test_message["parameters"][
        "dose_per_frame"
    ]
    output_relion_options["gain_ref"] = motioncorr_test_message["parameters"][
        "gain_ref"
    ]
    output_relion_options.update(
        motioncorr_test_message["parameters"]["relion_options"]
    )
    output_relion_options["eer_grouping"] = 0

    # Set up the mock service
    service = motioncorr.MotionCorr()
    service.transport = offline_transport
    service.start()

    # Work out the expected shifts
    service.x_shift_list = [-3.0, 3.0]
    service.y_shift_list = [4.0, -4.0]
    service.each_total_motion = [5.0, 5.0]
    total_motion = 10.0
    early_motion = 10.0
    late_motion = 0.0
    average_motion_per_frame = 5

    # Touch expected output file
    (tmp_path / "MotionCorr/job002/Movies").mkdir(parents=True, exist_ok=True)
    with open(
        tmp_path / "MotionCorr/job002/Movies/sample_motion_corrected.star", "w"
    ) as relion_output:
        relion_output.write(
            "data_global_shift\nloop_\n_rlnMicrographShiftX\n_rlnMicrographShiftY\n"
        )

    # Send a message to the service
    service.motion_correction(None, header=header, message=motioncorr_test_message)

    mc_command = [
        "relion_motion_correction",
        "--use_own",
        "--in_movie",
        str(movie),
        "--out_mic",
        motioncorr_test_message["parameters"]["mrc_out"],
        "--angpix",
        str(motioncorr_test_message["parameters"]["pixel_size"]),
        "--dose_per_frame",
        "1.0",
        "--patch_x",
        "5",
        "--patch_y",
        "5",
        "--j",
        "1",
        "--gainref",
        motioncorr_test_message["parameters"]["gain_ref"],
        "--dose_weighting",
        "--i",
        "dummy",
    ]

    assert mock_subprocess.call_count == 4
    mock_subprocess.assert_called_with(
        mc_command,
        capture_output=True,
    )

    # Check that the correct messages were sent
    offline_transport.send.assert_any_call(
        destination="ctffind",
        message={
            "parameters": {
                "ctf": "ctf",
                "input_image": motioncorr_test_message["parameters"]["mrc_out"],
                "output_image": f"{tmp_path}/CtfFind/job003/Movies/sample_motion_corrected.ctf",
                "mc_uuid": motioncorr_test_message["parameters"]["mc_uuid"],
                "picker_uuid": motioncorr_test_message["parameters"]["picker_uuid"],
                "relion_options": output_relion_options,
                "amplitude_contrast": output_relion_options["ampl_contrast"],
                "experiment_type": "tomography",
                "pixel_size": motioncorr_test_message["parameters"]["pixel_size"],
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="ispyb_connector",
        message={
            "parameters": {
                "first_frame": 1,
                "last_frame": 2,
                "total_motion": total_motion,
                "average_motion_per_frame": average_motion_per_frame,
                "drift_plot_full_path": f"{tmp_path}/MotionCorr/job002/Movies/sample_drift_plot.json",
                "micrograph_snapshot_full_path": f"{tmp_path}/MotionCorr/job002/Movies/sample_motion_corrected.jpeg",
                "micrograph_full_path": motioncorr_test_message["parameters"][
                    "mrc_out"
                ],
                "patches_used_x": motioncorr_test_message["parameters"]["patch_sizes"][
                    "x"
                ],
                "patches_used_y": motioncorr_test_message["parameters"]["patch_sizes"][
                    "y"
                ],
                "buffer_store": motioncorr_test_message["parameters"]["mc_uuid"],
                "dose_per_frame": motioncorr_test_message["parameters"][
                    "dose_per_frame"
                ],
                "ispyb_command": "buffer",
                "buffer_command": {"ispyb_command": "insert_motion_correction"},
            },
            "content": {"dummy": "dummy"},
        },
    )
    offline_transport.send.assert_any_call(
        destination="murfey_feedback",
        message={
            "register": "motion_corrected",
            "movie": str(movie),
            "mrc_out": motioncorr_test_message["parameters"]["mrc_out"],
            "movie_id": motioncorr_test_message["parameters"]["movie_id"],
        },
    )
    offline_transport.send.assert_any_call(
        destination="images",
        message={
            "image_command": "mrc_to_jpeg",
            "file": motioncorr_test_message["parameters"]["mrc_out"],
        },
    )
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "experiment_type": "tomography",
                "job_type": "relion.importtomo",
                "input_file": f"{movie}:{tmp_path}/Movies/*.mdoc",
                "output_file": f"{tmp_path}/Import/job001/Movies/sample.tiff",
                "relion_options": output_relion_options,
                "command": "",
                "stdout": "",
                "stderr": "",
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "experiment_type": "tomography",
                "job_type": "relion.motioncorr.own",
                "input_file": f"{tmp_path}/Import/job001/Movies/sample.tiff",
                "output_file": motioncorr_test_message["parameters"]["mrc_out"],
                "relion_options": output_relion_options,
                "command": " ".join(mc_command),
                "stdout": "stdout",
                "stderr": "stderr",
                "results": {
                    "total_motion": total_motion,
                    "early_motion": early_motion,
                    "late_motion": late_motion,
                },
            },
            "content": "dummy",
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.util.slurm_submission.subprocess.run")
def test_motioncor2_slurm_service_spa(mock_subprocess, offline_transport, tmp_path):
    """
    Send a test message to MotionCorr for SPA using MotionCor2 via slurm.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = (
        '{"job_id": "1", "jobs": [{"job_state": ["COMPLETED"]}]}'.encode("ascii")
    )
    mock_subprocess().stderr = "stderr".encode("ascii")

    movie = Path(f"{tmp_path}/Movies/sample.tiff")
    movie.parent.mkdir(parents=True)
    movie.touch()

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    motioncorr_test_message = {
        "parameters": {
            "movie": str(movie),
            "mrc_out": f"{tmp_path}/MotionCorr/job002/Movies/sample.mrc",
            "experiment_type": "spa",
            "pixel_size": 0.1,
            "dose_per_frame": 1,
            "use_motioncor2": True,
            "submit_to_slurm": True,
            "patch_sizes": {"x": 5, "y": 5},
            "movie_id": 1,
            "mc_uuid": 0,
            "picker_uuid": 0,
            "relion_options": {},
        },
        "content": "dummy",
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["pixel_size"] = motioncorr_test_message["parameters"][
        "pixel_size"
    ]
    output_relion_options["dose_per_frame"] = motioncorr_test_message["parameters"][
        "dose_per_frame"
    ]
    output_relion_options["eer_grouping"] = 0

    # Set up the mock service
    service = motioncorr.MotionCorr()
    service.transport = offline_transport
    service.start()

    # Work out the expected shifts
    service.x_shift_list = [-3.0, 3.0]
    service.y_shift_list = [4.0, -4.0]
    service.each_total_motion = [5.0, 5.0]
    total_motion = 10.0
    early_motion = 10.0
    late_motion = 0.0

    # Construct the file which contains rest api submission information
    os.environ["MOTIONCOR2_SIF"] = "MotionCor2_SIF"
    os.environ["SLURM_RESTAPI_CONFIG"] = str(tmp_path / "restapi.txt")
    with open(tmp_path / "restapi.txt", "w") as restapi_config:
        restapi_config.write(
            "user: user\n"
            "user_home: /home\n"
            f"user_token: {tmp_path}/token.txt\n"
            "required_directories: [directory1, directory2]\n"
            "partition: partition\n"
            "partition_preference: preference\n"
            "cluster: cluster\n"
            "url: /url/of/slurm/restapi\n"
            "api_version: v0.0.40\n"
        )
    with open(tmp_path / "token.txt", "w") as token:
        token.write("token_key")

    # Touch the expected output files
    (tmp_path / "MotionCorr/job002/Movies").mkdir(parents=True)
    (tmp_path / "MotionCorr/job002/Movies/sample.mrc.out").touch()
    (tmp_path / "MotionCorr/job002/Movies/sample.mrc.err").touch()

    # Send a message to the service
    service.motion_correction(None, header=header, message=motioncorr_test_message)

    # Get the expected motion correction command
    mc_command = [
        "MotionCor2",
        "-InTiff",
        str(movie),
        "-OutMrc",
        motioncorr_test_message["parameters"]["mrc_out"],
        "-PixSize",
        str(motioncorr_test_message["parameters"]["pixel_size"]),
        "-FmDose",
        "1.0",
        "-Patch",
        "5 5",
        "-Gpu",
        "0",
        "-FmRef",
        "0",
    ]

    # Check the slurm commands were run
    slurm_submit_command = (
        f'curl -H "X-SLURM-USER-NAME:user" -H "X-SLURM-USER-TOKEN:token_key" '
        '-H "Content-Type: application/json" -X POST '
        "/url/of/slurm/restapi/slurm/v0.0.40/job/submit "
        f"-d @{tmp_path}/MotionCorr/job002/Movies/sample.mrc.json"
    )
    slurm_status_command = (
        'curl -H "X-SLURM-USER-NAME:user" -H "X-SLURM-USER-TOKEN:token_key" '
        '-H "Content-Type: application/json" -X GET '
        "/url/of/slurm/restapi/slurm/v0.0.40/job/1"
    )
    assert mock_subprocess.call_count == 5
    mock_subprocess.assert_any_call(
        slurm_submit_command, capture_output=True, shell=True
    )
    mock_subprocess.assert_any_call(
        slurm_status_command, capture_output=True, shell=True
    )

    # Just check the node creator send to make sure all ran correctly
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "experiment_type": "spa",
                "job_type": "relion.motioncorr.motioncor2",
                "input_file": f"{tmp_path}/Import/job001/Movies/sample.tiff",
                "output_file": motioncorr_test_message["parameters"]["mrc_out"],
                "relion_options": output_relion_options,
                "command": " ".join(mc_command),
                "stdout": "",
                "stderr": "",
                "results": {
                    "total_motion": total_motion,
                    "early_motion": early_motion,
                    "late_motion": late_motion,
                },
            },
            "content": "dummy",
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.motioncorr.slurm_submission")
def test_motioncor2_slurm_parameters(mock_slurm, offline_transport, tmp_path):
    """
    Test the parameters used for slurm job submission when using MotionCor2
    """
    mock_slurm().returncode = 0
    mock_slurm().stdout = (
        '{"job_id": "1", "jobs": [{"job_state": ["COMPLETED"]}]}'.encode("ascii")
    )
    mock_slurm().stderr = "stderr".encode("ascii")

    movie = Path(f"{tmp_path}/Movies/sample.tiff")
    movie.parent.mkdir(parents=True)
    movie.touch()

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    motioncorr_test_message = {
        "parameters": {
            "movie": str(movie),
            "mrc_out": f"{tmp_path}/MotionCorr/job002/Movies/sample.mrc",
            "experiment_type": "spa",
            "pixel_size": 0.1,
            "dose_per_frame": 1,
            "use_motioncor2": True,
            "submit_to_slurm": True,
            "patch_sizes": {"x": 5, "y": 5},
            "movie_id": 1,
            "mc_uuid": 0,
            "picker_uuid": 0,
            "relion_options": {},
        },
        "content": "dummy",
    }

    # Set up the mock service
    service = motioncorr.MotionCorr()
    service.transport = offline_transport
    service.start()

    # Work out the expected shifts
    service.x_shift_list = [-3.0, 3.0]
    service.y_shift_list = [4.0, -4.0]
    service.each_total_motion = [5.0, 5.0]

    # Construct the file which contains rest api submission information
    os.environ["MOTIONCOR2_SIF"] = "MotionCor2_SIF"

    # Touch the expected output files
    (tmp_path / "MotionCorr/job002/Movies").mkdir(parents=True)
    (tmp_path / "MotionCorr/job002/Movies/sample.mrc.out").touch()
    (tmp_path / "MotionCorr/job002/Movies/sample.mrc.err").touch()
    (tmp_path / "MotionCorr/job002/Movies/sample.mrc.json").touch()

    # Send a message to the service
    service.motion_correction(None, header=header, message=motioncorr_test_message)

    # Get the expected motion correction command
    mc_command = [
        "MotionCor2",
        "-InTiff",
        str(movie),
        "-OutMrc",
        motioncorr_test_message["parameters"]["mrc_out"],
        "-PixSize",
        str(motioncorr_test_message["parameters"]["pixel_size"]),
        "-FmDose",
        "1.0",
        "-Patch",
        "5 5",
        "-Gpu",
        "0",
        "-FmRef",
        "0",
    ]

    # Check the slurm submission command
    assert mock_slurm.call_count == 4
    mock_slurm.assert_called_with(
        log=service.log,
        job_name="MotionCor2",
        command=mc_command,
        project_dir=tmp_path / "MotionCorr/job002/Movies/",
        output_file=tmp_path / "MotionCorr/job002/Movies/sample.mrc",
        cpus=1,
        use_gpu=True,
        use_singularity=True,
        cif_name="MotionCor2_SIF",
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.motioncorr.slurm_submission")
def test_motioncor_relion_slurm_parameters(mock_slurm, offline_transport, tmp_path):
    """
    Test the parameters used for slurm job submission when using Relion's own
    """
    mock_slurm().returncode = 0
    mock_slurm().stdout = (
        '{"job_id": "1", "jobs": [{"job_state": ["COMPLETED"]}]}'.encode("ascii")
    )
    mock_slurm().stderr = "stderr".encode("ascii")

    movie = Path(f"{tmp_path}/Movies/sample.tiff")
    movie.parent.mkdir(parents=True)
    movie.touch()

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    motioncorr_test_message = {
        "parameters": {
            "movie": str(movie),
            "mrc_out": f"{tmp_path}/MotionCorr/job002/Movies/sample.mrc",
            "experiment_type": "spa",
            "pixel_size": 0.1,
            "dose_per_frame": 1,
            "use_motioncor2": False,
            "submit_to_slurm": True,
            "patch_sizes": {"x": 5, "y": 5},
            "movie_id": 1,
            "mc_uuid": 0,
            "picker_uuid": 0,
            "relion_options": {},
        },
        "content": "dummy",
    }

    # Set up the mock service
    service = motioncorr.MotionCorr()
    service.transport = offline_transport
    service.start()

    # Work out the expected shifts
    service.x_shift_list = [-3.0, 3.0]
    service.y_shift_list = [4.0, -4.0]
    service.each_total_motion = [5.0, 5.0]

    # Construct the file which contains rest api submission information
    os.environ["MOTIONCOR2_SIF"] = "MotionCor2_SIF"

    # Touch the expected output files
    (tmp_path / "MotionCorr/job002/Movies").mkdir(parents=True)
    (tmp_path / "MotionCorr/job002/Movies/sample.mrc.out").touch()
    (tmp_path / "MotionCorr/job002/Movies/sample.mrc.err").touch()
    (tmp_path / "MotionCorr/job002/Movies/sample.mrc.json").touch()

    # Send a message to the service
    service.motion_correction(None, header=header, message=motioncorr_test_message)

    # Get the expected motion correction command
    mc_command = [
        "relion_motion_correction",
        "--use_own",
        "--in_movie",
        str(movie),
        "--out_mic",
        motioncorr_test_message["parameters"]["mrc_out"],
        "--angpix",
        str(motioncorr_test_message["parameters"]["pixel_size"]),
        "--dose_per_frame",
        "1.0",
        "--patch_x",
        "5",
        "--patch_y",
        "5",
        "--j",
        "1",
        "--dose_weighting",
        "--i",
        "dummy",
    ]

    # Check the slurm submission command
    assert mock_slurm.call_count == 4
    mock_slurm.assert_called_with(
        log=service.log,
        job_name="RelionMotionCorr",
        command=mc_command,
        project_dir=tmp_path / "MotionCorr/job002/Movies/",
        output_file=tmp_path / "MotionCorr/job002/Movies/sample.mrc",
        cpus=4,
        use_gpu=False,
        use_singularity=False,
        script_extras="module load EM/relion/motioncorr",
    )


def test_parse_motioncor2_output(offline_transport):
    """
    Send test lines to the output parser for MotionCor2
    to check the shift values are being read in
    """
    service = motioncorr.MotionCorr()
    service.transport = offline_transport
    service.start()

    # MotionCor2 v1.4.0 case
    motioncorr.MotionCorr.parse_mc2_stdout(
        service, "...... Frame (  1) shift:    -1.0      2.0"
    )
    motioncorr.MotionCorr.parse_mc2_stdout(
        service, "...... Frame (  2) shift:    1.0      -2.0"
    )
    assert service.x_shift_list == [-1.0, 1.0]
    assert service.y_shift_list == [2.0, -2.0]

    # MotionCor2 v1.6.3 case
    service.x_shift_list = []
    service.y_shift_list = []
    motioncorr.MotionCorr.parse_mc2_stdout(
        service, "Frame   x Shift   y Shift\n1    -3.0      4.0\n2    3.0      -4.0"
    )
    assert service.x_shift_list == [-3.0, 3.0]
    assert service.y_shift_list == [4.0, -4.0]


def test_parse_motioncor2_slurm_output(offline_transport, tmp_path):
    """
    Send test lines to the output parser
    to check the shift values are being read in
    """
    service = motioncorr.MotionCorr()
    service.transport = offline_transport
    service.start()

    with open(tmp_path / "mc_output.txt", "w") as mc_output:
        mc_output.write(
            "...... Frame (  1) shift:    -3.0      4.0\n"
            "...... Frame (  2) shift:    3.0      -4.0\n"
        )

    motioncorr.MotionCorr.parse_mc2_slurm_output(service, tmp_path / "mc_output.txt")
    assert service.x_shift_list == [-3.0, 3.0]
    assert service.y_shift_list == [4.0, -4.0]


def test_parse_relion_output(offline_transport, tmp_path):
    """
    Send a test file to the output parser for Relion's motion correction
    to check the shift values are being read in
    """
    service = motioncorr.MotionCorr()
    service.transport = offline_transport
    service.start()

    with open(tmp_path / "mc_output.star", "w") as mc_output:
        mc_output.write(
            "data_global_shift\nloop_\n_rlnMicrographShiftX\n_rlnMicrographShiftY\n"
            "-5.0 6.0\n"
            "5.0 -6.0\n"
        )

    motioncorr.MotionCorr.parse_relion_mc_output(service, tmp_path / "mc_output.star")
    assert service.x_shift_list == [-5.0, 5.0]
    assert service.y_shift_list == [6.0, -6.0]
