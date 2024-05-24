from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest
import zocalo.configuration
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import motioncorr_slurm
from cryoemservices.util.relion_service_options import RelionServiceOptions


@pytest.fixture
def mock_zocalo_configuration(tmp_path):
    mock_zc = mock.MagicMock(zocalo.configuration.Configuration)
    mock_zc.storage = {
        "zocalo.recipe_directory": tmp_path,
    }
    return mock_zc


@pytest.fixture
def mock_environment(mock_zocalo_configuration):
    return {"config": mock_zocalo_configuration}


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.util.slurm_submission.subprocess.run")
def test_motioncor2_service_spa(
    mock_subprocess, mock_environment, offline_transport, tmp_path
):
    """
    Send a test message to MotionCorr for SPA using MotionCor2
    This should call the mock subprocess then send messages on to
    the ispyb_connector and images services.
    It also creates the next jobs (ctffind and two icebreaker jobs)
    and the node_creator is called for both import and motion correction.
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
    service = motioncorr_slurm.MotionCorrSlurm(environment=mock_environment)
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
                "input_image": motioncorr_test_message["parameters"]["mrc_out"],
                "mc_uuid": motioncorr_test_message["parameters"]["mc_uuid"],
                "picker_uuid": motioncorr_test_message["parameters"]["picker_uuid"],
                "relion_options": output_relion_options,
                "amplitude_contrast": output_relion_options["ampl_contrast"],
                "experiment_type": "spa",
                "output_image": f"{tmp_path}/CtfFind/job006/Movies/sample.ctf",
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
@mock.patch("cryoemservices.util.slurm_submission.subprocess.run")
def test_motioncor2_service_tomo(
    mock_subprocess, mock_environment, offline_transport, tmp_path
):
    """
    Send a test message to MotionCorr for tomography using MotionCor2
    This should call the mock subprocess then send messages on to
    the murfey_feedback, ispyb_connector and images services.
    It also creates the ctffind job.
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
            "mrc_out": f"{tmp_path}/MotionCorr/job002/Movies/sample_motion_corrected.mrc",
            "experiment_type": "tomography",
            "pixel_size": 0.1,
            "dose_per_frame": 1,
            "use_motioncor2": True,
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
    service = motioncorr_slurm.MotionCorrSlurm(environment=mock_environment)
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
    (tmp_path / "MotionCorr/job002/Movies/sample_motion_corrected.mrc.out").touch()
    (tmp_path / "MotionCorr/job002/Movies/sample_motion_corrected.mrc.err").touch()

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
        f"-d @{tmp_path}/MotionCorr/job002/Movies/sample_motion_corrected.mrc.json"
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

    # Check that the correct messages were sent
    offline_transport.send.assert_any_call(
        destination="ctffind",
        message={
            "parameters": {
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
                "job_type": "relion.import.tilt_series",
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


def test_parse_motioncor2_output(mock_environment, offline_transport, tmp_path):
    """
    Send test lines to the output parser
    to check the shift values are being read in
    """
    service = motioncorr_slurm.MotionCorrSlurm(environment=mock_environment)
    service.transport = offline_transport
    service.start()

    with open(tmp_path / "mc_output.txt", "w") as mc_output:
        mc_output.write(
            "...... Frame (  1) shift:    -3.0      4.0\n"
            "...... Frame (  2) shift:    3.0      -4.0\n"
        )

    motioncorr_slurm.MotionCorrSlurm.parse_mc_slurm_output(
        service, tmp_path / "mc_output.txt"
    )
    assert service.x_shift_list == [-3.0, 3.0]
    assert service.y_shift_list == [4.0, -4.0]
