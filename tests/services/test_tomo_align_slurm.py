from __future__ import annotations

import os
import sys
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import tomo_align_slurm


def cluster_submission_configuration(tmp_path):
    # Create a config file
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as cf:
        cf.write("rabbitmq_credentials: rmq_creds\n")
        cf.write(f"recipe_directory: {tmp_path}/recipes\n")
        cf.write("slurm_credentials:\n")
        cf.write(f"  default: {tmp_path}/slurm_credentials.yaml\n")
    os.environ["USER"] = "user"

    # Create dummy slurm credentials files
    with open(tmp_path / "slurm_credentials.yaml", "w") as slurm_creds:
        slurm_creds.write(
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


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.tomo_align_slurm.subprocess.run")
@mock.patch("cryoemservices.services.tomo_align.px.scatter")
@mock.patch("cryoemservices.services.tomo_align.mrcfile")
@mock.patch("cryoemservices.services.tomo_align_slurm.transfer_files")
@mock.patch("cryoemservices.services.tomo_align_slurm.retrieve_files")
def test_tomo_align_slurm_service(
    mock_retrieve,
    mock_transfer,
    mock_mrcfile,
    mock_plotly,
    mock_subprocess,
    offline_transport,
    tmp_path,
):
    """
    Send a test message to TomoAlign for the slurm submission version
    This should call the mock subprocess then send messages on to
    the denoising, ispyb_connector and images services.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = (
        '{"job_id": "1", "jobs": [{"job_state": ["COMPLETED"]}]}'.encode("ascii")
    )
    mock_subprocess().stderr = "stderr".encode("ascii")

    mock_mrcfile.open().__enter__().header = {"nx": 2000, "ny": 3000}

    mock_transfer.return_value = ["test_stack.mrc", "angles.txt"]

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    tomo_align_test_message = {
        "stack_file": f"{tmp_path}/Tomograms/job006/tomograms/test_stack.mrc",
        "path_pattern": None,
        "input_file_list": str(
            [[f"{tmp_path}/MotionCorr/job002/Movies/input_file_1.mrc", "1.00"]]
        ),
        "vol_z": 1200,
        "align": None,
        "out_bin": 4,
        "tilt_axis": 85,
        "tilt_cor": 1,
        "flip_int": None,
        "flip_vol": 1,
        "wbp": None,
        "roi_file": [],
        "patch": None,
        "kv": None,
        "dose_per_frame": None,
        "frame_count": None,
        "align_file": None,
        "angle_file": f"{tmp_path}/angles.file",
        "align_z": None,
        "pixel_size": 1e-10,
        "refine_flag": 1,
        "out_imod": 1,
        "out_imod_xf": None,
        "dark_tol": None,
        "manual_tilt_offset": None,
        "tomogram_uuid": 0,
        "relion_options": {},
    }

    # Construct the file which contains rest api submission information
    os.environ["ARETOMO2_EXECUTABLE"] = "slurm_AreTomo"
    os.environ["EXTRA_LIBRARIES"] = "/lib/aretomo"
    cluster_submission_configuration(tmp_path)

    # Set up the mock service
    service = tomo_align_slurm.TomoAlignSlurm(
        environment={
            "config": f"{tmp_path}/config.yaml",
            "slurm_cluster": "default",
            "queue": "",
        }
    )
    service.transport = offline_transport
    service.start()

    service.rot_centre_z_list = [1.1, 2.1]
    service.tilt_offset = 1.1
    service.alignment_quality = 0.5

    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo_Imod").mkdir(
        parents=True
    )
    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo_Imod/tilt.com").touch()
    with open(tmp_path / "Tomograms/job006/tomograms/test_stack.aln", "w") as aln_file:
        aln_file.write("dummy 0 1000 1.2 2.3 5 6 7 8 4.5")

    # Touch the expected output files
    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc").touch()
    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc.out").touch()
    (tmp_path / "Tomograms/job006/tomograms/test_stack_aretomo.mrc.err").touch()

    # Send a message to the service
    service.tomo_align(None, header=header, message=tomo_align_test_message)

    # Check the newstack was run
    mock_subprocess.assert_any_call(
        [
            "newstack",
            "-fileinlist",
            f"{tmp_path}/Tomograms/job006/tomograms/test_stack_newstack.txt",
            "-output",
            f"{tmp_path}/Tomograms/job006/tomograms/test_stack.mrc",
            "-quiet",
        ]
    )

    # Check the angle file
    assert (
        tmp_path / "Tomograms/job006/tomograms/test_stack_tilt_angles.txt"
    ).is_file()
    with open(
        tmp_path / "Tomograms/job006/tomograms/test_stack_tilt_angles.txt", "r"
    ) as angfile:
        angles_data = angfile.read()
    assert angles_data == "1.00  0\n"

    # Check the slurm commands were run
    slurm_submit_command = (
        f'curl -H "X-SLURM-USER-NAME:user" -H "X-SLURM-USER-TOKEN:token_key" '
        '-H "Content-Type: application/json" -X POST '
        "/url/of/slurm/restapi/slurm/v0.0.40/job/submit "
        f"-d @{tmp_path}/Tomograms/job006/tomograms/test_stack_aretomo.mrc.json"
    )
    slurm_status_command = (
        'curl -H "X-SLURM-USER-NAME:user" -H "X-SLURM-USER-TOKEN:token_key" '
        '-H "Content-Type: application/json" -X GET '
        "/url/of/slurm/restapi/slurm/v0.0.40/job/1"
    )
    mock_subprocess.assert_any_call(
        slurm_submit_command, capture_output=True, shell=True
    )
    mock_subprocess.assert_any_call(
        slurm_status_command, capture_output=True, shell=True
    )

    # Check file transfer and retrieval
    assert mock_transfer.call_count == 1
    mock_transfer.assert_any_call(
        [
            tmp_path / "Tomograms/job006/tomograms/test_stack.mrc",
            tmp_path / "Tomograms/job006/tomograms/test_stack_tilt_angles.txt",
        ]
    )
    assert mock_retrieve.call_count == 1
    mock_retrieve.assert_any_call(
        job_directory=tmp_path / "Tomograms/job006/tomograms",
        files_to_skip=[
            tmp_path / "Tomograms/job006/tomograms/test_stack.mrc",
            tmp_path / "Tomograms/job006/tomograms/test_stack_tilt_angles.txt",
        ],
        basepath="test_stack",
    )
    assert mock_plotly.call_count == 1
    assert mock_subprocess.call_count == 6


def test_parse_tomo_align_output(offline_transport, tmp_path):
    """
    Send test lines to the output parser
    to check the rotations and offsets are being read in
    """
    service = tomo_align_slurm.TomoAlignSlurm(environment={"queue": ""})
    service.transport = offline_transport
    service.start()

    with open(tmp_path / "tomo_output.txt", "w") as tomo_output:
        tomo_output.write(
            "Rot center Z 100.0 200.0 300.0\n"
            "Rot center Z 150.0 250.0 350.0\n"
            "Tilt offset 1.0, CC: 0.5\n"
            "Best tilt axis:   57, Score:   0.07568\n"
        )

    tomo_align_slurm.TomoAlignSlurm.parse_tomo_output(
        service, str(tmp_path / "tomo_output.txt")
    )
    assert service.rot_centre_z_list == ["300.0", "350.0"]
    assert service.tilt_offset == 1.0
    assert service.alignment_quality == 0.07568


def test_transfer_files(tmp_path):
    """Test that existing files can be transferred, and non-existant files are not"""
    (tmp_path / "to_transfer").mkdir()
    (tmp_path / "to_transfer/file_exists").touch()
    transferred_files = tomo_align_slurm.transfer_files(
        [
            tmp_path / "to_transfer/file_exists",
            tmp_path / "to_transfer/file_does_not_exist",
        ],
        local_base=f"{tmp_path}/to_transfer",
        remote_base=f"{tmp_path}/destination",
    )

    assert transferred_files == [tmp_path / "to_transfer/file_exists"]
    assert (tmp_path / "destination/file_exists").is_file()
    assert not (tmp_path / "destination/file_does_not_exist").exists()


def test_retrieve_files(tmp_path):
    (tmp_path / "remote_system/job_dir/file_imod_dir").mkdir(parents=True)
    (tmp_path / "remote_system/job_dir/file_to_retrieve").touch()
    (tmp_path / "remote_system/job_dir/file_to_ignore").touch()
    (tmp_path / "remote_system/job_dir/different_basepath").touch()
    (tmp_path / "remote_system/job_dir/file_imod_dir/imod_file").touch()

    tomo_align_slurm.retrieve_files(
        job_directory=tmp_path / "local_system/job_dir",
        files_to_skip=[
            tmp_path / "local_system/job_dir/file_to_ignore",
            tmp_path / "local_system/job_dir/file_not_exists",
        ],
        basepath="file",
        local_base=f"{tmp_path}/local_system",
        remote_base=f"{tmp_path}/remote_system",
    )

    # File which should have been copied and removed
    assert (tmp_path / "local_system/job_dir/file_to_retrieve").is_file()
    assert not (tmp_path / "remote_system/job_dir/file_to_retrieve").exists()

    # File should have been ignored but removed anyway
    assert not (tmp_path / "local_system/job_dir/file_to_ignore").exists()
    assert not (tmp_path / "remote_system/job_dir/file_to_ignore").exists()

    # File in subfolder which should have been copied and removed
    assert (tmp_path / "local_system/job_dir/file_imod_dir/imod_file").is_file()
    assert not (tmp_path / "remote_system/job_dir/file_imod_dir").exists()

    # File with different basepath should have been left where it is
    assert not (tmp_path / "local_system/job_dir/different_basepath").exists()
    assert (tmp_path / "remote_system/job_dir/different_basepath").is_file()

    # File which doesn't exist
    assert not (tmp_path / "local_system/job_dir/file_not_exists").exists()
    assert not (tmp_path / "remote_system/job_dir/file_not_exists").exists()
