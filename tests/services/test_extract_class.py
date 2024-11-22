from __future__ import annotations

import os
import sys
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import extract_class
from cryoemservices.util.relion_service_options import RelionServiceOptions


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
@mock.patch("cryoemservices.util.slurm_submission.subprocess.run")
def test_extract_class_service(mock_subprocess, offline_transport, tmp_path):
    """
    Send a test message to the class extraction service
    This should run particle selection and launch re-extraction jobs with slurm
    then send messages on to refinement and the node_creator
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = (
        '{"job_id": "1", "jobs": [{"job_state": ["COMPLETED"]}]}'.encode("ascii")
    )
    mock_subprocess().stderr = "stderr".encode("ascii")

    # Create the expected input files
    (tmp_path / "CtfFind/job003").mkdir(parents=True)
    (tmp_path / "CtfFind/job003/micrographs_ctf.star").touch()
    (tmp_path / "Class3D/job010").mkdir(parents=True)
    with open(tmp_path / "Class3D/job010/run_it025_data.star", "w") as f:
        f.write(
            "data_particles\n\nloop_\n #1\n_rlnCoordinateY #2\n_rlnImageName #3\n"
            "_rlnMicrographName #4\n_rlnOpticsGroup #5\n_rlnClassNumber #6\n"
        )
        for i in range(5):
            f.write(f"\n1.0 2.0 {i}@Extract.mrcs sample.mrc 1 {i}")
    with open(tmp_path / "Class3D/job010/run_it025_model.star", "w") as f:
        f.write("_rlnPixelSize 4.25\n")
    with open(tmp_path / "Class3D/job010/run_it025_optimiser.star", "w") as f:
        f.write("_rlnParticleDiameter 100\n")

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    extract_class_test_message = {
        "parameters": {
            "extraction_executable": "EM/cryoemservices.reextract",
            "class3d_dir": str(tmp_path / "Class3D/job010"),
            "refine_job_dir": str(tmp_path / "Refine3D/job013"),
            "refine_class_nr": 1,
            "original_pixel_size": 0.5,
            "boxsize": 200,
            "nr_iter_3d": 25,
            "bg_radius": -1,
            "downscale_factor": 2,
            "downscale": True,
            "normalise": True,
            "invert_contrast": True,
            "relion_options": {},
        },
        "content": "dummy",
    }
    output_relion_options = RelionServiceOptions()
    output_relion_options.downscale = True
    output_relion_options.pixel_size_downscaled = 1.0
    output_relion_options = dict(output_relion_options)
    output_relion_options.update(
        extract_class_test_message["parameters"]["relion_options"]
    )
    output_relion_options["boxsize"] = 228
    output_relion_options["small_boxsize"] = 114

    # Construct the file which contains rest api submission information
    cluster_submission_configuration(tmp_path)

    # Touch the expected output files
    (tmp_path / "Extract/job012").mkdir(parents=True)
    (tmp_path / "Extract/job012/slurm_run.out").touch()
    (tmp_path / "Extract/job012/slurm_run.err").touch()

    # Set up the mock service and call it
    service = extract_class.ExtractClass(
        environment={"config": f"{tmp_path}/config.yaml"}
    )
    service.transport = offline_transport
    service.start()
    service.extract_class(None, header=header, message=extract_class_test_message)

    # Check the output files were made
    assert (tmp_path / "Select/job011/particles.star").is_file()

    # Check the slurm commands were run
    slurm_submit_command = (
        f'curl -H "X-SLURM-USER-NAME:user" -H "X-SLURM-USER-TOKEN:token_key" '
        '-H "Content-Type: application/json" -X POST '
        "/url/of/slurm/restapi/slurm/v0.0.40/job/submit "
        f"-d @{tmp_path}/Extract/job012/slurm_run.json"
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

    # Get the expected commands
    extract_command = [
        "EM/cryoemservices.reextract",
        "--extract_job_dir",
        str(tmp_path / "Extract/job012"),
        "--select_job_dir",
        str(tmp_path / "Select/job011"),
        "--original_dir",
        str(tmp_path),
        "--full_boxsize",
        "228",
        "--scaled_boxsize",
        "114",
        "--full_pixel_size",
        "0.5",
        "--scaled_pixel_size",
        "1.0",
        "--bg_radius",
        "-1",
        "--invert_contrast",
        "--normalise",
        "--downscale",
    ]
    rescaling_command = [
        "relion_image_handler",
        "--i",
        str(tmp_path / "Class3D/job010/run_it025_class001.mrc"),
        "--o",
        str(tmp_path / "Extract/job012/refinement_reference_class001.mrc"),
        "--angpix",
        "4.25",
        "--rescale_angpix",
        "1.0",
        "--force_header_angpix",
        "1.0",
        "--new_box",
        "114",
    ]

    # Check that the correct messages were sent
    offline_transport.send.assert_any_call(
        destination="refine_wrapper",
        message={
            "content": "dummy",
            "parameters": {
                "refine_job_dir": f"{tmp_path}/Refine3D/job013",
                "particles_file": f"{tmp_path}/Extract/job012/particles.star",
                "rescaling_command": rescaling_command,
                "rescaled_class_reference": str(
                    tmp_path / "Extract/job012/refinement_reference_class001.mrc"
                ),
                "is_first_refinement": True,
                "number_of_particles": 1,
                "batch_size": 1,
                "pixel_size": "1.0",
                "class_number": 1,
            },
        },
    )
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "job_type": "relion.select.onvalue",
                "input_file": str(tmp_path / "Class3D/job010/run_it025_data.star"),
                "output_file": f"{tmp_path}/Select/job011/particles.star",
                "relion_options": output_relion_options,
                "command": "",
                "stdout": "",
                "stderr": "",
                "success": True,
            },
            "content": "dummy",
        },
    )
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "job_type": "relion.extract",
                "input_file": (
                    f"{tmp_path}/Select/job011/particles.star"
                    f":CtfFind/job003/micrographs_ctf.star"
                ),
                "output_file": f"{tmp_path}/Extract/job012/particles.star",
                "relion_options": output_relion_options,
                "command": " ".join(extract_command),
                "stdout": "",
                "stderr": "",
                "success": True,
            },
            "content": "dummy",
        },
    )
