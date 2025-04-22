from __future__ import annotations

import copy
import json
import sys
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import cryolo
from cryoemservices.util.relion_service_options import RelionServiceOptions


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    mocker.spy(transport, "nack")
    return transport


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.cryolo.subprocess.run")
def test_cryolo_service_spa(mock_subprocess, offline_transport, tmp_path):
    """
    Send a test message to CrYOLO
    This should call the mock subprocess then send messages on to the
    node_creator, murfey_feedback, ispyb_connector and images services
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }

    output_path = tmp_path / "AutoPick/job007/STAR/sample.star"
    cbox_path = tmp_path / "AutoPick/job007/CBOX/sample.cbox"
    ctf_test_values = {
        "CtfMaxResolution": 0.00001,
        "DefocusU": 0.05,
        "DefocusV": 0.08,
    }
    cryolo_test_message = {
        "pixel_size": 0.1,
        "input_path": "MotionCorr/job002/sample.mrc",
        "output_path": str(output_path),
        "experiment_type": "spa",
        "cryolo_config_file": str(tmp_path) + "/config.json",
        "cryolo_model_weights": "sample_weights",
        "cryolo_threshold": 0.15,
        "retained_fraction": 0.5,
        "min_particles": 0,
        "mc_uuid": 0,
        "picker_uuid": 0,
        "particle_diameter": 1.1,
        "ctf_values": ctf_test_values,
        "cryolo_command": "cryolo_predict.py",
        "relion_options": {"batch_size": 20000, "downscale": True},
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options.update(cryolo_test_message["relion_options"])
    output_relion_options["cryolo_config_file"] = str(
        tmp_path / "AutoPick/job007/cryolo_config.json"
    )

    # Write a dummy config file expected by cryolo
    with open(tmp_path / "config.json", "w") as f:
        f.write('{\n"model": {\n"anchors": [160, 160]\n}\n}')

    # Write star co-ordinate file in the format cryolo will output
    output_path.parent.mkdir(parents=True)
    cbox_path.parent.mkdir(parents=True)
    with open(cbox_path, "w") as f:
        f.write(
            "data_cryolo\n\nloop_\n\n_EstWidth\n_EstHeight\n_Confidence\n"
            "_CoordinateX\n_CoordinateY\n_Width\n_Height\n"
            "100 200 0.6 0.1 0.2 2 4\n100 200 0.5 0.3 0.4 6 8"
        )

    # Set up the mock service and send the message to it
    service = cryolo.CrYOLO(environment={"queue": ""}, transport=offline_transport)
    service.initializing()
    service.cryolo(None, header=header, message=cryolo_test_message)

    assert mock_subprocess.call_count == 4
    mock_subprocess.assert_called_with(
        [
            "cryolo_predict.py",
            "--conf",
            str(tmp_path / "AutoPick/job007/cryolo_config.json"),
            "-o",
            str(output_path.parent.parent),
            "--otf",
            "-i",
            "MotionCorr/job002/sample.mrc",
            "--weights",
            "sample_weights",
            "--threshold",
            "0.15",
            "--distance",
            "0",
            "--norm_margin",
            "0",
        ],
        cwd=tmp_path / "AutoPick/job007",
        capture_output=True,
    )

    # Check the config file which was made
    assert (tmp_path / "AutoPick/job007/cryolo_config.json").is_file()
    with open(tmp_path / "AutoPick/job007/cryolo_config.json") as config_file:
        config_values = json.load(config_file)

    assert config_values["model"] == {
        "architecture": "PhosaurusNet",
        "input_size": 1024,
        "max_box_per_image": 600,
        "norm": "STANDARD",
        "num_patches": 1,
        "filter": [0.1, "filtered"],
        "anchors": [160, 160],
    }
    assert config_values["other"] == {"log_path": "logs/"}

    # Check that the correct messages were sent
    assert offline_transport.send.call_count == 4
    extraction_params = {
        "ctf_values": cryolo_test_message["ctf_values"],
        "micrographs_file": cryolo_test_message["input_path"],
        "coord_list_file": cryolo_test_message["output_path"],
        "extract_file": "Extract/job008/Movies/sample_extract.star",
    }
    offline_transport.send.assert_any_call(
        "ispyb_connector",
        {
            "particle_picking_template": "sample_weights",
            "number_of_particles": 1,
            "particle_diameter": 1.1,
            "summary_image_full_path": str(output_path.with_suffix(".jpeg")),
            "ispyb_command": "buffer",
            "buffer_lookup": {"motion_correction_id": 0},
            "buffer_command": {"ispyb_command": "insert_particle_picker"},
            "buffer_store": 0,
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "picked_particles",
            "file": cryolo_test_message["input_path"],
            "coordinates": [["1.1", "2.2"]],
            "pixel_size": 0.1,
            "diameter": 1.1,
            "outfile": str(output_path.with_suffix(".jpeg")),
        },
    )
    offline_transport.send.assert_any_call(
        "murfey_feedback",
        {
            "register": "picked_particles",
            "motion_correction_id": cryolo_test_message["mc_uuid"],
            "micrograph": cryolo_test_message["input_path"],
            "particle_diameters": [10.0, 20.0],
            "particle_count": 2,
            "resolution": ctf_test_values["CtfMaxResolution"],
            "astigmatism": ctf_test_values["DefocusV"] - ctf_test_values["DefocusU"],
            "defocus": (ctf_test_values["DefocusU"] + ctf_test_values["DefocusV"]) / 2,
            "extraction_parameters": extraction_params,
        },
    )
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "job_type": "cryolo.autopick",
            "input_file": cryolo_test_message["input_path"],
            "output_file": str(output_path),
            "relion_options": output_relion_options,
            "command": (
                f"cryolo_predict.py --conf {tmp_path}/AutoPick/job007/cryolo_config.json "
                f"-o {tmp_path}/AutoPick/job007 --otf "
                f"-i MotionCorr/job002/sample.mrc "
                f"--weights sample_weights --threshold 0.15 "
                "--distance 0 --norm_margin 0"
            ),
            "stdout": "stdout",
            "stderr": "stderr",
            "experiment_type": "spa",
            "success": True,
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.cryolo.subprocess.run")
def test_cryolo_service_tomography(mock_subprocess, offline_transport, tmp_path):
    """
    Send a test message to CrYOLO
    This should call the mock subprocess then send messages on to the
    node_creator, murfey_feedback, ispyb_connector and images services
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }

    output_path = tmp_path / "AutoPick/job007/STAR/sample.star"
    cryolo_test_message = {
        "input_path": "MotionCorr/job002/sample.mrc",
        "output_path": str(output_path),
        "experiment_type": "tomography",
        "cryolo_box_size": 40,
        "cryolo_model_weights": "sample_weights",
        "cryolo_threshold": 0.15,
        "retained_fraction": 0.5,
        "min_particles": 0,
        "particle_diameter": 1.1,
        "tomo_tracing_min_frames": 5,
        "tomo_tracing_missing_frames": 0,
        "tomo_tracing_search_range": -1,
        "ctf_values": {"dummy": "dummy"},
        "cryolo_command": "cryolo_predict.py",
        "relion_options": {},
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options["cryolo_config_file"] = str(
        tmp_path / "AutoPick/job007/cryolo_config.json"
    )

    # Set up the mock service and send the message to it
    service = cryolo.CrYOLO(environment={"queue": ""}, transport=offline_transport)
    service.initializing()
    service.cryolo(None, header=header, message=cryolo_test_message)

    assert mock_subprocess.call_count == 4
    mock_subprocess.assert_called_with(
        [
            "cryolo_predict.py",
            "--conf",
            str(tmp_path / "AutoPick/job007/cryolo_config.json"),
            "-o",
            str(output_path.parent.parent),
            "--tomogram",
            "-tsr",
            "-1",
            "-tmem",
            "0",
            "-tmin",
            "5",
            "--gpus",
            "0",
            "-i",
            "MotionCorr/job002/sample.mrc",
            "--weights",
            "sample_weights",
            "--threshold",
            "0.15",
            "--distance",
            "0",
            "--norm_margin",
            "0",
        ],
        cwd=tmp_path / "AutoPick/job007",
        capture_output=True,
    )

    # Check the config file which was made
    assert (tmp_path / "AutoPick/job007/cryolo_config.json").is_file()
    with open(tmp_path / "AutoPick/job007/cryolo_config.json") as config_file:
        config_values = json.load(config_file)

    assert config_values["model"] == {
        "architecture": "PhosaurusNet",
        "input_size": 1024,
        "max_box_per_image": 600,
        "norm": "STANDARD",
        "num_patches": 1,
        "anchors": [40, 40],
    }
    assert config_values["other"] == {"log_path": "logs/"}

    # Check that the correct messages were sent
    assert offline_transport.send.call_count == 4
    offline_transport.send.assert_any_call(
        "ispyb_connector",
        {
            "ispyb_command": "insert_processed_tomogram",
            "file_path": cryolo_test_message["output_path"],
            "processing_type": "Picked",
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "picked_particles_3d_apng",
            "file": cryolo_test_message["input_path"],
            "coordinates_file": cryolo_test_message["output_path"],
            "box_size": 40,
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "picked_particles_3d_central_slice",
            "file": cryolo_test_message["input_path"],
            "coordinates_file": cryolo_test_message["output_path"],
            "box_size": 40,
        },
    )
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "job_type": "cryolo.autopick",
            "input_file": cryolo_test_message["input_path"],
            "output_file": str(output_path),
            "relion_options": output_relion_options,
            "command": (
                f"cryolo_predict.py --conf {tmp_path}/AutoPick/job007/cryolo_config.json "
                f"-o {tmp_path}/AutoPick/job007 "
                f"--tomogram -tsr -1 -tmem 0 -tmin 5 --gpus 0 "
                "-i MotionCorr/job002/sample.mrc "
                "--weights sample_weights --threshold 0.15 "
                "--distance 0 --norm_margin 0"
            ),
            "stdout": "stdout",
            "stderr": "stderr",
            "experiment_type": "tomography",
            "success": True,
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.cryolo.subprocess.run")
def test_cryolo_spa_needs_uuids_and_pixel_size(
    mock_subprocess, offline_transport, tmp_path
):
    """
    Send a test message to CrYOLO without some of the necessary parameters
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }

    output_path = tmp_path / "AutoPick/job007/STAR/sample.star"
    cryolo_test_message = {
        "parameters": {
            "pixel_size": 0.1,
            "input_path": "MotionCorr/job002/sample.mrc",
            "output_path": str(output_path),
            "experiment_type": "spa",
            "min_particles": 0,
            "mc_uuid": 0,
            "picker_uuid": 0,
            "relion_options": {},
        },
        "content": "dummy",
    }

    # Set up the mock service
    service = cryolo.CrYOLO(environment={"queue": ""}, transport=offline_transport)
    service.initializing()

    # Send messages without pixel_size, mc_uuid and picker_uuid in turn
    no_pixel_size_message = copy.deepcopy(cryolo_test_message)
    no_pixel_size_message["parameters"]["pixel_size"] = None
    service.cryolo(None, header=header, message=no_pixel_size_message)

    no_mc_uuid_message = copy.deepcopy(cryolo_test_message)
    no_mc_uuid_message["parameters"]["mc_uuid"] = None
    service.cryolo(None, header=header, message=no_mc_uuid_message)

    no_picker_uuid_message = copy.deepcopy(cryolo_test_message)
    no_picker_uuid_message["parameters"]["picker_uuid"] = None
    service.cryolo(None, header=header, message=no_picker_uuid_message)

    # None of these should call subprocess or send, all should nack the message
    mock_subprocess.assert_not_called()
    offline_transport.send.assert_not_called()
    assert offline_transport.nack.call_count == 3


def test_parse_cryolo_output(offline_transport):
    """
    Send test lines to the output parser
    to check the number of particles is being read in
    """
    service = cryolo.CrYOLO(environment={"queue": ""}, transport=offline_transport)
    service.initializing()

    service.number_of_particles = 0
    cryolo.CrYOLO.parse_cryolo_output(service, "30 particles in total has been found")
    cryolo.CrYOLO.parse_cryolo_output(service, "Deleted 10 particles")
    assert service.number_of_particles == 20
