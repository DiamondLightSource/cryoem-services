from __future__ import annotations

import sys
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import ctffind
from cryoemservices.util.relion_service_options import RelionServiceOptions


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.ctffind.subprocess.run")
def test_ctffind4_service_spa(mock_subprocess, offline_transport, tmp_path):
    """
    Send a test message to CTFFind
    This should call the mock subprocess then send messages on to the
    cryolo, node_creator, ispyb_connector and images services
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("utf8")
    mock_subprocess().stderr = "stderr".encode("utf8")

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    ctffind_test_message = {
        "experiment_type": "spa",
        "pixel_size": 0.1,
        "voltage": 300.0,
        "spher_aber": 2.7,
        "ampl_contrast": 0.1,
        "ampl_spectrum": 512,
        "min_res": 30.0,
        "max_res": 5.0,
        "min_defocus": 5000.0,
        "max_defocus": 50000.0,
        "defocus_step": 100.0,
        "astigmatism_known": "no",
        "slow_search": "no",
        "astigmatism_restrain": "no",
        "additional_phase_shift": "no",
        "expert_options": "no",
        "input_image": f"{tmp_path}/MotionCorr/job002/sample.mrc",
        "output_image": f"{tmp_path}/CtfFind/job006/sample.ctf",
        "mc_uuid": 0,
        "picker_uuid": 0,
        "relion_options": {"cryolo_threshold": 0.3},
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options.update(ctffind_test_message["relion_options"])

    # Set up the mock service
    service = ctffind.CTFFind(environment={"queue": ""})
    service.transport = offline_transport
    service.start()

    # Set some parameters then send a message to the service
    service.defocus1 = 1
    service.defocus2 = 2
    service.astigmatism_angle = 3
    service.cc_value = 4
    service.estimated_resolution = 5
    service.ctf_find(None, header=header, message=ctffind_test_message)

    parameters_list = [
        ctffind_test_message["input_image"],
        ctffind_test_message["output_image"],
        ctffind_test_message["pixel_size"],
        ctffind_test_message["voltage"],
        ctffind_test_message["spher_aber"],
        ctffind_test_message["ampl_contrast"],
        ctffind_test_message["ampl_spectrum"],
        ctffind_test_message["min_res"],
        ctffind_test_message["max_res"],
        ctffind_test_message["min_defocus"],
        ctffind_test_message["max_defocus"],
        ctffind_test_message["defocus_step"],
        ctffind_test_message["astigmatism_known"],
        ctffind_test_message["slow_search"],
        ctffind_test_message["astigmatism_restrain"],
        ctffind_test_message["additional_phase_shift"],
        ctffind_test_message["expert_options"],
    ]
    parameters_string = "\n".join(map(str, parameters_list))

    assert mock_subprocess.call_count == 4
    mock_subprocess.assert_called_with(
        ["ctffind"],
        input=parameters_string.encode("ascii"),
        capture_output=True,
    )

    # Check that the correct messages were sent
    assert offline_transport.send.call_count == 4
    offline_transport.send.assert_any_call(
        "cryolo",
        {
            "input_path": ctffind_test_message["input_image"],
            "output_path": f"{tmp_path}/AutoPick/job007/STAR/sample.star",
            "ctf_values": {
                "CtfImage": ctffind_test_message["output_image"],
                "CtfMaxResolution": service.estimated_resolution,
                "CtfFigureOfMerit": service.cc_value,
                "DefocusU": service.defocus1,
                "DefocusV": service.defocus2,
                "DefocusAngle": service.astigmatism_angle,
            },
            "experiment_type": "spa",
            "relion_options": output_relion_options,
            "mc_uuid": ctffind_test_message["mc_uuid"],
            "picker_uuid": ctffind_test_message["picker_uuid"],
            "pixel_size": ctffind_test_message["pixel_size"],
        },
    )
    offline_transport.send.assert_any_call(
        "ispyb_connector",
        {
            "box_size_x": str(ctffind_test_message["ampl_spectrum"]),
            "box_size_y": str(ctffind_test_message["ampl_spectrum"]),
            "min_resolution": str(ctffind_test_message["min_res"]),
            "max_resolution": str(ctffind_test_message["max_res"]),
            "min_defocus": str(ctffind_test_message["min_defocus"]),
            "max_defocus": str(ctffind_test_message["max_defocus"]),
            "astigmatism": str(service.defocus1 - service.defocus2),
            "defocus_step_size": str(ctffind_test_message["defocus_step"]),
            "astigmatism_angle": str(service.astigmatism_angle),
            "estimated_resolution": str(service.estimated_resolution),
            "estimated_defocus": str((service.defocus1 + service.defocus2) / 2),
            "amplitude_contrast": str(ctffind_test_message["ampl_contrast"]),
            "cc_value": str(service.cc_value),
            "fft_theoretical_full_path": f"{tmp_path}/CtfFind/job006/sample.jpeg",
            "ispyb_command": "buffer",
            "buffer_lookup": {"motion_correction_id": 0},
            "buffer_command": {"ispyb_command": "insert_ctf"},
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "mrc_to_jpeg",
            "file": f"{tmp_path}/CtfFind/job006/sample.ctf",
        },
    )
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "experiment_type": "spa",
            "job_type": "relion.ctffind.ctffind4",
            "input_file": f"{tmp_path}/MotionCorr/job002/sample.mrc",
            "output_file": f"{tmp_path}/CtfFind/job006/sample.ctf",
            "relion_options": output_relion_options,
            "command": f"ctffind\n{' '.join(map(str, parameters_list))}",
            "stdout": "stdout",
            "stderr": "stderr",
            "success": True,
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.ctffind.subprocess.run")
def test_ctffind5_service_tomo(mock_subprocess, offline_transport, tmp_path):
    """
    Send a test message to CTFFind with the version 5 flags on
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("utf8")
    mock_subprocess().stderr = "stderr".encode("utf8")

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    ctffind_test_message = {
        "experiment_type": "tomography",
        "pixel_size": 0.1,
        "determine_tilt": "yes",
        "determine_thickness": "yes",
        "brute_force_1d": "yes",
        "refinement_2d": "yes",
        "node_min_res": 30.0,
        "node_max_res": 3.0,
        "node_rounded_square": "no",
        "node_downweight": "no",
        "ctffind_version": 5,
        "input_image": f"{tmp_path}/MotionCorr/job002/sample.mrc",
        "output_image": f"{tmp_path}/CtfFind/job006/sample.ctf",
        "mc_uuid": 0,
        "picker_uuid": 0,
        "relion_options": {},
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options.update(ctffind_test_message["relion_options"])

    # Set up the mock service
    service = ctffind.CTFFind(environment={"queue": ""})
    service.transport = offline_transport
    service.start()

    # Set some parameters then send a message to the service
    service.defocus1 = 1
    service.defocus2 = 2
    service.astigmatism_angle = 3
    service.cc_value = 4
    service.estimated_resolution = 5
    service.ctf_find(None, header=header, message=ctffind_test_message)

    parameters_list = [
        ctffind_test_message["input_image"],
        ctffind_test_message["output_image"],
        ctffind_test_message["pixel_size"],
        "300.0",
        "2.7",
        "0.1",
        "512",
        "30.0",
        "5.0",
        "5000.0",
        "50000.0",
        "100.0",
        "no",
        "no",
        "no",
        "no",
        ctffind_test_message["determine_tilt"],
        ctffind_test_message["determine_thickness"],
        ctffind_test_message["brute_force_1d"],
        ctffind_test_message["refinement_2d"],
        ctffind_test_message["node_min_res"],
        ctffind_test_message["node_max_res"],
        ctffind_test_message["node_rounded_square"],
        ctffind_test_message["node_downweight"],
        "no",
    ]
    parameters_string = "\n".join(map(str, parameters_list))

    assert mock_subprocess.call_count == 4
    mock_subprocess.assert_called_with(
        ["ctffind5"],
        input=parameters_string.encode("ascii"),
        capture_output=True,
    )

    # Check that the correct messages were sent (no need to recheck ones tested above)
    assert offline_transport.send.call_count == 3
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "experiment_type": "tomography",
            "job_type": "relion.ctffind.ctffind4",
            "input_file": f"{tmp_path}/MotionCorr/job002/sample.mrc",
            "output_file": f"{tmp_path}/CtfFind/job006/sample.ctf",
            "relion_options": output_relion_options,
            "command": f"ctffind5\n{' '.join(map(str, parameters_list))}",
            "stdout": "stdout",
            "stderr": "stderr",
            "success": True,
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.ctffind.subprocess.run")
def test_ctffind5_service_nothickness(mock_subprocess, offline_transport, tmp_path):
    """
    Send a test message to CTFFind version 5 without thickness determination
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("utf8")
    mock_subprocess().stderr = "stderr".encode("utf8")

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    ctffind_test_message = {
        "experiment_type": "tomography",
        "pixel_size": 0.2,
        "determine_tilt": "yes",
        "ctffind_version": 5,
        "input_image": f"{tmp_path}/MotionCorr/job002/sample.mrc",
        "output_image": f"{tmp_path}/CtfFind/job006/sample.ctf",
        "mc_uuid": 0,
        "picker_uuid": 0,
        "relion_options": {},
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options.update(ctffind_test_message["relion_options"])

    # Set up the mock service
    service = ctffind.CTFFind(environment={"queue": ""})
    service.transport = offline_transport
    service.start()

    # Set some parameters then send a message to the service
    service.defocus1 = 1
    service.defocus2 = 2
    service.astigmatism_angle = 3
    service.cc_value = 4
    service.estimated_resolution = 5
    service.ctf_find(None, header=header, message=ctffind_test_message)

    parameters_list = [
        ctffind_test_message["input_image"],
        ctffind_test_message["output_image"],
        ctffind_test_message["pixel_size"],
        "300.0",
        "2.7",
        "0.1",
        "512",
        "30.0",
        "5.0",
        "5000.0",
        "50000.0",
        "100.0",
        "no",
        "no",
        "no",
        "no",
        "yes",
        "no",
        "no",
    ]
    parameters_string = "\n".join(map(str, parameters_list))

    assert mock_subprocess.call_count == 4
    mock_subprocess.assert_called_with(
        ["ctffind5"],
        input=parameters_string.encode("ascii"),
        capture_output=True,
    )

    # Check that the correct messages were sent (no need to recheck ones tested above)
    assert offline_transport.send.call_count == 3


def test_parse_ctffind_output(offline_transport):
    """
    Send test lines to the output parser
    to check the ctf values are being read in
    """
    service = ctffind.CTFFind(environment={"queue": ""})
    service.transport = offline_transport
    service.start()

    ctffind.CTFFind.parse_ctf_output(service, "Estimated defocus values        : 1 , 2")
    ctffind.CTFFind.parse_ctf_output(service, "Estimated azimuth of astigmatism: 3")
    ctffind.CTFFind.parse_ctf_output(service, "Score                           : 4")
    ctffind.CTFFind.parse_ctf_output(service, "Thon rings with good fit up to  : 5")
    assert service.defocus1 == 1
    assert service.defocus2 == 2
    assert service.astigmatism_angle == 3
    assert service.cc_value == 4
    assert service.estimated_resolution == 5
