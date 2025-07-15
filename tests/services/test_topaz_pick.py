from __future__ import annotations

import sys
from unittest import mock

import numpy as np
import pytest
import starfile
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import topaz_pick
from cryoemservices.util.relion_service_options import RelionServiceOptions


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    mocker.spy(transport, "nack")
    return transport


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.topaz_pick.normalize_images")
@mock.patch("cryoemservices.services.topaz_pick.score_images")
@mock.patch("cryoemservices.services.topaz_pick.non_maximum_suppression")
def test_topaz_with_diameter(
    mock_topaz_nms, mock_topaz_score, mock_topaz_normalise, offline_transport, tmp_path
):
    """
    Send a test message to topaz picking
    This should call the mock subprocess then send messages on to the
    node_creator, murfey_feedback, ispyb_connector and images services
    """
    mock_topaz_score.return_value = [["name1", 1], ["name2", 2]]
    mock_topaz_nms.return_value = (
        [1, 2, 3],
        np.array([[1.1, 1.2], [2.1, 2.2], [3.1, 3.2]]),
    )

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }

    output_path = tmp_path / "AutoPick/job007/STAR/sample.star"
    topaz_test_message = {
        "pixel_size": 0.1,
        "input_path": "MotionCorr/job002/sample.mrc",
        "output_path": str(output_path),
        "scale": 8,
        "log_threshold": -6,
        "log_threshold_for_radius": 0,
        "max_particle_radius": 40,
        "topaz_model": "resnet16",
        "particle_diameter": 1.1,
        "mc_uuid": 0,
        "picker_uuid": 0,
        "ctf_values": {"dummy": "dummy"},
        "relion_options": {"batch_size": 20000, "downscale": True},
    }
    output_relion_options_model = RelionServiceOptions()
    output_relion_options_model.particle_diameter = 1.1
    output_relion_options = dict(output_relion_options_model)
    output_relion_options.update(topaz_test_message["relion_options"])

    # Set up the mock service and send the message to it
    service = topaz_pick.TopazPick(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    service.topaz(None, header=header, message=topaz_test_message)

    mock_topaz_normalise.assert_called_once()
    mock_topaz_score.assert_called_once()
    mock_topaz_nms.assert_called_once()

    assert output_path.is_file()
    written_coords = starfile.read(output_path)
    assert (
        written_coords.keys()
        == ["rlnCoordinateX", "rlnCoordinateY", "rlnAutopickFigureOfMerit"]
    ).all()
    assert (written_coords["rlnCoordinateX"] == [9, 17, 25]).all()
    assert (written_coords["rlnCoordinateY"] == [10, 18, 26]).all()
    assert (written_coords["rlnAutopickFigureOfMerit"] == [1, 2, 3]).all()

    # Check that the correct messages were sent
    assert offline_transport.send.call_count == 4
    extraction_params = {
        "ctf_values": topaz_test_message["ctf_values"],
        "micrographs_file": topaz_test_message["input_path"],
        "coord_list_file": topaz_test_message["output_path"],
        "extract_file": "Extract/job008/Movies/sample_extract.star",
    }
    offline_transport.send.assert_any_call(
        "ispyb_connector",
        {
            "particle_picking_template": "resnet16",
            "number_of_particles": 3,
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
            "file": topaz_test_message["input_path"],
            "coordinates": [["9", "10"], ["17", "18"], ["25", "26"]],
            "pixel_size": 0.1,
            "diameter": 1.1,
            "outfile": str(output_path.with_suffix(".jpeg")),
        },
    )
    offline_transport.send.assert_any_call(
        "murfey_feedback",
        {
            "register": "picked_particles",
            "motion_correction_id": topaz_test_message["mc_uuid"],
            "micrograph": topaz_test_message["input_path"],
            "particle_diameters": [1.1, 1.1, 1.1],
            "extraction_parameters": extraction_params,
        },
    )
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.autopick.topaz.pick",
            "input_file": topaz_test_message["input_path"],
            "output_file": str(output_path),
            "relion_options": output_relion_options,
            "command": (
                "topaz extract -m resnet16 -r 0 -s 8 -t -6.0 --per-micrograph -o "
                f"{output_path} MotionCorr/job002/sample.mrc"
            ),
            "stdout": "",
            "stderr": "",
            "success": True,
        },
    )
