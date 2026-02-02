from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
from gemmi import cif
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import extract
from cryoemservices.util.relion_service_options import RelionServiceOptions


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    mocker.spy(transport, "ack")
    mocker.spy(transport, "nack")
    return transport


@mock.patch("cryoemservices.services.extract.mrcfile.open")
def test_extract_service(mock_mrcfile, offline_transport, tmp_path):
    """
    Send a test message to the extract service
    This should call the mock file reader then send messages on to the
    node_creator and select services
    """
    mock_mrcfile().__enter__().data = np.random.rand(256, 256)

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }

    cryolo_file = tmp_path / "AutoPick/job007/STAR/sample.star"
    cryolo_file.parent.mkdir(parents=True)
    with open(cryolo_file, "w") as f:
        f.write(
            "data_particles\n\nloop_\n_rlnCoordinateX\n_rlnCoordinateY\n100.0 200.0"
        )
    output_path = tmp_path / "Extract/job008/Movies/sample.star"

    extract_test_message = {
        "pixel_size": 0.1,
        "ctf_image": f"{tmp_path}/CtFind/job006/Movies/sample.ctf",
        "ctf_max_resolution": "10",
        "ctf_figure_of_merit": "20",
        "defocus_u": "1.0",
        "defocus_v": "2.0",
        "defocus_angle": "0.0",
        "micrographs_file": f"{tmp_path}/MotionCorr/job002/sample.mrc",
        "coord_list_file": str(cryolo_file),
        "output_file": str(output_path),
        "particle_diameter": 200,
        "norm": True,
        "bg_radius": -1,
        "downscale": True,
        "invert_contrast": True,
        "confidence_threshold": 1,
        "batch_size": 20000,
        "voltage": 300,
        "relion_options": {"batch_size": 20000},
    }
    output_relion_options = RelionServiceOptions()
    output_relion_options.voltage = extract_test_message["voltage"]
    output_relion_options.pixel_size = extract_test_message["pixel_size"]
    output_relion_options.particle_diameter = extract_test_message["particle_diameter"]
    output_relion_options.downscale = extract_test_message["downscale"]
    output_relion_options.pixel_size_downscaled = (
        extract_test_message["pixel_size"]
        * output_relion_options.boxsize
        / output_relion_options.small_boxsize
    )
    output_relion_options = dict(output_relion_options)
    output_relion_options.update(extract_test_message["relion_options"])

    # Set up the mock service and send the message to it
    service = extract.Extract(environment={"queue": ""}, transport=offline_transport)
    service.initializing()
    service.extract(None, header=header, message=extract_test_message)

    assert mock_mrcfile().__enter__.call_count == 2

    # Check symlinks
    assert (tmp_path / "Extract/Live_all_particles").is_symlink()
    assert (tmp_path / "Extract/Live_all_particles").readlink() == (
        tmp_path / "Extract/job008"
    )

    # Check that the correct messages were sent
    offline_transport.send.assert_any_call(
        "select_particles",
        {
            "input_file": extract_test_message["output_file"],
            "relion_options": output_relion_options,
            "batch_size": output_relion_options["batch_size"],
            "image_size": 96,
        },
    )
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.extract",
            "input_file": (f"{cryolo_file}:{extract_test_message['ctf_image']}"),
            "output_file": str(output_path),
            "relion_options": output_relion_options,
            "command": "",
            "stdout": "",
            "stderr": "",
            "results": {"box_size": 96},
            "alias": "Live_all_particles",
        },
    )

    # Check the output files and their structure
    assert output_path.exists()
    assert output_path.with_suffix(".mrcs").exists()

    particles_file = cif.read_file(str(output_path))
    particles_data = particles_file.find_block("particles")
    assert list(particles_data.find_loop("_rlnCoordinateX")) == ["100.0"]
    assert list(particles_data.find_loop("_rlnCoordinateY")) == ["200.0"]
    assert list(particles_data.find_loop("_rlnImageName")) == [
        f"000000@{output_path.relative_to(tmp_path).with_suffix('.mrcs')}"
    ]
    assert list(particles_data.find_loop("_rlnMicrographName")) == [
        "MotionCorr/job002/sample.mrc"
    ]
    assert list(particles_data.find_loop("_rlnOpticsGroup")) == ["1"]
    assert list(particles_data.find_loop("_rlnCtfMaxResolution")) == ["10.0"]
    assert list(particles_data.find_loop("_rlnCtfFigureOfMerit")) == ["20.0"]
    assert list(particles_data.find_loop("_rlnDefocusU")) == ["1.0"]
    assert list(particles_data.find_loop("_rlnDefocusV")) == ["2.0"]
    assert list(particles_data.find_loop("_rlnDefocusAngle")) == ["0.0"]
    assert list(particles_data.find_loop("_rlnCtfBfactor")) == ["0.0"]
    assert list(particles_data.find_loop("_rlnCtfScalefactor")) == ["1.0"]
    assert list(particles_data.find_loop("_rlnPhaseShift")) == ["0.0"]


@mock.patch("cryoemservices.services.extract.mrcfile.open")
def test_extract_service_check_symlinks(mock_mrcfile, offline_transport, tmp_path):
    """
    Send a test message to the extract service
    to test the symlink setup
    """
    mock_mrcfile().__enter__().data = np.random.rand(256, 256)

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }

    cryolo_file = tmp_path / "AutoPick/job007/STAR/sample.star"
    cryolo_file.parent.mkdir(parents=True)
    with open(cryolo_file, "w") as f:
        f.write(
            "data_particles\n\nloop_\n_rlnCoordinateX\n_rlnCoordinateY\n100.0 200.0"
        )
    output_path = tmp_path / "Extract/job008/Movies/sample.star"

    extract_test_message = {
        "pixel_size": 0.1,
        "ctf_image": f"{tmp_path}/CtFind/job006/Movies/sample.ctf",
        "ctf_max_resolution": "10",
        "ctf_figure_of_merit": "20",
        "defocus_u": "1.0",
        "defocus_v": "2.0",
        "defocus_angle": "0.0",
        "micrographs_file": f"{tmp_path}/MotionCorr/job002/sample.mrc",
        "coord_list_file": str(cryolo_file),
        "output_file": str(output_path),
        "relion_options": {},
    }

    # Set up the mock service and send the message to it
    service = extract.Extract(environment={"queue": ""}, transport=offline_transport)
    service.initializing()

    # Case 1: no symlink
    service.extract(None, header=header, message=extract_test_message)
    offline_transport.ack.assert_called_once()

    # Case 2: ok symlink
    assert (tmp_path / "Extract/Live_all_particles").is_symlink()
    service.extract(None, header=header, message=extract_test_message)
    assert offline_transport.ack.call_count == 2

    # Case 3: bad symlink
    (tmp_path / "Extract/Live_all_particles").unlink()
    (tmp_path / "Extract/Live_all_particles").symlink_to(tmp_path / "AutoPick")
    service.extract(None, header=header, message=extract_test_message)
    offline_transport.nack.assert_called_once()
