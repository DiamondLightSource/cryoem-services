from __future__ import annotations

import sys
from unittest import mock

import pytest
import zocalo.configuration
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import bfactor_setup
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
def test_bfactor_service(mock_environment, offline_transport, tmp_path):
    """
    Send a test message to the BFactor setup service
    This should create a particles file then send messages on to
    refinement and the node_creator
    """
    # Create the expected input files
    bfactor_dir = tmp_path / "bfactor_run/bfactor_2"
    (tmp_path / "Extract/Reextract_class1").mkdir(parents=True)
    with open(tmp_path / "Extract/Reextract_class1/particles.star", "w") as f:
        f.write(
            "data_particles\n\nloop_\n_rlnCoordinateX\n_rlnCoordinateY\n_rlnImageName"
            "\n_rlnMicrographName\n_rlnOpticsGroup"
        )
        for i in range(5):
            f.write(f"\n1.0 2.0 {i}@Extract.mrcs sample.mrc 1")

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    bfactor_test_message = {
        "parameters": {
            "bfactor_directory": str(bfactor_dir),
            "rescaled_class_reference": str(tmp_path / "rescaled_class.mrc"),
            "class_number": 1,
            "number_of_particles": 2,
            "batch_size": 10000,
            "pixel_size": 0.5,
            "mask": str(tmp_path / "mask.mrc"),
            "relion_options": {},
        },
        "content": "dummy",
    }
    output_relion_options = dict(RelionServiceOptions())
    output_relion_options.update(bfactor_test_message["parameters"]["relion_options"])
    output_relion_options["pixel_size"] = 0.5
    output_relion_options["batch_size"] = 10000

    # Set up the mock service and call it
    service = bfactor_setup.BFactor(environment=mock_environment)
    service.transport = offline_transport
    service.start()
    service.bfactor_setup(None, header=header, message=bfactor_test_message)

    # Check the output files were made
    assert (bfactor_dir / "Select/job002/particles_split1.star").is_file()

    split_command = [
        "relion_star_handler",
        "--i",
        str(bfactor_dir / "Import/job001/particles.star"),
        "--o",
        "Select/job002/particles_split1.star",
        "--split",
        "--random_order",
        "--nr_split",
        "1",
        "--size_split",
        "2",
        "--pipeline_control",
        "Select/job002/",
    ]

    # Check that the correct messages were sent
    offline_transport.send.assert_any_call(
        destination="refine_wrapper",
        message={
            "content": "dummy",
            "parameters": {
                "refine_job_dir": f"{bfactor_dir}/Refine3D/job003",
                "particles_file": f"{bfactor_dir}/Select/job002/particles_split1.star",
                "rescaled_class_reference": str(
                    bfactor_dir / "Import/job001/refinement_ref.mrc"
                ),
                "is_first_refinement": False,
                "number_of_particles": 2,
                "batch_size": 10000,
                "pixel_size": 0.5,
                "mask": str(bfactor_dir / "Import/job001/mask.mrc"),
                "class_number": 1,
            },
        },
    )
    offline_transport.send.assert_any_call(
        destination="node_creator",
        message={
            "parameters": {
                "job_type": "relion.select.split",
                "input_file": str(bfactor_dir / "Import/job001/particles.star"),
                "output_file": f"{bfactor_dir}/Select/job002/particles_split1.star",
                "relion_options": output_relion_options,
                "command": " ".join(split_command),
                "stdout": "",
                "stderr": "",
                "success": True,
            },
            "content": "dummy",
        },
    )
