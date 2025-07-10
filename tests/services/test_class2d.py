from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.class2d import Class2D
from cryoemservices.util.relion_service_options import RelionServiceOptions


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    mocker.spy(transport, "ack")
    mocker.spy(transport, "nack")
    return transport


@mock.patch("cryoemservices.wrappers.class2d_wrapper.subprocess.run")
def test_class2d_service_complete_batch(mock_subprocess, offline_transport, tmp_path):
    """
    Send a test message to the Class2D service for a complete batch,
    which should then do images, ispyb and node_creator message sends,
    and also start icebreaker and class selection jobs
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("utf8")
    mock_subprocess().stderr = "stderr".encode("utf8")

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    class2d_test_message = {
        "batch_is_complete": True,
        "batch_size": "50000",
        "class2d_dir": f"{tmp_path}/Class2D/job010",
        "class2d_grp_uuid": "5",
        "class2d_nr_classes": "1",
        "class_uuids": "{'0': 10}",
        "do_icebreaker_jobs": True,
        "do_vdam": False,
        "gpus": "0:1:2:3",
        "particle_diameter": "180",
        "particles_file": f"{tmp_path}/Select/job009/particles_split2.star",
        "picker_id": "6",
        "relion_options": {},
    }
    output_relion_options = RelionServiceOptions()
    output_relion_options.particle_diameter = 180
    output_relion_options.class2d_nr_classes = 1
    output_relion_options.class2d_nr_iter = 20
    output_relion_options = dict(output_relion_options)

    # Create the expected output files
    (tmp_path / "Class2D/job010").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "Class2D/job010/run_it020_data.star", "w") as data_star:
        data_star.write(
            "data_optics\nloop_\n_rlnImagePixelSize\n2.5\n\n"
            "data_particles\nloop_\n_rlnCoordinateX\n1\n2\n3\n4\n5"
        )
    with open(tmp_path / "Class2D/job010/run_it020_model.star", "w") as model_star:
        model_star.write(
            "data_model_classes\nloop_\n"
            "_rlnReferenceImage\n_Fraction\n_Rotation\n_Translation\n"
            "_Resolution\n_Completeness\n_OffsetX\n_OffsetY\n"
            "1@Class2D/job010/run_it020_classes.mrcs 0.4 30.3 33.3 12.2 1.0 0.6 0.01\n"
        )

    # Set up and run the service
    service = Class2D(environment={"queue": ""}, rabbitmq_credentials=Path("."))
    service._transport = offline_transport
    service.initializing()
    service.class2d(None, header=header, message=class2d_test_message)

    # Check the expected command was run
    class2d_command = [
        "srun",
        "-n",
        "5",
        "relion_refine_mpi",
        "--i",
        "Select/job009/particles_split2.star",
        "--o",
        "Class2D/job010/run",
        "--particle_diameter",
        str(output_relion_options["mask_diameter"]),
        "--dont_combine_weights_via_disc",
        "--preread_images",
        "--pool",
        "100",
        "--pad",
        "2",
        "--ctf",
        "--tau2_fudge",
        "2",
        "--K",
        "1",
        "--flatten_solvent",
        "--zero_mask",
        "--center_classes",
        "--oversampling",
        "1",
        "--psi_step",
        "12.0",
        "--offset_range",
        "5",
        "--offset_step",
        "2",
        "--norm",
        "--scale",
        "--j",
        "8",
        "--gpu",
        "0:1:2:3",
        "--pipeline_control",
        "Class2D/job010/",
        "--iter",
        "20",
    ]
    mock_subprocess.assert_called_with(
        class2d_command, capture_output=True, cwd=str(tmp_path)
    )

    # Check the expected message sends to other processes
    assert offline_transport.send.call_count == 5
    # Don't need to re-test the node_creator, images or ispyb messages
    offline_transport.send.assert_any_call(
        "icebreaker",
        {
            "icebreaker_type": "particles",
            "input_micrographs": (
                f"{tmp_path}/IceBreaker/job003/grouped_micrographs.star"
            ),
            "input_particles": f"{tmp_path}/Select/job009/particles_split2.star",
            "output_path": f"{tmp_path}/IceBreaker/job011/",
            "mc_uuid": -1,
            "relion_options": output_relion_options,
        },
    )
    offline_transport.send.assert_any_call(
        "select_classes",
        {
            "input_file": f"{tmp_path}/Class2D/job010/run_it020_optimiser.star",
            "relion_options": output_relion_options,
            "class_uuids": "{'0': 10}",
        },
    )
    offline_transport.ack.assert_called_once()
