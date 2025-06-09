from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import mrcfile
import numpy as np
import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import select_classes
from cryoemservices.util.relion_service_options import RelionServiceOptions


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


def select_classes_common_setup(
    job_dir: Path, initial_particle_count: int, particles_to_add: int
):
    """Setup for the tests below: create the message for and output of autoselection"""
    particles_file = job_dir / "Select/job012/particles.star"
    particles_file.parent.mkdir(parents=True)
    with open(particles_file, "w") as f:
        f.write("data_optics\n\nloop_\n_group\nopticsGroup1\n\n")
        f.write("data_particles\n\nloop_\n_x\n_y\n_particle\n_movie\n")
        for i in range(particles_to_add):
            f.write(
                f"{i / 100} {i / 100} {i}@Extract/job008/classes.mrcs "
                f"MotionCorr/job002/Movies/movie.mrc\n"
            )

    if initial_particle_count:
        particles_file = job_dir / "Select/job013/particles_all.star"
        particles_file.parent.mkdir(parents=True)
        with open(particles_file, "w") as f:
            f.write("data_optics\n\nloop_\n_group\nopticsGroup1\n\n")
            f.write("data_particles\n\nloop_\n_x\n_y\n_particle\n_movie\n")
            for i in range(initial_particle_count):
                f.write(
                    f"{i/100} {i/100} {i}@Extract/job008/classes.mrcs "
                    f"MotionCorr/job002/Movies/movie.mrc\n"
                )

    Path(job_dir / "MotionCorr/job002/Movies").mkdir(parents=True, exist_ok=True)
    with mrcfile.new(job_dir / "MotionCorr/job002/Movies/movie.mrc") as mrc:
        mrc.set_data(np.random.random((64, 64)).astype(np.float32))
        mrc.header.mx = 100
        mrc.header.my = 50
        mrc.header.mz = 1
        mrc.header.cella.x = 50
        mrc.header.cella.y = 50
        mrc.header.cella.z = 1

    autoselect_file = job_dir / "Select/job012/rank_model.star"
    with open(autoselect_file, "w") as f:
        f.write(
            "data_model_classes\n\nloop_\n_rlnReferenceImage\n"
            "_rlnClassScore\n_rlnClassDistribution\n"
            "_rlnAccuracyRotations\n_rlnAccuracyTranslationsAngst\n"
            "_rlnEstimatedResolution\n_rlnOverallFourierCompleteness\n"
            "_rlnClassPriorOffsetX\n_rlnClassPriorOffsetY"
        )
        f.write(
            "\n000001@Class2D/job010/run_it020_classes.mrcs "
            "0.004 0.035 3.100 1.416 16.183 1.000 -0.133 -0.001"
            "\n000002@Class2D/job010/run_it020_classes.mrcs "
            "0.008 0.035 3.100 1.416 16.183 1.000 -0.133 -0.001"
        )

    classes_file = job_dir / "Select/job012/class_averages.star"
    with open(classes_file, "w") as f:
        f.write(
            "data_\n\nloop_\n_rlnReferenceImage\n_rlnPredictedClassScore\n"
            "_rlnClassDistribution\n_rlnAccuracyRotations\n"
            "_rlnAccuracyTranslationsAngst\n_rlnEstimatedResolution\n"
        )
        f.write(
            "\n000001@Class2D/job010/run_it020_classes.mrcs "
            "0.029585 0.063949 1.500000 0.752250 9.258462"
            "\n00002@Class2D/job010/run_it020_classes.mrcs "
            "0.015038 0.029918 1.500000 0.752250 9.628800"
        )

    tmp_relion_options = RelionServiceOptions()
    tmp_relion_options.class2d_fraction_of_classes_to_remove = 0.5
    tmp_relion_options.autoselect_min_score = 0.0
    tmp_relion_options.particle_diameter = 100
    output_relion_options = dict(tmp_relion_options)

    select_test_message = {
        "input_file": f"{job_dir}/Class2D/job010/run_it020_optimiser.star",
        "combine_star_job_number": 13,
        "particle_diameter": 100,
        "class2d_fraction_of_classes_to_remove": 0.5,
        "particles_file": "particles.star",
        "classes_file": "class_averages.star",
        "python_exe": "python",
        "min_score": 0,
        "min_particles": 500,
        "class3d_batch_size": 50000,
        "class3d_max_size": 200000,
        "class_uuids": "{'1': '1', '2': '2'}",
        "relion_options": {},
    }
    return select_test_message, output_relion_options


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.select_classes.subprocess.run")
def test_select_classes_service_first_batch(
    mock_subprocess, offline_transport, tmp_path
):
    """
    Send a test message to the select classes service when it is a new job.
    This should call the 2D auto-selection and star file combiner,
    then send messages to the node creator and ask Murfey to do 3D classification.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    select_test_message, relion_options = select_classes_common_setup(
        tmp_path, initial_particle_count=0, particles_to_add=60000
    )

    # Set up the mock service and send the message to it
    service = select_classes.SelectClasses(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    service.select_classes(None, header=header, message=select_test_message)

    # Check the correct particle counts were found and split files made
    assert service.previous_total_count == 0
    assert service.total_count == 60000
    assert (tmp_path / "Select/job013/particles_split1.star").is_file()
    assert (tmp_path / "Select/job013/particles_split2.star").is_file()
    assert (tmp_path / "Select/job013/particles_batch_50000.star").is_file()

    # Check the mock calls
    assert mock_subprocess.call_count == 5
    mock_subprocess.assert_any_call(
        [
            "relion_class_ranker",
            "--opt",
            select_test_message["input_file"],
            "--o",
            "Select/job012/",
            "--auto_select",
            "--fn_root",
            "rank",
            "--do_granularity_features",
            "--fn_sel_parts",
            "particles.star",
            "--fn_sel_classavgs",
            "class_averages.star",
            "--python",
            select_test_message["python_exe"],
            "--select_min_nr_particles",
            "500",
            "--pipeline_control",
            "Select/job012/",
            "--min_score",
            "0.006",
        ],
        cwd=str(tmp_path),
        capture_output=True,
    )

    # Check that the correct messages were sent
    offline_transport.send.assert_any_call(
        "ispyb_connector",
        {
            "ispyb_command": "multipart_message",
            "ispyb_command_list": [
                {
                    "ispyb_command": "buffer",
                    "buffer_lookup": {"particle_classification_id": "1"},
                    "buffer_command": {
                        "ispyb_command": "insert_particle_classification"
                    },
                    "selected": 1,
                },
                {
                    "ispyb_command": "buffer",
                    "buffer_lookup": {"particle_classification_id": "2"},
                    "buffer_command": {
                        "ispyb_command": "insert_particle_classification"
                    },
                    "selected": 1,
                },
            ],
        },
    )
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "job_type": "relion.select.class2dauto",
            "input_file": select_test_message["input_file"],
            "output_file": f"{tmp_path}/Select/job012/particles.star",
            "relion_options": relion_options,
            "command": (
                "relion_class_ranker --opt "
                f"{tmp_path}/Class2D/job010/run_it020_optimiser.star "
                "--o Select/job012/ --auto_select --fn_root rank "
                "--do_granularity_features --fn_sel_parts particles.star "
                "--fn_sel_classavgs class_averages.star --python python "
                "--select_min_nr_particles 500 "
                "--pipeline_control Select/job012/ --min_score 0.006"
            ),
            "stdout": "stdout",
            "stderr": "stderr",
            "success": True,
        },
    )
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "alias": "Best_particles",
            "job_type": "combine_star_files_job",
            "input_file": f"{tmp_path}/Select/job012/particles.star",
            "output_file": f"{tmp_path}/Select/job013/particles_all.star",
            "relion_options": relion_options,
            "command": (
                f"combine_star_files {tmp_path}/Select/job012/particles.star "
                f"--output_dir {tmp_path}/Select/job013"
            ),
            "stdout": "",
            "stderr": "",
            "success": True,
        },
    )
    offline_transport.send.assert_any_call(
        "node_creator",
        {
            "alias": "Best_particles",
            "job_type": "combine_star_files_job",
            "input_file": f"{tmp_path}/Select/job012/particles.star",
            "output_file": f"{tmp_path}/Select/job013/particles_all.star",
            "relion_options": relion_options,
            "command": (
                f"combine_star_files {tmp_path}/Select/job013/particles_all.star "
                f"--output_dir {tmp_path}/Select/job013 --split --split_size 50000"
            ),
            "stdout": "",
            "stderr": "",
            "success": True,
        },
    )
    offline_transport.send.assert_any_call(
        "images",
        {
            "image_command": "picked_particles",
            "file": f"{tmp_path}/MotionCorr/job002/Movies/movie.mrc",
            "coordinates": [],
            "selected_coordinates": [
                [str(i / 100), str(i / 100)] for i in range(60000)
            ],
            "pixel_size": 0.5,
            "diameter": 100.0,
            "outfile": f"{tmp_path}/AutoPick/job007/STAR/movie.jpeg",
            "remove_input": False,
            "contrast_factor": 6,
        },
    )
    offline_transport.send.assert_any_call(
        "murfey_feedback",
        {
            "register": "save_class_selection_score",
            "class_selection_score": 0.006,
        },
    )
    offline_transport.send.assert_any_call(
        "murfey_feedback",
        {
            "register": "run_class3d",
            "class3d_message": {
                "particles_file": f"{tmp_path}/Select/job013/particles_batch_50000.star",
                "class3d_dir": f"{tmp_path}/Class3D/job",
                "batch_size": 50000,
            },
        },
    )
    offline_transport.send.assert_any_call(
        "murfey_feedback",
        {
            "register": "done_class_selection",
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.select_classes.subprocess.run")
def test_select_classes_service_batch_threshold(
    mock_subprocess, offline_transport, tmp_path
):
    """
    Test the service for the case where the particle count crosses a batch threshold.
    In this case particles_all.star already exists so should be appended to,
    and 3D classification should be requested.
    For this test the particle count is increased from 90000 to 110000,
    crossing the threshold of 100000.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    select_test_message, relion_options = select_classes_common_setup(
        tmp_path, initial_particle_count=90000, particles_to_add=20000
    )

    # Set up the mock service and send the message to it
    service = select_classes.SelectClasses(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    service.select_classes(None, header=header, message=select_test_message)

    # Check the correct particle counts were found and split files made
    assert service.previous_total_count == 90000
    assert service.total_count == 110000
    assert (tmp_path / "Select/job013/particles_split1.star").is_file()
    assert (tmp_path / "Select/job013/particles_split2.star").is_file()
    assert (tmp_path / "Select/job013/particles_batch_100000.star").is_file()

    # Don't bother to check the auto-selection calls here, they are checked above
    # Do check the Murfey 3D calls
    offline_transport.send.assert_any_call(
        "murfey_feedback",
        {
            "register": "run_class3d",
            "class3d_message": {
                "particles_file": f"{tmp_path}/Select/job013/particles_batch_100000.star",
                "class3d_dir": f"{tmp_path}/Class3D/job",
                "batch_size": 100000,
            },
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.select_classes.subprocess.run")
def test_select_classes_service_two_thresholds(
    mock_subprocess, offline_transport, tmp_path
):
    """
    Test the service for the case where the particle count crosses two thresholds.
    In this case particles_all.star already exists so should be appended to,
    and 3D classification should be requested.
    For this test the particle count is increased from 10000 to 110000,
    crossing the thresholds of 50000 and 100000.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    select_test_message, relion_options = select_classes_common_setup(
        tmp_path, initial_particle_count=10000, particles_to_add=100000
    )

    # Set up the mock service and send the message to it
    service = select_classes.SelectClasses(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    service.select_classes(None, header=header, message=select_test_message)

    # Check the correct particle counts were found and split files made
    assert service.previous_total_count == 10000
    assert service.total_count == 110000
    assert (tmp_path / "Select/job013/particles_split1.star").is_file()
    assert (tmp_path / "Select/job013/particles_split2.star").is_file()
    assert (tmp_path / "Select/job013/particles_batch_100000.star").is_file()

    # Don't bother to check the auto-selection calls here, they are checked above
    # Do check the Murfey 3D calls
    offline_transport.send.assert_any_call(
        "murfey_feedback",
        {
            "register": "run_class3d",
            "class3d_message": {
                "particles_file": f"{tmp_path}/Select/job013/particles_batch_100000.star",
                "class3d_dir": f"{tmp_path}/Class3D/job",
                "batch_size": 100000,
            },
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.ctffind.subprocess.run")
def test_select_classes_service_last_threshold(
    mock_subprocess, offline_transport, tmp_path
):
    """
    Test the service for the case where the particle count crosses the maximum.
    In this case particles_all.star already exists so should be appended to,
    and 3D classification should be requested.
    For this test the particle count is increased from 190000 to 260000,
    crossing the thresholds of 200000 and 250000,
    but the maximum of 200000 should be used.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    select_test_message, relion_options = select_classes_common_setup(
        tmp_path, initial_particle_count=190000, particles_to_add=70000
    )

    # Set up the mock service and send the message to it
    service = select_classes.SelectClasses(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    service.select_classes(None, header=header, message=select_test_message)

    # Check the correct particle counts were found and split files made
    assert service.previous_total_count == 190000
    assert service.total_count == 260000
    assert (tmp_path / "Select/job013/particles_split1.star").is_file()
    assert (tmp_path / "Select/job013/particles_split2.star").is_file()
    assert (tmp_path / "Select/job013/particles_batch_200000.star").is_file()

    # Don't bother to check the auto-selection calls here, they are checked above
    # Do check the Murfey 3D calls
    offline_transport.send.assert_any_call(
        "murfey_feedback",
        {
            "register": "run_class3d",
            "class3d_message": {
                "particles_file": f"{tmp_path}/Select/job013/particles_batch_200000.star",
                "class3d_dir": f"{tmp_path}/Class3D/job",
                "batch_size": 200000,
            },
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.select_classes.subprocess.run")
def test_select_classes_service_not_threshold(
    mock_subprocess, offline_transport, tmp_path
):
    """
    Test the service for the case where the particle count doesn't cross a threshold.
    In this case particles_all.star already exists so should be appended to,
    but 3D classification should not be requested.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    select_test_message, relion_options = select_classes_common_setup(
        tmp_path, initial_particle_count=110000, particles_to_add=20000
    )

    # Set up the mock service and send the message to it
    service = select_classes.SelectClasses(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    service.select_classes(None, header=header, message=select_test_message)

    # Check the correct particle counts were found and split files made
    assert service.previous_total_count == 110000
    assert service.total_count == 130000
    assert (tmp_path / "Select/job013/particles_split1.star").is_file()
    assert not (tmp_path / "Select/job013/particles_split2.star").is_file()
    assert len(list(tmp_path.glob("Select/job013/particles_batch_*"))) == 0

    # Don't bother to check the auto-selection calls here, they are checked above
    # Do check the Murfey 3D calls
    assert len(offline_transport.send.call_args_list) == 7


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.select_classes.subprocess.run")
def test_select_classes_service_past_maximum(
    mock_subprocess, offline_transport, tmp_path
):
    """
    Test the service for the case where the existing particle count exceeds the maximum.
    In this case particles_all.star already exists so should be appended to,
    but 3D classification should not be requested.
    """
    mock_subprocess().returncode = 0
    mock_subprocess().stdout = "stdout".encode("ascii")
    mock_subprocess().stderr = "stderr".encode("ascii")

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    select_test_message, relion_options = select_classes_common_setup(
        tmp_path, initial_particle_count=290000, particles_to_add=20000
    )

    # Set up the mock service and send the message to it
    service = select_classes.SelectClasses(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()
    service.select_classes(None, header=header, message=select_test_message)

    # Check the correct particle counts were found and split files made
    assert service.previous_total_count == 290000
    assert service.total_count == 310000
    assert (tmp_path / "Select/job013/particles_split1.star").is_file()
    assert (tmp_path / "Select/job013/particles_split2.star").is_file()
    assert len(list(tmp_path.glob("Select/job013/particles_batch_*"))) == 0

    # Don't bother to check the auto-selection calls here, they are checked above
    # Do check the Murfey 3D calls
    assert len(offline_transport.send.call_args_list) == 7


def test_parse_combiner_output(offline_transport):
    """
    Send test lines to the output parser
    to check the number of particles are being read in
    """
    service = select_classes.SelectClasses(
        environment={"queue": ""}, transport=offline_transport
    )
    service.initializing()

    select_classes.SelectClasses.parse_combiner_output(
        service, "Adding Select/job/particles_all.star with 10 particles"
    )
    select_classes.SelectClasses.parse_combiner_output(
        service, "Combined 2 files into particles_all.star with 20 particles"
    )
    assert service.previous_total_count == 10
    assert service.total_count == 20
