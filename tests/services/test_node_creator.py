from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest
from gemmi import cif
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.util.relion_service_options import RelionServiceOptions

node_creator = pytest.importorskip(
    "cryoemservices.services.node_creator",
    reason="these tests require the ccpem pipeliner",
)


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


def setup_and_run_node_creation(
    relion_options: RelionServiceOptions,
    transport: OfflineTransport,
    project_dir: Path,
    job_dir: str,
    job_type: str,
    input_file: str,
    output_file: Path,
    skip_short_pipeline: bool = False,
    results: dict = {},
    experiment_type: str = "spa",
):
    """
    Run the node creation for any job and check the pipeline files are produced
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }

    # Write a dummy pipeline file expected by cryolo
    with open(project_dir / "default_pipeline.star", "w") as f:
        f.write("data_pipeline_general\n\n_rlnPipeLineJobCounter  1")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.touch()
    test_message = {
        "job_type": job_type,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "relion_options": relion_options,
        "command": "command",
        "stdout": "stdout",
        "stderr": "stderr",
        "results": results,
        "experiment_type": experiment_type,
    }

    # set up the mock service and send the message to it
    service = node_creator.NodeCreator()
    service.transport = transport
    service.start()
    service.node_creator(None, header=header, message=test_message)

    # Check that the correct general pipeline files have been made
    assert (project_dir / f"{job_type.replace('.', '_')}_job.star").exists()
    assert (project_dir / f".gui_{job_type.replace('.', '_')}job.star").exists()
    assert (project_dir / ".Nodes").is_dir()

    assert (project_dir / job_dir / "job.star").exists()
    assert (project_dir / job_dir / "note.txt").exists()
    assert (project_dir / job_dir / "run.out").exists()
    assert (project_dir / job_dir / "run.err").exists()
    assert (project_dir / job_dir / "run.job").exists()
    assert (project_dir / job_dir / "continue_job.star").exists()
    assert (project_dir / job_dir / "PIPELINER_JOB_EXIT_SUCCESS").exists()
    # assert (project_dir / job_dir / "job_metadata.json").exists()
    assert (project_dir / job_dir / "default_pipeline.star").exists()
    assert (project_dir / job_dir / ".CCPEM_pipeliner_jobinfo").exists()
    if experiment_type == "spa" and not skip_short_pipeline:
        assert (project_dir / "short_pipeline.star").exists()


# General tests
@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_failed_job(offline_transport, tmp_path):
    """
    Use motion correction to test that the node creator works for failed commands.
    This should set up the general pipeliner parts, and add a failure file to the job.
    """
    job_dir = "MotionCorr/job002"
    input_file = tmp_path / "Import/job001/Movies/sample.mrc"
    output_file = tmp_path / job_dir / "Movies/sample.mrc"
    relion_options = RelionServiceOptions()

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.touch()
    test_message = {
        "job_type": "relion.motioncorr.motioncor2",
        "input_file": str(input_file),
        "output_file": str(output_file),
        "relion_options": relion_options,
        "command": "command",
        "stdout": "stdout",
        "stderr": "stderr",
        "success": False,
    }

    # set up the mock service and send the message to it
    service = node_creator.NodeCreator()
    service.transport = offline_transport
    service.start()
    service.node_creator(None, header=header, message=test_message)

    # Check that the correct general pipeline files have been made
    assert (tmp_path / "relion_motioncorr_motioncor2_job.star").exists()
    assert (tmp_path / ".gui_projectdir").exists()
    assert (tmp_path / job_dir / "job.star").exists()
    assert (tmp_path / job_dir / "note.txt").exists()
    assert (tmp_path / job_dir / "run.out").exists()
    assert (tmp_path / job_dir / "run.err").exists()
    assert (tmp_path / job_dir / "run.job").exists()
    assert (tmp_path / job_dir / "continue_job.star").exists()
    assert (tmp_path / job_dir / "PIPELINER_JOB_EXIT_FAILED").exists()
    assert (tmp_path / job_dir / "default_pipeline.star").exists()
    assert (tmp_path / job_dir / ".CCPEM_pipeliner_jobinfo").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_rerun_job(offline_transport, tmp_path):
    """
    Use motion correction to test that the node creator works for failed commands.
    This should set up the general pipeliner parts, and add a failure file to the job.
    """
    job_dir = "MotionCorr/job002"
    input_file = tmp_path / "Import/job001/Movies/sample.mrc"
    output_file = tmp_path / job_dir / "Movies/sample.mrc"
    relion_options = RelionServiceOptions()

    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.touch()
    (tmp_path / job_dir / "PIPELINER_JOB_EXIT_SUCCESS").touch()
    test_message = {
        "job_type": "relion.motioncorr.motioncor2",
        "input_file": str(input_file),
        "output_file": str(output_file),
        "relion_options": relion_options,
        "command": "command",
        "stdout": "stdout",
        "stderr": "stderr",
        "success": True,
        "results": {"total_motion": "10", "early_motion": "4", "late_motion": "6"},
    }

    # set up the mock service and send the message to it
    service = node_creator.NodeCreator()
    service.transport = offline_transport
    service.start()
    service.node_creator(None, header=header, message=test_message)

    # Check that the correct general pipeline files have been made
    assert (tmp_path / "relion_motioncorr_motioncor2_job.star").exists()
    assert (tmp_path / ".gui_projectdir").exists()
    assert (tmp_path / job_dir / "job.star").exists()
    assert (tmp_path / job_dir / "note.txt").exists()
    assert (tmp_path / job_dir / "run.out").exists()
    assert (tmp_path / job_dir / "run.err").exists()
    assert not (tmp_path / job_dir / "run.job").exists()
    assert not (tmp_path / job_dir / "continue_job.star").exists()
    assert (tmp_path / job_dir / "PIPELINER_JOB_EXIT_SUCCESS").exists()
    assert not (tmp_path / job_dir / "default_pipeline.star").exists()
    assert not (tmp_path / job_dir / ".CCPEM_pipeliner_jobinfo").exists()


# SPA tests
@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_import(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.import.movies
    """
    job_dir = "Import/job001"
    input_file = f"{tmp_path}/Movies/sample.mrc"
    output_file = tmp_path / job_dir / "Movies/sample.mrc"
    relion_options = RelionServiceOptions()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.import.movies",
        input_file,
        output_file,
    )

    # Check the output file structure
    assert (tmp_path / job_dir / "movies.star").exists()
    micrographs_file = cif.read_file(str(tmp_path / job_dir / "movies.star"))

    micrographs_optics = micrographs_file.find_block("optics")
    assert list(micrographs_optics.find_loop("_rlnOpticsGroupName")) == ["opticsGroup1"]
    assert list(micrographs_optics.find_loop("_rlnOpticsGroup")) == ["1"]
    assert list(micrographs_optics.find_loop("_rlnMicrographOriginalPixelSize")) == [
        str(relion_options.pixel_size)
    ]
    assert list(micrographs_optics.find_loop("_rlnVoltage")) == [
        str(relion_options.voltage)
    ]
    assert list(micrographs_optics.find_loop("_rlnSphericalAberration")) == [
        str(relion_options.spher_aber)
    ]
    assert list(micrographs_optics.find_loop("_rlnAmplitudeContrast")) == [
        str(relion_options.ampl_contrast)
    ]
    assert list(micrographs_optics.find_loop("_rlnMicrographPixelSize")) == [
        str(relion_options.pixel_size)
    ]

    micrographs_data = micrographs_file.find_block("movies")
    assert list(micrographs_data.find_loop("_rlnMicrographMovieName")) == [
        "Import/job001/Movies/sample.mrc"
    ]
    assert list(micrographs_data.find_loop("_rlnOpticsGroup")) == ["1"]


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_motioncorr(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.motioncorr.motioncor2
    """
    job_dir = "MotionCorr/job002"
    input_file = f"{tmp_path}/Import/job001/Movies/sample.mrc"
    output_file = tmp_path / job_dir / "Movies/sample.mrc"
    relion_options = RelionServiceOptions()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.motioncorr.motioncor2",
        input_file,
        output_file,
        results={"total_motion": "10", "early_motion": "4", "late_motion": "6"},
    )

    # Check the output file structure
    assert (tmp_path / job_dir / "corrected_micrographs.star").exists()
    micrographs_file = cif.read_file(
        str(tmp_path / job_dir / "corrected_micrographs.star")
    )

    micrographs_optics = micrographs_file.find_block("optics")
    assert list(micrographs_optics.find_loop("_rlnOpticsGroupName")) == ["opticsGroup1"]
    assert list(micrographs_optics.find_loop("_rlnOpticsGroup")) == ["1"]
    assert list(micrographs_optics.find_loop("_rlnMicrographOriginalPixelSize")) == [
        str(relion_options.pixel_size)
    ]
    assert list(micrographs_optics.find_loop("_rlnVoltage")) == [
        str(relion_options.voltage)
    ]
    assert list(micrographs_optics.find_loop("_rlnSphericalAberration")) == [
        str(relion_options.spher_aber)
    ]
    assert list(micrographs_optics.find_loop("_rlnAmplitudeContrast")) == [
        str(relion_options.ampl_contrast)
    ]
    assert list(micrographs_optics.find_loop("_rlnMicrographPixelSize")) == [
        str(relion_options.pixel_size)
    ]

    micrographs_data = micrographs_file.find_block("micrographs")
    assert list(micrographs_data.find_loop("_rlnMicrographName")) == [
        "MotionCorr/job002/Movies/sample.mrc"
    ]
    assert list(micrographs_data.find_loop("_rlnMicrographMetadata")) == [
        "MotionCorr/job002/Movies/sample.star"
    ]
    assert list(micrographs_data.find_loop("_rlnOpticsGroup")) == ["1"]
    assert list(micrographs_data.find_loop("_rlnAccumMotionTotal")) == ["10"]
    assert list(micrographs_data.find_loop("_rlnAccumMotionEarly")) == ["4"]
    assert list(micrographs_data.find_loop("_rlnAccumMotionLate")) == ["6"]


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_icebreaker_micrographs(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    icebreaker.micrograph_analysis.micrographs
    """
    job_dir = "IceBreaker/job003"
    input_file = f"{tmp_path}/MotionCorr/job002/Movies/sample.mrc"
    output_file = tmp_path / job_dir
    output_file.mkdir(parents=True)
    relion_options = RelionServiceOptions()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "icebreaker.micrograph_analysis.micrographs",
        input_file,
        output_file,
        results={
            "icebreaker_type": "micrographs",
            "total_motion": 10,
            "early_motion": 2,
            "late_motion": 8,
        },
    )

    # Check the output file structure
    assert (tmp_path / job_dir / "grouped_micrographs.star").exists()
    micrographs_file = cif.read_file(
        str(tmp_path / job_dir / "grouped_micrographs.star")
    )

    micrographs_data = micrographs_file.find_block("micrographs")
    assert list(micrographs_data.find_loop("_rlnMicrographName")) == [
        "IceBreaker/job003/Movies/sample_grouped.mrc"
    ]
    assert list(micrographs_data.find_loop("_rlnMicrographMetadata")) == [
        "MotionCorr/job002/Movies/sample.star"
    ]
    assert list(micrographs_data.find_loop("_rlnOpticsGroup")) == ["1"]
    assert list(micrographs_data.find_loop("_rlnAccumMotionTotal")) == ["10"]
    assert list(micrographs_data.find_loop("_rlnAccumMotionEarly")) == ["2"]
    assert list(micrographs_data.find_loop("_rlnAccumMotionLate")) == ["8"]


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_icebreaker_enhancecontrast(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    icebreaker.micrograph_analysis.enhancecontrast
    """
    job_dir = "IceBreaker/job004"
    input_file = f"{tmp_path}/MotionCorr/job002/Movies/sample.mrc"
    output_file = tmp_path / job_dir
    output_file.mkdir(parents=True)
    relion_options = RelionServiceOptions()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "icebreaker.micrograph_analysis.enhancecontrast",
        input_file,
        output_file,
        results={
            "icebreaker_type": "enhancecontrast",
            "total_motion": 10,
            "early_motion": 2,
            "late_motion": 8,
        },
    )

    # Check the output file structure
    assert (tmp_path / job_dir / "flattened_micrographs.star").exists()
    micrographs_file = cif.read_file(
        str(tmp_path / job_dir / "flattened_micrographs.star")
    )

    micrographs_data = micrographs_file.find_block("micrographs")
    assert list(micrographs_data.find_loop("_rlnMicrographName")) == [
        "IceBreaker/job004/Movies/sample_flattened.mrc"
    ]
    assert list(micrographs_data.find_loop("_rlnMicrographMetadata")) == [
        "MotionCorr/job002/Movies/sample.star"
    ]
    assert list(micrographs_data.find_loop("_rlnOpticsGroup")) == ["1"]
    assert list(micrographs_data.find_loop("_rlnAccumMotionTotal")) == ["10"]
    assert list(micrographs_data.find_loop("_rlnAccumMotionEarly")) == ["2"]
    assert list(micrographs_data.find_loop("_rlnAccumMotionLate")) == ["8"]


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_icebreaker_summary(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    icebreaker.micrograph_analysis.summary
    """
    job_dir = "IceBreaker/job005"
    input_file = f"{tmp_path}/IceBreaker/job003/Movies/sample.mrc"
    output_file = tmp_path / job_dir
    output_file.mkdir(parents=True)
    relion_options = RelionServiceOptions()

    (output_file / "five_figs_test.csv").touch()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "icebreaker.micrograph_analysis.summary",
        input_file,
        output_file,
        results={
            "icebreaker_type": "summary",
            "total_motion": "10",
            "summary": ["0", "1", "2", "3", "4"],
        },
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_ctffind(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.ctffind.ctffind4
    """
    job_dir = "CtfFind/job006"
    input_file = f"{tmp_path}/MotionCorr/job002/Movies/sample.mrc"
    output_file = tmp_path / job_dir / "Movies/sample.ctf"
    relion_options = RelionServiceOptions()

    output_file.parent.mkdir(parents=True)
    with open(output_file.with_suffix(".txt"), "w") as f:
        f.write("0.0 1.0 2.0 3.0 4.0 5.0 6.0")
    with open(f"{output_file.with_suffix('')}_avrot.txt", "w") as f:
        f.write(
            "header\nheader\nheader\nheader\nheader\n0.24 0.26 0.27 0.29\n1 2 3 4\n"
        )

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.ctffind.ctffind4",
        input_file,
        output_file,
    )

    # Check the output file structure
    assert (tmp_path / job_dir / "micrographs_ctf.star").exists()
    micrographs_file = cif.read_file(str(tmp_path / job_dir / "micrographs_ctf.star"))

    micrographs_optics = micrographs_file.find_block("optics")
    assert list(micrographs_optics.find_loop("_rlnOpticsGroupName")) == ["opticsGroup1"]
    assert list(micrographs_optics.find_loop("_rlnOpticsGroup")) == ["1"]
    assert list(micrographs_optics.find_loop("_rlnMicrographOriginalPixelSize")) == [
        str(relion_options.pixel_size)
    ]
    assert list(micrographs_optics.find_loop("_rlnVoltage")) == [
        str(relion_options.voltage)
    ]
    assert list(micrographs_optics.find_loop("_rlnSphericalAberration")) == [
        str(relion_options.spher_aber)
    ]
    assert list(micrographs_optics.find_loop("_rlnAmplitudeContrast")) == [
        str(relion_options.ampl_contrast)
    ]
    assert list(micrographs_optics.find_loop("_rlnMicrographPixelSize")) == [
        str(relion_options.pixel_size)
    ]

    micrographs_data = micrographs_file.find_block("micrographs")
    assert list(micrographs_data.find_loop("_rlnMicrographName")) == [
        "MotionCorr/job002/Movies/sample.mrc"
    ]
    assert list(micrographs_data.find_loop("_rlnOpticsGroup")) == ["1"]
    assert list(micrographs_data.find_loop("_rlnCtfImage")) == [
        "CtfFind/job006/Movies/sample.ctf:mrc"
    ]
    assert list(micrographs_data.find_loop("_rlnDefocusU")) == ["1.0"]
    assert list(micrographs_data.find_loop("_rlnDefocusV")) == ["2.0"]
    assert list(micrographs_data.find_loop("_rlnCtfAstigmatism")) == ["1.0"]
    assert list(micrographs_data.find_loop("_rlnDefocusAngle")) == ["3.0"]
    assert list(micrographs_data.find_loop("_rlnCtfFigureOfMerit")) == ["5.0"]
    assert list(micrographs_data.find_loop("_rlnCtfMaxResolution")) == ["6.0"]
    assert list(micrographs_data.find_loop("_rlnCtfIceRingDensity")) == ["5.0"]


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_cryolo(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    cryolo.autopick
    """
    job_dir = "AutoPick/job007"
    input_file = f"{tmp_path}/MotionCorr/job002/Movies/sample.mrc"
    output_file = tmp_path / job_dir / "STAR/sample.star"
    relion_options = RelionServiceOptions()

    relion_options.cryolo_config_file = str(tmp_path / "cryolo_config.json")
    (tmp_path / "cryolo_config.json").touch()

    (tmp_path / "MotionCorr/job002/").mkdir(parents=True)
    (tmp_path / "MotionCorr/job002/corrected_micrographs.star").touch()

    (tmp_path / job_dir / "DISTR").mkdir(parents=True)
    (tmp_path / job_dir / "DISTR/confidence_distribution_summary_1.txt").touch()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "cryolo.autopick",
        input_file,
        output_file,
    )

    # Check the output file structure
    assert (tmp_path / job_dir / "autopick.star").exists()
    micrographs_file = cif.read_file(str(tmp_path / job_dir / "autopick.star"))

    micrographs_data = micrographs_file.find_block("coordinate_files")
    assert list(micrographs_data.find_loop("_rlnMicrographName")) == [
        "MotionCorr/job002/Movies/sample.mrc"
    ]
    assert list(micrographs_data.find_loop("_rlnMicrographCoordinates")) == [
        "AutoPick/job007/STAR/sample.star"
    ]


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_extract(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.extract
    """
    job_dir = "Extract/job008"
    input_file = (
        f"{tmp_path}/AutoPick/job007/STAR/sample.star"
        f":{tmp_path}/CtfFind/job006/Movies/sample.ctf"
    )
    output_file = tmp_path / job_dir / "Movies/sample.star"
    relion_options = RelionServiceOptions()

    output_file.parent.mkdir(parents=True)
    with open(output_file, "w") as f:
        f.write("data_particles\n\nloop_\n_rlnCoordinateX\n_rlnCoordinateY\n1.0 2.0")

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.extract",
        input_file,
        output_file,
        results={"box_size": 64},
    )

    # Check the output file structure
    assert (tmp_path / job_dir / "particles.star").exists()
    micrographs_file = cif.read_file(str(tmp_path / job_dir / "particles.star"))

    micrographs_optics = micrographs_file.find_block("optics")
    assert list(micrographs_optics.find_loop("_rlnOpticsGroupName")) == ["opticsGroup1"]
    assert list(micrographs_optics.find_loop("_rlnOpticsGroup")) == ["1"]
    assert list(micrographs_optics.find_loop("_rlnMicrographOriginalPixelSize")) == [
        str(relion_options.pixel_size)
    ]
    assert list(micrographs_optics.find_loop("_rlnVoltage")) == [
        str(relion_options.voltage)
    ]
    assert list(micrographs_optics.find_loop("_rlnSphericalAberration")) == [
        str(relion_options.spher_aber)
    ]
    assert list(micrographs_optics.find_loop("_rlnAmplitudeContrast")) == [
        str(relion_options.ampl_contrast)
    ]
    assert list(micrographs_optics.find_loop("_rlnImagePixelSize")) == [
        str(relion_options.pixel_size)
    ]
    assert list(micrographs_optics.find_loop("_rlnImageSize")) == ["64"]
    assert list(micrographs_optics.find_loop("_rlnImageDimensionality")) == ["2"]
    assert list(micrographs_optics.find_loop("_rlnCtfDataAreCtfPremultiplied")) == ["0"]

    micrographs_data = micrographs_file.find_block("particles")
    assert list(micrographs_data.find_loop("_rlnCoordinateX")) == ["1.0"]
    assert list(micrographs_data.find_loop("_rlnCoordinateY")) == ["2.0"]


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_select_particles(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.select.split
    """
    job_dir = "Select/job009"
    input_file = f"{tmp_path}/Extract/job007/Movies/sample.star"
    output_file = tmp_path / job_dir / "particles_split2.star"
    relion_options = RelionServiceOptions()

    (tmp_path / job_dir).mkdir(parents=True)
    (tmp_path / job_dir / "particles_split1.star").touch()
    (tmp_path / job_dir / "particles_split2.star").touch()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.select.split",
        input_file,
        output_file,
    )

    # Check the output file structure
    assert (
        tmp_path / ".Nodes/ParticleGroupMetadata/Select/job009/particles_split1.star"
    ).exists()
    assert (
        tmp_path / ".Nodes/ParticleGroupMetadata/Select/job009/particles_split2.star"
    ).exists()


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_icebreaker_particles(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    icebreaker.micrograph_analysis.particles
    """
    job_dir = "IceBreaker/job011"
    input_file = (
        f"{tmp_path}/IceBreaker/job003/grouped_micrographs.star"
        f":{tmp_path}/Select/job009/particles.star"
    )
    output_file = tmp_path / job_dir
    output_file.mkdir(parents=True)
    relion_options = RelionServiceOptions()

    # .Nodes directory doesn't get made by this job
    (tmp_path / ".Nodes").mkdir()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "icebreaker.micrograph_analysis.particles",
        input_file,
        output_file,
        results={
            "icebreaker_type": "particles",
        },
    )

    # Check the output file structure
    assert not (tmp_path / job_dir / "done_mics.txt").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_class2d_em(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.class2d.em
    """
    job_dir = "Class2D/job010"
    input_file = f"{tmp_path}/Select/job009/particles.star"
    output_file = tmp_path / job_dir
    output_file.mkdir(parents=True)
    relion_options = RelionServiceOptions()

    # .Nodes directory doesn't get made by this job
    (tmp_path / ".Nodes").mkdir()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.class2d.em",
        input_file,
        output_file,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_class2d_vdam(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.class2d.vdam
    """
    job_dir = "Class2D/job010"
    input_file = f"{tmp_path}/Select/job009/particles.star"
    output_file = tmp_path / job_dir
    output_file.mkdir(parents=True)
    relion_options = RelionServiceOptions()

    # .Nodes directory doesn't get made by this job
    (tmp_path / ".Nodes").mkdir()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.class2d.vdam",
        input_file,
        output_file,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_select_class(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.select.class2dauto
    """
    job_dir = "Select/job012"
    input_file = f"{tmp_path}/Class2D/job010/optimiser.star"
    output_file = tmp_path / job_dir / "particles.star"
    output_file.parent.mkdir(parents=True)
    relion_options = RelionServiceOptions()

    # .Nodes directory doesn't get made by this job
    (tmp_path / ".Nodes").mkdir()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.select.class2dauto",
        input_file,
        output_file,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_split_star(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    combine_star_files_job
    """
    job_dir = "Select/job013"
    input_file = f"{tmp_path}/Select/job012/particles.star"
    output_file = tmp_path / job_dir / "particles_all.star"
    output_file.parent.mkdir(parents=True)
    relion_options = RelionServiceOptions()

    with open(tmp_path / job_dir / "class_averages.star", "w") as class_file:
        class_file.write(
            "data_\nloop_\nrlnReferenceImage\n"
            "01@Class2D/job010/run_it025_classes.mrcs"
            "06@Class2D/job010/run_it025_classes.mrcs"
        )
    with open(tmp_path / job_dir / "particles_all.star", "w") as particles_file:
        particles_file.write(
            "data_optics\nloop_\n_rlnOpticsGroupName\nopticsGroup1 1\n\n"
            "data_particles\nloop_\n_rlnImageName\n"
            "000047@image1.mrcs\n000048@image1.mrcs\n"
        )

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "combine_star_files_job",
        input_file,
        output_file,
    )

    # Check the output file structure
    assert (tmp_path / job_dir / ".results_display000_pending.json").is_file()
    assert (tmp_path / job_dir / ".results_display001_pending.json").is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_initial_model(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.initialmodel
    """
    job_dir = "InitialModel/job014"
    input_file = f"{tmp_path}/Select/job013/particles.star"
    output_file = tmp_path / job_dir / "initial_model.star"
    output_file.parent.mkdir(parents=True)
    relion_options = RelionServiceOptions()

    # .Nodes directory doesn't get made by this job
    (tmp_path / ".Nodes").mkdir()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.initialmodel",
        input_file,
        output_file,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_class3d(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.class3d
    """
    job_dir = "Class3D/job015"
    input_file = (
        f"{tmp_path}/Select/job013/particles.star"
        + f":{tmp_path}/InitialModel/job014/initial_model.star"
    )
    output_file = tmp_path / job_dir
    output_file.mkdir(parents=True)
    relion_options = RelionServiceOptions()

    # .Nodes directory doesn't get made by this job
    (tmp_path / ".Nodes").mkdir()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.class3d",
        input_file,
        output_file,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_select_value(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.select.onvalue
    """
    job_dir = "Select/job019"
    input_file = f"{tmp_path}/Class3D/job015/optimiser.star"
    output_file = tmp_path / job_dir / "particles.star"
    output_file.parent.mkdir(parents=True)
    relion_options = RelionServiceOptions()

    # .Nodes directory doesn't get made by this job
    (tmp_path / ".Nodes").mkdir()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.select.onvalue",
        input_file,
        output_file,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_refine3d(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.refine3d
    """
    job_dir = "Refine3D/job021"
    input_file = (
        f"{tmp_path}/Extract/job020/particles.star"
        + f":{tmp_path}/Extract/job020/ref.mrc"
    )
    output_file = tmp_path / job_dir
    output_file.mkdir(parents=True)
    relion_options = RelionServiceOptions()

    # .Nodes directory doesn't get made by this job
    (tmp_path / ".Nodes").mkdir()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.refine3d",
        input_file,
        output_file,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_maskcreate(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.maskcreate
    """
    job_dir = "MaskCreate/job022"
    input_file = f"{tmp_path}/Refine3D/job021/run_class001.mrc"
    output_file = tmp_path / job_dir / "mask.mrc"
    output_file.parent.mkdir(parents=True)
    relion_options = RelionServiceOptions()

    # .Nodes directory doesn't get made by this job
    (tmp_path / ".Nodes").mkdir()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.maskcreate",
        input_file,
        output_file,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_postprocess(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.postprocess
    """
    job_dir = "PostProcess/job023"
    input_file = (
        f"{tmp_path}/Refine3D/job021/half_map.mrc"
        + f":{tmp_path}/MaskCreate/job022/mask.mrc"
    )
    output_file = tmp_path / job_dir
    output_file.mkdir(parents=True)
    relion_options = RelionServiceOptions()

    # .Nodes directory doesn't get made by this job
    (tmp_path / ".Nodes").mkdir()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.postprocess",
        input_file,
        output_file,
    )


# Tomography tests
@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_import_tomo(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.importtomo
    """
    job_dir = "Import/job001"
    input_file = (
        f"{tmp_path}/Movies/Position_1_2_001_0.00_fractions.tiff"
        + f":{tmp_path}/Movies/*.mdoc"
    )
    output_file = tmp_path / job_dir / "Movies/Position_1_2_001_1.50_fractions.tiff"
    relion_options = RelionServiceOptions()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.importtomo",
        input_file,
        output_file,
        experiment_type="tomography",
    )

    # Check the output file structure
    assert (tmp_path / job_dir / "tilt_series.star").exists()
    tilt_series_file = cif.read_file(str(tmp_path / job_dir / "tilt_series.star"))

    global_block = tilt_series_file.find_block("global")
    assert list(global_block.find_loop("_rlnTomoName")) == ["Position_1_2"]
    assert list(global_block.find_loop("_rlnTomoTiltSeriesStarFile")) == [
        f"{job_dir}/tilt_series/Position_1_2.star"
    ]
    assert list(global_block.find_loop("_rlnVoltage")) == [str(relion_options.voltage)]
    assert list(global_block.find_loop("_rlnSphericalAberration")) == [
        str(relion_options.spher_aber)
    ]
    assert list(global_block.find_loop("_rlnAmplitudeContrast")) == [
        str(relion_options.ampl_contrast)
    ]
    assert list(global_block.find_loop("_rlnMicrographOriginalPixelSize")) == [
        str(relion_options.pixel_size)
    ]
    assert list(global_block.find_loop("_rlnTomoHand")) == [
        str(relion_options.invert_hand)
    ]
    assert list(global_block.find_loop("_rlnOpticsGroupName")) == ["optics1"]
    assert list(global_block.find_loop("_rlnTomoTiltSeriesPixelSize")) == [
        str(relion_options.pixel_size)
    ]

    assert (tmp_path / job_dir / "tilt_series/Position_1_2.star").exists()
    tilts_file = cif.read_file(
        str(tmp_path / job_dir / "tilt_series/Position_1_2.star")
    )

    tilts_block = tilts_file.sole_block()
    assert list(tilts_block.find_loop("_rlnMicrographMovieName")) == [
        f"{job_dir}/Movies/Position_1_2_001_1.50_fractions.tiff"
    ]
    assert list(tilts_block.find_loop("_rlnTomoTiltMovieFrameCount")) == [
        str(relion_options.frame_count)
    ]
    assert list(tilts_block.find_loop("_rlnTomoNominalStageTiltAngle")) == ["1.50"]
    assert list(tilts_block.find_loop("_rlnTomoNominalTiltAxisAngle")) == [
        str(relion_options.tilt_axis_angle)
    ]
    assert list(tilts_block.find_loop("_rlnMicrographPreExposure")) == ["12.77"]
    assert list(tilts_block.find_loop("_rlnTomoNominalDefocus")) == [
        str(relion_options.defocus)
    ]


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_motioncorr_tomo(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.motioncorr.own
    """
    job_dir = "MotionCorr/job002"
    input_file = f"{tmp_path}/Import/job001/Movies/Position_1_2_001_1.50_fractions.tiff"
    output_file = tmp_path / job_dir / "Movies/Position_1_2_001_1.50_fractions.mrc"
    relion_options = RelionServiceOptions()

    # .Nodes directory doesn't get made by this job
    (tmp_path / ".Nodes").mkdir()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.motioncorr.own",
        input_file,
        output_file,
        results={"total_motion": "10", "early_motion": "4", "late_motion": "6"},
        experiment_type="tomography",
    )

    # Check the output file structure
    assert (tmp_path / job_dir / "corrected_tilt_series.star").exists()
    tilt_series_file = cif.read_file(
        str(tmp_path / job_dir / "corrected_tilt_series.star")
    )

    global_block = tilt_series_file.find_block("global")
    assert list(global_block.find_loop("_rlnTomoName")) == ["Position_1_2"]
    assert list(global_block.find_loop("_rlnTomoTiltSeriesStarFile")) == [
        f"{job_dir}/tilt_series/Position_1_2.star"
    ]

    assert (tmp_path / job_dir / "tilt_series/Position_1_2.star").exists()
    tilts_file = cif.read_file(
        str(tmp_path / job_dir / "tilt_series/Position_1_2.star")
    )

    tilts_block = tilts_file.sole_block()
    assert list(tilts_block.find_loop("_rlnMicrographMovieName")) == [
        "Import/job001/Movies/Position_1_2_001_1.50_fractions.tiff"
    ]
    assert list(tilts_block.find_loop("_rlnTomoTiltMovieFrameCount")) == [
        str(relion_options.frame_count)
    ]
    assert list(tilts_block.find_loop("_rlnTomoNominalStageTiltAngle")) == ["1.50"]
    assert list(tilts_block.find_loop("_rlnTomoNominalTiltAxisAngle")) == [
        str(relion_options.tilt_axis_angle)
    ]
    assert list(tilts_block.find_loop("_rlnMicrographPreExposure")) == ["12.77"]
    assert list(tilts_block.find_loop("_rlnTomoNominalDefocus")) == [
        str(relion_options.defocus)
    ]
    assert list(tilts_block.find_loop("_rlnMicrographName")) == [
        f"{job_dir}/Movies/Position_1_2_001_1.50_fractions.mrc"
    ]
    assert list(tilts_block.find_loop("_rlnMicrographMetadata")) == [
        f"{job_dir}/Movies/Position_1_2_001_1.50_fractions.star"
    ]
    assert list(tilts_block.find_loop("_rlnAccumMotionTotal")) == ["10"]
    assert list(tilts_block.find_loop("_rlnAccumMotionEarly")) == ["4"]
    assert list(tilts_block.find_loop("_rlnAccumMotionLate")) == ["6"]


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_ctffind_tomo(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.ctffind.ctffind4
    """
    job_dir = "CtfFind/job003"
    input_file = (
        f"{tmp_path}/MotionCorr/job002/Movies/Position_1_2_001_-1.50_fractions.mrc"
    )
    output_file = tmp_path / job_dir / "Movies/Position_1_2_001_-1.50_fractions.ctf"
    relion_options = RelionServiceOptions()

    output_file.parent.mkdir(parents=True)
    with open(output_file.with_suffix(".txt"), "w") as f:
        f.write("0.0 1.0 2.0 3.0 4.0 5.0 6.0")
    with open(f"{output_file.with_suffix('')}_avrot.txt", "w") as f:
        f.write(
            "header\nheader\nheader\nheader\nheader\n0.24 0.26 0.27 0.29\n1 2 3 4\n"
        )

    # .Nodes directory doesn't get made by this job
    (tmp_path / ".Nodes").mkdir()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.ctffind.ctffind4",
        input_file,
        output_file,
        experiment_type="tomography",
    )

    # Check the output file structure
    assert (tmp_path / job_dir / "tilt_series_ctf.star").exists()
    tilt_series_file = cif.read_file(str(tmp_path / job_dir / "tilt_series_ctf.star"))

    global_block = tilt_series_file.find_block("global")
    assert list(global_block.find_loop("_rlnTomoName")) == ["Position_1_2"]
    assert list(global_block.find_loop("_rlnTomoTiltSeriesStarFile")) == [
        f"{job_dir}/tilt_series/Position_1_2.star"
    ]

    assert (tmp_path / job_dir / "tilt_series/Position_1_2.star").exists()
    tilts_file = cif.read_file(
        str(tmp_path / job_dir / "tilt_series/Position_1_2.star")
    )

    tilts_block = tilts_file.sole_block()
    assert list(tilts_block.find_loop("_rlnTomoTiltMovieFrameCount")) == [
        str(relion_options.frame_count)
    ]
    assert list(tilts_block.find_loop("_rlnTomoNominalStageTiltAngle")) == ["-1.50"]
    assert list(tilts_block.find_loop("_rlnTomoNominalTiltAxisAngle")) == [
        str(relion_options.tilt_axis_angle)
    ]
    assert list(tilts_block.find_loop("_rlnMicrographPreExposure")) == ["12.77"]
    assert list(tilts_block.find_loop("_rlnTomoNominalDefocus")) == [
        str(relion_options.defocus)
    ]
    assert list(tilts_block.find_loop("_rlnMicrographName")) == [
        "MotionCorr/job002/Movies/Position_1_2_001_-1.50_fractions.mrc"
    ]
    assert list(tilts_block.find_loop("_rlnCtfImage")) == [
        f"{job_dir}/Movies/Position_1_2_001_-1.50_fractions.ctf:mrc"
    ]
    assert list(tilts_block.find_loop("_rlnDefocusU")) == ["1.0"]
    assert list(tilts_block.find_loop("_rlnDefocusV")) == ["2.0"]
    assert list(tilts_block.find_loop("_rlnCtfAstigmatism")) == ["1.0"]
    assert list(tilts_block.find_loop("_rlnDefocusAngle")) == ["3.0"]
    assert list(tilts_block.find_loop("_rlnCtfFigureOfMerit")) == ["5.0"]
    assert list(tilts_block.find_loop("_rlnCtfMaxResolution")) == ["6.0"]
    assert list(tilts_block.find_loop("_rlnCtfIceRingDensity")) == ["5.0"]


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_excludetilts(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.excludetilts
    """
    job_dir = "ExcludeTiltImages/job004"
    input_file = (
        f"{tmp_path}/MotionCorr/job002/Movies/Position_1_2_001_1.50_fractions.mrc"
    )
    output_file = tmp_path / job_dir / "tilts/Position_1_2_001_1.50_fractions.mrc"
    relion_options = RelionServiceOptions()

    # .Nodes directory doesn't get made by this job
    (tmp_path / ".Nodes").mkdir()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.excludetilts",
        input_file,
        output_file,
        experiment_type="tomography",
    )

    # Check the output file structure
    assert (tmp_path / job_dir / "selected_tilt_series.star").exists()
    tilt_series_file = cif.read_file(
        str(tmp_path / job_dir / "selected_tilt_series.star")
    )

    global_block = tilt_series_file.find_block("global")
    assert list(global_block.find_loop("_rlnTomoName")) == ["Position_1_2"]
    assert list(global_block.find_loop("_rlnTomoTiltSeriesStarFile")) == [
        f"{job_dir}/tilt_series/Position_1_2.star"
    ]

    assert (tmp_path / job_dir / "tilt_series/Position_1_2.star").exists()
    tilts_file = cif.read_file(
        str(tmp_path / job_dir / "tilt_series/Position_1_2.star")
    )

    tilts_block = tilts_file.sole_block()
    assert list(tilts_block.find_loop("_rlnTomoTiltMovieFrameCount")) == [
        str(relion_options.frame_count)
    ]
    assert list(tilts_block.find_loop("_rlnTomoNominalStageTiltAngle")) == ["1.50"]
    assert list(tilts_block.find_loop("_rlnTomoNominalTiltAxisAngle")) == [
        str(relion_options.tilt_axis_angle)
    ]
    assert list(tilts_block.find_loop("_rlnMicrographPreExposure")) == ["12.77"]
    assert list(tilts_block.find_loop("_rlnTomoNominalDefocus")) == [
        str(relion_options.defocus)
    ]
    assert list(tilts_block.find_loop("_rlnMicrographName")) == [
        "MotionCorr/job002/Movies/Position_1_2_001_1.50_fractions.mrc"
    ]


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_aligntiltseries(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.aligntiltseries
    """
    job_dir = "AlignTiltSeries/job005"
    input_file = (
        f"{tmp_path}/MotionCorr/job002/Movies/Position_1_2_001_1.50_fractions.mrc"
    )
    output_file = tmp_path / job_dir / "tilts/Position_1_2_001_1.50_fractions.mrc"
    relion_options = RelionServiceOptions()

    ctf_output_file = (
        tmp_path / "CtfFind/job003/Movies/Position_1_2_001_1.50_fractions.ctf"
    )

    ctf_output_file.parent.mkdir(parents=True)
    with open(ctf_output_file.with_suffix(".txt"), "w") as f:
        f.write("0.0 1.0 2.0 3.0 4.0 5.0 6.0")
    with open(f"{ctf_output_file.with_suffix('')}_avrot.txt", "w") as f:
        f.write(
            "header\nheader\nheader\nheader\nheader\n0.24 0.26 0.27 0.29\n1 2 3 4\n"
        )

    # .Nodes directory doesn't get made by this job
    (tmp_path / ".Nodes").mkdir()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.aligntiltseries",
        input_file,
        output_file,
        experiment_type="tomography",
        results={
            "TomoXTilt": "0.00",
            "TomoYTilt": "4.00",
            "TomoZRot": "83.5",
            "TomoXShiftAngst": "1.5",
            "TomoYShiftAngst": "4.2",
        },
    )

    # Check the output file structure
    assert (tmp_path / job_dir / "aligned_tilt_series.star").exists()
    tilt_series_file = cif.read_file(
        str(tmp_path / job_dir / "aligned_tilt_series.star")
    )

    global_block = tilt_series_file.find_block("global")
    assert list(global_block.find_loop("_rlnTomoName")) == ["Position_1_2"]
    assert list(global_block.find_loop("_rlnTomoTiltSeriesStarFile")) == [
        f"{job_dir}/tilt_series/Position_1_2.star"
    ]

    assert (tmp_path / job_dir / "tilt_series/Position_1_2.star").exists()
    tilts_file = cif.read_file(
        str(tmp_path / job_dir / "tilt_series/Position_1_2.star")
    )

    tilts_block = tilts_file.sole_block()
    assert list(tilts_block.find_loop("_rlnTomoTiltMovieFrameCount")) == [
        str(relion_options.frame_count)
    ]
    assert list(tilts_block.find_loop("_rlnTomoNominalStageTiltAngle")) == ["1.50"]
    assert list(tilts_block.find_loop("_rlnTomoNominalTiltAxisAngle")) == [
        str(relion_options.tilt_axis_angle)
    ]
    assert list(tilts_block.find_loop("_rlnMicrographPreExposure")) == ["12.77"]
    assert list(tilts_block.find_loop("_rlnTomoNominalDefocus")) == [
        str(relion_options.defocus)
    ]
    assert list(tilts_block.find_loop("_rlnMicrographName")) == [
        "MotionCorr/job002/Movies/Position_1_2_001_1.50_fractions.mrc"
    ]
    assert list(tilts_block.find_loop("_rlnDefocusU")) == ["1.0"]
    assert list(tilts_block.find_loop("_rlnDefocusV")) == ["2.0"]
    assert list(tilts_block.find_loop("_rlnTomoXTilt")) == ["0.00"]
    assert list(tilts_block.find_loop("_rlnTomoYTilt")) == ["4.00"]
    assert list(tilts_block.find_loop("_rlnTomoZRot")) == ["83.5"]
    assert list(tilts_block.find_loop("_rlnTomoXShiftAngst")) == ["1.5"]
    assert list(tilts_block.find_loop("_rlnTomoYShiftAngst")) == ["4.2"]


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_tomograms(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.reconstructtomograms
    """
    job_dir = "Tomograms/job006"
    input_file = (
        f"{tmp_path}/MotionCorr/job002/Movies/Position_1_2_001_1.50_fractions.mrc"
    )
    output_file = tmp_path / job_dir / "tomograms/Position_1_2_stack_aretomo.mrc"
    relion_options = RelionServiceOptions()

    # .Nodes directory doesn't get made by this job
    (tmp_path / ".Nodes").mkdir()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.reconstructtomograms",
        input_file,
        output_file,
        experiment_type="tomography",
    )

    # Check the output file structure
    assert (tmp_path / job_dir / "tomograms.star").exists()
    tilt_series_file = cif.read_file(str(tmp_path / job_dir / "tomograms.star"))

    global_block = tilt_series_file.find_block("global")
    assert list(global_block.find_loop("_rlnTomoName")) == ["Position_1_2"]
    assert list(global_block.find_loop("_rlnVoltage")) == [str(relion_options.voltage)]
    assert list(global_block.find_loop("_rlnSphericalAberration")) == [
        str(relion_options.spher_aber)
    ]
    assert list(global_block.find_loop("_rlnAmplitudeContrast")) == [
        str(relion_options.ampl_contrast)
    ]
    assert list(global_block.find_loop("_rlnMicrographOriginalPixelSize")) == [
        str(relion_options.pixel_size)
    ]
    assert list(global_block.find_loop("_rlnTomoHand")) == [
        str(relion_options.invert_hand)
    ]
    assert list(global_block.find_loop("_rlnOpticsGroupName")) == ["optics1"]
    assert list(global_block.find_loop("_rlnTomoTiltSeriesPixelSize")) == [
        str(relion_options.pixel_size_downscaled)
    ]
    assert list(global_block.find_loop("_rlnTomoTiltSeriesStarFile")) == [
        "AlignTiltSeries/job005/tilt_series/Position_1_2.star"
    ]
    assert list(global_block.find_loop("_rlnTomoTomogramBinning")) == [
        str(relion_options.pixel_size_downscaled / relion_options.pixel_size)
    ]
    assert list(global_block.find_loop("_rlnTomoSizeX")) == [
        str(relion_options.tomo_size_x)
    ]
    assert list(global_block.find_loop("_rlnTomoSizeY")) == [
        str(relion_options.tomo_size_y)
    ]
    assert list(global_block.find_loop("_rlnTomoSizeZ")) == [str(relion_options.vol_z)]
    assert list(global_block.find_loop("_rlnTomoReconstructedTomogram")) == [
        str(output_file.relative_to(tmp_path))
    ]


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_denoisetomo(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    relion.denoisetomo
    """
    job_dir = "Denoise/job007"
    input_file = f"{tmp_path}/Denoise/job007/tomograms/Position_1_2_stack_aretomo.mrc"
    output_file = (
        tmp_path / job_dir / "tomograms/Position_1_2_stack_aretomo.denoised.mrc"
    )
    relion_options = RelionServiceOptions()

    # .Nodes directory doesn't get made by this job
    (tmp_path / ".Nodes").mkdir()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "relion.denoisetomo",
        input_file,
        output_file,
        experiment_type="tomography",
    )

    # Check the output file structure
    assert (tmp_path / job_dir / "tomograms.star").exists()
    tilt_series_file = cif.read_file(str(tmp_path / job_dir / "tomograms.star"))

    global_block = tilt_series_file.find_block("global")
    assert list(global_block.find_loop("_rlnTomoName")) == ["Position_1_2"]
    assert list(global_block.find_loop("_rlnVoltage")) == [str(relion_options.voltage)]
    assert list(global_block.find_loop("_rlnSphericalAberration")) == [
        str(relion_options.spher_aber)
    ]
    assert list(global_block.find_loop("_rlnAmplitudeContrast")) == [
        str(relion_options.ampl_contrast)
    ]
    assert list(global_block.find_loop("_rlnMicrographOriginalPixelSize")) == [
        str(relion_options.pixel_size)
    ]
    assert list(global_block.find_loop("_rlnTomoHand")) == [
        str(relion_options.invert_hand)
    ]
    assert list(global_block.find_loop("_rlnOpticsGroupName")) == ["optics1"]
    assert list(global_block.find_loop("_rlnTomoTiltSeriesPixelSize")) == [
        str(relion_options.pixel_size_downscaled)
    ]
    assert list(global_block.find_loop("_rlnTomoTiltSeriesStarFile")) == [
        "AlignTiltSeries/job005/tilt_series/Position_1_2.star"
    ]
    assert list(global_block.find_loop("_rlnTomoTomogramBinning")) == [
        str(relion_options.pixel_size_downscaled / relion_options.pixel_size)
    ]
    assert list(global_block.find_loop("_rlnTomoSizeX")) == [
        str(relion_options.tomo_size_x)
    ]
    assert list(global_block.find_loop("_rlnTomoSizeY")) == [
        str(relion_options.tomo_size_y)
    ]
    assert list(global_block.find_loop("_rlnTomoSizeZ")) == [str(relion_options.vol_z)]
    assert list(global_block.find_loop("_rlnTomoReconstructedTomogram")) == [
        str(Path(input_file).relative_to(tmp_path))
    ]
    assert list(global_block.find_loop("_rlnTomoReconstructedTomogramDenoised")) == [
        str(output_file.relative_to(tmp_path))
    ]


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_node_creator_cryolo_tomo(offline_transport, tmp_path):
    """
    Send a test message to the node creator for
    cryolo.autopick running on a tomogram
    """
    job_dir = "AutoPick/job009"
    (tmp_path / job_dir / "CBOX_3D").mkdir(parents=True)

    input_file = f"{tmp_path}/Denoise/job007/Movies/tomograms/Position_1_2_stack_aretomo.denoised.mrc"
    output_file = (
        tmp_path / job_dir / "CBOX_3D/Position_1_2_stack_aretomo.denoised.cbox"
    )
    relion_options = RelionServiceOptions()

    relion_options.cryolo_config_file = str(tmp_path / job_dir / "cryolo_config.json")
    (tmp_path / job_dir / "cryolo_config.json").touch()

    with open(
        tmp_path / job_dir / "CBOX_3D/Position_1_2_stack_aretomo.denoised.cbox", "w"
    ) as particles_file:
        particles_file.write(
            "data_global\n\n_cbox_format_version   1.0\n"
            "data_cryolo\n\nloop_\n"
            "_CoordinateX\n_CoordinateY\n_CoordinateZ\n_EstWidth\n_EstHeight\n"
            "60 70 80 5 6\n90 100 110 8 10\n"
        )

    (tmp_path / job_dir / "DISTR").mkdir(parents=True)
    (tmp_path / job_dir / "DISTR/confidence_distribution_summary_1.txt").touch()

    setup_and_run_node_creation(
        relion_options,
        offline_transport,
        tmp_path,
        job_dir,
        "cryolo.autopick",
        input_file,
        output_file,
        experiment_type="tomography",
    )

    # Check the output file structure
    assert (tmp_path / job_dir / "optimisation_set.star").exists()
    optimiser_file = cif.read_file(str(tmp_path / job_dir / "optimisation_set.star"))
    optimiser_block = optimiser_file.find_block("optimisation_set")
    assert list(optimiser_block.find_loop("_rlnTomoParticlesFile")) == [
        "AutoPick/job009/particles.star"
    ]
    assert list(optimiser_block.find_loop("_rlnTomoTomogramsFile")) == [
        "Denoise/job007/tomograms.star"
    ]

    assert (tmp_path / job_dir / "particles.star").exists()
    particles_file = cif.read_file(str(tmp_path / job_dir / "particles.star"))
    particles_block = particles_file.find_block("particles")
    assert list(particles_block.find_loop("_rlnTomoName")) == [
        "Position_1_2",
        "Position_1_2",
    ]
    assert list(particles_block.find_loop("_rlnCenteredCoordinateXAngst")) == [
        "62.5",
        "94.0",
    ]
    assert list(particles_block.find_loop("_rlnCenteredCoordinateYAngst")) == [
        "73.0",
        "105.0",
    ]
    assert list(particles_block.find_loop("_rlnCenteredCoordinateZAngst")) == [
        "80",
        "110",
    ]
