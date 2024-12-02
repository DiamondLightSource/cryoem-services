from __future__ import annotations

from datetime import datetime
from unittest import mock

from cryoemservices.util import ispyb_commands


@mock.patch("cryoemservices.util.ispyb_commands.models")
def test_insert_movie_notime(mock_models):
    def mock_movie_parameters(p):
        movie_params = {
            "dcid": 101,
            "movie_number": 1,
            "movie_path": "/path/to/movie",
            "timestamp": None,
        }
        return movie_params[p]

    mock_session = mock.MagicMock()

    return_value = ispyb_commands.insert_movie({}, mock_movie_parameters, mock_session)
    assert return_value.get("success")
    assert return_value["return_value"]

    mock_models.Movie.assert_called_with(
        dataCollectionId=101, movieNumber=1, movieFullPath="/path/to/movie"
    )

    mock_session.add.assert_called()
    mock_session.commit.assert_called()


@mock.patch("cryoemservices.util.ispyb_commands.models")
def test_insert_movie_timestamp(mock_models):
    def mock_movie_parameters(p):
        movie_params = {
            "dcid": 101,
            "movie_number": 1,
            "movie_path": "/path/to/movie",
            "timestamp": 1,
        }
        return movie_params[p]

    mock_session = mock.MagicMock()
    return_value = ispyb_commands.insert_movie({}, mock_movie_parameters, mock_session)
    assert return_value.get("success")
    assert return_value["return_value"]

    mock_models.Movie.assert_called_with(
        dataCollectionId=101,
        movieNumber=1,
        movieFullPath="/path/to/movie",
        createdTimeStamp=datetime.fromtimestamp(1).strftime("%Y-%m-%d %H:%M:%S"),
    )
    mock_session.add.assert_called()
    mock_session.commit.assert_called()


@mock.patch("cryoemservices.util.ispyb_commands.models")
def test_insert_motion_correction_has_movie(mock_models):
    def mock_mc_parameters(p):
        mc_parameters = {
            "dcid": 101,
            "movie_id": 1,
            "program_id": 2,
            "image_number": 3,
            "first_frame": 0,
            "last_frame": 50,
            "dose_per_frame": 0.5,
            "dose_weight": 1,
            "total_motion": 8.5,
            "average_motion_per_frame": 8.5 / 50,
            "drift_plot_full_path": "/path/to/drift/plot",
            "micrograph_full_path": "/path/to/micrograph",
            "micrograph_snapshot_full_path": "/path/to/snapshot",
            "patches_used_x": 5,
            "patches_used_y": 6,
            "fft_full_path": "/path/to/fft",
            "fft_corrected_full_path": "/path/to/fft/corrected",
            "comments": "comment",
        }
        return mc_parameters[p]

    mock_session = mock.MagicMock()
    return_value = ispyb_commands.insert_motion_correction(
        {}, mock_mc_parameters, mock_session
    )
    assert return_value.get("success")
    assert return_value["return_value"]

    mock_models.Movie.assert_not_called()
    mock_models.MotionCorrection.assert_called_with(
        dataCollectionId=101,
        movieId=1,
        autoProcProgramId=2,
        imageNumber=3,
        firstFrame=0,
        lastFrame=50,
        dosePerFrame=0.5,
        doseWeight=1,
        totalMotion=8.5,
        averageMotionPerFrame=8.5 / 50,
        driftPlotFullPath="/path/to/drift/plot",
        micrographFullPath="/path/to/micrograph",
        micrographSnapshotFullPath="/path/to/snapshot",
        patchesUsedX=5,
        patchesUsedY=6,
        fftFullPath="/path/to/fft",
        fftCorrectedFullPath="/path/to/fft/corrected",
        comments="comment",
    )
    mock_session.add.assert_called()
    mock_session.commit.assert_called()


@mock.patch("cryoemservices.util.ispyb_commands.models")
def test_insert_motion_correction_without_movie(mock_models):
    def mock_mc_parameters(p):
        mc_parameters = {
            "dcid": 101,
            "image_number": 3,
            "micrograph_full_path": "/path/to/micrograph",
            "created_time_stamp": 1,
            "movie_id": None,
        }
        return mc_parameters[p] if p in mc_parameters.keys() else 1

    mock_session = mock.MagicMock()
    return_value = ispyb_commands.insert_motion_correction(
        {}, mock_mc_parameters, mock_session
    )
    assert return_value.get("success")
    assert return_value["return_value"]

    mock_models.Movie.assert_called_with(
        dataCollectionId=101,
        movieNumber=3,
        movieFullPath="/path/to/micrograph",
        createdTimeStamp=datetime.fromtimestamp(1).strftime("%Y-%m-%d %H:%M:%S"),
    )
    mock_models.MotionCorrection.assert_called()
    mock_session.add.assert_called()
    mock_session.commit.assert_called()


@mock.patch("cryoemservices.util.ispyb_commands.models")
def test_insert_ice_thickness(mock_models):
    def mock_ice_parameters(p):
        ice_parameters = {
            "dcid": 10,
            "motion_correction_id": 101,
            "program_id": 1,
            "minimum": 5.1,
            "q1": 6.1,
            "median": 7.1,
            "q3": 8.1,
            "maximum": 9.1,
        }
        return ice_parameters[p]

    mock_session = mock.MagicMock()
    return_value = ispyb_commands.insert_relative_ice_thickness(
        {}, mock_ice_parameters, mock_session
    )
    assert return_value.get("success")
    assert return_value["return_value"]

    mock_models.RelativeIceThickness.assert_called_with(
        motionCorrectionId=101,
        autoProcProgramId=1,
        minimum=5.1,
        q1=6.1,
        median=7.1,
        q3=8.1,
        maximum=9.1,
    )
    mock_session.add.assert_called()
    mock_session.commit.assert_called()


@mock.patch("cryoemservices.util.ispyb_commands.models")
def test_insert_ctf(mock_models):
    def mock_ctf_parameters(p):
        ctf_parameters = {
            "dcid": 10,
            "ctf_id": 201,
            "motion_correction_id": 101,
            "program_id": 1,
            "box_size_x": 5000,
            "box_size_y": 4000,
            "min_resolution": 3.1,
            "max_resolution": 5.1,
            "min_defocus": 0.5,
            "max_defocus": 3.5,
            "defocus_step_size": 0.5,
            "astigmatism": -500,
            "astigmatism_angle": 60,
            "estimated_resolution": 3.2,
            "estimated_defocus": 2.2,
            "amplitude_contrast": 0.1,
            "cc_value": 10000,
            "fft_theoretical_full_path": "/path/to/fft",
            "comments": "comment",
        }
        return ctf_parameters[p]

    mock_session = mock.MagicMock()
    return_value = ispyb_commands.insert_ctf({}, mock_ctf_parameters, mock_session)
    assert return_value.get("success")
    assert return_value["return_value"]

    mock_models.CTF.assert_called_with(
        ctfId=201,
        motionCorrectionId=101,
        autoProcProgramId=1,
        boxSizeX=5000,
        boxSizeY=4000,
        minResolution=3.1,
        maxResolution=5.1,
        minDefocus=0.5,
        maxDefocus=3.5,
        defocusStepSize=0.5,
        astigmatism=-500,
        astigmatismAngle=60,
        estimatedResolution=3.2,
        estimatedDefocus=2.2,
        amplitudeContrast=0.1,
        ccValue=10000,
        fftTheoreticalFullPath="/path/to/fft",
        comments="comment",
    )
    mock_session.add.assert_called()
    mock_session.commit.assert_called()


@mock.patch("cryoemservices.util.ispyb_commands.models")
def test_insert_particle_picker(mock_models):
    def mock_picker_parameters(p):
        picker_parameters = {
            "dcid": 10,
            "particle_picker_id": 301,
            "motion_correction_id": 101,
            "program_id": 1,
            "particle_picking_template": "/path/to/picker/template",
            "particle_diameter": 180,
            "number_of_particles": 72,
            "summary_image_full_path": "/path/to/summary/image",
        }
        return picker_parameters[p]

    mock_session = mock.MagicMock()
    return_value = ispyb_commands.insert_particle_picker(
        {}, mock_picker_parameters, mock_session
    )
    assert return_value.get("success")
    assert return_value["return_value"]

    mock_models.ParticlePicker.assert_called_with(
        particlePickerId=301,
        programId=1,
        firstMotionCorrectionId=101,
        particlePickingTemplate="/path/to/picker/template",
        particleDiameter=180,
        numberOfParticles=72,
        summaryImageFullPath="/path/to/summary/image",
    )
    mock_session.add.assert_called()
    mock_session.commit.assert_called()


@mock.patch("cryoemservices.util.ispyb_commands.models")
def test_insert_particle_classification_new(mock_models):
    def mock_class_parameters(p):
        class_parameters = {
            "dcid": 10,
            "particle_classification_id": 401,
            "particle_classification_group_id": 501,
            "class_number": 1,
            "class_image_full_path": "/path/to/class/image",
            "particles_per_class": 72,
            "rotation_accuracy": 7.4,
            "translation_accuracy": 6.5,
            "estimated_resolution": 8.5,
            "overall_fourier_completeness": 0.95,
            "class_distribution": 0.25,
            "selected": 1,
            "bfactor_fit_intercept": 0.2,
            "bfactor_fit_linear": 50,
            "bfactor_fit_quadratic": 0,
        }
        return class_parameters[p]

    # Mock which returns None for existing objects
    mock_session = mock.MagicMock()
    mock_session.query().filter().first.return_value = None

    return_value = ispyb_commands.insert_particle_classification(
        {}, mock_class_parameters, mock_session
    )
    assert return_value.get("success")
    assert return_value["return_value"]

    assert mock_session.query.call_count == 2
    assert mock_session.query().filter.call_count == 2
    assert mock_session.query().filter().first.call_count == 1

    mock_models.ParticleClassification.assert_called_with(
        particleClassificationId=401,
        particleClassificationGroupId=501,
        classNumber=1,
        classImageFullPath="/path/to/class/image",
        particlesPerClass=72,
        rotationAccuracy=7.4,
        translationAccuracy=6.5,
        estimatedResolution=8.5,
        overallFourierCompleteness=0.95,
        classDistribution=0.25,
        selected=1,
        bFactorFitIntercept=0.2,
        bFactorFitLinear=50,
        bFactorFitQuadratic=0,
    )
    mock_session.add.assert_called()
    mock_session.commit.assert_called()


def test_insert_particle_classification_update():
    def mock_class_parameters(p):
        class_parameters = {
            "dcid": 10,
            "particle_classification_id": 401,
            "particle_classification_group_id": 501,
            "class_number": 1,
            "class_image_full_path": "/path/to/class/image",
            "particles_per_class": 72,
            "rotation_accuracy": 7.4,
            "translation_accuracy": 6.5,
            "estimated_resolution": 8.5,
            "overall_fourier_completeness": 0.95,
            "class_distribution": 0.25,
            "selected": 1,
            "bfactor_fit_intercept": 0.2,
            "bfactor_fit_linear": 50,
            "bfactor_fit_quadratic": 0,
        }
        return class_parameters[p]

    # Mock which returns an existing object
    mock_session = mock.MagicMock()
    mock_session.query().filter().first.return_value = 1

    return_value = ispyb_commands.insert_particle_classification(
        {}, mock_class_parameters, mock_session
    )
    assert return_value.get("success")
    assert return_value["return_value"]

    assert mock_session.query.call_count == 3
    assert mock_session.query().filter.call_count == 3
    assert mock_session.query().filter().first.call_count == 1

    # Don't check the model call here, instead look at the update
    mock_session.query().filter().update.assert_called_with(
        {
            "particleClassificationGroupId": 501,
            "classNumber": 1,
            "classImageFullPath": "/path/to/class/image",
            "particlesPerClass": 72,
            "rotationAccuracy": 7.4,
            "translationAccuracy": 6.5,
            "estimatedResolution": 8.5,
            "overallFourierCompleteness": 0.95,
            "classDistribution": 0.25,
            "selected": 1,
            "bFactorFitIntercept": 0.2,
            "bFactorFitLinear": 50,
            "bFactorFitQuadratic": 0,
        }
    )
    mock_session.add.assert_not_called()
    mock_session.commit.assert_called()


@mock.patch("cryoemservices.util.ispyb_commands.models")
def test_insert_particle_classification_group_new(mock_models):
    def mock_group_parameters(p):
        group_parameters = {
            "dcid": 10,
            "particle_picker_id": 301,
            "particle_classification_group_id": 501,
            "program_id": 1,
            "type": "2D",
            "batch_number": 2,
            "number_of_particles_per_batch": 50000,
            "number_of_classes_per_batch": 50,
            "symmetry": "C1",
        }
        return group_parameters[p]

    # Mock which returns None for existing objects
    mock_session = mock.MagicMock()
    mock_session.query().filter().first.return_value = None

    return_value = ispyb_commands.insert_particle_classification_group(
        {}, mock_group_parameters, mock_session
    )
    assert return_value.get("success")
    assert return_value["return_value"]

    assert mock_session.query.call_count == 2
    assert mock_session.query().filter.call_count == 2
    assert mock_session.query().filter().first.call_count == 1

    mock_models.ParticleClassificationGroup.assert_called_with(
        particleClassificationGroupId=501,
        particlePickerId=301,
        programId=1,
        type="2D",
        batchNumber=2,
        numberOfParticlesPerBatch=50000,
        numberOfClassesPerBatch=50,
        symmetry="C1",
    )
    mock_session.add.assert_called()
    mock_session.commit.assert_called()


def test_insert_particle_classification_group_update():
    def mock_group_parameters(p):
        group_parameters = {
            "dcid": 10,
            "particle_picker_id": 301,
            "particle_classification_group_id": 501,
            "program_id": 1,
            "type": "2D",
            "batch_number": 2,
            "number_of_particles_per_batch": 50000,
            "number_of_classes_per_batch": 50,
            "symmetry": "C1",
        }
        return group_parameters[p]

    # Mock which returns an existing object
    mock_session = mock.MagicMock()
    mock_session.query().filter().first.return_value = 1

    return_value = ispyb_commands.insert_particle_classification_group(
        {}, mock_group_parameters, mock_session
    )
    assert return_value.get("success")
    assert return_value["return_value"]

    assert mock_session.query.call_count == 3
    assert mock_session.query().filter.call_count == 3
    assert mock_session.query().filter().first.call_count == 1

    # Don't check the model call here, instead look at the update
    mock_session.query().filter().update.assert_called_with(
        {
            "particlePickerId": 301,
            "programId": 1,
            "type": "2D",
            "batchNumber": 2,
            "numberOfParticlesPerBatch": 50000,
            "numberOfClassesPerBatch": 50,
            "symmetry": "C1",
        }
    )
    mock_session.add.assert_not_called()
    mock_session.commit.assert_called()
