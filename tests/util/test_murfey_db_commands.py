from unittest import mock

import pytest

from cryoemservices.util import murfey_db_commands

bad_buffer_commands = [
    {},
    {"buffer_command": "not a dict"},
    {"buffer_command": {"ispyb_command": None}},
    {"buffer_command": {"ispyb_command": "non_existant"}},
    {
        "buffer_command": {
            "ispyb_command": "insert_movie",
        },
        "buffer_lookup": "not a dict",
    },
]


@pytest.mark.parametrize("commands", bad_buffer_commands)
def test_buffer_bad_commands(commands):
    def ispyb_parameters(p):
        return p

    assert (
        murfey_db_commands.buffer(commands, ispyb_parameters, mock.MagicMock()) is False
    )


@mock.patch(
    "cryoemservices.util.murfey_db_commands.ispyb_commands.insert_particle_classification"
)
def test_buffer(mock_insert_class, tmp_path):
    """
    Test that multipart message calls run reinjection.
    Use a 3D classification example for this
    """

    def mock_parameters(p):
        params = {"program_id": 1}
        return params.get(p)

    ispyb_test_message = {
        "buffer_command": {"ispyb_command": "insert_particle_classification"},
        "buffer_lookup": {"particle_classification_group_id": 5},
        "buffer_store": 10,
        "class_distribution": "0.4",
        "class_image_full_path": ("/path/to/Class3D/job015/run_it025_class001.mrc"),
        "class_number": 1,
        "estimated_resolution": 12.2,
        "ispyb_command": "buffer",
        "overall_fourier_completeness": 1.0,
        "particles_per_class": 20000.0,
        "rotation_accuracy": "30.3",
        "translation_accuracy": "33.3",
        "store_result": "value_to_store",
    }

    # Mock up the individual insert command
    mock_insert_class.return_value = {"success": True, "return_value": "dummy_class"}
    mock_db_session = mock.MagicMock()

    result = murfey_db_commands.buffer(
        ispyb_test_message, mock_parameters, mock_db_session
    )

    mock_insert_class.assert_called_once_with(
        message={
            "ispyb_command": "insert_particle_classification",
            "particle_classification_group_id": 5,
        },
        parameters=mock_parameters,
        session=mock_db_session,
    )
    assert result["success"]
    assert result["return_value"] == "dummy_class"
    assert result["store_result"] == "value_to_store"


def test_insert_movie():
    def mock_parameters(p):
        return {}.get(p)

    mock_session = mock.MagicMock()
    return_value = murfey_db_commands.insert_movie({}, mock_parameters, mock_session)
    assert return_value.get("success")
    assert return_value["return_value"] == 0


@mock.patch("cryoemservices.util.murfey_db_commands.models")
def test_insert_initial_model(mock_models):
    def mock_model_parameters(p):
        model_parameters = {
            "cryoem_initial_model_id": "None",
            "particle_classification_id": 401,
            "resolution": 15.1,
            "number_of_particles": 21000,
        }
        return model_parameters[p]

    mock_session = mock.MagicMock()
    return_value = murfey_db_commands.insert_cryoem_initial_model(
        {}, mock_model_parameters, mock_session
    )
    assert return_value.get("success")
    assert return_value["return_value"]

    mock_models.CryoemInitialModel.assert_called_with(
        resolution=15.1,
        numberOfParticles=21000,
        particleClassificationId=401,
    )
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()


@mock.patch("cryoemservices.util.murfey_db_commands.models")
def test_insert_tilts_has_movie(mock_models):
    def mock_tilt_parameters(p):
        tilt_parameters = {
            "dcid": 10,
            "movie_id": 1,
            "tomogram_id": 801,
            "defocus_u": 20000,
            "defocus_v": 25000,
            "psd_file": "/path/to/psd/file",
            "resolution": 10.4,
            "fit_quality": 0.15,
            "refined_magnification": 18000,
            "refined_tilt_angle": -12,
            "refined_tilt_axis": 83.6,
            "residual_error": 0.6,
        }
        return tilt_parameters[p]

    mock_session = mock.MagicMock()
    return_value = murfey_db_commands.insert_tilt_image_alignment(
        {}, mock_tilt_parameters, mock_session
    )
    assert return_value.get("success")
    assert return_value["return_value"]

    mock_models.TiltImageAlignment.assert_called_with(
        movieId=1,
        tomogramId=801,
        defocusU=20000,
        defocusV=25000,
        psdFile="/path/to/psd/file",
        resolution=10.4,
        fitQuality=0.15,
        refinedMagnification=18000,
        refinedTiltAngle=-12,
        refinedTiltAxis=83.6,
        residualError=0.6,
    )
    mock_session.add.assert_called()
    mock_session.commit.assert_called()


@mock.patch("cryoemservices.util.murfey_db_commands.models")
def test_insert_tilts_without_movie(mock_models):
    def mock_tilt_parameters(p):
        tilt_parameters = {
            "dcid": 10,
            "movie_id": None,
            "path": "/path/to/test_movie_motion_corrected.mrc",
            "tomogram_id": 801,
            "defocus_u": 20000,
            "defocus_v": 25000,
            "psd_file": "/path/to/psd/file",
            "resolution": 10.4,
            "fit_quality": 0.15,
            "refined_magnification": 18000,
            "refined_tilt_angle": -12,
            "refined_tilt_axis": 83.6,
            "residual_error": 0.6,
        }
        return tilt_parameters[p]

    class MockMovieParameters:
        data_collection_id = 10
        path = "/original/path/test_movie.mrc"
        murfey_id = 5

    mock_session = mock.MagicMock()
    mock_session.query().filter().all.return_value = [MockMovieParameters]

    return_value = murfey_db_commands.insert_tilt_image_alignment(
        {}, mock_tilt_parameters, mock_session
    )
    assert return_value.get("success")
    assert return_value["return_value"]

    # Check the movie lookup
    assert mock_session.query.call_count == 2
    assert mock_session.query().filter.call_count == 2
    assert mock_session.query().filter().all.call_count == 1

    mock_models.TiltImageAlignment.assert_called_with(
        movieId=5,
        tomogramId=801,
        defocusU=20000,
        defocusV=25000,
        psdFile="/path/to/psd/file",
        resolution=10.4,
        fitQuality=0.15,
        refinedMagnification=18000,
        refinedTiltAngle=-12,
        refinedTiltAxis=83.6,
        residualError=0.6,
    )
    mock_session.add.assert_called()
    mock_session.commit.assert_called()


def test_update_processing_status():
    def mock_parameters(p):
        return {}.get(p)

    mock_session = mock.MagicMock()
    return_value = murfey_db_commands.update_processing_status(
        {}, mock_parameters, mock_session
    )
    assert return_value.get("success")
    assert return_value["return_value"] == 0


def test_register_processing():
    def mock_parameters(p):
        return {}.get(p)

    mock_session = mock.MagicMock()
    return_value = murfey_db_commands.register_processing(
        {}, mock_parameters, mock_session
    )
    assert return_value.get("success")
    assert return_value["return_value"] == 0
