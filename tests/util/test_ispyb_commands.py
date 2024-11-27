from __future__ import annotations

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
        createdTimeStamp="1970-01-01 01:00:01",
    )

    mock_session.add.assert_called()
    mock_session.commit.assert_called()
