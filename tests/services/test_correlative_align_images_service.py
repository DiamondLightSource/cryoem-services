from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.correlative_align_images import (
    AlignImagesParameters,
    AlignImagesService,
)
from cryoemservices.util.models import MockRW


@pytest.fixture
def offline_transport(mocker: MockerFixture):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    mocker.spy(transport, "nack")
    return transport


@pytest.fixture
def mock_config_file(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    lines = [
        "rabbitmq_credentials: rmq_creds",
        f"recipe_directory: {tmp_path}/recipes",
        f"ispyb_credentials: {tmp_path}/ispyb.cfg",
    ]
    with open(config_file, "w") as file:
        for line in lines:
            file.write(line + "\n")
    return config_file


@pytest.mark.parametrize(
    "test_params",
    (  # Use recwrap | Ref type | Mov type
        (True, "Tomography", "FIB"),
        (False, "Tomography", "FIB"),
        (True, "Tomography", "Single Particle"),
        (False, "Tomography", "Single Particle"),
        (True, "Tomography", "CLEM"),
        (False, "Tomography", "CLEM"),
    ),
)
def test_align_images_service(
    mocker: MockerFixture,
    tmp_path: Path,
    mock_config_file: Path,
    offline_transport: OfflineTransport,
    test_params: tuple[bool, str, str],
):
    # Set up the message parameters
    use_recwrap, ref_type, mov_type = test_params
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    align_images_test_message = {
        "id_ref": 1,
        "image_ref": str(tmp_path / "ref.png"),
        "pixel_size_ref": 1e-6,
        "id_mov": 2,
        "image_mov": str(tmp_path / "mov.png"),
        "pixel_size_mov": 1e-6,
    }
    params = AlignImagesParameters(**align_images_test_message)

    # Mock the ISPyB session creation logic
    mock_sqlalchemy = mocker.patch(
        "cryoemservices.services.correlative_align_images.sqlalchemy"
    )
    mocker.patch("cryoemservices.services.correlative_align_images.ispyb.sqlalchemy")
    mock_ispyb_session = MagicMock()
    mock_sqlalchemy.orm.sessionmaker()().__enter__.return_value = mock_ispyb_session

    # Mock the query function and its returns
    mock_atlas_ref = MagicMock()
    mock_dcg_ref = MagicMock()
    mock_experiment_ref = MagicMock()
    mock_experiment_ref.name = ref_type

    mock_atlas_mov = MagicMock()
    mock_dcg_mov = MagicMock()
    mock_experiment_mov = MagicMock()
    mock_experiment_mov.name = mov_type

    mock_get = mocker.patch(
        "cryoemservices.services.correlative_align_images._get_atlas_dcg_experiment_type",
        side_effect=[
            (mock_atlas_ref, mock_dcg_ref, mock_experiment_ref),
            (mock_atlas_mov, mock_dcg_mov, mock_experiment_mov),
        ],
    )

    # Set up and run the service
    service = AlignImagesService(
        environment={"config": str(mock_config_file), "queue": ""},
        transport=offline_transport,
    )
    service.log = MagicMock()  # Mock the logger to evaluate calls
    service.initializing()
    if use_recwrap:
        recwrap = MockRW(offline_transport)
        recwrap.recipe_step = {"parameters": align_images_test_message}
        service.call_align_images(
            recwrap,
            header=header,
            message=None,
        )
    else:
        service.call_align_images(
            None,
            header=header,
            message=align_images_test_message,
        )

    # Queries for the Atlas, DCG, and ExperimentType were made
    mock_get.assert_any_call(
        session=mock_ispyb_session,
        atlas_id=params.id_ref,
    )
    mock_get.assert_any_call(
        session=mock_ispyb_session,
        atlas_id=params.id_mov,
    )
    # It goes into the correct case block
    match (ref_type, mov_type):
        case ("Tomography", "FIB"):
            service.log.info.assert_called_with("Aligning FIB atlas to tomography one")
        case _:
            service.log.info.assert_called_with(
                "No image alignment algorithm implemented for this case yet"
            )
