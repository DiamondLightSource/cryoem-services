from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.correlative_align_images import (
    AlignImagesParameters,
    AlignImagesService,
    _get_atlas_proposal_session_experiment_type,
)
from cryoemservices.util.models import MockRW


def test_get_atlas_dcg_experiment_type(mocker: MockerFixture):
    # Create mock return results
    mock_atlas = MagicMock()
    mock_proposal = MagicMock()
    mock_bl_session = MagicMock()
    mock_experiment = MagicMock()

    mock_result = MagicMock()
    mock_result.Atlas = mock_atlas
    mock_result.Proposal = mock_proposal
    mock_result.BLSession = mock_bl_session
    mock_result.ExperimentType = mock_experiment

    # Create the mock SQLAlchemy session
    mock_session = MagicMock()
    mock_session.execute.return_value.one.return_value = mock_result

    # Run the function
    assert _get_atlas_proposal_session_experiment_type(mock_session, 1) == (
        mock_atlas,
        mock_proposal,
        mock_bl_session,
        mock_experiment,
    )


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
    (  # Use recwrap | Use ISPyB | Ref type | Mov type
        (True, True, "Tomography", "FIB"),
        (False, True, "Tomography", "FIB"),
        (True, False, "Tomography", "Single Particle"),
        (False, False, "Tomography", "Single Particle"),
        (True, True, "Tomography", "CLEM"),
        (False, True, "Tomography", "CLEM"),
        (True, True, "Lamella Tomography", "FIB"),
        (False, True, "Lamella Tomography", "FIB"),
    ),
)
def test_align_images_service(
    mocker: MockerFixture,
    tmp_path: Path,
    mock_config_file: Path,
    offline_transport: OfflineTransport,
    test_params: tuple[bool, bool, str, str],
):
    # Set up the message parameters
    use_recwrap, use_ispyb, ref_type, mov_type = test_params

    # Set up reference image and moving image parameters
    id_ref = 1
    id_mov = 2

    proposal_code = "cm"
    proposal_number = "12345"

    visit_ref = f"{proposal_code}{proposal_number}-{id_ref}"
    image_ref = tmp_path / visit_ref / "processed" / "atlas" / "test.png"
    pixel_size_ref = 1e-6

    visit_mov = f"{proposal_code}{proposal_number}-{id_mov}"
    image_mov = tmp_path / visit_mov / "processed" / "atlas" / "test.png"
    pixel_size_mov = 1e-6

    save_dir = (
        tmp_path / visit_ref / "processed" / "correlation" / visit_mov / image_mov.stem
    )

    # Populate message based on whether to use ISPyB
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    align_images_test_message = {
        "id_ref": id_ref if use_ispyb else None,
        "id_mov": id_mov if use_ispyb else None,
        "visit_ref": None if use_ispyb else visit_ref,
        "experiment_type_ref": None if use_ispyb else ref_type,
        "image_ref": None if use_ispyb else str(image_ref),
        "pixel_size_ref": None if use_ispyb else pixel_size_ref,
        "visit_mov": None if use_ispyb else visit_mov,
        "experiment_type_mov": None if use_ispyb else mov_type,
        "image_mov": None if use_ispyb else str(image_mov),
        "pixel_size_mov": None if use_ispyb else pixel_size_mov,
        "save_dir": None if use_ispyb else save_dir,
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
    mock_proposal_ref = MagicMock()
    mock_bl_session_ref = MagicMock()
    mock_experiment_ref = MagicMock()

    mock_atlas_mov = MagicMock()
    mock_proposal_mov = MagicMock()
    mock_bl_session_mov = MagicMock()
    mock_experiment_mov = MagicMock()

    if use_ispyb:
        mock_atlas_ref.atlasImage = str(image_ref)
        mock_atlas_ref.pixelSize = pixel_size_ref
        mock_proposal_ref.proposalCode = proposal_code
        mock_proposal_ref.proposalNumber = proposal_number
        mock_bl_session_ref.visit_number = id_ref
        mock_experiment_ref.name = ref_type

        mock_atlas_mov.atlasImage = str(image_mov)
        mock_atlas_mov.pixelSize = pixel_size_mov
        mock_proposal_mov.proposalCode = proposal_code
        mock_proposal_mov.proposalNumber = proposal_number
        mock_bl_session_mov.visit_number = id_mov
        mock_experiment_mov.name = mov_type

    mock_get_ispyb = mocker.patch(
        "cryoemservices.services.correlative_align_images._get_atlas_proposal_session_experiment_type",
        side_effect=[
            (
                mock_atlas_ref,
                mock_proposal_ref,
                mock_bl_session_ref,
                mock_experiment_ref,
            ),
            (
                mock_atlas_mov,
                mock_proposal_mov,
                mock_bl_session_mov,
                mock_experiment_mov,
            ),
        ],
    )

    # Mock the '_handle_fib_tomo_case' class function
    mock_handle_fib_tomo = mocker.patch(
        "cryoemservices.services.correlative_align_images.AlignImagesService._handle_fib_tomo_case"
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
    if use_ispyb:
        mock_get_ispyb.assert_any_call(
            session=mock_ispyb_session,
            atlas_id=params.id_ref,
        )
        mock_get_ispyb.assert_any_call(
            session=mock_ispyb_session,
            atlas_id=params.id_mov,
        )
    else:
        mock_get_ispyb.assert_not_called()

    # It goes into the correct case block
    match sorted((ref_type, mov_type)):
        case ["FIB", "Tomography"] | ["FIB", "Lamella Tomography"]:
            mock_handle_fib_tomo.assert_called_with(
                image_ref,
                pixel_size_ref,
                image_mov,
                pixel_size_mov,
                save_dir,
            )
        case _:
            service.log.info.assert_called_with(
                "No image alignment algorithm implemented for this case yet"
            )
