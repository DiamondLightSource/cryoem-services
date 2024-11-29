from __future__ import annotations

import copy
import sys
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services import ispyb_connector


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    return transport


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb_commands.insert_movie")
def test_ispyb_service_run(
    mock_command, mock_sqlalchemy, mock_ispyb_api, offline_transport, tmp_path
):
    """
    Send a test message to the ispyb service
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    ispyb_test_message = {
        "parameters": {
            "ispyb_command": "insert_movie",
        },
        "content": "dummy",
    }

    mock_command.return_value = {"success": True, "return_value": "dummy_result"}

    # Set up the mock service and call it
    service = ispyb_connector.EMISPyB()
    service.transport = offline_transport
    service.start()
    service.insert_into_ispyb(None, header=header, message=ispyb_test_message)

    mock_ispyb_api.url.assert_called()
    mock_sqlalchemy.create_engine.assert_called()
    mock_sqlalchemy.orm.sessionmaker.assert_called()
    mock_sqlalchemy.orm.sessionmaker()().__enter__.assert_called()

    mock_command.assert_called_with(
        message="dummy",
        parameters=mock.ANY,
        session=mock_sqlalchemy.orm.sessionmaker()().__enter__(),
    )

    # Check that the correct messages were sent
    offline_transport.send.assert_any_call(
        destination="output",
        message={"result": "dummy_result"},
    )


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb_commands.insert_movie")
def test_ispyb_service_store_result(
    mock_command, mock_sqlalchemy, mock_ispyb_api, offline_transport, tmp_path
):
    """
    Send a test message to the ispyb service with a stored result
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    ispyb_test_message = {
        "parameters": {
            "ispyb_command": "insert_movie",
            "store_result": "full_result",
        },
        "content": "dummy",
    }

    mock_command.return_value = {
        "success": True,
        "return_value": "result_value",
        "store_result": "result_name",
    }

    mock_rw = mock.MagicMock()
    mock_rw.recipe_step = {"parameters": ispyb_test_message["parameters"]}
    mock_rw.environment = {}

    # Set up the mock service and call it
    service = ispyb_connector.EMISPyB()
    service.transport = offline_transport
    service.start()
    service.insert_into_ispyb(rw=mock_rw, header=header, message=ispyb_test_message)

    # Check that the correct messages were sent
    mock_command.assert_called()
    mock_rw.send_to.assert_called_with(
        "output",
        {"result": "result_value"},
    )

    # Check the results were stored in the environment
    assert mock_rw.environment["full_result"] == "result_value"
    assert mock_rw.environment["result_name"] == "result_value"


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb_commands.insert_movie")
def test_ispyb_service_checkpoint(
    mock_command, mock_sqlalchemy, mock_ispyb_api, offline_transport, tmp_path
):
    """
    Send a test message to the ispyb service with a checkpoint return
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    ispyb_test_message = {
        "parameters": {
            "ispyb_command": "insert_movie",
        },
        "content": "dummy",
    }

    mock_command.return_value = {
        "checkpoint": True,
        "return_value": "dummy_result",
        "checkpoint_dict": {"checkpoint": 1},
    }

    mock_rw = mock.MagicMock()
    mock_rw.recipe_step = {"parameters": ispyb_test_message["parameters"]}

    # Set up the mock service and call it
    service = ispyb_connector.EMISPyB()
    service.transport = offline_transport
    service.start()
    service.insert_into_ispyb(rw=mock_rw, header=header, message=ispyb_test_message)

    # Check that the correct messages were sent - this checkpoints but does not send
    mock_rw.send_to.assert_not_called()
    offline_transport.send.assert_not_called()
    mock_rw.checkpoint.assert_called_with({"checkpoint": 1})


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.sqlalchemy")
@mock.patch(
    "cryoemservices.services.ispyb_connector.ispyb_commands.insert_particle_classification_group"
)
@mock.patch(
    "cryoemservices.services.ispyb_connector.ispyb_commands.insert_particle_classification"
)
@mock.patch(
    "cryoemservices.services.ispyb_connector.ispyb_commands.insert_cryoem_initial_model"
)
def test_ispyb_multipart_message(
    mock_insert_model,
    mock_insert_class,
    mock_insert_group,
    mock_sqlalchemy,
    mock_ispyb_api,
    offline_transport,
):
    """
    Test that multipart message calls run reinjection.
    Use a 3D classification example for this
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    ispyb_test_message = {
        "parameters": {"dcid": 1000, "program_id": 100},
        "content": {
            "ispyb_command": "multipart_message",
            "ispyb_command_list": [
                {
                    "batch_number": "1",
                    "buffer_command": {
                        "ispyb_command": "insert_particle_classification_group"
                    },
                    "buffer_store": 5,
                    "ispyb_command": "buffer",
                    "number_of_classes_per_batch": 2,
                    "number_of_particles_per_batch": 50000,
                    "particle_picker_id": 6,
                    "symmetry": "C1",
                    "type": "3D",
                },
                {
                    "buffer_command": {
                        "ispyb_command": "insert_particle_classification"
                    },
                    "buffer_lookup": {"particle_classification_group_id": 5},
                    "buffer_store": 10,
                    "class_distribution": "0.4",
                    "class_image_full_path": (
                        "/path/to/Class3D/job015/run_it025_class001.mrc"
                    ),
                    "class_number": 1,
                    "estimated_resolution": 12.2,
                    "ispyb_command": "buffer",
                    "overall_fourier_completeness": 1.0,
                    "particles_per_class": 20000.0,
                    "rotation_accuracy": "30.3",
                    "translation_accuracy": "33.3",
                },
                {
                    "buffer_command": {"ispyb_command": "insert_cryoem_initial_model"},
                    "buffer_lookup": {"particle_classification_id": 10},
                    "ispyb_command": "buffer",
                    "number_of_particles": 20000.0,
                    "resolution": "30.3",
                    "store_result": "ispyb_initial_model_id",
                },
            ],
        },
    }

    # The output will be the input, but without the first command as that has run
    output_commands = copy.deepcopy(ispyb_test_message["content"]["ispyb_command_list"])
    output_commands.pop(0)

    # After a second run two commands will have been removed
    second_output_commands = copy.deepcopy(
        ispyb_test_message["content"]["ispyb_command_list"]
    )
    second_output_commands.pop(0)
    second_output_commands.pop(0)

    # Mock up the individual insert command
    mock_insert_group.return_value = {"success": True, "return_value": "dummy_group"}
    mock_insert_class.return_value = {"success": True, "return_value": "dummy_class"}
    mock_insert_model.return_value = {"success": True, "return_value": "dummy_model"}

    mock_rw = mock.MagicMock()
    mock_rw.recipe_step = {"parameters": ispyb_test_message["parameters"]}
    mock_rw.environment = {}

    # Set up the mock service and call it
    service = ispyb_connector.EMISPyB()
    service.transport = offline_transport
    service.start()
    service.insert_into_ispyb(
        rw=mock_rw, header=header, message=ispyb_test_message["content"]
    )

    # Check that the correct messages were sent - this checkpoints but does not send
    mock_rw.send_to.assert_not_called()
    offline_transport.send.assert_not_called()
    mock_rw.checkpoint.assert_any_call(
        {
            "checkpoint": 1,
            "ispyb_command": "multipart_message",
            "ispyb_command_list": output_commands,
        }
    )

    # This command should have buffer stored
    mock_ispyb_api.ZcZocaloBuffer.assert_called_with(
        AutoProcProgramID=100,
        UUID=5,
        Reference="dummy_group",
    )
    mock_sqlalchemy.orm.sessionmaker()().__enter__().merge.assert_called()
    mock_sqlalchemy.orm.sessionmaker()().__enter__().commit.assert_called()

    # It should then do lookups and another checkpoint on resubmission
    service.insert_into_ispyb(
        rw=mock_rw,
        header=header,
        message={
            "checkpoint": 1,
            "ispyb_command": "multipart_message",
            "ispyb_command_list": output_commands,
        },
    )
    mock_rw.checkpoint.assert_called_with(
        {
            "checkpoint": 2,
            "ispyb_command": "multipart_message",
            "ispyb_command_list": second_output_commands,
        }
    )
    mock_sqlalchemy.orm.sessionmaker()().__enter__().query.assert_called()
    mock_sqlalchemy.orm.sessionmaker()().__enter__().query().filter.assert_called()
    mock_sqlalchemy.orm.sessionmaker()().__enter__().query().filter().filter.assert_called()

    # Then a final round of submission should store the result and return success
    service.insert_into_ispyb(
        rw=mock_rw,
        header=header,
        message={
            "checkpoint": 2,
            "ispyb_command": "multipart_message",
            "ispyb_command_list": second_output_commands,
        },
    )
    mock_rw.send_to.assert_called_with(
        "output",
        {"result": "dummy_model"},
    )
    assert mock_rw.environment["ispyb_initial_model_id"] == "dummy_model"
