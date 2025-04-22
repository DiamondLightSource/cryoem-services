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
    mocker.spy(transport, "nack")
    return transport


def write_config_file(tmp_path):
    # Create a config file
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as cf:
        cf.write("rabbitmq_credentials: rmq_creds\n")
        cf.write(f"recipe_directory: {tmp_path}/recipes\n")
        cf.write(f"ispyb_credentials: {tmp_path}/ispyb.cfg")


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.ispyb_connector.MockRW")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb_commands.insert_movie")
def test_ispyb_service_run(
    mock_command, mock_sqlalchemy, mock_ispyb_api, mock_rw, offline_transport, tmp_path
):
    """
    Send a test message to the ispyb service
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    ispyb_test_message = {
        "ispyb_command": "insert_movie",
    }

    mock_command.return_value = {"success": True, "return_value": "dummy_result"}
    write_config_file(tmp_path)

    # Set up the mock service and call it
    service = ispyb_connector.EMISPyB(
        environment={"config": f"{tmp_path}/config.yaml", "queue": ""},
        transport=offline_transport,
    )
    service.initializing()
    service.insert_into_ispyb(None, header=header, message=ispyb_test_message)

    mock_ispyb_api.url.assert_called_with(credentials=tmp_path / "ispyb.cfg")
    mock_sqlalchemy.create_engine.assert_called()
    mock_sqlalchemy.orm.sessionmaker.assert_called()
    mock_sqlalchemy.orm.sessionmaker()().__enter__.assert_called()

    mock_command.assert_called_with(
        message={
            "expiry_time": mock.ANY,
            "ispyb_command": "insert_movie",
        },
        parameters=mock.ANY,
        session=mock_sqlalchemy.orm.sessionmaker()().__enter__(),
    )

    # Check that the correct messages were sent
    mock_rw.assert_called_with(offline_transport)
    mock_rw().set_default_channel.assert_called_with("output")
    mock_rw().send.assert_called_with({"result": "dummy_result"})


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
        "ispyb_command": "insert_movie",
        "store_result": "full_result",
    }

    mock_command.return_value = {
        "success": True,
        "return_value": "result_value",
        "store_result": "result_name",
    }

    mock_rw = mock.MagicMock()
    mock_rw.recipe_step = {"parameters": ispyb_test_message}
    mock_rw.environment = {}
    write_config_file(tmp_path)

    # Set up the mock service and call it
    service = ispyb_connector.EMISPyB(
        environment={"config": f"{tmp_path}/config.yaml", "queue": ""},
        transport=offline_transport,
    )
    service.initializing()
    service.insert_into_ispyb(rw=mock_rw, header=header, message=ispyb_test_message)

    # Check that the correct messages were sent
    mock_command.assert_called()
    mock_rw.set_default_channel.assert_called_with("output")
    mock_rw.send.assert_called_with({"result": "result_value"})

    # Check the results were stored in the environment
    assert mock_rw.environment["full_result"] == "result_value"
    assert mock_rw.environment["result_name"] == "result_value"


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb_commands.insert_movie")
def test_ispyb_service_env_keys(
    mock_command, mock_sqlalchemy, mock_ispyb_api, offline_transport, tmp_path
):
    """
    Test as above, but read the key values from the environment in $ form
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    ispyb_test_message = {
        "ispyb_command": "${command}",
        "store_result": "$output",
    }
    ispyb_test_environment = {"command": "insert_movie", "output": "full_result"}

    mock_command.return_value = {
        "success": True,
        "return_value": "result_value",
        "store_result": "result_name",
    }

    mock_rw = mock.MagicMock()
    mock_rw.recipe_step = {"parameters": ispyb_test_message}
    mock_rw.environment = ispyb_test_environment
    write_config_file(tmp_path)

    # Set up the mock service and call it
    service = ispyb_connector.EMISPyB(
        environment={"config": f"{tmp_path}/config.yaml", "queue": ""},
        transport=offline_transport,
    )
    service.initializing()
    service.insert_into_ispyb(rw=mock_rw, header=header, message=ispyb_test_message)

    # Check that the correct messages were sent
    mock_command.assert_called()
    mock_rw.set_default_channel.assert_called_with("output")
    mock_rw.send.assert_called_with({"result": "result_value"})

    # Check the results were stored in the environment
    assert mock_rw.environment["full_result"] == "result_value"
    assert mock_rw.environment["result_name"] == "result_value"


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.sqlalchemy")
@mock.patch("cryoemservices.util.ispyb_commands.models")
def test_ispyb_service_multipart_env_keys(
    mock_models, mock_sqlalchemy, mock_ispyb_api, offline_transport, tmp_path
):
    """
    Test as above, but with a nested $ key in a multipart message
    Using initial model as example of where this happens
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    ispyb_test_message = {
        "ispyb_command": "multipart_message",
        "ispyb_command_list": [
            {
                "ispyb_command": "insert_cryoem_initial_model",
                "particle_classification_id": 401,
                "cryoem_initial_model_id": "$ispyb_initial_model_id",
                "number_of_particles": 10000,
                "resolution": "6.6",
            }
        ],
    }
    ispyb_test_environment = {"output": "full_result", "ispyb_initial_model_id": 601}

    mock_rw = mock.MagicMock()
    mock_rw.recipe_step = {"parameters": ispyb_test_message}
    mock_rw.environment = ispyb_test_environment
    write_config_file(tmp_path)

    # Set up the mock service and call it
    service = ispyb_connector.EMISPyB(
        environment={"config": f"{tmp_path}/config.yaml", "queue": ""},
        transport=offline_transport,
    )
    service.initializing()
    service.insert_into_ispyb(rw=mock_rw, header=header, message=ispyb_test_message)

    # Check the sub-command calls were made
    mock_models.CryoemInitialModel.assert_not_called()
    mock_models.t_ParticleClassification_has_CryoemInitialModel.insert().values.assert_called_with(
        cryoemInitialModelId="601",
        particleClassificationId=401,
    )

    # Check that the correct messages were sent
    mock_rw.set_default_channel.assert_called_with("output")
    mock_rw.send.assert_called_with({"result": "601"})


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
        "ispyb_command": "insert_movie",
    }

    mock_command.return_value = {
        "checkpoint": True,
        "return_value": "dummy_result",
        "checkpoint_dict": {"checkpoint": 1},
    }

    mock_rw = mock.MagicMock()
    mock_rw.recipe_step = {"parameters": ispyb_test_message}
    write_config_file(tmp_path)

    # Set up the mock service and call it
    service = ispyb_connector.EMISPyB(
        environment={"config": f"{tmp_path}/config.yaml", "queue": ""},
        transport=offline_transport,
    )
    service.initializing()
    service.insert_into_ispyb(rw=mock_rw, header=header, message=ispyb_test_message)

    # Check that the correct messages were sent - this checkpoints but does not send
    mock_rw.set_default_channel.assert_not_called()
    mock_rw.send.assert_not_called()
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
    tmp_path,
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
                "buffer_command": {"ispyb_command": "insert_particle_classification"},
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
    }

    # The output will be the input, but without the first command as that has run
    output_commands = copy.deepcopy(ispyb_test_message["ispyb_command_list"])
    output_commands.pop(0)

    # After a second run two commands will have been removed
    second_output_commands = copy.deepcopy(ispyb_test_message["ispyb_command_list"])
    second_output_commands.pop(0)
    second_output_commands.pop(0)

    # Mock up the individual insert command
    mock_insert_group.return_value = {"success": True, "return_value": "dummy_group"}
    mock_insert_class.return_value = {"success": True, "return_value": "dummy_class"}
    mock_insert_model.return_value = {"success": True, "return_value": "dummy_model"}

    mock_rw = mock.MagicMock()
    mock_rw.recipe_step = {"parameters": {"dcid": 1000, "program_id": 100}}
    mock_rw.environment = {}
    write_config_file(tmp_path)

    # Set up the mock service and call it
    service = ispyb_connector.EMISPyB(
        environment={"config": f"{tmp_path}/config.yaml", "queue": ""},
        transport=offline_transport,
    )
    service.initializing()
    service.insert_into_ispyb(rw=mock_rw, header=header, message=ispyb_test_message)

    # Check that the correct messages were sent - this checkpoints but does not send
    mock_rw.send.assert_not_called()
    mock_rw.checkpoint.assert_any_call(
        {
            "checkpoint": 1,
            "expiry_time": mock.ANY,
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
            "expiry_time": mock.ANY,
            "ispyb_command": "multipart_message",
            "ispyb_command_list": output_commands,
        },
    )
    mock_rw.checkpoint.assert_called_with(
        {
            "checkpoint": 2,
            "expiry_time": mock.ANY,
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
            "expiry_time": mock.ANY,
            "ispyb_command": "multipart_message",
            "ispyb_command_list": second_output_commands,
        },
    )
    mock_rw.set_default_channel.assert_called_with("output")
    mock_rw.send.assert_called_with({"result": "dummy_model"})
    assert mock_rw.environment["ispyb_initial_model_id"] == "dummy_model"


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.sqlalchemy")
@mock.patch("cryoemservices.services.ispyb_connector.ispyb_commands.insert_movie")
def test_ispyb_service_failed_lookup(
    mock_command, mock_sqlalchemy, mock_ispyb_api, offline_transport, tmp_path
):
    """
    Send a test message to the ispyb service for a command which fails
    This currently nacks, but maybe should checkpoint ready for rerunning
    """
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    ispyb_test_message = {
        "ispyb_command": "insert_movie",
    }

    mock_command.return_value = False

    mock_rw = mock.MagicMock(environment={"config": f"{tmp_path}/config.yaml"})
    mock_rw.recipe_step = {"parameters": ispyb_test_message}
    write_config_file(tmp_path)

    # Set up the mock service and call it
    service = ispyb_connector.EMISPyB(
        environment={"config": f"{tmp_path}/config.yaml", "queue": ""},
        transport=offline_transport,
    )
    service.initializing()
    service.insert_into_ispyb(rw=mock_rw, header=header, message=ispyb_test_message)

    # Check that the correct messages were sent - this checkpoints but does not send
    mock_rw.set_default_channel.assert_not_called()
    mock_rw.send.assert_not_called()
    mock_rw.checkpoint.assert_not_called()
    mock_rw.transport.nack.assert_called()
