from __future__ import annotations

import copy
from pathlib import Path
from unittest import mock

import pytest
from workflows.transport.offline_transport import OfflineTransport

from cryoemservices.services.class3d import Class3D


@pytest.fixture
def offline_transport(mocker):
    transport = OfflineTransport()
    mocker.spy(transport, "send")
    mocker.spy(transport, "ack")
    mocker.spy(transport, "nack")
    return transport


@mock.patch("cryoemservices.services.class3d.run_class3d")
def test_class3d_service_failed_resends(mock_class3d, offline_transport, tmp_path):
    """Failures of the processing should lead to reinjection of the message"""

    def raise_exception():
        raise ValueError

    mock_class3d.side_effect = raise_exception

    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    class3d_test_message = {
        "batch_size": "100000",
        "class_uuids": "{'0': 10, '1': 11}",
        "class3d_dir": f"{tmp_path}/Class3D/job015",
        "class3d_grp_uuid": "5",
        "class3d_nr_classes": "2",
        "do_initial_model": False,
        "initial_model_file": f"{tmp_path}/initial_model.mrc",
        "mask_diameter": "190.0",
        "particle_diameter": "180",
        "particles_file": f"{tmp_path}/Select/job013/particles_100000.star",
        "picker_id": "6",
        "relion_options": {},
    }
    end_message = copy.deepcopy(class3d_test_message)

    # Set up and run the service
    service = Class3D(environment={"queue": ""}, rabbitmq_credentials=Path("."))
    service._transport = offline_transport
    service.initializing()
    service.class3d(None, header=header, message=class3d_test_message)

    end_message["requeue"] = 1
    offline_transport.send.assert_any_call("class3d", end_message)
    offline_transport.ack.assert_called_once()


def test_class3d_service_nack_on_requeue(offline_transport, tmp_path):
    """Messages reinjected 5 times should nack"""
    # Set up the parameters
    header = {
        "message-id": mock.sentinel,
        "subscription": mock.sentinel,
    }
    class3d_test_message = {
        "batch_size": "100000",
        "class_uuids": "{'0': 10, '1': 11}",
        "class3d_dir": f"{tmp_path}/Class3D/job015",
        "class3d_grp_uuid": "5",
        "class3d_nr_classes": "2",
        "do_initial_model": False,
        "initial_model_file": f"{tmp_path}/initial_model.mrc",
        "mask_diameter": "190.0",
        "particle_diameter": "180",
        "particles_file": f"{tmp_path}/Select/job013/particles_100000.star",
        "picker_id": "6",
        "relion_options": {},
        "requeue": 5,
    }

    # Set up and run the service
    service = Class3D(environment={"queue": ""}, rabbitmq_credentials=Path("."))
    service._transport = offline_transport
    service.initializing()
    service.class3d(None, header=header, message=class3d_test_message)

    assert offline_transport.send.call_count == 0
    offline_transport.nack.assert_called_once()
