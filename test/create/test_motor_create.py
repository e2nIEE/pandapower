# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import numpy as np

from pandapower.create import create_bus, create_motor
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_create_motor():
    # Test basic motor creation with required parameters
    net = pandapowerNet(name="test_create_motor")
    b1 = create_bus(net, 110.0)

    # Create motor with required parameters
    midx = create_motor(
        net,
        bus=b1,
        pn_mech_mw=0.5,
        cos_phi=0.9,
    )

    assert len(net.motor) == 1
    assert net.motor.at[midx, "bus"] == b1
    assert np.isclose(net.motor.at[midx, "pn_mech_mw"], 0.5)
    assert np.isclose(net.motor.at[midx, "cos_phi"], 0.9)

    validate_network(net)


def test_create_motor_with_optional_params():
    # Test motor creation with optional parameters
    net = pandapowerNet(name="test_create_motor_with_optional_params")
    b1 = create_bus(net, 110.0)

    midx = create_motor(
        net,
        bus=b1,
        pn_mech_mw=0.120,
        cos_phi=0.9,
        name="test_motor",
        efficiency_percent=90.0,
        loading_percent=40.0,
        scaling=1.0,
        lrc_pu=6.0,
        vn_kv=0.6,
        rx=0.5,
        in_service=True,
        cos_phi_n=0.85,
        efficiency_n_percent=92.0,
        test_kwargs="dummy_string",
    )

    assert len(net.motor) == 1
    assert net.motor.at[midx, "bus"] == b1
    assert net.motor.at[midx, "name"] == "test_motor"
    assert np.isclose(net.motor.at[midx, "pn_mech_mw"], 0.120)
    assert np.isclose(net.motor.at[midx, "cos_phi"], 0.9)
    assert np.isclose(net.motor.at[midx, "efficiency_percent"], 90.0)
    assert np.isclose(net.motor.at[midx, "loading_percent"], 40.0)
    assert np.isclose(net.motor.at[midx, "scaling"], 1.0)
    assert np.isclose(net.motor.at[midx, "lrc_pu"], 6.0)
    assert np.isclose(net.motor.at[midx, "vn_kv"], 0.6)
    assert np.isclose(net.motor.at[midx, "rx"], 0.5)
    assert net.motor.at[midx, "in_service"]
    assert np.isclose(net.motor.at[midx, "cos_phi_n"], 0.85)
    assert np.isclose(net.motor.at[midx, "efficiency_n_percent"], 92.0)
    assert net.motor.test_kwargs.at[midx] == "dummy_string"

    validate_network(net)


def test_create_motor_out_of_service():
    # Test motor creation with in_service=False
    net = pandapowerNet(name="test_create_motor_out_of_service")
    b1 = create_bus(net, 110.0)

    midx = create_motor(
        net,
        bus=b1,
        pn_mech_mw=0.5,
        cos_phi=0.9,
        in_service=False,
    )

    assert len(net.motor) == 1
    assert not net.motor.at[midx, "in_service"]

    validate_network(net)


def test_create_motor_with_index():
    # Test motor creation with custom index
    net = pandapowerNet(name="test_create_motor_with_index")
    b1 = create_bus(net, 110.0)

    midx = create_motor(
        net,
        bus=b1,
        pn_mech_mw=0.5,
        cos_phi=0.9,
        index=5,
    )

    assert midx == 5
    assert len(net.motor) == 1

    validate_network(net)


def test_create_motor_nonexistent_bus():
    # Test that creating a motor with non-existent bus raises an error
    net = pandapowerNet(name="test_create_motor_nonexistent_bus")

    with pytest.raises(Exception):
        create_motor(
            net,
            bus=0,  # Bus doesn't exist
            pn_mech_mw=0.5,
            cos_phi=0.9,
        )
