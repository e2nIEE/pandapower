# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import numpy as np

from pandapower.create import create_bus, create_buses, create_gen, create_gens
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network

def test_create_gen():
    # Test basic generator creation with required parameters
    net = pandapowerNet(name="test_create_gen")
    b1 = create_bus(net, 110.0)

    # Create generator with required parameters (bus, p_mw)
    gidx = create_gen(
        net,
        bus=b1,
        p_mw=50.0,
    )

    assert len(net.gen) == 1
    assert net.gen.at[gidx, "bus"] == b1
    assert np.isclose(net.gen.at[gidx, "p_mw"], 50.0)
    assert np.isclose(net.gen.at[gidx, "vm_pu"], 1.0)  # default value
    assert np.isclose(net.gen.at[gidx, "scaling"], 1.0)  # default value
    assert net.gen.at[gidx, "in_service"]  # default value
    assert not net.gen.at[gidx, "slack"]  # default value

    validate_network(net)


def test_create_gen_with_optional_params():
    # Test generator creation with optional parameters
    net = pandapowerNet(name="test_create_gen_with_optional_params")
    b1 = create_bus(net, 110.0)

    gidx = create_gen(
        net,
        bus=b1,
        p_mw=120.0,
        vm_pu=1.02,
        sn_mva=150.0,
        name="test_generator",
        scaling=0.8,
        type="sync",
        slack=True,
        max_p_mw=150.0,
        min_p_mw=20.0,
        max_q_mvar=50.0,
        min_q_mvar=-50.0,
        min_vm_pu=0.9,
        max_vm_pu=1.1,
        vn_kv=10.5,
        xdss_pu=0.2,
        rdss_ohm=0.01,
        cos_phi=0.85,
        pg_percent=10.0,
        in_service=True,
        slack_weight=1.0,
        test_kwargs="dummy_string",
        controllable=True,
    )

    assert len(net.gen) == 1
    assert net.gen.at[gidx, "bus"] == b1
    assert np.isclose(net.gen.at[gidx, "p_mw"], 120.0)
    assert np.isclose(net.gen.at[gidx, "vm_pu"], 1.02)
    assert np.isclose(net.gen.at[gidx, "sn_mva"], 150.0)
    assert net.gen.at[gidx, "name"] == "test_generator"
    assert np.isclose(net.gen.at[gidx, "scaling"], 0.8)
    assert net.gen.at[gidx, "type"] == "sync"
    assert net.gen.at[gidx, "slack"]
    assert np.isclose(net.gen.at[gidx, "max_p_mw"], 150.0)
    assert np.isclose(net.gen.at[gidx, "min_p_mw"], 20.0)
    assert np.isclose(net.gen.at[gidx, "max_q_mvar"], 50.0)
    assert np.isclose(net.gen.at[gidx, "min_q_mvar"], -50.0)
    assert np.isclose(net.gen.at[gidx, "min_vm_pu"], 0.9)
    assert np.isclose(net.gen.at[gidx, "max_vm_pu"], 1.1)
    assert np.isclose(net.gen.at[gidx, "vn_kv"], 10.5)
    assert np.isclose(net.gen.at[gidx, "xdss_pu"], 0.2)
    assert np.isclose(net.gen.at[gidx, "rdss_ohm"], 0.01)
    assert np.isclose(net.gen.at[gidx, "cos_phi"], 0.85)
    assert np.isclose(net.gen.at[gidx, "pg_percent"], 10.0)
    assert net.gen.at[gidx, "in_service"]
    assert np.isclose(net.gen.at[gidx, "slack_weight"], 1.0)
    assert net.gen.test_kwargs.at[gidx] == "dummy_string"

    validate_network(net)


def test_create_gen_out_of_service():
    # Test generator creation with in_service=False
    net = pandapowerNet(name="test_create_gen_out_of_service")
    b1 = create_bus(net, 110.0)

    gidx = create_gen(
        net,
        bus=b1,
        p_mw=50.0,
        in_service=False,
    )

    assert len(net.gen) == 1
    assert not net.gen.at[gidx, "in_service"]

    validate_network(net)


def test_create_gen_with_index():
    # Test generator creation with custom index
    net = pandapowerNet(name="test_create_gen_with_index")
    b1 = create_bus(net, 110.0)

    gidx = create_gen(
        net,
        bus=b1,
        p_mw=50.0,
        index=5,
    )

    assert gidx == 5
    assert len(net.gen) == 1

    validate_network(net)


def test_create_gen_nonexistent_bus():
    # Test that creating a generator with non-existent bus raises an error
    net = pandapowerNet(name="test_create_gen_nonexistent_bus")

    with pytest.raises(Exception):
        create_gen(
            net,
            bus=0,  # Bus doesn't exist
            p_mw=50.0,
        )


def test_create_gens():
    net = pandapowerNet(name="test_create_gens")
    # standard
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 110)
    create_gens(
        net,
        buses=[b1, b2, b3],
        p_mw=[0, 0, 1],
        vm_pu=1.0,
        controllable=[True, False, False],
        id_q_capability_characteristic=[0, 1, 2],
        reactive_capability_curve=False,
        curve_style=["straightLineYValues", "straightLineYValues", "straightLineYValues"],
        max_p_mw=0.2,
        min_p_mw=[0, 0.1, 0],
        max_q_mvar=0.2,
        min_q_mvar=[0, 0.1, 0],
        min_vm_pu=0.85,
        max_vm_pu=1.15,
        vn_kv=0.4,
        xdss_pu=0.1,
        rdss_pu=0.1,
        cos_phi=1.0,
        test_kwargs="dummy_string",
    )
    assert net.gen.bus.at[0] == b1
    assert net.gen.bus.at[1] == b2
    assert net.gen.bus.at[2] == b3
    assert net.gen.p_mw.at[0] == 0
    assert net.gen.p_mw.at[1] == 0
    assert net.gen.p_mw.at[2] == 1
    assert net.gen.controllable.dtype == bool
    assert net.gen.controllable.at[0]
    assert not net.gen.controllable.at[1]
    assert not net.gen.controllable.at[2]
    assert np.allclose(net.gen.max_p_mw, 0.2)
    assert all(net.gen.min_p_mw.values == [0, 0.1, 0])
    assert np.allclose(net.gen.max_q_mvar.values, 0.2)
    assert all(net.gen.min_q_mvar.values == [0, 0.1, 0])
    assert np.allclose(net.gen.min_vm_pu, 0.85)
    assert np.allclose(net.gen.max_vm_pu, 1.15)
    assert np.allclose(net.gen.vn_kv, 0.4)
    assert np.allclose(net.gen.xdss_pu, 0.1)
    assert np.allclose(net.gen.rdss_pu, 0.1)
    assert np.allclose(net.gen.cos_phi, 1.0)
    assert all(net.gen.test_kwargs == "dummy_string")
    assert all(net.gen.id_q_capability_characteristic.values == [0, 1, 2])
    assert all(net.gen.curve_style == "straightLineYValues")
    assert all(net.gen.reactive_capability_curve == [False, False, False])

    validate_network(net)


def test_create_gen_controllable():
    net = pandapowerNet(name="test_create_gen_controllable")

    b1 = create_bus(net, 110)
    s1 = create_gen(net, b1, 50)
    # controllable column should not exist
    assert 'controllable' not in net.gen.columns
    s2 = create_gen(net, b1, 50, controllable=True)
    # controllable should be created with default value False
    assert not net.gen.loc[s1, 'controllable']
    assert net.gen.loc[s2, 'controllable']


def test_create_gens_controllable():
    net = pandapowerNet(name="test_create_gens_controllable")

    b1 = create_bus(net, 110)
    s1 = create_gens(net, [b1], 50)[0]
    # controllable column should not exist
    assert 'controllable' not in net.gen.columns
    s2 = create_gens(net, [b1], 50, controllable=True)[0]
    # controllable should be created with default value False
    assert not net.gen.loc[s1, 'controllable']
    assert net.gen.loc[s2, 'controllable']


def test_create_gens_raise_errorexcept():
    net = pandapowerNet(name="test_create_gens_raise_errorexcept")
    # standard
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 110)

    with pytest.raises(
            UserWarning, match=r"Cannot attach to buses \{3, 4, 5\}, they do not exist"
    ):
        create_gens(
            net,
            buses=[3, 4, 5],
            p_mw=[0, 0, 1],
            vm_pu=1.0,
            controllable=[True, False, False],
            max_p_mw=0.2,
            min_p_mw=[0, 0.1, 0],
            max_q_mvar=0.2,
            min_q_mvar=[0, 0.1, 0],
            min_vm_pu=0.85,
            max_vm_pu=1.15,
            vn_kv=0.4,
            xdss_pu=0.1,
            rdss_pu=0.1,
            cos_phi=1.0,
        )
    g = create_gens(
        net,
        buses=[b1, b2, b3],
        p_mw=[0, 0, 1],
        vm_pu=1.0,
        controllable=[True, False, False],
        max_p_mw=0.2,
        min_p_mw=[0, 0.1, 0],
        max_q_mvar=0.2,
        min_q_mvar=[0, 0.1, 0],
        min_vm_pu=0.85,
        max_vm_pu=1.15,
        vn_kv=0.4,
        xdss_pu=0.1,
        rdss_pu=0.1,
        cos_phi=1.0,
    )

    with pytest.raises(UserWarning, match=r"Gens with indexes \[0 1 2\] already exist"):
        create_gens(
            net,
            buses=[b1, b2, b3],
            p_mw=[0, 0, 1],
            vm_pu=1.0,
            controllable=[True, False, False],
            max_p_mw=0.2,
            min_p_mw=[0, 0.1, 0],
            max_q_mvar=0.2,
            min_q_mvar=[0, 0.1, 0],
            min_vm_pu=0.85,
            max_vm_pu=1.15,
            vn_kv=0.4,
            xdss_pu=0.1,
            rdss_pu=0.1,
            cos_phi=1.0,
            index=g,
        )

    validate_network(net)

def test_create_gen_uses_bus_voltage_limits_as_defaults():
    net = pandapowerNet(name="test_create_gen_uses_bus_voltage_limits_as_defaults")
    bus = create_bus(net, vn_kv=20.0, min_vm_pu=0.95, max_vm_pu=1.05)

    gen = create_gen(net, bus=bus, p_mw=1.0)

    assert np.isclose(net.gen.at[gen, "min_vm_pu"], 0.95)
    assert np.isclose(net.gen.at[gen, "max_vm_pu"], 1.05)


def test_create_gen_keeps_explicit_voltage_limits():
    net = pandapowerNet(name="test_create_gen_keeps_explicit_voltage_limits")
    bus = create_bus(net, vn_kv=20.0, min_vm_pu=0.95, max_vm_pu=1.05)

    gen = create_gen(
        net,
        bus=bus,
        p_mw=1.0,
        min_vm_pu=0.90,
        max_vm_pu=1.10,
    )

    assert np.isclose(net.gen.at[gen, "min_vm_pu"], 0.90)
    assert np.isclose(net.gen.at[gen, "max_vm_pu"], 1.10)


def test_create_gens_use_bus_voltage_limits_as_defaults():
    net = pandapowerNet(name="test_create_gens_use_bus_voltage_limits_as_defaults")
    buses = create_buses(
        net,
        nr_buses=2,
        vn_kv=20.0,
        min_vm_pu=[0.95, 0.96],
        max_vm_pu=[1.05, 1.06],
    )

    gens = create_gens(net, buses=buses, p_mw=[1.0, 2.0])

    assert np.allclose(net.gen.loc[gens, "min_vm_pu"].values, [0.95, 0.96])
    assert np.allclose(net.gen.loc[gens, "max_vm_pu"].values, [1.05, 1.06])

