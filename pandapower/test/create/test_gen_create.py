# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest

from pandapower.create import create_empty_network, create_bus, create_gen, create_gens
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_create_gens():
    net = create_empty_network()
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
    assert all(net.gen.max_p_mw.values == 0.2)
    assert all(net.gen.min_p_mw.values == [0, 0.1, 0])
    assert all(net.gen.max_q_mvar.values == 0.2)
    assert all(net.gen.min_q_mvar.values == [0, 0.1, 0])
    assert all(net.gen.min_vm_pu.values == 0.85)
    assert all(net.gen.max_vm_pu.values == 1.15)
    assert all(net.gen.vn_kv.values == 0.4)
    assert all(net.gen.xdss_pu.values == 0.1)
    assert all(net.gen.rdss_pu.values == 0.1)
    assert all(net.gen.cos_phi.values == 1.0)
    assert all(net.gen.test_kwargs == "dummy_string")
    assert all(net.gen.id_q_capability_characteristic.values == [0, 1, 2])
    assert all(net.gen.curve_style == "straightLineYValues")
    assert all(net.gen.reactive_capability_curve == [False, False, False])

    validate_network(net)


def test_create_gen_controllable():
    net = create_empty_network()

    b1 = create_bus(net, 110)
    s1 = create_gen(net, b1, 50)
    # controllable column should not exist
    assert 'controllable' not in net.gen.columns
    s2 = create_gen(net, b1, 50, controllable=True)
    # controllable should be created with default value False
    assert not net.gen.loc[s1, 'controllable']
    assert net.gen.loc[s2, 'controllable']


def test_create_gens_controllable():
    net = create_empty_network()

    b1 = create_bus(net, 110)
    s1 = create_gens(net, [b1], 50)[0]
    # controllable column should not exist
    assert 'controllable' not in net.gen.columns
    s2 = create_gens(net, [b1], 50, controllable=True)[0]
    # controllable should be created with default value False
    assert not net.gen.loc[s1, 'controllable']
    assert net.gen.loc[s2, 'controllable']


def test_create_gens_raise_errorexcept():
    net = create_empty_network()
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

