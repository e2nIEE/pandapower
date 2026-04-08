# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest

from pandapower.create import create_empty_network, create_bus, create_sgen, create_sgens
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_create_sgens():
    net = create_empty_network()
    # standard
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 110)
    create_sgens(
        net,
        buses=[b1, b2, b3],
        p_mw=[0, 0, 1],
        q_mvar=0.0,
        controllable=[True, False, False],
        id_q_capability_characteristic=[0, 1, 2],
        reactive_capability_curve=False,
        curve_style=["straightLineYValues", "straightLineYValues", "straightLineYValues"],
        max_p_mw=0.2,
        min_p_mw=[0, 0.1, 0],
        max_q_mvar=0.2,
        min_q_mvar=[0, 0.1, 0],
        k=1.3,
        rx=0.4,
        current_source=True,
        test_kwargs="dummy_string",
    )

    assert net.sgen.bus.at[0] == b1
    assert net.sgen.bus.at[1] == b2
    assert net.sgen.bus.at[2] == b3
    assert net.sgen.p_mw.at[0] == 0
    assert net.sgen.p_mw.at[1] == 0
    assert net.sgen.p_mw.at[2] == 1
    assert net.sgen.q_mvar.at[0] == 0
    assert net.sgen.q_mvar.at[1] == 0
    assert net.sgen.q_mvar.at[2] == 0
    assert net.sgen.controllable.dtype == bool
    assert net.sgen.controllable.at[0]
    assert not net.sgen.controllable.at[1]
    assert not net.sgen.controllable.at[2]
    assert all(net.sgen.max_p_mw.values == 0.2)
    assert all(net.sgen.min_p_mw.values == [0, 0.1, 0])
    assert all(net.sgen.max_q_mvar.values == 0.2)
    assert all(net.sgen.min_q_mvar.values == [0, 0.1, 0])
    assert all(net.sgen.k.values == 1.3)
    assert all(net.sgen.rx.values == 0.4)
    assert all(net.sgen.current_source)
    assert all(net.sgen.test_kwargs == "dummy_string")
    assert all(net.sgen.id_q_capability_characteristic.values == [0, 1, 2])
    assert all(net.sgen.curve_style == "straightLineYValues")
    assert all(net.sgen.reactive_capability_curve == [False, False, False])


def test_create_sgen_controllable():
    net = create_empty_network()

    b1 = create_bus(net, 110)
    s1 = create_sgen(net, b1, 50)
    # controllable column should not exist
    assert 'controllable' not in net.sgen.columns
    s2 = create_sgen(net, b1, 50, controllable=True)
    # controllable should be created with default value False
    assert not net.sgen.loc[s1, 'controllable']
    assert net.sgen.loc[s2, 'controllable']


def test_create_sgens_controllable():
    net = create_empty_network()

    b1 = create_bus(net, 110)
    s1 = create_sgens(net, [b1], 50)[0]
    # controllable column should not exist
    assert 'controllable' not in net.sgen.columns
    s2 = create_sgens(net, [b1], 50, controllable=True)[0]
    # controllable should be created with default value False
    assert not net.sgen.loc[s1, 'controllable']
    assert net.sgen.loc[s2, 'controllable']

    validate_network(net)


def test_create_sgens_raise_errorexcept():
    net = create_empty_network()
    # standard
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 110)

    with pytest.raises(
            UserWarning, match=r"Cannot attach to buses \{3, 4, 5\}, they do not exist"
    ):
        create_sgens(
            net,
            buses=[3, 4, 5],
            p_mw=[0, 0, 1],
            q_mvar=0.0,
            controllable=[True, False, False],
            max_p_mw=0.2,
            min_p_mw=[0, 0.1, 0],
            max_q_mvar=0.2,
            min_q_mvar=[0, 0.1, 0],
            k=1.3,
            rx=0.4,
            current_source=True,
        )
    sg = create_sgens(
        net,
        buses=[b1, b2, b3],
        p_mw=[0, 0, 1],
        q_mvar=0.0,
        controllable=[True, False, False],
        max_p_mw=0.2,
        min_p_mw=[0, 0.1, 0],
        max_q_mvar=0.2,
        min_q_mvar=[0, 0.1, 0],
        k=1.3,
        rx=0.4,
        current_source=True,
    )
    with pytest.raises(
            UserWarning, match=r"Sgens with indexes \[0 1 2\] already exist"
    ):
        create_sgens(
            net,
            buses=[b1, b2, b3],
            p_mw=[0, 0, 1],
            q_mvar=0.0,
            controllable=[True, False, False],
            max_p_mw=0.2,
            min_p_mw=[0, 0.1, 0],
            max_q_mvar=0.2,
            min_q_mvar=[0, 0.1, 0],
            k=1.3,
            rx=0.4,
            current_source=True,
            index=sg,
        )

    validate_network(net)
