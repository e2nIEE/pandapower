# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest

from pandapower.create import create_empty_network, create_bus, create_loads
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_create_loads():
    net = create_empty_network()
    # standard
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 110)
    create_loads(
        net,
        buses=[b1, b2, b3],
        p_mw=[0, 0, 1],
        q_mvar=0.0,
        controllable=[True, False, False],
        max_p_mw=0.2,
        min_p_mw=[0, 0.1, 0],
        max_q_mvar=0.2,
        min_q_mvar=[0, 0.1, 0],
        test_kwargs=["dummy_string_1", "dummy_string_2", "dummy_string_3"],
    )

    assert net.load.bus.at[0] == b1
    assert net.load.bus.at[1] == b2
    assert net.load.bus.at[2] == b3
    assert net.load.p_mw.at[0] == 0
    assert net.load.p_mw.at[1] == 0
    assert net.load.p_mw.at[2] == 1
    assert net.load.q_mvar.at[0] == 0
    assert net.load.q_mvar.at[1] == 0
    assert net.load.q_mvar.at[2] == 0
    assert net.load.controllable.dtype == bool
    assert net.load.controllable.at[0]
    assert not net.load.controllable.at[1]
    assert not net.load.controllable.at[2]
    assert all(net.load.max_p_mw.values == 0.2)
    assert all(net.load.min_p_mw.values == [0, 0.1, 0])
    assert all(net.load.max_q_mvar.values == 0.2)
    assert all(net.load.min_q_mvar.values == [0, 0.1, 0])
    assert all(
        net.load.test_kwargs.values
        == ["dummy_string_1", "dummy_string_2", "dummy_string_3"]
    )

    validate_network(net)


def test_create_loads_raise_errorexcept():
    net = create_empty_network()
    # standard
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 110)

    with pytest.raises(
            UserWarning, match=r"Cannot attach to buses {3, 4, 5}, they do not exist"
    ):
        create_loads(
            net,
            buses=[3, 4, 5],
            p_mw=[0, 0, 1],
            q_mvar=0.0,
            controllable=[True, False, False],
            max_p_mw=0.2,
            min_p_mw=[0, 0.1, 0],
            max_q_mvar=0.2,
            min_q_mvar=[0, 0.1, 0],
        )
    l = create_loads(
        net,
        buses=[b1, b2, b3],
        p_mw=[0, 0, 1],
        q_mvar=0.0,
        controllable=[True, False, False],
        max_p_mw=0.2,
        min_p_mw=[0, 0.1, 0],
        max_q_mvar=0.2,
        min_q_mvar=[0, 0.1, 0],
    )
    with pytest.raises(
            UserWarning, match=r"Loads with indexes \[0 1 2\] already exist"
    ):
        create_loads(
            net,
            buses=[b1, b2, b3],
            p_mw=[0, 0, 1],
            q_mvar=0.0,
            controllable=[True, False, False],
            max_p_mw=0.2,
            min_p_mw=[0, 0.1, 0],
            max_q_mvar=0.2,
            min_q_mvar=[0, 0.1, 0],
            index=l,
        )

    validate_network(net)

