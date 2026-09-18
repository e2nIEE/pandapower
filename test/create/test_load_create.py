# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import numpy as np

from pandapower.create import (
    create_bus,
    create_load,
    create_loads,
    create_asymmetric_load,
    create_load_from_cosphi,
    create_bus_dc,
    create_load_dc,
)
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_create_load():
    net = pandapowerNet(name="test_create_load")
    b1 = create_bus(net, 110)

    # Test basic load creation
    idx = create_load(net, bus=b1, p_mw=10.0, q_mvar=2.0)
    assert net.load.at[idx, "bus"] == b1
    assert np.isclose(net.load.at[idx, "p_mw"], 10.0)
    assert np.isclose(net.load.at[idx, "q_mvar"], 2.0)

    # Test load with all optional parameters
    idx2 = create_load(
        net, bus=b1, p_mw=5.0, q_mvar=1.0,
        name="test_load",
        sn_mva=6.0,
        scaling=0.8,
        in_service=False,
        type="delta",
        max_p_mw=6.0,
        min_p_mw=4.0,
        max_q_mvar=1.5,
        min_q_mvar=0.5,
        controllable=True,
    )
    assert net.load.at[idx2, "name"] == "test_load"
    assert np.isclose(net.load.at[idx2, "sn_mva"], 6.0)
    assert np.isclose(net.load.at[idx2, "scaling"], 0.8)
    assert net.load.at[idx2, "in_service"] == False
    assert net.load.at[idx2, "type"] == "delta"
    assert np.isclose(net.load.at[idx2, "max_p_mw"], 6.0)
    assert np.isclose(net.load.at[idx2, "min_p_mw"], 4.0)
    assert np.isclose(net.load.at[idx2, "max_q_mvar"], 1.5)
    assert np.isclose(net.load.at[idx2, "min_q_mvar"], 0.5)
    assert net.load.at[idx2, "controllable"] == True

    # Test load with const_z/const_i percent
    idx3 = create_load(
        net, bus=b1, p_mw=3.0, q_mvar=0.5,
        const_z_p_percent=50.0,
        const_i_p_percent=30.0,
        const_z_q_percent=40.0,
        const_i_q_percent=20.0,
    )
    assert np.isclose(net.load.at[idx3, "const_z_p_percent"], 50.0)
    assert np.isclose(net.load.at[idx3, "const_i_p_percent"], 30.0)
    assert np.isclose(net.load.at[idx3, "const_z_q_percent"], 40.0)
    assert np.isclose(net.load.at[idx3, "const_i_q_percent"], 20.0)

    validate_network(net)


def test_create_loads():
    net = pandapowerNet(name="test_create_loads")
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
    assert np.allclose(net.load.max_p_mw, 0.2)
    assert all(net.load.min_p_mw.values == [0, 0.1, 0])
    assert np.allclose(net.load.max_q_mvar, 0.2)
    assert all(net.load.min_q_mvar.values == [0, 0.1, 0])
    assert all(
        net.load.test_kwargs.values
        == ["dummy_string_1", "dummy_string_2", "dummy_string_3"]
    )

    validate_network(net)


def test_create_loads_raise_errorexcept():
    net = pandapowerNet(name="test_create_loads_raise_errorexcept")
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


def test_create_asymmetric_load():
    net = pandapowerNet(name="test_create_asymmetric_load")
    b1 = create_bus(net, 110)

    # Test basic asymmetric load creation
    idx = create_asymmetric_load(
        net, bus=b1,
        p_a_mw=1.0, p_b_mw=2.0, p_c_mw=3.0,
        q_a_mvar=0.1, q_b_mvar=0.2, q_c_mvar=0.3,
    )
    assert net.asymmetric_load.at[idx, "bus"] == b1
    assert np.isclose(net.asymmetric_load.at[idx, "p_a_mw"], 1.0)
    assert np.isclose(net.asymmetric_load.at[idx, "p_b_mw"], 2.0)
    assert np.isclose(net.asymmetric_load.at[idx, "p_c_mw"], 3.0)
    assert np.isclose(net.asymmetric_load.at[idx, "q_a_mvar"], 0.1)
    assert np.isclose(net.asymmetric_load.at[idx, "q_b_mvar"], 0.2)
    assert np.isclose(net.asymmetric_load.at[idx, "q_c_mvar"], 0.3)

    # Test asymmetric load with sn_mva values
    idx2 = create_asymmetric_load(
        net, bus=b1,
        p_a_mw=1.0, p_b_mw=2.0, p_c_mw=3.0,
        sn_a_mva=1.5, sn_b_mva=2.5, sn_c_mva=3.5, sn_mva=7.5,
        name="asym_test",
        scaling=0.9,
        in_service=False,
        type="delta",
    )
    assert np.isclose(net.asymmetric_load.at[idx2, "sn_a_mva"], 1.5)
    assert np.isclose(net.asymmetric_load.at[idx2, "sn_b_mva"], 2.5)
    assert np.isclose(net.asymmetric_load.at[idx2, "sn_c_mva"], 3.5)
    assert np.isclose(net.asymmetric_load.at[idx2, "sn_mva"], 7.5)
    assert net.asymmetric_load.at[idx2, "name"] == "asym_test"
    assert np.isclose(net.asymmetric_load.at[idx2, "scaling"], 0.9)
    assert net.asymmetric_load.at[idx2, "in_service"] == False
    assert net.asymmetric_load.at[idx2, "type"] == "delta"

    validate_network(net)


def test_create_load_from_cosphi():
    net = pandapowerNet(name="test_create_load_from_cosphi")
    b1 = create_bus(net, 110)

    # Test underexcited (inductive)
    idx1 = create_load_from_cosphi(
        net, bus=b1, sn_mva=10.0, cos_phi=0.9, mode="underexcited"
    )
    assert net.load.at[idx1, "bus"] == b1
    assert np.isclose(net.load.at[idx1, "sn_mva"], 10.0)
    assert net.load.at[idx1, "p_mw"] > 0
    assert net.load.at[idx1, "q_mvar"] > 0  # underexcited has positive Q

    # Test overexcited (capacitive)
    idx2 = create_load_from_cosphi(
        net, bus=b1, sn_mva=5.0, cos_phi=0.8, mode="overexcited"
    )
    assert net.load.at[idx2, "p_mw"] > 0
    assert net.load.at[idx2, "q_mvar"] < 0  # overexcited has negative Q

    # Test with additional parameters
    idx3 = create_load_from_cosphi(
        net, bus=b1, sn_mva=2.0, cos_phi=0.95, mode="underexcited",
        name="cosphi_load", scaling=1.2
    )
    assert net.load.at[idx3, "name"] == "cosphi_load"
    assert np.isclose(net.load.at[idx3, "scaling"], 1.2)

    validate_network(net)


def test_create_load_dc():
    net = pandapowerNet(name="test_create_load_dc")
    b1 = create_bus_dc(net, 100)

    # Test basic DC load creation
    idx = create_load_dc(net, bus_dc=b1, p_dc_mw=5.0)
    assert net.load_dc.at[idx, "bus_dc"] == b1
    assert np.isclose(net.load_dc.at[idx, "p_dc_mw"], 5.0)

    # Test DC load with all optional parameters
    idx2 = create_load_dc(
        net, bus_dc=b1, p_dc_mw=2.5,
        name="dc_load_test",
        scaling=0.75,
        in_service=False,
        type="resistive",
        controllable=True,
    )
    assert net.load_dc.at[idx2, "name"] == "dc_load_test"
    assert np.isclose(net.load_dc.at[idx2, "scaling"], 0.75)
    assert net.load_dc.at[idx2, "in_service"] == False
    assert net.load_dc.at[idx2, "type"] == "resistive"
    assert net.load_dc.at[idx2, "controllable"] == True

    validate_network(net)
