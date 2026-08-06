# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from copy import deepcopy

import numpy as np
import pandas as pd

from pandapower.create import create_bus, create_ward, create_wards, create_xward
from pandapower.toolbox import nets_equal
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_create_ward():
    net = pandapowerNet(name="test_create_ward")
    bus = create_bus(net, 110)

    ward_id = create_ward(net, bus, ps_mw=1.0, qs_mvar=0.5, pz_mw=0.2, qz_mvar=0.1)

    assert ward_id == 0
    assert net.ward.bus.at[0] == bus
    assert np.isclose(net.ward.ps_mw.at[0], 1.0)
    assert np.isclose(net.ward.qs_mvar.at[0], 0.5)
    assert np.isclose(net.ward.pz_mw.at[0], 0.2)
    assert np.isclose(net.ward.qz_mvar.at[0], 0.1)
    assert net.ward.in_service.at[0]

    # Test with all parameters
    bus2 = create_bus(net, 110)
    ward_id2 = create_ward(net, bus2, 2.0, 1.0, 0.4, 0.2, name="test_ward", in_service=False)

    assert ward_id2 == 1
    assert net.ward.bus.at[1] == bus2
    assert np.isclose(net.ward.ps_mw.at[1], 2.0)
    assert np.isclose(net.ward.qs_mvar.at[1], 1.0)
    assert np.isclose(net.ward.pz_mw.at[1], 0.4)
    assert np.isclose(net.ward.qz_mvar.at[1], 0.2)
    assert net.ward.at[1, "name"] == "test_ward"
    assert not net.ward.in_service.at[1]


def test_create_wards():
    net = pandapowerNet(name="test_create_wards")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 110)
    net_bulk = deepcopy(net)
    vals = np.c_[[b1, b2, b3], np.reshape(np.arange(12), (3, 4)), ["asd", None, "123"], [True, False, False]]

    create_ward(net, *vals[0, :])
    create_ward(net, *vals[1, :])
    create_ward(net, *vals[2, :])

    create_wards(net_bulk, vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3], vals[:, 4],
                    vals[:, 5], vals[:, 6])

    assert net.ward.bus.at[0] == b1
    assert net.ward.bus.at[1] == b2
    assert net.ward.bus.at[2] == b3
    assert net.ward.ps_mw.at[0] == 0
    assert net.ward.ps_mw.at[1] == 4
    assert net.ward.ps_mw.at[2] == 8
    assert net.ward.qs_mvar.at[0] == 1
    assert net.ward.qs_mvar.at[1] == 5
    assert net.ward.qs_mvar.at[2] == 9
    assert net.ward.pz_mw.at[0] == 2
    assert net.ward.pz_mw.at[1] == 6
    assert net.ward.pz_mw.at[2] == 10
    assert net.ward.qz_mvar.at[0] == 3
    assert net.ward.qz_mvar.at[1] == 7
    assert net.ward.qz_mvar.at[2] == 11
    assert net.ward.name.at[0] == "asd"
    assert pd.isna(net.ward.name.at[1])
    assert net.ward.name.at[2] == "123"
    assert net.ward.in_service.at[0]
    assert not net.ward.in_service.at[1]
    assert not net.ward.in_service.at[2]
    assert nets_equal(net, net_bulk)

    validate_network(net)


def test_create_xward():
    net = pandapowerNet(name="test_create_xward")
    bus = create_bus(net, 110)

    xward_id = create_xward(
        net, bus,
        ps_mw=1.0,
        qs_mvar=0.5,
        pz_mw=0.2,
        qz_mvar=0.1,
        r_ohm=10.0,
        x_ohm=5.0,
        vm_pu=1.0
    )

    assert xward_id == 0
    assert net.xward.bus.at[0] == bus
    assert np.isclose(net.xward.ps_mw.at[0], 1.0)
    assert np.isclose(net.xward.qs_mvar.at[0], 0.5)
    assert np.isclose(net.xward.pz_mw.at[0], 0.2)
    assert np.isclose(net.xward.qz_mvar.at[0], 0.1)
    assert np.isclose(net.xward.r_ohm.at[0], 10.0)
    assert np.isclose(net.xward.x_ohm.at[0], 5.0)
    assert np.isclose(net.xward.vm_pu.at[0], 1.0)
    assert net.xward.in_service.at[0]
    assert np.isclose(net.xward.slack_weight.at[0], 0.0)

    # Test with all parameters
    bus2 = create_bus(net, 110)
    xward_id2 = create_xward(
        net, bus2,
        ps_mw=2.0, qs_mvar=1.0,
        pz_mw=0.4, qz_mvar=0.2,
        r_ohm=20.0, x_ohm=10.0, vm_pu=1.02,
        name="test_xward",
        in_service=False,
        slack_weight=1.5
    )

    assert xward_id2 == 1
    assert net.xward.bus.at[1] == bus2
    assert np.isclose(net.xward.ps_mw.at[1], 2.0)
    assert np.isclose(net.xward.qs_mvar.at[1], 1.0)
    assert np.isclose(net.xward.pz_mw.at[1], 0.4)
    assert np.isclose(net.xward.qz_mvar.at[1], 0.2)
    assert np.isclose(net.xward.r_ohm.at[1], 20.0)
    assert np.isclose(net.xward.x_ohm.at[1], 10.0)
    assert np.isclose(net.xward.vm_pu.at[1], 1.02)
    assert net.xward.name.at[1] == "test_xward"
    assert not net.xward.in_service.at[1]
    assert np.isclose(net.xward.slack_weight.at[1], 1.5)
