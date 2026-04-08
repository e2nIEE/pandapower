# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from copy import deepcopy

import numpy as np
import pandas as pd

from pandapower.create import create_empty_network, create_bus, create_ward, create_wards
from pandapower.toolbox import nets_equal
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_create_ward(): raise NotImplementedError()


def test_create_wards():
    net = create_empty_network()
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
    # assert net.ward.name.at[1] == None
    assert pd.isna(net.ward.name.at[1])  #TODO: recheck, if <NA> would also be ok
    assert net.ward.name.at[2] == "123"
    assert net.ward.in_service.at[0]
    assert not net.ward.in_service.at[1]
    assert not net.ward.in_service.at[2]
    assert nets_equal(net, net_bulk)

    validate_network(net)


def test_create_xward(): raise NotImplementedError()
