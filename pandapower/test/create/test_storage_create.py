# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from copy import deepcopy

from pandapower.create import create_empty_network, create_bus, create_storage, create_storages
from pandapower.toolbox import nets_equal
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_create_storage(): raise NotImplementedError()


def test_create_storages():
    net = create_empty_network()
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 110)
    net_bulk = deepcopy(net)

    create_storage(net, b1, 0, 3, 0.5, controllable=True, max_p_mw=0.2, min_p_mw=0,
                      max_q_mvar=0.2, min_q_mvar=0, test_kwargs="dummy_string_1")
    create_storage(net, b2, 0, 5, 0.5, controllable=False, max_p_mw=0.2, min_p_mw=0.1,
                      max_q_mvar=0.2, min_q_mvar=0.1, test_kwargs="dummy_string_2")
    create_storage(net, b3, 1, 7, 0.5, max_p_mw=0.2, min_p_mw=0,
                      max_q_mvar=0.2, min_q_mvar=0, test_kwargs="dummy_string_3")

    create_storages(
        net_bulk,
        buses=[b1, b2, b3],
        p_mw=[0, 0, 1],
        max_e_mwh=[3, 5, 7],
        q_mvar=0.5,
        controllable=[True, False, False],
        max_p_mw=0.2,
        min_p_mw=[0, 0.1, 0],
        max_q_mvar=0.2,
        min_q_mvar=[0, 0.1, 0],
        test_kwargs=["dummy_string_1", "dummy_string_2", "dummy_string_3"],
    )

    assert net.storage.bus.at[0] == b1
    assert net.storage.bus.at[1] == b2
    assert net.storage.bus.at[2] == b3
    assert net.storage.p_mw.at[0] == 0
    assert net.storage.p_mw.at[1] == 0
    assert net.storage.p_mw.at[2] == 1
    assert net.storage.max_e_mwh.at[0] == 3
    assert net.storage.max_e_mwh.at[1] == 5
    assert net.storage.max_e_mwh.at[2] == 7
    assert net.storage.q_mvar.at[0] == 0.5
    assert net.storage.q_mvar.at[1] == 0.5
    assert net.storage.q_mvar.at[2] == 0.5
    assert net.storage.controllable.dtype == bool
    assert net.storage.controllable.at[0]
    assert not net.storage.controllable.at[1]
    assert not net.storage.controllable.at[2]
    assert all(net.storage.max_p_mw.values == 0.2)
    assert all(net.storage.min_p_mw.values == [0, 0.1, 0])
    assert all(net.storage.max_q_mvar.values == 0.2)
    assert all(net.storage.min_q_mvar.values == [0, 0.1, 0])
    assert all(net.storage.test_kwargs.values == ["dummy_string_1", "dummy_string_2", "dummy_string_3"])
    for col in ["name", "type"]:
        if col in net.storage.columns:
            net.storage.loc[net.storage[col].isnull(), col] = "" #TODO: why is this here ?
    assert nets_equal(net, net_bulk)

    validate_network(net)
