# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pandas as pd
import numpy as np

from pandapower.create import create_bus, create_storage, create_storages
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network

def _check_storage_table(table: pd.DataFrame, buses: tuple[int, int, int]):
    assert table.bus.at[0] == buses[0]
    assert table.bus.at[1] == buses[1]
    assert table.bus.at[2] == buses[2]
    assert table.p_mw.at[0] == 0
    assert table.p_mw.at[1] == 0
    assert table.p_mw.at[2] == 1
    assert table.max_e_mwh.at[0] == 3
    assert table.max_e_mwh.at[1] == 5
    assert table.max_e_mwh.at[2] == 7
    assert np.isclose(table.q_mvar.at[0], 0.5)
    assert np.isclose(table.q_mvar.at[1], 0.5)
    assert np.isclose(table.q_mvar.at[2], 0.5)
    assert table.controllable.dtype == bool
    assert table.controllable.at[0]
    assert not table.controllable.at[1]
    assert not table.controllable.at[2]
    assert np.allclose(table.max_p_mw, 0.2)
    assert all(table.min_p_mw.values == [0, 0.1, 0])  # type: ignore[arg-type]
    assert np.allclose(table.max_q_mvar, 0.2)
    assert all(table.min_q_mvar.values == [0, 0.1, 0])  # type: ignore[arg-type]
    assert all(table.test_kwargs.values == ["dummy_string_1", "dummy_string_2", "dummy_string_3"])  # type: ignore[arg-type]


def test_create_storage():
    net = pandapowerNet(name="test_create_storage")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 110)

    create_storage(
        net, b1, 0, 3, 0.5, controllable=True, max_p_mw=0.2, min_p_mw=0, max_q_mvar=0.2,
        min_q_mvar=0, test_kwargs="dummy_string_1"
    )
    create_storage(
        net, b2, 0, 5, 0.5, controllable=False, max_p_mw=0.2, min_p_mw=0.1, max_q_mvar=0.2,
        min_q_mvar=0.1, test_kwargs="dummy_string_2"
    )
    create_storage(
        net, b3, 1, 7, 0.5, max_p_mw=0.2, min_p_mw=0, max_q_mvar=0.2, min_q_mvar=0,
        test_kwargs="dummy_string_3"
    )

    _check_storage_table(net.storage, (b1, b2, b3))
    validate_network(net)


def test_create_storages():
    net = pandapowerNet(name="test_create_storages")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 110)

    create_storages(
        net,
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

    _check_storage_table(net.storage, (b1, b2, b3))
    validate_network(net)
