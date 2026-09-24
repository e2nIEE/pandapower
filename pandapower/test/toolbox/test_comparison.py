# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy

import numpy as np
import pandas as pd
import pytest

from pandapower.control.controller.trafo.ContinuousTapControl import ContinuousTapControl
from pandapower.create import create_bus
from pandapower.networks.cigre_networks import create_cigre_network_lv
from pandapower.toolbox.comparison import dataframes_equal, nets_equal, logger as tbc_logger


@pytest.mark.parametrize(
    "left,right",
    [(None, np.nan), (None, pd.NA), (None, pd.NaT), (np.nan, pd.NA), (np.nan, pd.NaT), (pd.NA, pd.NaT)],
)
def test_dataframes_equal_object_null_values(left, right):
    df1 = pd.DataFrame({"value": pd.Series([left], dtype=object)})
    df2 = pd.DataFrame({"value": pd.Series([right], dtype=object)})
    df1_original = df1.copy(deep=True)
    df2_original = df2.copy(deep=True)

    assert dataframes_equal(df1, df2)
    pd.testing.assert_frame_equal(df1, df1_original)
    pd.testing.assert_frame_equal(df2, df2_original)


def test_dataframes_equal_preserves_value_and_dtype_checks():
    assert not dataframes_equal(pd.DataFrame({"value": [1]}), pd.DataFrame({"value": [2]}))
    assert not dataframes_equal(
        pd.DataFrame({"value": pd.Series([1], dtype=object)}),
        pd.DataFrame({"value": pd.Series([1], dtype="int64")}),
    )


def test_nets_equal():
    tbc_logger.setLevel(40)
    original = create_cigre_network_lv()
    net = copy.deepcopy(original)

    # should be equal
    assert nets_equal(original, net)
    assert nets_equal(net, original)

    # detecting additional element
    create_bus(net, vn_kv=.4)
    assert not nets_equal(original, net)
    assert not nets_equal(net, original)
    net = copy.deepcopy(original)

    # detecting removed element
    net["bus"] = net["bus"].drop(net.bus.index[0])
    assert not nets_equal(original, net)
    assert not nets_equal(net, original)
    net = copy.deepcopy(original)

    # detecting alternated value
    net["load"].loc[net["load"].index[0], "p_mw"] += 0.1
    assert not nets_equal(original, net)
    assert not nets_equal(net, original)
    net = copy.deepcopy(original)

    # detecting added column
    net["load"]["new_col"] = 0.1
    assert not nets_equal(original, net)
    assert not nets_equal(net, original)
    net = copy.deepcopy(original)

    # not detecting alternated value if difference is beyond tolerance
    net["load"].loc[net["load"].index[0], "p_mw"] += 0.0001
    assert nets_equal(original, net, atol=0.1)
    assert nets_equal(net, original, atol=0.1)

    # check controllers
    original.trafo.tap_side = original.trafo.tap_side.fillna("hv")
    net1 = copy.deepcopy(original)
    net2 = copy.deepcopy(original)
    ContinuousTapControl(net1, 0, 1.0)
    ContinuousTapControl(net2, 0, 1.0)
    c1 = net1.controller.at[0, "object"]
    c2 = net2.controller.at[0, "object"]
    assert c1 == c2
    assert c1 is not c2
    assert nets_equal(net1, net2)
    c1.vm_set_pu = 1.01
    assert c1 != c2
    assert nets_equal(net1, net2, exclude_elms=["controller"])
    assert not nets_equal(net1, net2)

    # check geo
    net1 = copy.deepcopy(original)
    net2 = copy.deepcopy(original)
    net1.bus["geo"] = pd.Series(index=net1.bus.index, data=[100] * len(net1.bus.index))
    net2.bus["geo"] = pd.Series(index=net2.bus.index, data=[100] * len(net2.bus.index))
    assert nets_equal(
        net1, net2, assume_geojson_strings=False, exclude_elms=set(net.keys()) - {"bus"}
    )


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
