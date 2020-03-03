# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest
import copy
import numpy as np
import pandas as pd

from pandapower.auxiliary import get_indices

import pandapower as pp
import pandapower.networks
import pandapower.control
import pandapower.timeseries


def test_get_indices():
    a = [i+100 for i in range(10)]
    lookup = {idx: pos for pos, idx in enumerate(a)}
    lookup["before_fuse"] = a

    # First without fused buses no magic here
    # after fuse
    result = get_indices([102, 107], lookup, fused_indices=True)
    assert np.array_equal(result, [2, 7])

    # before fuse
    result = get_indices([2, 7], lookup, fused_indices=False)
    assert np.array_equal(result, [102, 107])

    # Same setup EXCEPT we have fused buses now (bus 102 and 107 are fused)
    lookup[107] = lookup[102]

    # after fuse
    result = get_indices([102, 107], lookup, fused_indices=True)
    assert np.array_equal(result, [2, 2])

    # before fuse
    result = get_indices([2, 7], lookup, fused_indices=False)
    assert np.array_equal(result, [102, 107])


def test_net_deepcopy():
    net = pp.networks.example_simple()
    net.line_geodata.loc[0, 'coords'] = [[0,1], [1,2]]

    pp.control.ContinuousTapControl(net, tid=0, vm_set_pu=1)
    ds = pp.timeseries.DFData(pd.DataFrame(data=[[0,1,2], [3,4,5]]))
    pp.control.ConstControl(net, element='load', variable='p_mw', element_index=[0], profile_name=[0], data_source=ds)

    net1 = copy.deepcopy(net)
    assert net1.controller.object.at[0].net is net1
    assert net1.controller.object.at[1].net is net1

    assert not net1.controller.object.at[0].net is net
    assert not net1.controller.object.at[1].net is net

    assert not net1.controller.object.at[1].data_source is ds
    assert not net1.controller.object.at[1].data_source.df is ds.df

    assert not net1.line_geodata.coords.at[0] is net.line_geodata.coords.at[0]


if __name__ == '__main__':
    pytest.main([__file__, "-x"])
