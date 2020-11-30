# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import pandas as pd
import numpy as np

import pandapower as pp
import pandapower.networks as nw
import pandapower.control
import pandapower.timeseries
import logging as log

logger = log.getLogger(__name__)


class ModifiedConstControl(pp.control.ConstControl):
    def modify_values(self):
        self.values = np.square(self.values)


def test_write():
    net = nw.simple_four_bus_system()
    ds = pp.timeseries.DFData(pd.DataFrame(data=[[0., 1., 2.], [2., 3., 4.]]))
    c1 = pp.control.ConstControl(net, 'sgen', 'p_mw', element_index=[0, 1], profile_name=[0, 1], data_source=ds)
    pp.create_sgen(net, 0, 0)
    c2 = pp.control.ConstControl(net, 'sgen', 'p_mw', element_index=[2], profile_name=[2], data_source=ds,
                                 scale_factor=0.5)
    for t in range(2):
        c1.time_step(net, t)
        c1.control_step(net)
        c2.time_step(net, t)
        c2.control_step(net)
        assert np.all(net.sgen.p_mw.values == ds.df.loc[t].values * np.array([1, 1, 0.5]))


def test_modified_values():
    net = nw.simple_four_bus_system()
    ds = pp.timeseries.DFData(pd.DataFrame(data=[[0., 1., 2.], [2., 3., 4.]]))
    c1 = ModifiedConstControl(net, 'sgen', 'p_mw', element_index=[0, 1], profile_name=[0, 1], data_source=ds,
                              scale_factor=0.5)
    pp.create_sgen(net, 0, 0)
    c2 = ModifiedConstControl(net, 'sgen', 'p_mw', element_index=[2], profile_name=[2], data_source=ds,
                              scale_factor=0.25)
    for t in range(2):
        c1.time_step(net, t)
        c1.control_step(net)
        c2.time_step(net, t)
        c2.control_step(net)
        assert np.all(net.sgen.p_mw.values == np.square(ds.df.loc[t].values * np.array([0.5, 0.5, 0.25])))


if __name__ == '__main__':
    pytest.main([__file__])
