# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from copy import deepcopy
import pytest
import pandas as pd

import pandapower as pp


def test_set_line_geodata_from_bus_geodata():
    net = pp.networks.case9()
    bus_geo_data = deepcopy(net.bus.geo)

    empty_line_geo = pd.Series(None, index=net.line.index, dtype=object) # ensure that line geo data
    assert not net.bus.geo.isnull().any() # ensure that bus geo data is available

    pp.plotting.simple_plot(net, show_plot=False) # test that plotting works with case9 file

    # --- create line geo data from complete net.bus.geo to empty net.line.geo
    net.line.geo = deepcopy(empty_line_geo) # ensure that line geo data is missing
    pp.plotting.set_line_geodata_from_bus_geodata(net)
    assert not net.line.geo.isnull().any()
    pp.plotting.simple_plot(net, show_plot=False) # test that plotting works with case9 file

    # --- create line geo data from complete net.bus.geo to incomplete net.line.geo
    net.line.at[2, "geo"] = None
    net.line.at[4, "geo"] = None
    pp.plotting.set_line_geodata_from_bus_geodata(net)
    assert not net.line.geo.isnull().any()
    pp.plotting.simple_plot(net, show_plot=False) # test that plotting works with case9 file

    # --- create line geo data from complete net.bus.geo to incomplete net.line.geo using overwrite
    net.line.at[2, "geo"] = None
    net.line.at[4, "geo"] = None
    pp.plotting.set_line_geodata_from_bus_geodata(net, overwrite=True)
    assert not net.line.geo.isnull().any()
    pp.plotting.simple_plot(net, show_plot=False) # test that plotting works with case9 file

    # --- create line geo data from incomplete net.bus.geo to incomplete net.line.geo
    # (-> no warning expected since all missing data can be filled by available bus data)
    net.bus.at[2, "geo"] = None
    net.bus.at[4, "geo"] = None
    net.line.at[0, "geo"] = None
    net.line.at[5, "geo"] = None
    pp.plotting.set_line_geodata_from_bus_geodata(net)
    assert not net.line.geo.isnull().any()
    net.bus.at[2, "geo"] = bus_geo_data.at[2]
    net.bus.at[4, "geo"] = bus_geo_data.at[4]
    pp.plotting.simple_plot(net, show_plot=False) # test that plotting works with case9 file

    # --- create line geo data from incomplete net.bus.geo to incomplete net.line.geo
    # using line_index (-> no warning expected since all missing data can be filled by available bus
    # data)
    net.bus.at[2, "geo"] = None
    net.bus.at[4, "geo"] = None
    net.line.at[6, "geo"] = None
    net.line.at[7, "geo"] = None
    pp.plotting.set_line_geodata_from_bus_geodata(net, line_index=[6, 7])
    assert not net.line.geo.isnull().any()
    net.bus.at[2, "geo"] = bus_geo_data.at[2]
    net.bus.at[4, "geo"] = bus_geo_data.at[4]
    pp.plotting.simple_plot(net, show_plot=False) # test that plotting works with case9 file

    # --- create line geo data from incomplete net.bus.geo to empty net.line.geo
    # (-> warning expected)
    net.bus.at[2, "geo"] = None
    net.bus.at[4, "geo"] = None
    net.line.geo = deepcopy(empty_line_geo) # ensure that line geo data is missing
    pp.plotting.set_line_geodata_from_bus_geodata(net)


if __name__ == "__main__":
    pytest.main([__file__])
