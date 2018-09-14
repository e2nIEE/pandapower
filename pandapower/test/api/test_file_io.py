# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os
import pytest
import copy

import pandas as pd
import pandapower as pp
from pandapower.test.toolbox import assert_net_equal, create_test_network, tempdir, net_in
from pandapower.io_utils import collect_all_dtypes_df, restore_all_dtypes


def test_pickle(net_in, tempdir):
    filename = os.path.join(tempdir, "testfile.p")
    pp.to_pickle(net_in, filename)
    net_out = pp.from_pickle(filename)
    assert_net_equal(net_in, net_out)


def test_excel(net_in, tempdir):
    filename = os.path.join(tempdir, "testfile.xlsx")
    pp.to_excel(net_in, filename)
    net_out = pp.from_excel(filename)
    assert_net_equal(net_in, net_out)

    pp.set_user_pf_options(net_in, tolerance_kva=1e3)
    pp.to_excel(net_in, filename)
    net_out = pp.from_excel(filename)
    assert_net_equal(net_in, net_out)
    assert net_out.user_pf_options == net_in.user_pf_options


def test_json(net_in, tempdir):
    filename = os.path.join(tempdir, "testfile.json")
    try:
        net_geo = copy.deepcopy(net_in)
        # make GeodataFrame
        from shapely.geometry import Point, LineString
        from fiona.crs import from_epsg
        import geopandas as gpd

        for tab in ('bus_geodata', 'line_geodata'):
            net_geo[tab]['geometry'] = None
            net_geo[tab] = gpd.GeoDataFrame(net_geo[tab], geometry='geometry', crs=from_epsg(4326))

        for idx, row in net_geo.bus_geodata.iterrows():
            net_geo.bus_geodata.at[idx, 'geometry'] = Point(row.x, row.y)

        for idx, row in net_geo.line_geodata.iterrows():
            net_geo.line_geodata.at[idx, 'geometry'] = LineString(row.coords)

        pp.to_json(net_geo, filename)
        net_out = pp.from_json(filename)
        assert_net_equal(net_geo, net_out)
    except (NameError, ImportError):
        pass

    # check if restore_all_dtypes works properly:
    net_in.line['test'] = 123
    # this will not work:
    # net_in.res_line['test'] = 123
    pp.to_json(net_in, filename)
    net_out = pp.from_json(filename)
    assert_net_equal(net_in, net_out)


def test_type_casting_json(net_in, tempdir):
    filename = os.path.join(tempdir, "testfile.json")
    net_in.sn_kva = 1000
    pp.to_json(net_in, filename)
    net = pp.from_json(filename)
    assert_net_equal(net_in, net)


def test_sqlite(net_in, tempdir):
    filename = os.path.join(tempdir, "testfile.db")
    pp.to_sqlite(net_in, filename)
    net_out = pp.from_sqlite(filename)
    assert_net_equal(net_in, net_out)


def test_convert_format():  # TODO what is this thing testing ?
    folder = os.path.abspath(os.path.dirname(pp.__file__))
    net = pp.from_pickle(os.path.join(folder, "test", "api", "old_net.p"))
    pp.runpp(net)
    assert net.converged


def test_restore_all_dtypes():
    net = create_test_network()
    pp.runpp(net)
    net['res_test'] = pd.DataFrame(columns=['test'], data=[1, 2, 3])
    net['test'] = pd.DataFrame(columns=['test'], data=[1, 2, 3])
    net.line['test'] = 123
    net.res_line['test'] = 123
    net.bus['test'] = 123
    net.res_bus['test'] = 123
    net.res_load['test'] = 123
    dtdf = collect_all_dtypes_df(net)
    restore_all_dtypes(net, dtdf)


def test_to_json_dtypes(tempdir):
    filename = os.path.join(tempdir, "testfile.json")
    net = create_test_network()
    pp.runpp(net)
    net['res_test'] = pd.DataFrame(columns=['test'], data=[1, 2, 3])
    net['test'] = pd.DataFrame(columns=['test'], data=[1, 2, 3])
    net.line['test'] = 123
    net.res_line['test'] = 123
    net.bus['test'] = 123
    net.res_bus['test'] = 123
    net.res_load['test'] = 123
    pp.to_json(net, filename)
    net1 = pp.from_json(filename)
    assert_net_equal(net, net1)


if __name__ == "__main__":
    pytest.main(["test_file_io.py", "-x"])
