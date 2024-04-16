# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import json
import os
import tempfile

import pandas as pd
import pandas.testing as pdt
import numpy as np
import pytest
import time

import pandapower as pp
import pandapower.networks
import pandapower.control
from pandapower import pp_dir
from pandapower.auxiliary import _preserve_dtypes
from pandapower.sql_io import download_sql_table
from pandapower.test import assert_res_equal

try:
    import psycopg2
    import psycopg2.errors

    PSYCOPG2_INSTALLED = True
except ImportError:
    psycopg2 = None
    PSYCOPG2_INSTALLED = False

try:
    import sqlite3

    SQLITE_INSTALLED = True
except ImportError:
    sqlite3 = None
    SQLITE_INSTALLED = False


@pytest.fixture(params=["case9", "case14", "case39", "simple_mv_open_ring_net",
                        "create_cigre_network_hv", "mv_oberrhein"])
def net_in(request):
    method = pp.networks.__dict__[request.param]
    net = method()
    net.line_geodata.loc[0, "coords"] = [(1.1, 2.2), (3.3, 4.4)]
    net.line_geodata.loc[11, "coords"] = [(5.5, 5.5), (6.6, 6.6), (7.7, 7.7)]
    if len(net.trafo) > 0:
        net.trafo.tap_side = "lv"
        pp.control.DiscreteTapControl(net, net.trafo.index.values[0], 0.98, 1.02)
    return net


def get_postgresql_connection_data():
    filename = os.path.join(pp_dir, "test", "test_files", "postgresql_connect_data.json")
    if not os.path.isfile(filename):
        return {}, None
    with open(filename) as fp:
        connect_data = json.load(fp)
        schema = connect_data.pop("schema")

    return connect_data, schema


def postgresql_listening(**connect_data):
    if len(connect_data) == 0:
        return False
    try:
        conn = psycopg2.connect(**connect_data)
        conn.close()
        return True
    except psycopg2.OperationalError as ex:
        return False


def assert_postgresql_roundtrip(net_in, **kwargs):
    net = net_in.deepcopy()
    include_results = kwargs.pop("include_results", False)
    if not include_results:
        pp.reset_results(net)
    else:
        pp.runpp(net)
    connection_data, schema = get_postgresql_connection_data()
    grid_id = pp.to_postgresql(net, schema=schema, include_results=include_results, **connection_data, **kwargs)

    net_out = pp.from_postgresql(grid_id=grid_id, schema=schema, **connection_data, **kwargs)

    if not include_results:
        pp.runpp(net)
        pp.runpp(net_out)

    assert_res_equal(net, net_out)

    for element, table in net.items():
        # dictionaries (e.g. std_type) not included
        # json serialization/deserialization of objects not implemented
        if not isinstance(table, pd.DataFrame) or table.empty or "geodata" in element:
            continue
        # code below: very difficult to compare columns with NaN values due to None vs np.nan and dtypes,
        # "1" vs 1 and dtype object
        # also sometimes order of rows is not same
        columns = table.columns
        table_in = table.fillna(np.nan)
        table_out = net_out[element][columns].loc[table_in.index].fillna(np.nan)
        _preserve_dtypes(table_out, table_in.dtypes)
        pdt.assert_frame_equal(table_in, table_out, check_dtype=False)

    # clean-up
    pp.delete_postgresql_net(grid_id=grid_id, schema=schema, **connection_data)


POSTGRESQL_AVAILABLE = PSYCOPG2_INSTALLED and postgresql_listening(**get_postgresql_connection_data()[0])


@pytest.mark.skipif(not POSTGRESQL_AVAILABLE,
                    reason="testing happens on GitHub Actions where we create a temporary instance of PostgreSQL")
def test_postgresql(net_in):
    assert_postgresql_roundtrip(net_in, include_results=False)
    assert_postgresql_roundtrip(net_in, include_results=True)


@pytest.mark.skipif(not POSTGRESQL_AVAILABLE,
                    reason="testing happens on GitHub Actions where we create a temporary instance of PostgreSQL")
def test_unique():
    net = pp.networks.case9()
    connection_data, schema = get_postgresql_connection_data()
    grid_id = pp.to_postgresql(net, **connection_data, schema=schema)
    with pytest.raises(UserWarning):
        pp.to_postgresql(net, **connection_data, schema=schema, grid_id=grid_id)
    # clean-up:
    pp.delete_postgresql_net(grid_id=grid_id, schema=schema, **connection_data)


@pytest.mark.skipif(not POSTGRESQL_AVAILABLE,
                    reason="testing happens on GitHub Actions where we create a temporary instance of PostgreSQL")
def test_delete():
    connection_data, schema = get_postgresql_connection_data()
    # cannot delete if the net does not exist
    with pytest.raises(UserWarning):
        pp.delete_postgresql_net(grid_id=int(time.time()), schema=schema, **connection_data)

    # check that net is deleted
    net = pp.networks.case9()
    grid_id = pp.to_postgresql(net, **connection_data, schema=schema)
    pp.delete_postgresql_net(grid_id=grid_id, schema=schema, **connection_data)
    with pytest.raises(UserWarning):
        _ = pp.from_postgresql(grid_id=grid_id, schema=schema, **connection_data)

    # check that it is not only deleted from the grid catalogue
    conn = psycopg2.connect(**connection_data)
    cursor = conn.cursor()
    for element in ("bus", "line", "load", "ext_grid", "gen"):
        tab = download_sql_table(cursor, f"{schema}.{element}", grid_id=grid_id)
        assert tab.empty


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
