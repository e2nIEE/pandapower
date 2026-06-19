# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import os
import copy

import pandas as pd
import pandas.testing as pdt
import numpy as np
import pytest
import time

from pandapower import reset_results, runpp
from pandapower.networks import case9, case14, case39, simple_mv_open_ring_net, create_cigre_network_hv, mv_oberrhein
from pandapower.plotting.geo import convert_geodata_to_geojson
from pandapower.auxiliary import _preserve_dtypes
from pandapower.sql_io import (
    download_sql_table, to_postgresql, from_postgresql, delete_postgresql_net, PSYCOPG_INSTALLED
)
from pandapower.test import assert_res_equal

if PSYCOPG_INSTALLED:
    import psycopg
    import psycopg.errors

@pytest.fixture(params=[case9, case14, case39, simple_mv_open_ring_net,
                        create_cigre_network_hv, mv_oberrhein])
def net_in(request):
    net = request.param()
    # net.line.loc[0, "geo"] = '{"coordinates": [[1.1, 2.2], [3.3, 4.4]], "type": "LineString"}'
    # net.line.loc[11, "geo"] = '{"coordinates": [[5.5, 5.5], [6.6, 6.6], [7.7, 7.7]], "type": "LineString"}'
    # if len(net.trafo) > 0:
    #     net.trafo.tap_side = "lv"
    #     pp.control.DiscreteTapControl(net, net.trafo.index.values[0], 0.98, 1.02)
    return net


def get_postgresql_connection_data() -> tuple[str | None, str | None]:
    dsn = os.getenv('DSN', None)
    schema = os.getenv('SCHEMA', None)
    return dsn, schema


def postgresql_listening(dsn: str | None) -> bool:
    if dsn is None:
        return False
    try:
        conn = psycopg.connect(dsn)
        conn.close()
        return True
    except psycopg.OperationalError:
        return False


def assert_postgresql_roundtrip(net_in, **kwargs):
    net = copy.deepcopy(net_in)
    if hasattr(net, "bus_geodata") or hasattr(net, "line_geodata"):
        convert_geodata_to_geojson(net)
    include_results = kwargs.pop("include_results", False)
    if not include_results:
        reset_results(net)
    else:
        runpp(net)
    dsn, schema = get_postgresql_connection_data()
    grid_id = to_postgresql(net, dsn=dsn, schema=schema, include_results=include_results, **kwargs)

    net_out = from_postgresql(dsn=dsn, grid_id=grid_id, schema=schema, **kwargs)

    if not include_results:
        runpp(net)
        runpp(net_out)

    assert_res_equal(net, net_out)

    for element, table in net.items():
        # dictionaries (e.g. std_type) not included
        # json serialization/deserialization of objects not implemented
        if not isinstance(table, pd.DataFrame) or table.empty:
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
    delete_postgresql_net(dsn=dsn, schema=schema, grid_id=grid_id)


POSTGRESQL_AVAILABLE = PSYCOPG_INSTALLED and postgresql_listening(get_postgresql_connection_data()[0])


@pytest.mark.skipif(not POSTGRESQL_AVAILABLE,
                    reason="testing happens on GitHub Actions where we create a temporary instance of PostgreSQL")
def test_postgresql(net_in):
    assert_postgresql_roundtrip(net_in, include_results=False)
    assert_postgresql_roundtrip(net_in, include_results=True)


@pytest.mark.skipif(not POSTGRESQL_AVAILABLE,
                    reason="testing happens on GitHub Actions where we create a temporary instance of PostgreSQL")
def test_unique():
    net = case9()
    dsn, schema = get_postgresql_connection_data()
    grid_id = to_postgresql(net, dsn=dsn, schema=schema)
    with pytest.raises(UserWarning):
        to_postgresql(net, dsn=dsn, schema=schema, grid_id=grid_id)
    # clean-up:
    delete_postgresql_net(dsn=dsn, schema=schema, grid_id=grid_id)


@pytest.mark.skipif(not POSTGRESQL_AVAILABLE,
                    reason="testing happens on GitHub Actions where we create a temporary instance of PostgreSQL")
def test_delete():
    dsn, schema = get_postgresql_connection_data()
    # cannot delete if the net does not exist
    with pytest.raises(UserWarning):
        delete_postgresql_net(dsn=dsn, schema=schema, grid_id=int(time.time()))

    # check that net is deleted
    net = case9()
    grid_id = to_postgresql(net, dsn=dsn, schema=schema)
    delete_postgresql_net(dsn=dsn, schema=schema, grid_id=grid_id)
    with pytest.raises(UserWarning):
        _ = from_postgresql(dsn=dsn, schema=schema, grid_id=grid_id)

    # check that it is not only deleted from the grid catalogue
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cursor:
            for element in ("bus", "line", "load", "ext_grid", "gen"):
                tab = download_sql_table(cursor, f"{schema}.{element}", grid_id=grid_id)
                assert tab.empty

if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
