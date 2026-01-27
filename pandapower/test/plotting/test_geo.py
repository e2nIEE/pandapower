# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import math
import os

import geojson
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal

from pandapower.networks.mv_oberrhein import mv_oberrhein
from pandapower.plotting.geo import _node_geometries_from_geodata, _branch_geometries_from_geodata, \
    _transform_node_geometry_to_geodata, _transform_branch_geometry_to_coords, _convert_xy_epsg, \
    convert_geodata_to_geojson, dump_to_geojson, convert_crs
from pandapower.test.helper_functions import create_test_network


def _bus_geojson_to_geodata_(_net):
    _net["bus_geodata"] = pd.DataFrame(
        _net.bus.geo.dropna().apply(geojson.loads).apply(geojson.utils.coords).apply(next).to_list(),
        index=_net.bus.geo.dropna().index,
        columns=["x", "y"]
    )
    _net["bus_geodata"]["coords"] = math.nan
    _net.bus_geodata.coords = _net.bus_geodata.coords.astype("object")
    _net.bus.drop("geo", axis=1, inplace=True)


def _line_geojson_to_geodata_(_net):
    _net["line_geodata"] = _net.line.geo.dropna().apply(geojson.loads).apply(geojson.utils.coords).apply(
        list).to_frame().rename(columns={"geo": "coords"})
    _net.line.drop("geo", axis=1, inplace=True)


@pytest.fixture(name='net', params=(create_test_network(), mv_oberrhein()))
def test_network(request):
    """
    Fixture which yields different networks for testing.
    It should yield the test network, a network with geodata and a network with graph layout coordinates.
    """
    yield copy.deepcopy(request.param)


@pytest.fixture
def get_network_and_result(net, request):
    """
    Fixture which yields the network, and its expected result based on running test.
    """
    test_file_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(test_file_path, "test_geo", request.keywords.node.originalname, f"{net.name}.pkl")
    df = pd.read_pickle(full_path)
    return net, df


@pytest.mark.xfail
def test__node_geometries_from_geodata(get_network_and_result):
    pytest.importorskip("geopandas")

    _net, expected = get_network_and_result
    _bus_geojson_to_geodata_(_net)

    result = _node_geometries_from_geodata(_net.bus_geodata)
    # is mostly the same as assert_geodataframe_equal with check_less_precise=True, but the tolerance in the function
    # can't be adapted
    assert result.shape == expected.shape
    assert isinstance(result, type(expected))
    assert result.geom_equals_exact(expected.geometry, tolerance=1 * 10 ** (-6)).all()
    assert not (result.geometry.is_empty & expected.geometry.is_empty).all()
    assert not (result.geometry.isna() & expected.geometry.isna()).all()
    left2 = result.select_dtypes(exclude="geometry")
    right2 = expected.select_dtypes(exclude="geometry")
    assert_index_equal(result.columns, expected.columns, exact="equiv", obj="GeoDataFrame.columns")
    assert_frame_equal(left2, right2, check_dtype=True, check_index_type="equiv", check_column_type="equiv",
                       obj="GeoDataFrame")


@pytest.mark.xfail
def test__branch_geometries_from_geodata(get_network_and_result):
    pytest.importorskip("geopandas")

    _net, expected = get_network_and_result

    _line_geojson_to_geodata_(_net)

    result = _branch_geometries_from_geodata(_net.line_geodata)
    # is mostly the same as assert_geodataframe_equal with check_less_precise=True, but the tolerance in the function
    # can't be adapted
    assert result.shape == expected.shape
    assert isinstance(result, type(expected))
    assert (result.geom_equals_exact(expected.geometry, tolerance=1 * 10 ** (-6)) |
            (result.geometry.is_empty & expected.geometry.is_empty) |
            (result.geometry.isna() & expected.geometry.isna())).all()
    left2 = result.select_dtypes(exclude="geometry")
    right2 = expected.select_dtypes(exclude="geometry")
    assert_index_equal(result.columns, expected.columns, exact="equiv", obj="GeoDataFrame.columns")
    assert_frame_equal(left2, right2, check_dtype=True, check_index_type="equiv", check_column_type="equiv",
                       obj="GeoDataFrame")


@pytest.mark.xfail
def test__transform_node_geometry_to_geodata(get_network_and_result):
    pytest.importorskip("geopandas")

    _net, expected = get_network_and_result
    _bus_geojson_to_geodata_(_net)

    # Transforming to geodata to test the inverse...
    _net.bus_geodata = _node_geometries_from_geodata(_net.bus_geodata)
    result = _transform_node_geometry_to_geodata(_net.bus_geodata)
    # is mostly the same as assert_geodataframe_equal with check_less_precise=True, but the tolerance in the function
    # can't be adapted
    assert result.shape == expected.shape
    assert isinstance(result, type(expected))
    assert (result.geom_equals_exact(expected.geometry, tolerance=1 * 10 ** (-6)) |
            (result.geometry.is_empty & expected.geometry.is_empty) |
            (result.geometry.isna() & expected.geometry.isna())).all()
    left2 = result.select_dtypes(exclude="geometry")
    right2 = expected.select_dtypes(exclude="geometry")
    assert_index_equal(result.columns, expected.columns, exact="equiv", obj="GeoDataFrame.columns")
    assert_frame_equal(left2, right2, check_dtype=True, check_index_type="equiv", check_column_type="equiv",
                       obj="GeoDataFrame")

@pytest.mark.xfail
def test__transform_branch_geometry_to_coords(get_network_and_result):
    pytest.importorskip("geopandas")

    _net, expected = get_network_and_result
    _line_geojson_to_geodata_(_net)

    _net.line_geodata = _branch_geometries_from_geodata(_net.line_geodata)
    result = _transform_branch_geometry_to_coords(_net.line_geodata)
    # is mostly the same as assert_geodataframe_equal with check_less_precise=True, but the tolerance in the function
    # can't be adapted
    assert result.shape == expected.shape
    assert isinstance(result, type(expected))
    assert (result.geom_equals_exact(expected.geometry, tolerance=1 * 10 ** (-6)) |
            (result.geometry.is_empty & expected.geometry.is_empty) |
            (result.geometry.isna() & expected.geometry.isna())).all()
    left2 = result.select_dtypes(exclude="geometry")
    right2 = expected.select_dtypes(exclude="geometry")
    assert_index_equal(result.columns, expected.columns, exact="equiv", obj="GeoDataFrame.columns")
    assert_frame_equal(left2, right2, check_dtype=True, check_index_type="equiv", check_column_type="equiv",
                       obj="GeoDataFrame")


def test__convert_xy_epsg():
    x = 9.487
    y = 51.320
    result = _convert_xy_epsg(x, y, 4326, 31467)
    expected = (3534023, 5687359)
    assert result == pytest.approx(expected)
    result = _convert_xy_epsg(x, y, 4326, 3857)
    expected = (1056088, 6678094)
    assert result == pytest.approx(expected)
    x = 3534023
    y = 5687359
    result = _convert_xy_epsg(x, y, 31467, 4326)
    expected = (9.487, 51.320)
    assert result == pytest.approx(expected, abs=1e-3)
    x = [9.487, 9]
    y = [51.320, 51]
    result_x, result_y = _convert_xy_epsg(x, y, 4326, 31467)
    expected_x, expected_y = ([3534023, 3500073], [5687359, 5651645])
    assert result_x == pytest.approx(expected_x)
    assert result_y == pytest.approx(expected_y)

def test_convert_epsg_bus_geodata():
    pytest.skip("Not implemented")


def test_convert_crs():
    pytest.skip("Not implemented")

def test_convert_crs__bus_only():
    def get_net_with_bus_geo():
        net = mv_oberrhein()
        net.line.geo = None
        return net
    # Test with geojson
    net = get_net_with_bus_geo()
    convert_crs(net)
    # Test with geodata
    net = get_net_with_bus_geo()
    _bus_geojson_to_geodata_(net)
    convert_crs(net)
    # Test with geodata without coord column
    net = get_net_with_bus_geo()
    _bus_geojson_to_geodata_(net)
    net.bus_geodata = net.bus_geodata[["x", "y"]]
    convert_crs(net)
    assert "coords" not in net.bus_geodata.columns

    # Test with geodata, populate geodata coord
    net = get_net_with_bus_geo()
    _bus_geojson_to_geodata_(net)
    for index, row in net.bus_geodata.iterrows():
        net.bus_geodata.at[index, 'coords'] = [(row['x'], row['y']), (row['x'], row['y'])]
    convert_crs(net)
    assert "coords" in net.bus_geodata.columns
    assert not (pd.isna(net.bus_geodata.coords)).any()

def test_convert_crs__branch_only():
    net = mv_oberrhein()
    net.bus.geo = None
    convert_crs(net)
    _line_geojson_to_geodata_(net)
    convert_crs(net)

def test_dump_to_geojson():
    pytest.importorskip("geojson")
    from geojson import FeatureCollection, dumps

    # test with no parameters
    _net = create_test_network()
    _bus_geojson_to_geodata_(_net)
    _line_geojson_to_geodata_(_net)
    convert_geodata_to_geojson(_net)

    result = dump_to_geojson(_net)
    assert isinstance(result, FeatureCollection)
    assert dumps(result, sort_keys=True) == '{"features": [], "type": "FeatureCollection"}'

    # test exporting buses
    result = dump_to_geojson(_net, buses=True)
    assert isinstance(result, FeatureCollection)
    assert dumps(result, sort_keys=True) == ('{"features": [{"geometry": {"coordinates": [1.0, 2.0], "type": "Point"}, '
                                             '"id": "bus-1", "properties": {"in_service": true, "name": "bus2", '
                                             '"pp_index": 1, "pp_type": "bus", "type": "b", "vn_kv": 0.4, '
                                             '"zone": null}, "type": "Feature"}, {"geometry": {"coordinates": [1.0, '
                                             '3.0], "type": "Point"}, "id": "bus-7", "properties": {"in_service": '
                                             'true, "name": "bus3", "pp_index": 7, "pp_type": "bus", "type": "b", '
                                             '"vn_kv": 0.4, "zone": null}, "type": "Feature"}], '
                                             '"type": "FeatureCollection"}')

    # test exporting lines
    result = dump_to_geojson(_net, lines=True)
    assert isinstance(result, FeatureCollection)
    assert dumps(result, sort_keys=True) == ('{"features": [{"geometry": {"coordinates": [[1.0, 2.0], [3.0, 4.0]], '
                                             '"type": "LineString"}, "id": "line-0", "properties": {"c_nf_per_km": '
                                             '720.0, "df": 1.0, "from_bus": 1, "g_us_per_km": 0.0, "ices": 0.389985, '
                                             '"in_service": true, "length_km": 1.0, "max_i_ka": 0.328, '
                                             '"name": "line1", "parallel": 1, "pp_index": 0, "pp_type": "line", '
                                             '"r_ohm_per_km": 0.2067, "std_type": null, "to_bus": 7, "type": null, '
                                             '"x_ohm_per_km": 0.1897522}, "type": "Feature"}], '
                                             '"type": "FeatureCollection"}')

    # test exporting both
    result = dump_to_geojson(_net, buses=True, lines=True)
    assert isinstance(result, FeatureCollection)
    assert dumps(result, sort_keys=True) == ('{"features": [{"geometry": {"coordinates": [1.0, 2.0], "type": "Point"}, '
                                             '"id": "bus-1", "properties": {"in_service": true, "name": "bus2", '
                                             '"pp_index": 1, "pp_type": "bus", "type": "b", "vn_kv": 0.4, '
                                             '"zone": null}, "type": "Feature"}, {"geometry": {"coordinates": [1.0, '
                                             '3.0], "type": "Point"}, "id": "bus-7", "properties": {"in_service": '
                                             'true, "name": "bus3", "pp_index": 7, "pp_type": "bus", "type": "b", '
                                             '"vn_kv": 0.4, "zone": null}, "type": "Feature"}, {"geometry": {'
                                             '"coordinates": [[1.0, 2.0], [3.0, 4.0]], "type": "LineString"}, '
                                             '"id": "line-0", "properties": {"c_nf_per_km": 720.0, "df": 1.0, '
                                             '"from_bus": 1, "g_us_per_km": 0.0, "ices": 0.389985, "in_service": '
                                             'true, "length_km": 1.0, "max_i_ka": 0.328, "name": "line1", "parallel": '
                                             '1, "pp_index": 0, "pp_type": "line", "r_ohm_per_km": 0.2067, '
                                             '"std_type": null, "to_bus": 7, "type": null, "x_ohm_per_km": '
                                             '0.1897522}, "type": "Feature"}], "type": "FeatureCollection"}')

    # test exporting specific buses
    result = dump_to_geojson(_net, buses=[1])
    assert isinstance(result, FeatureCollection)
    assert dumps(result, sort_keys=True) == ('{"features": [{"geometry": {"coordinates": [1.0, 2.0], "type": "Point"}, '
                                             '"id": "bus-1", "properties": {"in_service": true, "name": "bus2", '
                                             '"pp_index": 1, "pp_type": "bus", "type": "b", "vn_kv": 0.4, '
                                             '"zone": null}, "type": "Feature"}], "type": "FeatureCollection"}')

    # test exporting specific lines
    result = dump_to_geojson(_net, lines=[0])
    assert isinstance(result, FeatureCollection)
    assert dumps(result, sort_keys=True) == ('{"features": [{"geometry": {"coordinates": [[1.0, 2.0], [3.0, 4.0]], '
                                             '"type": "LineString"}, "id": "line-0", "properties": {"c_nf_per_km": '
                                             '720.0, "df": 1.0, "from_bus": 1, "g_us_per_km": 0.0, "ices": 0.389985, '
                                             '"in_service": true, "length_km": 1.0, "max_i_ka": 0.328, '
                                             '"name": "line1", "parallel": 1, "pp_index": 0, "pp_type": "line", '
                                             '"r_ohm_per_km": 0.2067, "std_type": null, "to_bus": 7, "type": null, '
                                             '"x_ohm_per_km": 0.1897522}, "type": "Feature"}], '
                                             '"type": "FeatureCollection"}')

    # test exporting props from bus and res_bus
    _net.res_bus.loc[1, ["vm_pu", "va_degree", "p_mw", "q_mvar"]] = [1.0, 1.0, 1.0, 1.0]
    result = dump_to_geojson(_net, buses=[1])
    assert isinstance(result, FeatureCollection)
    assert dumps(result, sort_keys=True) == ('{"features": [{"geometry": {"coordinates": [1.0, 2.0], "type": "Point"}, '
                                             '"id": "bus-1", "properties": {"in_service": true, "name": "bus2", '
                                             '"p_mw": 1.0, "pp_index": 1, "pp_type": "bus", "q_mvar": 1.0, '
                                             '"type": "b", "va_degree": 1.0, "vm_pu": 1.0, "vn_kv": 0.4, '
                                             '"zone": null}, "type": "Feature"}], "type": "FeatureCollection"}')

    # test exporting props from line and res_line
    _net.res_line.loc[0, _net.res_line.columns] = [7.0] * len(_net.res_line.columns)
    result = dump_to_geojson(_net, lines=[0])
    assert isinstance(result, FeatureCollection)
    assert dumps(result, sort_keys=True) == ('{"features": [{"geometry": {"coordinates": [[1.0, 2.0], [3.0, 4.0]], '
                                             '"type": "LineString"}, "id": "line-0", "properties": {"c_nf_per_km": '
                                             '720.0, "df": 1.0, "from_bus": 1, "g_us_per_km": 0.0, "i_from_ka": 7.0, '
                                             '"i_ka": 7.0, "i_to_ka": 7.0, "ices": 0.389985, "in_service": true, '
                                             '"length_km": 1.0, "loading_percent": 7.0, "max_i_ka": 0.328, '
                                             '"name": "line1", "p_from_mw": 7.0, "p_to_mw": 7.0, "parallel": 1, '
                                             '"pl_mw": 7.0, "pp_index": 0, "pp_type": "line", "q_from_mvar": 7.0, '
                                             '"q_to_mvar": 7.0, "ql_mvar": 7.0, "r_ohm_per_km": 0.2067, "std_type": '
                                             'null, "to_bus": 7, "type": null, "va_from_degree": 7.0, "va_to_degree": '
                                             '7.0, "vm_from_pu": 7.0, "vm_to_pu": 7.0, "x_ohm_per_km": 0.1897522}, '
                                             '"type": "Feature"}], "type": "FeatureCollection"}')


def test_convert_geodata_to_geojson():
    pytest.importorskip("geojson")
    pytest.importorskip("pandapower")
    import geojson
    from pandapower.create import create_empty_network, create_bus, create_line
    # Erstelle ein Beispielnetzwerk
    _net = create_empty_network()

    # Füge Busse hinzu
    create_bus(_net, 0, geodata=(10, 20))
    create_bus(_net, 1, geodata=(30, 40))

    # Füge Leitungen hinzu
    create_line(_net, 0, 1, 1, std_type="NAYY 4x50 SE", geodata=[[10, 20], [30, 40]])

    _bus_geojson_to_geodata_(_net)
    _line_geojson_to_geodata_(_net)

    # Rufe die Funktion zum Konvertieren auf
    convert_geodata_to_geojson(_net)

    # Überprüfe die Ergebnisse
    assert _net.bus.at[0, "geo"] == geojson.dumps(geojson.Point((10., 20.)), sort_keys=True)
    assert _net.bus.at[1, "geo"] == geojson.dumps(geojson.Point((30., 40.)), sort_keys=True)
    assert _net.line.at[0, "geo"] == geojson.dumps(geojson.LineString([(10., 20.), (30., 40.)]), sort_keys=True)
    # TODO: Test could be more exhaustive (e.g. test delete=False, lonlat=True, geo_str=False)


def test_convert_gis_to_geojson():
    # TODO: implement
    pytest.skip("Not implemented")


if __name__ == "__main__":
    pytest.main(["test_geo.py"])
