# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import pandas

import pandapower.plotting.geo as geo
from pandapower.test.helper_functions import create_test_network


def test__node_geometries_from_geodata():
    pytest.importorskip("geopandas")
    from geopandas import GeoDataFrame, points_from_xy, testing

    net = create_test_network()
    result = geo._node_geometries_from_geodata(net.bus_geodata)
    pdf = pandas.DataFrame({'x': [1., 1.], 'y': [2., 3.], 'coords': [float('nan'), float('nan')]})
    pdf = pdf.astype({'coords': 'object'})
    pdf.set_index(pandas.Index([1, 7]), inplace=True)
    expected = GeoDataFrame(crs="epsg:31467", geometry=points_from_xy(pdf.x, pdf.y), data=pdf)
    testing.assert_geodataframe_equal(result, expected)


def test__branch_geometries_from_geodata():
    pytest.importorskip("geopandas")
    pytest.importorskip("shapely")
    from geopandas import GeoDataFrame, testing
    from shapely.geometry import LineString

    net = create_test_network()
    result = geo._branch_geometries_from_geodata(net.line_geodata)
    pdf = pandas.DataFrame({'coords': [[[1, 2], [3, 4]]], 'geometry': LineString([[1, 2], [3, 4]])})
    expected = GeoDataFrame(crs="epsg:31467", geometry=pdf.geometry, data=pdf)
    testing.assert_geodataframe_equal(result, expected)


def test__transform_node_geometry_to_geodata():
    pytest.importorskip("geopandas")
    pytest.importorskip("shapely")
    from shapely.geometry import Point

    net = create_test_network()
    # Transforming to geodata to test the inverse...
    net.bus_geodata = geo._node_geometries_from_geodata(net.bus_geodata)
    result = geo._transform_node_geometry_to_geodata(net.bus_geodata)
    expected = pandas.DataFrame({'x': [1., 1.], 'y': [2., 3.], 'coords': [float('nan'), float('nan')], 'geometry': [Point(1., 2.), Point(1., 3.)]})
    expected.set_index(pandas.Index([1, 7]), inplace=True)
    result.equals(expected)


def test__transform_branch_geometry_to_coords():
    pytest.importorskip("geopandas")
    pytest.importorskip("shapely")
    from shapely.geometry import LineString

    net = create_test_network()
    net.line_geodata = geo._branch_geometries_from_geodata(net.line_geodata)
    result = geo._transform_branch_geometry_to_coords(net.line_geodata)
    expected = pandas.DataFrame({'coords': [[(1., 2.), (3., 4.)]], 'geometry': LineString([[1, 2], [3, 4]])})
    result.equals(expected)


def test__convert_xy_epsg():
    x = 9.487
    y = 51.320
    result = geo._convert_xy_epsg(x, y, 4326, 31467)
    expected = (3534023, 5687359)
    assert result == pytest.approx(expected)
    result = geo._convert_xy_epsg(x, y, 4326, 3857)
    expected = (1056088, 6678094)
    assert result == pytest.approx(expected)
    x = 3534023
    y = 5687359
    result = geo._convert_xy_epsg(x, y, 31467, 4326)
    expected = (9.487, 51.320)
    assert result == pytest.approx(expected, abs=1e-3)
    x = [9.487, 9]
    y = [51.320, 51]
    result_x, result_y = geo._convert_xy_epsg(x, y, 4326, 31467)
    expected_x, expected_y = ([3534023, 3500073], [5687359, 5651645])
    assert result_x == pytest.approx(expected_x)
    assert result_y == pytest.approx(expected_y)


def test_convert_gis_to_geodata():
    pytest.importorskip("geopandas")
    pytest.importorskip("shapely")
    from shapely.geometry import Point, LineString
    from geopandas import testing

    converted_node = pandas.DataFrame({'x': [1., 1.], 'y': [2., 3.], 'coords': [float('nan'), float('nan')], 'geometry': [Point(1., 2.), Point(1., 3.)]})
    converted_node.set_index(pandas.Index([1, 7]), inplace=True)
    converted_branch = pandas.DataFrame({'coords': [[(1., 2.), (3., 4.)]], 'geometry': LineString([[1, 2], [3, 4]])})

    net = create_test_network()
    geo.convert_geodata_to_gis(net)
    node_geodata = net.bus_geodata
    branch_geodata = net.line_geodata

    geo.convert_gis_to_geodata(net)
    net.bus_geodata.equals(converted_node)
    net.line_geodata.equals(converted_branch)

    net = create_test_network()
    geo.convert_geodata_to_gis(net)
    geo.convert_gis_to_geodata(net, node_geodata=False)
    testing.assert_geodataframe_equal(net.bus_geodata, node_geodata)
    net.line_geodata.equals(converted_branch)

    net = create_test_network()
    geo.convert_geodata_to_gis(net)
    geo.convert_gis_to_geodata(net, branch_geodata=False)
    net.bus_geodata.equals(converted_node)
    testing.assert_geodataframe_equal(net.line_geodata, branch_geodata)


def test_convert_geodata_to_gis():
    pytest.importorskip("geopandas")
    pytest.importorskip("shapely")
    from geopandas import GeoDataFrame, testing, points_from_xy
    from shapely.geometry import LineString

    pdf = pandas.DataFrame({'x': [1., 1.], 'y': [2., 3.], 'coords': [float('nan'), float('nan')]})
    pdf = pdf.astype({'coords': 'object'})
    pdf.set_index(pandas.Index([1, 7]), inplace=True)
    converted_node = GeoDataFrame(crs="epsg:31467", geometry=points_from_xy(pdf.x, pdf.y), data=pdf)

    pdf = pandas.DataFrame({'coords': [[[1, 2], [3, 4]]], 'geometry': LineString([[1, 2], [3, 4]])})
    converted_branch = GeoDataFrame(crs="epsg:31467", geometry=pdf.geometry, data=pdf)

    net = create_test_network()
    node_geodata = net.bus_geodata
    branch_geodata = net.line_geodata

    geo.convert_geodata_to_gis(net)
    testing.assert_geodataframe_equal(net.bus_geodata, converted_node)
    testing.assert_geodataframe_equal(net.line_geodata, converted_branch)

    net = create_test_network()
    geo.convert_geodata_to_gis(net, node_geodata=False)
    net.bus_geodata.equals(node_geodata)
    testing.assert_geodataframe_equal(net.line_geodata, converted_branch)

    net = create_test_network()
    geo.convert_geodata_to_gis(net, branch_geodata=False)
    testing.assert_geodataframe_equal(net.bus_geodata, converted_node)
    net.line_geodata.equals(branch_geodata)


def test_convert_epsg_bus_geodata():
    pytest.skip("Not implemented")


def test_convert_crs():
    pytest.skip("Not implemented")


def test_dump_to_geojson():
    pytest.importorskip("geojson")
    from geojson import FeatureCollection, dumps

    # test with no parameters
    net = create_test_network()
    result = geo.dump_to_geojson(net)
    assert isinstance(result, FeatureCollection)
    assert dumps(result) == '{"type": "FeatureCollection", "features": []}'

    # test exporting nodes
    result = geo.dump_to_geojson(net, nodes=True)
    assert isinstance(result, FeatureCollection)
    assert dumps(result) == '{"type": "FeatureCollection", "features": [{"type": "Feature", "id": 1, "geometry": {"type": "Point", "coordinates": [1.0, 2.0]}, "properties": {"pp_type": "bus", "pp_index": 1, "name": "bus2", "vn_kv": 0.4, "type": "b", "zone": "None", "in_service": 1.0}}, {"type": "Feature", "id": 7, "geometry": {"type": "Point", "coordinates": [1.0, 3.0]}, "properties": {"pp_type": "bus", "pp_index": 7, "name": "bus3", "vn_kv": 0.4, "type": "b", "zone": "None", "in_service": 1.0}}]}'

    # test exporting branches
    result = geo.dump_to_geojson(net, branches=True)
    assert isinstance(result, FeatureCollection)
    assert dumps(result) == '{"type": "FeatureCollection", "features": [{"type": "Feature", "id": 0, "geometry": {"type": "LineString", "coordinates": [[1, 2], [3, 4]]}, "properties": {"pp_type": "line", "pp_index": 0, "name": "line1", "std_type": "None", "from_bus": 1.0, "to_bus": 7.0, "length_km": 1.0, "r_ohm_per_km": 0.2067, "x_ohm_per_km": 0.1897522, "c_nf_per_km": 720.0, "g_us_per_km": 0.0, "max_i_ka": 0.328, "df": 1.0, "parallel": 1.0, "type": "None", "in_service": 1.0, "ices": 0.389985}}]}'

    # test exporting both
    result = geo.dump_to_geojson(net, nodes=True, branches=True)
    assert isinstance(result, FeatureCollection)
    assert dumps(result) == '{"type": "FeatureCollection", "features": [{"type": "Feature", "id": 1, "geometry": {"type": "Point", "coordinates": [1.0, 2.0]}, "properties": {"pp_type": "bus", "pp_index": 1, "name": "bus2", "vn_kv": 0.4, "type": "b", "zone": "None", "in_service": 1.0}}, {"type": "Feature", "id": 7, "geometry": {"type": "Point", "coordinates": [1.0, 3.0]}, "properties": {"pp_type": "bus", "pp_index": 7, "name": "bus3", "vn_kv": 0.4, "type": "b", "zone": "None", "in_service": 1.0}}, {"type": "Feature", "id": 0, "geometry": {"type": "LineString", "coordinates": [[1, 2], [3, 4]]}, "properties": {"pp_type": "line", "pp_index": 0, "name": "line1", "std_type": "None", "from_bus": 1.0, "to_bus": 7.0, "length_km": 1.0, "r_ohm_per_km": 0.2067, "x_ohm_per_km": 0.1897522, "c_nf_per_km": 720.0, "g_us_per_km": 0.0, "max_i_ka": 0.328, "df": 1.0, "parallel": 1.0, "type": "None", "in_service": 1.0, "ices": 0.389985}}]}'

    # test exporting specific nodes
    result = geo.dump_to_geojson(net, nodes=[1])
    assert isinstance(result, FeatureCollection)
    assert dumps(result) == '{"type": "FeatureCollection", "features": [{"type": "Feature", "id": 1, "geometry": {"type": "Point", "coordinates": [1.0, 2.0]}, "properties": {"pp_type": "bus", "pp_index": 1, "name": "bus2", "vn_kv": 0.4, "type": "b", "zone": "None", "in_service": 1.0}}]}'

    # test exporting specific branches
    result = geo.dump_to_geojson(net, branches=[0])
    assert isinstance(result, FeatureCollection)
    assert dumps(result) == '{"type": "FeatureCollection", "features": [{"type": "Feature", "id": 0, "geometry": {"type": "LineString", "coordinates": [[1, 2], [3, 4]]}, "properties": {"pp_type": "line", "pp_index": 0, "name": "line1", "std_type": "None", "from_bus": 1.0, "to_bus": 7.0, "length_km": 1.0, "r_ohm_per_km": 0.2067, "x_ohm_per_km": 0.1897522, "c_nf_per_km": 720.0, "g_us_per_km": 0.0, "max_i_ka": 0.328, "df": 1.0, "parallel": 1.0, "type": "None", "in_service": 1.0, "ices": 0.389985}}]}'


if __name__ == "__main__":
    pytest.main(["test_geo.py"])
