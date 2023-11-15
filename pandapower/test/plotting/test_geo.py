# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import pandas
from pathlib import Path

import pandapower.plotting.geo as geo
from pandapower.test.helper_functions import create_test_network
import pandapower.networks as pn


@pytest.fixture(name='net', params=(create_test_network(), pn.mv_oberrhein()))
def test_network(request):
    """
    Fixture which yields different networks for testing.
    It should yield the test network, a network with geodata and a network with graph layout coordinates.
    """
    yield request.param


@pytest.fixture
def get_network_and_result(net, request):
    """
    Fixture which yields the network, and its expected result based on running test.
    """
    file = Path(
        request.config.invocation_dir,
        request.fspath.purebasename,
        request.keywords.node.originalname,
        net.name
    ).with_suffix('.pkl')
    df = pandas.read_pickle(file)
    return net, df


def test__node_geometries_from_geodata(get_network_and_result):
    pytest.importorskip("geopandas")
    from geopandas import testing

    net, expected = get_network_and_result

    result = geo._node_geometries_from_geodata(net.bus_geodata)
    testing.assert_geodataframe_equal(result, expected)


def test__branch_geometries_from_geodata(get_network_and_result):
    pytest.importorskip("geopandas")
    from geopandas import testing

    net, expected = get_network_and_result

    result = geo._branch_geometries_from_geodata(net.line_geodata)
    testing.assert_geodataframe_equal(result, expected)


def test__transform_node_geometry_to_geodata(get_network_and_result):
    pytest.importorskip("geopandas")
    from geopandas import testing

    net, expected = get_network_and_result


    # Transforming to geodata to test the inverse...
    net.bus_geodata = geo._node_geometries_from_geodata(net.bus_geodata)
    result = geo._transform_node_geometry_to_geodata(net.bus_geodata)
    result.to_pickle(f'test_geo/test__transform_node_geometry_to_geodata/{net.name}.pkl')
    testing.assert_geodataframe_equal(result, expected)


def test__transform_branch_geometry_to_coords(get_network_and_result):
    pytest.importorskip("geopandas")
    from geopandas import testing

    net, expected = get_network_and_result

    net.line_geodata = geo._branch_geometries_from_geodata(net.line_geodata)
    result = geo._transform_branch_geometry_to_coords(net.line_geodata)
    testing.assert_geodataframe_equal(result, expected)


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

    converted_node = pandas.DataFrame({'x': [1., 1.], 'y': [2., 3.], 'coords': [float('nan'), float('nan')],
                                       'geometry': [Point(1., 2.), Point(1., 3.)]})
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
    geo.convert_geodata_to_geojson(net)

    result = geo.dump_to_geojson(net)
    assert isinstance(result, FeatureCollection)
    assert dumps(result, sort_keys=True) == '{"features": [], "type": "FeatureCollection"}'

    # test exporting nodes
    result = geo.dump_to_geojson(net, nodes=True)
    assert isinstance(result, FeatureCollection)
    assert dumps(result, sort_keys=True) == '{"features": [{"geometry": {"coordinates": [1.0, 2.0], "type": "Point"}, "id": "bus-1", "properties": {"in_service": 1.0, "name": "bus2", "pp_index": 1, "pp_type": "bus", "type": "b", "vn_kv": 0.4, "zone": "None"}, "type": "Feature"}, {"geometry": {"coordinates": [1.0, 3.0], "type": "Point"}, "id": "bus-7", "properties": {"in_service": 1.0, "name": "bus3", "pp_index": 7, "pp_type": "bus", "type": "b", "vn_kv": 0.4, "zone": "None"}, "type": "Feature"}], "type": "FeatureCollection"}'

    # test exporting branches
    result = geo.dump_to_geojson(net, branches=True)
    assert isinstance(result, FeatureCollection)
    assert dumps(result, sort_keys=True) == '{"features": [{"geometry": {"coordinates": [[1, 2], [3, 4]], "type": "LineString"}, "id": "line-0", "properties": {"c_nf_per_km": 720.0, "df": 1.0, "from_bus": 1.0, "g_us_per_km": 0.0, "ices": 0.389985, "in_service": 1.0, "length_km": 1.0, "max_i_ka": 0.328, "name": "line1", "parallel": 1.0, "pp_index": 0, "pp_type": "line", "r_ohm_per_km": 0.2067, "std_type": "None", "to_bus": 7.0, "type": "None", "x_ohm_per_km": 0.1897522}, "type": "Feature"}], "type": "FeatureCollection"}'

    # test exporting both
    result = geo.dump_to_geojson(net, nodes=True, branches=True)
    assert isinstance(result, FeatureCollection)
    assert dumps(result, sort_keys=True) == '{"features": [{"geometry": {"coordinates": [1.0, 2.0], "type": "Point"}, "id": "bus-1", "properties": {"in_service": 1.0, "name": "bus2", "pp_index": 1, "pp_type": "bus", "type": "b", "vn_kv": 0.4, "zone": "None"}, "type": "Feature"}, {"geometry": {"coordinates": [1.0, 3.0], "type": "Point"}, "id": "bus-7", "properties": {"in_service": 1.0, "name": "bus3", "pp_index": 7, "pp_type": "bus", "type": "b", "vn_kv": 0.4, "zone": "None"}, "type": "Feature"}, {"geometry": {"coordinates": [[1, 2], [3, 4]], "type": "LineString"}, "id": "line-0", "properties": {"c_nf_per_km": 720.0, "df": 1.0, "from_bus": 1.0, "g_us_per_km": 0.0, "ices": 0.389985, "in_service": 1.0, "length_km": 1.0, "max_i_ka": 0.328, "name": "line1", "parallel": 1.0, "pp_index": 0, "pp_type": "line", "r_ohm_per_km": 0.2067, "std_type": "None", "to_bus": 7.0, "type": "None", "x_ohm_per_km": 0.1897522}, "type": "Feature"}], "type": "FeatureCollection"}'

    # test exporting specific nodes
    result = geo.dump_to_geojson(net, nodes=[1])
    assert isinstance(result, FeatureCollection)
    assert dumps(result, sort_keys=True) == '{"features": [{"geometry": {"coordinates": [1.0, 2.0], "type": "Point"}, "id": "bus-1", "properties": {"in_service": 1.0, "name": "bus2", "pp_index": 1, "pp_type": "bus", "type": "b", "vn_kv": 0.4, "zone": "None"}, "type": "Feature"}], "type": "FeatureCollection"}'

    # test exporting specific branches
    result = geo.dump_to_geojson(net, branches=[0])
    assert isinstance(result, FeatureCollection)
    assert dumps(result, sort_keys=True) == '{"features": [{"geometry": {"coordinates": [[1, 2], [3, 4]], "type": "LineString"}, "id": "line-0", "properties": {"c_nf_per_km": 720.0, "df": 1.0, "from_bus": 1.0, "g_us_per_km": 0.0, "ices": 0.389985, "in_service": 1.0, "length_km": 1.0, "max_i_ka": 0.328, "name": "line1", "parallel": 1.0, "pp_index": 0, "pp_type": "line", "r_ohm_per_km": 0.2067, "std_type": "None", "to_bus": 7.0, "type": "None", "x_ohm_per_km": 0.1897522}, "type": "Feature"}], "type": "FeatureCollection"}'


def test_convert_geodata_to_geojson():
    pytest.importorskip("geojson")
    pytest.importorskip("pandapower")
    import pandapower as pp
    import geojson
    # Erstelle ein Beispielnetzwerk
    net = pp.create_empty_network()

    # Füge Busse hinzu
    pp.create_bus(net, 0, geodata=(10, 20))
    pp.create_bus(net, 1, geodata=(30, 40))

    # Füge Leitungen hinzu
    pp.create_line(net, 0, 1, 1, std_type="NAYY 4x50 SE", geodata=[(10, 20), (30, 40)])

    # Rufe die Funktion zum Konvertieren auf
    geo.convert_geodata_to_geojson(net)

    # Überprüfe die Ergebnisse
    assert geojson.loads(net.bus.at[0, "geo"]) == geojson.Point((10.0, 20.0))
    assert geojson.loads(net.bus.at[1, "geo"]) == geojson.Point((30.0, 40.0))
    assert geojson.loads(net.line.at[0, "geo"]) == geojson.LineString([(10.0, 20.0), (30.0, 40.0)])
    # TODO: Test could be more exhaustive (e.g. test delete=False, lonlat=True, geo_str=False)


def test_convert_gis_to_geojson():
    # TODO: implement
    pytest.skip("Not implemented")


if __name__ == "__main__":
    pytest.main(["test_geo.py"])
