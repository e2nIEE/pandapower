# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import sys
import math
import pandas as pd
from numpy import array, setdiff1d

from pandapower.auxiliary import soft_dependency_error

try:
    from shapely.geometry import Point, LineString
    shapely_INSTALLED = True
except ImportError:
    shapely_INSTALLED = False

try:
    from geopandas import GeoDataFrame, GeoSeries
    geopandas_INSTALLED = True
except ImportError:
    geopandas_INSTALLED = False

try:
    from pyproj import Proj, transform, Transformer

    pyproj_INSTALLED = True
except ImportError:
    pyproj_INSTALLED = False

try:
    import geojson
    geojson_INSTALLED = True
except ImportError:
    geojson_INSTALLED = False


def _node_geometries_from_geodata(node_geo, epsg=31467):
    """
    Creates a geopandas geodataframe from a given dataframe of with node coordinates as x and y
    values.

    :param node_geo: The dataframe containing the node coordinates (x and y values)
    :type node_geo: pandas.dataframe
    :param epsg: The epsg projection of the node coordinates
    :type epsg: int, default 31467 (= Gauss-Krüger Zone 3)
    :return: node_geodata - a geodataframe containing the node_geo and Points in the geometry column
    """
    missing_packages = array(["shapely", "geopandas"])[~array([
        shapely_INSTALLED, geopandas_INSTALLED])]
    if len(missing_packages):
        soft_dependency_error(str(sys._getframe().f_code.co_name)+"()", missing_packages)
    geoms = [Point(x, y) for x, y in node_geo[["x", "y"]].values]
    return GeoDataFrame(node_geo, crs=f"epsg:{epsg}", geometry=geoms, index=node_geo.index)


def _branch_geometries_from_geodata(branch_geo, epsg=31467):
    missing_packages = array(["shapely", "geopandas"])[~array([
        shapely_INSTALLED, geopandas_INSTALLED])]
    if len(missing_packages):
        soft_dependency_error(str(sys._getframe().f_code.co_name)+"()", missing_packages)
    geoms = GeoSeries([LineString(x) for x in branch_geo.coords.values], index=branch_geo.index,
                      crs=f"epsg:{epsg}")
    return GeoDataFrame(branch_geo, crs=f"epsg:{epsg}", geometry=geoms, index=branch_geo.index)


def _transform_node_geometry_to_geodata(node_geo):
    """
    Create x and y values from geodataframe

    :param node_geo: The dataframe containing the node geometries (as shapely points)
    :type node_geo: geopandas.GeoDataFrame
    :return: bus_geo - The given geodataframe with x and y values
    """
    node_geo["x"] = [p.x for p in node_geo.geometry]
    node_geo["y"] = [p.y for p in node_geo.geometry]
    return node_geo


def _transform_branch_geometry_to_coords(branch_geo):
    """
    Create coords entries from geodataframe geometries

    :param branch_geo: The dataframe containing the branch geometries (as shapely LineStrings)
    :type branch_geo: geopandas.GeoDataFrame
    :return: branch_geo - The given geodataframe with coords
    """
    branch_geo["coords"] = branch_geo["coords"].geometry.apply(lambda x: list(x.coords))
    return branch_geo


def _convert_xy_epsg(x, y, epsg_in=4326, epsg_out=31467):
    """
    Converts the given x and y coordinates according to the defined epsg projections.

    :param x: x-values of coordinates
    :type x: iterable
    :param y: y-values of coordinates
    :type y: iterable
    :param epsg_in: current epsg projection
    :type epsg_in: int, default 4326 (= WGS84)
    :param epsg_out: epsg projection to be transformed to
    :type epsg_out: int, default 31467 (= Gauss-Krüger Zone 3)
    :return: transformed_coords - x and y values in new coordinate system
    """
    if not pyproj_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name)+"()", "pyproj")
    in_proj = Proj(init='epsg:%i' % epsg_in)
    out_proj = Proj(init='epsg:%i' % epsg_out)
    return transform(in_proj, out_proj, x, y)


def convert_gis_to_geodata(net, node_geodata=True, branch_geodata=True):
    """
    Extracts information on bus and line geodata from the geometries of a geopandas geodataframe.

    :param net: The net for which to convert the geodata
    :type net: pandapowerNet
    :param node_geodata: flag if to extract x and y values for bus geodata
    :type node_geodata: bool, default True
    :param branch_geodata: flag if to extract coordinates values for line geodata
    :type branch_geodata: bool, default True
    :return: No output.
    """
    if node_geodata:
        _transform_node_geometry_to_geodata(net.bus_geodata)
    if branch_geodata:
        _transform_branch_geometry_to_coords(net.line_geodata)


def convert_geodata_to_gis(net, epsg=31467, node_geodata=True, branch_geodata=True):
    """
    Transforms the bus and line geodata of a net into a geopandaas geodataframe with the respective
    geometries.

    :param net: The net for which to convert the geodata
    :type net: pandapowerNet
    :param epsg: current epsg projection
    :type epsg: int, default 4326 (= WGS84)
    :param node_geodata: flag if to transform the bus geodata table
    :type node_geodata: bool, default True
    :param branch_geodata: flag if to transform the line geodata table
    :type branch_geodata: bool, default True
    :return: No output.
    """
    if node_geodata:
        net["bus_geodata"] = _node_geometries_from_geodata(net["bus_geodata"], epsg)
    if branch_geodata:
        net["line_geodata"] = _branch_geometries_from_geodata(net["line_geodata"], epsg)
    net["gis_epsg_code"] = epsg


def convert_epsg_bus_geodata(net, epsg_in=4326, epsg_out=31467):
    """
    Converts bus geodata in net from epsg_in to epsg_out

    :param net: The pandapower network
    :type net: pandapowerNet
    :param epsg_in: current epsg projection
    :type epsg_in: int, default 4326 (= WGS84)
    :param epsg_out: epsg projection to be transformed to
    :type epsg_out: int, default 31467 (= Gauss-Krüger Zone 3)
    :return: net - the given pandapower network (no copy!)
    """
    net['bus_geodata'].loc[:, "x"], net['bus_geodata'].loc[:, "y"] = _convert_xy_epsg(
        net['bus_geodata'].loc[:, "x"], net['bus_geodata'].loc[:, "y"], epsg_in, epsg_out)
    return net


def convert_crs(net, epsg_in=4326, epsg_out=31467, switch=False):
    """
    Converts bus and line geodata in net from epsg_in to epsg_out
    if GeoDataFrame data is present convert_geodata_to_gis should be used to update geometries after crs conversion

    :param net: The pandapower network
    :type net: pandapowerNet
    :param epsg_in: current epsg projection
    :type epsg_in: int, default 4326 (= WGS84)
    :param epsg_out: epsg projection to be transformed to
    :type epsg_out: int, default 31467 (= Gauss-Krüger Zone 3)
    :param switch: swap x, y coordinate while transforming
    :type switch: bool, default True
    :return: net - the given pandapower network (no copy!)
    """
    if epsg_in == epsg_out and not switch:
        return

    if not pyproj_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name)+"()", "pyproj")
    transformer = Transformer.from_crs(epsg_in, epsg_out)

    def _geo_bus_transformer(r):
        if switch:
            (y, x) = transformer.transform(r.y, r.x)
        else:
            (y, x) = transformer.transform(r.x, r.y)
        coords = r.coords
        if coords and not pd.isna(coords):
            iterator = transformer.itransform(coords, switch=switch)
            coords = list(iterator)
        return pd.Series([x, y, coords], ["x", "y", "coords"])

    def _geo_line_transformer(r):
        iterator = transformer.itransform(r, switch=switch)
        line = list(iterator)
        return line

    net.bus_geodata = net.bus_geodata.apply(lambda r: _geo_bus_transformer(r), axis=1)
    net.line_geodata.coords = net.line_geodata.coords.apply(lambda r: _geo_line_transformer(r))


def dump_to_geojson(net, nodes=False, branches=False):
    """
    Dumps all primitive values from bus, bus_geodata, res_bus, line, line_geodata and res_line into a geojson object.
    It is recommended to only dump networks using WGS84 for GeoJSON specification compliance.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param nodes: if True return contains all bus data, can be a list of bus ids that should be contained
    :type nodes: bool | list, default True
    :param branches: if True return contains all line data, can be a list of line ids that should be contained
    :type branches: bool | list, default True
    :return: geojson
    :return type: geojson.FeatureCollection
    """
    if not geojson_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "geojson")

    features = []
    # build geojson features for nodes
    if nodes:
        props = {}
        for table in ['bus', 'res_bus']:
            cols = net[table].columns
            # I use uid for the id of the feature, but it is NOT a unique identifier in the geojson structure,
            # as line and bus can have same ids.
            for uid, row in net[table].iterrows():
                prop = {
                    'pp_type': 'bus',
                    'pp_index': uid,
                }
                for c in cols:
                    try:
                        prop[c] = float(row[c])
                        if math.isnan(prop[c]):
                            prop[c] = None
                    except (ValueError, TypeError):
                        prop[c] = str(row[c])
                if uid not in props:
                    props[uid] = {}
                props[uid].update(prop)
        if isinstance(nodes, bool):
            iterator = net.bus_geodata.iterrows()
        else:
            iterator = net.bus_geodata.loc[nodes].iterrows()
        for uid, row in iterator:
            if row.coords is not None and not pd.isna(row.coords):
                # [(x, y), (x2, y2)] start and end of bus bar
                geom = geojson.LineString(row.coords)
            else:
                # this is just a bus with x, y
                geom = geojson.Point((row.x, row.y))
            features.append(geojson.Feature(geometry=geom, id=uid, properties=props[uid]))

    # build geojson features for branches
    if branches:
        props = {}
        for table in ['line', 'res_line']:
            cols = net[table].columns
            for uid, row in net[table].iterrows():
                prop = {
                    'pp_type': 'line',
                    'pp_index': uid,
                }
                for c in cols:
                    try:
                        prop[c] = float(row[c])
                        if math.isnan(prop[c]):
                            prop[c] = None
                    except (ValueError, TypeError):
                        prop[c] = str(row[c])
                if uid not in props:
                    props[uid] = {}
                props[uid].update(prop)
        if isinstance(branches, bool):
            iterator = net.line_geodata.iterrows()
        else:
            iterator = net.line_geodata.loc[branches].iterrows()
        for uid, row in iterator:
            geom = geojson.LineString(row.coords)
            features.append(geojson.Feature(geometry=geom, id=uid, properties=props[uid]))

    return geojson.FeatureCollection(features)
