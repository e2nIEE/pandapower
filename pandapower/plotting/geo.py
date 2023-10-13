# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
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
    from pyproj import Transformer

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
    branch_geo["coords"] = branch_geo.geometry.apply(lambda x: list(x.coords))
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
    transformer = Transformer.from_crs(f'EPSG:{epsg_in}', f'EPSG:{epsg_out}', always_xy=True)
    return transformer.transform(x, y)


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


def convert_crs(net, epsg_in=4326, epsg_out=31467):
    """
    This function works for pandapowerNet and pandapipesNet. Documentation will refer to names from pandapower.
    Converts bus and line geodata in net from epsg_in to epsg_out
    if GeoDataFrame data is present convert_geodata_to_gis should be used to update geometries after crs conversion

    :param net: The pandapower network
    :type net: pandapowerNet|pandapipesNet
    :param epsg_in: current epsg projection
    :type epsg_in: int, default 4326 (= WGS84)
    :param epsg_out: epsg projection to be transformed to
    :type epsg_out: int, default 31467 (= Gauss-Krüger Zone 3)
    :return: net - the given pandapower network (no copy!)
    """
    is_pandapower = net.__class__.__name__ == 'pandapowerNet'
    if epsg_in == epsg_out:
        return

    if not pyproj_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name)+"()", "pyproj")
    transformer = Transformer.from_crs(epsg_in, epsg_out, always_xy=True)

    def _geo_node_transformer(r):
        (x, y) = transformer.transform(r.x, r.y)
        if is_pandapower:
            coords = r.coords
            if coords and not pd.isna(coords):
                coords = _geo_branch_transformer(coords)
            return pd.Series([x, y, coords], ["x", "y", "coords"])
        else:
            return pd.Series([x, y], ["x", "y"])

    def _geo_branch_transformer(r):
        return list(transformer.itransform(r))

    if is_pandapower:
        net.bus_geodata = net.bus_geodata.apply(lambda r: _geo_node_transformer(r), axis=1)
        net.line_geodata.coords = net.line_geodata.coords.apply(lambda r: _geo_branch_transformer(r))
        net.bus_geodata.attrs = {"crs": f"EPSG:{epsg_out}"}
        net.line_geodata.attrs = {"crs": f"EPSG:{epsg_out}"}
    else:
        net.junction_geodata = net.junction_geodata.apply(lambda r: _geo_node_transformer(r), axis=1)
        net.pipe_geodata.coords = net.pipe_geodata.coords.apply(lambda r: _geo_branch_transformer(r))
        net.junction_geodata.attrs = {"crs": f"EPSG:{epsg_out}"}
        net.pipe_geodata.attrs = {"crs": f"EPSG:{epsg_out}"}


def dump_to_geojson(net, nodes=False, branches=False):
    """
    This function works for pandapowerNet and pandapipesNet. Documentation will refer to names from pandapower.
    Dumps all primitive values from bus, bus_geodata, res_bus, line, line_geodata and res_line into a geojson object.
    It is recommended to only dump networks using WGS84 for GeoJSON specification compliance.

    :param net: The pandapower network
    :type net: pandapowerNet|pandapipesNet
    :param nodes: if True return contains all bus data, can be a list of bus ids that should be contained
    :type nodes: bool | list, default False
    :param branches: if True return contains all line data, can be a list of line ids that should be contained
    :type branches: bool | list, default False
    :return: geojson
    :return type: geojson.FeatureCollection
    """
    is_pandapower = net.__class__.__name__ == 'pandapowerNet'
    if is_pandapower:
        node_geodata = net.bus_geodata
        branch_geodata = net.line_geodata
    else:
        node_geodata = net.junction_geodata
        branch_geodata = net.pipe_geodata

    if not geojson_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "geojson")

    features = []
    # build geojson features for nodes
    if nodes:
        props = {}
        for table in (['bus', 'res_bus'] if is_pandapower else ['junction', 'res_junction']):
            if table not in net.keys():
                continue
            cols = net[table].columns
            # I use uid for the id of the feature, but it is NOT a unique identifier in the geojson structure,
            # as line and bus (pipe and junction) can have same ids.
            for uid, row in net[table].iterrows():
                prop = {
                    'pp_type': 'bus' if is_pandapower else 'junction',
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
            iterator = node_geodata.iterrows()
        else:
            iterator = node_geodata.loc[nodes].iterrows()
        for uid, row in iterator:
            if is_pandapower and row.coords is not None and not pd.isna(row.coords):
                # [(x, y), (x2, y2)] start and end of bus bar
                geom = geojson.LineString(row.coords)
            else:
                # this is just a bus with x, y
                geom = geojson.Point((row.x, row.y))
            features.append(geojson.Feature(geometry=geom, id=uid, properties=props[uid]))

    # build geojson features for branches
    if branches:
        props = {}
        for table in (['line', 'res_line'] if is_pandapower else ['pipe', 'res_pipe']):
            if table not in net.keys():
                continue
            cols = net[table].columns
            for uid, row in net[table].iterrows():
                prop = {
                    'pp_type': 'line' if is_pandapower else 'pipe',
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

        # Iterating over pipe_geodata won't work
        # pipe_geodata only contains pipes that have inflection points!
        if isinstance(branches, bool):
            # if all iterating over pipe
            iterator = net.line_geodata.iterrows() if is_pandapower else net.pipe.iterrows()
        else:
            iterator = net.line_geodata.loc[branches].iterrows() if is_pandapower else net.pipe.loc[branches].iterrows()
        for uid, row in iterator:
            if not is_pandapower:
                coords = []
                from_coords = net.junction_geodata.loc[row.from_junction]
                to_coords = net.junction_geodata.loc[row.to_junction]
                coords.append([float(from_coords.x), float(from_coords.y)])
                if uid in net.pipe_geodata:
                    coords.append(net.pipe_geodata.loc[uid].coords)
                coords.append([float(to_coords.x), float(to_coords.y)])

            geom = geojson.LineString(row.coords if is_pandapower else coords)
            features.append(geojson.Feature(geometry=geom, id=uid, properties=props[uid]))
    # find and set crs if available
    crs_node = None
    if nodes and "crs" in node_geodata.attrs:
        crs_node = node_geodata.attrs["crs"]
    crs_branch = None
    if branches and "crs" in branch_geodata.attrs:
        crs_branch = branch_geodata.attrs["crs"]

    crs = {
        "type": "name",
        "properties": {
            "name": ""
        }
    }
    if crs_node:
        if crs_branch and crs_branch != crs_node:
            raise ValueError("Node and Branch crs mismatch")
        crs["properties"]["name"] = crs_node
    elif crs_branch:
        crs["properties"]["name"] = crs_branch
    else:
        crs = None
    if crs:
        return geojson.FeatureCollection(features, crs=crs)
    return geojson.FeatureCollection(features)
