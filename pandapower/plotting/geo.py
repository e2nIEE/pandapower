# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from typing import List, Tuple, TYPE_CHECKING, Dict, Any, Union

import numpy as np

# TYPE_CHECKING is used to avoid circular imports, see https://stackoverflow.com/a/39757388
if TYPE_CHECKING:
    import pandapipes
from typing_extensions import deprecated

import sys
import math
import pandas as pd
from numpy import array

import pandapower
from pandapower.auxiliary import soft_dependency_error, pandapowerNet

# get logger (same as in simple_plot)
try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

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


def _node_geometries_from_geodata(node_geo, epsg=31467, remove_xy=False):
    """
    Creates a geopandas geodataframe from a given dataframe of with node coordinates as x and y
    values.

    :param node_geo: The dataframe containing the node coordinates (x and y values)
    :type node_geo: pandas.dataframe
    :param epsg: The epsg projection of the node coordinates
    :type epsg: int, default 31467 (= Gauss-Krüger Zone 3)
    :param remove_xy: If x/y and coords columns should be removed from the geodataframe
    :type remove_xy: bool, default False
    :return: node_geodata - a geodataframe containing the node_geo and Points in the geometry column
    """
    missing_packages = array(["shapely", "geopandas"])[~array([
        shapely_INSTALLED, geopandas_INSTALLED])]
    if len(missing_packages):
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", missing_packages)
    geoms = [Point(x, y) for x, y in node_geo[["x", "y"]].values]
    if remove_xy:
        return GeoDataFrame(crs=f"epsg:{epsg}", geometry=geoms, index=node_geo.index)
    return GeoDataFrame(node_geo, crs=f"epsg:{epsg}", geometry=geoms, index=node_geo.index)


def _branch_geometries_from_geodata(branch_geo, epsg=31467, remove_xy=False):
    missing_packages = array(["shapely", "geopandas"])[~array([
        shapely_INSTALLED, geopandas_INSTALLED])]
    if len(missing_packages):
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", missing_packages)
    geoms = GeoSeries([LineString(x) for x in branch_geo.coords.values], index=branch_geo.index,
                      crs=f"epsg:{epsg}")
    if remove_xy:
        return GeoDataFrame(crs=f"epsg:{epsg}", geometry=geoms, index=branch_geo.index)
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


def _transform_node_geometry_to_geojson(node_geo):
    """
    Create geojson string from geodataframe geometries

    ! forces projection to wgs84 !

    :param node_geo: The dataframe containing the node geometries (as shapely points)
    :type node_geo: geopandas.GeoDataFrame
    :return: A geojson string for the node_geo
    """
    return node_geo.to_json(na='drop', show_bbox=True, drop_id=False, to_wgs84=True)


def _transform_branch_geometry_to_geojson(branch_geo):
    """
    Create geojson from geodataframe geometries

    ! forces projection to wgs84 !

    :param branch_geo: The dataframe containing the branch geometries (as shapely LineStrings)
    :type branch_geo: geopandas.GeoDataFrame
    :return: A geojson object for the branch_geo
    """
    return branch_geo.to_json(na='drop', show_bbox=True, drop_id=False, to_wgs84=True)


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
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "pyproj")
    transformer = Transformer.from_crs(f'EPSG:{epsg_in}', f'EPSG:{epsg_out}', always_xy=True)
    return transformer.transform(x, y)


@deprecated("Use convert_gis_to_geojson instead. Support for geodata will be dropped.")
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


@deprecated(
    "Use convert_geodata_to_geojson instead. Support for gis will be dropped.\
    To get a geodataframe use GeoDataFrame.from_features."
)
def convert_geodata_to_gis(net, epsg=31467, node_geodata=True, branch_geodata=True, remove_xy=False):
    """
    Transforms the bus and line geodata of a net into a geopandas geodataframe with the respective
    geometries.

    :param net: The net for which to convert the geodata
    :type net: pandapowerNet
    :param epsg: current epsg projection
    :type epsg: int, default 4326 (= WGS84)
    :param node_geodata: flag if to transform the bus geodata table
    :type node_geodata: bool, default True
    :param branch_geodata: flag if to transform the line geodata table
    :type branch_geodata: bool, default True
    :param remove_xy: flag if to remove x,y and coords columns from geodata tables
    :return: No output.
    """
    converted = False
    if node_geodata and "bus_geodata" in net:
        net["bus_geodata"] = _node_geometries_from_geodata(net["bus_geodata"], epsg, remove_xy)
        converted = True
    if branch_geodata and "line_geodata" in net:
        net["line_geodata"] = _branch_geometries_from_geodata(net["line_geodata"], epsg, remove_xy)
        converted = True
    if converted:
        net["gis_epsg_code"] = epsg


@deprecated("Use convert_crs instead. Networks should not use different crs for bus and line geodata.")
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


def convert_crs(net: pandapowerNet or 'pandapipes.pandapipesNet', epsg_in=4326, epsg_out=31467):
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

    if ('geo' in net.bus and not all(net.bus.geo.isna()) and
            'geo' in net.line and not all(net.line.geo.isna()) and
            epsg_out == 4326):
        # by definition geojson is in wgs84
        return

    if not pyproj_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "pyproj")
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


def dump_to_geojson(
        net: pandapowerNet or 'pandapipes.pandapipesNet',
        nodes: Union[bool, List[int]] = False,
        branches: Union[bool, List[int]] = False,
        switches: Union[bool,  List[int]] = False,
        trafos: Union[bool, List[int]] = False,
        t_is_3w: bool = False
) -> geojson.FeatureCollection:
    """
    This function works for pandapowerNet and pandapipesNet. Documentation will refer to names from pandapower.
    Dumps all primitive values from bus, bus_geodata, res_bus, line, line_geodata and res_line into a geojson object.
    It is recommended to only dump networks using WGS84 for GeoJSON specification compliance.

    Since switches and trafos do not contain their own geodata in pandapower, geodata is taken from the components
    connected to them. Trafos are always given the geodata from the lv_bus connected at the trafo. Switches are given
    geodata from the bus the switch is connected to. They do not carry info about where the other components connected
    to them are located!

    :param net: The pandapower network
    :type net: pandapowerNet|pandapipesNet
    :param nodes: if True return contains all bus data, can be a list of bus ids that should be contained
    :type nodes: bool | list, default False
    :param branches: if True return contains all line data, can be a list of line ids that should be contained
    :type branches: bool | list, default False
    :param switches: if True return contains all switch data, can be a list of switch ids that should be contained (only supported for pandapowerNet)
    :type switches: bool | list, default False
    :param trafos: if True return contains all trafo data, can be a list of trafo ids that should be contained (only supported for pandapowerNet)
    :type trafos: bool | list, default False
    :param t_is_3w: if True, the trafos are treated as 3W-trafos
    :type t_is_3w: bool, default False
    :return: A geojson object.
    :return type: geojson.FeatureCollection
    """
    is_pandapower = net.__class__.__name__ == 'pandapowerNet'

    if not geojson_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "geojson")

    try:
        if is_pandapower:
            if hasattr(net, "bus_geodata") or hasattr(net, "line_geodata"):
                raise UserWarning("""The supplied network uses an outdated geodata format. Please update your geodata by
                                     \rrunning `pandapower.plotting.geo.convert_geodata_to_geojson(net)`""")
            else:
                node_geodata = net.bus.geo
                branch_geodata = net.line.geo
        else:
            if hasattr(net, "junction_geodata") or hasattr(net, "pipe_geodata"):
                raise UserWarning("""The supplied network uses an outdated geodata format. Please update your geodata by
                                     \rrunning `pandapower.plotting.geo.convert_geodata_to_geojson(net)`""")
            else:
                node_geodata = net.junction.geo
                branch_geodata = net.pipe.geo
    except UserWarning as e:
        logger.warning(e)
        return geojson.FeatureCollection([])

    def _get_props(r, c, p) -> None:
        for col in c:
            try:
                p[col] = float(r[col])
                if math.isnan(p[col]):
                    p[col] = None
            except (ValueError, TypeError):
                p[col] = str(r[col])

    def update_props(r: pd.Series) -> None:
        if r.name not in props:
            props[r.name] = {}
        props[r.name].update(r.to_dict())

    missing_geom: List[int] = [0, 0, 0, 0]  # missing nodes, branches, switches, trafos
    features = []
    # build geojson features for nodes
    if nodes:
        props = {}
        for table in (['bus', 'res_bus'] if is_pandapower else ['junction', 'res_junction']):
            if table not in net.keys():
                continue

            tempdf = net[table].copy(deep=True)
            tempdf['pp_type'] = 'bus' if is_pandapower else 'junction'
            tempdf['pp_index'] = tempdf.index
            tempdf.index = tempdf.apply(lambda r: f"{r['pp_type']}-{r['pp_index']}", axis=1)
            tempdf.drop(columns=['geo'], inplace=True, axis=1, errors='ignore')

            tempdf.apply(update_props, axis=1)
        if isinstance(nodes, bool):
            iterator = node_geodata.items()
        else:
            iterator = node_geodata.loc[nodes].items()
        for ind, geom in iterator:
            if geom is None or pd.isna(geom) or geom == "[]":
                missing_geom[0] += 1
                continue
            uid = f"{'bus' if is_pandapower else 'junction'}-{ind}"
            features.append(geojson.Feature(geometry=geojson.loads(geom), id=uid, properties=props[uid]))

    # build geojson features for branches
    if branches:
        props = {}
        for table in (['line', 'res_line'] if is_pandapower else ['pipe', 'res_pipe']):
            if table not in net.keys():
                continue

            tempdf = net[table].copy(deep=True)
            tempdf['pp_type'] = 'line' if is_pandapower else 'pipe'
            tempdf['pp_index'] = tempdf.index
            tempdf.index = tempdf.apply(lambda r: f"{r['pp_type']}-{r['pp_index']}", axis=1)
            tempdf.drop(columns=['geo'], inplace=True, axis=1, errors='ignore')

            tempdf.apply(update_props, axis=1)

        # Iterating over pipe_geodata won't work
        # pipe_geodata only contains pipes that have inflection points!
        if isinstance(branches, bool):
            # if all iterating over pipe
            iterator = branch_geodata.items()
        else:
            iterator = branch_geodata.loc[branches].items()
        for ind, geom in iterator:
            if geom is None or pd.isna(geom) or geom == "[]":
                missing_geom[1] += 1
                continue
            uid = f"{'line' if is_pandapower else 'pipe'}-{ind}"
            features.append(geojson.Feature(geometry=geojson.loads(geom), id=uid, properties=props[uid]))

    if switches and is_pandapower:
        if isinstance(switches, bool):
            switches = net.switch.index
        if 'switch' in net.keys():
            cols = net.switch.columns
            for ind, row in net.switch.loc[switches].iterrows():
                if pd.isna(row.bus):
                    # switch is not connected to a bus! Will count this as missing geometry.
                    missing_geom[2] += 1
                    continue
                prop = {
                    'pp_type': 'switch',
                    'pp_index': ind,
                }
                uid = f"switch-{ind}"
                _get_props(row, cols, prop)

                # getting geodata for switches
                geom = geojson.loads(net.bus.geo.at[row.bus])
                if isinstance(geom, geojson.LineString):
                    logger.warning(f"LineString geometry not supported for type 'switch'. Skipping switch {ind}")
                    geom = None
                if geom is None or geom == "[]":
                    missing_geom[2] += 1
                    continue
                features.append(geojson.Feature(geometry=geom, id=uid, properties=prop))

        if trafos and is_pandapower:
            t_type = 'trafo3w' if t_is_3w else 'trafo'
            if isinstance(trafos, bool):
                trafos = net[t_type].index
            if t_type in net.keys():
                cols = net[t_type].columns
                for ind, row in net[t_type].loc[trafos].iterrows():
                    prop = {
                        'pp_type': t_type,
                        'pp_index': ind,
                    }
                    uid = f"{t_type}-{ind}"
                    _get_props(row, cols, prop)

                    # getting geodata for switches
                    geom = geojson.loads(net.bus.geo.at[row.lv_bus])
                    if isinstance(geom, geojson.LineString):
                        logger.warning(f"LineString geometry not supported for type '{t_type}'. Skipping trafo {ind}")
                    if geom is None or geom == "[]":
                        missing_geom[3] += 1
                        continue
                    features.append(geojson.Feature(geometry=geom, id=uid, properties=prop))

    if any(missing_geom):
        missing_str = []
        if missing_geom[0]:
            missing_str.append(f"{missing_geom[0]} branch geometries")
        if missing_geom[1]:
            missing_str.append(f"{missing_geom[1]} node geometries")
        if missing_geom[2]:
            missing_str.append(f"{missing_geom[2]} switch geometries")
        if missing_geom[3]:
            missing_str.append(f"{missing_geom[3]} trafo geometries")
        logger.warning(f"{', '.join(missing_str)} could not be converted to geojson. Please update network's geodata!")

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


def convert_geodata_to_geojson(
        net: pandapowerNet or 'pandapipes.pandapipesNet',
        delete: bool = True,
        lonlat: bool = False) -> None:
    """
    Converts bus_geodata and line_geodata to bus.geo and line.geo column entries.
    If used on pandapipesNet, the junction_geodata and pipe_geodata are converted.

    It is expected that any input network has its coords in WGS84 (epsg:4326) projection.
    If this is not the case use convert_crs to convert the network to WGS84.

    :param net: The pandapower network containing line_geodata and bus_geodata in WGS84!
    :type net: pandapowerNet
    :param delete: If True, the geodataframes are deleted after conversion
    :type delete: bool, default True
    :param lonlat: If True, the coordinates are expected to be in lonlat format (x=lon, y=lat)
    :type lonlat: bool, default False
    """
    is_pandapower = net.__class__.__name__ == 'pandapowerNet'

    if is_pandapower:
        df = net.bus
        ldf = net.line
        geo_df = net.bus_geodata if (hasattr(net, 'bus_geodata') and isinstance(net.bus_geodata, pd.DataFrame)) else pd.DataFrame()
        geo_ldf = net.line_geodata if (hasattr(net, 'line_geodata') and isinstance(net.line_geodata, pd.DataFrame)) else pd.DataFrame()
    else:
        df = net.junction
        ldf = net.pipe
        geo_df = net.junction_geodata if (hasattr(net, 'junction_geodata') and isinstance(net.junction_geodata, pd.DataFrame)) else pd.DataFrame()
        geo_ldf = net.pipe_geodata if (hasattr(net, 'pipe_geodata') and isinstance(net.pipe_geodata, pd.DataFrame)) else pd.DataFrame()

    a, b = "yx" if lonlat else "xy"  # substitute x and y with a and b to reverse them if necessary
    if not geo_df.empty:
        df["geo"] = geo_df.apply(lambda r: f'{{"coordinates": [{r[a]}, {r[b]}], "type": "Point"}}', axis=1)

    ldf["geo"] = np.nan
    for l_id in ldf.index:
        if l_id not in geo_ldf.index:
            continue
        # pandapipes currently only stores inflection points for pipes. This function will inject start and end points.
        if is_pandapower:
            coords: List[List[float]] = [[y, x] if lonlat else [x, y] for x, y in geo_ldf.coords.at[l_id]]
        else:
            coords: List[List[float]] = []
            from_coords = geo_df.loc[ldf[l_id].from_junction]
            to_coords = geo_df.loc[ldf[l_id].to_junction]
            coords.append([float(from_coords.x), float(from_coords.y)])
            if l_id in net.pipe_geodata:
                coords.append(geo_ldf.loc[l_id].coords)
            coords.append([float(to_coords.x), float(to_coords.y)])
        if not coords:
            continue
        ls = f'{{"coordinates": {coords}, "type": "LineString"}}'
        ldf["geo"] = ldf["geo"].astype(object)
        ldf.geo.at[l_id] = ls

    if delete:
        if is_pandapower:
            if hasattr(net, 'bus_geodata'):del net.bus_geodata
            if hasattr(net, 'line_geodata'): del net.line_geodata
        else:
            if hasattr(net, 'junction_geodata'): del net.junction_geodata
            if hasattr(net, 'pipe_geodata'): del net.pipe_geodata


def convert_gis_to_geojson(
        net: pandapowerNet or 'pandapipes.pandapipesNet',
        delete: bool = True) -> None:
    """
    Transforms the bus and line geodataframes of a net into a geojson object.

    :param net: The net for which to convert the geodataframes
    :type net: pandapowerNet
    :param delete: If True, the geodataframes are deleted after conversion
    :type delete: bool, default True
    :return: No output.
    """

    is_pandapower = net.__class__.__name__ == 'pandapowerNet'

    if is_pandapower:
        net.bus["geo"] = _transform_node_geometry_to_geojson(net["bus_geodata"])
        net.line["geo"] = _transform_branch_geometry_to_geojson(net["line_geodata"])

        if delete:
            del net.bus_geodata
            del net.line_geodata
    else:
        net.junction["geo"] = _transform_node_geometry_to_geojson(net["junction_geodata"])
        net.pipe["geo"] = _transform_branch_geometry_to_geojson(net["pipe_geodata"])

        if delete:
            del net.junction_geodata
            del net.pipe_geodata
