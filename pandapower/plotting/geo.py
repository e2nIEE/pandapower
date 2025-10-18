# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging

import numpy as np

import sys
import math
import pandas as pd
from numpy import array

from pandapower.auxiliary import soft_dependency_error, pandapowerNet, ADict
# ADict is used as a type to ensure compatibility with pandapipes


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
    missing_packages = array(["shapely", "geopandas"])[~array([shapely_INSTALLED, geopandas_INSTALLED])]
    if len(missing_packages):
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", missing_packages)
    geoms = [Point(x, y) for x, y in node_geo[["x", "y"]].values]
    if remove_xy:
        return GeoDataFrame(crs=f"epsg:{epsg}", geometry=geoms, index=node_geo.index)
    return GeoDataFrame(node_geo, crs=f"epsg:{epsg}", geometry=geoms, index=node_geo.index)


def _branch_geometries_from_geodata(branch_geo, epsg=31467, remove_xy=False):
    missing_packages = array(["shapely", "geopandas"])[~array([shapely_INSTALLED, geopandas_INSTALLED])]
    if len(missing_packages):
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", missing_packages)
    geoms = GeoSeries(
        [LineString(x) for x in branch_geo.coords.values],
        index=branch_geo.index,
        crs=f"epsg:{epsg}",
    )
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
    return node_geo.to_json(na="drop", show_bbox=True, drop_id=False, to_wgs84=True)


def _transform_branch_geometry_to_geojson(branch_geo):
    """
    Create geojson from geodataframe geometries

    ! forces projection to wgs84 !

    :param branch_geo: The dataframe containing the branch geometries (as shapely LineStrings)
    :type branch_geo: geopandas.GeoDataFrame
    :return: A geojson object for the branch_geo
    """
    return branch_geo.to_json(na="drop", show_bbox=True, drop_id=False, to_wgs84=True)


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
    transformer = Transformer.from_crs(f"EPSG:{epsg_in}", f"EPSG:{epsg_out}", always_xy=True)
    return transformer.transform(x, y)


def abstract_convert_crs(
    net: ADict,
    epsg_in: int = 4326,
    epsg_out: int = 31467,
    component_name: str = "bus",
) -> None:
    """
    function to convert the crs of a network in place

    :param ADict net: A network subclassed from pandapower.auxiliary.ADict
    :param int epsg_in: the ESRI CRS number to convert from
    :param int epsg_out: the ESRI CRS number to convert to
    :param str component_name: name of the nodes DataFrame
    :param str branch_name: name of the branches DataFrame
    :return:
    """
    if epsg_in == epsg_out:
        return

    if not pyproj_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "pyproj")
    transformer = Transformer.from_crs(epsg_in, epsg_out, always_xy=True)

    if (
            ("geo" in net[component_name])
            and (not (net[component_name].empty or all(net[component_name].geo.isna())))
    ):
        if epsg_out != 4326:
            logger.warning("Converting geojson to crs other than WGS84 is highly discouraged.")

        def _geojson_transformer(geojson_str):
            geometry = geojson.loads(geojson_str)

            if geometry["type"] == "Point":
                x, y = geometry["coordinates"]
                x_new, y_new = transformer.transform(x, y)
                geometry["coordinates"] = [x_new, y_new]
            elif geometry["type"] == "LineString":
                new_coordinates = []
                for coord in geometry["coordinates"]:
                    x, y = coord
                    x_new, y_new = transformer.transform(x, y)
                    new_coordinates.append([x_new, y_new])
                geometry["coordinates"] = new_coordinates
            return geojson.dumps(geometry)

        net[component_name].geo = net[component_name].geo.apply(_geojson_transformer)
        return

    def _geo_component_transformer(r):
        if isinstance(r, list):
            return list(transformer.itraform(r))
        (x, y) = transformer.transform(r.x, r.y)
        if "coords" in r:
            coords = r.coords
            if coords and not any(pd.isna(coords)):
                coords = _geo_branch_transformer(coords)
            return pd.Series([x, y, coords], ["x", "y", "coords"])
        return pd.Series([x, y], ["x", "y"])

    component_geo_name = f"{component_name}_geodata"
    try:
        net[component_geo_name] = net[component_geo_name].apply(lambda r: _geo_component_transformer(r), axis=1)
        net[component_geo_name].attrs = {"crs": f"EPSG:{epsg_out}"}
    except Exception:
        pass


def convert_crs(
    net: pandapowerNet,
    epsg_in: int = 4326,
    epsg_out: int = 31467,
) -> None:
    """
    Converts bus and line geodata in net from epsg_in to epsg_out
    Supported geojson geometries are Point and LineString. Although Conversion away from WGS84 is highly discouraged.

    :param net: network
    :type net: pandapowerNet
    :param epsg_in: current epsg projection
    :type epsg_in: int, default 4326 (= WGS84)
    :param epsg_out: epsg projection to be transformed to
    :type epsg_out: int, default 31467 (= Gauss-Krüger Zone 3)
    """
    abstract_convert_crs(net, epsg_in, epsg_out, "bus")
    abstract_convert_crs(net, epsg_in, epsg_out, "line")


def dump_to_geojson_node_branch(
    net: ADict,
    node_name: str = "bus",
    branch_name: str = "line",
    nodes: bool | list[int] = False,
    branches: bool | list[int] = False,
    include_type_id: bool = True,
):
    """
    function to dump node and branch information to geojson feature collection

    :param net: the network to get the information from
    :param node_name: name of the table for node information
    :param branch_name: name of the table for branch information
    :param nodes:
    :param branches:
    :param include_type_id:
    :return:
    """
    def update_props(r: pd.Series) -> None:
        if r.name not in props:
            props[r.name] = {}
        props[r.name].update(r.to_dict())

    features = []
    elements = {node_name: nodes, branch_name: branches}
    geodata = {node_name: net[node_name].geo, branch_name: net[branch_name].geo}
    missing_geom = {node_name: 0, branch_name: 0}
    # build geojson features for nodes and branches

    for name in [node_name, branch_name]:
        element = elements[name]
        if element:
            props: dict = {}
            for table in [name, f"res_{name}"]:
                if table not in net.keys():
                    continue

                tempdf = net[table].copy(deep=True)
                if include_type_id:
                    tempdf["pp_type"] = name
                    tempdf["pp_index"] = tempdf.index
                tempdf.index = tempdf.apply(lambda r: f"{r['pp_type']}-{r['pp_index']}", axis=1)
                tempdf.drop(columns=["geo"], inplace=True, axis=1, errors="ignore")

                tempdf.apply(update_props, axis=1)
            if isinstance(element, bool):
                iterator = geodata[name].items()
            else:
                iterator = geodata[name].loc[element].items()
            for ind, geom in iterator:
                if geom is None:
                    missing_geom[name] += 1
                    continue
                uid = f"{name}-{ind}"
                features.append(geojson.Feature(geometry=geojson.loads(geom), id=uid, properties=props[uid]))
    return features, missing_geom[node_name], missing_geom[branch_name]


def dump_to_geojson(
    net: pandapowerNet,
    buses: bool | list[int] = False,
    lines: bool | list[int] = False,
    switches: bool | list[int] = False,
    trafos: bool | list[int] = False,
    t_is_3w: bool = False,
    include_type_id: bool = True,
) -> geojson.FeatureCollection:
    """
    Dumps all primitive values into a geojson object. Supports bus, line, switche, trafo, trafo3w.
    It is recommended to only dump networks using WGS84 for GeoJSON specification compliance.

    Since switches and trafos do not contain their own geodata in pandapower, geodata is taken from the components
    connected to them. Trafos are always given the geodata from the lv_bus connected at the trafo. Switches are given
    geodata from the bus the switch is connected to. They do not carry info about where the other components connected
    to them are located!

    :param net: The pandapower network
    :type net: pandapowerNet
    :param buses: if True return contains all bus data, can be a list of bus ids that should be contained
    :type buses: bool | list, default False
    :param lines: if True return contains all line data, can be a list of line ids that should be contained
    :type lines: bool | list, default False
    :param switches: if True return contains all switch data, can be a list of switch ids that should be contained (only supported for pandapowerNet)
    :type switches: bool | list, default False
    :param trafos: if True return contains all trafo data, can be a list of trafo ids that should be contained (only supported for pandapowerNet)
    :type trafos: bool | list, default False
    :param t_is_3w: if True, the trafos are treated as 3W-trafos
    :type t_is_3w: bool, default False
    :param include_type_id: if True, pp_type and pp_index is added to every feature
    :type include_type_id: bool, default true
    :return: A geojson object.
    :return type: geojson.FeatureCollection
    """
    if not geojson_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "geojson")

    try:
        if hasattr(net, "bus_geodata") or hasattr(net, "line_geodata"):
            raise UserWarning("""The supplied network uses an outdated geodata format. Please update your geodata by
                                 \rrunning `pandapower.plotting.geo.convert_geodata_to_geojson(net)`""")
    except UserWarning as e:
        logger.warning(e)
        return geojson.FeatureCollection([])

    missing_geom: dict[str, int] = {}
    features, missing_geom["bus"], missing_geom["line"] = dump_to_geojson_node_branch(
        net, "bus", "line", buses, lines, include_type_id=include_type_id
    )

    def _get_props(r, c, p) -> None:
        for col in c:
            try:
                p[col] = float(r[col])
                if math.isnan(p[col]):
                    p[col] = None
            except (ValueError, TypeError):
                p[col] = str(r[col])

    if switches:
        if isinstance(switches, bool):
            switches = net.switch.index
        if "switch" in net.keys():
            cols = net.switch.columns
            for ind, row in net.switch.loc[switches].iterrows():
                if pd.isna(row.bus):
                    # switch is not connected to a bus! Will count this as missing geometry.
                    missing_geom["switch"] += 1
                    continue
                prop = {}
                if include_type_id:
                    prop = {
                        "pp_type": "switch",
                        "pp_index": ind,
                    }
                uid = f"switch-{ind}"
                _get_props(row, cols, prop)

                # getting geodata for switches
                geom = geojson.loads(net.bus.geo.at[row.bus])
                if isinstance(geom, geojson.LineString):
                    logger.warning(f"LineString geometry not supported for type 'switch'. Skipping switch {ind}")
                    geom = None
                if geom is None:
                    missing_geom["switch"] += 1
                    continue
                features.append(geojson.Feature(geometry=geom, id=uid, properties=prop))

    if trafos:
        t_type = "trafo3w" if t_is_3w else "trafo"
        if isinstance(trafos, bool):
            trafos = net[t_type].index
        if t_type in net.keys():
            cols = net[t_type].columns
            for ind, row in net[t_type].loc[trafos].iterrows():
                prop = {}
                if include_type_id:
                    prop = {
                        "pp_type": t_type,
                        "pp_index": ind,
                    }
                uid = f"{t_type}-{ind}"
                _get_props(row, cols, prop)

                # getting geodata for trafos
                geom = geojson.loads(net.bus.geo.at[row.lv_bus])
                if isinstance(geom, geojson.LineString):
                    logger.warning(f"LineString geometry not supported for type '{t_type}'. Skipping trafo {ind}")
                if geom is None:
                    missing_geom[t_type] += 1
                    continue
                features.append(geojson.Feature(geometry=geom, id=uid, properties=prop))

    if any(missing_geom):
        missing_str = []
        for count, name in missing_geom.items():
            if count:
                missing_str.append(f"{count} {name} geometries")
        logger.warning(f"{', '.join(missing_str)} could not be converted to geojson. Please update network's geodata!")

    # find and set crs if available
    crs_node = None
    if buses and "crs" in net.bus.attrs:
        crs_node = net.bus.attrs["crs"]
    crs_branch = None
    if lines and "crs" in net.line.attrs:
        crs_branch = net.line.attrs["crs"]

    crs: dict = {"type": "name", "properties": {"name": ""}}
    if crs_node:
        if crs_branch and crs_branch != crs_node:
            raise ValueError("Node and Branch crs mismatch")
        crs["properties"]["name"] = crs_node
    elif crs_branch:
        crs["properties"]["name"] = crs_branch
    else:
        return geojson.FeatureCollection(features)
    return geojson.FeatureCollection(features, crs=crs)


def _geodata_node_check(df, geo_df, lonlat=False, drop_invalid_geodata=True):
    if not geo_df.empty:
        df["geo"] = None
        a, b = "yx" if lonlat else "xy"  # substitute x and y with a and b to reverse them if necessary
        geo_df = geo_df.astype({"x": float, "y": float})
        coords_na = geo_df[["x", "y"]].isna().sum(axis=1)
        if not drop_invalid_geodata and any(coords_na == 1):
            raise ValueError(
                f"There exists invalid bus geodata at index {list(geo_df[coords_na == 1].index)}. "
                f"Please clean up your data first or set 'drop_invalid_geodata' to True"
            )
        if any(coords_na == 1):
            logger.warning(f"bus geodata at index {list(geo_df[coords_na == 1].index)} is invalid and replaced by None")
        geo_df.dropna(inplace=True)
        geo_as_json = pd.Series(
            [f'{{"coordinates": [{x}, {y}], "type": "Point"}}' for x, y in (zip(geo_df[a], geo_df[b]))],
            index=geo_df.index,
            name="geo",
        )
        df.loc[geo_df.index, "geo"] = geo_as_json


def _geodata_branch_check(df, geo_df, lonlat=False, drop_invalid_geodata=True):
    if not geo_df.empty:
        df["geo"] = None
        geo_df = geo_df.explode("coords")
        geo_df.dropna(inplace=True)
        coords_na = geo_df["coords"].apply(lambda x: sum(pd.isna(list(x))) if isinstance(x, (list, tuple)) else 999)
        if not drop_invalid_geodata and any(coords_na == 1):
            raise ValueError(
                f"There exists invalid bus geodata at index {list(geo_df[coords_na == 1].index)}. "
                f"Please clean up your data first or set 'drop_invalid_geodata' to True"
            )
        if any(coords_na == 1):
            logger.warning(
                f"line geodata at index {list(geo_df[coords_na == 1].index)} is invalid and replaced by None"
            )
        geo_df = geo_df[coords_na == 0]
        geo_df["coords"] = geo_df["coords"].apply(
            lambda coord_list: [float(x) for x in (coord_list[::-1] if lonlat else coord_list)]
        )
        geo_as_json = geo_df.groupby(geo_df.index).apply(
            lambda x: f'{{"coordinates": {list(x.coords)}, "type": "LineString"}}'
        )
        coords_na_sum = coords_na.groupby(coords_na.index).sum()
        geo_df.loc[coords_na_sum != 0] = None
        df.loc[geo_df.index, "geo"] = geo_as_json


def abstract_convert_geodata_to_geojson(
    net: ADict,
    node_name: str = "bus",
    branch_name: str = "line",
    delete: bool = True,
    lonlat: bool = False,
    drop_invalid_geodata: bool = True,
) -> None:
    df = net[node_name]
    ldf = net[branch_name]
    bus_geo_name = node_name + "_geodata"
    line_geo_name = branch_name + "_geodata"
    geo_df = (
        net[bus_geo_name][["x", "y"]]
        if (hasattr(net, bus_geo_name) and isinstance(net[bus_geo_name], pd.DataFrame))
        else pd.DataFrame()
    )
    geo_ldf = (
        net[line_geo_name]
        if (hasattr(net, line_geo_name) and isinstance(net[line_geo_name], pd.DataFrame))
        else pd.DataFrame()
    )

    _geodata_node_check(df, geo_df, lonlat=lonlat, drop_invalid_geodata=drop_invalid_geodata)
    _geodata_branch_check(ldf, geo_ldf, lonlat=lonlat, drop_invalid_geodata=drop_invalid_geodata)

    if delete:
        if hasattr(net, bus_geo_name):
            del net[bus_geo_name]
        if hasattr(net, line_geo_name):
            del net[line_geo_name]


def convert_geodata_to_geojson(
    net: pandapowerNet, delete: bool = True, lonlat: bool = False, drop_invalid_geodata: bool = True
) -> None:
    """
    Converts bus_geodata and line_geodata to bus.geo and line.geo column entries.

    It is expected that any input network has its coords in WGS84 (epsg:4326) projection.
    If this is not the case use convert_crs to convert the network to WGS84.

    :param net: The pandapower network containing line_geodata and bus_geodata in WGS84!
    :type net: pandapowerNet
    :param delete: If True, the geodataframes are deleted after conversion
    :type delete: bool, default True
    :param lonlat: If True, the coordinates are expected to be in lonlat format (x=lon, y=lat)
    :type lonlat: bool, default False
    :param drop_invalid_geodata: If True, entries containing invalid geo coordinates e.g. None, np.nan will be dropped
    :type drop_invalid_geodata: bool, default True
    """
    abstract_convert_geodata_to_geojson(net, "bus", "line", delete, lonlat, drop_invalid_geodata)
    abstract_convert_geodata_to_geojson(net, "bus_dc", "line_dc", delete, lonlat, drop_invalid_geodata)


def _is_valid_number(value):
    try:
        float_value = float(value)
        return not (isinstance(value, float) and np.isnan(float_value))
    except (ValueError, TypeError):
        return False


def abstract_convert_gis_to_geojson(
    net: ADict, node_name: str = "bus", branch_name: str = "line", delete: bool = True
) -> None:
    net[node_name]["geo"] = _transform_node_geometry_to_geojson(net[node_name + "_geodata"])
    net[branch_name]["geo"] = _transform_branch_geometry_to_geojson(net[branch_name + "_geodata"])

    if delete:
        del net[node_name + "_geodata"]
        del net[branch_name + "_geodata"]


def convert_gis_to_geojson(net: pandapowerNet, delete: bool = True) -> None:
    """
    Transforms the bus and line geodataframes of a net into a geojson object.

    :param net: The net for which to convert the geodataframes
    :type net: pandapowerNet
    :param delete: If True, the geodataframes are deleted after conversion
    :type delete: bool, default True
    :return: No output.
    """
    abstract_convert_gis_to_geojson(net, "bus", "line", delete)
