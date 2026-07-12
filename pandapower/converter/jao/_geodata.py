# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""
Geodata extraction from the JAO HTML map and its transfer onto buses and lines.

The published HTML embeds the line geometry inside a leaflet ``htmlwidget`` JSON blob.
:func:`parse_html_str` extracts the line endpoints; :func:`add_bus_geo` assigns a coordinate to
each bus from the geometry of its connected lines.
"""

import re
import json
import logging

import numpy as np
import pandas as pd

from pandapower.io_utils import pandapowerNet
from pandapower.plotting import set_line_geodata_from_bus_geodata

logger = logging.getLogger(__name__)

# The leaflet widget id changes with every JAO release, so it must not be hard-coded. Match any
# hex id and pick the JSON block that actually carries the map data (an "addPolylines" call).
_WIDGET_SCRIPT = re.compile(
    r'<script[^>]*type="application/json"[^>]*data-for="htmlwidget-[0-9a-f]+"[^>]*>(.*?)</script>',
    re.DOTALL)


def parse_html_str(html_str: str) -> pd.DataFrame:
    """Parse the embedded leaflet JSON and return per-line endpoint coordinates.

    Returns a tidy DataFrame with columns ``[EIC_Code, name, bus, geo_dim, value]`` where ``bus``
    is ``from``/``to`` and ``geo_dim`` is ``lng``/``lat``.

    Raises ``json.JSONDecodeError``/``KeyError`` if no map widget is found and ``AssertionError``
    if the EIC and geometry lists have different lengths.
    """
    geo_data = _extract_widget_calls(html_str)
    methods_pos = pd.Series({item["method"]: i for i, item in enumerate(geo_data)})
    polylines = geo_data[methods_pos.at["addPolylines"]]["args"]

    if len(polylines[6]) != len(polylines[0]):
        raise AssertionError("The lists of EIC Code data and geo data are not of the same length.")

    eic_start = "EIC Code:<b> "
    line_eic = [tip[tip.find(eic_start) + len(eic_start):] for tip in polylines[6]]
    line_name = [_filter_name(tip) for tip in polylines[6]]
    line_geo_data = pd.concat(
        [_lng_lat_to_df(polylines[0][i][0][0], line_eic[i], line_name[i])
         for i in range(len(polylines[0]))],
        ignore_index=True)

    for col in ("EIC_Code", "name"):
        line_geo_data[col] = line_geo_data[col].str.strip()
    return line_geo_data


def _extract_widget_calls(html_str: str) -> list:
    """Return the ``x.calls`` list of the leaflet map widget from the HTML.

    Scans every ``application/json`` htmlwidget script and returns the calls of the first one
    that contains an ``addPolylines`` method (the map layer).
    """
    for match in _WIDGET_SCRIPT.finditer(html_str):
        try:
            calls = json.loads(match.group(1))["x"]["calls"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if any(item.get("method") == "addPolylines" for item in calls):
            return calls
    raise json.JSONDecodeError("No leaflet map widget with polyline data found.", html_str, 0)


def _filter_name(tooltip: str) -> str:
    name_start, name_end = "<b>NE name: ", "</b>"
    pos0 = tooltip.find(name_start) + len(name_start)
    pos1 = tooltip.find(name_end, pos0)
    if pos0 < len(name_start) or pos1 < pos0:
        raise AssertionError("Could not parse the NE name from a map tooltip.")
    return tooltip[pos0:pos1]


def _lng_lat_to_df(coords: dict, line_eic: str, line_name: str) -> pd.DataFrame:
    """Turn a ``{"lng": [..], "lat": [..]}`` endpoint dict into four tidy rows."""
    return pd.DataFrame([
        [line_eic, line_name, "from", "lng", coords["lng"][0]],
        [line_eic, line_name, "to", "lng", coords["lng"][1]],
        [line_eic, line_name, "from", "lat", coords["lat"][0]],
        [line_eic, line_name, "to", "lat", coords["lat"][1]],
    ], columns=["EIC_Code", "name", "bus", "geo_dim", "value"])


def add_bus_geo(net: pandapowerNet, line_geo_data: pd.DataFrame) -> None:
    """Assign a coordinate to every bus from the geometry of its connected lines.

    Coordinates are looked up primarily by EIC code and fall back to the line name where the EIC
    is duplicated or missing. Ambiguous cases are reduced by rounding and by picking the most
    frequent coordinate. The result is written as GeoJSON strings into ``net.bus.geo``.
    """
    slicer = pd.IndexSlice
    lgd_eic_bus = line_geo_data.pivot_table(values="value", index=["EIC_Code", "bus"],
                                            columns="geo_dim")
    lgd_name_bus = line_geo_data.pivot_table(values="value", index=["name", "bus"],
                                             columns="geo_dim")
    lgd_bus = pd.concat([
        lgd_eic_bus.set_axis(_extend_index(lgd_eic_bus, "EIC_Code")),
        lgd_name_bus.set_axis(_extend_index(lgd_name_bus, "name")),
    ])
    dupl_eics = net.line.EIC_Code.loc[net.line.EIC_Code.duplicated()]
    dupl_names = net.line.name.loc[net.line.name.duplicated()]

    def geo_json(bus_geo: pd.Series) -> str:
        return f'{{"coordinates": [{bus_geo.at["lng"]}, {bus_geo.at["lat"]}], "type": "Point"}}'

    def bus_geo(bus: int) -> str | None:
        from_excerpt = net.line.loc[net.line.from_bus == bus, ["EIC_Code", "name", "Tieline"]]
        to_excerpt = net.line.loc[net.line.to_bus == bus, ["EIC_Code", "name", "Tieline"]]
        line_excerpt = pd.concat([from_excerpt, to_excerpt])
        n_ends = len(line_excerpt)
        if n_ends == 0:
            logger.error(f"Bus {bus} (name {net.bus.at[bus, 'name']}) is not found in "
                         "line_geo_data.")
            return None

        is_dupl = pd.concat([
            _isin_frame(from_excerpt, dupl_eics, dupl_names, "from"),
            _isin_frame(to_excerpt, dupl_eics, dupl_names, "to"),
        ])
        is_missing = pd.DataFrame({
            "EIC": ~line_excerpt.EIC_Code.isin(
                lgd_bus.loc["EIC_Code"].index.get_level_values("identifier")),
            "name": ~line_excerpt.name.isin(
                lgd_bus.loc["name"].index.get_level_values("identifier")),
        }).set_axis(is_dupl.index, axis=0)
        is_tieline = pd.Series(net.line.loc[is_dupl.index.get_level_values("line_index"),
                                            "Tieline"].values, index=is_dupl.index)

        # default to the EIC code, switch to the line name where the EIC is duplicated/missing
        access = pd.DataFrame({
            "col_name": "EIC_Code",
            "identifier": line_excerpt.EIC_Code.values,
            "bus": is_dupl.index.get_level_values("bus").values,
        })
        take_from_name = ((is_dupl.EIC | is_missing.EIC)
                          & (~is_dupl.name & ~is_missing.name)).values
        access.loc[take_from_name, "col_name"] = "name"
        access.loc[take_from_name, "identifier"] = line_excerpt.name.loc[take_from_name].values

        keep = (~(is_dupl | is_missing)).any(axis=1).values
        if np.all(is_missing):
            msg = (f"For bus {bus} (name {net.bus.at[bus, 'name']}), {n_ends} line ends were found "
                   "but no EIC_Codes or names of the connected lines exist in the HTML geo data.")
            logger.debug(msg) if is_tieline.all() else logger.warning(msg)
            return None
        if keep.sum() == 0:
            logger.info(f"For {bus=}, all EIC_Codes and names of connected lines are ambiguous. "
                        "No geo data is dropped at this point.")
            keep[(~is_missing).any(axis=1)] = True
        access = access.loc[keep]

        this_bus_geo = lgd_bus.loc[slicer[access.col_name, access.identifier, access.bus], :]
        if len(this_bus_geo) > 1:
            this_bus_geo = this_bus_geo.loc[this_bus_geo.round(2).drop_duplicates().index]

        if len(this_bus_geo) == 1:
            return geo_json(this_bus_geo.iloc[0])
        if len(this_bus_geo) == 2:
            how_often = pd.Series(
                [int((np.isclose(lgd_eic_bus["lat"], this_bus_geo["lat"].iat[i])
                      & np.isclose(lgd_eic_bus["lng"], this_bus_geo["lng"].iat[i])).sum())
                 for i in range(len(this_bus_geo))],
                index=this_bus_geo.index)
            if how_often.at[how_often.idxmax()] >= 1:
                logger.warning(f"Bus {bus} (name {net.bus.at[bus, 'name']}) was found multiple "
                               "times in line_geo_data. The first of the most used geo positions "
                               "is used.")
            return geo_json(this_bus_geo.loc[how_often.idxmax()])
        return None

    net.bus.geo = [bus_geo(bus) for bus in net.bus.index]


def _extend_index(pivot: pd.DataFrame, col_name: str) -> pd.MultiIndex:
    """Prepend a constant ``col_name`` level to a ``(identifier, bus)`` pivot index."""
    frame = pivot.index.to_frame().assign(col_name=col_name).rename(
        columns={pivot.index.names[0]: "identifier"})
    return pd.MultiIndex.from_frame(frame.loc[:, ["col_name", "identifier", "bus"]])


def _isin_frame(excerpt: pd.DataFrame, dupl_eics: pd.Series, dupl_names: pd.Series,
                side: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"EIC": excerpt.EIC_Code.isin(dupl_eics).values,
         "name": excerpt.name.isin(dupl_names).values},
        index=pd.MultiIndex.from_product([[side], excerpt.index],
                                         names=["bus", "line_index"]))


def fill_geo_at_one_sided_branches_without_geo_extent(net: pandapowerNet) -> None:
    """Propagate bus geodata across branches that have geodata on only one end."""

    def availability(net: pandapowerNet) -> dict:
        av = {}
        with_geo = net.bus.index[~net.bus.geo.isnull()]
        av["lines_fbw_tbwo"] = net.line.index[net.line.from_bus.isin(with_geo)
                                              & ~net.line.to_bus.isin(with_geo)]
        av["lines_fbwo_tbw"] = net.line.index[~net.line.from_bus.isin(with_geo)
                                              & net.line.to_bus.isin(with_geo)]
        av["trafos_hvbw_lvbwo"] = net.trafo.index[net.trafo.hv_bus.isin(with_geo)
                                                  & ~net.trafo.lv_bus.isin(with_geo)]
        av["trafos_hvbwo_lvbw"] = net.trafo.index[~net.trafo.hv_bus.isin(with_geo)
                                                  & net.trafo.lv_bus.isin(with_geo)]
        av["n_lines_one_side_geo"] = len(av["lines_fbw_tbwo"]) + len(av["lines_fbwo_tbw"])
        return av

    geo_avail = availability(net)
    while geo_avail["n_lines_one_side_geo"]:
        for et, bus_w_geo, bus_wo_geo, idx_key in zip(
                ["line", "line", "trafo", "trafo"],
                ["to_bus", "from_bus", "lv_bus", "hv_bus"],
                ["from_bus", "to_bus", "hv_bus", "lv_bus"],
                ["lines_fbwo_tbw", "lines_fbw_tbwo", "trafos_hvbwo_lvbw", "trafos_hvbw_lvbwo"]):
            net.bus.loc[net[et].loc[geo_avail[idx_key], bus_wo_geo].values, "geo"] = \
                net.bus.loc[net[et].loc[geo_avail[idx_key], bus_w_geo].values, "geo"].values
        geo_avail = availability(net)
    set_line_geodata_from_bus_geodata(net)
