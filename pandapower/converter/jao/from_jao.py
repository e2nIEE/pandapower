# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""
Convert the JAO (Joint Allocation Office) Core static grid model into a pandapower network.

The heavy lifting is split across sibling modules:

  * :mod:`._schema` resolves the drifting two-level Excel column headers,
  * :mod:`._correction` cleans the sheets and unifies the many spellings of each location,
  * :mod:`._elements` builds buses, lines and transformers,
  * :mod:`._geodata` extracts geodata from the HTML map.

This module wires those steps together and provides the grid-group utilities used afterwards.
"""

import os
import json
import logging
from functools import reduce

import numpy as np
import pandas as pd

from pandapower.io_utils import pandapowerNet
from pandapower.create import create_empty_network
from pandapower.topology import create_nxgraph, connected_components
from pandapower.plotting import set_line_geodata_from_bus_geodata
from pandapower.toolbox import drop_buses, fuse_buses

from pandapower.converter.jao._correction import data_correction, report_problematic_names
from pandapower.converter.jao._elements import (
    create_buses_from_line_data, create_lines, create_transformers_and_buses, get_bus_idx)
from pandapower.converter.jao._geodata import parse_html_str, add_bus_geo

logger = logging.getLogger(__name__)


def from_jao(excel_file_path: str,
             html_file_path: str | None,
             extend_data_for_grid_group_connections: bool,
             drop_grid_groups_islands: bool = False,
             apply_data_correction: bool = True,
             max_i_ka_fillna: float | int = 999,
             strict: bool = True,
             **kwargs) -> pandapowerNet:
    """
    Converts European (Core) EHV grid data provided by JAO (Joint Allocation Office), the
    "Single Allocation Platform (SAP) for all European Transmission System Operators (TSOs) that
    operate in accordance to EU legislation".

    **Data Sources and Availability:**
    The data are available at the website
    `JAO Static Grid Model <https://www.jao.eu/static-grid-model>`_. There, a map is provided to
    get a fine overview of the geographical extent and the scope of the data. These include
    information about European (Core) lines, tielines, and transformers.

    **Limitations:**
    No information is available on load or generation. The data quality with regard to the
    interconnection of the equipment, the information provided and the (incomplete) geodata should
    be considered with caution. The published data is maintained largely by hand, so the converter
    applies robust fuzzy matching to cope with inconsistent column headers and location names.

    **Features of the converter:**

    - **Data Correction:** corrects known data inconsistencies, such as inconsistent spellings and
      missing necessary information.
    - **Geographical Data Parsing:** Parses geodata from the HTML file to add geolocation
      information to buses and lines.
    - **Grid Group Connections:** Optionally extends the network by connecting islanded grid
      groups to avoid disconnected components.
    - **Data Customization:** Allows for customization through additional parameters to control
      transformer creation, grid group dropping, and voltage level deviations.

    Parameters:
        excel_file_path: input data including electrical parameters of the grid's utilities, stored
            in multiple sheets of an Excel file (typically "Lines", "Tielines", "Transformers").
        html_file_path: input data for geo information provided by an HTML file. Pass None to run
            the converter without geo information.
        extend_data_for_grid_group_connections: if True, connections (additional transformers and
            merged buses) are created to avoid islanded grid groups.
        drop_grid_groups_islands: if True, islanded grid groups are dropped if their number of
            buses is below `min_bus_number` (default 6). (default: False)
        apply_data_correction: if True, apply the data-correction routines. (default: True)
        max_i_ka_fillna: value to fill missing or invalid max_i_ka of lines and transformers.
            Pass np.nan to leave such values unset. (default: 999)
        strict: how to handle transformers whose location cannot be matched to any bus (which
            happens for hand-maintained data with missing buses or inconsistent naming). If True,
            a ValueError is raised listing every affected transformer. If False, those
            transformers are skipped (with a warning listing them) and the conversion still
            completes; the resulting network is then incomplete and should be used with care.
            (default: True)

    Keyword Arguments:
        minimal_trafo_invention (bool): applies if extend_data_for_grid_group_connections is True.
            If True, adding transformers stops once no grid group is islanded anymore. (default:
            False)
        min_bus_number (int|str): threshold to decide which small grid groups are dropped. Use
            "max" to keep only the largest group, or "unsupplied" to drop groups without a slack
            element. (default: 6)
        rel_deviation_threshold_for_trafo_bus_creation (float): if a transformer location's voltage
            deviates from the transformer data by more than this fraction, an additional bus is
            created. (default: 0.2)
        log_rel_vn_deviation (float): range below rel_deviation_threshold_for_trafo_bus_creation in
            which a warning is logged instead of creating additional buses. (default: 0.12)
        sn_mva (float): system base apparent power (MVA) for the pandapower net.

    Returns:
        pandapowerNet: the network created from the JAO data.

    Example:
        >>> from pathlib import Path
        >>> import os
        >>> from pandapower.converter.jao import from_jao
        >>> home = str(Path.home())
        >>> excel_file_path = os.path.join(home, "desktop", "202409_Core Static Grid Model.xlsx")
        >>> html_file_path = os.path.join(home, "desktop", "2024-09-13_Core_SGM_publication.html")
        >>> net = from_jao(excel_file_path, html_file_path, True, drop_grid_groups_islands=True)
    """
    # --- read data
    data = pd.read_excel(excel_file_path, sheet_name=None, header=[0, 1])
    report_problematic_names(data)
    if html_file_path is not None:
        with open(html_file_path, mode="r", encoding=kwargs.get("encoding", "utf-8")) as f:
            html_str = f.read()
    else:
        html_str = ""

    # --- correct data
    if apply_data_correction:
        html_str = data_correction(data, html_str, max_i_ka_fillna)

    # --- parse geodata from html
    line_geo_data = None
    if html_str:
        try:
            line_geo_data = parse_html_str(html_str)
        except (json.JSONDecodeError, KeyError, AssertionError) as e:
            logger.error(f"html data were ignored due to this error:\n{e}")

    # --- create the pandapower net
    net = create_empty_network(
        name=os.path.splitext(os.path.basename(excel_file_path))[0],
        **{key: val for key, val in kwargs.items() if key == "sn_mva"})
    create_buses_from_line_data(net, data)
    create_lines(net, data, max_i_ka_fillna)
    create_transformers_and_buses(net, data, strict=strict, **kwargs)

    # --- invent connections between grid groups
    if extend_data_for_grid_group_connections:
        _invent_connections_between_grid_groups(net, **kwargs)

    # --- drop islanded grid groups
    if drop_grid_groups_islands:
        drop_islanded_grid_groups(net, kwargs.get("min_bus_number", 6))

    # --- add geodata to buses and lines
    if line_geo_data is not None:
        add_bus_geo(net, line_geo_data)
        set_line_geodata_from_bus_geodata(net)

    return net


# ==================================================================================================
# Grid groups
# ==================================================================================================

def get_grid_groups(net: pandapowerNet, **kwargs) -> pd.DataFrame:
    """Return the connected components (grid groups) of the network and their bus counts."""
    notravbuses = {"notravbuses": kwargs.pop("notravbuses")} if "notravbuses" in kwargs else {}
    grid_group_buses = list(connected_components(create_nxgraph(net, **kwargs), **notravbuses))
    grid_groups = pd.DataFrame({"buses": grid_group_buses})
    grid_groups["n_buses"] = grid_groups["buses"].apply(len)
    return grid_groups


def drop_islanded_grid_groups(net: pandapowerNet, min_bus_number: int | str, **kwargs) -> None:
    """Drop islanded grid groups by size or supply condition.

    ``min_bus_number`` may be an int (drop groups smaller than this), ``"max"`` (keep only the
    largest group) or ``"unsupplied"`` (drop groups without any slack element).
    """
    def to_drop_by_size():
        return grid_groups.loc[grid_groups["n_buses"] < min_bus_number]

    grid_groups = get_grid_groups(net, **kwargs)

    if min_bus_number == "unsupplied":
        slack_buses = set(net.ext_grid.loc[net.ext_grid.in_service, "bus"]) | \
            set(net.gen.loc[net.gen.in_service & net.gen.slack, "bus"])
        grid_groups_to_drop = grid_groups.loc[
            ~grid_groups.buses.apply(lambda x: not x.isdisjoint(slack_buses))]
    elif min_bus_number == "max":
        min_bus_number = grid_groups["n_buses"].max()
        grid_groups_to_drop = to_drop_by_size()
    elif isinstance(min_bus_number, int):
        grid_groups_to_drop = to_drop_by_size()
    else:
        raise NotImplementedError(
            f"{min_bus_number=} is not implemented. Use an int, 'max', or 'unsupplied' instead.")

    buses_to_drop = reduce(set.union, grid_groups_to_drop.buses)
    drop_buses(net, buses_to_drop)
    logger.info(f"drop_islanded_grid_groups() drops {len(grid_groups_to_drop)} grid groups with a "
                f"total of {grid_groups_to_drop.n_buses.sum()} buses.")


def _invent_connections_between_grid_groups(
        net: pandapowerNet, minimal_trafo_invention: bool = False, **kwargs) -> None:
    """
    Add connections between islanded grid groups by:

    - adding transformers between equally named buses of different voltage level in different
      groups,
    - merging buses of the same voltage level, equal name base and different grid groups,
    - fusing selected known close-by bus pairs.

    :param pandapowerNet net: net to be manipulated
    :param bool minimal_trafo_invention: if True, adding transformers stops once no grid group is
        islanded anymore.
    """
    grid_groups = get_grid_groups(net)
    bus_idx = get_bus_idx(net)
    bus_grid_groups = pd.concat([pd.Series(group, index=buses) for group, buses in zip(
        grid_groups.index, grid_groups.buses)]).sort_index()

    # treat for example "Wuergau" equally as "Wuergau (2)":
    location_names = pd.Series(bus_idx.index.get_level_values(0))
    location_names = location_names.str.replace(r"(.) \([0-9]+\)", r"\1", regex=True)
    bus_idx.index = pd.MultiIndex.from_arrays(
        [location_names.values, bus_idx.index.get_level_values(1).to_numpy()],
        names=bus_idx.index.names)

    # --- 1) add transformers between equally named buses of different voltage level in different
    #        groups
    connected_vn_kvs_by_trafos = pd.DataFrame({
        "hv": net.bus.vn_kv.loc[net.trafo.hv_bus.values].values,
        "lv": net.bus.vn_kv.loc[net.trafo.lv_bus.values].values,
        "index": net.trafo.index}).set_index(["hv", "lv"]).sort_index()
    dupl_location_names = location_names[location_names.duplicated()]

    for location_name in dupl_location_names:
        if minimal_trafo_invention and len(bus_grid_groups.unique()) <= 1:
            break
        grid_groups_at_location = bus_grid_groups.loc[bus_idx.loc[location_name].values]
        grid_groups_at_location = grid_groups_at_location.drop_duplicates()
        if len(grid_groups_at_location) < 2:
            continue
        elif len(grid_groups_at_location) > 2:
            raise NotImplementedError("Code is not provided to invent Transformer connections "
                                      "between locations with more than two grid groups, i.e. "
                                      "voltage levels.")
        TSO = net.bus.zone.at[grid_groups_at_location.index[0]]
        vn_kvs = net.bus.vn_kv.loc[grid_groups_at_location.index].sort_values(ascending=False)
        try:
            trafos_same_vn = connected_vn_kvs_by_trafos.loc[tuple(vn_kvs)]
        except KeyError:
            logger.info(f"For location {location_name}, no transformer data can be reused since "
                        f"no transformer connects {vn_kvs.iat[0]} kV and {vn_kvs.iat[1]} kV.")
            continue
        trafos_same_tso = trafos_same_vn.loc[(net.bus.zone.loc[
            net.trafo.hv_bus.loc[trafos_same_vn.values.flatten()].values]
            == TSO).values].values.flatten()

        # choose the transformer to copy parameters from
        tr_to_copy = trafos_same_tso[0] if len(trafos_same_tso) else \
            trafos_same_vn.values.flatten()[0]

        duplicated_row = net.trafo.loc[[tr_to_copy]].copy()
        duplicated_row.index = [net.trafo.index.max() + 1]
        duplicated_row.hv_bus = vn_kvs.index[0]
        duplicated_row.lv_bus = vn_kvs.index[1]
        duplicated_row.name = "additional transformer to connect the grid"
        net.trafo = pd.concat([net.trafo, duplicated_row])

        bus_grid_groups.loc[bus_grid_groups == grid_groups_at_location.iat[1]] = \
            grid_groups_at_location.iat[0]

    # --- 2) merge buses of same voltage level, different grid groups and equal name base
    bus_name_splits = net.bus.name.str.split(r"[ -/]+", expand=True)
    buses_with_single_base = net.bus.name.loc[(~bus_name_splits.isnull()).sum(axis=1) == 1]
    for idx, name_base in buses_with_single_base.items():
        # a previous fuse in this loop may already have removed this bus; buses invented after
        # bus_grid_groups was built (e.g. duplicated same-bus trafo buses) are not tracked in it
        if idx not in net.bus.index or idx not in bus_grid_groups.index:
            continue
        same_name_base = net.bus.drop(idx).name.str.contains(name_base)
        if not any(same_name_base):
            continue
        other_group = bus_grid_groups.drop(idx) != bus_grid_groups.at[idx]
        same_vn = net.bus.drop(idx).vn_kv == net.bus.vn_kv.at[idx]
        is_fuse_candidate = same_name_base & other_group & same_vn
        if not any(is_fuse_candidate):
            continue
        to_fuse = bus_grid_groups.drop(idx).loc[is_fuse_candidate].drop_duplicates()
        fuse_buses(net, idx, set(to_fuse.index))

        bus_grid_groups.loc[bus_grid_groups.isin(bus_grid_groups.drop(idx).loc[
            is_fuse_candidate].unique())] = grid_groups_at_location.iat[0]
        bus_grid_groups = bus_grid_groups.drop(to_fuse.index)

    # --- 3) fuse buses that are close to each other (known cases)
    for name1, name2 in [("CROISIERE", "BOLLENE (POSTE RESEAU)"),
                         ("CAEN", "DRONNIERE (LA)"),
                         ("TRINITE-VICTOR", "MENTON/TRINITE VICTOR")]:
        b1 = net.bus.index[net.bus.name == name1]
        b2 = net.bus.index[net.bus.name == name2]
        if len(b1) == 1 and len(b2) >= 1:
            fuse_buses(net, b1[0], set(b2))
            bus_grid_groups = bus_grid_groups.drop(b2)
        else:
            logger.info("Buses of the following names were intended to be fused but were not found."
                        f"\n'{name1}' and '{name2}'")
