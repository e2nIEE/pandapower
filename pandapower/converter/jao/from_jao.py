# -*- coding: utf-8 -*-nt

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from copy import deepcopy
import os
import json
from functools import reduce
from typing import Optional, Union
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_object_dtype
from pandapower.io_utils import pandapowerNet
from pandapower.create import create_empty_network, create_buses, create_lines_from_parameters, \
    create_transformers_from_parameters
from pandapower.topology import create_nxgraph, connected_components
from pandapower.plotting import set_line_geodata_from_bus_geodata
from pandapower.toolbox import drop_buses, fuse_buses
import re
import unicodedata
import difflib
import logging

logger = logging.getLogger(__name__)


def from_jao(excel_file_path: str,
             html_file_path: Optional[str],
             extend_data_for_grid_group_connections: bool,
             drop_grid_groups_islands: bool = False,
             apply_data_correction: bool = True,
             max_i_ka_fillna: Union[float, int] = 999,
             **kwargs) -> pandapowerNet:
    """
    Converts European (Core) EHV grid data provided by JAO (Joint Allocation Office), the
    "Single Allocation Platform (SAP) for all European Transmission System Operators (TSOs) that
    operate in accordance to EU legislation".

    **Data Sources and Availability:**
    The data are available at the website
    `JAO Static Grid Model <https://www.jao.eu/static-grid-model>`_ (November 2024).
    There, a map is provided to get an fine overview of the geographical extent and the scope of
    the data. These inlcude information about European (Core) lines, tielines, and transformers.

    **Limitations:**
    No information is available on load or generation.
    The data quality with regard to the interconnection of the equipment, the information provided
    and the (incomplete) geodata should be considered with caution.

    **Features of the converter:**
    - **Data Correction:** corrects known data inconsistencies, such as inconsistent spellings and missing necessary information.
    - **Geographical Data Parsing:** Parses geographical data from the HTML file to add geolocation information to buses and lines.
    - **Grid Group Connections:** Optionally extends the network by connecting islanded grid groups to avoid disconnected components.
    - **Data Customization:** Allows for customization through additional parameters to control transformer creation, grid group dropping, and voltage level deviations.

    :param str excel_file_path:
        input data including electrical parameters of grids' utilities, stored in multiple sheets
        of an excel file

    :param str html_file_path:
        input data for geo information. If The converter should be run without geo information, None
        can be passed., provided by an html file

    :param bool extend_data_for_grid_group_connections:
        if True, connections (additional transformers and merging buses) are created to avoid
        islanded grid groups, by default False

    :param Optional[bool] drop_grid_groups_islands:
        if True, islanded grid groups will be dropped if their number of buses is below
        `min_bus_number` default for this is 6 (default: False)

    :param Optional[bool] apply_data_correction:
        _description_ (default: True)

    :param Optional[float|int] max_i_ka_fillna:
        value to fill missing values or data of false type in max_i_ka of lines and transformers.
        If no value should be set, you can also pass np.nan. (default: 999)

    :param '**'kwargs: following params are available

    :param Optional[bool] minimal_trafo_invention:
        applies if extend_data_for_grid_group_connections is True. Then, if minimal_trafo_invention
        is True, adding transformers stops when no grid groups is islanded anymore (does not apply
        for release version 5 or 6, i.e. it does not care what value is passed to
        minimal_trafo_invention). If False, all equally named buses that have different voltage
        level and lay in different groups will be connected via additional transformers (default: False)

    :param Optional[int|str] min_bus_number:
        Threshold value to decide which small grid groups should be dropped and which large grid
        groups should be kept. If all islanded grid groups should be dropped except of the one
        largest, set "max". If all grid groups that do not contain a slack element should be
        dropped, set "unsupplied". (default: 6)

    :param Optional[float] rel_deviation_threshold_for_trafo_bus_creation:
        If the voltage level of transformer locations is far different than the transformer data,
        additional buses are created. rel_deviation_threshold_for_trafo_bus_creation defines the
        tolerance in which no additional buses are created. (default: 0.2)

    :param Optional[float] log_rel_vn_deviation:
        This parameter allows a range below rel_deviation_threshold_for_trafo_bus_creation in which
        a warning is logged instead of a creating additional buses. (default: 0.12)

    :return: net created from the jao data
    :rtype: pandapowerNet

    :example:
        >>> from pathlib import Path
        >>> import os
        >>> import pandapower as pp
        >>> net = pp.converter.from_jao()
        >>> home = str(Path.home())
        >>> # assume that the files are located at your desktop:
        >>> excel_file_path = os.path.join(home, "desktop", "202409_Core Static Grid Mode_6th release")
        >>> html_file_path = os.path.join(home, "desktop", "2024-09-13_Core_SGM_publication.html")
        >>> net = from_jao(excel_file_path, html_file_path, True, drop_grid_groups_islands=True)
    """

    # --- read data
    data = pd.read_excel(excel_file_path, sheet_name=None, header=[0, 1])
    if html_file_path is not None:
        with open(html_file_path, mode='r', encoding=kwargs.get("encoding", "utf-8")) as f:
            html_str = f.read()
    else:
        html_str = ""

    # --- manipulate data / data corrections
    if apply_data_correction:
        html_str = _data_correction(data, html_str, max_i_ka_fillna)

    # --- parse html_str to line_geo_data
    line_geo_data = None
    if html_str:
        try:
            line_geo_data = _parse_html_str(html_str)
        except (json.JSONDecodeError, KeyError, AssertionError) as e:
            logger.error(f"html data were ignored due to this error:\n{e}")

    # --- create the pandapower net
    net = create_empty_network(name=os.path.splitext(os.path.basename(excel_file_path))[0],
                               **{key: val for key, val in kwargs.items() if key == "sn_mva"})
    _create_buses_from_line_data(net, data)
    _create_lines(net, data, max_i_ka_fillna)
    _create_transformers_and_buses(net, data, **kwargs)

    # --- invent connections between grid groups
    if extend_data_for_grid_group_connections:
        _invent_connections_between_grid_groups(net, **kwargs)

    # --- drop islanded grid groups
    if drop_grid_groups_islands:
        drop_islanded_grid_groups(net, kwargs.get("min_bus_number", 6))

    # --- add geo data to buses and lines
    if line_geo_data is not None:
        _add_bus_geo(net, line_geo_data)
        set_line_geodata_from_bus_geodata(net)

    return net

# --- secondary functions --------------------------------------------------------------------------


# geänderte Funktion: _data_correction
def _data_correction(
        data: dict[str, pd.DataFrame],
        html_str: Optional[str],
        max_i_ka_fillna: Union[float, int]) -> Optional[str]:
    """
    Korrigiert numerische Eingabeschwächen (Komma-Floats, fehlende Imax),
    trimmt Whitespaces etc. KEINE hartkodierten Location-Renames mehr;
    Namens-Matching erfolgt später dynamisch in _find_trafo_locations().
    """

    # --- Line und Tieline ---
    for key in ["Lines", "Tielines"]:
        # Spaltenköpfe justieren
        cols = data[key].columns.to_frame().reset_index(drop=True)
        cols.loc[cols[1] == "Voltage_level(kV)", 0] = None
        cols.loc[cols[1] == "Comment", 0] = None
        cols.loc[cols[0].str.startswith("Unnamed:").astype(bool), 0] = None
        cols.loc[cols[1] == "Length_(km)", 0] = "Electrical Parameters"
        data[key].columns = pd.MultiIndex.from_arrays(cols.values.T)

        # Komma zu Punkt und auf float casten
        data[key][("Maximum Current Imax (A)", "Fixed")] = \
            data[key][("Maximum Current Imax (A)", "Fixed")].replace(
                "\xa0", max_i_ka_fillna*1e3).replace(
                "-", max_i_ka_fillna*1e3).replace(" ", max_i_ka_fillna*1e3)
        col_names = [("Electrical Parameters", col) for col in
                     ["Length_(km)", "Resistance_R(Ω)", "Reactance_X(Ω)", "Susceptance_B(μS)", "Length_(km)"]] \
                    + [("Maximum Current Imax (A)", "Fixed")]
        _float_col_comma_correction(data, key, col_names)

        # Name-Whitespaces trimmen (keine Renames)
        lvl1 = data[key].columns.get_level_values(1)
        ne_cols = data[key].columns[lvl1 == "NE_name"]  # or .isin(["NE_name","NE_names"]) if needed
        if len(ne_cols):
            data[key].loc[:, ne_cols] = data[key].loc[:, ne_cols].apply(
                lambda s: s.astype(str).str.strip()
            )

        # keep explicit handling for Full_name
        for loc_name in [("Substation_1", "Full_name"), ("Substation_2", "Full_name")]:
            data[key].loc[:, loc_name] = data[key].loc[:, loc_name].astype(str).str.strip()

    # --- Transformer ---
    key = "Transformers"
    # Location/Name trimmen
    data[key].loc[:, ("Location", "Full Name")] = data[key].loc[:, ("Location", "Full Name")].astype(str).str.strip()

    # Taps reparieren
    taps = data[key].loc[:, ("Phase Shifting Properties", "Taps used for RAO")].fillna("").astype(str).str.replace(" ", "")
    nonnull = taps.apply(len).astype(bool)
    nonnull_taps = taps.loc[nonnull]
    surrounded = nonnull_taps.str.startswith("<") & nonnull_taps.str.endswith(">")
    nonnull_taps.loc[surrounded] = nonnull_taps.loc[surrounded].str[1:-1]
    slash_sep = (~nonnull_taps.str.contains(";")) & nonnull_taps.str.contains("/")
    nonnull_taps.loc[slash_sep] = nonnull_taps.loc[slash_sep].str.replace("/", ";")
    nonnull_taps.loc[nonnull_taps == "0"] = "0;0"
    data[key].loc[nonnull, ("Phase Shifting Properties", "Taps used for RAO")] = nonnull_taps
    data[key].loc[~nonnull, ("Phase Shifting Properties", "Taps used for RAO")] = "0;0"

    # Doppelinfos bei PST
    cols = ["Phase Regulation δu (%)", "Angle Regulation δu (%)"]
    for col in cols:
        if is_object_dtype(data[key].loc[:, ("Phase Shifting Properties", col)]):
            tr_double = data[key].index[data[key].loc[:, ("Phase Shifting Properties", col)].str.contains("/").fillna(0).astype(bool)]
            data[key].loc[tr_double, ("Phase Shifting Properties", col)] = data[key].loc[
                tr_double, ("Phase Shifting Properties", col)].str.split("/", expand=True)[1].str.replace(",", ".").astype(float).values

    return html_str


def _parse_html_str(html_str: str) -> pd.DataFrame:
    """
    Converts ths geodata from the html file (information hidden in the string), from Lines in
    particular, to a DataFrame that can be used later in _add_bus_geo()

    :param str html_str: html file that includes geodata information

    :return: extracted geodata for a later and easy use
    :rtype: pd.DataFrame
    """
    def _filter_name(st: str) -> str:
        name_start = "<b>NE name: "
        name_end = "</b>"
        pos0 = st.find(name_start) + len(name_start)
        pos1 = st.find(name_end, pos0)
        assert pos0 >= 0
        assert pos1 >= len(name_start)
        return st[pos0:pos1]

    json_start_str = '<script type="application/json" data-for="htmlwidget-216030e6806f328c00fb">'
    json_start_pos = html_str.find(json_start_str) + len(json_start_str)
    json_end_pos = html_str[json_start_pos:].find('</script>')
    json_str = html_str[json_start_pos:(json_start_pos+json_end_pos)]
    geo_data = json.loads(json_str)
    geo_data = geo_data["x"]["calls"]
    methods_pos = pd.Series({item["method"]: i for i, item in enumerate(geo_data)})
    polylines = geo_data[methods_pos.at["addPolylines"]]["args"]
    EIC_start = "EIC Code:<b> "
    if len(polylines[6]) != len(polylines[0]):
        raise AssertionError("The lists of EIC Code data and geo data are not of the same length.")
    line_EIC = [polylines[6][i][polylines[6][i].find(EIC_start)+len(EIC_start):] for i in range(
        len(polylines[6]))]
    line_name = [_filter_name(polylines[6][i]) for i in range(len(polylines[6]))]
    line_geo_data = pd.concat([_lng_lat_to_df(polylines[0][i][0][0], line_EIC[i], line_name[i]) for
                               i in range(len(polylines[0]))], ignore_index=True)

    # remove trailing whitespaces
    for col in ["EIC_Code", "name"]:
        line_geo_data[col] = line_geo_data[col].str.strip()

    return line_geo_data


def _create_buses_from_line_data(net: pandapowerNet, data: dict[str, pd.DataFrame]) -> None:
    """Creates buses to the pandapower net using information from the lines and tielines sheets
    (excel file).

    :param pandapowerNet net: net to be filled by buses
    :param dict[str, pd.DataFrame data: data provided by the excel file which will be corrected
    """
    bus_df_empty = pd.DataFrame({"name": str(), "vn_kv": float(), "TSO": str()}, index=[])
    bus_df = deepcopy(bus_df_empty)
    for key in ["Lines", "Tielines"]:
        for subst in ['Substation_1', 'Substation_2']:
            data_col_tuples = [(subst, "Full_name"), (None, "Voltage_level(kV)"), (None, "TSO")]
            to_add = data[key].loc[:, data_col_tuples].set_axis(bus_df.columns, axis="columns")
            if len(bus_df):
                bus_df = pd.concat([bus_df, to_add])
            else:
                bus_df = to_add
    bus_df = _drop_duplicates_and_join_TSO(bus_df)
    new_bus_idx = create_buses(
        net, len(bus_df), vn_kv=bus_df.vn_kv, name=bus_df.name, zone=bus_df.TSO)
    assert np.allclose(new_bus_idx, bus_df.index)


def _create_lines(
        net: pandapowerNet,
        data: dict[str, pd.DataFrame],
        max_i_ka_fillna: Union[float, int]) -> None:
    """
    Creates lines to the pandapower net using information from the lines and tielines sheets (excel file).

    :param pandapowerNet net: net to be filled by buses
    :param dict[str, pd.DataFrame] data: data provided by the excel file which will be corrected
    :param float|int max_i_ka_fillna: value to fill missing values or data of false type in max_i_ka of lines and transformers.
        If no value should be set, you can also pass np.nan.
    """

    bus_idx = _get_bus_idx(net)

    for key in ["Lines", "Tielines"]:
        length_km = data[key][("Electrical Parameters", "Length_(km)")].values
        zero_length = np.isclose(length_km, 0)
        no_length = np.isnan(length_km)
        if sum(zero_length) or sum(no_length):
            logger.warning(f"According to given data, {sum(zero_length)} {key.lower()} have zero "
                           f"length and {sum(zero_length)} {key.lower()} have no length data. "
                           "Both types of wrong data are replaced by 1 km.")
            length_km[zero_length | no_length] = 1
        vn_kvs = data[key].loc[:, (None, "Voltage_level(kV)")].values

        _ = create_lines_from_parameters(
            net,
            bus_idx.loc[list(tuple(zip(data[key].loc[:, ("Substation_1", "Full_name")].values,
                        vn_kvs)))].values,
            bus_idx.loc[list(tuple(zip(data[key].loc[:, ("Substation_2", "Full_name")].values,
                        vn_kvs)))].values,
            length_km,
            data[key][("Electrical Parameters", "Resistance_R(Ω)")].values / length_km,
            data[key][("Electrical Parameters", "Reactance_X(Ω)")].values / length_km,
            data[key][("Electrical Parameters", "Susceptance_B(μS)")].values / length_km,
            data[key][("Maximum Current Imax (A)", "Fixed")].fillna(
                max_i_ka_fillna*1e3).values / 1e3,
            name=data[key].xs("NE_name", level=1, axis=1).values[:, 0],
            EIC_Code=data[key].xs("EIC_Code", level=1, axis=1).values[:, 0],
            TSO=data[key].xs("TSO", level=1, axis=1).values[:, 0],
            Comment=data[key].xs("Comment", level=1, axis=1).values[:, 0],
            Tieline=key == "Tielines",
        )


def _create_transformers_and_buses(
        net: pandapowerNet, data: dict[str, pd.DataFrame], **kwargs) -> None:
    """
    Creates transformers to the pandapower net using information from the transformers sheet (excel file).

    :param pandapowerNet net: net to be filled by buses
    :param dict[str, pd.DataFrame] data: data provided by the excel file which will be corrected
    """

    # --- data preparations
    key = "Transformers"
    bus_idx = _get_bus_idx(net)
    vn_hv_kv, vn_lv_kv = _get_transformer_voltages(data, bus_idx)
    trafo_connections = _allocate_trafos_to_buses_and_create_buses(
        net, data, bus_idx, vn_hv_kv, vn_lv_kv, **kwargs)
    max_i_a = data[key].loc[:, ("Maximum Current Imax (A) primary", "Fixed")]
    empty_i_idx = max_i_a.index[max_i_a.isnull()]
    max_i_a.loc[empty_i_idx] = data[key].loc[empty_i_idx, (
        "Maximum Current Imax (A) primary", "Max")].values
    sn_mva = np.sqrt(3) * max_i_a * vn_hv_kv / 1e3
    z_pu = vn_lv_kv**2 / sn_mva
    rk = data[key].xs("Resistance_R(Ω)", level=1, axis=1).values[:, 0] / z_pu
    xk = data[key].xs("Reactance_X(Ω)", level=1, axis=1).values[:, 0] / z_pu
    b0 = data[key].xs("Susceptance_B (µS)", level=1, axis=1).values[:, 0] * 1e-6 * z_pu
    g0 = data[key].xs("Conductance_G (µS)", level=1, axis=1).values[:, 0] * 1e-6 * z_pu
    zk = np.sqrt(rk**2 + xk**2)
    vk_percent = np.sign(xk) * zk * 100
    vkr_percent = rk * 100
    pfe_kw = g0 * sn_mva * 1e3
    i0_percent = 100 * np.sqrt(b0**2 + g0**2) * net.sn_mva / sn_mva
    taps = data[key].loc[:, ("Phase Shifting Properties", "Taps used for RAO")].str.split(
        ";", expand=True).astype(int).set_axis(["tap_min", "tap_max"], axis=1)

    du = _get_float_column(data[key], ("Phase Shifting Properties", "Phase Regulation δu (%)"))
    dphi = _get_float_column(data[key], ("Phase Shifting Properties", "Angle Regulation δu (%)"))
    phase_shifter = np.isclose(du, 0) & (~np.isclose(dphi, 0))  # Symmetrical/Asymmetrical not
    # considered

    _ = create_transformers_from_parameters(
        net,
        trafo_connections.hv_bus.values,
        trafo_connections.lv_bus.values,
        sn_mva,
        vn_hv_kv,
        vn_lv_kv,
        vkr_percent,
        vk_percent,
        pfe_kw,
        i0_percent,
        shift_degree=data[key].xs("Theta θ (°)", level=1, axis=1).values[:, 0],
        tap_pos=0,
        tap_neutral=0,
        tap_side="lv",
        tap_min=taps["tap_min"].values,
        tap_max=taps["tap_max"].values,
        tap_phase_shifter=phase_shifter,
        tap_step_percent=du,
        tap_step_degree=dphi,
        name=data[key].loc[:, ("Location", "Full Name")].str.strip().values,
        EIC_Code=data[key].xs("EIC_Code", level=1, axis=1).values[:, 0],
        TSO=data[key].xs("TSO", level=1, axis=1).values[:, 0],
        Comment=data[key].xs("Comment", level=1, axis=1).replace("\xa0", "").values[:, 0],
    )


def _invent_connections_between_grid_groups(
        net: pandapowerNet, minimal_trafo_invention: bool = False, **kwargs) -> None:
    """
    Adds connections between islanded grid groups via:

    - adding transformers between equally named buses that have different voltage level and lay in different groups
    - merge buses of same voltage level, different grid groups and equal name base
    - fuse buses that are close to each other

    :param pandapowerNet net: net to be manipulated
    :param Optional[bool] minimal_trafo_invention: if True, adding transformers stops when no grid groups is islanded anymore (does not apply
        for release version 5 or 6, i.e. it does not care what value is passed to
        minimal_trafo_invention). If False, all equally named buses that have different voltage
        level and lay in different groups will be connected via additional transformers,
        (default: False)
    """
    grid_groups = get_grid_groups(net)
    bus_idx = _get_bus_idx(net)
    bus_grid_groups = pd.concat([pd.Series(group, index=buses) for group, buses in zip(
        grid_groups.index, grid_groups.buses)]).sort_index()

    # treat for example "Wuergau" equally as "Wuergau (2)":
    location_names = pd.Series(bus_idx.index.get_level_values(0))
    location_names = location_names.str.replace(r"(.) \([0-9]+\)", r"\1", regex=True)
    bus_idx.index = pd.MultiIndex.from_arrays(
        [location_names.values, bus_idx.index.get_level_values(1).to_numpy()],
        names=bus_idx.index.names)

    # --- add Transformers between equally named buses that have different voltage level and lay in
    # --- different groups
    connected_vn_kvs_by_trafos = pd.DataFrame({
        "hv": net.bus.vn_kv.loc[net.trafo.hv_bus.values].values,
        "lv": net.bus.vn_kv.loc[net.trafo.lv_bus.values].values,
        "index": net.trafo.index}).set_index(["hv", "lv"]).sort_index()
    dupl_location_names = location_names[location_names.duplicated()]

    for location_name in dupl_location_names:
        if minimal_trafo_invention and len(bus_grid_groups.unique()) <= 1:
            break  # break with regard to minimal_trafo_invention
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
            trafos_connecting_same_voltage_levels = \
                connected_vn_kvs_by_trafos.loc[tuple(vn_kvs)]
        except KeyError:
            logger.info(f"For location {location_name}, no transformer data can be reused since "
                        f"no transformer connects {vn_kvs.sort_values(ascending=False).iat[0]} kV "
                        f"and {vn_kvs.sort_values(ascending=False).iat[1]} kV.")
            continue
        trafos_of_same_TSO = trafos_connecting_same_voltage_levels.loc[(net.bus.zone.loc[
            net.trafo.hv_bus.loc[trafos_connecting_same_voltage_levels.values.flatten(
            )].values] == TSO).values].values.flatten()

        # from which trafo parameters are copied:
        tr_to_be_copied = trafos_of_same_TSO[0] if len(trafos_of_same_TSO) else \
            trafos_connecting_same_voltage_levels.values.flatten()[0]

        # copy transformer data
        duplicated_row = net.trafo.loc[[tr_to_be_copied]].copy()
        duplicated_row.index = [net.trafo.index.max() + 1]  # adjust index
        duplicated_row.hv_bus = vn_kvs.index[0]  # adjust hv_bus, lv_bus
        duplicated_row.lv_bus = vn_kvs.index[1]  # adjust hv_bus, lv_bus
        duplicated_row.name = "additional transformer to connect the grid"
        net.trafo = pd.concat([net.trafo, duplicated_row])

        bus_grid_groups.loc[bus_grid_groups == grid_groups_at_location.iat[1]] = \
            grid_groups_at_location.iat[0]

    # --- merge buses of same voltage level, different grid groups and equal name base
    bus_name_splits = net.bus.name.str.split(r"[ -/]+", expand=True)
    buses_with_single_base = net.bus.name.loc[(~bus_name_splits.isnull()).sum(axis=1) == 1]
    for idx, name_base in buses_with_single_base.items():
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

    # --- fuse buses that are close to each other
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


def drop_islanded_grid_groups(
        net: pandapowerNet,
        min_bus_number: Union[int, str],
        **kwargs) -> None:
    """
    Drops grid groups that are islanded and include a number of buses below min_bus_number.

    :param panadpowerNet net: net in which islanded grid groups will be dropped
    :param Optional[int|str] min_bus_number: Threshold value to decide which small grid groups should be dropped and which large grid
        groups should be kept. If all islanded grid groups should be dropped except of the one
        largest, set "max". If all grid groups that do not contain a slack element should be
        dropped, set "unsupplied".
    """
    def _grid_groups_to_drop_by_min_bus_number():
        return grid_groups.loc[grid_groups["n_buses"] < min_bus_number]

    grid_groups = get_grid_groups(net, **kwargs)

    if min_bus_number == "unsupplied":
        slack_buses = set(net.ext_grid.loc[net.ext_grid.in_service, "bus"]) | \
            set(net.gen.loc[net.gen.in_service & net.gen.slack, "bus"])
        grid_groups_to_drop = grid_groups.loc[~grid_groups.buses.apply(
            lambda x: not x.isdisjoint(slack_buses))]

    elif min_bus_number == "max":
        min_bus_number = grid_groups["n_buses"].max()
        grid_groups_to_drop = _grid_groups_to_drop_by_min_bus_number()

    elif isinstance(min_bus_number, int):
        grid_groups_to_drop = _grid_groups_to_drop_by_min_bus_number()

    else:
        raise NotImplementedError(
            f"{min_bus_number=} is not implemented. Use an int, 'max', or 'unsupplied' instead.")

    buses_to_drop = reduce(set.union, grid_groups_to_drop.buses)
    drop_buses(net, buses_to_drop)
    logger.info(f"drop_islanded_grid_groups() drops {len(grid_groups_to_drop)} grid groups with a "
                f"total of {grid_groups_to_drop.n_buses.sum()} buses.")


def _add_bus_geo(net: pandapowerNet, line_geo_data: pd.DataFrame) -> None:
    """Adds geodata to the buses. The function needs to handle cases where line_geo_data does not
    include no or multiple geodata per bus. Primarly, the geodata are allocate via EIC Code names,
    if ambigous, names are considered.

    :param pandapowerNet net: net in which geodata are added to the buses
    :param pd.DataFrame: line_geo_data: Converted geodata from the html file
    """
    iSl = pd.IndexSlice
    lgd_EIC_bus = line_geo_data.pivot_table(values="value", index=["EIC_Code", "bus"],
                                            columns="geo_dim")
    lgd_name_bus = line_geo_data.pivot_table(values="value", index=["name", "bus"],
                                             columns="geo_dim")
    lgd_EIC_bus_idx_extended = pd.MultiIndex.from_frame(lgd_EIC_bus.index.to_frame().assign(
        **dict(col_name="EIC_Code")).rename(columns=dict(EIC_Code="identifier")).loc[
        :, ["col_name", "identifier", "bus"]])
    lgd_name_bus_idx_extended = pd.MultiIndex.from_frame(lgd_name_bus.index.to_frame().assign(
        **dict(col_name="name")).rename(columns=dict(name="identifier")).loc[
        :, ["col_name", "identifier", "bus"]])
    lgd_bus = pd.concat([lgd_EIC_bus.set_axis(lgd_EIC_bus_idx_extended),
                         lgd_name_bus.set_axis(lgd_name_bus_idx_extended)])
    dupl_EICs = net.line.EIC_Code.loc[net.line.EIC_Code.duplicated()]
    dupl_names = net.line.name.loc[net.line.name.duplicated()]

    def _geo_json_str(this_bus_geo: pd.Series) -> str:
        return f'{{"coordinates": [{this_bus_geo.at["lng"]}, {this_bus_geo.at["lat"]}], "type": "Point"}}'

    def _add_bus_geo_inner(bus: int) -> Optional[str]:
        from_bus_line_excerpt = net.line.loc[net.line.from_bus ==
                                             bus, ["EIC_Code", "name", "Tieline"]]
        to_bus_line_excerpt = net.line.loc[net.line.to_bus == bus, ["EIC_Code", "name", "Tieline"]]
        line_excerpt = pd.concat([from_bus_line_excerpt, to_bus_line_excerpt])
        n_connected_line_ends = len(line_excerpt)
        if n_connected_line_ends == 0:
            logger.error(
                f"Bus {bus} (name {net.bus.at[bus, 'name']}) is not found in line_geo_data.")
            return None
        is_dupl = pd.concat([
            pd.DataFrame({"EIC": from_bus_line_excerpt.EIC_Code.isin(dupl_EICs).values,
                          "name": from_bus_line_excerpt.name.isin(dupl_names).values},
                         index=pd.MultiIndex.from_product([["from"], from_bus_line_excerpt.index],
                                                          names=["bus", "line_index"])),
            pd.DataFrame({"EIC": to_bus_line_excerpt.EIC_Code.isin(dupl_EICs).values,
                          "name": to_bus_line_excerpt.name.isin(dupl_names).values},
                         index=pd.MultiIndex.from_product([["to"], to_bus_line_excerpt.index],
                                                          names=["bus", "line_index"]))
        ])
        is_missing = pd.DataFrame({
            "EIC": ~line_excerpt.EIC_Code.isin(
                lgd_bus.loc["EIC_Code"].index.get_level_values("identifier")),
            "name": ~line_excerpt.name.isin(
                lgd_bus.loc["name"].index.get_level_values("identifier"))
        }).set_axis(is_dupl.index)
        is_tieline = pd.Series(net.line.loc[is_dupl.index.get_level_values("line_index"),
                                            "Tieline"].values, index=is_dupl.index)

        # --- construct access_vals, i.e. values to take line geo data from lgd_bus
        # --- if not duplicated, take "EIC_Code". Otherwise and if not dupl, take "name".
        # --- Otherwise ignore. Do it for both from and to bus
        access_vals = pd.DataFrame({
            "col_name": "EIC_Code",
            "identifier": line_excerpt.EIC_Code.values,
            "bus": is_dupl.index.get_level_values("bus").values
        })  # default is EIC_Code
        take_from_name = ((is_dupl.EIC | is_missing.EIC) & (
            ~is_dupl.name & ~is_missing.name)).values
        access_vals.loc[take_from_name, "col_name"] = "name"
        access_vals.loc[take_from_name, "identifier"] = line_excerpt.name.loc[take_from_name].values
        keep = (~(is_dupl | is_missing)).any(axis=1).values
        if np.all(is_missing):
            log_msg = (f"For bus {bus} (name {net.bus.at[bus, 'name']}), {n_connected_line_ends} "
                       "were found but no EIC_Codes or names of corresponding lines were found ."
                       "in the geo data from the html file.")
            if is_tieline.all():
                logger.debug(log_msg)
            else:
                logger.warning(log_msg)
            return None
        elif sum(keep) == 0:
            logger.info(f"For {bus=}, all EIC_Codes and names of connected lines are ambiguous. "
                        "No geo data is dropped at this point.")
            keep[(~is_missing).any(axis=1)] = True
        access_vals = access_vals.loc[keep]

        # --- get this_bus_geo from EIC_Code or name with regard to access_vals
        this_bus_geo = lgd_bus.loc[iSl[
            access_vals.col_name, access_vals.identifier, access_vals.bus], :]

        if len(this_bus_geo) > 1:
            # reduce similar/equal lines
            this_bus_geo = this_bus_geo.loc[this_bus_geo.round(2).drop_duplicates().index]

        # --- return geo_json_str
        len_this_bus_geo = len(this_bus_geo)
        if len_this_bus_geo == 1:
            return _geo_json_str(this_bus_geo.iloc[0])
        elif len_this_bus_geo == 2:
            how_often = pd.Series(
                [sum(np.isclose(lgd_EIC_bus["lat"], this_bus_geo["lat"].iat[i]) &
                     np.isclose(lgd_EIC_bus["lng"], this_bus_geo["lng"].iat[i])) for i in
                 range(len_this_bus_geo)], index=this_bus_geo.index)
            if how_often.at[how_often.idxmax()] >= 1:
                logger.warning(f"Bus {bus} (name {net.bus.at[bus, 'name']}) was found multiple times"
                               " in line_geo_data. No value exists more often than others. "
                               "The first of most used geo positions is used.")
            return _geo_json_str(this_bus_geo.loc[how_often.idxmax()])

    net.bus.geo = [_add_bus_geo_inner(bus) for bus in net.bus.index]


# --- tertiary functions ---------------------------------------------------------------------------

def _float_col_comma_correction(data: dict[str, pd.DataFrame], key: str, col_names: list):
    for col_name in col_names:
        data[key][col_name] = pd.to_numeric(data[key][col_name].astype(str).str.replace(
            ",", "."), errors="coerce")


def _get_transformer_voltages(
        data: dict[str, pd.DataFrame], bus_idx: pd.Series) -> tuple[np.ndarray, np.ndarray]:

    key = "Transformers"
    vn = data[key].loc[:, [("Voltage_level(kV)", "Primary"),
                           ("Voltage_level(kV)", "Secondary")]].values
    vn_hv_kv = np.max(vn, axis=1)
    vn_lv_kv = np.min(vn, axis=1)
    if is_integer_dtype(list(bus_idx.index.dtypes)[1]):
        vn_hv_kv = vn_hv_kv.astype(int)
        vn_lv_kv = vn_lv_kv.astype(int)

    return vn_hv_kv, vn_lv_kv


def _allocate_trafos_to_buses_and_create_buses(
        net: pandapowerNet, data: dict[str, pd.DataFrame], bus_idx: pd.Series,
        vn_hv_kv: np.ndarray, vn_lv_kv: np.ndarray,
        rel_deviation_threshold_for_trafo_bus_creation: float = 0.2,
        log_rel_vn_deviation: float = 0.12, **kwargs) -> pd.DataFrame:
    """
    Zuordnung der Trafos zu Bussen mit Standort-Scoring:
    - bevorzugt Standorte, die beide Zielspannungen (hv & lv) haben,
    - bevorzugt Standorte mit vorhandenen Leitungen,
    - tie-breaker nach minimaler vn-Abweichung.
    """

    if rel_deviation_threshold_for_trafo_bus_creation < log_rel_vn_deviation:
        logger.warning(
            f"Given parameters violates the ineqation "
            f"{rel_deviation_threshold_for_trafo_bus_creation=} >= {log_rel_vn_deviation=}. "
            f"Therefore, rel_deviation_threshold_for_trafo_bus_creation={log_rel_vn_deviation} is assumed.")
        rel_deviation_threshold_for_trafo_bus_creation = log_rel_vn_deviation

    key = "Transformers"
    bus_location_names = set(net.bus.name)
    trafo_bus_names = data[key].loc[:, ("Location", "Full Name")].astype(str).str.strip()
    trafo_location_names = _find_trafo_locations(trafo_bus_names, bus_location_names)

    empties = -1*np.ones(len(vn_hv_kv), dtype=int)
    trafo_connections = pd.DataFrame({
        "name": trafo_location_names,
        "hv_bus": empties,
        "lv_bus": empties,
        "vn_hv_kv": vn_hv_kv,
        "vn_lv_kv": vn_lv_kv,
        "vn_hv_kv_next_bus": vn_hv_kv,
        "vn_lv_kv_next_bus": vn_lv_kv,
        "hv_rel_deviation": np.zeros(len(vn_hv_kv)),
        "lv_rel_deviation": np.zeros(len(vn_hv_kv)),
    }, dtype=object)
    trafo_connections[["hv_bus", "lv_bus"]] = trafo_connections[["hv_bus", "lv_bus"]].astype(np.int64)

    # Hilfsfunktion: Standort-Score
    def _candidate_score(locname: str, target_hv: float, target_lv: float) -> int:
        score = 0
        # verfügbare VNs am Standort
        try:
            loc_vns = set(bus_idx.loc[locname].index.values)
            if target_hv in loc_vns and target_lv in loc_vns:
                score += 2   # beide VNs vorhanden
            elif (target_hv in loc_vns) or (target_lv in loc_vns):
                score += 1   # mindestens eine VN vorhanden
            # Leitungen am Standort?
            loc_buses = bus_idx.loc[locname].values
            has_lines = any(((net.line.from_bus.isin(loc_buses)) | (net.line.to_bus.isin(loc_buses))).values)
            if has_lines:
                score += 1
        except KeyError:
            pass
        return score

    for side in ["hv", "lv"]:
        bus_col, trafo_vn_col, next_col, rel_dev_col, has_dev_col = \
            f"{side}_bus", f"vn_{side}_kv", f"vn_{side}_kv_next_bus", f"{side}_rel_deviation", \
            f"trafo_{side}_to_bus_deviation"

        # exakte (name,vn)-Treffer
        name_vn_series = pd.Series(tuple(zip(trafo_location_names, trafo_connections[trafo_vn_col])), dtype=object)
        isin = name_vn_series.isin(bus_idx.index)
        trafo_connections[has_dev_col] = ~isin
        if isin.any():
            trafo_connections.loc[isin, bus_col] = bus_idx.loc[name_vn_series.loc[isin]].values
            trafo_connections.loc[isin, next_col] = trafo_connections.loc[isin, trafo_vn_col].values
            trafo_connections.loc[isin, rel_dev_col] = 0.0

        # nicht-exakte Treffer: Scoring + minimaler Abstand
        not_isin = ~isin
        if not_isin.any():
            chosen_vns = []
            chosen_buses = []
            rel_devs = []
            for tln in trafo_connections.loc[not_isin, ["name", trafo_vn_col, "vn_hv_kv", "vn_lv_kv"]].itertuples():
                locname = getattr(tln, "name")
                target_vn = getattr(tln, trafo_vn_col)
                target_hv = getattr(tln, "vn_hv_kv")
                target_lv = getattr(tln, "vn_lv_kv")

                if locname in bus_idx.index.get_level_values(0):
                    # Kandidaten am Standort
                    loc_series = bus_idx.loc[locname]  # Index = verfügbare VNs, Werte = Bus-IDs
                    # Score einmal pro Standort
                    base_score = _candidate_score(locname, target_hv, target_lv)
                    candidates = []
                    for cand_vn, cand_bus in loc_series.items():
                        diff = abs(float(cand_vn) - float(target_vn))
                        # Gesamtscore: Standortscore, dann geringer vn-Abstand
                        candidates.append((base_score, diff, float(cand_vn), int(cand_bus)))
                    candidates.sort(key=lambda x: (-x[0], x[1]))  # bester Score, dann geringste Abweichung
                    chosen_score, chosen_diff, chosen_vn, chosen_bus = candidates[0]
                    chosen_vns.append(chosen_vn)
                    chosen_buses.append(chosen_bus)
                    rel_devs.append(chosen_diff / max(chosen_vn, 1e-9))
                else:
                    # Standort nicht vorhanden -> globale vn-Nachbarschaft
                    all_bus_vn = net.bus.vn_kv.astype(float)
                    nearest_bus = (all_bus_vn - float(target_vn)).abs().idxmin()
                    chosen_vn = float(net.bus.vn_kv.at[nearest_bus])
                    chosen_vns.append(chosen_vn)
                    chosen_buses.append(int(nearest_bus))
                    rel_devs.append(abs(chosen_vn - float(target_vn)) / max(chosen_vn, 1e-9))
                    logger.warning(f"Location '{locname}' not present in bus_idx; fallback to global nearest vn (bus {nearest_bus}).")

            trafo_connections.loc[not_isin, next_col] = np.array(chosen_vns)
            trafo_connections.loc[not_isin, rel_dev_col] = np.array(rel_devs, dtype=float)
            trafo_connections.loc[not_isin, bus_col] = np.array(chosen_buses, dtype=int)

        # ggf. neue Busse anlegen, wenn Abweichung zu groß
        need_bus_creation = trafo_connections[rel_dev_col].astype(float) > rel_deviation_threshold_for_trafo_bus_creation
        if need_bus_creation.any():
            new_bus_data = pd.DataFrame({
                "vn_kv": trafo_connections.loc[need_bus_creation, trafo_vn_col].values.astype(float),
                "name": trafo_connections.loc[need_bus_creation, "name"].astype(str).values,
                "TSO": data[key].loc[need_bus_creation, ("Location", "TSO")].astype(str).values
            })
            new_bus_data_dd = _drop_duplicates_and_join_TSO(new_bus_data)
            if len(new_bus_data_dd):
                new_bus_idx = create_buses(net, len(new_bus_data_dd), vn_kv=new_bus_data_dd.vn_kv,
                                           name=new_bus_data_dd.name, zone=new_bus_data_dd.TSO)
                # Map zurück auf neu angelegte Busse
                trafo_connections.loc[need_bus_creation, bus_col] = net.bus.loc[new_bus_idx, ["name","vn_kv"]].reset_index().set_index(["name","vn_kv"]).loc[
                    list(new_bus_data[["name","vn_kv"]].itertuples(index=False, name=None))
                ].values
                trafo_connections.loc[need_bus_creation, next_col] = trafo_connections.loc[need_bus_creation, trafo_vn_col].values
                trafo_connections.loc[need_bus_creation, rel_dev_col] = 0.0
                trafo_connections.loc[need_bus_creation, has_dev_col] = False

    # gleicher Bus an beiden Seiten -> duplizieren
    same_bus_connection = trafo_connections.hv_bus == trafo_connections.lv_bus
    duplicated_buses = net.bus.loc[trafo_connections.loc[same_bus_connection, "lv_bus"]].copy()
    duplicated_buses["name"] += " (2)"
    duplicated_buses.index = list(range(net.bus.index.max()+1, net.bus.index.max()+1+len(duplicated_buses)))
    trafo_connections.loc[same_bus_connection, "lv_bus"] = duplicated_buses.index
    net.bus = pd.concat([net.bus, duplicated_buses])
    if len(duplicated_buses):
        tr_names = data[key].loc[trafo_connections.index[same_bus_connection], ("Location", "Full Name")]
        are_PSTs = tr_names.str.contains("PST", na=False)
        logger.info(f"{len(duplicated_buses)} additional buses created to avoid same-bus trafo connections. PST count: {int(are_PSTs.sum())}")

    # Logging bei Abweichung
    for side in ["hv","lv"]:
        bus_col, trafo_vn_col, next_col, rel_dev_col, has_dev_col = \
            f"{side}_bus", f"vn_{side}_kv", f"vn_{side}_kv_next_bus", f"{side}_rel_deviation", f"trafo_{side}_to_bus_deviation"
        need_logging = trafo_connections.loc[trafo_connections[has_dev_col], rel_dev_col].astype(float) > log_rel_vn_deviation
        if need_logging.any():
            idx_max = trafo_connections.loc[trafo_connections[has_dev_col], rel_dev_col].astype(float).idxmax()
            max_dev = float(trafo_connections.at[idx_max, rel_dev_col])
            logger.warning(
                f"For {int(need_logging.sum())} transformers ({side} side) vn deviation > {log_rel_vn_deviation}. "
                f"Max deviation {max_dev:.3f} at vn_trafo={trafo_connections.at[idx_max, trafo_vn_col]}, "
                f"vn_bus={trafo_connections.at[idx_max, next_col]}."
            )

    assert (trafo_connections.hv_bus.astype(int) > -1).all()
    assert (trafo_connections.lv_bus.astype(int) > -1).all()
    assert (trafo_connections.hv_bus.astype(int) != trafo_connections.lv_bus.astype(int)).all()

    return trafo_connections


def _strip_diacritics(s: str) -> str:
    if s is None:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def _canon_base(s: str) -> str:
    s = _strip_diacritics(s).upper()
    s = re.sub(r"[-/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokenize_location(s: str) -> list[str]:
    # entferne Suffix-Codes -Axx/-TDxx/-PFxx, technische Tokens, Zahlen und Ein-Zeichen
    s = re.sub(r"-(A|TD|PF)\d+", " ", s)
    tokens = [t for t in re.split(r"\s+", s) if t]
    drop = {"TR", "TRAFO", "PST", "TFO", "KV", "/", "EHPST", "LIPST"}
    def is_code(t: str) -> bool:
        return any(ch.isdigit() for ch in t)
    tokens = [t for t in tokens if t not in drop and not is_code(t) and len(t) > 1]
    return tokens

def _equiv_token(a: str, b: str) -> bool:
    if a == b:
        return True
    # toleriert MATK ~ MATKI (trailing 'I')
    if a.rstrip("I") == b or b.rstrip("I") == a or a.rstrip("I") == b.rstrip("I"):
        if min(len(a), len(b)) >= 4:
            return True
    # Romanzahlen als separate Tokens ignorieren
    if a in {"I", "II", "III", "IV", "V"} or b in {"I", "II", "III", "IV", "V"}:
        return True
    # einfacher gemeinsamer Präfix für längere Tokens
    if min(len(a), len(b)) >= 5:
        pref = 0
        for x, y in zip(a, b):
            if x == y:
                pref += 1
            else:
                break
        if pref >= 4:
            return True
    return False

def _tokens_match_score(ta: list[str], tb: list[str]) -> int:
    score, used = 0, set()
    for x in ta:
        for j, y in enumerate(tb):
            if j in used:
                continue
            if _equiv_token(x, y):
                score += 1
                used.add(j)
                break
    return score

def _normalize_for_matching(series: pd.Series) -> pd.DataFrame:
    canon = series.astype(str).apply(_canon_base)
    tokens = canon.apply(_tokenize_location)
    joined = tokens.apply(lambda t: " ".join(t))
    return pd.DataFrame({"original": series.astype(str), "canon": canon, "tokens": tokens, "joined": joined})

# geänderte Funktion: _find_trafo_locations
def _find_trafo_locations(trafo_bus_names: pd.Series, bus_location_names: set[str]) -> pd.Series:
    """
    Dynamische, robuste Standort-Zuordnung: normalisiert beide Seiten und gibt
    originale Busnamen (aus bus_location_names) zurück.
    """
    # Busseite normalisieren
    bus_series = pd.Series(sorted(bus_location_names))
    bus_df = _normalize_for_matching(bus_series)

    # Varianten -> Original-Mapping
    variant_to_original = {}
    for i in range(len(bus_df)):
        for key in (bus_df.joined.iat[i], bus_df.canon.iat[i]):
            if key and key not in variant_to_original:
                variant_to_original[key] = bus_df.original.iat[i]

    # Trafoseite normalisieren
    trafo_df = _normalize_for_matching(trafo_bus_names.astype(str).str.strip())

    mapped = [None] * len(trafo_df)
    unresolved_idx = []

    # Direkte Treffer
    for i in range(len(trafo_df)):
        j = trafo_df.joined.iat[i]
        c = trafo_df.canon.iat[i]
        if j in variant_to_original:
            mapped[i] = variant_to_original[j]
        elif c in variant_to_original:
            mapped[i] = variant_to_original[c]
        else:
            unresolved_idx.append(i)

    # Fuzzy/Token-Matching
    if unresolved_idx:
        for i in unresolved_idx:
            t_tokens = trafo_df.tokens.iat[i]
            significant = [x for x in t_tokens if len(x) >= 4]

            best_score, best_original = 0, None
            if significant:
                for k in range(len(bus_df)):
                    b_tokens = bus_df.tokens.iat[k]
                    # Vorfilter
                    if not any(len(x) >= 4 and (x in b_tokens or any(_equiv_token(x, y) for y in b_tokens)) for x in significant):
                        continue
                    score = _tokens_match_score(t_tokens, b_tokens)
                    if score > best_score:
                        best_score = score
                        best_original = bus_df.original.iat[k]
                if best_original is not None and best_score >= 2:
                    mapped[i] = best_original

            # Letzter Fallback via SequenceMatcher
            if mapped[i] is None:
                ratios = bus_df.canon.apply(lambda b: difflib.SequenceMatcher(None, trafo_df.canon.iat[i], b).ratio())
                kbest = int(ratios.idxmax())
                mapped[i] = bus_df.original.iat[kbest]
                logger.warning(f"Fuzzy fallback (ratio={ratios.max():.2f}) for transformer location '{trafo_df.original.iat[i]}' -> '{mapped[i]}'")

    # Safety: None vermeiden
    for i, m in enumerate(mapped):
        if m is None:
            mapped[i] = bus_df.original.iat[0]
            logger.error(f"Hard fallback mapping '{trafo_df.original.iat[i]}' -> '{mapped[i]}'")

    return pd.Series(mapped, index=trafo_df.index)


def _drop_duplicates_and_join_TSO(bus_df: pd.DataFrame) -> pd.DataFrame:
    bus_df = bus_df.drop_duplicates(ignore_index=True)
    # just keep one bus per name and vn_kv. If there are multiple buses of different TSOs, join the
    # TSO strings:
    bus_df = bus_df.groupby(["name", "vn_kv"], as_index=False).agg({"TSO": lambda x: '/'.join(x)})
    assert not bus_df.duplicated(["name", "vn_kv"]).any()
    return bus_df


def _get_float_column(df, col_tuple, fill=0):
    series = df.loc[:, col_tuple]
    series.loc[series == "\xa0"] = fill
    return series.astype(float).fillna(fill)


def _get_bus_idx(net: pandapowerNet) -> pd.Series:
    return net.bus[["name", "vn_kv"]].rename_axis("index").reset_index().set_index([
        "name", "vn_kv"])["index"]


def get_grid_groups(net: pandapowerNet, **kwargs) -> pd.DataFrame:
    notravbuses_dict = dict() if "notravbuses" not in kwargs.keys() else {
        "notravbuses": kwargs.pop("notravbuses")}
    grid_group_buses = [set_ for set_ in connected_components(create_nxgraph(net, **kwargs),
                                                              **notravbuses_dict)]
    grid_groups = pd.DataFrame({"buses": grid_group_buses})
    grid_groups["n_buses"] = grid_groups["buses"].apply(len)
    return grid_groups


def _lng_lat_to_df(dict_: dict, line_EIC: str, line_name: str) -> pd.DataFrame:
    return pd.DataFrame([
        [line_EIC, line_name, "from", "lng", dict_["lng"][0]],
        [line_EIC, line_name,   "to", "lng", dict_["lng"][1]],
        [line_EIC, line_name, "from", "lat", dict_["lat"][0]],
        [line_EIC, line_name,   "to", "lat", dict_["lat"][1]],
    ], columns=["EIC_Code", "name", "bus", "geo_dim", "value"])


def _fill_geo_at_one_sided_branches_without_geo_extent(net: pandapowerNet):

    def _check_geo_availablitiy(net: pandapowerNet) -> dict[str, Union[pd.Index, int]]:
        av = dict()  # availablitiy of geodata
        av["bus_with_geo"] = net.bus.index[~net.bus.geo.isnull()]
        av["lines_fbw_tbwo"] = net.line.index[net.line.from_bus.isin(av["bus_with_geo"]) &
                                              (~net.line.to_bus.isin(av["bus_with_geo"]))]
        av["lines_fbwo_tbw"] = net.line.index[(~net.line.from_bus.isin(av["bus_with_geo"])) &
                                              net.line.to_bus.isin(av["bus_with_geo"])]
        av["trafos_hvbw_lvbwo"] = net.trafo.index[net.trafo.hv_bus.isin(av["bus_with_geo"]) &
                                                  (~net.trafo.lv_bus.isin(av["bus_with_geo"]))]
        av["trafos_hvbwo_lvbw"] = net.trafo.index[(~net.trafo.hv_bus.isin(av["bus_with_geo"])) &
                                                  net.trafo.lv_bus.isin(av["bus_with_geo"])]
        av["n_lines_one_side_geo"] = len(av["lines_fbw_tbwo"])+len(av["lines_fbwo_tbw"])
        return av

    geo_avail = _check_geo_availablitiy(net)
    while geo_avail["n_lines_one_side_geo"]:

        # copy available geodata to the other end of branches where geodata are missing
        for et, bus_w_geo, bus_wo_geo, idx_key in zip(
                ["line", "line", "trafo", "trafo"],
                ["to_bus", "from_bus", "lv_bus", "hv_bus"],
                ["from_bus", "to_bus", "hv_bus", "lv_bus"],
                ["lines_fbwo_tbw", "lines_fbw_tbwo", "trafos_hvbwo_lvbw", "trafos_hvbw_lvbwo"]):
            net.bus.loc[net[et].loc[geo_avail[idx_key], bus_wo_geo].values, "geo"] = \
                net.bus.loc[net[et].loc[geo_avail[idx_key], bus_w_geo].values, "geo"].values
        geo_avail = _check_geo_availablitiy(net)

    set_line_geodata_from_bus_geodata(net)


def _multi_str_repl(st: str, repl: list[tuple]) -> str:
    for (old, new) in repl:
        st = st.replace(old, new)
    return st

def _canon_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.upper()
    s = re.sub(r"[-_/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _bus_df(net):
    df = net.bus.reset_index()[["index","name","vn_kv"]]
    df["key"] = list(zip(df["name"].astype(str), df["vn_kv"].astype(float)))
    df["canon"] = df["name"].map(_canon_name)
    return df

def _usage_table(net, bus_indices):
    rows=[]
    for idx in bus_indices:
        n_lines = int(((net.line.from_bus == idx) | (net.line.to_bus == idx)).sum())
        hv_tr = net.trafo.index[net.trafo.hv_bus == idx].tolist()
        lv_tr = net.trafo.index[net.trafo.lv_bus == idx].tolist()
        rows.append((idx, n_lines, hv_tr, lv_tr))
    return pd.DataFrame(rows, columns=["bus_idx","n_lines","hv_trafos","lv_trafos"]).set_index("bus_idx")

def analyze_added_buses(net_dyn, net_hc):
    d_dyn = _bus_df(net_dyn)
    d_hc  = _bus_df(net_hc)

    set_dyn = set(d_dyn["key"])
    set_hc  = set(d_hc["key"])
    extras_keys  = set_dyn - set_hc
    missing_keys = set_hc - set_dyn

    extras  = d_dyn[d_dyn["key"].isin(extras_keys)].copy()
    missing = d_hc[d_hc["key"].isin(missing_keys)].copy()

    # Paare reine Umbenennungen per (canon, vn_kv)
    paired = []
    used_missing = set()
    for i, r in extras.iterrows():
        cand = missing[(missing["canon"] == r["canon"]) & (np.isclose(missing["vn_kv"], r["vn_kv"]))]
        if len(cand):
            j = cand.index[0]
            paired.append((i, j))
            used_missing.add(missing.at[j, "key"])
    extras["is_paired"] = extras.index.isin([i for i,_ in paired])
    missing["is_paired"] = missing.index.isin([j for _,j in paired])

    # Netto hinzugekommen / weggefallen (echte Topologie-/Zähländerung)
    net_added   = extras[~extras["is_paired"]].copy()
    net_removed = missing[~missing["is_paired"]].copy()

    # Nutzung annotieren
    na = _usage_table(net_dyn, net_added["index"]).join(net_dyn.bus.loc[net_added["index"], ["name","vn_kv"]])
    nr = _usage_table(net_hc,  net_removed["index"]).join(net_hc.bus.loc[net_removed["index"], ["name","vn_kv"]])

    return na.reset_index(), nr.reset_index(), extras, missing


if __name__ == "__main__":
    from pathlib import Path
    import os
    import pandapower as pp

    home = str(Path.home())
    jao_data_folder = os.path.join(home, "Documents", "JAO Static Grid Model")

    release5 = os.path.join(jao_data_folder, "20240329_Core Static Grid Model – 5th release")
    excel_file_path = os.path.join(release5, "20240329_Core Static Grid Model_public.xlsx")
    html_file_path = os.path.join(release5, "20240329_Core Static Grid Model Map_public",
                                  "2024-03-18_Core_SGM_publication.html")

    release6 = os.path.join(jao_data_folder, "202409_Core Static Grid Mode_6th release")
    excel_file_path = os.path.join(release6, "20240916_Core Static Grid Model_for publication.xlsx")
    html_file_path = os.path.join(release6, "2024-09-13_Core_SGM_publication_files",
                                  "2024-09-13_Core_SGM_publication.html")

    pp_net_json_file = os.path.join(home, "desktop", "jao_grid.json")

    if 1:  # read from original data
        net = from_jao(excel_file_path, html_file_path, True, drop_grid_groups_islands=True)
        pp.to_json(net, pp_net_json_file)
    else:  # load net from already converted and stored net
        net = pp.from_json(pp_net_json_file)

    print(net)
    grid_groups = get_grid_groups(net)
    print(grid_groups)

    _fill_geo_at_one_sided_branches_without_geo_extent(net)
