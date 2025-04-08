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

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def from_jao(excel_file_path: str,
             html_file_path: Optional[str],
             extend_data_for_grid_group_connections: bool,
             drop_grid_groups_islands: bool = False,
             apply_data_correction: bool = True,
             max_i_ka_fillna: Union[float, int] = 999,
             **kwargs) -> pandapowerNet:
    """Converts European (Core) EHV grid data provided by JAO (Joint Allocation Office), the
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

    Parameters
    ----------
    excel_file_path : str
        input data including electrical parameters of grids' utilities, stored in multiple sheets
        of an excel file
    html_file_path : str
        input data for geo information. If The converter should be run without geo information, None
        can be passed., provided by an html file
    extend_data_for_grid_group_connections : bool
        if True, connections (additional transformers and merging buses) are created to avoid
        islanded grid groups, by default False
    drop_grid_groups_islands : bool, optional
        if True, islanded grid groups will be dropped if their number of buses is below
        min_bus_number (default is 6), by default False
    apply_data_correction : bool, optional
        _description_, by default True
    max_i_ka_fillna : float | int, optional
        value to fill missing values or data of false type in max_i_ka of lines and transformers.
        If no value should be set, you can also pass np.nan. By default 999

    Returns
    -------
    pandapowerNet
        net created from the jao data

    Additional Parameters
    ---------------------
    minimal_trafo_invention : bool, optional
        applies if extend_data_for_grid_group_connections is True. Then, if minimal_trafo_invention
        is True, adding transformers stops when no grid groups is islanded anymore (does not apply
        for release version 5 or 6, i.e. it does not care what value is passed to
        minimal_trafo_invention). If False, all equally named buses that have different voltage
        level and lay in different groups will be connected via additional transformers,
        by default False
    min_bus_number : Union[int,str], optional
        Threshold value to decide which small grid groups should be dropped and which large grid
        groups should be kept. If all islanded grid groups should be dropped except of the one
        largest, set "max". If all grid groups that do not contain a slack element should be
        dropped, set "unsupplied". By default 6
    rel_deviation_threshold_for_trafo_bus_creation : float, optional
        If the voltage level of transformer locations is far different than the transformer data,
        additional buses are created. rel_deviation_threshold_for_trafo_bus_creation defines the
        tolerance in which no additional buses are created. By default 0.2
    log_rel_vn_deviation : float, optional
        This parameter allows a range below rel_deviation_threshold_for_trafo_bus_creation in which
        a warning is logged instead of a creating additional buses. By default 0.12

    Examples
    --------
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


def _data_correction(
        data: dict[str, pd.DataFrame],
        html_str: Optional[str],
        max_i_ka_fillna: Union[float, int]) -> Optional[str]:
    """Corrects input data in particular with regard to obvious weaknesses in the data provided,
    such as inconsistent spellings and missing necessary information

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        data provided by the excel file which will be corrected
    html_str : str | None
        data provided by the html file which will be corrected
    max_i_ka_fillna : float | int
        value to fill missing values or data of false type in max_i_ka of lines and transformers.
        If no value should be set, you can also pass np.nan.

    Returns
    -------
    str
        corrected html_str
    """
    # old name -> new name
    rename_locnames = [("PSTMIKULOWA", "PST MIKULOWA"),
                       ("Chelm", "CHELM"),
                       ("OLSZTYN-MATK", "OLSZTYN-MATKI"),
                       ("STANISLAWOW", "Stanislawow"),
                       ("VIERRADEN", "Vierraden")]

    # --- Line and Tieline data ---------------------------
    for key in ["Lines", "Tielines"]:

        # --- correct column names
        cols = data[key].columns.to_frame().reset_index(drop=True)
        cols.loc[cols[1] == "Voltage_level(kV)", 0] = None
        cols.loc[cols[1] == "Comment", 0] = None
        cols.loc[cols[0].str.startswith("Unnamed:").astype(bool), 0] = None
        cols.loc[cols[1] == "Length_(km)", 0] = "Electrical Parameters"  # might be wrong in
        # Tielines otherwise
        data[key].columns = pd.MultiIndex.from_arrays(cols.values.T)

        # --- correct comma separation and cast to floats
        data[key][("Maximum Current Imax (A)", "Fixed")] = \
            data[key][("Maximum Current Imax (A)", "Fixed")].replace(
                "\xa0", max_i_ka_fillna*1e3).replace(
                "-", max_i_ka_fillna*1e3).replace(" ", max_i_ka_fillna*1e3)
        col_names = [("Electrical Parameters", col_level1) for col_level1 in [
            "Length_(km)", "Resistance_R(Ω)", "Reactance_X(Ω)", "Susceptance_B(μS)",
            "Length_(km)"]] + [("Maximum Current Imax (A)", "Fixed")]
        _float_col_comma_correction(data, key, col_names)

        # --- consolidate to one way of name capitalization
        for loc_name in [(None, "NE_name"), ("Substation_1", "Full_name"),
                         ("Substation_2", "Full_name")]:
            data[key].loc[:, loc_name] = data[key].loc[:, loc_name].str.strip().apply(
                _multi_str_repl, repl=rename_locnames)
    html_str = _multi_str_repl(html_str, rename_locnames)

    # --- Transformer data --------------------------------
    key = "Transformers"

    # --- fix Locations
    loc_name = ("Location", "Full Name")
    data[key].loc[:, loc_name] = data[key].loc[:, loc_name].str.strip().apply(
        _multi_str_repl, repl=rename_locnames)

    # --- fix data in nonnull_taps
    taps = data[key].loc[:, ("Phase Shifting Properties", "Taps used for RAO")].fillna("").astype(
        str).str.replace(" ", "")
    nonnull = taps.apply(len).astype(bool)
    nonnull_taps = taps.loc[nonnull]
    surrounded = nonnull_taps.str.startswith("<") & nonnull_taps.str.endswith(">")
    nonnull_taps.loc[surrounded] = nonnull_taps.loc[surrounded].str[1:-1]
    slash_sep = (~nonnull_taps.str.contains(";")) & nonnull_taps.str.contains("/")
    nonnull_taps.loc[slash_sep] = nonnull_taps.loc[slash_sep].str.replace("/", ";")
    nonnull_taps.loc[nonnull_taps == "0"] = "0;0"
    data[key].loc[nonnull, ("Phase Shifting Properties", "Taps used for RAO")] = nonnull_taps
    data[key].loc[~nonnull, ("Phase Shifting Properties", "Taps used for RAO")] = "0;0"

    # --- phase shifter with double info
    cols = ["Phase Regulation δu (%)", "Angle Regulation δu (%)"]
    for col in cols:
        if is_object_dtype(data[key].loc[:, ("Phase Shifting Properties", col)]):
            tr_double = data[key].index[data[key].loc[:, (
                "Phase Shifting Properties", col)].str.contains("/").fillna(0).astype(bool)]
            data[key].loc[tr_double, ("Phase Shifting Properties", col)] = data[key].loc[
                tr_double, ("Phase Shifting Properties", col)].str.split("/", expand=True)[
                1].str.replace(",", ".").astype(float).values  # take second info and correct
                # separation: , -> .

    return html_str


def _parse_html_str(html_str: str) -> pd.DataFrame:
    """Converts ths geodata from the html file (information hidden in the string), from Lines in
    particular, to a DataFrame that can be used later in _add_bus_geo()

    Parameters
    ----------
    html_str : str
        html file that includes geodata information

    Returns
    -------
    pd.DataFrame
        extracted geodata for a later and easy use
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

    Parameters
    ----------
    net : pandapowerNet
        net to be filled by buses
    data : dict[str, pd.DataFrame]
        data provided by the excel file which will be corrected
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
    """Creates lines to the pandapower net using information from the lines and tielines sheets
    (excel file).

    Parameters
    ----------
    net : pandapowerNet
        net to be filled by buses
    data : dict[str, pd.DataFrame]
        data provided by the excel file which will be corrected
    max_i_ka_fillna : float | int
        value to fill missing values or data of false type in max_i_ka of lines and transformers.
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
    """Creates transformers to the pandapower net using information from the transformers sheet
    (excel file).

    Parameters
    ----------
    net : pandapowerNet
        net to be filled by buses
    data : dict[str, pd.DataFrame]
        data provided by the excel file which will be corrected
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
    """Adds connections between islanded grid groups via:

    - adding transformers between equally named buses that have different voltage level and lay in different groups
    - merge buses of same voltage level, different grid groups and equal name base
    - fuse buses that are close to each other

    Parameters
    ----------
    net : pandapowerNet
        net to be manipulated
    minimal_trafo_invention : bool, optional
        if True, adding transformers stops when no grid groups is islanded anymore (does not apply
        for release version 5 or 6, i.e. it does not care what value is passed to
        minimal_trafo_invention). If False, all equally named buses that have different voltage
        level and lay in different groups will be connected via additional transformers,
        by default False
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
    """Drops grid groups that are islanded and include a number of buses below min_bus_number.

    Parameters
    ----------
    net : pandapowerNet
        net in which islanded grid groups will be dropped
    min_bus_number : int | str, optional
        Threshold value to decide which small grid groups should be dropped and which large grid
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

    Parameters
    ----------
    net : pandapowerNet
        net in which geodata are added to the buses
    line_geo_data : pd.DataFrame
        Converted geodata from the html file
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
    """Provides a DataFrame of data to allocate transformers to the buses according to their
    location names. If locations of transformers do not exist due to the data of the lines and
    tielines sheets, additional buses are created. If locations exist but have a far different
    voltage level than the transformer, either a warning is logged or additional buses are created
    according to rel_deviation_threshold_for_trafo_bus_creation and log_rel_vn_deviation.

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    data : dict[str, pd.DataFrame]
        _description_
    bus_idx : pd.Series
        Series of indices and corresponding location names and voltage levels in the MultiIndex of
        the Series
    vn_hv_kv : np.ndarray
        nominal voltages of the hv side of the transformers
    vn_lv_kv : np.ndarray
        Nominal voltages of the lv side of the transformers
    rel_deviation_threshold_for_trafo_bus_creation : float, optional
        If the voltage level of transformer locations is far different than the transformer data,
        additional buses are created. rel_deviation_threshold_for_trafo_bus_creation defines the
        tolerance in which no additional buses are created. By default 0.2
    log_rel_vn_deviation : float, optional
        This parameter allows a range below rel_deviation_threshold_for_trafo_bus_creation in which
        a warning is logged instead of a creating additional buses. By default 0.12

    Returns
    -------
    pd.DataFrame
        information to which bus the trafos should be connected to. Columns are
        ["name", "hv_bus", "lv_bus", "vn_hv_kv", "vn_lv_kv", ...]
    """

    if rel_deviation_threshold_for_trafo_bus_creation < log_rel_vn_deviation:
        logger.warning(
            f"Given parameters violates the ineqation "
            f"{rel_deviation_threshold_for_trafo_bus_creation=} >= {log_rel_vn_deviation=}. "
            f"Therefore, rel_deviation_threshold_for_trafo_bus_creation={log_rel_vn_deviation} "
            "is assumed.")
        rel_deviation_threshold_for_trafo_bus_creation = log_rel_vn_deviation

    key = "Transformers"
    bus_location_names = set(net.bus.name)
    trafo_bus_names = data[key].loc[:, ("Location", "Full Name")]
    trafo_location_names = _find_trafo_locations(trafo_bus_names, bus_location_names)

    # --- construct DataFrame trafo_connections including all information on trafo allocation to
    # --- buses
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
    })
    trafo_connections[["hv_bus", "lv_bus"]] = trafo_connections[[
        "hv_bus", "lv_bus"]].astype(np.int64)

    for side in ["hv", "lv"]:
        bus_col, trafo_vn_col, next_col, rel_dev_col, has_dev_col = \
            f"{side}_bus", f"vn_{side}_kv", f"vn_{side}_kv_next_bus", f"{side}_rel_deviation", \
            f"trafo_{side}_to_bus_deviation"
        name_vn_series = pd.Series(
            tuple(zip(trafo_location_names, trafo_connections[trafo_vn_col])))
        isin = name_vn_series.isin(bus_idx.index)
        trafo_connections[has_dev_col] = ~isin
        trafo_connections.loc[isin, bus_col] = bus_idx.loc[name_vn_series.loc[isin]].values

        # --- code to find bus locations with vn deviation
        next_vn = np.array([bus_idx.loc[tln.name].index.values[
            (pd.Series(bus_idx.loc[tln.name].index) - getattr(tln, trafo_vn_col)).abs().idxmin(
            )] for tln in trafo_connections.loc[~isin, ["name", trafo_vn_col]].itertuples()])
        trafo_connections.loc[~isin, next_col] = next_vn
        rel_dev = np.abs(next_vn - trafo_connections.loc[~isin, trafo_vn_col].values) / next_vn
        trafo_connections.loc[~isin, rel_dev_col] = rel_dev
        trafo_connections.loc[~isin, bus_col] = \
            bus_idx.loc[list(tuple(zip(trafo_connections.loc[~isin, "name"],
                                       trafo_connections.loc[~isin, next_col])))].values

        # --- create buses to avoid too large vn deviations between nodes and transformers
        need_bus_creation = trafo_connections[rel_dev_col] > \
            rel_deviation_threshold_for_trafo_bus_creation
        new_bus_data = pd.DataFrame({
            "vn_kv": trafo_connections.loc[need_bus_creation, trafo_vn_col].values,
            "name": trafo_connections.loc[need_bus_creation, "name"].values,
            "TSO": data[key].loc[need_bus_creation, ("Location", "TSO")].values
        })
        new_bus_data_dd = _drop_duplicates_and_join_TSO(new_bus_data)
        new_bus_idx = create_buses(net, len(new_bus_data_dd), vn_kv=new_bus_data_dd.vn_kv,
                                   name=new_bus_data_dd.name, zone=new_bus_data_dd.TSO)
        trafo_connections.loc[need_bus_creation, bus_col] = net.bus.loc[new_bus_idx, [
            "name", "vn_kv"]].reset_index().set_index(["name", "vn_kv"]).loc[list(new_bus_data[[
                "name", "vn_kv"]].itertuples(index=False, name=None))].values
        trafo_connections.loc[need_bus_creation, next_col] = \
            trafo_connections.loc[need_bus_creation, trafo_vn_col].values
        trafo_connections.loc[need_bus_creation, rel_dev_col] = 0
        trafo_connections.loc[need_bus_creation, has_dev_col] = False

    # --- create buses for trafos that are connected to the same bus at both sides (possible if
    # --- vn_hv_kv < vn_lv_kv *(1+rel_deviation_threshold_for_trafo_bus_creation) which usually
    # --- occurs for PSTs only)
    same_bus_connection = trafo_connections.hv_bus == trafo_connections.lv_bus
    duplicated_buses = net.bus.loc[trafo_connections.loc[same_bus_connection, "lv_bus"]].copy()
    duplicated_buses["name"] += " (2)"
    duplicated_buses.index = list(range(net.bus.index.max()+1,
                                        net.bus.index.max()+1+len(duplicated_buses)))
    trafo_connections.loc[same_bus_connection, "lv_bus"] = duplicated_buses.index
    net.bus = pd.concat([net.bus, duplicated_buses])
    if n_add_buses := len(duplicated_buses):
        tr_names = data[key].loc[trafo_connections.index[same_bus_connection],
                                 ("Location", "Full Name")]
        are_PSTs = tr_names.str.contains("PST")
        logger.info(f"{n_add_buses} additional buses were created to avoid that transformers are "
                    f"connected to the same bus at both side, hv and lv. Of the causing "
                    f"{len(tr_names)} transformers, {sum(are_PSTs)} contain 'PST' in their name. "
                    f"According to this converter, the power flows over all these transformers will"
                    f" end at the additional buses. Please consider to connect lines with the "
                    f"additional buses, so that the power flow is over the (PST) transformers into "
                    f"the lines.")

    # --- log according to log_rel_vn_deviation
    for side in ["hv", "lv"]:
        need_logging = trafo_connections.loc[trafo_connections[has_dev_col],
                                             rel_dev_col] > log_rel_vn_deviation
        if n_need_logging := sum(need_logging):
            max_dev = trafo_connections[rel_dev_col].max()
            idx_max_dev = trafo_connections[rel_dev_col].idxmax()
            logger.warning(
                f"For {n_need_logging} Transformers ({side} side), only locations were found (orig"
                f"in are the line and tieline data) that have a higher relative deviation than "
                f"{log_rel_vn_deviation}. The maximum relative deviation is {max_dev} which "
                f"results from a Transformer rated voltage of "
                f"{trafo_connections.at[idx_max_dev, trafo_vn_col]} and a bus "
                f"rated voltage (taken from Lines/Tielines data sheet) of "
                f"{trafo_connections.at[idx_max_dev, next_col]}. The best locations were "
                f"nevertheless applied, due to {rel_deviation_threshold_for_trafo_bus_creation=}")

    assert (trafo_connections.hv_bus > -1).all()
    assert (trafo_connections.lv_bus > -1).all()
    assert (trafo_connections.hv_bus != trafo_connections.lv_bus).all()

    return trafo_connections


def _find_trafo_locations(trafo_bus_names, bus_location_names):
    # --- split (original and lower case) strings at " " separators to remove impeding parts for
    # identifying the location names
    trafo_bus_names_expended = trafo_bus_names.str.split(r"[ ]+|-A[0-9]+|-TD[0-9]+|-PF[0-9]+",
                                                         expand=True).fillna("").replace(" ", "")
    trafo_bus_names_expended_lower = trafo_bus_names.str.lower().str.split(
        r"[ ]+|-A[0-9]+|-TD[0-9]+|-PF[0-9]+", expand=True).fillna("").replace(" ", "")

    # --- identify impeding parts
    contains_number = trafo_bus_names_expended.map(lambda x: any(char.isdigit() for char in x))
    to_drop = (trafo_bus_names_expended_lower == "tr") | (trafo_bus_names_expended_lower == "pst") \
        | (trafo_bus_names_expended == "") | (trafo_bus_names_expended == "/") | (
        trafo_bus_names_expended == "LIPST") | (trafo_bus_names_expended == "EHPST") | (
        trafo_bus_names_expended == "TFO") | (trafo_bus_names_expended_lower == "trafo") | (
        trafo_bus_names_expended_lower == "kv") | contains_number
    trafo_bus_names_expended[to_drop] = ""

    # --- reconstruct name strings for identification
    trafo_bus_names_joined = trafo_bus_names_expended.where(~to_drop).fillna('').agg(
        ' '.join, axis=1).str.strip()
    trafo_bus_names_longest_part = trafo_bus_names_expended.apply(
        lambda row: max(row, key=len), axis=1)
    joined_in_buses = trafo_bus_names_joined.isin(bus_location_names)
    longest_part_in_buses = trafo_bus_names_longest_part.isin(bus_location_names)

    # --- check whether all name strings point at location names of the buses
    if False:  # for easy testing
        fail = ~(joined_in_buses | longest_part_in_buses)
        a = pd.concat([trafo_bus_names_joined.loc[fail],
                      trafo_bus_names_longest_part.loc[fail]], axis=1)

    if n_bus_names_not_found := len(joined_in_buses) - sum(joined_in_buses | longest_part_in_buses):
        raise ValueError(
            f"For {n_bus_names_not_found} Tranformers, no suitable bus location names were found, "
            f"i.e. the algorithm did not find a (part) of Transformers-Location-Full Name that fits"
            " to Substation_1 or Substation_2 data in Lines or Tielines sheet.")

    # --- set the trafo location names and trafo bus indices respectively
    trafo_location_names = trafo_bus_names_longest_part
    trafo_location_names.loc[joined_in_buses] = trafo_bus_names_joined

    return trafo_location_names


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
        return st.replace(old, new)


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
