# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import difflib
import json
import logging
import os
import re
import unicodedata
from copy import deepcopy
from functools import reduce

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_object_dtype

from pandapower.create import create_buses, create_lines_from_parameters, create_transformers_from_parameters
from pandapower.network import pandapowerNet
from pandapower.plotting import set_line_geodata_from_bus_geodata
from pandapower.toolbox.grid_modification import drop_buses, fuse_buses
from pandapower.topology import connected_components, create_nxgraph

logger = logging.getLogger(__name__)


def from_jao(
    excel_file_path: str,
    html_file_path: str | None,
    extend_data_for_grid_group_connections: bool,
    drop_grid_groups_islands: bool = False,
    apply_data_correction: bool = True,
    max_i_ka_fillna: float = 999,
    **kwargs,
) -> pandapowerNet:
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

    Parameters:
        excel_file_path: input data including electrical parameters of grids' utilities, stored in multiple sheets
            of an excel file
        html_file_path: input data for geo information. If The converter should be run without geo information, None
            can be passed., provided by an html file
        extend_data_for_grid_group_connections: if True, connections (additional transformers and merging buses) are
            created to avoid islanded grid groups, by default False
        drop_grid_groups_islands: if True, islanded grid groups will be dropped if their number of buses is below
            `min_bus_number` default for this is 6 (default: False)
        apply_data_correction: _description_ (default: True)
        max_i_ka_fillna: value to fill missing values or data of false type in max_i_ka of lines and transformers.
            If no value should be set, you can also pass np.nan. (default: 999)

    Keyword Arguments:
        minimal_trafo_invention (Optional[bool]): applies if extend_data_for_grid_group_connections is True. Then,
            if minimal_trafo_invention is True, adding transformers stops when no grid groups is islanded anymore
            (does not apply for release version 5 or 6, i.e. it does not care what value is passed to
            minimal_trafo_invention). If False, all equally named buses that have different voltage level and lay in
            different groups will be connected via additional transformers (default: False)
        min_bus_number (Optional[int|str]): Threshold value to decide which small grid groups should be dropped and
            which large grid groups should be kept. If all islanded grid groups should be dropped except of the one
            largest, set "max". If all grid groups that do not contain a slack element should be dropped, set
            "unsupplied". (default: 6)
        rel_deviation_threshold_for_trafo_bus_creation (Optional[float]): If the voltage level of transformer locations
            is far different than the transformer data, additional buses are created.
            rel_deviation_threshold_for_trafo_bus_creation defines the tolerance in which no additional buses are
            created. (default: 0.2)
        log_rel_vn_deviation (Optional[float]): This parameter allows a range below
            rel_deviation_threshold_for_trafo_bus_creation in which a warning is logged instead of a creating additional
            buses. (default: 0.12)

    Returns:
        net created from the jao data

    Example:
        >>> from pathlib import Path
        >>> import os
        >>> from pandapower.converter.jao.from_jao import from_jao
        >>> net = from_jao()
        >>> home = str(Path.home())
        >>> # assume that the files are located at your desktop:
        >>> excel_file_path = os.path.join(home, "desktop", "202409_Core Static Grid Mode_6th release")
        >>> html_file_path = os.path.join(home, "desktop", "2024-09-13_Core_SGM_publication.html")
        >>> net = from_jao(excel_file_path, html_file_path, True, drop_grid_groups_islands=True)
    """

    # --- read data
    data = pd.read_excel(excel_file_path, sheet_name=None, header=[0, 1])
    if html_file_path is not None:
        with open(html_file_path, mode="r", encoding=kwargs.get("encoding", "utf-8")) as f:
            html_str = f.read()
    else:
        html_str = ""

    # --- manipulate data / data corrections
    if apply_data_correction:
        html_str = _data_correction(data, html_str, max_i_ka_fillna)  # type: ignore[assignment]

    # --- parse html_str to line_geo_data
    line_geo_data = None
    if html_str:
        try:
            line_geo_data = _parse_html_str(html_str)
        except (json.JSONDecodeError, KeyError, AssertionError) as e:
            logger.error(f"html data were ignored due to this error:\n{e}")

    # --- create the pandapower net
    net = pandapowerNet(name=os.path.splitext(os.path.basename(excel_file_path))[0], sn_mva=kwargs.get("sn_mva", 1.0))
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


def _data_correction(data: dict[str, pd.DataFrame], html_str: str | None, max_i_ka_fillna: float) -> str | None:
    """
    Corrects input data in particular with regard to obvious weaknesses in the data provided,
    such as inconsistent spellings and missing necessary information

    :param dict[str, pd.DataFrame] data: data provided by the excel file which will be corrected
    :param str|None html_str: data provided by the html file which will be corrected
    :param float|int max_i_ka_fillna: value to fill missing values or data of false type in max_i_ka of lines and transformers.
        If no value should be set, you can also pass np.nan.

    :return: corrected html_str
    :rtype: str
    """
    # --- Line and Tieline data ---------------------------
    for key in ["Lines", "Tielines"]:
        # --- correct column names
        cols = data[key].columns.to_frame().reset_index(drop=True)
        cols.loc[cols[1] == "Voltage_level(kV)", 0] = None
        cols.loc[cols[1] == "Comment", 0] = None
        cols.loc[cols[0].str.startswith("Unnamed:").astype(bool), 0] = None
        cols.loc[cols[1] == "Length_(km)", 0] = "Electrical Parameters"  # might be wrong in
        # Tielines otherwise
        data[key].columns = pd.MultiIndex.from_arrays(cols.values.T.tolist())

        # --- correct comma separation and cast to floats
        data[key][("Maximum Current Imax (A)", "Fixed")] = (
            data[key][("Maximum Current Imax (A)", "Fixed")]
            .replace("\xa0", max_i_ka_fillna * 1e3)
            .replace("-", max_i_ka_fillna * 1e3)
            .replace(" ", max_i_ka_fillna * 1e3)
            .infer_objects(copy=False)
        )
        col_names = [
            ("Electrical Parameters", col_level1)
            for col_level1 in ["Length_(km)", "Resistance_R(Ω)", "Reactance_X(Ω)", "Susceptance_B(μS)", "Length_(km)"]
        ] + [("Maximum Current Imax (A)", "Fixed")]
        _float_col_comma_correction(data, key, col_names)

    # --- repair incomplete line data
    _repair_incomplete_line_data(data)

    # --- consolidate inconsistent location name spellings
    # Build the raw -> display name mapping from the bus names in the Lines and Tielines sheets.
    # Names that only differ in case, spacing, punctuation, accents or special characters are
    # merged onto a single display spelling.
    bus_name_series: list[pd.Series] = []
    for key in ["Lines", "Tielines"]:
        for subst in ["Substation_1", "Substation_2"]:
            bus_name_series.append(data[key][(subst, "Full_name")])
    bus_names: pd.Series = pd.concat(bus_name_series, ignore_index=True)
    locname_mapping = _locname_display_names(bus_names)
    rename_locnames = sorted(
        ((raw, disp) for raw, disp in locname_mapping.items() if raw != disp), key=lambda x: len(x[0]), reverse=True
    )

    for key in ["Lines", "Tielines"]:
        data[key][(None, "NE_name")] = _replace_substrings(data[key][(None, "NE_name")], rename_locnames)
        for loc_name in [("Substation_1", "Full_name"), ("Substation_2", "Full_name")]:
            data[key][loc_name] = _replace_locnames(data[key][loc_name], locname_mapping)
    html_str = _multi_str_repl(html_str, rename_locnames)  # type: ignore[arg-type]

    # --- Transformer data --------------------------------
    key = "Transformers"

    # --- correct current data (some files hold non-breaking spaces) and drop rows without any
    # --- electrical data, which cannot be converted
    current_cols = [("Maximum Current Imax (A) primary", col) for col in ("Min", "Max", "Fixed")]
    for col in current_cols:
        data[key][col] = pd.to_numeric(data[key][col].replace("\xa0", np.nan), errors="coerce")
    vn_cols = [("Voltage_level(kV)", "Primary"), ("Voltage_level(kV)", "Secondary")]
    no_data = data[key][vn_cols].isna().all(axis=1) & data[key][current_cols].isna().all(axis=1)
    if no_data.any():
        logger.warning(
            f"Transformers: dropping {no_data.sum()} row(s) without electrical data: {list(data[key].index[no_data])}"
        )
        data[key] = data[key].drop(index=data[key].index[no_data])

    # --- fix Locations
    loc_name = ("Location", "Full Name")
    data[key][loc_name] = _replace_substrings(data[key][loc_name], rename_locnames)

    # --- fix data in nonnull_taps
    taps = (
        data[key].loc[:, ("Phase Shifting Properties", "Taps used for RAO")].fillna("").astype(str).str.replace(" ", "")
    )
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
    cols = ["Phase Regulation δu (%)", "Angle Regulation δu (%)"]  # type: ignore[assignment]
    for col in cols:
        if is_object_dtype(data[key].loc[:, ("Phase Shifting Properties", col)]):
            tr_double = data[key].index[
                data[key].loc[:, ("Phase Shifting Properties", col)].str.contains("/").fillna(0).astype(bool)
            ]
            data[key].loc[tr_double, ("Phase Shifting Properties", col)] = (
                data[key]
                .loc[tr_double, ("Phase Shifting Properties", col)]
                .str.split("/", expand=True)[1]
                .str.replace(",", ".")
                .astype(float)
                .values
            )  # take second info and correct
            # separation: , -> .

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
        if pos0 < 0:
            raise AssertionError("pos0 < 0")
        if pos1 < len(name_start):
            raise AssertionError("pos1 < len(name_start)")
        return st[pos0:pos1]

    json_start_str = '<script type="application/json" data-for="htmlwidget-216030e6806f328c00fb">'
    json_start_pos = html_str.find(json_start_str) + len(json_start_str)
    json_end_pos = html_str[json_start_pos:].find("</script>")
    json_str = html_str[json_start_pos : (json_start_pos + json_end_pos)]
    geo_data = json.loads(json_str)
    geo_data = geo_data["x"]["calls"]
    methods_pos = pd.Series({item["method"]: i for i, item in enumerate(geo_data)})
    polylines = geo_data[methods_pos.at["addPolylines"]]["args"]
    EIC_start = "EIC Code:<b> "
    if len(polylines[6]) != len(polylines[0]):
        raise AssertionError("The lists of EIC Code data and geo data are not of the same length.")
    line_EIC = [polylines[6][i][polylines[6][i].find(EIC_start) + len(EIC_start) :] for i in range(len(polylines[6]))]
    line_name = [_filter_name(polylines[6][i]) for i in range(len(polylines[6]))]
    line_geo_data = pd.concat(
        [_lng_lat_to_df(polylines[0][i][0][0], line_EIC[i], line_name[i]) for i in range(len(polylines[0]))],
        ignore_index=True,
    )

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
    bus_df_empty = pd.DataFrame({"name": "", "vn_kv": 0.0, "TSO": ""}, index=[])
    bus_df = deepcopy(bus_df_empty)
    for key in ["Lines", "Tielines"]:
        for subst in ["Substation_1", "Substation_2"]:
            data_col_tuples = [(subst, "Full_name"), (None, "Voltage_level(kV)"), (None, "TSO")]
            to_add = data[key].loc[:, data_col_tuples].set_axis(bus_df.columns, axis="columns")
            if len(bus_df):
                bus_df = pd.concat([bus_df, to_add])
            else:
                bus_df = to_add
    bus_df = _drop_duplicates_and_join_TSO(bus_df)
    new_bus_idx = create_buses(net, len(bus_df), vn_kv=bus_df.vn_kv, name=bus_df.name, zone=bus_df.TSO)
    if any(new_bus_idx != bus_df.index):  # type: ignore[arg-type]
        raise AssertionError("Not all values of now_bus_idx are identical to bus_df.index")


def _create_lines(net: pandapowerNet, data: dict[str, pd.DataFrame], max_i_ka_fillna: float) -> None:
    """
    Creates lines to the pandapower net using information from the lines and tielines sheets (excel file).

    :param pandapowerNet net: net to be filled by buses
    :param dict[str, pd.DataFrame] data: data provided by the excel file which will be corrected
    :param float|int max_i_ka_fillna: value to fill missing values or data of false type in max_i_ka of lines and transformers.
        If no value should be set, you can also pass np.nan.
    """

    bus_idx = _get_bus_idx(net)

    for key in ["Lines", "Tielines"]:
        length_km = data[key][("Electrical Parameters", "Length_(km)")].to_numpy(dtype=float, copy=True)
        zero_length = np.isclose(length_km, 0)
        no_length = np.isnan(length_km)
        if sum(zero_length) or sum(no_length):  # type: ignore[call-overload]
            logger.warning(
                f"According to given data, {sum(zero_length)} {key.lower()} have zero "  # type: ignore[call-overload]
                f"length and {sum(zero_length)} {key.lower()} have no length data. "  # type: ignore[call-overload]
                "Both types of wrong data are replaced by 1 km."
            )
            length_km[zero_length | no_length] = 1
        vn_kvs = data[key].loc[:, (None, "Voltage_level(kV)")].values
        keys_1 = list(zip(data[key].loc[:, ("Substation_1", "Full_name")].values, vn_kvs))  # type: ignore[arg-type]
        keys_2 = list(zip(data[key].loc[:, ("Substation_2", "Full_name")].values, vn_kvs))  # type: ignore[arg-type]
        _raise_for_missing_buses(bus_idx, keys_1 + keys_2, key)

        _ = create_lines_from_parameters(
            net,
            bus_idx.loc[keys_1].values,
            bus_idx.loc[keys_2].values,
            length_km,
            data[key][("Electrical Parameters", "Resistance_R(Ω)")].values / length_km,  # type: ignore[operator]
            data[key][("Electrical Parameters", "Reactance_X(Ω)")].values / length_km,  # type: ignore[operator]
            data[key][("Electrical Parameters", "Susceptance_B(μS)")].values / length_km,  # type: ignore[operator]
            data[key][("Maximum Current Imax (A)", "Fixed")].fillna(max_i_ka_fillna * 1e3).values / 1e3,  # type: ignore[operator]
            name=data[key].xs("NE_name", level=1, axis=1).values[:, 0],  # type: ignore[call-overload]
            EIC_Code=data[key].xs("EIC_Code", level=1, axis=1).values[:, 0],  # type: ignore[call-overload]
            TSO=data[key].xs("TSO", level=1, axis=1).values[:, 0],  # type: ignore[call-overload]
            Comment=data[key].xs("Comment", level=1, axis=1).values[:, 0],  # type: ignore[call-overload]
            Tieline=key == "Tielines",
        )


def _create_transformers_and_buses(net: pandapowerNet, data: dict[str, pd.DataFrame], **kwargs) -> None:
    """
    Creates transformers to the pandapower net using information from the transformers sheet (excel file).

    :param pandapowerNet net: net to be filled by buses
    :param dict[str, pd.DataFrame] data: data provided by the excel file which will be corrected
    """

    # --- data preparations
    key = "Transformers"
    bus_idx = _get_bus_idx(net)
    vn_hv_kv, vn_lv_kv = _get_transformer_voltages(data, bus_idx)
    trafo_connections = _allocate_trafos_to_buses_and_create_buses(net, data, bus_idx, vn_hv_kv, vn_lv_kv, **kwargs)
    max_i_a = (
        data[key]
        .loc[:, ("Maximum Current Imax (A) primary", "Fixed")]
        .fillna(data[key].loc[:, ("Maximum Current Imax (A) primary", "Max")])
    )
    sn_mva = np.sqrt(3) * max_i_a * vn_hv_kv / 1e3
    z_pu = vn_hv_kv**2 / sn_mva
    rk = data[key].xs("Resistance_R(Ω)", level=1, axis=1).values[:, 0] / z_pu
    xk = data[key].xs("Reactance_X(Ω)", level=1, axis=1).values[:, 0] / z_pu
    b0 = data[key].xs("Susceptance_B (µS)", level=1, axis=1).values[:, 0] * 1e-6 * z_pu  # type: ignore[operator]
    g0 = data[key].xs("Conductance_G (µS)", level=1, axis=1).values[:, 0] * 1e-6 * z_pu  # type: ignore[operator]
    zk = np.sqrt(rk**2 + xk**2)
    vk_percent = np.sign(xk) * zk * 100
    vkr_percent = rk * 100
    pfe_kw = g0 * sn_mva * 1e3
    i0_percent = 100 * np.sqrt(b0**2 + g0**2) * net.sn_mva / sn_mva
    taps = (
        data[key]
        .loc[:, ("Phase Shifting Properties", "Taps used for RAO")]
        .str.split(";", expand=True)
        .astype(int)
        .set_axis(["tap_min", "tap_max"], axis=1)
    )

    du = _get_float_column(data[key], ("Phase Shifting Properties", "Phase Regulation δu (%)"))
    dphi = _get_float_column(data[key], ("Phase Shifting Properties", "Angle Regulation δu (%)"))
    phase_shifter = np.isclose(du, 0) & (~np.isclose(dphi, 0))  # Symmetrical/Asymmetrical not
    # considered

    _ = create_transformers_from_parameters(
        net,
        trafo_connections.hv_bus.values,  # type: ignore[arg-type]
        trafo_connections.lv_bus.values,  # type: ignore[arg-type]
        sn_mva,
        vn_hv_kv,
        vn_lv_kv,
        vkr_percent,
        vk_percent,
        pfe_kw,
        i0_percent,
        shift_degree=data[key].xs("Theta θ (°)", level=1, axis=1).values[:, 0],  # type: ignore[call-overload]
        tap_pos=0,
        tap_neutral=0,
        tap_side="lv",
        tap_min=taps["tap_min"].values,
        tap_max=taps["tap_max"].values,
        tap_phase_shifter=phase_shifter,
        tap_step_percent=du,
        tap_step_degree=dphi,
        name=data[key].loc[:, ("Location", "Full Name")].str.strip().values,
        EIC_Code=data[key].xs("EIC_Code", level=1, axis=1).values[:, 0],  # type: ignore[call-overload]
        TSO=data[key].xs("TSO", level=1, axis=1).values[:, 0],  # type: ignore[call-overload]
        Comment=data[key].xs("Comment", level=1, axis=1).replace("\xa0", "").values[:, 0],  # type: ignore[call-overload]
    )


def _invent_connections_between_grid_groups(
    net: pandapowerNet, minimal_trafo_invention: bool = False, **kwargs
) -> None:
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
    bus_grid_groups = pd.concat(
        [pd.Series(group, index=buses) for group, buses in zip(grid_groups.index, grid_groups.buses)]
    ).sort_index()

    # treat for example "Wuergau" equally as "Wuergau (2)":
    location_names = pd.Series(bus_idx.index.get_level_values(0))
    location_names = location_names.str.replace(r"(.) \([0-9]+\)", r"\1", regex=True)
    bus_idx.index = pd.MultiIndex.from_arrays(
        [location_names.values, bus_idx.index.get_level_values(1).to_numpy()], names=bus_idx.index.names
    )

    # --- add Transformers between equally named buses that have different voltage level and lay in
    # --- different groups
    connected_vn_kvs_by_trafos = (
        pd.DataFrame(
            {
                "hv": net.bus.vn_kv.loc[net.trafo.hv_bus].to_list(),
                "lv": net.bus.vn_kv.loc[net.trafo.lv_bus].to_list(),
                "index": net.trafo.index,
            }
        )
        .set_index(["hv", "lv"])
        .sort_index()
    )
    dupl_location_names = location_names[location_names.duplicated()]

    for location_name in dupl_location_names:
        if minimal_trafo_invention and len(bus_grid_groups.unique()) <= 1:
            break  # break with regard to minimal_trafo_invention
        grid_groups_at_location = bus_grid_groups.loc[bus_idx.loc[location_name].values]
        grid_groups_at_location = grid_groups_at_location.drop_duplicates()
        if len(grid_groups_at_location) < 2:
            continue
        elif len(grid_groups_at_location) > 2:
            raise NotImplementedError(
                "Code is not provided to invent Transformer connections "
                "between locations with more than two grid groups, i.e. "
                "voltage levels."
            )
        TSO = net.bus.zone.at[grid_groups_at_location.index[0]]
        vn_kvs = net.bus.vn_kv.loc[grid_groups_at_location.index].sort_values(ascending=False)
        try:
            trafos_connecting_same_voltage_levels = connected_vn_kvs_by_trafos.loc[tuple(vn_kvs)]
        except KeyError:
            logger.info(
                f"For location {location_name}, no transformer data can be reused since "
                f"no transformer connects {vn_kvs.sort_values(ascending=False).iat[0]} kV "
                f"and {vn_kvs.sort_values(ascending=False).iat[1]} kV."
            )
            continue
        trafos_of_same_TSO = trafos_connecting_same_voltage_levels.loc[
            (
                net.bus.zone.loc[net.trafo.hv_bus.loc[trafos_connecting_same_voltage_levels.values.flatten()].values]
                == TSO
            ).values
        ].values.flatten()

        # from which trafo parameters are copied:
        tr_to_be_copied = (
            trafos_of_same_TSO[0]
            if len(trafos_of_same_TSO)
            else trafos_connecting_same_voltage_levels.values.flatten()[0]
        )

        # copy transformer data
        duplicated_row = net.trafo.loc[[tr_to_be_copied]].copy()
        duplicated_row.index = [net.trafo.index.max() + 1]  # adjust index
        duplicated_row.hv_bus = vn_kvs.index[0]  # type: ignore[attr-defined] # adjust hv_bus, lv_bus
        duplicated_row.lv_bus = vn_kvs.index[1]  # type: ignore[attr-defined] # adjust hv_bus, lv_bus
        duplicated_row.name = "additional transformer to connect the grid"  # type: ignore[attr-defined]
        net.trafo = pd.concat([net.trafo, duplicated_row])

        bus_grid_groups.loc[bus_grid_groups == grid_groups_at_location.iat[1]] = grid_groups_at_location.iat[0]

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

        bus_grid_groups.loc[bus_grid_groups.isin(bus_grid_groups.drop(idx).loc[is_fuse_candidate].unique())] = (
            grid_groups_at_location.iat[0]
        )
        bus_grid_groups = bus_grid_groups.drop(to_fuse.index)

    # --- fuse buses that are close to each other
    for name1, name2 in [
        ("CROISIERE", "BOLLENE (POSTE RESEAU)"),
        ("CAEN", "DRONNIERE (LA)"),
        ("TRINITE-VICTOR", "MENTON/TRINITE VICTOR"),
    ]:
        b1 = net.bus.index[net.bus.name == name1]
        b2 = net.bus.index[net.bus.name == name2]
        if len(b1) == 1 and len(b2) >= 1:
            fuse_buses(net, b1[0], set(b2))
            bus_grid_groups = bus_grid_groups.drop(b2)
        else:
            logger.info(
                f"Buses of the following names were intended to be fused but were not found.\n'{name1}' and '{name2}'"
            )


def drop_islanded_grid_groups(net: pandapowerNet, min_bus_number: int | str, **kwargs) -> None:
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
        slack_buses = set(net.ext_grid.loc[net.ext_grid.in_service, "bus"]) | set(
            net.gen.loc[net.gen.in_service & net.gen.slack, "bus"]
        )
        grid_groups_to_drop = grid_groups.loc[~grid_groups.buses.apply(lambda x: not x.isdisjoint(slack_buses))]

    elif min_bus_number == "max":
        min_bus_number = grid_groups["n_buses"].max()
        grid_groups_to_drop = _grid_groups_to_drop_by_min_bus_number()

    elif isinstance(min_bus_number, int):
        grid_groups_to_drop = _grid_groups_to_drop_by_min_bus_number()

    else:
        raise NotImplementedError(f"{min_bus_number=} is not implemented. Use an int, 'max', or 'unsupplied' instead.")

    buses_to_drop = reduce(set.union, grid_groups_to_drop.buses)
    drop_buses(net, buses_to_drop)
    logger.info(
        f"drop_islanded_grid_groups() drops {len(grid_groups_to_drop)} grid groups with a "
        f"total of {grid_groups_to_drop.n_buses.sum()} buses."
    )


def _add_bus_geo(net: pandapowerNet, line_geo_data: pd.DataFrame) -> None:
    """
    Adds geodata to the buses. The function needs to handle cases where line_geo_data includes multiple or no geodata
    per bus. Primarily, the geodata are allocate via EIC Code names, if ambiguous, names are considered.

    Parameters:
        net: net in which geodata are added to the buses
        line_geo_data: Converted geodata from the html file
    """
    iSl = pd.IndexSlice
    lgd_EIC_bus = line_geo_data.pivot_table(values="value", index=["EIC_Code", "bus"], columns="geo_dim")
    lgd_name_bus = line_geo_data.pivot_table(values="value", index=["name", "bus"], columns="geo_dim")
    lgd_EIC_bus_idx_extended = pd.MultiIndex.from_frame(
        lgd_EIC_bus.index.to_frame()
        .assign(col_name="EIC_Code")
        .rename(columns={"EIC_Code": "identifier"})
        .loc[:, ["col_name", "identifier", "bus"]]
    )
    lgd_name_bus_idx_extended = pd.MultiIndex.from_frame(
        lgd_name_bus.index.to_frame()
        .assign(col_name="name")
        .rename(columns={"name": "identifier"})
        .loc[:, ["col_name", "identifier", "bus"]]
    )
    lgd_bus = pd.concat(
        [lgd_EIC_bus.set_axis(lgd_EIC_bus_idx_extended), lgd_name_bus.set_axis(lgd_name_bus_idx_extended)]
    )
    dupl_EICs = net.line.EIC_Code.loc[net.line.EIC_Code.duplicated()]
    dupl_names = net.line.name.loc[net.line.name.duplicated()]

    def _geo_json_str(this_bus_geo: pd.Series) -> str:
        return f'{{"coordinates": [{this_bus_geo.at["lng"]}, {this_bus_geo.at["lat"]}], "type": "Point"}}'

    def _add_bus_geo_inner(bus: int) -> str | None:
        from_bus_line_excerpt = net.line.loc[net.line.from_bus == bus, ["EIC_Code", "name", "Tieline"]]
        to_bus_line_excerpt = net.line.loc[net.line.to_bus == bus, ["EIC_Code", "name", "Tieline"]]

        line_excerpt = pd.concat([from_bus_line_excerpt, to_bus_line_excerpt])
        n_connected_line_ends = len(line_excerpt)
        if n_connected_line_ends == 0:
            logger.error(f"Bus {bus} (name {net.bus.at[bus, 'name']}) is not found in line_geo_data.")
            return None
        is_dupl = pd.concat(
            [
                pd.DataFrame(
                    {
                        "EIC": from_bus_line_excerpt.EIC_Code.isin(dupl_EICs).values,
                        "name": from_bus_line_excerpt.name.isin(dupl_names).values,
                    },
                    index=pd.MultiIndex.from_product(
                        [["from"], from_bus_line_excerpt.index], names=["bus", "line_index"]
                    ),
                ),
                pd.DataFrame(
                    {
                        "EIC": to_bus_line_excerpt.EIC_Code.isin(dupl_EICs).values,
                        "name": to_bus_line_excerpt.name.isin(dupl_names).values,
                    },
                    index=pd.MultiIndex.from_product([["to"], to_bus_line_excerpt.index], names=["bus", "line_index"]),
                ),
            ]
        )
        is_missing = pd.DataFrame(
            {
                "EIC": ~line_excerpt.EIC_Code.isin(lgd_bus.loc["EIC_Code"].index.get_level_values("identifier")),
                "name": ~line_excerpt.name.isin(lgd_bus.loc["name"].index.get_level_values("identifier")),
            }
        ).set_axis(is_dupl.index, axis=0)
        is_tieline = pd.Series(
            net.line.loc[is_dupl.index.get_level_values("line_index"), "Tieline"].values, index=is_dupl.index
        )

        # --- construct access_vals, i.e. values to take line geo data from lgd_bus
        # --- if not duplicated, take "EIC_Code". Otherwise and if not dupl, take "name".
        # --- Otherwise ignore. Do it for both from and to bus
        access_vals = pd.DataFrame(
            {
                "col_name": "EIC_Code",
                "identifier": line_excerpt.EIC_Code.values,
                "bus": is_dupl.index.get_level_values("bus").values,
            }
        )  # default is EIC_Code
        take_from_name = ((is_dupl.EIC | is_missing.EIC) & (~is_dupl.name & ~is_missing.name)).values
        access_vals.loc[take_from_name, "col_name"] = "name"
        access_vals.loc[take_from_name, "identifier"] = line_excerpt.name.loc[take_from_name].values  # type: ignore[index]
        keep = (~(is_dupl | is_missing)).any(axis=1).values
        if np.all(is_missing):
            log_msg = (
                f"For bus {bus} (name {net.bus.at[bus, 'name']}), {n_connected_line_ends} "
                "were found but no EIC_Codes or names of corresponding lines were found ."
                "in the geo data from the html file."
            )
            if is_tieline.all():
                logger.debug(log_msg)
            else:
                logger.warning(log_msg)
            return None
        elif sum(keep) == 0:
            logger.info(
                f"For {bus=}, all EIC_Codes and names of connected lines are ambiguous. "
                "No geo data is dropped at this point."
            )
            keep[(~is_missing).any(axis=1)] = True  # type: ignore[index]
        access_vals = access_vals.loc[keep]  # type: ignore[index]

        # --- get this_bus_geo from EIC_Code or name with regard to access_vals
        this_bus_geo = lgd_bus.loc[iSl[access_vals.col_name, access_vals.identifier, access_vals.bus], :]

        if len(this_bus_geo) > 1:
            # reduce similar/equal lines
            this_bus_geo = this_bus_geo.loc[this_bus_geo.round(2).drop_duplicates().index]

        # --- return geo_json_str
        len_this_bus_geo = len(this_bus_geo)
        if len_this_bus_geo == 1:
            return _geo_json_str(this_bus_geo.iloc[0])
        elif len_this_bus_geo == 2:
            how_often: pd.Series[int] = pd.Series(
                [
                    int(
                        (
                            np.isclose(lgd_EIC_bus["lat"], this_bus_geo["lat"].iat[i])
                            & np.isclose(lgd_EIC_bus["lng"], this_bus_geo["lng"].iat[i])
                        ).sum()
                    )
                    for i in range(len_this_bus_geo)
                ],
                index=this_bus_geo.index,
                dtype=int,
            )
            if how_often.max() >= 1:
                logger.warning(
                    f"Bus {bus} (name {net.bus.at[bus, 'name']}) was found multiple times in line_geo_data. "
                    f"No geo positions was used more often than all other positions. "
                    "The first of the most used positions is used."
                )
            return _geo_json_str(this_bus_geo.loc[how_often.idxmax()].iloc[0])  # type: ignore[call-overload]

        return None

    net.bus["geo"] = [_add_bus_geo_inner(bus) for bus in net.bus.index]


# --- tertiary functions ---------------------------------------------------------------------------


def _float_col_comma_correction(data: dict[str, pd.DataFrame], key: str, col_names: list):
    for col_name in col_names:
        data[key][col_name] = pd.to_numeric(data[key][col_name].astype(str).str.replace(",", "."), errors="coerce")


def _repair_incomplete_line_data(data: dict[str, pd.DataFrame]) -> None:
    """Drops Lines/Tielines rows without any data and fills missing substation names and voltage
    levels from unambiguous information of the same file.

    Some JAO releases contain malformed rows, e.g. rows that only repeat an NE_name without any
    substation or electrical data, lines with one missing substation name, or lines with a missing
    voltage level. Rows without any information are dropped, the other gaps are filled if the
    information is unambiguous.

    :param dict[str, pd.DataFrame] data: data provided by the excel file which will be corrected
    """
    s1, s2 = ("Substation_1", "Full_name"), ("Substation_2", "Full_name")
    vn_col, len_col, ne_col = (None, "Voltage_level(kV)"), ("Electrical Parameters", "Length_(km)"), (None, "NE_name")

    # --- names and voltage levels that are known from complete rows of all sheets
    known_names: set[str] = set()
    name_vns: dict[str, set] = {}
    for key in ("Lines", "Tielines"):
        for subst in (s1, s2):
            pairs = data[key][[subst, vn_col]].dropna()
            known_names |= set(pairs[subst].astype(str))
            for name, vn in pairs.itertuples(index=False):
                name_vns.setdefault(name, set()).add(vn)

    for key in ("Lines", "Tielines"):
        df = data[key].copy()

        empty = df[[s1, s2, vn_col, len_col]].isna().all(axis=1)
        if empty.any():
            logger.warning(f"{key}: dropping {empty.sum()} row(s) without any data: {list(df.index[empty])}")
            df = df.drop(index=df.index[empty])

        for idx in df.index[df[s1].isna() | df[s2].isna()]:
            missing = s1 if pd.isna(df.at[idx, s1]) else s2
            other = df.at[idx, s2] if missing == s1 else df.at[idx, s1]
            tokens = [token.strip().upper() for token in re.sub(r"\s*\(.*?\)", "", str(df.at[idx, ne_col])).split("-")]
            candidates = [
                name
                for name in known_names
                if name != other
                and any(token == name.upper() or token.startswith(name.upper() + " ") for token in tokens)
            ]
            if len(candidates) == 1:
                df.at[idx, missing] = candidates[0]
                logger.warning(
                    f"{key} row {idx}: missing {missing[0]} name set to {candidates[0]!r} "
                    f"(derived from NE_name {df.at[idx, ne_col]!r})"
                )

        for idx in df.index[df[vn_col].isna()]:
            vns = {
                next(iter(name_vns[name]))
                for name in (df.at[idx, s1], df.at[idx, s2])
                if len(name_vns.get(name, ())) == 1
            }
            if len(vns) == 1:
                df.at[idx, vn_col] = vns.pop()
                logger.warning(f"{key} row {idx}: missing Voltage_level(kV) set to {df.at[idx, vn_col]}")

        data[key] = df


def _get_transformer_voltages(data: dict[str, pd.DataFrame], bus_idx: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    key = "Transformers"
    vn = data[key].loc[:, [("Voltage_level(kV)", "Primary"), ("Voltage_level(kV)", "Secondary")]].values
    vn_hv_kv = np.max(vn, axis=1)
    vn_lv_kv = np.min(vn, axis=1)
    if is_integer_dtype(list(bus_idx.index.dtypes)[1]):  # type: ignore[attr-defined]
        vn_hv_kv = vn_hv_kv.astype(int)
        vn_lv_kv = vn_lv_kv.astype(int)

    return vn_hv_kv, vn_lv_kv


def _allocate_trafos_to_buses_and_create_buses(
    net: pandapowerNet,
    data: dict[str, pd.DataFrame],
    bus_idx: pd.Series,
    vn_hv_kv: np.ndarray,
    vn_lv_kv: np.ndarray,
    rel_deviation_threshold_for_trafo_bus_creation: float = 0.2,
    log_rel_vn_deviation: float = 0.12,
    **kwargs,
) -> pd.DataFrame:
    """Provides a DataFrame of data to allocate transformers to the buses according to their
    location names. If locations of transformers do not exist due to the data of the lines and
    tielines sheets, additional buses are created. If locations exist but have a far different
    voltage level than the transformer, either a warning is logged or additional buses are created
    according to rel_deviation_threshold_for_trafo_bus_creation and log_rel_vn_deviation.

    :param pandapowerNet net: pandapower net
    :param dict[str, pd.DataFrame] data: _description_
    :param pd.Series bus_idx: Series of indices and corresponding location names and voltage levels in the MultiIndex of
        the Series
    :param np.ndarray vn_hv_kv: nominal voltages of the hv side of the transformers
    :param np.ndarray vn_lv_kv: Nominal voltages of the lv side of the transformers
    :param Optional[float] rel_deviation_threshold_for_trafo_bus_creation: If the voltage level of transformer locations
        is far different than the transformer data,
        additional buses are created. rel_deviation_threshold_for_trafo_bus_creation defines the
        tolerance in which no additional buses are created. (default: 0.2)
    :param Optional[float] log_rel_vn_deviation: This parameter allows a range below rel_deviation_threshold_for_trafo_bus_creation in which
        a warning is logged instead of a creating additional buses. (default: 0.12)

    :rtype: pd.DataFrame
    :return: information to which bus the trafos should be connected to. Columns are
        ["name", "hv_bus", "lv_bus", "vn_hv_kv", "vn_lv_kv", ...]
    """

    if rel_deviation_threshold_for_trafo_bus_creation < log_rel_vn_deviation:
        logger.warning(
            f"Given parameters violates the ineqation "
            f"{rel_deviation_threshold_for_trafo_bus_creation=} >= {log_rel_vn_deviation=}. "
            f"Therefore, rel_deviation_threshold_for_trafo_bus_creation={log_rel_vn_deviation} "
            "is assumed."
        )
        rel_deviation_threshold_for_trafo_bus_creation = log_rel_vn_deviation

    key = "Transformers"
    bus_location_names = set(net.bus.name)
    trafo_bus_names = data[key].loc[:, ("Location", "Full Name")]
    trafo_location_names = _find_trafo_locations(trafo_bus_names, bus_location_names)

    # --- construct DataFrame trafo_connections including all information on trafo allocation to
    # --- buses
    empties: np.typing.NDArray[np.int_] = -1 * np.ones(len(vn_hv_kv), dtype=int)
    trafo_connections = pd.DataFrame(
        {
            "name": trafo_location_names,
            "hv_bus": empties,
            "lv_bus": empties,
            "vn_hv_kv": vn_hv_kv,
            "vn_lv_kv": vn_lv_kv,
            "vn_hv_kv_next_bus": vn_hv_kv,
            "vn_lv_kv_next_bus": vn_lv_kv,
            "hv_rel_deviation": np.zeros(len(vn_hv_kv)),
            "lv_rel_deviation": np.zeros(len(vn_hv_kv)),
        }
    )
    trafo_connections[["hv_bus", "lv_bus"]] = trafo_connections[["hv_bus", "lv_bus"]].astype(np.int64)

    for side in ["hv", "lv"]:
        bus_col, trafo_vn_col, next_col, rel_dev_col, has_dev_col = (
            f"{side}_bus",
            f"vn_{side}_kv",
            f"vn_{side}_kv_next_bus",
            f"{side}_rel_deviation",
            f"trafo_{side}_to_bus_deviation",
        )
        name_vn_series = pd.Series(tuple(zip(trafo_location_names, trafo_connections[trafo_vn_col])))
        isin = name_vn_series.isin(bus_idx.index).to_numpy()
        has_name = trafo_location_names.notna().to_numpy()
        trafo_connections[has_dev_col] = ~isin
        trafo_connections.loc[isin, bus_col] = bus_idx.loc[name_vn_series.loc[isin]].values

        # --- unresolved locations: a new bus at the trafo's rated voltage is created later
        unresolved = ~has_name
        trafo_connections.loc[unresolved, next_col] = trafo_connections.loc[unresolved, trafo_vn_col].values
        trafo_connections.loc[unresolved, rel_dev_col] = np.inf

        # --- resolved but no exact (name, vn) match: code to find bus locations with vn deviation
        need_nearest = ~isin & has_name
        next_vn = np.array(
            [
                bus_idx.loc[tln.name].index.values[
                    (pd.Series(bus_idx.loc[tln.name].index) - getattr(tln, trafo_vn_col)).abs().idxmin()
                ]
                for tln in trafo_connections.loc[need_nearest, ["name", trafo_vn_col]].itertuples()
            ]
        )
        trafo_connections.loc[need_nearest, next_col] = next_vn
        rel_dev = np.abs(next_vn - trafo_connections.loc[need_nearest, trafo_vn_col].values) / next_vn
        trafo_connections.loc[need_nearest, rel_dev_col] = rel_dev
        trafo_connections.loc[need_nearest, bus_col] = bus_idx.loc[
            list(zip(trafo_connections.loc[need_nearest, "name"], trafo_connections.loc[need_nearest, next_col]))
        ].values

        # --- create buses to avoid too large vn deviations between nodes and transformers
        need_bus_creation = trafo_connections[rel_dev_col] > rel_deviation_threshold_for_trafo_bus_creation
        fallback_names = pd.Series(trafo_bus_names.astype(str).str.strip().to_numpy(), index=trafo_connections.index)
        new_bus_data = pd.DataFrame(
            {
                "vn_kv": trafo_connections.loc[need_bus_creation, trafo_vn_col].values,
                "name": trafo_connections.loc[need_bus_creation, "name"].fillna(fallback_names).values,
                "TSO": data[key].loc[need_bus_creation, ("Location", "TSO")].values,
            }
        )
        new_bus_data_dd = _drop_duplicates_and_join_TSO(new_bus_data)
        new_bus_idx = create_buses(
            net, len(new_bus_data_dd), vn_kv=new_bus_data_dd.vn_kv, name=new_bus_data_dd.name, zone=new_bus_data_dd.TSO
        )
        trafo_connections.loc[need_bus_creation, bus_col] = (
            net.bus.loc[new_bus_idx, ["name", "vn_kv"]]
            .reset_index()
            .set_index(["name", "vn_kv"])
            .loc[list(new_bus_data[["name", "vn_kv"]].itertuples(index=False, name=None))]
            .values
        )
        trafo_connections.loc[need_bus_creation, next_col] = trafo_connections.loc[
            need_bus_creation, trafo_vn_col
        ].values
        trafo_connections.loc[need_bus_creation, rel_dev_col] = 0
        trafo_connections.loc[need_bus_creation, has_dev_col] = False

    # --- create buses for trafos that are connected to the same bus at both sides (possible if
    # --- vn_hv_kv < vn_lv_kv *(1+rel_deviation_threshold_for_trafo_bus_creation) which usually
    # --- occurs for PSTs only)
    same_bus_connection = trafo_connections.hv_bus == trafo_connections.lv_bus
    duplicated_buses = net.bus.loc[trafo_connections.loc[same_bus_connection, "lv_bus"]].copy()
    duplicated_buses["name"] += " (2)"
    duplicated_buses.index = list(range(net.bus.index.max() + 1, net.bus.index.max() + 1 + len(duplicated_buses)))
    if same_bus_connection.any():
        trafo_connections.loc[same_bus_connection, "lv_bus"] = duplicated_buses.index
    net.bus = pd.concat([net.bus, duplicated_buses])
    if n_add_buses := len(duplicated_buses):
        tr_names = data[key].loc[trafo_connections.index[same_bus_connection], ("Location", "Full Name")]
        are_PSTs = tr_names.str.contains("PST")
        logger.info(
            f"{n_add_buses} additional buses were created to avoid that transformers are "
            f"connected to the same bus at both side, hv and lv. Of the causing "
            f"{len(tr_names)} transformers, {sum(are_PSTs)} contain 'PST' in their name. "
            f"According to this converter, the power flows over all these transformers will"
            f" end at the additional buses. Please consider to connect lines with the "
            f"additional buses, so that the power flow is over the (PST) transformers into "
            f"the lines."
        )

    # --- log according to log_rel_vn_deviation
    for side in ["hv", "lv"]:
        need_logging = trafo_connections.loc[trafo_connections[has_dev_col], rel_dev_col] > log_rel_vn_deviation
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
                f"nevertheless applied, due to {rel_deviation_threshold_for_trafo_bus_creation=}"
            )

    if (
        not (trafo_connections.hv_bus > -1).all()
        or not (trafo_connections.lv_bus > -1).all()
        or not (trafo_connections.hv_bus != trafo_connections.lv_bus).all()
    ):
        raise AssertionError("A trafo")

    return trafo_connections


def _find_trafo_locations(trafo_bus_names, bus_location_names):
    """Identifies the bus location name for each transformer from its "Location / Full Name".

    The transformer names are reduced to the impeding-part-free location (joined and longest part)
    and compared to the bus location names on normalized keys. Remaining names are resolved by
    fuzzy matching (only clear best matches) and, if the name contains a voltage level, by matching
    the name base with that voltage. Unresolved transformers are returned as ``None``; the caller
    then creates a new bus at the transformer's rated voltage.

    Returns a Series of bus location names (or ``None``) aligned with ``trafo_bus_names``.
    """
    bus_location_names = sorted(set(bus_location_names))
    bus_key_to_name: dict[str, str] = {}
    for name in bus_location_names:
        bus_key_to_name.setdefault(_normalize_locname(name), name)

    # --- split strings at " " separators to remove impeding parts for identifying the location
    # --- names
    trafo_bus_names_expended = (
        trafo_bus_names.str.split(r"[ ]+|-A[0-9]+|-TD[0-9]+|-PF[0-9]+", expand=True).fillna("").replace(" ", "")
    )
    trafo_bus_names_expended_lower = (
        trafo_bus_names.str.lower()
        .str.split(r"[ ]+|-A[0-9]+|-TD[0-9]+|-PF[0-9]+", expand=True)
        .fillna("")
        .replace(" ", "")
    )

    # --- identify impeding parts
    contains_number = trafo_bus_names_expended.map(lambda x: any(char.isdigit() for char in x))
    to_drop = (
        (trafo_bus_names_expended_lower == "tr")
        | (trafo_bus_names_expended_lower == "pst")
        | (trafo_bus_names_expended == "")
        | (trafo_bus_names_expended == "/")
        | (trafo_bus_names_expended == "LIPST")
        | (trafo_bus_names_expended == "EHPST")
        | (trafo_bus_names_expended == "TFO")
        | (trafo_bus_names_expended_lower == "trafo")
        | (trafo_bus_names_expended_lower == "kv")
        | contains_number
    )
    trafo_bus_names_expended[to_drop] = ""

    # --- reconstruct name strings for identification and compare on normalized keys
    trafo_bus_names_joined = trafo_bus_names_expended.where(~to_drop).fillna("").agg(" ".join, axis=1).str.strip()
    trafo_bus_names_longest_part = trafo_bus_names_expended.apply(lambda row: max(row, key=len), axis=1)
    joined_keys = trafo_bus_names_joined.map(_normalize_locname)
    longest_keys = trafo_bus_names_longest_part.map(_normalize_locname)
    joined_in_buses = joined_keys.isin(bus_key_to_name)
    longest_part_in_buses = longest_keys.isin(bus_key_to_name)

    # --- resolve glued noise prefixes, e.g. "PSTMIKULOWA" -> "MIKULOWA"
    def _strip_glued_prefix(part: str) -> str:
        stripped = re.sub(r"^(?:PST|TR|TFO)(?=[A-Za-z]{3,})", "", part)
        return stripped if stripped != part else ""

    unresolved = ~(joined_in_buses | longest_part_in_buses)
    resolved_from_glued = pd.Series("", index=trafo_bus_names_longest_part.index)
    for idx in trafo_bus_names_longest_part.index[unresolved]:
        for part in [p for p in trafo_bus_names_expended.loc[idx] if p]:
            stripped_key = _normalize_locname(_strip_glued_prefix(part))
            if stripped_key in bus_key_to_name:
                resolved_from_glued.at[idx] = bus_key_to_name[stripped_key]
                break
    glued_in_buses = resolved_from_glued.astype(bool)

    # --- further attempts for still unresolved names: fuzzy matching and voltage-aware matching
    unresolved = ~(joined_in_buses | longest_part_in_buses | glued_in_buses)
    bus_by_base: dict[str, list[tuple[int | None, str]]] = {}
    for bus_key, bus_name in bus_key_to_name.items():
        base, voltage = _split_name_voltage(bus_key)
        bus_by_base.setdefault(base, []).append((voltage, bus_name))

    resolved_extra = pd.Series("", index=trafo_bus_names_longest_part.index)
    unresolved_names: list[str] = []
    if unresolved.any():
        bus_keys = list(bus_key_to_name)
        for idx in trafo_bus_names_longest_part.index[unresolved]:
            raw_name = trafo_bus_names.at[idx]
            search_key = longest_keys.at[idx] or _normalize_locname(raw_name)

            # --- fuzzy matching: accept only a clear best match
            scored = sorted(
                ((difflib.SequenceMatcher(None, search_key, key).ratio(), key) for key in bus_keys), reverse=True
            )
            best_ratio, best_key = scored[0]
            second_ratio = scored[1][0] if len(scored) > 1 else 0.0
            if best_ratio >= 0.85 and best_ratio - second_ratio >= 0.03:
                resolved_extra.at[idx] = bus_key_to_name[best_key]
                logger.info(
                    f"Transformer {raw_name!r} was matched to bus location {bus_key_to_name[best_key]!r} "
                    f"by fuzzy matching (similarity {best_ratio:.3f})."
                )
                continue

            # --- voltage-aware matching: same name base, bus voltage equals the voltage in the name
            voltage = _parse_voltage_kv(raw_name)
            if voltage is not None:
                base, _ = _split_name_voltage(search_key)
                candidates = sorted({name for v, name in bus_by_base.get(base, []) if v == voltage})
                if len(candidates) == 1:
                    resolved_extra.at[idx] = candidates[0]
                    logger.info(
                        f"Transformer {raw_name!r} was matched to bus location {candidates[0]!r} "
                        f"by name and voltage level ({voltage} kV)."
                    )
                    continue

            unresolved_names.append(raw_name)

    if unresolved_names:
        logger.warning(
            f"For {len(unresolved_names)} transformer(s), no bus location could be determined from the "
            f"transformer location names. A new bus at the transformer's rated voltage will be created for "
            f"each of them:\n" + "\n".join(f"  {name!r}" for name in unresolved_names)
        )

    # --- set the trafo location names (actual bus names) to be used for the bus allocation
    trafo_location_names = longest_keys.map(bus_key_to_name).where(longest_part_in_buses)
    trafo_location_names = trafo_location_names.where(~joined_in_buses, joined_keys.map(bus_key_to_name))
    trafo_location_names = trafo_location_names.where(~glued_in_buses, resolved_from_glued)
    trafo_location_names = trafo_location_names.where(~resolved_extra.astype(bool), resolved_extra)

    return trafo_location_names


def _drop_duplicates_and_join_TSO(bus_df: pd.DataFrame) -> pd.DataFrame:
    bus_df = bus_df.drop_duplicates(ignore_index=True)
    # just keep one bus per name and vn_kv. If there are multiple buses of different TSOs, join the
    # TSO strings:
    bus_df = bus_df.groupby(["name", "vn_kv"], as_index=False).agg({"TSO": lambda x: "/".join(x)})  # type: ignore[misc]
    if bus_df.duplicated(["name", "vn_kv"]).any():
        raise AssertionError("bus_df contains duplicate names with identical vn_kv")
    return bus_df


def _get_float_column(df, col_tuple, fill=0):
    series = df.loc[:, col_tuple].copy()
    series.loc[series == "\xa0"] = fill
    return series.astype(float).fillna(fill)


def _get_bus_idx(net: pandapowerNet) -> pd.Series:
    return net.bus[["name", "vn_kv"]].rename_axis("index").reset_index().set_index(["name", "vn_kv"])["index"]


def _raise_for_missing_buses(bus_idx: pd.Series, endpoints: list[tuple], sheet: str) -> None:
    """Raises a descriptive error if line endpoints do not match any bus (name, vn_kv)."""
    missing = sorted({endpoint for endpoint in endpoints if endpoint not in bus_idx.index}, key=str)
    if missing:
        raise ValueError(
            f"{len(missing)} endpoint(s) of sheet {sheet!r} have no matching bus (name, vn_kv): "
            + ", ".join(f"{name!r} at {vn} kV" for name, vn in missing)
            + ". Rows with missing substation name or voltage level cannot be converted."
        )


def get_grid_groups(net: pandapowerNet, **kwargs) -> pd.DataFrame:
    notravbuses_dict = {} if "notravbuses" not in kwargs else {"notravbuses": kwargs.pop("notravbuses")}
    grid_group_buses = [set_ for set_ in connected_components(create_nxgraph(net, **kwargs), **notravbuses_dict)]
    grid_groups = pd.DataFrame({"buses": grid_group_buses})
    grid_groups["n_buses"] = grid_groups["buses"].apply(len)
    return grid_groups


def _lng_lat_to_df(dict_: dict, line_EIC: str, line_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [line_EIC, line_name, "from", "lng", dict_["lng"][0]],
            [line_EIC, line_name, "to", "lng", dict_["lng"][1]],
            [line_EIC, line_name, "from", "lat", dict_["lat"][0]],
            [line_EIC, line_name, "to", "lat", dict_["lat"][1]],
        ],
        columns=["EIC_Code", "name", "bus", "geo_dim", "value"],
    )


def _fill_geo_at_one_sided_branches_without_geo_extent(net: pandapowerNet):
    def _check_geo_availablitiy(net: pandapowerNet) -> dict[str, pd.Index | int]:
        av: dict[str, pd.Index | int] = {}  # availablitiy of geodata
        av["bus_with_geo"] = net.bus.index[~net.bus.geo.isnull()]
        av["lines_fbw_tbwo"] = net.line.index[
            net.line.from_bus.isin(av["bus_with_geo"]) & (~net.line.to_bus.isin(av["bus_with_geo"]))  # type: ignore[arg-type]
        ]
        av["lines_fbwo_tbw"] = net.line.index[
            (~net.line.from_bus.isin(av["bus_with_geo"])) & net.line.to_bus.isin(av["bus_with_geo"])  # type: ignore[arg-type]
        ]
        av["trafos_hvbw_lvbwo"] = net.trafo.index[
            net.trafo.hv_bus.isin(av["bus_with_geo"]) & (~net.trafo.lv_bus.isin(av["bus_with_geo"]))  # type: ignore[arg-type]
        ]
        av["trafos_hvbwo_lvbw"] = net.trafo.index[
            (~net.trafo.hv_bus.isin(av["bus_with_geo"])) & net.trafo.lv_bus.isin(av["bus_with_geo"])  # type: ignore[arg-type]
        ]
        av["n_lines_one_side_geo"] = len(av["lines_fbw_tbwo"]) + len(av["lines_fbwo_tbw"])  # type: ignore[assignment,arg-type]
        return av

    geo_avail = _check_geo_availablitiy(net)
    while geo_avail["n_lines_one_side_geo"]:
        # copy available geodata to the other end of branches where geodata are missing
        for et, bus_w_geo, bus_wo_geo, idx_key in zip(
            ["line", "line", "trafo", "trafo"],
            ["to_bus", "from_bus", "lv_bus", "hv_bus"],
            ["from_bus", "to_bus", "hv_bus", "lv_bus"],
            ["lines_fbwo_tbw", "lines_fbw_tbwo", "trafos_hvbwo_lvbw", "trafos_hvbw_lvbwo"],
        ):
            net.bus.loc[net[et].loc[geo_avail[idx_key], bus_wo_geo].values, "geo"] = net.bus.loc[
                net[et].loc[geo_avail[idx_key], bus_w_geo].values, "geo"
            ].values
        geo_avail = _check_geo_availablitiy(net)

    set_line_geodata_from_bus_geodata(net)


def _normalize_locname(name: object) -> str:
    """Returns a comparison key for a location name.

    The key is insensitive to case, accents, spacing, punctuation and special characters
    (e.g. "Rohrsdorf" and "Röhrsdorf", "T HAL-BYC" and "T HAL-BYĆ", "BAULER" and "Bauler" all map
    to the same key).
    """
    if not isinstance(name, str):
        return ""
    st = name.replace("\xa0", " ").replace("\u00ad", "")
    st = st.replace("\ufb00", "ff").replace("\ufb01", "fi").replace("\ufb02", "fl")
    st = st.replace("\u00df", "ss").replace("\u1e9e", "SS")
    st = unicodedata.normalize("NFKD", st)
    st = "".join(char for char in st if not unicodedata.combining(char))
    return re.sub(r"[^0-9a-z]+", " ", st.casefold()).strip()


def _parse_voltage_kv(name: object) -> int | None:
    """Returns the voltage level (kV) explicitly written in a name, e.g. "TR Ens 380kV TR 404" -> 380.

    Only voltages in the range 60...800 kV are considered, so that transformer numbers such as "441"
    are not mistaken for a voltage level.
    """
    for match in re.findall(r"(\d{2,3})\s*kV", str(name), flags=re.IGNORECASE):
        if 60 <= int(match) <= 800:
            return int(match)
    return None


def _split_name_voltage(normalized_name: str) -> tuple[str, int | None]:
    """Splits a trailing voltage level off a normalized name, e.g. "eemshaven 380" -> ("eemshaven", 380)."""
    tokens = normalized_name.split()
    if tokens and tokens[-1].isdigit():
        return " ".join(tokens[:-1]), int(tokens[-1])
    return normalized_name, None


def _locname_display_names(names: pd.Series) -> dict[str, str]:
    """Maps each raw location name spelling to the display spelling of its normalized group.

    For each group of spellings that share a normalization key, the display name is the most
    frequent raw spelling, preferring mixed-case over ALL-CAPS, then alphabetical order.
    """
    names = names.dropna().astype(str).str.strip()
    counts = names.value_counts()
    groups: dict[str, list[str]] = {}
    for name in counts.index:
        groups.setdefault(_normalize_locname(name), []).append(name)

    mapping: dict[str, str] = {}
    for raws in groups.values():
        mixed_case = [raw for raw in raws if raw != raw.upper()] or raws
        display = min(mixed_case, key=lambda raw: (-counts[raw], raw))
        for raw in raws:
            mapping[raw] = display

    merged = {key: raws for key, raws in groups.items() if len(raws) > 1}
    if merged:
        logger.info(f"Merged {len(merged)} groups of inconsistent location name spellings:")
        for raws in merged.values():
            logger.info(f"  {mapping[raws[0]]!r} <- {sorted(raws)}")

    return mapping


def _replace_locnames(names: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Applies the raw -> display name mapping to a Series, keeping missing values untouched."""
    stripped = names.where(names.isna(), names.astype(str).str.strip())
    return stripped.map(lambda name: mapping.get(name, name) if isinstance(name, str) else name)


def _replace_substrings(names: pd.Series, repl: list[tuple]) -> pd.Series:
    """Strips and applies substring replacements to all non-missing string values of a Series."""
    return names.map(lambda st: _multi_str_repl(st.strip(), repl) if isinstance(st, str) else st)


def _multi_str_repl(st: str, repl: list[tuple]) -> str:
    for old, new in repl:
        st = st.replace(old, new)
    return st


if __name__ == "__main__":
    import os
    from pathlib import Path

    from pandapower.file_io import from_json, to_json

    home = str(Path.home())
    jao_data_folder = os.path.join(home, "Documents", "JAO Static Grid Model")

    release5 = os.path.join(jao_data_folder, "20240329_Core Static Grid Model – 5th release")
    excel_file_path = os.path.join(release5, "20240329_Core Static Grid Model_public.xlsx")
    html_file_path = os.path.join(
        release5, "20240329_Core Static Grid Model Map_public", "2024-03-18_Core_SGM_publication.html"
    )

    release6 = os.path.join(jao_data_folder, "202409_Core Static Grid Mode_6th release")
    excel_file_path = os.path.join(release6, "20240916_Core Static Grid Model_for publication.xlsx")
    html_file_path = os.path.join(
        release6, "2024-09-13_Core_SGM_publication_files", "2024-09-13_Core_SGM_publication.html"
    )

    pp_net_json_file = os.path.join(home, "desktop", "jao_grid.json")

    if 1:  # read from original data
        net = from_jao(excel_file_path, html_file_path, True, drop_grid_groups_islands=True)
        to_json(net, pp_net_json_file)
    else:  # load net from already converted and stored net
        net = from_json(pp_net_json_file)

    print(net)
    grid_groups = get_grid_groups(net)
    print(grid_groups)

    _fill_geo_at_one_sided_branches_without_geo_extent(net)
