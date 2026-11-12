# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Parsing of the JAO html map and assignment of geodata to buses and lines."""

import json
import logging

import numpy as np
import pandas as pd

from pandapower.network import pandapowerNet
from pandapower.plotting import set_line_geodata_from_bus_geodata

logger = logging.getLogger(__name__)


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


def _lng_lat_to_df(dict_: dict, line_EIC: str, line_name: str) -> pd.DataFrame:
    """Converts one polyline's lng/lat lists into a long-format geo DataFrame."""
    return pd.DataFrame(
        [
            [line_EIC, line_name, "from", "lng", dict_["lng"][0]],
            [line_EIC, line_name, "to", "lng", dict_["lng"][1]],
            [line_EIC, line_name, "from", "lat", dict_["lat"][0]],
            [line_EIC, line_name, "to", "lat", dict_["lat"][1]],
        ],
        columns=["EIC_Code", "name", "bus", "geo_dim", "value"],
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
        [
            lgd_EIC_bus.set_axis(lgd_EIC_bus_idx_extended, axis=0),
            lgd_name_bus.set_axis(lgd_name_bus_idx_extended, axis=0),
        ]
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


def _fill_geo_at_one_sided_branches_without_geo_extent(net: pandapowerNet):
    """Copies known bus geodata to the other end of branches with only one georeferenced side."""

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
