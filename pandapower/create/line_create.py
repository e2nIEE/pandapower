# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging
from operator import itemgetter
from typing import Iterable, Sequence

from numpy import nan, isnan, any as np_any, bool_, all as np_all, float64
import numpy.typing as npt

from pandapower.auxiliary import pandapowerNet
from pandapower.std_types import load_std_type
from pandapower.pp_types import Int, LineType
from pandapower.create._utils import (
    _add_branch_geodata,
    _add_multiple_branch_geodata,
    _add_to_entries_if_not_nan,
    _check_branch_element,
    _check_multiple_branch_elements,
    _get_index_with_check,
    _get_multiple_index_with_check,
    _set_entries,
    _set_multiple_entries,
    _set_value_if_not_nan,
)

logger = logging.getLogger(__name__)


def create_line(
    net: pandapowerNet,
    from_bus: Int,
    to_bus: Int,
    length_km: float,
    std_type: str,
    name: str | None = None,
    index: Int | None = None,
    geodata: Iterable[tuple[float, float]] | None = None,
    df: float = 1.0,
    parallel: int = 1,
    in_service: bool = True,
    max_loading_percent: float = nan,
    alpha: float = nan,
    temperature_degree_celsius: float = nan,
    **kwargs,
) -> Int:
    """
    Creates a line element in net["line"]
    The line parameters are defined through the standard type library.


    INPUT:
        **net** - The net within this line should be created

        **from_bus** (int) - ID of the bus on one side which the line will be connected with

        **to_bus** (int) - ID of the bus on the other side which the line will be connected with

        **length_km** (float) - The line length in km

        **std_type** (string) - Name of a standard line type:

                                - Pre-defined in standard_linetypes

                                **or**

                                - Customized std_type made using **create_std_type()**

    OPTIONAL:
        **name** (string, None) - A custom name for this line

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **geodata**
        (Iterable[Tuple[int, int]|Tuple[float, float]], default None) -
        The geodata of the line. The first element should be the coordinates
        of from_bus and the last should be the coordinates of to_bus. The points
        in the middle represent the bending points of the line

        **in_service** (boolean, True) - True for in_service or False for out of service

        **df** (float, 1) - derating factor: maximum current of line in relation to nominal current\
            of line (from 0 to 1)

        **parallel** (integer, 1) - number of parallel line systems

        **max_loading_percent (float)** - maximum current loading (only needed for OPF)

        **alpha (float)** - temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0))

        **temperature_degree_celsius (float)** - line temperature for which line resistance is adjusted

        **tdpf (bool)** - whether the line is considered in the TDPF calculation

        **wind_speed_m_per_s (float)** - wind speed at the line in m/s (TDPF)

        **wind_angle_degree (float)** - angle of attack between the wind direction and the line (TDPF)

        **conductor_outer_diameter_m (float)** - outer diameter of the line conductor in m (TDPF)

        **air_temperature_degree_celsius (float)** - ambient temperature in °C (TDPF)

        **reference_temperature_degree_celsius (float)** - reference temperature in °C for which \
            r_ohm_per_km for the line is specified (TDPF)

        **solar_radiation_w_per_sq_m (float)** - solar radiation on horizontal plane in W/m² (TDPF)

        **solar_absorptivity (float)** - Albedo factor for absorptivity of the lines (TDPF)

        **emissivity (float)** - Albedo factor for emissivity of the lines (TDPF)

        **r_theta_kelvin_per_mw (float)** - thermal resistance of the line (TDPF, only for \
            simplified method)

        **mc_joule_per_m_k (float)** - specific mass of the conductor multiplied by the specific \
            thermal capacity of the material (TDPF, only for thermal inertia consideration with \
                tdpf_delay_s parameter)

    OUTPUT:
        **index** (int) - The unique ID of the created line

    EXAMPLE:
        create_line(net, from_bus=0, to_bus=1, length_km=0.1,  std_type="NAYY 4x50 SE", name="line1")

    """

    # check if bus exist to attach the line to
    _check_branch_element(net, "Line", index, from_bus, to_bus)

    index = _get_index_with_check(net, "line", index)

    tdpf_columns = (
        "wind_speed_m_per_s",
        "wind_angle_degree",
        "conductor_outer_diameter_m",
        "air_temperature_degree_celsius",
        "reference_temperature_degree_celsius",
        "solar_radiation_w_per_sq_m",
        "solar_absorptivity",
        "emissivity",
        "r_theta_kelvin_per_mw",
        "mc_joule_per_m_k",
    )
    tdpf_parameters = {c: kwargs.pop(c) for c in tdpf_columns if c in kwargs}

    entries = {
        "name": name,
        "length_km": length_km,
        "from_bus": from_bus,
        "to_bus": to_bus,
        "in_service": in_service,
        "std_type": std_type,
        "df": df,
        "parallel": parallel,
        **kwargs,
    }

    lineparam = load_std_type(net, std_type, "line")

    entries.update({param: lineparam[param] for param in ["r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km", "max_i_ka"]})
    if "r0_ohm_per_km" in lineparam:
        entries.update({param: lineparam[param] for param in ["r0_ohm_per_km", "x0_ohm_per_km", "c0_nf_per_km"]})

    entries["g_us_per_km"] = lineparam["g_us_per_km"] if "g_us_per_km" in lineparam else 0.0

    if "type" in lineparam:
        entries["type"] = lineparam["type"]

    # only add alpha from std_type if any line already has an alpha # TODO inconsistent behavior: Document this CLEARLY!
    if "alpha" in net.line.columns and "alpha" in lineparam:
        entries["alpha"] = lineparam["alpha"]

    _set_entries(net, "line", index, entries=entries)

    _set_value_if_not_nan(net, index, max_loading_percent, "max_loading_percent", "line")
    _set_value_if_not_nan(net, index, alpha, "alpha", "line")
    _set_value_if_not_nan(net, index, temperature_degree_celsius, "temperature_degree_celsius", "line")
    # add optional columns for TDPF if parameters passed to kwargs:
    _set_value_if_not_nan(net, index, kwargs.get("tdpf"), "tdpf", "line", bool_)
    for column, value in tdpf_parameters.items():
        _set_value_if_not_nan(net, index, value, column, "line", float64)

    _add_branch_geodata(net, geodata, index)

    return index


def create_line_dc(
    net: pandapowerNet,
    from_bus_dc: Int,
    to_bus_dc: Int,
    length_km: float,
    std_type: str,
    name: str | None = None,
    index: Int | None = None,
    geodata: Iterable[tuple[float, float]] | None = None,
    df: float = 1.0,
    parallel: int = 1,
    in_service: bool = True,
    max_loading_percent: float = nan,
    alpha: float = nan,
    temperature_degree_celsius: float = nan,
    **kwargs,
) -> Int:
    """
    Creates a line element in net["line_dc"]
    The line_dc parameters are defined through the standard type library.


    INPUT:
        **net** - The net within this line should be created

        **from_bus_dc** (int) - ID of the bus_dc on one side which the line will be connected with

        **to_bus_dc** (int) - ID of the bus_dc on the other side which the line will be connected with

        **length_km** (float) - The line length in km

        **std_type** (string) - Name of a standard line type :

                                - Pre-defined in standard_linetypes

                                **or**

                                - Customized std_type made using **create_std_type()**

    OPTIONAL:
        **name** (string, None) - A custom name for this line_dc

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **geodata**
        (array, default None, shape= (,2L)) -
        The line geodata of the line_dc. The first row should be the coordinates
        of bus a and the last should be the coordinates of bus b. The points
        in the middle represent the bending points of the line

        **in_service** (boolean, True) - True for in_service or False for out of service

        **df** (float, 1) - derating factor: maximum current of line_dc in relation to nominal current\
            of line (from 0 to 1)

        **parallel** (integer, 1) - number of parallel line systems

        **max_loading_percent (float)** - maximum current loading (only needed for OPF)

        **alpha (float)** - temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0))

        **temperature_degree_celsius (float)** - line temperature for which line resistance is adjusted

        **tdpf (bool)** - whether the line is considered in the TDPF calculation

        **wind_speed_m_per_s (float)** - wind speed at the line in m/s (TDPF)

        **wind_angle_degree (float)** - angle of attack between the wind direction and the line (TDPF)

        **conductor_outer_diameter_m (float)** - outer diameter of the line conductor in m (TDPF)

        **air_temperature_degree_celsius (float)** - ambient temperature in °C (TDPF)

        **reference_temperature_degree_celsius (float)** - reference temperature in °C for which \
            r_ohm_per_km for the line_dc is specified (TDPF)

        **solar_radiation_w_per_sq_m (float)** - solar radiation on horizontal plane in W/m² (TDPF)

        **solar_absorptivity (float)** - Albedo factor for absorptivity of the lines (TDPF)

        **emissivity (float)** - Albedo factor for emissivity of the lines (TDPF)

        **r_theta_kelvin_per_mw (float)** - thermal resistance of the line (TDPF, only for \
            simplified method)

        **mc_joule_per_m_k (float)** - specific mass of the conductor multiplied by the specific \
            thermal capacity of the material (TDPF, only for thermal inertia consideration with \
                tdpf_delay_s parameter)

    OUTPUT:
        **index** (int) - The unique ID of the created dc line

    EXAMPLE:
        create_line_dc(net, from_bus_dc=0, to_bus_dc=1, length_km=0.1,  std_type="NAYY 4x50 SE", name="line_dc1")

    """

    # check if bus exist to attach the line to
    _check_branch_element(net, "Line_dc", index, from_bus_dc, to_bus_dc, node_name="bus_dc")

    index = _get_index_with_check(net, "line_dc", index)

    tdpf_columns = (
        "wind_speed_m_per_s",
        "wind_angle_degree",
        "conductor_outer_diameter_m",
        "air_temperature_degree_celsius",
        "reference_temperature_degree_celsius",
        "solar_radiation_w_per_sq_m",
        "solar_absorptivity",
        "emissivity",
        "r_theta_kelvin_per_mw",
        "mc_joule_per_m_k",
    )
    tdpf_parameters = {c: kwargs.pop(c) for c in tdpf_columns if c in kwargs}

    entries = {
        "name": name,
        "length_km": length_km,
        "from_bus_dc": from_bus_dc,
        "to_bus_dc": to_bus_dc,
        "in_service": in_service,
        "std_type": std_type,
        "df": df,
        "parallel": parallel,
        **kwargs,
    }

    lineparam = load_std_type(net, std_type, "line_dc")

    entries.update({param: lineparam[param] for param in ["r_ohm_per_km", "max_i_ka"]})
    if "r0_ohm_per_km" in lineparam:
        entries.update({param: lineparam[param] for param in ["r0_ohm_per_km"]})

    entries["g_us_per_km"] = lineparam["g_us_per_km"] if "g_us_per_km" in lineparam else 0.0

    if "type" in lineparam:
        entries["type"] = lineparam["type"]

    # if net.line column already has alpha, add it from std_type
    if "alpha" in net.line.columns and "alpha" in lineparam:
        entries["alpha"] = lineparam["alpha"]

    _set_entries(net, "line_dc", index, entries=entries)

    if geodata and hasattr(geodata, "__iter__"):
        geo = [[x, y] for x, y in geodata]
        net.line_dc.at[index, "geo"] = f'{{"coordinates": {geo}, "type": "LineString"}}'

    _set_value_if_not_nan(net, index, max_loading_percent, "max_loading_percent", "line_dc")
    _set_value_if_not_nan(net, index, alpha, "alpha", "line_dc")
    _set_value_if_not_nan(net, index, temperature_degree_celsius, "temperature_degree_celsius", "line_dc")
    # add optional columns for TDPF if parameters passed to kwargs:
    _set_value_if_not_nan(net, index, kwargs.get("tdpf"), "tdpf", "line_dc", bool_)
    for column, value in tdpf_parameters.items():
        _set_value_if_not_nan(net, index, value, column, "line_dc", float64)

    _add_branch_geodata(net, geodata, index, "line_dc")

    return index


def create_lines(
    net: pandapowerNet,
    from_buses: Sequence,
    to_buses: Sequence,
    length_km: float | Iterable[float],
    std_type: str | Iterable[str],
    name: Iterable[str] | None = None,
    index: Int | Iterable[Int] | None = None,
    geodata: Iterable[Iterable[tuple[float, float]]] | Iterable[tuple[float, float]] | None = None,
    df: float | Iterable[float] = 1.0,
    parallel: int | Iterable[int] = 1,
    in_service: bool | Iterable[bool] = True,
    max_loading_percent: float | Iterable[float] = nan,
    **kwargs,
) -> npt.NDArray[Int]:
    """ Convenience function for creating many lines at once. Parameters 'from_buses' and 'to_buses'
        must be arrays of equal length. Other parameters may be either arrays of the same length or
        single or values. In any case the line parameters are defined through a single standard
        type, so all lines have the same standard type.


        INPUT:
            **net** - The net within this line should be created

            **from_buses** (list of int) - ID of the bus on one side which the line will be \
                connected with

            **to_buses** (list of int) - ID of the bus on the other side which the line will be \
                connected with

            **length_km** (list of float) - The line length in km

            **std_type** (string) - The line type of the lines.

        OPTIONAL:
            **name** (list of string, None) - A custom name for this line

            **index** (list of int, None) - Force a specified ID if it is available. If None, the\
                index one higher than the highest already existing index is selected.

            **geodata**
            (Iterable[Iterable[Tuple[x, y]]] or Iterable[Tuple[x, y]], default None) -
            The geodata of the line. The first element should be the coordinates
            of from_bus and the last should be the coordinates of to_bus. The points
            in the middle represent the bending points of the line

            **in_service** (list of boolean, True) - True for in_service or False for out of service

            **df** (list of float, 1) - derating factor: maximum current of line in relation to \
                nominal current of line (from 0 to 1)

            **parallel** (list of integer, 1) - number of parallel line systems

            **max_loading_percent (list of float)** - maximum current loading (only needed for OPF)

            **alpha (float)** - temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0))

            **temperature_degree_celsius (float)** - line temperature for which line resistance is adjusted

            **tdpf (bool)** - whether the line is considered in the TDPF calculation

            **wind_speed_m_per_s (float)** - wind speed at the line in m/s (TDPF)

            **wind_angle_degree (float)** - angle of attack between the wind direction and the line (TDPF)

            **conductor_outer_diameter_m (float)** - outer diameter of the line conductor in m (TDPF)

            **air_temperature_degree_celsius (float)** - ambient temperature in °C (TDPF)

            **reference_temperature_degree_celsius (float)** - reference temperature in °C for \
                which r_ohm_per_km for the line is specified (TDPF)

            **solar_radiation_w_per_sq_m (float)** - solar radiation on horizontal plane in W/m² (TDPF)

            **solar_absorptivity (float)** - Albedo factor for absorptivity of the lines (TDPF)

            **emissivity (float)** - Albedo factor for emissivity of the lines (TDPF)

            **r_theta_kelvin_per_mw (float)** - thermal resistance of the line (TDPF, only for \
                simplified method)

            **mc_joule_per_m_k (float)** - specific mass of the conductor multiplied by the \
                specific thermal capacity of the material (TDPF, only for thermal inertia \
                consideration with tdpf_delay_s parameter)

        OUTPUT:
            **index** (list of int) - The unique ID of the created lines

        EXAMPLE:
            create_lines(net, from_buses=[0,1], to_buses=[2,3], length_km=0.1, std_type="NAYY 4x50 SE",
                         name=["line1", "line2"])

    """
    _check_multiple_branch_elements(net, from_buses, to_buses, "Lines")

    index = _get_multiple_index_with_check(net, "line", index, len(from_buses))

    entries = {
        "from_bus": from_buses,
        "to_bus": to_buses,
        "length_km": length_km,
        "std_type": std_type,
        "name": name,
        "df": df,
        "parallel": parallel,
        "in_service": in_service,
        **kwargs,
    }

    # add std type data
    if isinstance(std_type, str):
        lineparam = load_std_type(net, std_type, "line")
        entries["r_ohm_per_km"] = lineparam["r_ohm_per_km"]
        entries["x_ohm_per_km"] = lineparam["x_ohm_per_km"]
        entries["c_nf_per_km"] = lineparam["c_nf_per_km"]
        entries["max_i_ka"] = lineparam["max_i_ka"]
        entries["g_us_per_km"] = lineparam["g_us_per_km"] if "g_us_per_km" in lineparam else 0.0
        if "type" in lineparam:
            entries["type"] = lineparam["type"]
    else:
        lineparam = list(map(load_std_type, [net] * len(index), std_type, ["line"] * len(index)))
        entries["r_ohm_per_km"] = list(map(itemgetter("r_ohm_per_km"), lineparam))
        entries["x_ohm_per_km"] = list(map(itemgetter("x_ohm_per_km"), lineparam))
        entries["c_nf_per_km"] = list(map(itemgetter("c_nf_per_km"), lineparam))
        entries["max_i_ka"] = list(map(itemgetter("max_i_ka"), lineparam))
        entries["g_us_per_km"] = [line_param_dict.get("g_us_per_km", 0) for line_param_dict in lineparam]
        entries["type"] = [line_param_dict.get("type", None) for line_param_dict in lineparam]

    _add_to_entries_if_not_nan(net, "line", entries, index, "max_loading_percent", max_loading_percent)

    # add optional columns for TDPF if parameters passed to kwargs:
    _add_to_entries_if_not_nan(net, "line", entries, index, "tdpf", kwargs.get("tdpf"), bool_)
    tdpf_columns = (
        "wind_speed_m_per_s",
        "wind_angle_degree",
        "conductor_outer_diameter_m",
        "air_temperature_degree_celsius",
        "reference_temperature_degree_celsius",
        "solar_radiation_w_per_sq_m",
        "solar_absorptivity",
        "emissivity",
        "r_theta_kelvin_per_mw",
        "mc_joule_per_m_k",
    )
    tdpf_parameters = {c: kwargs.pop(c) for c in tdpf_columns if c in kwargs}
    for column, value in tdpf_parameters.items():
        _add_to_entries_if_not_nan(net, "line", entries, index, column, value, float64)

    _set_multiple_entries(net, "line", index, entries=entries)

    _add_multiple_branch_geodata(net, geodata, index)

    return index


def create_lines_dc(
    net: pandapowerNet,
    from_buses_dc: Sequence,
    to_buses_dc: Sequence,
    length_km: float | Iterable[float],
    std_type: str | Sequence[str],
    name: Iterable[str] | None = None,
    index: Int | Iterable[Int] | None = None,
    geodata: Iterable[Iterable[tuple[float, float]]] | None = None,
    df: float | Iterable[float] = 1.0,
    parallel: int | Iterable[int] = 1,
    in_service: bool | Iterable[bool] = True,
    max_loading_percent: float | Iterable[float] = nan,
    **kwargs,
) -> npt.NDArray[Int]:
    """ Convenience function for creating many dc lines at once. Parameters 'from_buses_dc' and 'to_buses_dc'
        must be arrays of equal length. Other parameters may be either arrays of the same length or
        single or values. In any case the dc line parameters are defined through a single standard
        type, so all lines have the same standard type.


        INPUT:
            **net** - The net within this dc line should be created

            **from_buses_dc** (list of int) - ID of the dc buses on one side which the dc lines will be \
                connected with

            **to_buses_dc** (list of int) - ID of the dc buses on the other side which the dc lines will be \
                connected with

            **length_km** (list of float) - The dc line length in km

            **std_type** (list of strings) - The dc line type of the dc lines.

        OPTIONAL:
            **name** (list of string, None) - A custom name for these dc lines

            **index** (list of int, None) - Force a specified ID if it is available. If None, the\
                index one higher than the highest already existing index is selected.

            **geodata**
            (list of arrays, default None, shape of arrays (,2L)) -
            The linegeodata of the dc line. The first row should be the coordinates
            of dc bus a and the last should be the coordinates of dc bus b. The points
            in the middle represent the bending points of the dc line

            **in_service** (list of boolean, True) - True for in_service or False for out of service

            **df** (list of float, 1) - derating factor: maximum current of line in relation to \
                nominal current of line (from 0 to 1)

            **parallel** (list of integer, 1) - number of parallel line systems

            **max_loading_percent (list of float)** - maximum current loading (only needed for OPF)

            **alpha (float)** - temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0))

            **temperature_degree_celsius (float)** - line temperature for which line resistance is adjusted

            **tdpf (bool)** - whether the line is considered in the TDPF calculation

            **wind_speed_m_per_s (float)** - wind speed at the line in m/s (TDPF)

            **wind_angle_degree (float)** - angle of attack between the wind direction and the line (TDPF)

            **conductor_outer_diameter_m (float)** - outer diameter of the line conductor in m (TDPF)

            **air_temperature_degree_celsius (float)** - ambient temperature in °C (TDPF)

            **reference_temperature_degree_celsius (float)** - reference temperature in °C for \
                which r_ohm_per_km for the line is specified (TDPF)

            **solar_radiation_w_per_sq_m (float)** - solar radiation on horizontal plane in W/m² (TDPF)

            **solar_absorptivity (float)** - Albedo factor for absorptivity of the lines (TDPF)

            **emissivity (float)** - Albedo factor for emissivity of the lines (TDPF)

            **r_theta_kelvin_per_mw (float)** - thermal resistance of the line (TDPF, only for \
                simplified method)

            **mc_joule_per_m_k (float)** - specific mass of the conductor multiplied by the \
                specific thermal capacity of the material (TDPF, only for thermal inertia \
                consideration with tdpf_delay_s parameter)

        OUTPUT:
            **index** (list of int) - The unique ID of the created dc lines

        EXAMPLE:
            create_lines_dc(net, from_buses_dc=[0,1], to_buses_dc=[2,3], length_km=0.1,
            std_type="Not specified yet", name=["line_dc1","line_dc2"])

    """
    _check_multiple_branch_elements(
        net, from_buses_dc, to_buses_dc, "Lines_dc", node_name="bus_dc", plural="(all dc buses)"
    )

    index = _get_multiple_index_with_check(net, "line_dc", index, len(from_buses_dc))

    entries = {
        "from_bus_dc": from_buses_dc,
        "to_bus_dc": to_buses_dc,
        "length_km": length_km,
        "std_type": std_type,
        "name": name,
        "df": df,
        "parallel": parallel,
        "in_service": in_service,
        **kwargs,
    }

    # add std type data
    if isinstance(std_type, str):
        lineparam = load_std_type(net, std_type, "line_dc")
        entries["r_ohm_per_km"] = lineparam["r_ohm_per_km"]
        entries["max_i_ka"] = lineparam["max_i_ka"]
        entries["g_us_per_km"] = lineparam["g_us_per_km"] if "g_us_per_km" in lineparam else 0.0
        if "type" in lineparam:
            entries["type"] = lineparam["type"]
    else:
        lineparam = list(map(load_std_type, [net] * len(std_type), std_type, ["line_dc"] * len(std_type)))
        entries["r_ohm_per_km"] = list(map(itemgetter("r_ohm_per_km"), lineparam))
        entries["max_i_ka"] = list(map(itemgetter("max_i_ka"), lineparam))
        entries["g_us_per_km"] = [line_param_dict.get("g_us_per_km", 0) for line_param_dict in lineparam]
        entries["type"] = [line_param_dict.get("type", None) for line_param_dict in lineparam]

    _add_to_entries_if_not_nan(net, "line_dc", entries, index, "max_loading_percent", max_loading_percent)

    # add optional columns for TDPF if parameters passed to kwargs:
    _add_to_entries_if_not_nan(net, "line_dc", entries, index, "tdpf", kwargs.get("tdpf"), bool_)
    tdpf_columns = (
        "wind_speed_m_per_s",
        "wind_angle_degree",
        "conductor_outer_diameter_m",
        "air_temperature_degree_celsius",
        "reference_temperature_degree_celsius",
        "solar_radiation_w_per_sq_m",
        "solar_absorptivity",
        "emissivity",
        "r_theta_kelvin_per_mw",
        "mc_joule_per_m_k",
    )
    tdpf_parameters = {c: kwargs.pop(c) for c in tdpf_columns if c in kwargs}
    for column, value in tdpf_parameters.items():
        _add_to_entries_if_not_nan(net, "line_dc", entries, index, column, value, float64)

    _set_multiple_entries(net, "line_dc", index, entries=entries)

    _add_multiple_branch_geodata(net, geodata, index, "line_dc")

    return index


def create_line_from_parameters(
    net: pandapowerNet,
    from_bus: Int,
    to_bus: Int,
    length_km: float,
    r_ohm_per_km: float,
    x_ohm_per_km: float,
    c_nf_per_km: float,
    max_i_ka: float,
    name: str | None = None,
    index: Int | None = None,
    type: LineType | None = None,
    geodata: Iterable[tuple[float, float]] | None = None,
    in_service: bool = True,
    df: float = 1.0,
    parallel: int = 1,
    g_us_per_km: float = 0.0,
    max_loading_percent: float = nan,
    alpha: float = nan,
    temperature_degree_celsius: float = nan,
    r0_ohm_per_km: float = nan,
    x0_ohm_per_km: float = nan,
    c0_nf_per_km: float = nan,
    g0_us_per_km: float = 0,
    endtemp_degree: float = nan,
    **kwargs,
) -> Int:
    """
    Creates a line element in net["line"] from line parameters.

    INPUT:
        **net** - The net within this line should be created

        **from_bus** (int) - ID of the bus on one side which the line will be connected with

        **to_bus** (int) - ID of the bus on the other side which the line will be connected with

        **length_km** (float) - The line length in km

        **r_ohm_per_km** (float) - line resistance in ohm per km

        **x_ohm_per_km** (float) - line reactance in ohm per km

        **c_nf_per_km** (float) - line capacitance (line-to-earth) in nano Farad per km

        **r0_ohm_per_km** (float) - zero sequence line resistance in ohm per km

        **x0_ohm_per_km** (float) - zero sequence line reactance in ohm per km

        **c0_nf_per_km** (float) - zero sequence line capacitance in nano Farad per km

        **max_i_ka** (float) - maximum thermal current in kilo Ampere

    OPTIONAL:
        **name** (string, None) - A custom name for this line

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **in_service** (boolean, True) - True for in_service or False for out of service

        **type** (str, None) - type of line ("ol" for overhead line or "cs" for cable system)

        **df** (float, 1) - derating factor: maximum current of line in relation to nominal current\
            of line (from 0 to 1)

        **g_us_per_km** (float, 0) - dielectric conductance in micro Siemens per km

        **g0_us_per_km** (float, 0) - zero sequence dielectric conductance in micro Siemens per km

        **parallel** (integer, 1) - number of parallel line systems

        **geodata**
        (array, default None, shape= (,2)) -
        The geodata of the line. The first row should be the coordinates
        of bus a and the last should be the coordinates of bus b. The points
        in the middle represent the bending points of the line

        **max_loading_percent (float)** - maximum current loading (only needed for OPF)

        **alpha (float)** - temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0)))

        **temperature_degree_celsius (float)** - line temperature for which line resistance is adjusted

        **tdpf (bool)** - whether the line is considered in the TDPF calculation

        **wind_speed_m_per_s (float)** - wind speed at the line in m/s (TDPF)

        **wind_angle_degree (float)** - angle of attack between the wind direction and the line (TDPF)

        **conductor_outer_diameter_m (float)** - outer diameter of the line conductor in m (TDPF)

        **air_temperature_degree_celsius (float)** - ambient temperature in °C (TDPF)

        **reference_temperature_degree_celsius (float)** - reference temperature in °C for which \
            r_ohm_per_km for the line is specified (TDPF)

        **solar_radiation_w_per_sq_m (float)** - solar radiation on horizontal plane in W/m² (TDPF)

        **solar_absorptivity (float)** - Albedo factor for absorptivity of the lines (TDPF)

        **emissivity (float)** - Albedo factor for emissivity of the lines (TDPF)

        **r_theta_kelvin_per_mw (float)** - thermal resistance of the line (TDPF, only for \
            simplified method)

        **mc_joule_per_m_k (float)** - specific mass of the conductor multiplied by the specific \
            thermal capacity of the material (TDPF, only for thermal inertia consideration with \
            tdpf_delay_s parameter)

    OUTPUT:
        **index** (int) - The unique ID of the created line

    EXAMPLE:
        create_line_from_parameters(net, from_bus=0, to_bus=1, length_km=0.1,
        r_ohm_per_km=.01, x_ohm_per_km=0.05, c_nf_per_km=10,
        max_i_ka=0.4, name="line1")

    """

    # check if bus exist to attach the line to
    _check_branch_element(net, "Line", index, from_bus, to_bus)

    index = _get_index_with_check(net, "line", index)

    tdpf_columns = (
        "wind_speed_m_per_s",
        "wind_angle_degree",
        "conductor_outer_diameter_m",
        "air_temperature_degree_celsius",
        "reference_temperature_degree_celsius",
        "solar_radiation_w_per_sq_m",
        "solar_absorptivity",
        "emissivity",
        "r_theta_kelvin_per_mw",
        "mc_joule_per_m_k",
    )
    tdpf_parameters = {c: kwargs.pop(c) for c in tdpf_columns if c in kwargs}

    entries = {
        "name": name,
        "length_km": length_km,
        "from_bus": from_bus,
        "to_bus": to_bus,
        "in_service": in_service,
        "std_type": None,
        "df": df,
        "r_ohm_per_km": r_ohm_per_km,
        "x_ohm_per_km": x_ohm_per_km,
        "c_nf_per_km": c_nf_per_km,
        "max_i_ka": max_i_ka,
        "parallel": parallel,
        "type": type,
        "g_us_per_km": g_us_per_km,
        **kwargs,
    }
    _set_entries(net, "line", index, entries=entries)

    nan_0_values = [isnan(r0_ohm_per_km), isnan(x0_ohm_per_km), isnan(c0_nf_per_km)]
    if not np_any(nan_0_values):
        _set_value_if_not_nan(net, index, r0_ohm_per_km, "r0_ohm_per_km", "line")
        _set_value_if_not_nan(net, index, x0_ohm_per_km, "x0_ohm_per_km", "line")
        _set_value_if_not_nan(net, index, c0_nf_per_km, "c0_nf_per_km", "line")
        _set_value_if_not_nan(net, index, g0_us_per_km, "g0_us_per_km", "line", default_val=0.0)
    elif not np_all(nan_0_values):
        logger.warning(
            "Zero sequence values are given for only some parameters. Please specify "
            "them for all parameters, otherwise they are not set!"
        )

    _set_value_if_not_nan(net, index, max_loading_percent, "max_loading_percent", "line")
    _set_value_if_not_nan(net, index, alpha, "alpha", "line")
    _set_value_if_not_nan(net, index, temperature_degree_celsius, "temperature_degree_celsius", "line")
    _set_value_if_not_nan(net, index, endtemp_degree, "endtemp_degree", "line")

    # add optional columns for TDPF if parameters passed to kwargs:
    _set_value_if_not_nan(net, index, kwargs.get("tdpf"), "tdpf", "line", bool_)
    for column, value in tdpf_parameters.items():
        _set_value_if_not_nan(net, index, value, column, "line", float64)

    _add_branch_geodata(net, geodata, index)
    return index


def create_line_dc_from_parameters(
    net: pandapowerNet,
    from_bus_dc: Int,
    to_bus_dc: Int,
    length_km: float,
    r_ohm_per_km: float,
    max_i_ka: float,
    name: str | None = None,
    index: Int | None = None,
    type: LineType | None = None,
    geodata: Iterable[tuple[float, float]] | None = None,
    in_service: bool = True,
    df: float = 1.0,
    parallel: int = 1,
    max_loading_percent: float = nan,
    alpha: float = nan,
    temperature_degree_celsius: float = nan,
    g_us_per_km: float = 0.0,
    **kwargs,
) -> Int:
    """
    Creates a dc line element in net["line_dc"] from dc line parameters.

    INPUT:
        **net** - The net within this dc line should be created

        **from_bus_dc** (int) - ID of the dc bus on one side which the dc line will be connected with

        **to_bus_dc** (int) - ID of the dc bus on the other side which the dc line will be connected with

        **length_km** (float) - The dc line length in km

        **r_ohm_per_km** (float) - dc line resistance in ohm per km

        **max_i_ka** (float) - maximum thermal current in kilo Ampere

    OPTIONAL:
        **name** (string, None) - A custom name for this line

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **in_service** (boolean, True) - True for in_service or False for out of service

        **type** (str, None) - type of dc line ("ol" for overhead dc line or "cs" for cable system)

        **df** (float, 1) - derating factor: maximum current of dc line in relation to nominal current\
            of line (from 0 to 1)

        **g_us_per_km** (float, 0) - dielectric conductance in micro Siemens per km

        **g0_us_per_km** (float, 0) - zero sequence dielectric conductance in micro Siemens per km

        **parallel** (integer, 1) - number of parallel line systems

        **geodata**
        (array, default None, shape= (,2L)) -
        The linegeodata of the dc line. The first row should be the coordinates
        of dc bus a and the last should be the coordinates of dc bus b. The points
        in the middle represent the bending points of the line

        **max_loading_percent (float)** - maximum current loading (only needed for OPF)

        **alpha (float)** - temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0)))

        **temperature_degree_celsius (float)** - line temperature for which line resistance is adjusted

        **tdpf (bool)** - whether the line is considered in the TDPF calculation

        **wind_speed_m_per_s (float)** - wind speed at the line in m/s (TDPF)

        **wind_angle_degree (float)** - angle of attack between the wind direction and the line (TDPF)

        **conductor_outer_diameter_m (float)** - outer diameter of the line conductor in m (TDPF)

        **air_temperature_degree_celsius (float)** - ambient temperature in °C (TDPF)

        **reference_temperature_degree_celsius (float)** - reference temperature in °C for which \
            r_ohm_per_km for the line is specified (TDPF)

        **solar_radiation_w_per_sq_m (float)** - solar radiation on horizontal plane in W/m² (TDPF)

        **solar_absorptivity (float)** - Albedo factor for absorptivity of the lines (TDPF)

        **emissivity (float)** - Albedo factor for emissivity of the lines (TDPF)

        **r_theta_kelvin_per_mw (float)** - thermal resistance of the line (TDPF, only for \
            simplified method)

        **mc_joule_per_m_k (float)** - specific mass of the conductor multiplied by the specific \
            thermal capacity of the material (TDPF, only for thermal inertia consideration with \
            tdpf_delay_s parameter)

    OUTPUT:
        **index** (int) - The unique ID of the created line

    EXAMPLE:
        create_line_dc_from_parameters(net, from_bus_dc=0, to_bus_dc=1, length_km=0.1,
        r_ohm_per_km=.01, max_i_ka=0.4, name="line_dc1")

    """

    # check if bus exist to attach the dc line to
    _check_branch_element(net, "Line_dc", index, from_bus_dc, to_bus_dc, node_name="bus_dc")

    index = _get_index_with_check(net, "line_dc", index)

    tdpf_columns = (
        "wind_speed_m_per_s",
        "wind_angle_degree",
        "conductor_outer_diameter_m",
        "air_temperature_degree_celsius",
        "reference_temperature_degree_celsius",
        "solar_radiation_w_per_sq_m",
        "solar_absorptivity",
        "emissivity",
        "r_theta_kelvin_per_mw",
        "mc_joule_per_m_k",
    )
    tdpf_parameters = {c: kwargs.pop(c) for c in tdpf_columns if c in kwargs}

    entries = {
        "name": name,
        "length_km": length_km,
        "from_bus_dc": from_bus_dc,
        "to_bus_dc": to_bus_dc,
        "in_service": in_service,
        "std_type": None,
        "df": df,
        "r_ohm_per_km": r_ohm_per_km,
        "max_i_ka": max_i_ka,
        "parallel": parallel,
        "type": type,
        "g_us_per_km": g_us_per_km,
        **kwargs,
    }
    _set_entries(net, "line_dc", index, entries=entries)

    if geodata and hasattr(geodata, "__iter__"):
        geo = [[x, y] for x, y in geodata]
        net.line_dc.at[index, "geo"] = f'{{"coordinates": {geo}, "type": "LineString"}}'

    _set_value_if_not_nan(net, index, max_loading_percent, "max_loading_percent", "line_dc")
    _set_value_if_not_nan(net, index, alpha, "alpha", "line_dc")
    _set_value_if_not_nan(net, index, temperature_degree_celsius, "temperature_degree_celsius", "line_dc")

    # add optional columns for TDPF if parameters passed to kwargs:
    _set_value_if_not_nan(net, index, kwargs.get("tdpf"), "tdpf", "line_dc", bool_)
    for column, value in tdpf_parameters.items():
        _set_value_if_not_nan(net, index, value, column, "line_dc", float64)

    _add_branch_geodata(net, geodata, index, "line_dc")

    return index


def create_lines_from_parameters(
    net: pandapowerNet,
    from_buses: Sequence,
    to_buses: Sequence,
    length_km: float | Iterable[float],
    r_ohm_per_km: float | Iterable[float],
    x_ohm_per_km: float | Iterable[float],
    c_nf_per_km: float | Iterable[float],
    max_i_ka: float | Iterable[float],
    name: Iterable[str] | None = None,
    index: Int | Iterable[Int] | None = None,
    type: LineType | Iterable[str] | None = None,
    geodata: Iterable[Iterable[tuple[float, float]]] | None = None,
    in_service: bool | Iterable[bool] = True,
    df: float | Iterable[float] = 1.0,
    parallel: int | Iterable[int] = 1,
    g_us_per_km: float | Iterable[float] = 0.0,
    max_loading_percent: float | Iterable[float] = nan,
    alpha: float = nan,
    temperature_degree_celsius: float = nan,
    r0_ohm_per_km: float | Iterable[float] = nan,
    x0_ohm_per_km: float | Iterable[float] = nan,
    c0_nf_per_km: float | Iterable[float] = nan,
    g0_us_per_km: float | Iterable[float] = nan,
    **kwargs,
) -> npt.NDArray[Int]:
    """
    Convenience function for creating many lines at once. Parameters 'from_buses' and 'to_buses'
        must be arrays of equal length. Other parameters may be either arrays of the same length or
        single or values.

    INPUT:
        **net** - The net within this line should be created

        **from_buses** (list of int) - ID of the buses on one side which the lines will be connected with

        **to_buses** (list of int) - ID of the buses on the other side which the lines will be connected\
            with

        **length_km** (list of float) - The line length in km

        **r_ohm_per_km** (list of float) - line resistance in ohm per km

        **x_ohm_per_km** (list of float) - line reactance in ohm per km

        **c_nf_per_km** (list of float) - line capacitance in nano Farad per km

        **r0_ohm_per_km** (list of float) - zero sequence line resistance in ohm per km

        **x0_ohm_per_km** (list of float) - zero sequence line reactance in ohm per km

        **c0_nf_per_km** (list of float) - zero sequence line capacitance in nano Farad per km

        **max_i_ka** (list of float) - maximum thermal current in kilo Ampere

    OPTIONAL:
        **name** (list of string, None) - A custom name for this line

        **index** (list of int, None) - Force a specified ID if it is available. If None, the\
            index one higher than the highest already existing index is selected.

        **in_service** (list of boolean, True) - True for in_service or False for out of service

        **type** (list of string, None) - type of line ("ol" for overhead line or "cs" for cable system)

        **df** (list of float, 1) - derating factor: maximum current of line in relation to nominal current\
            of line (from 0 to 1)

        **g_us_per_km** (list of float, 0) - dielectric conductance in micro Siemens per km

        **g0_us_per_km** (list of float, 0) - zero sequence dielectric conductance in micro Siemens per km

        **parallel** (list of integer, 1) - number of parallel line systems

        **geodata**
        (array, default None, shape= (,2)) -
        The geodata of the line. The first row should be the coordinates
        of bus a and the last should be the coordinates of bus b. The points
        in the middle represent the bending points of the line

        **max_loading_percent (list of float)** - maximum current loading (only needed for OPF)

        **alpha (float)** - temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0)))

        **temperature_degree_celsius (float)** - line temperature for which line resistance is adjusted

        **tdpf (bool)** - whether the line is considered in the TDPF calculation

        **wind_speed_m_per_s (float)** - wind speed at the line in m/s (TDPF)

        **wind_angle_degree (float)** - angle of attack between the wind direction and the line (TDPF)

        **conductor_outer_diameter_m (float)** - outer diameter of the line conductor in m (TDPF)

        **air_temperature_degree_celsius (float)** - ambient temperature in °C (TDPF)

        **reference_temperature_degree_celsius (float)** - reference temperature in °C for which \
            r_ohm_per_km for the line is specified (TDPF)

        **solar_radiation_w_per_sq_m (float)** - solar radiation on horizontal plane in W/m² (TDPF)

        **solar_absorptivity (float)** - Albedo factor for absorptivity of the lines (TDPF)

        **emissivity (float)** - Albedo factor for emissivity of the lines (TDPF)

        **r_theta_kelvin_per_mw (float)** - thermal resistance of the line (TDPF, only for \
            simplified method)

        **mc_joule_per_m_k (float)** - specific mass of the conductor multiplied by the specific \
            thermal capacity of the material (TDPF, only for thermal inertia consideration with \
            tdpf_delay_s parameter)

    OUTPUT:
        **index** (list of int) - The unique ID of the created lines

    EXAMPLE:
        create_lines_from_parameters(net, from_buses=[0,1], to_buses=[2,3], length_km=0.1,
        r_ohm_per_km=.01, x_ohm_per_km=0.05, c_nf_per_km=10, max_i_ka=0.4, name=["line1","line2"])

    """
    _check_multiple_branch_elements(net, from_buses, to_buses, "Lines")

    index = _get_multiple_index_with_check(net, "line", index, len(from_buses))

    entries = {
        "from_bus": from_buses,
        "to_bus": to_buses,
        "length_km": length_km,
        "type": type,
        "r_ohm_per_km": r_ohm_per_km,
        "x_ohm_per_km": x_ohm_per_km,
        "c_nf_per_km": c_nf_per_km,
        "max_i_ka": max_i_ka,
        "g_us_per_km": g_us_per_km,
        "name": name,
        "df": df,
        "parallel": parallel,
        "in_service": in_service,
        **kwargs,
    }

    _add_to_entries_if_not_nan(net, "line", entries, index, "max_loading_percent", max_loading_percent)
    _add_to_entries_if_not_nan(net, "line", entries, index, "r0_ohm_per_km", r0_ohm_per_km)
    _add_to_entries_if_not_nan(net, "line", entries, index, "x0_ohm_per_km", x0_ohm_per_km)
    _add_to_entries_if_not_nan(net, "line", entries, index, "c0_nf_per_km", c0_nf_per_km)
    _add_to_entries_if_not_nan(net, "line", entries, index, "g0_us_per_km", g0_us_per_km)
    _add_to_entries_if_not_nan(net, "line", entries, index, "temperature_degree_celsius", temperature_degree_celsius)
    _add_to_entries_if_not_nan(net, "line", entries, index, "alpha", alpha)

    # add optional columns for TDPF if parameters passed to kwargs:
    _add_to_entries_if_not_nan(net, "line", entries, index, "tdpf", kwargs.get("tdpf"), bool_)
    tdpf_columns = (
        "wind_speed_m_per_s",
        "wind_angle_degree",
        "conductor_outer_diameter_m",
        "air_temperature_degree_celsius",
        "reference_temperature_degree_celsius",
        "solar_radiation_w_per_sq_m",
        "solar_absorptivity",
        "emissivity",
        "r_theta_kelvin_per_mw",
        "mc_joule_per_m_k",
    )
    tdpf_parameters = {c: kwargs.pop(c) for c in tdpf_columns if c in kwargs}
    for column, value in tdpf_parameters.items():
        _add_to_entries_if_not_nan(net, "line", entries, index, column, value, float64)

    _set_multiple_entries(net, "line", index, entries=entries)

    _add_multiple_branch_geodata(net, geodata, index)

    return index


def create_lines_dc_from_parameters(
    net: pandapowerNet,
    from_buses_dc: Sequence,
    to_buses_dc: Sequence,
    length_km: float | Iterable[float],
    r_ohm_per_km: float | Iterable[float],
    max_i_ka: float | Iterable[float],
    name: Iterable[str] | None = None,
    index: Int | Iterable[Int] | None = None,
    type: LineType | Iterable[str] | None = None,
    geodata: Iterable[Iterable[tuple[float, float]]] | None = None,
    in_service: bool | Iterable[bool] = True,
    df: float | Iterable[float] = 1.0,
    parallel: int | Iterable[int] = 1,
    g_us_per_km: float | Iterable[float] = 0.0,
    max_loading_percent: float | Iterable[float] = nan,
    alpha: float = nan,
    temperature_degree_celsius: float = nan,
    **kwargs,
) -> npt.NDArray[Int]:
    """
    Convenience function for creating many dc lines at once. Parameters 'from_buses_dc' and 'to_buses_dc'
        must be arrays of equal length. Other parameters may be either arrays of the same length or
        single or values.

    INPUT:
        **net** - The net within this dc lines should be created

        **from_buses_dc** (list of int) - ID of the dc buses on one side which the dc lines will be connected with

        **to_buses_dc** (list of int) - ID of the dc buses on the other side which the dc lines will be connected\
            with

        **length_km** (list of float) - The dc line length in km

        **r_ohm_per_km** (list of float) - dc line resistance in ohm per km

        **max_i_ka** (list of float) - maximum thermal current in kilo Ampere

    OPTIONAL:
        **name** (list of string, None) - A custom name for this dc line

        **index** (list of int, None) - Force a specified ID if it is available. If None, the\
            index one higher than the highest already existing index is selected.

        **in_service** (list of boolean, True) - True for in_service or False for out of service

        **type** (list of str, None) - type of dc line ("ol" for overhead dc line or "cs" for cable system)

        **df** (list of float, 1) - derating factor: maximum current of line in relation to nominal current\
            of line (from 0 to 1)

        **g_us_per_km** (list of float, 0) - dielectric conductance in micro Siemens per km

        **parallel** (list of integer, 1) - number of parallel line systems

        **geodata**
        (array, default None, shape= (,2L)) -
        The linegeodata of the dc lines. The first row should be the coordinates
        of dc bus a and the last should be the coordinates of dc bus b. The points
        in the middle represent the bending points of the line

        **max_loading_percent (list of float)** - maximum current loading (only needed for OPF)

        **alpha (float)** - temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0)))

        **temperature_degree_celsius (float)** - line temperature for which line resistance is adjusted

        **tdpf (bool)** - whether the line is considered in the TDPF calculation

        **wind_speed_m_per_s (float)** - wind speed at the line in m/s (TDPF)

        **wind_angle_degree (float)** - angle of attack between the wind direction and the line (TDPF)

        **conductor_outer_diameter_m (float)** - outer diameter of the line conductor in m (TDPF)

        **air_temperature_degree_celsius (float)** - ambient temperature in °C (TDPF)

        **reference_temperature_degree_celsius (float)** - reference temperature in °C for which \
            r_ohm_per_km for the line is specified (TDPF)

        **solar_radiation_w_per_sq_m (float)** - solar radiation on horizontal plane in W/m² (TDPF)

        **solar_absorptivity (float)** - Albedo factor for absorptivity of the lines (TDPF)

        **emissivity (float)** - Albedo factor for emissivity of the lines (TDPF)

        **r_theta_kelvin_per_mw (float)** - thermal resistance of the line (TDPF, only for \
            simplified method)

        **mc_joule_per_m_k (float)** - specific mass of the conductor multiplied by the specific \
            thermal capacity of the material (TDPF, only for thermal inertia consideration with \
            tdpf_delay_s parameter)

    OUTPUT:
        **index** (list of int) - The list of IDs of the created dc lines

    EXAMPLE:
        create_lines_dc_from_parameters(net, from_buses_dc=[0,1], to_buses_dc=[2,3], length_km=0.1,
        r_ohm_per_km=.01, max_i_ka=0.4, name=["line_dc1","line_dc2"])

    """
    _check_multiple_branch_elements(
        net, from_buses_dc, to_buses_dc, "Lines_dc", node_name="bus_dc", plural="(all dc buses)"
    )

    index = _get_multiple_index_with_check(net, "line", index, len(from_buses_dc))

    entries = {
        "from_bus_dc": from_buses_dc,
        "to_bus_dc": to_buses_dc,
        "length_km": length_km,
        "type": type,
        "r_ohm_per_km": r_ohm_per_km,
        "max_i_ka": max_i_ka,
        "g_us_per_km": g_us_per_km,
        "name": name,
        "df": df,
        "parallel": parallel,
        "in_service": in_service,
        **kwargs,
    }

    _add_to_entries_if_not_nan(net, "line_dc", entries, index, "max_loading_percent", max_loading_percent)
    _add_to_entries_if_not_nan(net, "line_dc", entries, index, "temperature_degree_celsius", temperature_degree_celsius)
    _add_to_entries_if_not_nan(net, "line_dc", entries, index, "alpha", alpha)

    # add optional columns for TDPF if parameters passed to kwargs:
    _add_to_entries_if_not_nan(net, "line_dc", entries, index, "tdpf", kwargs.get("tdpf"), bool_)
    tdpf_columns = (
        "wind_speed_m_per_s",
        "wind_angle_degree",
        "conductor_outer_diameter_m",
        "air_temperature_degree_celsius",
        "reference_temperature_degree_celsius",
        "solar_radiation_w_per_sq_m",
        "solar_absorptivity",
        "emissivity",
        "r_theta_kelvin_per_mw",
        "mc_joule_per_m_k",
    )
    tdpf_parameters = {c: kwargs.pop(c) for c in tdpf_columns if c in kwargs}
    for column, value in tdpf_parameters.items():
        _add_to_entries_if_not_nan(net, "line_dc", entries, index, column, value, float64)

    _set_multiple_entries(net, "line_dc", index, entries=entries)

    _add_multiple_branch_geodata(net, geodata, index, "line_dc")

    return index


def create_dcline(
    net: pandapowerNet,
    from_bus: Int,
    to_bus: Int,
    p_mw: float,
    loss_percent: float,
    loss_mw: float,
    vm_from_pu: float,
    vm_to_pu: float,
    index: Int | None = None,
    name: str | None = None,
    max_p_mw: float = nan,
    min_q_from_mvar: float = nan,
    min_q_to_mvar: float = nan,
    max_q_from_mvar: float = nan,
    max_q_to_mvar: float = nan,
    in_service: bool = True,
    **kwargs,
) -> Int:
    """
    Creates a dc line.

    INPUT:
        **from_bus** (int) - ID of the bus on one side which the line will be connected with

        **to_bus** (int) - ID of the bus on the other side which the line will be connected with

        **p_mw** - (float) Active power transmitted from 'from_bus' to 'to_bus'

        **loss_percent** - (float) Relative transmission loss in percent of active power
            transmission

        **loss_mw** - (float) Total transmission loss in MW

        **vm_from_pu** - (float) Voltage set point at from bus

        **vm_to_pu** - (float) Voltage set point at to bus

    OPTIONAL:
        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **name** (str, None) - A custom name for this dc line

        **in_service** (boolean) - True for in_service or False for out of service

        **max_p_mw** - Maximum active power flow. Only respected for OPF

        **min_q_from_mvar** - Minimum reactive power at from bus. Necessary for OPF

        **min_q_to_mvar** - Minimum reactive power at to bus. Necessary for OPF

        **max_q_from_mvar** - Maximum reactive power at from bus. Necessary for OPF

        **max_q_to_mvar** - Maximum reactive power at to bus. Necessary for OPF

    OUTPUT:
        **index** (int) - The unique ID of the created element

    EXAMPLE:
        create_dcline(net, from_bus=0, to_bus=1, p_mw=1e4, loss_percent=1.2, loss_mw=25, \
            vm_from_pu=1.01, vm_to_pu=1.02)
    """
    index = _get_index_with_check(net, "dcline", index)

    _check_branch_element(net, "DCLine", index, from_bus, to_bus)

    entries = {
        "name": name,
        "from_bus": from_bus,
        "to_bus": to_bus,
        "p_mw": p_mw,
        "loss_percent": loss_percent,
        "loss_mw": loss_mw,
        "vm_from_pu": vm_from_pu,
        "vm_to_pu": vm_to_pu,
        "max_p_mw": max_p_mw,
        "min_q_from_mvar": min_q_from_mvar,
        "max_q_from_mvar": max_q_from_mvar,
        "max_q_to_mvar": max_q_to_mvar,
        "min_q_to_mvar": min_q_to_mvar,
        "in_service": in_service,
        **kwargs,
    }
    _set_entries(net, "dcline", index, entries=entries)

    return index
