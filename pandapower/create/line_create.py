# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging
from collections.abc import Iterable, Sequence
from operator import itemgetter
from typing import NotRequired, TypedDict

import numpy.typing as npt
from numpy import nan
from typing_extensions import deprecated

from pandapower import ensure_iterability, pandapowerNet
from pandapower.auxiliary import warn_and_fix_parameter_renaming
from pandapower.create._utils import (
    _add_multiple_branch_geodata,
    _add_to_entries_if_not_nan,
    _check_branch_element,
    _check_multiple_branch_elements,
    _get_index_with_check,
    _get_multiple_index_with_check,
    _set_entries,
    _set_multiple_entries,
)
from pandapower.network_schema.tools.validation.network_validation import validate_network
from pandapower.network_structure import get_default_value
from pandapower.pp_types import Int, LineType
from pandapower.std_types import load_std_type

logger = logging.getLogger(__name__)

G_US_PER_KM_DEFAULT = get_default_value("line", "g_us_per_km")
G0_US_PER_KM_DEFAULT = get_default_value("line", "g0_us_per_km")


class LineParams(TypedDict):
    """
    Parameters for the create_lines method (alternative to std_type)
    """

    r_ohm_per_km: float | Iterable[float]
    """line resistance in ohm per km"""
    x_ohm_per_km: float | Iterable[float]
    """line reactance in ohm per km"""
    c_nf_per_km: float | Iterable[float]
    """line capacitance (line-to-earth) in nano Farad per km"""
    g_us_per_km: NotRequired[float | Iterable[float]]  # filled by default value G_US_PER_KM_DEFAULT
    """dielectric conductance in micro Siemens per km. If not provided a default is used if column is present."""
    max_i_ka: float | Iterable[float]
    """maximum thermal current in kilo Ampere"""
    type: NotRequired[LineType | Iterable[str]]
    """type of line ("ol" for overhead line or "cs" for cable system)"""
    r0_ohm_per_km: NotRequired[float | Iterable[float]]
    """zero sequence line resistance in ohm per km"""
    x0_ohm_per_km: NotRequired[float | Iterable[float]]
    """zero sequence line reactance in ohm per km"""
    c0_nf_per_km: NotRequired[float | Iterable[float]]
    """zero sequence line capacitance in nano Farad per km"""
    g0_us_per_km: NotRequired[float | Iterable[float]]  # filled by default value G0_US_PER_KM_DEFAULT
    """
    zero sequence dielectric conductance in micro Siemens per km.
    If not provided a default is used if column is present.
    """
    endtemp_degree: NotRequired[float | Iterable[float]]


def create_lines(
    net: pandapowerNet,
    from_buses: Int | Sequence[Int],
    to_buses: Int | Sequence[Int],
    length_km: float | Iterable[float],
    # None is only for deprecation support and should be removed once std_type is fully deprecated
    line_params: LineParams | str | Iterable[str] | None = None,
    name: Iterable[str] | None = None,
    index: Int | Iterable[Int] | None = None,
    geodata: Iterable[Iterable[tuple[float, float]]] | Iterable[tuple[float, float]] | None = None,
    df: float | Iterable[float] = get_default_value("line", "df"),
    parallel: int | Iterable[int] = get_default_value("line", "parallel"),
    in_service: bool | Iterable[bool] = get_default_value("line", "in_service"),
    max_loading_percent: float | Iterable[float] = nan,
    alpha: float | Iterable[float] = nan,
    temperature_degree_celsius: float | Iterable[float] = nan,
    skip_validation: bool = False,
    **kwargs,
) -> npt.NDArray[Int]:
    """
    Convenience function for creating many lines at once. Parameters 'from_buses' and 'to_buses'
    must be arrays of equal length. Other parameters may be either arrays of the same length or
    single or values. In any case the line parameters are defined through a single standard
    type, so all lines have the same standard type.


    Parameters:
        net: The net within this line should be created
        from_buses: ID of the bus on one side which the line will be connected with
        to_buses: ID of the bus on the other side which the line will be connected with
        length_km: The line length in km
        line_params: The std_type of the lines or a LineParams dict.
        name: A custom name for this line
        index: Force a specified ID if it is available. If None, the index one higher than the highest already existing
            index is selected.
        geodata: The geodata of the line. The first element should be the coordinates of from_bus and the last should be
            the coordinates of to_bus. The points in the middle represent the bending points of the line
        in_service: True for in_service or False for out of service
        df: derating factor: maximum current of line in relation to nominal current of line (from 0 to 1)
        parallel: number of parallel line systems
        max_loading_percent: maximum current loading (only needed for OPF)
        alpha: temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0))
            .. attention::
                This column will only be filled from std_type if the column already exists in the line DataFrame
        temperature_degree_celsius: line temperature for which line resistance is adjusted
        skip_validation: if set to true validate_network will not be run after creating.
            (Only use this if performance is critical)

    Keyword Arguments:
        alpha (float): temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0))
        temperature_degree_celsius (float): line temperature for which line resistance is adjusted
        tdpf (bool): whether the line is considered in the TDPF calculation
        wind_speed_m_per_s (float): wind speed at the line in m/s (TDPF)
        wind_angle_degree (float): angle of attack between the wind direction and the line (TDPF)
        conductor_outer_diameter_m (float): outer diameter of the line conductor in m (TDPF)
        air_temperature_degree_celsius (float): ambient temperature in °C (TDPF)
        reference_temperature_degree_celsius (float): reference temperature in °C for which r_ohm_per_km for the line is
            specified (TDPF)
        solar_radiation_w_per_sq_m (float): solar radiation on horizontal plane in W/m² (TDPF)
        solar_absorptivity (float): Albedo factor for absorptivity of the lines (TDPF)
        emissivity (float): Albedo factor for emissivity of the lines (TDPF)
        r_theta_kelvin_per_mw (float): thermal resistance of the line (TDPF, only for simplified method)
        mc_joule_per_m_k (float): specific mass of the conductor multiplied by the specific thermal capacity of the
            material (TDPF, only for thermal inertia consideration with tdpf_delay_s parameter)

        Returns:
            The unique ID of the created lines

        Example:
            >>> create_lines(
            >>>   net, from_buses=[0,1], to_buses=[2,3], length_km=0.1, std_type="NAYY 4x50 SE", name=["line1", "line2"]
            >>> )
    """
    # std_type deprecated as of v4.0.0
    line_params, kwargs = warn_and_fix_parameter_renaming("std_type", "line_params", line_params, None, kwargs)
    if line_params is None:
        raise AttributeError("line_params may not be None")

    from_buses = ensure_iterability(from_buses)
    to_buses = ensure_iterability(to_buses)

    _check_multiple_branch_elements(net, from_buses, to_buses, "Lines")

    index = _get_multiple_index_with_check(net, "line", index, len(from_buses))
    index = ensure_iterability(index)

    entries = {
        "from_bus": from_buses,
        "to_bus": to_buses,
        "length_km": length_km,
        "name": name,
        "df": df,
        "parallel": parallel,
        "in_service": in_service,
        **kwargs,
    }
    if isinstance(line_params, dict):
        if "g_us_per_km" not in line_params:
            line_params["g_us_per_km"] = G_US_PER_KM_DEFAULT
        if "g0_us_per_km" in net.line and "g0_us_per_km" not in line_params:
            line_params["g0_us_per_km"] = G0_US_PER_KM_DEFAULT

        for param in ("r0_ohm_per_km", "x0_ohm_per_km", "c0_nf_per_km", "endtemp_degree"):
            value = line_params.pop(param, nan)  # type: ignore[misc]
            _add_to_entries_if_not_nan(net, "line", entries, index, param, value)

        entries.update(line_params)
    # add std type data
    elif isinstance(line_params, str):
        entries["std_type"] = line_params
        lineparam = load_std_type(net, line_params, "line")
        entries["r_ohm_per_km"] = lineparam["r_ohm_per_km"]
        entries["x_ohm_per_km"] = lineparam["x_ohm_per_km"]
        entries["c_nf_per_km"] = lineparam["c_nf_per_km"]
        entries["max_i_ka"] = lineparam["max_i_ka"]
        entries["g_us_per_km"] = lineparam.get("g_us_per_km", G_US_PER_KM_DEFAULT)
        if "alpha" in net.line.columns and "alpha" in lineparam:
            entries["alpha"] = lineparam["alpha"]
        if "type" in lineparam:
            entries["type"] = lineparam["type"]
    elif isinstance(line_params, Iterable):  # Iterable of str (std_type)
        entries["std_type"] = line_params

        lineparam = list(map(load_std_type, [net] * len(index), line_params, ["line"] * len(index)))
        entries["r_ohm_per_km"] = list(map(itemgetter("r_ohm_per_km"), lineparam))
        entries["x_ohm_per_km"] = list(map(itemgetter("x_ohm_per_km"), lineparam))
        entries["c_nf_per_km"] = list(map(itemgetter("c_nf_per_km"), lineparam))
        entries["max_i_ka"] = list(map(itemgetter("max_i_ka"), lineparam))
        entries["g_us_per_km"] = [line_param_dict.get("g_us_per_km", 0) for line_param_dict in lineparam]
        entries["type"] = [line_param_dict.get("type", None) for line_param_dict in lineparam]
    else:
        raise TypeError(f"line_params is not a valid type: {type(line_params)}")

    _add_to_entries_if_not_nan(net, "line", entries, index, "max_loading_percent", max_loading_percent)
    _add_to_entries_if_not_nan(net, "line", entries, index, "alpha", alpha)
    _add_to_entries_if_not_nan(net, "line", entries, index, "temperature_degree_celsius", temperature_degree_celsius)

    # add optional columns for TDPF if parameters passed to kwargs:
    _add_to_entries_if_not_nan(net, "line", entries, index, "tdpf", kwargs.get("tdpf"))
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
        _add_to_entries_if_not_nan(net, "line", entries, index, column, value)

    _set_multiple_entries(net, "line", index, entries=entries)

    _add_multiple_branch_geodata(net, geodata, index)

    # FIXME: fix test failing validation
    # if not skip_validation:
    #     validate_network(net)

    return index


@deprecated("Use create_lines instead.")  # deprecate since v4.0.0
def create_line(
    net: pandapowerNet,
    from_bus: Int,
    to_bus: Int,
    length_km: float,
    std_type: str,
    name: str | None = None,
    index: Int | None = None,
    geodata: Iterable[tuple[float, float]] | None = None,
    df: float = get_default_value("line", "df"),
    parallel: int = get_default_value("line", "parallel"),
    in_service: bool = get_default_value("line", "in_service"),
    max_loading_percent: float = nan,
    alpha: float = nan,
    temperature_degree_celsius: float = nan,
    **kwargs,
) -> Int:
    """
    Creates a line element in net["line"]
    The line parameters are defined through the standard type library.

    Parameters:
        net: The net within this line should be created
        from_bus: ID of the bus on one side which the line will be connected with
        to_bus: ID of the bus on the other side which the line will be connected with
        length_km: The line length in km
        std_type: Name of a standard line type:
            - Pre-defined in standard_linetypes
            - Customized std_type made using **create_std_type()**
        name: A custom name for this line
        index: Force a specified ID if it is available. If None, the index one higher than the highest already existing
            index is selected.
        geodata: Iterable[Tuple[int, int]|Tuple[float, float]], The geodata of the line. The first element should be the
            coordinates of from_bus and the last should be the coordinates of to_bus. The points in the middle represent
            the bending points of the line
        in_service: True for in_service or False for out of service
        df: derating factor: maximum current of line in relation to nominal current of line (from 0 to 1)
        parallel: number of parallel line systems
        max_loading_percent: maximum current loading (only needed for OPF)
        alpha: temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0))
        temperature_degree_celsius: line temperature for which line resistance is adjusted

    Keyword arguments:
        tdpf (bool): whether the line is considered in the TDPF calculation
        wind_speed_m_per_s (float): wind speed at the line in m/s (TDPF)
        wind_angle_degree (float): angle of attack between the wind direction and the line (TDPF)
        conductor_outer_diameter_m (float): outer diameter of the line conductor in m (TDPF)
        air_temperature_degree_celsius (float): ambient temperature in °C (TDPF)
        reference_temperature_degree_celsius (float): reference temperature in °C for which r_ohm_per_km for the line is
            specified (TDPF)
        solar_radiation_w_per_sq_m (float): solar radiation on horizontal plane in W/m² (TDPF)
        solar_absorptivity: Albedo factor for absorptivity of the lines (TDPF)
        emissivity (float): Albedo factor for emissivity of the lines (TDPF)
        r_theta_kelvin_per_mw (float): thermal resistance of the line (TDPF, only for simplified method)
        mc_joule_per_m_k (float): specific mass of the conductor multiplied by the specific thermal capacity of the
            material (TDPF, only for thermal inertia consideration with tdpf_delay_s parameter)

    Returns:
        The unique ID of the created line

    Example:
        >>> create_line(net, from_bus=0, to_bus=1, length_km=0.1,  std_type="NAYY 4x50 SE", name="line1")
    """
    return create_lines(
        net,
        from_bus,
        to_bus,
        length_km,
        std_type,
        [name] if name is not None else None,
        [index] if index is not None else None,
        geodata,
        df,
        parallel,
        in_service,
        max_loading_percent,
        alpha,
        temperature_degree_celsius,
        **kwargs,
    )[0]


@deprecated("Use create_lines_dc instead.")  # deprecate since v4.0.0
def create_line_dc(
    net: pandapowerNet,
    from_bus_dc: Int,
    to_bus_dc: Int,
    length_km: float,
    std_type: str,
    name: str | None = None,
    index: Int | None = None,
    geodata: Iterable[tuple[float, float]] | None = None,
    df: float = get_default_value("line", "df"),
    parallel: int = get_default_value("line", "parallel"),
    in_service: bool = get_default_value("line", "in_service"),
    max_loading_percent: float = nan,
    alpha: float = nan,
    temperature_degree_celsius: float = nan,
    **kwargs,
) -> Int:
    """
    Creates a line element in net["line_dc"]
    The line_dc parameters are defined through the standard type library.

    Parameters:
        net: The net within this line should be created
        from_bus_dc: ID of the bus_dc on one side which the line will be connected with
        to_bus_dc: ID of the bus_dc on the other side which the line will be connected with
        length_km: The line length in km
        std_type: Name of a standard line type:
            - Pre-defined in standard_linetypes
            - Customized std_type made using **create_std_type()**
        name: A custom name for this line_dc
        index: Force a specified ID if it is available. If None, the index one higher than the highest already existing
            index is selected.
        geodata: The line geodata of the line_dc. The first row should be the coordinates of bus a and the last should
            be the coordinates of bus b. The points in the middle represent the bending points of the line
        in_service: True for in_service or False for out of service
        df: derating factor, maximum current of line_dc in relation to nominal current of line (from 0 to 1)
        parallel: number of parallel line systems
        max_loading_percent: maximum current loading (only needed for OPF)
        alpha: temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0))
        temperature_degree_celsius: line temperature for which line resistance is adjusted

    Keyword Arguments:
        tdpf (bool): whether the line is considered in the TDPF calculation
        wind_speed_m_per_s (float): wind speed at the line in m/s (TDPF)
        wind_angle_degree (float): angle of attack between the wind direction and the line (TDPF)
        conductor_outer_diameter_m (float): outer diameter of the line conductor in m (TDPF)
        air_temperature_degree_celsius (float): ambient temperature in °C (TDPF)
        reference_temperature_degree_celsius (float): reference temperature in °C for which r_ohm_per_km for the line_dc
            is specified (TDPF)
        solar_radiation_w_per_sq_m (float): solar radiation on horizontal plane in W/m² (TDPF)
        solar_absorptivity (float): Albedo factor for absorptivity of the lines (TDPF)
        emissivity (float): Albedo factor for emissivity of the lines (TDPF)
        r_theta_kelvin_per_mw (float): thermal resistance of the line (TDPF, only for simplified method)
        mc_joule_per_m_k (float): specific mass of the conductor multiplied by the specific thermal capacity of the
            material (TDPF, only for thermal inertia consideration with tdpf_delay_s parameter)

    Returns:
        The ID of the created dc line

    Example:
        >>> create_line_dc(net, from_bus_dc=0, to_bus_dc=1, length_km=0.1,  std_type="NAYY 4x50 SE", name="line_dc1")
    """
    return create_lines_dc(
        net,
        from_bus_dc,
        to_bus_dc,
        length_km,
        std_type,
        name,
        [index] if index is not None else None,
        geodata,
        df,
        parallel,
        in_service,
        max_loading_percent,
        alpha,
        temperature_degree_celsius,
        **kwargs,
    )[0]


def create_lines_dc(
    net: pandapowerNet,
    from_buses_dc: Int | Sequence[Int],
    to_buses_dc: Int | Sequence[Int],
    length_km: float | Iterable[float],
    std_type: str | Sequence[str],
    name: Iterable[str] | None = None,
    index: Int | Iterable[Int] | None = None,
    geodata: Iterable[Iterable[tuple[float, float]]] | Iterable[tuple[float, float]] | None = None,
    df: float | Iterable[float] = get_default_value("line", "df"),
    parallel: int | Iterable[int] = get_default_value("line", "parallel"),
    in_service: bool | Iterable[bool] = get_default_value("line", "in_service"),
    max_loading_percent: float | Iterable[float] = nan,
    alpha: float = nan,
    temperature_degree_celsius: float = nan,
    skip_validation: bool = False,
    **kwargs,
) -> npt.NDArray[Int]:
    """
    Convenience function for creating many dc lines at once. Parameters 'from_buses_dc' and 'to_buses_dc'
    must be arrays of equal length. Other parameters may be either arrays of the same length or
    single or values. In any case the dc line parameters are defined through a single standard
    type, so all lines have the same standard type.


    Parameters:
        net: The net within this dc line should be created
        from_buses_dc: ID of the dc buses on one side which the dc lines will be connected with
        to_buses_dc: ID of the dc buses on the other side which the dc lines will be connected with
        length_km: The dc line length in km
        std_type: The dc line type of the dc lines.
        name: A custom name for these dc lines
        index: Force a specified ID if it is available. If None, the index one higher than the highest already existing
            index is selected.
        geodata: The linegeodata of the dc line. The first row should be the coordinates of dc bus a and the last should
            be the coordinates of dc bus b. The points in the middle represent the bending points of the dc line
        in_service: True for in_service or False for out of service
        df: derating factor: maximum current of line in relation to nominal current of line (from 0 to 1)
        parallel: number of parallel line systems
        max_loading_percent: maximum current loading (only needed for OPF)
        alpha: temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0))
            .. attention::
                This column will only be filled from std_type if the column already exists in the line DataFrame
        temperature_degree_celsius: line temperature for which line resistance is adjusted
        skip_validation: if set to true validate_network will not be run after creating.
            (Only use this if performance is critical)

    Keyword Arguments:
        alpha (float): temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0))
        temperature_degree_celsius (float): line temperature for which line resistance is adjusted
        tdpf (bool): whether the line is considered in the TDPF calculation
        wind_speed_m_per_s (float): wind speed at the line in m/s (TDPF)
        wind_angle_degree (float): angle of attack between the wind direction and the line (TDPF)
        conductor_outer_diameter_m (float): outer diameter of the line conductor in m (TDPF)
        air_temperature_degree_celsius (float): ambient temperature in °C (TDPF)
        reference_temperature_degree_celsius (float): reference temperature in °C for which r_ohm_per_km for the line is
            specified (TDPF)
        solar_radiation_w_per_sq_m (float): solar radiation on horizontal plane in W/m² (TDPF)
        solar_absorptivity (float): Albedo factor for absorptivity of the lines (TDPF)
        emissivity (float): Albedo factor for emissivity of the lines (TDPF)
        r_theta_kelvin_per_mw (float): thermal resistance of the line (TDPF, only for simplified method)
        mc_joule_per_m_k (float): specific mass of the conductor multiplied by the specific thermal capacity of the
            material (TDPF, only for thermal inertia consideration with tdpf_delay_s parameter)

    Returns:
        The unique ID of the created dc lines

    Example:
        >>> create_lines_dc(net, from_buses_dc=[0,1], to_buses_dc=[2,3], length_km=0.1,
        >>>   std_type="Not specified yet", name=["line_dc1","line_dc2"]
        >>> )
    """
    from_buses_dc = ensure_iterability(from_buses_dc)
    to_buses_dc = ensure_iterability(to_buses_dc)

    _check_multiple_branch_elements(
        net, from_buses_dc, to_buses_dc, "lines_dc", node_name="bus_dc", plural="(all dc buses)"
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
        entries["g_us_per_km"] = lineparam.get("g_us_per_km", 0.0)
        if "alpha" in net.line_dc.columns and "alpha" in lineparam:
            entries["alpha"] = lineparam["alpha"]
        if "type" in lineparam:
            entries["type"] = lineparam["type"]
    else:
        lineparam = list(map(load_std_type, [net] * len(std_type), std_type, ["line_dc"] * len(std_type)))
        entries["r_ohm_per_km"] = list(map(itemgetter("r_ohm_per_km"), lineparam))
        entries["max_i_ka"] = list(map(itemgetter("max_i_ka"), lineparam))
        entries["g_us_per_km"] = [line_param_dict.get("g_us_per_km", 0) for line_param_dict in lineparam]
        entries["type"] = [line_param_dict.get("type", None) for line_param_dict in lineparam]

    _add_to_entries_if_not_nan(net, "line_dc", entries, index, "max_loading_percent", max_loading_percent)
    _add_to_entries_if_not_nan(net, "line_dc", entries, index, "alpha", alpha)
    _add_to_entries_if_not_nan(net, "line_dc", entries, index, "temperature_degree_celsius", temperature_degree_celsius)

    # add optional columns for TDPF if parameters passed to kwargs:
    _add_to_entries_if_not_nan(net, "line_dc", entries, index, "tdpf", kwargs.get("tdpf"))
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
        _add_to_entries_if_not_nan(net, "line_dc", entries, index, column, value)

    _set_multiple_entries(net, "line_dc", index, entries=entries)

    _add_multiple_branch_geodata(net, geodata, index, "line_dc")

    # FIXME: fix test failing validation
    # if not skip_validation:
    #     validate_network(net)

    return index


@deprecated("Use create_lines with a LineParams dict instead.")  # deprecate since v4.0.0
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
    in_service: bool = get_default_value("line", "in_service"),
    df: float = get_default_value("line", "df"),
    parallel: int = get_default_value("line", "parallel"),
    g_us_per_km: float = get_default_value("line", "g_us_per_km"),
    max_loading_percent: float = nan,
    alpha: float = nan,
    temperature_degree_celsius: float = nan,
    r0_ohm_per_km: float = nan,
    x0_ohm_per_km: float = nan,
    c0_nf_per_km: float = nan,
    g0_us_per_km: float = get_default_value("line", "g0_us_per_km"),
    endtemp_degree: float = nan,
    **kwargs,
) -> Int:
    """
    Creates a line element in net["line"] from line parameters.

    Parameters:
        net: The net within this line should be created
        from_bus: ID of the bus on one side which the line will be connected with
        to_bus: ID of the bus on the other side which the line will be connected with
        length_km: The line length in km
        r_ohm_per_km: line resistance in ohm per km
        x_ohm_per_km: line reactance in ohm per km
        c_nf_per_km: line capacitance (line-to-earth) in nano Farad per km
        r0_ohm_per_km: zero sequence line resistance in ohm per km
        x0_ohm_per_km: zero sequence line reactance in ohm per km
        c0_nf_per_km: zero sequence line capacitance in nano Farad per km
        max_i_ka: maximum thermal current in kilo Ampere
        name: A custom name for this line
        index: Force a specified ID if it is available. If None, the index one higher than the highest already existing
            index is selected.
        in_service: True for in_service or False for out of service
        type: type of line ("ol" for overhead line or "cs" for cable system)
        df: derating factor: maximum current of line in relation to nominal current of line (from 0 to 1)
        g_us_per_km: dielectric conductance in micro Siemens per km
        g0_us_per_km: zero sequence dielectric conductance in micro Siemens per km
        parallel: number of parallel line systems
        geodata: The geodata of the line. The first row should be the coordinates of bus a and the last should be the
            coordinates of bus b. The points in the middle represent the bending points of the line
        max_loading_percent: maximum current loading (only needed for OPF)
        alpha: temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0)))
        temperature_degree_celsius: line temperature for which line resistance is adjusted

    Keyword Arguments:
        tdpf: whether the line is considered in the TDPF calculation
        wind_speed_m_per_s: wind speed at the line in m/s (TDPF)
        wind_angle_degree: angle of attack between the wind direction and the line (TDPF)
        conductor_outer_diameter_m: outer diameter of the line conductor in m (TDPF)
        air_temperature_degree_celsius: ambient temperature in °C (TDPF)
        reference_temperature_degree_celsius: reference temperature in °C for which r_ohm_per_km for the line is
            specified (TDPF)
        solar_radiation_w_per_sq_m: solar radiation on horizontal plane in W/m² (TDPF)
        solar_absorptivity: Albedo factor for absorptivity of the lines (TDPF)
        emissivity: Albedo factor for emissivity of the lines (TDPF)
        r_theta_kelvin_per_mw: thermal resistance of the line (TDPF, only for simplified method)
        mc_joule_per_m_k: specific mass of the conductor multiplied by the specific thermal capacity of the material
            (TDPF, only for thermal inertia consideration with tdpf_delay_s parameter)

    Returns:
        The unique ID of the created line

    Example:
        >>> create_line_from_parameters(net, from_bus=0, to_bus=1, length_km=0.1,
        >>>   r_ohm_per_km=.01, x_ohm_per_km=0.05, c_nf_per_km=10, max_i_ka=0.4, name="line1"
        >>> )
    """
    params: LineParams = {
        "r_ohm_per_km": r_ohm_per_km,
        "x_ohm_per_km": x_ohm_per_km,
        "c_nf_per_km": c_nf_per_km,
        "g_us_per_km": g_us_per_km,
        "max_i_ka": max_i_ka,
        "endtemp_degree": endtemp_degree,
        "r0_ohm_per_km": r0_ohm_per_km,
        "x0_ohm_per_km": x0_ohm_per_km,
        "c0_nf_per_km": c0_nf_per_km,
        "g0_us_per_km": g0_us_per_km,
    }
    if type is not None:
        params["type"] = type
    return create_lines(
        net,
        from_bus,
        to_bus,
        length_km,
        params,
        [name] if name is not None else None,
        [index] if index is not None else None,
        geodata,
        df,
        parallel,
        in_service,
        max_loading_percent,
        alpha,
        temperature_degree_celsius,
        **kwargs,
    )[0]


@deprecated("use create_lines_dc_from_parameters instead")  # deprecate since v4.0.0
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
    in_service: bool = get_default_value("line_dc", "in_service"),
    df: float = get_default_value("line_dc", "df"),
    parallel: int = get_default_value("line_dc", "parallel"),
    max_loading_percent: float = nan,
    alpha: float = nan,
    temperature_degree_celsius: float = nan,
    g_us_per_km: float = get_default_value("line_dc", "g_us_per_km"),
    **kwargs,
) -> Int:
    """
    Creates a dc line element in net["line_dc"] from dc line parameters.

    Parameters:
        net: The net within this dc line should be created
        from_bus_dc: ID of the dc bus on one side which the dc line will be connected with
        to_bus_dc: ID of the dc bus on the other side which the dc line will be connected with
        length_km: The dc line length in km
        r_ohm_per_km: dc line resistance in ohm per km
        max_i_ka: maximum thermal current in kilo Ampere
        name: A custom name for this line
        index: Force a specified ID if it is available. If None, the index one higher than the highest already existing
            index is selected.
        in_service: True for in_service or False for out of service
        type: type of dc line ("ol" for overhead dc line or "cs" for cable system)
        df: derating factor: maximum current of dc line in relation to nominal current of line (from 0 to 1)
        g_us_per_km: dielectric conductance in micro Siemens per km
        g0_us_per_km: zero sequence dielectric conductance in micro Siemens per km
        parallel: number of parallel line systems
        geodata: The linegeodata of the dc line. The first row should be the coordinates of dc bus a and the last should
            be the coordinates of dc bus b. The points in the middle represent the bending points of the line
        max_loading_percent: maximum current loading (only needed for OPF)
        alpha: temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0)))
        temperature_degree_celsius: line temperature for which line resistance is adjusted

    Keyword Arguments:
        tdpf: whether the line is considered in the TDPF calculation
        wind_speed_m_per_s: wind speed at the line in m/s (TDPF)
        wind_angle_degree: angle of attack between the wind direction and the line (TDPF)
        conductor_outer_diameter_m: outer diameter of the line conductor in m (TDPF)
        air_temperature_degree_celsius: ambient temperature in °C (TDPF)
        reference_temperature_degree_celsius: reference temperature in °C for which r_ohm_per_km for the line is
            specified (TDPF)
        solar_radiation_w_per_sq_m: solar radiation on horizontal plane in W/m² (TDPF)
        solar_absorptivity: Albedo factor for absorptivity of the lines (TDPF)
        emissivity: Albedo factor for emissivity of the lines (TDPF)
        r_theta_kelvin_per_mw: thermal resistance of the line (TDPF, only for simplified method)
        mc_joule_per_m_k: specific mass of the conductor multiplied by the specific thermal capacity of the material
            (TDPF, only for thermal inertia consideration with tdpf_delay_s parameter)

    Returns:
        The ID of the created line

    Example:
        >>> create_line_dc_from_parameters(
        >>>   net, from_bus_dc=0, to_bus_dc=1, length_km=0.1, r_ohm_per_km=.01, max_i_ka=0.4, name="line_dc1"
        >>> )
    """
    return create_lines_dc_from_parameters(
        net,
        from_bus_dc,
        to_bus_dc,
        length_km,
        r_ohm_per_km,
        max_i_ka,
        [name] if name is not None else None,
        [index] if index is not None else None,
        type,
        geodata,
        in_service,
        df,
        parallel,
        g_us_per_km,
        max_loading_percent,
        alpha,
        temperature_degree_celsius,
        **kwargs,
    )[0]


@deprecated("use create_lines with a LineParams")  # deprecate since v4.0.0
def create_lines_from_parameters(
    net: pandapowerNet,
    from_buses: Int | Sequence[Int],
    to_buses: Int | Sequence[Int],
    length_km: float | Iterable[float],
    r_ohm_per_km: float | Iterable[float],
    x_ohm_per_km: float | Iterable[float],
    c_nf_per_km: float | Iterable[float],
    max_i_ka: float | Iterable[float],
    name: Iterable[str] | None = None,
    index: Int | Iterable[Int] | None = None,
    type: LineType | Iterable[str] | None = None,
    geodata: Iterable[Iterable[tuple[float, float]]] | Iterable[tuple[float, float]] | None = None,
    in_service: bool | Iterable[bool] = get_default_value("line", "in_service"),
    df: float | Iterable[float] = get_default_value("line", "df"),
    parallel: int | Iterable[int] = get_default_value("line", "parallel"),
    g_us_per_km: float | Iterable[float] = get_default_value("line", "g_us_per_km"),
    max_loading_percent: float | Iterable[float] = nan,
    alpha: float = nan,
    temperature_degree_celsius: float = nan,
    r0_ohm_per_km: float | Iterable[float] = nan,
    x0_ohm_per_km: float | Iterable[float] = nan,
    c0_nf_per_km: float | Iterable[float] = nan,
    g0_us_per_km: float | Iterable[float] = nan,
    endtemp_degree: float | Iterable[float] = nan,
    **kwargs,
) -> npt.NDArray[Int]:
    """
    Convenience function for creating many lines at once. Parameters 'from_buses' and 'to_buses'
    must be arrays of equal length. Other parameters may be either arrays of the same length or
    single or values.

    Parameters:
        net: The net within this line should be created
        from_buses: ID of the buses on one side which the lines will be connected with
        to_buses: ID of the buses on the other side which the lines will be connected with
        length_km: The line length in km
        r_ohm_per_km: line resistance in ohm per km
        x_ohm_per_km: line reactance in ohm per km
        c_nf_per_km: line capacitance in nano Farad per kma
        r0_ohm_per_km: zero sequence line resistance in ohm per km
        x0_ohm_per_km: zero sequence line reactance in ohm per km
        c0_nf_per_km: zero sequence line capacitance in nano Farad per km
        max_i_ka: maximum thermal current in kilo Ampere
        name: A custom name for this line
        index: Force a specified ID if it is available. If None, the index one higher than the highest already existing
            index is selected.
        in_service: True for in_service or False for out of service
        type: type of line ("ol" for overhead line or "cs" for cable system)
        df: derating factor: maximum current of line in relation to nominal current of line (from 0 to 1)
        g_us_per_km: dielectric conductance in micro Siemens per km
        g0_us_per_km: zero sequence dielectric conductance in micro Siemens per km
        parallel: number of parallel line systems
        geodata: The geodata of the line. The first row should be the coordinates of bus a and the last should be the
            coordinates of bus b. The points in the middle represent the bending points of the line
        max_loading_percent: maximum current loading (only needed for OPF)
        alpha: temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0)))
            .. attention::
                This column will only be filled from std_type if the column already exists in the line DataFrame
        temperature_degree_celsius: line temperature for which line resistance is adjusted
        endtemp_degree:

    Keyword Arguments:
        tdpf (bool): whether the line is considered in the TDPF calculation
        wind_speed_m_per_s (float): wind speed at the line in m/s (TDPF)
        wind_angle_degree (float): angle of attack between the wind direction and the line (TDPF)
        conductor_outer_diameter_m (float): outer diameter of the line conductor in m (TDPF)
        air_temperature_degree_celsius (float): ambient temperature in °C (TDPF)
        reference_temperature_degree_celsius (float): reference temperature in °C for which r_ohm_per_km for the line is
            specified (TDPF)
        solar_radiation_w_per_sq_m (float): solar radiation on horizontal plane in W/m² (TDPF)
        solar_absorptivity (float): Albedo factor for absorptivity of the lines (TDPF)
        emissivity (float): Albedo factor for emissivity of the lines (TDPF)
        r_theta_kelvin_per_mw (float): thermal resistance of the line (TDPF, only for simplified method)
        mc_joule_per_m_k (float): specific mass of the conductor multiplied by the specific thermal capacity of the
            material (TDPF, only for thermal inertia consideration with tdpf_delay_s parameter)

    Returns:
        The ID of the created lines

    Example:
        >>> create_lines_from_parameters(
        >>>   net, from_buses=[0,1], to_buses=[2,3], length_km=0.1, r_ohm_per_km=.01, x_ohm_per_km=0.05, c_nf_per_km=10,
        >>>   max_i_ka=0.4, name=["line1","line2"]
        >>> )
    """
    params: LineParams = {
        "r_ohm_per_km": r_ohm_per_km,
        "x_ohm_per_km": x_ohm_per_km,
        "c_nf_per_km": c_nf_per_km,
        "max_i_ka": max_i_ka,
        "g_us_per_km": g_us_per_km,
        "r0_ohm_per_km": r0_ohm_per_km,
        "x0_ohm_per_km": x0_ohm_per_km,
        "c0_nf_per_km": c0_nf_per_km,
        "g0_us_per_km": g0_us_per_km,
        "endtemp_degree": endtemp_degree,
    }
    if type is not None:
        params["type"] = [type] if isinstance(type, str) else type
    return create_lines(
        net,
        from_buses,
        to_buses,
        length_km,
        params,
        name,
        index,
        geodata,
        df,
        parallel,
        in_service,
        max_loading_percent,
        alpha,
        temperature_degree_celsius,
        **kwargs,
    )


def create_lines_dc_from_parameters(
    net: pandapowerNet,
    from_buses_dc: Int | Sequence[Int],
    to_buses_dc: Int | Sequence[Int],
    length_km: float | Iterable[float],
    r_ohm_per_km: float | Iterable[float],
    max_i_ka: float | Iterable[float],
    name: Iterable[str] | None = None,
    index: Int | Iterable[Int] | None = None,
    type: LineType | Iterable[str] | None = None,
    geodata: Iterable[Iterable[tuple[float, float]]] | Iterable[tuple[float, float]] | None = None,
    in_service: bool | Iterable[bool] = get_default_value("line_dc", "in_service"),
    df: float | Iterable[float] = get_default_value("line_dc", "df"),
    parallel: int | Iterable[int] = get_default_value("line_dc", "parallel"),
    g_us_per_km: float | Iterable[float] = get_default_value("line_dc", "g_us_per_km"),
    max_loading_percent: float | Iterable[float] = nan,
    alpha: float = nan,
    temperature_degree_celsius: float = nan,
    skip_validation: bool = False,
    **kwargs,
) -> npt.NDArray[Int]:
    """
    Convenience function for creating many dc lines at once. Parameters 'from_buses_dc' and 'to_buses_dc'
        must be arrays of equal length. Other parameters may be either arrays of the same length or
        single or values.

    Parameters:
        net: The net within this dc lines should be created
        from_buses_dc: ID of the dc buses on one side which the dc lines will be connected with
        to_buses_dc: ID of the dc buses on the other side which the dc lines will be connected with
        length_km: The dc line length in km
        r_ohm_per_km: dc line resistance in ohm per km
        max_i_ka: maximum thermal current in kilo Ampere
        name: A custom name for this dc line
        index: Force a specified ID if it is available. If None, the index one higher than the highest already existing
            index is selected.
        in_service: True for in_service or False for out of service
        type: type of dc line ("ol" for overhead dc line or "cs" for cable system)
        df: derating factor: maximum current of line in relation to nominal current of line (from 0 to 1)
        g_us_per_km: dielectric conductance in micro Siemens per km
        parallel: number of parallel line systems
        geodata: The line geodata of the dc lines. The first row should be the coordinates of dc bus a and the last
            should be the coordinates of dc bus b. The points in the middle represent the bending points of the line
        max_loading_percent: maximum current loading (only needed for OPF)
        alpha: temperature coefficient of resistance: R(T) = R(T_0) * (1 + alpha * (T - T_0)))
        temperature_degree_celsius: line temperature for which line resistance is adjusted
        skip_validation: if set to true validate_network will not be run after creating.
            (Only use this if performance is critical)

    Keyword Arguments:
        tdpf (bool): whether the line is considered in the TDPF calculation
        wind_speed_m_per_s (float): wind speed at the line in m/s (TDPF)
        wind_angle_degree (float): angle of attack between the wind direction and the line (TDPF)
        conductor_outer_diameter_m (float): outer diameter of the line conductor in m (TDPF)
        air_temperature_degree_celsius (float): ambient temperature in °C (TDPF)
        reference_temperature_degree_celsius (float): reference temperature in °C for which r_ohm_per_km for the line is
            specified (TDPF)
        solar_radiation_w_per_sq_m (float): solar radiation on horizontal plane in W/m² (TDPF)
        solar_absorptivity (float): Albedo factor for absorptivity of the lines (TDPF)
        emissivity (float): Albedo factor for emissivity of the lines (TDPF)
        r_theta_kelvin_per_mw (float): thermal resistance of the line (TDPF, only for simplified method)
        mc_joule_per_m_k (float): specific mass of the conductor multiplied by the specific thermal capacity of the
            material (TDPF, only for thermal inertia consideration with tdpf_delay_s parameter)

    Return:
        List of IDs of the created dc lines

    Example:
        >>> create_lines_dc_from_parameters(net, from_buses_dc=[0,1], to_buses_dc=[2,3], length_km=0.1,
        >>>   r_ohm_per_km=.01, max_i_ka=0.4, name=["line_dc1", "line_dc2"]
        >>> )
    """
    from_buses_dc = ensure_iterability(from_buses_dc)
    to_buses_dc = ensure_iterability(to_buses_dc)

    _check_multiple_branch_elements(
        net, from_buses_dc, to_buses_dc, "lines_dc", node_name="bus_dc", plural="(all dc buses)"
    )

    index = _get_multiple_index_with_check(net, "line_dc", index, len(from_buses_dc))

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
    _add_to_entries_if_not_nan(net, "line_dc", entries, index, "tdpf", kwargs.get("tdpf"))
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
        _add_to_entries_if_not_nan(net, "line_dc", entries, index, column, value)

    _set_multiple_entries(net, "line_dc", index, entries=entries)

    _add_multiple_branch_geodata(net, geodata, index, "line_dc")

    # FIXME: fix test failing validation
    # if not skip_validation:
    #     validate_network(net)

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
    in_service: bool = get_default_value("dcline", "in_service"),
    skip_validation: bool = False,
    **kwargs,
) -> Int:
    """
    Creates a dc line.

    Parameters:
        from_bus: ID of the bus on one side which the line will be connected with
        to_bus: ID of the bus on the other side which the line will be connected with
        p_mw: Active power transmitted from 'from_bus' to 'to_bus'
        loss_percent: Relative transmission loss in percent of active power transmission
        loss_mw: Total transmission loss in MW
        vm_from_pu: Voltage set point at from bus
        vm_to_pu: Voltage set point at to bus
        index: Force a specified ID if it is available. If None, the index one higher than the highest already existing
            index is selected.
        name: A custom name for this dc line
        in_service: True for in_service or False for out of service
        max_p_mw: Maximum active power flow. Only respected for OPF
        min_q_from_mvar: Minimum reactive power at from bus. Necessary for OPF
        min_q_to_mvar: Minimum reactive power at to bus. Necessary for OPF
        max_q_from_mvar: Maximum reactive power at from bus. Necessary for OPF
        max_q_to_mvar: Maximum reactive power at to bus. Necessary for OPF
        skip_validation: if set to true validate_network will not be run after creating.
            (Only use this if performance is critical)

    Return:
        ID of the created element

    Example:
        >>> create_dcline(
        >>>     net, from_bus=0, to_bus=1, p_mw=1e4, loss_percent=1.2, loss_mw=25, vm_from_pu=1.01, vm_to_pu=1.02
        >>> )
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

    # FIXME: fix test failing validation
    # if not skip_validation:
    #     validate_network(net)

    return index
