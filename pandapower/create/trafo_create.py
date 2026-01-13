# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging
import warnings
from typing import Iterable, Sequence
from functools import partial

import pandas as pd
from numpy import nan, isnan, isin, array, bool_, float64, full, integer, all as all_
import numpy.typing as npt

from pandapower import pandapowerNet
from pandapower.std_types import load_std_type
from pandapower.pp_types import HVMVLVType, HVLVType, Int, TapChangerType, TapChangerWithTabularType
from pandapower.create._utils import (
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


def create_transformer(
    net: pandapowerNet,
    hv_bus: Int,
    lv_bus: Int,
    std_type: str,
    name: str | None = None,
    tap_pos: int | float = nan,
    in_service: bool = True,
    index: Int | None = None,
    max_loading_percent: float = nan,
    parallel: int = 1,
    df: float = 1.0,
    tap_changer_type: str | None = None,
    tap_dependency_table: bool = False,
    id_characteristic_table: int | None = None,
    pt_percent: float = nan,
    oltc: bool = False,
    xn_ohm: float = nan,
    tap2_pos: int | float = nan,
    **kwargs,
) -> Int:
    """
    Creates a two-winding transformer in table net.trafo.
    The trafo parameters are defined through the standard type library.

    INPUT:
        **net** (pandapowerNet) - the net within this transformer should be created

        **hv_bus** (int) - the bus on the high-voltage side on which the transformer will be connected to

        **lv_bus** (int) - τhe bus on the low-voltage side on which the transformer will be connected to

        **std_type** (str) - τhe used standard type from the standard type library

    **Zero sequence parameters** (added through std_type for three-phase load flow):

        **vk0_percent** (float) - zero sequence relative short-circuit voltage

        **vkr0_percent** (float) - real part of zero sequence relative short-circuit voltage

        **mag0_percent** (float) - ratio between magnetizing and short circuit impedance (zero sequence)

                                   z_mag0 / z0

        **mag0_rx** (float) - zero sequence magnetizing r/x ratio

        **si0_hv_partial** (float) - zero sequence short circuit impedance distribution in hv side

    OPTIONAL:
        **name** (str, None) - a custom name for this transformer

        **tap_pos** (int, nan) - current tap position of the transformer. Defaults to the medium position (tap_neutral)

        **in_service** (boolean, True) - True for in_service or False for out of service

        **index** (int, None) - force a specified ID if it is available. If None, the index one higher than the \
                                highest already existing index is selected

        **max_loading_percent** (float) - maximum current loading (only needed for OPF)

        **parallel** (int) - number of parallel transformers

        **df** (float) - derating factor: maximum current of transformer in relation to nominal current of transformer \
                         (from 0 to 1)

        **tap_dependency_table** (boolean, False) - True if transformer parameters (voltage ratio, angle, impedance) \
            must be adjusted dependent on the tap position of the transformer. Requires the additional column \
            "id_characteristic_table". The function pandapower.control.trafo_characteristic_table_diagnostic \
            can be used for sanity checks. \
            The function pandapower.control.create_trafo_characteristic_object can be used to create \
            SplineCharacteristic objects in the net.trafo_characteristic_spline table and add the additional column \
            "id_characteristic_spline" to set up the reference to the spline characteristics.

        **id_characteristic_table** (int, None) - references the index of the characteristic from the lookup table \
                                                 net.trafo_characteristic_table

        **tap_changer_type** (str, None) - specifies the phase shifter type ("Ratio", "Symmetrical", "Ideal", \
                                           "Tabular", None: no tap changer)

        **xn_ohm** (float) - impedance of the grounding reactor (Z_N) for short circuit calculation

        **tap2_pos** (int, float, nan) - current tap position of the second tap changer of the transformer. \
                                         Defaults to the medium position (tap2_neutral)

    OUTPUT:
        **index** (int) - the unique ID of the created transformer

    EXAMPLE:
        create_transformer(net, hv_bus=0, lv_bus=1, std_type="0.4 MVA 10/0.4 kV", name="trafo1")
    """

    from pandapower.convert_format import convert_trafo_pst_logic

    # Check if bus exist to attach the trafo to
    _check_branch_element(net, "Trafo", index, hv_bus, lv_bus)

    index = _get_index_with_check(net, "trafo", index, name="transformer")

    if df <= 0:
        raise ValueError(f"derating factor 'df' must be positive: df = {df:.3f}")

    entries: dict[str, str | Int | bool | float | None] = {
        "name": name,
        "hv_bus": hv_bus,
        "lv_bus": lv_bus,
        "in_service": in_service,
        "std_type": std_type,
        **kwargs,
    }
    ti = load_std_type(net, std_type, "trafo")

    updates = {
        "sn_mva": ti["sn_mva"],
        "vn_hv_kv": ti["vn_hv_kv"],
        "vn_lv_kv": ti["vn_lv_kv"],
        "vk_percent": ti["vk_percent"],
        "vkr_percent": ti["vkr_percent"],
        "pfe_kw": ti["pfe_kw"],
        "i0_percent": ti["i0_percent"],
        "parallel": parallel,
        "df": df,
        "shift_degree": ti.get("shift_degree", 0),
    }
    for zero_param in ["vk0_percent", "vkr0_percent", "mag0_percent", "mag0_rx", "si0_hv_partial", "vector_group"]:
        if zero_param in ti:
            updates[zero_param] = ti[zero_param]
    entries.update(updates)
    for s, tap_pos_var in (("", tap_pos), ("2", tap2_pos)):  # to enable a second tap changer if available
        for tp in (
            f"tap{s}_neutral",
            f"tap{s}_max",
            f"tap{s}_min",
            f"tap{s}_side",
            f"tap{s}_step_percent",
            f"tap{s}_step_degree",
            f"tap{s}_changer_type",
        ):
            if tp in ti:
                entries[tp] = ti[tp]
        if (f"tap{s}_neutral" in entries) and (tap_pos_var is nan):
            entries[f"tap{s}_pos"] = entries[f"tap{s}_neutral"]
        elif tap_pos_var is not nan:
            entries[f"tap{s}_pos"] = tap_pos_var
            if isinstance(tap_pos_var, float):
                net.trafo[f"tap{s}_pos"] = net.trafo.get(f"tap{s}_pos", full(len(net.trafo), nan)).astype(float64)

    for key in ["tap_dependent_impedance", "vk_percent_characteristic", "vkr_percent_characteristic"]:
        if key in kwargs:
            del kwargs[key]
            warnings.warn(
                DeprecationWarning(
                    f"The {key} parameter is not supported in pandapower version 3.0 or later. "
                    f"The transformer with index {index} will be created without tap_dependent_impedance characteristics. "
                    "To set up tap-dependent characteristics for this transformer, provide the "
                    "net.trafo_characteristic_table and populate the tap_dependency_table and id_characteristic_table "
                    "parameters."
                )
            )

    _set_entries(net, "trafo", index, entries=entries)

    if any(key in kwargs for key in ["tap_phase_shifter", "tap2_phase_shifter"]):
        convert_trafo_pst_logic(net)
        warnings.warn(
            DeprecationWarning(
                "The tap_phase_shifter/tap2_phase_shifter parameters are not supported in pandapower as of version 3.0. "
                f"The transformer parameters (index {index}) have been updated to the new format."
            )
        )

    _set_value_if_not_nan(net, index, max_loading_percent, "max_loading_percent", "trafo")
    _set_value_if_not_nan(net, index, id_characteristic_table, "id_characteristic_table", "trafo", dtype="Int64")
    _set_value_if_not_nan(
        net, index, tap_dependency_table, "tap_dependency_table", "trafo", dtype=bool_, default_val=False
    )
    _set_value_if_not_nan(net, index, tap_changer_type, "tap_changer_type", "trafo", dtype=object, default_val=None)
    _set_value_if_not_nan(net, index, pt_percent, "pt_percent", "trafo")
    _set_value_if_not_nan(net, index, oltc, "oltc", "trafo", dtype=bool_, default_val=False)
    _set_value_if_not_nan(net, index, xn_ohm, "xn_ohm", "trafo")

    return index


def create_transformers(
    net: pandapowerNet,
    hv_buses: Sequence,
    lv_buses: Sequence,
    std_type: str,
    name: Iterable[str] | None = None,
    tap_pos: int | Iterable[int] | float = nan,
    in_service: bool | Iterable[bool] = True,
    index: Int | Iterable[Int] | None = None,
    max_loading_percent: float | Iterable[float] = nan,
    parallel: int | Iterable[int] = 1,
    df: float | Iterable[float] = 1.0,
    tap_changer_type: TapChangerWithTabularType | Iterable[str] | None = None,
    tap_dependency_table: bool | Iterable[bool] = False,
    id_characteristic_table: int | Iterable[int] | None = None,
    pt_percent: float | Iterable[float] = nan,
    oltc: bool | Iterable[bool] = False,
    xn_ohm: float | Iterable[float] = nan,
    tap2_pos: int | Iterable[int] | float = nan,
    **kwargs,
) -> npt.NDArray[Int]:
    """
    Creates several two-winding transformers in table net.trafo.
    Additional parameters passed will be added to the transformers dataframe. If keywords are passed that are present
    in the std_type they will override any setting from the standard type.

    :param net: the pandapower network to which the transformers should be added
    :type net: pandapower.pandapowerNet
    :param Sequence hv_buses: a Sequence of bus ids that are the high voltage buses for the transformers
    :param Sequence lv_buses: a Sequence of bus ids that are the low valtage buses for the transformers
    :param str std_type: the transformer std_type to get the not specified parameters from
    :param name: names for the transformers, default None
    :type name: Iterable[str]
    :param tap_pos: current tap position of the transformers. Defaults to the medium position (tap_neutral), default nan
    :type tap_pos: int | Iterable[int] | float
    :param in_service: Wheather the transforers are in or out of service, default True
    :type in_service: bool | Itreable[bool]
    :param index: the index to use for the new elements, default None
    :type index: Int | Iterable[Int] | None
    :param max_loading_percent: the maximum loading percentage of the transformer, default nan
    :type max_loading_percent: float | Iterable[float]
    :param parallel: number of parallel transformer, default 1
    :param df: derating factor: maximum current of transformer in relation to nominal current of transformer (0 - 1), default 1.0
    :param tap_changer_type: specifies the phase shifter type ("Ratio", "Symmetrical", "Ideal", "Tabular" or None), default None
    :param tap_dependency_table: True if sanity checks should be performed. See SplineCharacteristics, default False
    :param id_characteristic_table: id of the SplineCharacteristic, default None
    :param pt_percent: default nan
    :param oltc: default False
    :param xn_ohm: impedance of the grounding reactor (Z_N) for short circuit calculation, default nan
    :param tap2_pos: current tap position of the second tap changer ot the transformer. Defaults to the medium position (tap2_neutral), default nan

    :example:
        >>> create_transformers(
        >>>     net, hv_bus=[0, 1], lv_bus=[2, 3], std_type="0.4 MVA 10/0.4 kV", name=["trafo1", "trafo2"]
        >>> )
    """

    std_params = load_std_type(net, std_type, "trafo")

    required_params = ("sn_mva", "vn_lv_kv", "vn_hv_kv", "vk_percent", "vkr_percent", "pfe_kw")
    if not all(param in std_params for param in required_params):
        raise ValueError(f"std_type is missing a required value. Required values: {', '.join(required_params)}")
    params_from_std_type = (
        "i0_percent", "vk0_percent", "vkr0_percent", "mag0_percent", "mag0_rx", "si0_hv_partial", "vector_group",
        *required_params
    )
    params = {param: std_params[param] for param in params_from_std_type if param in std_params}
    params.update(kwargs)

    return create_transformers_from_parameters(
        net=net, hv_buses=hv_buses, lv_buses=lv_buses, name=name, tap_pos=tap_pos, in_service=in_service, index=index,
        max_loading_percent=max_loading_percent, parallel=parallel, df=df, tap_changer_type=tap_changer_type,
        tap_dependency_table=tap_dependency_table, id_characteristic_table=id_characteristic_table,
        pt_percent=pt_percent, oltc=oltc, xn_ohm=xn_ohm, tap2_pos=tap2_pos, std_type=std_type,
        **params
    )


def create_transformer_from_parameters(
    net: pandapowerNet,
    hv_bus: Int,
    lv_bus: Int,
    sn_mva: float,
    vn_hv_kv: float,
    vn_lv_kv: float,
    vkr_percent: float,
    vk_percent: float,
    pfe_kw: float,
    i0_percent: float,
    shift_degree: float = 0,
    tap_side: HVLVType | None = None,
    tap_neutral: int | float = nan,
    tap_max: int | float = nan,
    tap_min: int | float = nan,
    tap_step_percent: float = nan,
    tap_step_degree: float = nan,
    tap_pos: int | float = nan,
    tap_changer_type: TapChangerWithTabularType | None = None,
    id_characteristic_table: int | None = None,
    in_service: bool = True,
    name: str | None = None,
    vector_group: str | None = None,
    index: Int | None = None,
    max_loading_percent: float = nan,
    parallel: int = 1,
    df: float = 1.0,
    vk0_percent: float = nan,
    vkr0_percent: float = nan,
    mag0_percent: float = nan,
    mag0_rx: float = nan,
    si0_hv_partial: float = nan,
    pt_percent: float = nan,
    oltc: bool = False,
    tap_dependency_table: bool = False,
    xn_ohm: float = nan,
    tap2_side: HVLVType | None = None,
    tap2_neutral: int | float = nan,
    tap2_max: int | float = nan,
    tap2_min: int | float = nan,
    tap2_step_percent: float = nan,
    tap2_step_degree: float = nan,
    tap2_pos: int | float = nan,
    tap2_changer_type: TapChangerType | None = None,
    **kwargs,
) -> Int:
    """
    Creates a two-winding transformer in table net.trafo with the specified parameters.

    INPUT:
        **net** (pandapowerNet) - the net within this transformer should be created

        **hv_bus** (int) - the bus on the high-voltage side on which the transformer will be connected to

        **lv_bus** (int) - the bus on the low-voltage side on which the transformer will be connected to

        **sn_mva** (float) - rated apparent power

        **vn_hv_kv** (float) - rated voltage on high voltage side

        **vn_lv_kv** (float) - rated voltage on low voltage side

        **vkr_percent** (float) - real part of relative short-circuit voltage

        **vk_percent** (float) - relative short-circuit voltage

        **pfe_kw** (float)  - iron losses in kW

        **i0_percent** (float) - open loop losses in percent of rated current

        **vector_group** (str) - vector group of the transformer

                                    HV side is Uppercase letters and LV side is lower case

        **vk0_percent** (float) - zero sequence relative short-circuit voltage

        **vkr0_percent** (float) - real part of zero sequence relative short-circuit voltage

        **mag0_percent** (float) - zero sequence magnetizing impedance/ vk0

        **mag0_rx** (float) - zero sequence magnetizing R/X ratio

        **si0_hv_partial** (float) - Distribution of zero sequence leakage impedance's for HV side

    OPTIONAL:

        **in_service** (boolean) - True for in_service or False for out of service

        **parallel** (int) - number of parallel transformers

        **name** (str) - A custom name for this transformer

        **shift_degree** (float) - angle shift over the transformer*

        **tap_side** (str) - position of tap changer ("hv", "lv")

        **tap_pos** (int, nan) - current tap position of the transformer. Defaults to the medium position (tap_neutral)

        **tap_neutral** (int, nan) - tap position where the transformer ratio is equal to the ratio of the \
                                     rated voltages

        **tap_max** (int, nan) - maximum allowed tap position

        **tap_min** (int, nan) - minimum allowed tap position

        **tap_step_percent** (float) - tap step size for voltage magnitude in percent

        **tap_step_degree** (float) - tap step size for voltage angle in degree*

        **tap_changer_type** (str, None) - specifies the phase shifter type ("Ratio", "Symmetrical", "Ideal", \
                                           "Tabular", None: no tap changer)*

        **index** (int, None) - force a specified ID if it is available. If None, the index one higher than the \
                                highest already existing index is selected.

        **max_loading_percent** (float) - maximum current loading (only needed for OPF)

        **df** (float) - derating factor: maximum current of transformer in relation to nominal \
                                          current of transformer (from 0 to 1)

        **tap_dependency_table** (boolean, False) - True if transformer parameters (voltage ratio, angle, impedance) \
            must be adjusted dependent on the tap position of the transformer. Requires the additional column \
            "id_characteristic_table". The function pandapower.control.trafo_characteristic_table_diagnostic \
            can be used for sanity checks. \
            The function pandapower.control.create_trafo_characteristic_object can be used to create \
            SplineCharacteristic objects in the net.trafo_characteristic_spline table and add the additional column \
            "id_characteristic_spline" to set up the reference to the spline characteristics.

        **id_characteristic_table** (int, None) - references the index of the characteristic from the lookup table \
                                                 net.trafo_characteristic_table

        **pt_percent** (float, nan) - (short circuit only)

        **oltc** (boolean, False) - (short circuit only)

        **xn_ohm** (float) - impedance of the grounding reactor (Z_N) for short circuit calculation

        **tap2_side** (str) - position of the second tap changer ("hv", "lv")

        **tap2_pos** (int, nan) - current tap position of the second tap changer of the transformer. \
                                  Defaults to the medium position (tap2_neutral)

        **tap2_neutral** (int, nan) - second tap position where the transformer ratio is equal to the \
                                      ratio of the rated voltages

        **tap2_max** (int, nan) - maximum allowed tap position of the second tap changer

        **tap2_min** (int, nan) - minimum allowed tap position of the second tap changer

        **tap2_step_percent** (float, nan) - second tap step size for voltage magnitude in percent

        **tap2_step_degree** (float, nan) - second tap step size for voltage angle in degree*

        **tap2_changer_type** (str, None) - specifies the tap changer type ("Ratio", "Symmetrical", "Ideal", \
                                            None: no tap changer)*

        **leakage_resistance_ratio_hv** (bool) - ratio of transformer short-circuit resistance on HV side (default 0.5)

        **leakage_reactance_ratio_hv** (bool) - ratio of transformer short-circuit reactance on HV side (default 0.5)

        \\* only considered in load flow if calculate_voltage_angles = True

    OUTPUT:
        **index** (int) - the unique ID of the created transformer

    EXAMPLE:
        create_transformer_from_parameters(net, hv_bus=0, lv_bus=1, name="trafo1", sn_mva=40, vn_hv_kv=110, \
                                           vn_lv_kv=10, vk_percent=10, vkr_percent=0.3, pfe_kw=30, i0_percent=0.1, \
                                           shift_degree=30)
    """

    from pandapower.convert_format import convert_trafo_pst_logic

    # Check if bus exist to attach the trafo to
    _check_branch_element(net, "Trafo", index, hv_bus, lv_bus)

    index = _get_index_with_check(net, "trafo", index, name="transformer")

    if df <= 0:
        raise UserWarning("derating factor df must be positive: df = %.3f" % df)

    if tap_pos is nan:
        tap_pos = tap_neutral
        # store dtypes

    entries = {
        "name": name,
        "hv_bus": hv_bus,
        "lv_bus": lv_bus,
        "in_service": in_service,
        "std_type": None,
        "sn_mva": sn_mva,
        "vn_hv_kv": vn_hv_kv,
        "vn_lv_kv": vn_lv_kv,
        "vk_percent": vk_percent,
        "vkr_percent": vkr_percent,
        "pfe_kw": pfe_kw,
        "i0_percent": i0_percent,
        "tap_neutral": tap_neutral,
        "tap_max": tap_max,
        "tap_min": tap_min,
        "shift_degree": shift_degree,
        "tap_side": tap_side,
        "tap_step_percent": tap_step_percent,
        "tap_step_degree": tap_step_degree,
        "parallel": parallel,
        "df": df,
        **kwargs,
    }

    if ("tap_neutral" in entries) and (tap_pos is nan):
        entries["tap_pos"] = entries["tap_neutral"]
    else:
        entries["tap_pos"] = tap_pos
        if type(tap_pos) is float:
            net.trafo.tap_pos = net.trafo.tap_pos.astype(float)

    for key in ["tap_dependent_impedance", "vk_percent_characteristic", "vkr_percent_characteristic"]:
        if key in kwargs:
            del kwargs[key]
            warnings.warn(
                DeprecationWarning(
                    f"The {key} parameter is not supported in pandapower version 3.0 or later. "
                    f"The transformer with index {index} will be created without tap_dependent_impedance characteristics. "
                    "To set up tap-dependent characteristics for this transformer, provide the "
                    "net.trafo_characteristic_table and populate the tap_dependency_table and id_characteristic_table "
                    "parameters."
                )
            )

    entries.update(kwargs)
    _set_entries(net, "trafo", index, entries=entries)

    _set_value_if_not_nan(net, index, id_characteristic_table, "id_characteristic_table", "trafo", dtype="Int64")
    _set_value_if_not_nan(net, index, tap_changer_type, "tap_changer_type", "trafo", dtype=object, default_val=None)
    _set_value_if_not_nan(
        net, index, tap_dependency_table, "tap_dependency_table", "trafo", dtype=bool_, default_val=False
    )

    _set_value_if_not_nan(net, index, tap2_side, "tap2_side", "trafo", dtype=str)
    _set_value_if_not_nan(net, index, tap2_neutral, "tap2_neutral", "trafo", dtype=float64)
    _set_value_if_not_nan(net, index, tap2_min, "tap2_min", "trafo", dtype=float64)
    _set_value_if_not_nan(net, index, tap2_max, "tap2_max", "trafo", dtype=float64)
    _set_value_if_not_nan(net, index, tap2_step_percent, "tap2_step_percent", "trafo", dtype=float64)
    _set_value_if_not_nan(net, index, tap2_step_degree, "tap2_step_degree", "trafo", dtype=float64)
    _set_value_if_not_nan(
        net, index, tap2_pos if pd.notnull(tap2_pos) else tap2_neutral, "tap2_pos", "trafo", dtype=float64
    )
    _set_value_if_not_nan(net, index, tap2_changer_type, "tap2_changer_type", "trafo", dtype=object)

    if any(key in kwargs for key in ["tap_phase_shifter", "tap2_phase_shifter"]):
        convert_trafo_pst_logic(net)
        warnings.warn(
            DeprecationWarning(
                "The tap_phase_shifter/tap2_phase_shifter parameter is not supported in pandapower version 3.0 or later. "
                f"The transformer parameters (index {index}) have been updated to the new format."
            )
        )

    if not (
        isnan(vk0_percent)
        and isnan(vkr0_percent)
        and isnan(mag0_percent)
        and isnan(mag0_rx)
        and isnan(si0_hv_partial)
        and vector_group is None
    ):
        _set_value_if_not_nan(net, index, vk0_percent, "vk0_percent", "trafo")
        _set_value_if_not_nan(net, index, vkr0_percent, "vkr0_percent", "trafo")
        _set_value_if_not_nan(net, index, mag0_percent, "mag0_percent", "trafo")
        _set_value_if_not_nan(net, index, mag0_rx, "mag0_rx", "trafo")
        _set_value_if_not_nan(net, index, si0_hv_partial, "si0_hv_partial", "trafo")
        _set_value_if_not_nan(net, index, vector_group, "vector_group", "trafo", dtype=str)
    _set_value_if_not_nan(net, index, max_loading_percent, "max_loading_percent", "trafo")
    _set_value_if_not_nan(net, index, pt_percent, "pt_percent", "trafo")
    _set_value_if_not_nan(net, index, oltc, "oltc", "trafo", dtype=bool_, default_val=False)
    _set_value_if_not_nan(net, index, xn_ohm, "xn_ohm", "trafo")

    return index


def create_transformers_from_parameters(  # index missing ?
    net: pandapowerNet,
    hv_buses: Sequence,
    lv_buses: Sequence,
    sn_mva: float | Iterable[float],
    vn_hv_kv: float | Iterable[float],
    vn_lv_kv: float | Iterable[float],
    vkr_percent: float | Iterable[float],
    vk_percent: float | Iterable[float],
    pfe_kw: float | Iterable[float],
    i0_percent: float | Iterable[float],
    shift_degree: float | Iterable[float] = 0,
    tap_side: HVLVType | Iterable[str] | None = None,
    tap_neutral: int | Iterable[int] | float = nan,
    tap_max: int | Iterable[int] | float = nan,
    tap_min: int | Iterable[int] | float = nan,
    tap_step_percent: float | Iterable[float] = nan,
    tap_step_degree: float | Iterable[float] = nan,
    tap_pos: int | Iterable[int] | float = nan,
    tap_changer_type: TapChangerWithTabularType | Iterable[str] | None = None,
    id_characteristic_table: int | Iterable[int] | None = None,
    in_service: bool | Iterable[bool] = True,
    name: Iterable[str] | None = None,
    vector_group: str | Iterable[str] | None = None,
    index: Int | Iterable[Int] | None = None,
    max_loading_percent: float | Iterable[float] = nan,
    parallel: int | Iterable[int] = 1,
    df: float | Iterable[float] = 1.0,
    vk0_percent: float | Iterable[float] = nan,
    vkr0_percent: float | Iterable[float] = nan,
    mag0_percent: float | Iterable[float] = nan,
    mag0_rx: float | Iterable[float] = nan,
    si0_hv_partial: float | Iterable[float] = nan,
    pt_percent: float | Iterable[float] = nan,
    oltc: bool | Iterable[bool] = False,
    tap_dependency_table: bool | Iterable[bool] = False,
    xn_ohm: float | Iterable[float] = nan,
    tap2_side: HVLVType | Iterable[str] | None = None,
    tap2_neutral: int | Iterable[int] | float = nan,
    tap2_max: int | Iterable[int] | float = nan,
    tap2_min: int | Iterable[int] | float = nan,
    tap2_step_percent: float | Iterable[float] = nan,
    tap2_step_degree: float | Iterable[float] = nan,
    tap2_pos: int | Iterable[int] | float = nan,
    tap2_changer_type: TapChangerType | Iterable[str] | None = None,
    **kwargs,
) -> npt.NDArray[Int]:
    """
    Creates several two-winding transformers in table net.trafo with the specified parameters.

    INPUT:
        **net** (pandapowerNet) - the net within this transformer should be created

        **hv_bus** (list of int) - the bus on the high-voltage side on which the transformer will be connected to

        **lv_bus** (list of int) - the bus on the low-voltage side on which the transformer will be connected to

        **sn_mva** (list of float) - rated apparent power

        **vn_hv_kv** (list of float) - rated voltage on high voltage side

        **vn_lv_kv** (list of float) - rated voltage on low voltage side

        **vkr_percent** (list of float) - real part of relative short-circuit voltage

        **vk_percent** (list of float) - relative short-circuit voltage

        **pfe_kw** (list of float)  - iron losses in kW

        **i0_percent** (list of float) - open loop losses in percent of rated current

        **vector_group** (list of str) - Vector group of the transformer

            HV side is Uppercase letters and LV side is lower case

        **vk0_percent** (list of float) - zero sequence relative short-circuit voltage

        **vkr0_percent** (list of float) - real part of zero sequence relative short-circuit voltage

        **mag0_percent** (list of float) - zero sequence magnetizing impedance/ vk0

        **mag0_rx** (list of float) - zero sequence magnetizing R/X ratio

        **si0_hv_partial** (list of float) - distribution of zero sequence leakage impedance's for HV side

    OPTIONAL:

        **in_service** (list of boolean) - True for in_service or False for out of service

        **parallel** (list of int) - number of parallel transformers

        **name** (list of str) - a custom name for this transformer

        **shift_degree** (list of float) - angle shift over the transformer*

        **tap_side** (list of str) - position of tap changer ("hv", "lv")

        **tap_pos** (list of int, nan) - current tap position of the transformer. Defaults to the neutral \
                                         tap position (tap_neutral)

        **tap_neutral** (list of int, nan) - tap position where the transformer ratio is equal to the ratio \
                                             of the rated voltages

        **tap_max** (list of int, nan) - maximum allowed tap position

        **tap_min** (list of int, nan) - minimum allowed tap position

        **tap_step_percent** (list of float) - tap step size for voltage magnitude in percent

        **tap_step_degree** (list of float) - tap step size for voltage angle in degree*

        **tap_changer_type** (list of str, None) - specifies the tap changer type ("Ratio", "Symmetrical", "Ideal", \
                                                   "Tabular", None: no tap changer)*

        **index** (list of int, None) - force a specified ID if it is available. If None, the index one \
                                        higher than the highest already existing index is selected.

        **max_loading_percent (list of float)** - maximum current loading (only needed for OPF)

        **df** (list of float) - derating factor: maximum current of transformer in relation to nominal \
                                                  current of transformer (from 0 to 1)

        **tap_dependency_table** (list of boolean, False) - True if transformer parameters (voltage ratio, angle, \
            impedance) must be adjusted dependent on the tap position of the transformer. Requires the additional \
            column "id_characteristic_table". The function pandapower.control.trafo_characteristic_table_diagnostic \
            can be used for sanity checks. \
            The function pandapower.control.create_trafo_characteristic_object can be used to create \
            SplineCharacteristic objects in the net.trafo_characteristic_spline table and add the additional column \
            "id_characteristic_spline" to set up the reference to the spline characteristics.

        **id_characteristic_table** (list of int, None) - references the index of the characteristic from the lookup \
            table net.trafo_characteristic_table

        **pt_percent** (list of float, nan) - (short circuit only)

        **oltc** (list of bool, False) - (short circuit only)

        **xn_ohm** (list of float) - impedance of the grounding reactor (Z_N) for short circuit calculation

        **tap2_side** (list of str) - position of the second tap changer ("hv", "lv")

        **tap2_pos** (list of int, nan) - current tap position of the second tap changer of the transformer. \
                                          Defaults to the medium position (tap2_neutral)

        **tap2_neutral** (list of int, nan) - second tap position where the transformer ratio is equal to the \
                                              ratio of the rated voltages

        **tap2_max** (list of int, nan) - maximum allowed tap position of the second tap changer

        **tap2_min** (list of int, nan) - minimum allowed tap position of the second tap changer

        **tap2_step_percent** (list of float) - second tap step size for voltage magnitude in percent

        **tap2_step_degree** (list of float) - second tap step size for voltage angle in degree*

        **tap2_changer_type** (list of str, None) - specifies the tap changer type ("Ratio", "Symmetrical", "Ideal", \
                                                    None: no tap changer)*

        \\* only considered in load flow if calculate_voltage_angles = True

    OUTPUT:
        **index** (list of int) - The list of IDs of the created transformers

    EXAMPLE:
        create_transformers_from_parameters(net, hv_bus=[0, 1], lv_bus=[2, 3], name="trafo1", sn_mva=40, \
                                            vn_hv_kv=110, vn_lv_kv=10, vk_percent=10, vkr_percent=0.3, pfe_kw=30, \
                                            i0_percent=0.1, shift_degree=30)
    """

    from pandapower.convert_format import convert_trafo_pst_logic

    _check_multiple_branch_elements(net, hv_buses, lv_buses, "Transformers")

    index = _get_multiple_index_with_check(net, "trafo", index, len(hv_buses))

    tp_neutral = pd.Series(tap_neutral, index=index, dtype=float64)
    tp_pos = pd.Series(tap_pos, index=index, dtype=float64).fillna(tp_neutral)
    entries = {
        "name": name,
        "hv_bus": hv_buses,
        "lv_bus": lv_buses,
        "in_service": array(in_service).astype(bool_),
        "std_type": None,
        "sn_mva": sn_mva,
        "vn_hv_kv": vn_hv_kv,
        "vn_lv_kv": vn_lv_kv,
        "vk_percent": vk_percent,
        "vkr_percent": vkr_percent,
        "pfe_kw": pfe_kw,
        "i0_percent": i0_percent,
        "tap_neutral": tp_neutral,
        "tap_max": tap_max,
        "tap_min": tap_min,
        "shift_degree": shift_degree,
        "tap_pos": tp_pos,
        "tap_side": tap_side,
        "tap_step_percent": tap_step_percent,
        "tap_step_degree": tap_step_degree,
        "tap_changer_type": tap_changer_type,
        "parallel": parallel,
        "df": df,
        "tap_dependency_table": tap_dependency_table,
        **kwargs,
    }

    _add_to_entries_if_not_nan(
        net, "trafo", entries, index, "id_characteristic_table", id_characteristic_table, dtype="Int64"
    )
    _add_to_entries_if_not_nan(net, "trafo", entries, index, "vk0_percent", vk0_percent)
    _add_to_entries_if_not_nan(net, "trafo", entries, index, "vkr0_percent", vkr0_percent)
    _add_to_entries_if_not_nan(net, "trafo", entries, index, "mag0_percent", mag0_percent)
    _add_to_entries_if_not_nan(net, "trafo", entries, index, "mag0_rx", mag0_rx)
    _add_to_entries_if_not_nan(net, "trafo", entries, index, "si0_hv_partial", si0_hv_partial)
    _add_to_entries_if_not_nan(net, "trafo", entries, index, "max_loading_percent", max_loading_percent)
    _add_to_entries_if_not_nan(net, "trafo", entries, index, "vector_group", vector_group, dtype=str)
    _add_to_entries_if_not_nan(net, "trafo", entries, index, "oltc", oltc, bool_, False)
    _add_to_entries_if_not_nan(net, "trafo", entries, index, "pt_percent", pt_percent)
    _add_to_entries_if_not_nan(net, "trafo", entries, index, "xn_ohm", xn_ohm)

    _add_to_entries_if_not_nan(net, "trafo", entries, index, "tap2_side", tap2_side, dtype=str)
    _add_to_entries_if_not_nan(net, "trafo", entries, index, "tap2_neutral", tap2_neutral)
    _add_to_entries_if_not_nan(net, "trafo", entries, index, "tap2_min", tap2_min)
    _add_to_entries_if_not_nan(net, "trafo", entries, index, "tap2_max", tap2_max)
    _add_to_entries_if_not_nan(net, "trafo", entries, index, "tap2_step_percent", tap2_step_percent)
    _add_to_entries_if_not_nan(net, "trafo", entries, index, "tap2_step_degree", tap2_step_degree)
    _add_to_entries_if_not_nan(net, "trafo", entries, index, "tap2_pos", tap2_pos)
    _add_to_entries_if_not_nan(net, "trafo", entries, index, "tap2_changer_type", tap2_changer_type, dtype=object)

    defaults_to_fill = [("tap_dependency_table", False)]

    for key in ["tap_dependent_impedance", "vk_percent_characteristic", "vkr_percent_characteristic"]:
        if key in kwargs:
            del kwargs[key]
            warnings.warn(
                DeprecationWarning(
                    f"The {key} parameter is not supported in pandapower version 3.0 or later. "
                    f"The transformer with index {index} will be created without tap_dependent_impedance characteristics. "
                    "To set up tap-dependent characteristics for this transformer, provide the "
                    "net.trafo_characteristic_table and populate the tap_dependency_table and id_characteristic_table "
                    "parameters."
                )
            )

    _set_multiple_entries(net, "trafo", index, defaults_to_fill=defaults_to_fill, entries=entries)

    if any(key in kwargs for key in ["tap_phase_shifter", "tap2_phase_shifter"]):
        convert_trafo_pst_logic(net)
        warnings.warn(
            DeprecationWarning(
                "The tap_phase_shifter/tap2_phase_shifter parameter is not supported in pandapower version 3.0 or later. "
                f"The transformer parameters (index {index}) have been updated to the new format."
            )
        )

    return index


def create_transformer3w(
    net: pandapowerNet,
    hv_bus: Int,
    mv_bus: Int,
    lv_bus: Int,
    std_type: str,
    name: str | None = None,
    tap_pos: int | float = nan,
    in_service: bool = True,
    index: Int | None = None,
    max_loading_percent: float = nan,
    tap_changer_type: TapChangerWithTabularType | None = None,
    tap_at_star_point: bool = False,
    tap_dependency_table: bool = False,
    id_characteristic_table: int | None = None,
    **kwargs,
) -> Int:
    """
    Creates a three-winding transformer in table net.trafo3w.
    The trafo parameters are defined through the standard type library.

    INPUT:
        **net** (pandapowerNet) - the net within this transformer should be created

        **hv_bus** (int) - The bus on the high-voltage side on which the transformer will be connected to

        **mv_bus** (int) - The medium voltage bus on which the transformer will be connected to

        **lv_bus** (int) - The bus on the low-voltage side on which the transformer will be connected to

        **std_type** (str) - the used standard type from the standard type library

    OPTIONAL:
        **name** (str) - a custom name for this transformer

        **tap_pos** (int, nan) - current tap position of the transformer. Defaults to the medium position (tap_neutral)

        **tap_changer_type** (str, None) - specifies the tap changer type ("Ratio", "Symmetrical", "Ideal", "Tabular", \
                                           None: no tap changer)*

        **tap_at_star_point** (boolean) - whether tap changer is located at the star point of the 3w-transformer \
                                          or at the bus

        **in_service** (boolean) - True for in_service or False for out of service

        **index** (int, None) - force a specified ID if it is available. If None, the index one \
                                higher than the highest already existing index is selected.

        **max_loading_percent** (float) - maximum current loading (only needed for OPF)

        **tap_at_star_point** (bool) - whether tap changer is modelled at star point or at the bus

        **tap_dependency_table** (boolean, False) - True if transformer parameters (voltage ratio, angle, impedance) \
            must be adjusted dependent on the tap position of the transformer. Requires the additional column \
            "id_characteristic_table". The function pandapower.control.trafo_characteristic_table_diagnostic \
            can be used for sanity checks. \
            The function pandapower.control.create_trafo_characteristic_object can be used to create \
            SplineCharacteristic objects in the net.trafo_characteristic_spline table and add the additional column \
            "id_characteristic_spline" to set up the reference to the spline characteristics.

        **id_characteristic_table** (int, None) - references the index of the characteristic from the lookup table \
                                                 net.trafo_characteristic_table

    OUTPUT:
        **index** (int) - the unique ID of the created transformer

    EXAMPLE:
        create_transformer3w(net, hv_bus=0, mv_bus=1, lv_bus=2, name="trafo1", std_type="63/25/38 MVA 110/20/10 kV")
    """

    # Check if bus exist to attach the trafo to
    for b in [hv_bus, mv_bus, lv_bus]:
        if b not in net["bus"].index.values:
            raise UserWarning("Trafo tries to attach to bus %s" % b)

    entries: dict[str, str | None | Int | bool | float] = {
        "name": name,
        "hv_bus": hv_bus,
        "mv_bus": mv_bus,
        "lv_bus": lv_bus,
        "in_service": in_service,
        "std_type": std_type,
    }
    ti = load_std_type(net, std_type, "trafo3w")

    index = _get_index_with_check(net, "trafo3w", index, "three winding transformer")

    entries.update(
        {
            "sn_hv_mva": ti["sn_hv_mva"],
            "sn_mv_mva": ti["sn_mv_mva"],
            "sn_lv_mva": ti["sn_lv_mva"],
            "vn_hv_kv": ti["vn_hv_kv"],
            "vn_mv_kv": ti["vn_mv_kv"],
            "vn_lv_kv": ti["vn_lv_kv"],
            "vk_hv_percent": ti["vk_hv_percent"],
            "vk_mv_percent": ti["vk_mv_percent"],
            "vk_lv_percent": ti["vk_lv_percent"],
            "vkr_hv_percent": ti["vkr_hv_percent"],
            "vkr_mv_percent": ti["vkr_mv_percent"],
            "vkr_lv_percent": ti["vkr_lv_percent"],
            "pfe_kw": ti["pfe_kw"],
            "i0_percent": ti["i0_percent"],
            "shift_mv_degree": ti.get("shift_mv_degree", 0),
            "shift_lv_degree": ti.get("shift_lv_degree", 0),
            "tap_at_star_point": tap_at_star_point,
        }
    )
    for tp in (
        "tap_neutral",
        "tap_max",
        "tap_min",
        "tap_side",
        "tap_step_percent",
        "tap_step_degree",
        "tap_changer_type",
    ):
        if tp in ti:
            entries.update({tp: ti[tp]})

    if ("tap_neutral" in entries) and (tap_pos is nan):
        entries["tap_pos"] = entries["tap_neutral"]
    else:
        entries["tap_pos"] = tap_pos
        if type(tap_pos) is float:
            net.trafo3w.tap_pos = net.trafo3w.tap_pos.astype(float)

    dd = pd.DataFrame(entries, index=[index])
    net["trafo3w"] = pd.concat([net["trafo3w"], dd], sort=True).reindex(net["trafo3w"].columns, axis=1)

    _set_value_if_not_nan(net, index, max_loading_percent, "max_loading_percent", "trafo3w")
    _set_value_if_not_nan(net, index, id_characteristic_table, "id_characteristic_table", "trafo3w", dtype="Int64")
    _set_value_if_not_nan(
        net, index, tap_dependency_table, "tap_dependency_table", "trafo3w", dtype=bool_, default_val=False
    )
    _set_value_if_not_nan(net, index, tap_changer_type, "tap_changer_type", "trafo3w", dtype=str, default_val=None)

    for key in [
        "tap_dependent_impedance",
        "vk_hv_percent_characteristic",
        "vkr_hv_percent_characteristic",
        "vk_mv_percent_characteristic",
        "vkr_mv_percent_characteristic",
        "vk_lv_percent_characteristic",
        "vkr_lv_percent_characteristic",
    ]:
        if key in kwargs:
            del kwargs[key]
            warnings.warn(
                DeprecationWarning(
                    f"The {key} parameter is not supported in pandapower version 3.0 or later. "
                    f"The 3w-transformer with index {index} will be created without tap_dependent_impedance "
                    "characteristics. To set up tap-dependent characteristics for this 3w-transformer, provide the "
                    "net.trafo_characteristic_table and populate the tap_dependency_table and id_characteristic_table "
                    "parameters."
                )
            )

    return index


def create_transformers3w(
    net: pandapowerNet,
    hv_buses: Sequence,
    mv_buses: Sequence,
    lv_buses: Sequence,
    std_type: str,
    tap_pos: int | Iterable[int] | float = nan,
    name: Iterable[str] | None = None,
    in_service: bool | Iterable[bool] = True,
    index: Iterable[Int] | None = None,
    max_loading_percent: float | Iterable[float] = nan,
    tap_at_star_point: bool | Iterable[bool] = False,
    tap_changer_type: float | Iterable[float] | None = None,
    tap_dependency_table: bool | Iterable[bool] = False,
    id_characteristic_table: int | Iterable[int] | None = None,
    **kwargs,
) -> npt.NDArray[Int]:
    """
    Creates several two-winding transformers in table net.trafo.
    Additional parameters passed will be added to the transformers dataframe. If keywords are passed that are present
    in the std_type they will override any setting from the standard type.

    :param net: the pandapower network to which the transformers should be added
    :type net: pandapower.pandapowerNet
    :param Sequence hv_buses: a Sequence of bus ids that are the high voltage buses for the transformers
    :param Sequence mv_buses: a Sequence of bus ids that are the medium voltage buses for the transformers
    :param Sequence lv_buses: a Sequence of bus ids that are the low valtage buses for the transformers
    :param str std_type: the transformer std_type to get the not specified parameters from
    :param tap_pos: current tap position of the transformers. Defaults to the medium position (tap_neutral), default nan
    :type tap_pos: int | Iterable[int] | float
    :param name: names for the transformers, default None
    :type name: Iterable[str]
    :param in_service: Wheather the transforers are in or out of service, default True
    :type in_service: bool | Itreable[bool]
    :param index: the index to use for the new elements, default None
    :type index: Int | Iterable[Int] | None
    :param max_loading_percent: the maximum loading percentage of the transformer, default nan
    :type max_loading_percent: float | Iterable[float]
    :param tap_at_star_point: whether tap changer is modelled at star point or at the bus
    :param tap_changer_type: specifies the phase shifter type ("Ratio", "Symmetrical", "Ideal", "Tabular" or None), default None
    :param tap_dependency_table: True if sanity checks should be performed. See SplineCharacteristics, default False
    :param id_characteristic_table: id of the SplineCharacteristic, default None

    :example:
        >>> create_transformers3w(
        >>>     net, hv_bus=[0, 1], lv_bus=[2, 3], std_type="63/25/38 MVA 110/20/10 kV", name=["trafo1", "trafo2"]
        >>> )
    """

    std_params = load_std_type(net, std_type, "trafo3w")

    params = {
        "shift_mv_degree": std_params.get("shift_mv_degree", 0),
        "shift_lv_degree": std_params.get("shift_lv_degree", 0),
    }

    required_params = (
        "sn_hv_mva", "sn_mv_mva", "sn_lv_mva", "vn_hv_kv", "vn_mv_kv", "vn_lv_kv",
        "vk_hv_percent", "vk_mv_percent", "vk_lv_percent",
        "vkr_hv_percent", "vkr_mv_percent", "vkr_lv_percent", "pfe_kw", "i0_percent")
    if not all(param in std_params for param in required_params):
        raise ValueError(f"std_type is missing a required value. Required values: {', '.join(required_params)}")
    params_from_std_type = (
        "tap_neutral", "tap_max", "tap_min", "tap_side", "tap_step_percent", "tap_step_degree", "tap_changer_type",
        *required_params
    )

    params.update({param: std_params[param] for param in params_from_std_type if param in std_params})
    if tap_changer_type is not None:
        params["tap_changer_type"] = tap_changer_type
    params.update(kwargs)

    return create_transformers3w_from_parameters(
        net=net, hv_buses=hv_buses, mv_buses=mv_buses, lv_buses=lv_buses, name=name, tap_pos=tap_pos, std_type=std_type,
        in_service=in_service, max_loading_percent=max_loading_percent, tap_dependency_table=tap_dependency_table,
        id_characteristic_table=id_characteristic_table, tap_at_star_point=tap_at_star_point, index=index,
        **params
    )


def create_transformer3w_from_parameters(
    net: pandapowerNet,
    hv_bus: Int,
    mv_bus: Int,
    lv_bus: Int,
    vn_hv_kv: float,
    vn_mv_kv: float,
    vn_lv_kv: float,
    sn_hv_mva: float,
    sn_mv_mva: float,
    sn_lv_mva: float,
    vk_hv_percent: float,
    vk_mv_percent: float,
    vk_lv_percent: float,
    vkr_hv_percent: float,
    vkr_mv_percent: float,
    vkr_lv_percent: float,
    pfe_kw: float,
    i0_percent: float,
    shift_mv_degree: float = 0.0,
    shift_lv_degree: float = 0.0,
    tap_side: HVMVLVType | None = None,
    tap_step_percent: float = nan,
    tap_step_degree: float = nan,
    tap_pos: int | float = nan,
    tap_neutral: int | float = nan,
    tap_max: int | float = nan,
    tap_changer_type: TapChangerWithTabularType | None = None,
    tap_min: float | None = nan,
    name: str | None = None,
    in_service: bool = True,
    index: Int | None = None,
    max_loading_percent: float = nan,
    tap_at_star_point: bool = False,
    vk0_hv_percent: float = nan,
    vk0_mv_percent: float = nan,
    vk0_lv_percent: float = nan,
    vkr0_hv_percent: float = nan,
    vkr0_mv_percent: float = nan,
    vkr0_lv_percent: float = nan,
    vector_group: str | None = None,
    tap_dependency_table: bool = False,
    id_characteristic_table: int | None = None,
    **kwargs,
) -> Int:
    """
    Adds a three-winding transformer in table net.trafo3w with the specified parameters.
    The model currently only supports one tap changer per 3w-transformer.

    Input:
        **net** (pandapowerNet) - the net within this transformer should be created

        **hv_bus** (int) - the bus on the high-voltage side on which the transformer will be connected to

        **mv_bus** (int) - The bus on the middle-voltage side on which the transformer will be connected to

        **lv_bus** (int) - The bus on the low-voltage side on which the transformer will be connected to

        **vn_hv_kv** (float) - rated voltage on high voltage side

        **vn_mv_kv** (float) - rated voltage on medium voltage side

        **vn_lv_kv** (float) - rated voltage on low voltage side

        **sn_hv_mva** (float) - rated apparent power on high voltage side

        **sn_mv_mva** (float) - rated apparent power on medium voltage side

        **sn_lv_mva** (float) - rated apparent power on low voltage side

        **vk_hv_percent** (float) - short circuit voltage from high to medium voltage

        **vk_mv_percent** (float) - short circuit voltage from medium to low voltage

        **vk_lv_percent** (float) - short circuit voltage from high to low voltage

        **vkr_hv_percent** (float) - real part of short circuit voltage from high to medium voltage

        **vkr_mv_percent** (float) - real part of short circuit voltage from medium to low voltage

        **vkr_lv_percent** (float) - real part of short circuit voltage from high to low voltage

        **pfe_kw** (float) - iron losses in kW

        **i0_percent** (float) - open loop losses

    OPTIONAL:
        **shift_mv_degree** (float, 0) - angle shift to medium voltage side*

        **shift_lv_degree** (float, 0) - angle shift to low voltage side*

        **tap_step_percent** (float) - tap step in percent

        **tap_step_degree** (float) - tap phase shift angle in degrees

        **tap_side** (str, None) - "hv", "mv", "lv"

        **tap_neutral** (int, nan) - default tap position

        **tap_min** (int, nan) - Minimum tap position

        **tap_max** (int, nan) - Maximum tap position

        **tap_pos** (int, nan) - current tap position of the transformer. Defaults to the medium position (tap_neutral)

        **tap_changer_type** (str, None) - specifies the tap changer type ("Ratio", "Symmetrical", "Ideal", "Tabular", \
                                           None: no tap changer)

        **tap_at_star_point** (boolean) - Whether tap changer is located at the star point of the 3w-transformer \
                                          or at the bus

        **name** (str, None) - name of the 3-winding transformer

        **in_service** (boolean, True) - True for in_service or False for out of service

        **max_loading_percent** (float) - maximum current loading (only needed for OPF)

        **tap_dependency_table** (boolean, False) - True if transformer parameters (voltage ratio, angle, impedance) \
            must be adjusted dependent on the tap position of the transformer. Requires the additional column \
            "id_characteristic_table". The function pandapower.control.trafo_characteristic_table_diagnostic \
            can be used for sanity checks. \
            The function pandapower.control.create_trafo_characteristic_object can be used to create \
            SplineCharacteristic objects in the net.trafo_characteristic_spline table and add the additional column \
            "id_characteristic_spline" to set up the reference to the spline characteristics.

        **id_characteristic_table** (int, None) - references the index of the characteristic from the lookup table \
                                                 net.trafo_characteristic_table

        **vk0_hv_percent** (float) - zero sequence short circuit voltage from high to medium voltage

        **vk0_mv_percent** (float) - zero sequence short circuit voltage from medium to low voltage

        **vk0_lv_percent** (float) - zero sequence short circuit voltage from high to low voltage

        **vkr0_hv_percent** (float) - zero sequence real part of short circuit voltage from high to medium voltage

        **vkr0_mv_percent** (float) - zero sequence real part of short circuit voltage from medium to low voltage

        **vkr0_lv_percent** (float) - zero sequence real part of short circuit voltage from high to low voltage

        **vector_group** (str) - vector group of the 3w-transformer

    OUTPUT:
        **trafo_id** - the unique trafo_id of the created 3w-transformer

    Example:
        create_transformer3w_from_parameters(net, hv_bus=0, mv_bus=1, lv_bus=2, name="trafo1", sn_hv_mva=40, \
                                             sn_mv_mva=20, sn_lv_mva=20, vn_hv_kv=110, vn_mv_kv=20, vn_lv_kv=10, \
                                             vk_hv_percent=10,vk_mv_percent=11, vk_lv_percent=12, vkr_hv_percent=0.3, \
                                             vkr_mv_percent=0.31, vkr_lv_percent=0.32, pfe_kw=30, i0_percent=0.1, \
                                             shift_mv_degree=30, shift_lv_degree=30)
    """

    # Check if bus exist to attach the trafo to
    for b in [hv_bus, mv_bus, lv_bus]:
        if b not in net["bus"].index.values:
            raise UserWarning("Trafo tries to attach to non-existent bus %s" % b)

    index = _get_index_with_check(net, "trafo3w", index, "three winding transformer")

    if tap_pos is nan:
        tap_pos = tap_neutral

    for key in [
        "tap_dependent_impedance",
        "vk_hv_percent_characteristic",
        "vkr_hv_percent_characteristic",
        "vk_mv_percent_characteristic",
        "vkr_mv_percent_characteristic",
        "vk_lv_percent_characteristic",
        "vkr_lv_percent_characteristic",
    ]:
        if key in kwargs:
            del kwargs[key]
            warnings.warn(
                DeprecationWarning(
                    f"The {key} parameter is not supported in pandapower version 3.0 or later. "
                    f"The 3w-transformer with index {index} will be created without tap_dependent_impedance "
                    "characteristics. To set up tap-dependent characteristics for this 3w-transformer, provide the "
                    "net.trafo_characteristic_table and populate the tap_dependency_table and id_characteristic_table "
                    "parameters."
                )
            )

    entries = {
        "lv_bus": lv_bus,
        "mv_bus": mv_bus,
        "hv_bus": hv_bus,
        "vn_hv_kv": vn_hv_kv,
        "vn_mv_kv": vn_mv_kv,
        "vn_lv_kv": vn_lv_kv,
        "sn_hv_mva": sn_hv_mva,
        "sn_mv_mva": sn_mv_mva,
        "sn_lv_mva": sn_lv_mva,
        "vk_hv_percent": vk_hv_percent,
        "vk_mv_percent": vk_mv_percent,
        "vk_lv_percent": vk_lv_percent,
        "vkr_hv_percent": vkr_hv_percent,
        "vkr_mv_percent": vkr_mv_percent,
        "vkr_lv_percent": vkr_lv_percent,
        "pfe_kw": pfe_kw,
        "i0_percent": i0_percent,
        "shift_mv_degree": shift_mv_degree,
        "shift_lv_degree": shift_lv_degree,
        "tap_side": tap_side,
        "tap_step_percent": tap_step_percent,
        "tap_step_degree": tap_step_degree,
        "tap_pos": tap_pos,
        "tap_neutral": tap_neutral,
        "tap_max": tap_max,
        "tap_min": tap_min,
        "in_service": in_service,
        "name": name,
        "std_type": None,
        "tap_at_star_point": tap_at_star_point,
        "vk0_hv_percent": vk0_hv_percent,
        "vk0_mv_percent": vk0_mv_percent,
        "vk0_lv_percent": vk0_lv_percent,
        "vkr0_hv_percent": vkr0_hv_percent,
        "vkr0_mv_percent": vkr0_mv_percent,
        "vkr0_lv_percent": vkr0_lv_percent,
        "vector_group": vector_group,
    }
    _set_entries(net, "trafo3w", index, entries=entries)

    _set_value_if_not_nan(net, index, max_loading_percent, "max_loading_percent", "trafo3w")
    _set_value_if_not_nan(net, index, id_characteristic_table, "id_characteristic_table", "trafo3w", dtype="Int64")
    _set_value_if_not_nan(net, index, tap_changer_type, "tap_changer_type", "trafo3w", dtype=str, default_val=None)
    _set_value_if_not_nan(
        net, index, tap_dependency_table, "tap_dependency_table", "trafo3w", dtype=bool_, default_val=False
    )

    return index


def create_transformers3w_from_parameters(  # no index ?
    net: pandapowerNet,
    hv_buses: Sequence,
    mv_buses: Sequence,
    lv_buses: Sequence,
    vn_hv_kv: float | Iterable[float],
    vn_mv_kv: float | Iterable[float],
    vn_lv_kv: float | Iterable[float],
    sn_hv_mva: float | Iterable[float],
    sn_mv_mva: float | Iterable[float],
    sn_lv_mva: float | Iterable[float],
    vk_hv_percent: float | Iterable[float],
    vk_mv_percent: float | Iterable[float],
    vk_lv_percent: float | Iterable[float],
    vkr_hv_percent: float | Iterable[float],
    vkr_mv_percent: float | Iterable[float],
    vkr_lv_percent: float | Iterable[float],
    pfe_kw: float | Iterable[float],
    i0_percent: float | Iterable[float],
    shift_mv_degree: float | Iterable[float] = 0.0,
    shift_lv_degree: float | Iterable[float] = 0.0,
    tap_side: HVMVLVType | Iterable[str] | None = None,
    tap_step_percent: float | Iterable[float] = nan,
    tap_step_degree: float | Iterable[float] = nan,
    tap_pos: int | Iterable[int] | float = nan,
    tap_neutral: int | Iterable[int] | float = nan,
    tap_max: int | Iterable[int] | float = nan,
    tap_min: int | Iterable[int] | float = nan,
    name: Iterable[str] | None = None,
    in_service: bool | Iterable[bool] = True,
    index: Iterable[Int] | None = None,
    max_loading_percent: float | Iterable[float] = nan,
    tap_at_star_point: bool | Iterable[bool] = False,
    tap_changer_type: float | Iterable[float] | None = None,
    vk0_hv_percent: float | Iterable[float] = nan,
    vk0_mv_percent: float | Iterable[float] = nan,
    vk0_lv_percent: float | Iterable[float] = nan,
    vkr0_hv_percent: float | Iterable[float] = nan,
    vkr0_mv_percent: float | Iterable[float] = nan,
    vkr0_lv_percent: float | Iterable[float] = nan,
    vector_group: str | Iterable[str] | None = None,
    tap_dependency_table: bool | Iterable[bool] = False,
    id_characteristic_table: int | Iterable[int] | None = None,
    **kwargs,
) -> npt.NDArray[integer]:
    """
    Adds multiple three-winding transformers in table net.trafo3w with the specified parameters.
    The model currently only supports one tap changer per 3w-transformer.

    Input:
        **net** (pandapowerNet) - the net within this transformer should be created

        **hv_bus** (list of int) - The bus on the high-voltage side on which the transformer will be connected to

        **mv_bus** (list of int) - The bus on the middle-voltage side on which the transformer will be connected to

        **lv_bus** (list of int) - The bus on the low-voltage side on which the transformer will be connected to

        **vn_hv_kv** (list of float) - rated voltage on high voltage side

        **vn_mv_kv** (list of float) - rated voltage on medium voltage side

        **vn_lv_kv** (list of float) - rated voltage on low voltage side

        **sn_hv_mva** (list of float) - rated apparent power on high voltage side

        **sn_mv_mva** (list of float) - rated apparent power on medium voltage side

        **sn_lv_mva** (list of float) - rated apparent power on low voltage side

        **vk_hv_percent** (list of float) - short circuit voltage from high to medium voltage

        **vk_mv_percent** (list of float) - short circuit voltage from medium to low voltage

        **vk_lv_percent** (list of float) - short circuit voltage from high to low voltage

        **vkr_hv_percent** (list of float) - real part of short circuit voltage from high to medium voltage

        **vkr_mv_percent** (list of float) - real part of short circuit voltage from medium to low voltage

        **vkr_lv_percent** (list of float) - real part of short circuit voltage from high to low voltage

        **pfe_kw** (list of float) - iron losses in kW

        **i0_percent** (list of float) - open loop losses

    OPTIONAL:

        **shift_mv_degree** (list of float, 0) - angle shift to medium voltage side*

        **shift_lv_degree** (list of float, 0) - angle shift to low voltage side*

        **tap_step_percent** (list of float) - tap step in percent

        **tap_step_degree** (list of float) - tap phase shift angle in degrees*

        **tap_side** (list of string, None) - "hv", "mv", "lv"

        **tap_neutral** (list of int, nan) - default tap position

        **tap_min** (list of int, nan) - minimum tap position

        **tap_max** (list of int, nan) - maximum tap position

        **tap_pos** (list of int, nan) - current tap position of the transformer. Defaults to the medium position \
                                         (tap_neutral)

        **tap_changer_type** (list of str, None) - specifies the tap changer type ("Ratio", "Symmetrical", "Ideal", \
                                                   "Tabular", None: no tap changer)*

        **tap_at_star_point** (list of boolean) - whether tap changer is located at the star point of the \
                                                  3w-transformer or at the bus

        **name** (list of str, None) - name of the 3-winding transformer

        **in_service** (list of boolean, True) - True for in_service or False for out of service

        **max_loading_percent** (list of float) - maximum current loading (only needed for OPF)

        **tap_dependency_table** (list of boolean, False) - True if transformer parameters (voltage ratio, angle, \
            impedance) must be adjusted dependent on the tap position of the transformer. Requires the additional \
            column "id_characteristic_table". The function pandapower.control.trafo_characteristic_table_diagnostic \
            can be used for sanity checks. \
            The function pandapower.control.create_trafo_characteristic_object can be used to create \
            SplineCharacteristic objects in the net.trafo_characteristic_spline table and add the additional column \
            "id_characteristic_spline" to set up the reference to the spline characteristics.

        **id_characteristic_table** (list of int, nan) - references the index of the characteristic from the \
                                                         lookup table net.trafo_characteristic_table

        **vk0_hv_percent** (list of float) - zero sequence short circuit voltage from high to medium voltage

        **vk0_mv_percent** (list of float) - zero sequence short circuit voltage from medium to low voltage

        **vk0_lv_percent** (list of float) - zero sequence short circuit voltage from high to low voltage

        **vkr0_hv_percent** (list of float) - zero sequence real part of short circuit voltage from high to \
                                              medium voltage

        **vkr0_mv_percent** (list of float) - zero sequence real part of short circuit voltage from medium to \
                                              low voltage

        **vkr0_lv_percent** (list of float) - zero sequence real part of short circuit voltage from high to low voltage

        **vector_group** (list of str) - vector group of the 3w-transformers

        \\* only considered in load flow if calculate_voltage_angles = True

    OUTPUT:
        **trafo_id** (list of int) - list of trafo_ids of the created 3w-transformers

    Example:
        create_transformers3w_from_parameters(net, hv_bus=[0, 3], mv_bus=[1, 4], lv_bus=[2, 5], name="trafo1", \
                                              sn_hv_mva=40, sn_mv_mva=20, sn_lv_mva=20, vn_hv_kv=110, vn_mv_kv=20, \
                                              vn_lv_kv=10, vk_hv_percent=10,vk_mv_percent=11, vk_lv_percent=12, \
                                              vkr_hv_percent=0.3, vkr_mv_percent=0.31, vkr_lv_percent=0.32, pfe_kw=30, \
                                              i0_percent=0.1, shift_mv_degree=30, shift_lv_degree=30)
    """

    index = _get_multiple_index_with_check(net, "trafo3w", index, len(hv_buses), name="Three winding transformers")

    if not all_([isin(hv_buses, net.bus.index), isin(mv_buses, net.bus.index), isin(lv_buses, net.bus.index)]):
        bus_not_exist = (set(hv_buses) | set(mv_buses) | set(lv_buses)) - set(net.bus.index)
        raise UserWarning(f"Transformers trying to attach to non existing buses {bus_not_exist}")

    tp_neutral = pd.Series(tap_neutral, index=index, dtype=float64)
    tp_pos = pd.Series(tap_pos, index=index, dtype=float64).fillna(tp_neutral)
    entries = {
        "lv_bus": lv_buses,
        "mv_bus": mv_buses,
        "hv_bus": hv_buses,
        "vn_hv_kv": vn_hv_kv,
        "vn_mv_kv": vn_mv_kv,
        "vn_lv_kv": vn_lv_kv,
        "sn_hv_mva": sn_hv_mva,
        "sn_mv_mva": sn_mv_mva,
        "sn_lv_mva": sn_lv_mva,
        "vk_hv_percent": vk_hv_percent,
        "vk_mv_percent": vk_mv_percent,
        "vk_lv_percent": vk_lv_percent,
        "vkr_hv_percent": vkr_hv_percent,
        "vkr_mv_percent": vkr_mv_percent,
        "vkr_lv_percent": vkr_lv_percent,
        "pfe_kw": pfe_kw,
        "i0_percent": i0_percent,
        "shift_mv_degree": shift_mv_degree,
        "shift_lv_degree": shift_lv_degree,
        "tap_side": tap_side,
        "tap_step_percent": tap_step_percent,
        "tap_step_degree": tap_step_degree,
        "tap_pos": tp_pos,
        "tap_neutral": tp_neutral,
        "tap_max": tap_max,
        "tap_min": tap_min,
        "in_service": array(in_service).astype(bool_),
        "name": name,
        "tap_at_star_point": array(tap_at_star_point).astype(bool_),
        "std_type": None,
        "vk0_hv_percent": vk0_hv_percent,
        "vk0_mv_percent": vk0_mv_percent,
        "vk0_lv_percent": vk0_lv_percent,
        "vkr0_hv_percent": vkr0_hv_percent,
        "vkr0_mv_percent": vkr0_mv_percent,
        "vkr0_lv_percent": vkr0_lv_percent,
        "vector_group": vector_group,
        "tap_dependency_table": tap_dependency_table,
        **kwargs,
    }

    _add_to_entries_if_not_nan(net, "trafo3w", entries, index, "max_loading_percent", max_loading_percent)
    _add_to_entries_if_not_nan(
        net, "trafo3w", entries, index, "id_characteristic_table", id_characteristic_table, dtype="Int64"
    )
    _add_to_entries_if_not_nan(
        net, "trafo3w", entries, index, "tap_changer_type", tap_changer_type, dtype=str, default_val=None
    )
    defaults_to_fill = [("tap_dependency_table", False)]

    for key in [
        "tap_dependent_impedance",
        "vk_hv_percent_characteristic",
        "vkr_hv_percent_characteristic",
        "vk_mv_percent_characteristic",
        "vkr_mv_percent_characteristic",
        "vk_lv_percent_characteristic",
        "vkr_lv_percent_characteristic",
    ]:
        if key in kwargs:
            del kwargs[key]
            warnings.warn(
                DeprecationWarning(
                    f"The {key} parameter is not supported in pandapower version 3.0 or later. "
                    f"The 3w-transformer with index {index} will be created without tap_dependent_impedance "
                    "characteristics. To set up tap-dependent characteristics for this 3w-transformer, provide the "
                    "net.trafo_characteristic_table and populate the tap_dependency_table and id_characteristic_table "
                    "parameters."
                )
            )

    _set_multiple_entries(net, "trafo3w", index, defaults_to_fill=defaults_to_fill, entries=entries)

    return index
