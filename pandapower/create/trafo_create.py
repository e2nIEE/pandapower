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

from pandapower.auxiliary import pandapowerNet
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

    Parameters:
        net: the net within this transformer should be created
        hv_bus: the bus on the high-voltage side on which the transformer will be connected to
        lv_bus: the bus on the low-voltage side on which the transformer will be connected to
        std_type: the used standard type from the standard type library
         
            **Zero sequence parameters** (added through std_type for three-phase load flow):
            
            - vk0_percent (float): zero sequence relative short-circuit voltage
            - vkr0_percent (float): real part of zero sequence relative short-circuit voltage
            - mag0_percent (float): ratio between magnetizing and short circuit impedance (zero sequence) as a percent
                                   (z_mag0 / z0) * 100 %
            - mag0_rx (float): zero sequence magnetizing r/x ratio
            - si0_hv_partial (float): zero sequence short circuit impedance distribution in hv side

        name: a custom name for this transformer
        tap_pos: current tap position of the transformer. Defaults to the medium position (tap_neutral)
        in_service: True for in_service or False for out of service
        index: force a specified ID if it is available. If None, the index one higher than the highest already existing
            index is selected
        max_loading_percent: maximum current loading (only needed for OPF)
        parallel: number of parallel transformers
        df: derating factor: maximum current of transformer in relation to nominal current of transformer (from 0 to 1)
        tap_dependency_table: True if transformer parameters (voltage ratio, angle, impedance) must be adjusted
            dependent on the tap position of the transformer. Requires the additional column "id_characteristic_table".
            The function pandapower.control.trafo_characteristic_table_diagnostic can be used for sanity checks.
            The function pandapower.control.create_trafo_characteristic_object can be used to create
            SplineCharacteristic objects in the net.trafo_characteristic_spline table and add the additional column
            "id_characteristic_spline" to set up the reference to the spline characteristics.
        id_characteristic_table: references the index of the characteristic from the lookup table
            net.trafo_characteristic_table
        tap_changer_type: specifies the phase shifter type ("Ratio", "Symmetrical", "Ideal", "Tabular", None: no tap
            changer)
        xn_ohm: impedance of the grounding reactor (Z_N) for short circuit calculation
        tap2_pos: current tap position of the second tap changer of the transformer. Defaults to the medium position
            (tap2_neutral)

    Returns:
        The ID of the created transformer

    Example:
        >>> create_transformer(net, hv_bus=0, lv_bus=1, std_type="0.4 MVA 10/0.4 kV", name="trafo1")
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
    
    Parameters:
        net: the pandapower network to which the transformers should be added
        Sequence hv_buses: a Sequence of bus ids that are the high voltage buses for the transformers
        Sequence lv_buses: a Sequence of bus ids that are the low valtage buses for the transformers
        str std_type: the transformer std_type to get the not specified parameters from
        name: names for the transformers, default None
        tap_pos: current tap position of the transformers. Defaults to the medium position (tap_neutral), default nan
        in_service: Wheather the transforers are in or out of service, default True
        index: the index to use for the new elements, default None
        max_loading_percent: the maximum loading percentage of the transformer, default nan
        parallel: number of parallel transformer, default 1
        df: derating factor: maximum current of transformer in relation to nominal current of transformer (0 - 1),
            default 1.0
        tap_changer_type: specifies the phase shifter type ("Ratio", "Symmetrical", "Ideal", "Tabular" or None),
            default None
        tap_dependency_table: True if sanity checks should be performed. See SplineCharacteristics, default False
        id_characteristic_table: id of the SplineCharacteristic, default None
        pt_percent: default nan
        oltc: default False
        xn_ohm: impedance of the grounding reactor (Z_N) for short circuit calculation, default nan
        tap2_pos: current tap position of the second tap changer ot the transformer. Defaults to the medium position
            (tap2_neutral), default nan

    Returns:
        The IDs of the created transformers

    Example:
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

    Parameters:
        net: the net within this transformer should be created
        hv_bus: the bus on the high-voltage side on which the transformer will be connected to
        lv_bus: the bus on the low-voltage side on which the transformer will be connected to
        sn_mva: rated apparent power
        vn_hv_kv: rated voltage on high voltage side
        vn_lv_kv: rated voltage on low voltage side
        vkr_percent: real part of relative short-circuit voltage
        vk_percent: relative short-circuit voltage
        pfe_kw: iron losses in kW
        i0_percent: open loop losses in percent of rated current
        vector_group: vector group of the transformer HV side is Uppercase letters and LV side is lower case
        vk0_percent: zero sequence relative short-circuit voltage
        vkr0_percent: real part of zero sequence relative short-circuit voltage
        mag0_percent: ratio between magnetizing and short circuit impedance (zero sequence) as a percent
                                   (z_mag0 / z0) * 100 %
        mag0_rx: zero sequence magnetizing R/X ratio
        si0_hv_partial: Distribution of zero sequence leakage impedance's for HV side
        in_service: True for in_service or False for out of service
        parallel: number of parallel transformers
        name: A custom name for this transformer
        shift_degree: angle shift over the transformer*
        tap_side: position of tap changer ("hv", "lv")
        tap_pos: current tap position of the transformer. Defaults to the medium position (tap_neutral)
        tap_neutral: tap position where the transformer ratio is equal to the ratio of the rated voltages
        tap_max: maximum allowed tap position
        tap_min: minimum allowed tap position
        tap_step_percent: tap step size for voltage magnitude in percent
        tap_step_degree: tap step size for voltage angle in degree*
        tap_changer_type: specifies the phase shifter type ("Ratio", "Symmetrical", "Ideal", "Tabular", None: no tap
            changer)*
        index: force a specified ID if it is available. If None, the index one higher than the highest already existing
            index is selected.
        max_loading_percent: maximum current loading (only needed for OPF)
        df: derating factor - maximum current of transformer in relation to nominal current of transformer (from 0 to 1)
        tap_dependency_table: True if transformer parameters (voltage ratio, angle, impedance) must be adjusted
            dependent on the tap position of the transformer. Requires the additional column "id_characteristic_table".
            The function pandapower.control.trafo_characteristic_table_diagnostic can be used for sanity checks.
            The function pandapower.control.create_trafo_characteristic_object can be used to create
            SplineCharacteristic objects in the net.trafo_characteristic_spline table and add the additional column
            "id_characteristic_spline" to set up the reference to the spline characteristics.
        id_characteristic_table: references the index of the characteristic from the lookup table
            net.trafo_characteristic_table
        pt_percent: (short circuit only)
        oltc: (short circuit only)
        xn_ohm: impedance of the grounding reactor (Z_N) for short circuit calculation
        tap2_side: position of the second tap changer ("hv", "lv")
        tap2_pos: current tap position of the second tap changer of the transformer. Defaults to the medium position
            (tap2_neutral)
        tap2_neutral: second tap position where the transformer ratio is equal to the ratio of the rated voltages
        tap2_max: maximum allowed tap position of the second tap changer
        tap2_min: minimum allowed tap position of the second tap changer
        tap2_step_percent: second tap step size for voltage magnitude in percent
        tap2_step_degree: second tap step size for voltage angle in degree*
        tap2_changer_type: specifies the tap changer type ("Ratio", "Symmetrical", "Ideal", None: no tap changer)*
        
            \\* only considered in load flow if calculate_voltage_angles = True
        
    Keyword Arguments:
        leakage_resistance_ratio_hv: ratio of transformer short-circuit resistance on HV side (default 0.5)
        leakage_reactance_ratio_hv: ratio of transformer short-circuit reactance on HV side (default 0.5)
        
    Returns:
        **index** (int) - the unique ID of the created transformer

    Example:
        >>> create_transformer_from_parameters(
        >>>     net, hv_bus=0, lv_bus=1, name="trafo1", sn_mva=40, vn_hv_kv=110, vn_lv_kv=10, vk_percent=10,
        >>>     vkr_percent=0.3, pfe_kw=30, i0_percent=0.1, shift_degree=30
        >>> )
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

    Parameters:
        net: the net within this transformer should be created
        hv_buses: the bus on the high-voltage side on which the transformer will be connected to
        lv_buses: the bus on the low-voltage side on which the transformer will be connected to
        sn_mva: rated apparent power
        vn_hv_kv: rated voltage on high voltage side
        vn_lv_kv: rated voltage on low voltage side
        vkr_percent: real part of relative short-circuit voltage
        vk_percent: relative short-circuit voltage
        pfe_kw: iron losses in kW
        i0_percent: open loop losses in percent of rated current
        vector_group: Vector group of the transformer HV side is Uppercase letters and LV side is lower case
        vk0_percent: zero sequence relative short-circuit voltage
        vkr0_percent: real part of zero sequence relative short-circuit voltage
        mag0_percent: ratio between magnetizing and short circuit impedance (zero sequence) as a percent
                                   (z_mag0 / z0) * 100 %
        mag0_rx: zero sequence magnetizing R/X ratio
        si0_hv_partial: distribution of zero sequence leakage impedance's for HV side
        in_service: True for in_service or False for out of service
        parallel: number of parallel transformers
        name: a custom name for this transformer
        shift_degree: angle shift over the transformer*
        tap_side: position of tap changer ("hv", "lv")
        tap_pos: current tap position of the transformer. Defaults to the neutral tap position (tap_neutral)
        tap_neutral: tap position where the transformer ratio is equal to the ratio of the rated voltages
        tap_max: maximum allowed tap position
        tap_min: minimum allowed tap position
        tap_step_percent: tap step size for voltage magnitude in percent
        tap_step_degree: tap step size for voltage angle in degree*
        tap_changer_type: specifies the tap changer type ("Ratio", "Symmetrical", "Ideal", "Tabular", None: no tap
            changer)*
        index: force a specified ID if it is available. If None, the index one higher than the highest already existing
            index is selected.
        max_loading_percent: maximum current loading (only needed for OPF)
        df: derating factor - maximum current of transformer in relation to nominal current of transformer (from 0 to 1)
        tap_dependency_table: True if transformer parameters (voltage ratio, angle, impedance) must be adjusted
            dependent on the tap position of the transformer. Requires the additional column "id_characteristic_table".
            The function pandapower.control.trafo_characteristic_table_diagnostic can be used for sanity checks.
            The function pandapower.control.create_trafo_characteristic_object can be used to create
            SplineCharacteristic objects in the net.trafo_characteristic_spline table and add the additional column
            "id_characteristic_spline" to set up the reference to the spline characteristics.
        id_characteristic_table: references the index of the characteristic from the lookup table
            net.trafo_characteristic_table
        pt_percent: (short circuit only)
        oltc: (short circuit only)
        xn_ohm: impedance of the grounding reactor (Z_N) for short circuit calculation
        tap2_side: position of the second tap changer ("hv", "lv")
        tap2_pos: current tap position of the second tap changer of the transformer. Defaults to the medium position
            (tap2_neutral)
        tap2_neutral: second tap position where the transformer ratio is equal to the ratio of the rated voltages
        tap2_max: maximum allowed tap position of the second tap changer
        tap2_min: minimum allowed tap position of the second tap changer
        tap2_step_percent: second tap step size for voltage magnitude in percent
        tap2_step_degree: second tap step size for voltage angle in degree*
        tap2_changer_type: specifies the tap changer type ("Ratio", "Symmetrical", "Ideal", None: no tap changer)*

            \\* only considered in load flow if calculate_voltage_angles = True

    Returns:
        The list of IDs of the created transformers

    Example:
        >>> create_transformers_from_parameters(
        >>>     net, hv_bus=[0, 1], lv_bus=[2, 3], name="trafo1", sn_mva=40, vn_hv_kv=110, vn_lv_kv=10, vk_percent=10,
        >>>     vkr_percent=0.3, pfe_kw=30, i0_percent=0.1, shift_degree=30
        >>> )
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
        net: the net within this transformer should be created
        hv_bus: The bus on the high-voltage side on which the transformer will be connected to
        mv_bus: The medium voltage bus on which the transformer will be connected to
        lv_bus: The bus on the low-voltage side on which the transformer will be connected to
        std_type: the used standard type from the standard type library
        name: a custom name for this transformer
        tap_pos: current tap position of the transformer. Defaults to the medium position (tap_neutral)
        tap_changer_type: specifies the tap changer type ("Ratio", "Symmetrical", "Ideal", "Tabular", None: no tap
            changer)*
        tap_at_star_point: whether tap changer is located at the star point of the 3w-transformer or at the bus
        in_service: True for in_service or False for out of service
        index: force a specified ID if it is available. If None, the index one higher than the highest already existing
            index is selected.
        max_loading_percent: maximum current loading (only needed for OPF)
        tap_at_star_point: whether tap changer is modelled at star point or at the bus
        tap_dependency_table: True if transformer parameters (voltage ratio, angle, impedance) must be adjusted
            dependent on the tap position of the transformer. Requires the additional column "id_characteristic_table".
            The function pandapower.control.trafo_characteristic_table_diagnostic can be used for sanity checks.
            The function pandapower.control.create_trafo_characteristic_object can be used to create
            SplineCharacteristic objects in the net.trafo_characteristic_spline table and add the additional column
            "id_characteristic_spline" to set up the reference to the spline characteristics.
        id_characteristic_table: references the index of the characteristic from the lookup table
            net.trafo_characteristic_table

    Returns:
        The ID of the created transformer

    Example:
        >>> create_transformer3w(net, hv_bus=0, mv_bus=1, lv_bus=2, name="trafo1", std_type="63/25/38 MVA 110/20/10 kV")
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

    Parameters:
        net: the pandapower network to which the transformers should be added
        hv_buses: a Sequence of bus ids that are the high voltage buses for the transformers
        mv_buses: a Sequence of bus ids that are the medium voltage buses for the transformers
        lv_buses: a Sequence of bus ids that are the low valtage buses for the transformers
        std_type: the transformer std_type to get the not specified parameters from
        tap_pos: current tap position of the transformers. Defaults to the medium position (tap_neutral), default nan
        name: names for the transformers, default None
        in_service: Wheather the transforers are in or out of service, default True
        index: the index to use for the new elements, default None
        max_loading_percent: the maximum loading percentage of the transformer, default nan
        tap_at_star_point: whether tap changer is modelled at star point or at the bus
        tap_changer_type: specifies the phase shifter type ("Ratio", "Symmetrical", "Ideal", "Tabular" or None),
            default None
        tap_dependency_table: True if sanity checks should be performed. See SplineCharacteristics, default False
        id_characteristic_table: id of the SplineCharacteristic, default None

    Example:
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

    Parameters:
        net: the net within this transformer should be created
        hv_bus: the bus on the high-voltage side on which the transformer will be connected to
        mv_bus: The bus on the middle-voltage side on which the transformer will be connected to
        lv_bus: The bus on the low-voltage side on which the transformer will be connected to
        vn_hv_kv: rated voltage on high voltage side
        vn_mv_kv: rated voltage on medium voltage side
        vn_lv_kv: rated voltage on low voltage side
        sn_hv_mva: rated apparent power on high voltage side
        sn_mv_mva: rated apparent power on medium voltage side
        sn_lv_mva: rated apparent power on low voltage side
        vk_hv_percent: short circuit voltage from high to medium voltage
        vk_mv_percent: short circuit voltage from medium to low voltage
        vk_lv_percent: short circuit voltage from high to low voltage
        vkr_hv_percent: real part of short circuit voltage from high to medium voltage
        vkr_mv_percent: real part of short circuit voltage from medium to low voltage
        vkr_lv_percent: real part of short circuit voltage from high to low voltage
        pfe_kw: iron losses in kW
        i0_percent: open loop losses
        shift_mv_degree: angle shift to medium voltage side*
        shift_lv_degree: angle shift to low voltage side*
        tap_step_percent: tap step in percent
        tap_step_degree: tap phase shift angle in degrees
        tap_side: "hv", "mv", "lv"
        tap_neutral: default tap position
        tap_min: Minimum tap position
        tap_max: Maximum tap position
        tap_pos: current tap position of the transformer. Defaults to the medium position (tap_neutral)
        tap_changer_type: specifies the tap changer type ("Ratio", "Symmetrical", "Ideal", "Tabular", None: no tap
            changer)
        tap_at_star_point: Whether tap changer is located at the star point of the 3w-transformer or at the bus
        name: name of the 3-winding transformer
        in_service: True for in_service or False for out of service
        max_loading_percent: maximum current loading (only needed for OPF)
        tap_dependency_table: True if transformer parameters (voltage ratio, angle, impedance) must be adjusted
            dependent on the tap position of the transformer. Requires the additional column "id_characteristic_table".
            The function pandapower.control.trafo_characteristic_table_diagnostic can be used for sanity checks.
            The function pandapower.control.create_trafo_characteristic_object can be used to create
            SplineCharacteristic objects in the net.trafo_characteristic_spline table and add the additional column
            "id_characteristic_spline" to set up the reference to the spline characteristics.
        id_characteristic_table: references the index of the characteristic from the lookup table
            net.trafo_characteristic_table
        vk0_hv_percent: zero sequence short circuit voltage from high to medium voltage
        vk0_mv_percent: zero sequence short circuit voltage from medium to low voltage
        vk0_lv_percent: zero sequence short circuit voltage from high to low voltage
        vkr0_hv_percent: zero sequence real part of short circuit voltage from high to medium voltage
        vkr0_mv_percent: zero sequence real part of short circuit voltage from medium to low voltage
        vkr0_lv_percent: zero sequence real part of short circuit voltage from high to low voltage
        vector_group: vector group of the 3w-transformer
    
    Returns:
        The ID of the created 3w-transformer

    Example:
        >>> create_transformer3w_from_parameters(
        >>>     net, hv_bus=0, mv_bus=1, lv_bus=2, name="trafo1", sn_hv_mva=40, sn_mv_mva=20, sn_lv_mva=20,
        >>>     vn_hv_kv=110, vn_mv_kv=20, vn_lv_kv=10, vk_hv_percent=10,vk_mv_percent=11, vk_lv_percent=12,
        >>>     vkr_hv_percent=0.3, vkr_mv_percent=0.31, vkr_lv_percent=0.32, pfe_kw=30, i0_percent=0.1,
        >>>     shift_mv_degree=30, shift_lv_degree=30
        >>> )
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
        net: the net within this transformer should be created
        hv_bus: The bus on the high-voltage side on which the transformer will be connected to
        mv_bus: The bus on the middle-voltage side on which the transformer will be connected to
        lv_bus: The bus on the low-voltage side on which the transformer will be connected to
        vn_hv_kv: rated voltage on high voltage side
        vn_mv_kv: rated voltage on medium voltage side
        vn_lv_kv: rated voltage on low voltage side
        sn_hv_mva: rated apparent power on high voltage side
        sn_mv_mva: rated apparent power on medium voltage side
        sn_lv_mva: rated apparent power on low voltage side
        vk_hv_percent: short circuit voltage from high to medium voltage
        vk_mv_percent: short circuit voltage from medium to low voltage
        vk_lv_percent: short circuit voltage from high to low voltage
        vkr_hv_percent: real part of short circuit voltage from high to medium voltage
        vkr_mv_percent: real part of short circuit voltage from medium to low voltage
        vkr_lv_percent: real part of short circuit voltage from high to low voltage
        pfe_kw: iron losses in kW
        i0_percent: open loop losses
        shift_mv_degree: angle shift to medium voltage side*
        shift_lv_degree: angle shift to low voltage side*
        tap_step_percent: tap step in percent
        tap_step_degree: tap phase shift angle in degrees*
        tap_side: "hv", "mv", "lv"
        tap_neutral: default tap position
        tap_min: minimum tap position
        tap_max: maximum tap position
        tap_pos: current tap position of the transformer. Defaults to the medium position (tap_neutral)
        tap_changer_type: specifies the tap changer type ("Ratio", "Symmetrical", "Ideal", "Tabular", None: no tap
            changer)*
        tap_at_star_point: whether tap changer is located at the star point of the 3w-transformer or at the bus
        name: name of the 3-winding transformer
        in_service: True for in_service or False for out of service
        max_loading_percent: maximum current loading (only needed for OPF)
        tap_dependency_table: True if transformer parameters (voltage ratio, angle, impedance) must be adjusted
            dependent on the tap position of the transformer. Requires the additional column "id_characteristic_table".
            The function pandapower.control.trafo_characteristic_table_diagnostic can be used for sanity checks.
            The function pandapower.control.create_trafo_characteristic_object can be used to create
            SplineCharacteristic objects in the net.trafo_characteristic_spline table and add the additional column
            "id_characteristic_spline" to set up the reference to the spline characteristics.
        id_characteristic_table: references the index of the characteristic from the lookup table
            net.trafo_characteristic_table
        vk0_hv_percent: zero sequence short circuit voltage from high to medium voltage
        vk0_mv_percent: zero sequence short circuit voltage from medium to low voltage
        vk0_lv_percent: zero sequence short circuit voltage from high to low voltage
        vkr0_hv_percent: zero sequence real part of short circuit voltage from high to medium voltage
        vkr0_mv_percent: zero sequence real part of short circuit voltage from medium to low voltage
        vkr0_lv_percent: zero sequence real part of short circuit voltage from high to low voltage
        vector_group: vector group of the 3w-transformers

    Returns:
        list of ids of the created 3w-transformers

    Example:
        >>> create_transformers3w_from_parameters(
        >>>     net, hv_bus=[0, 3], mv_bus=[1, 4], lv_bus=[2, 5], name="trafo1", sn_hv_mva=40, sn_mv_mva=20,
        >>>     sn_lv_mva=20, vn_hv_kv=110, vn_mv_kv=20, vn_lv_kv=10, vk_hv_percent=10,vk_mv_percent=11,
        >>>     vk_lv_percent=12, vkr_hv_percent=0.3, vkr_mv_percent=0.31, vkr_lv_percent=0.32, pfe_kw=30,
        >>>     i0_percent=0.1, shift_mv_degree=30, shift_lv_degree=30
        >>> )

    \\* only considered in load flow if calculate_voltage_angles = True
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
