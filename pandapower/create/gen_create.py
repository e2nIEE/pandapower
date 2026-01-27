# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import pandas as pd
from numpy import nan, bool_
import numpy.typing as npt

from pandapower.auxiliary import pandapowerNet
from pandapower.pp_types import Int
from pandapower.create._utils import (
    _add_to_entries_if_not_nan,
    _check_element,
    _check_multiple_elements,
    _get_index_with_check,
    _get_multiple_index_with_check,
    _set_entries,
    _set_multiple_entries,
    _set_value_if_not_nan,
)

logger = logging.getLogger(__name__)


def create_gen(
    net: pandapowerNet,
    bus: Int,
    p_mw: float,
    vm_pu: float = 1.0,
    sn_mva: float = pd.NA,
    name: str = pd.NA,
    index: Int | None = None,
    max_q_mvar: float = nan,
    min_q_mvar: float = nan,
    min_p_mw: float = nan,
    max_p_mw: float = nan,
    min_vm_pu: float = nan,
    max_vm_pu: float = nan,
    scaling: float = 1.0,
    type: str = pd.NA,
    slack: bool = False,
    id_q_capability_characteristic: int | None = pd.NA,
    reactive_capability_curve: bool | None = None,
    curve_style: str | None = pd.NA,
    controllable: bool | Iterable[bool] = pd.NA,
    vn_kv: float = pd.NA,
    xdss_pu: float = pd.NA,
    rdss_ohm: float = pd.NA,
    cos_phi: float = pd.NA,
    pg_percent: float = pd.NA,
    power_station_trafo: int = pd.NA,
    in_service: bool = True,
    slack_weight: float = nan,
    **kwargs,
) -> Int:
    """
    Adds a generator to the network.

    Generators are always modelled as voltage controlled PV nodes, which is why the input parameter
    is active power and a voltage set point. If you want to model a generator as PQ load with fixed
    reactive power and variable voltage, please use a static generator instead.

    Parameters::
        net: The net within this generator should be created
        bus: The bus id to which the generator is connected
        p_mw: The active power of the generator (positive for generation!)
        vm_pu: The voltage set point of the generator.
        sn_mva: Nominal power of the generator
        name: The name for this generator
        index: Force a specified ID if it is available. If None, the index one higher than the highest already existing
            index is selected.
        scaling: scaling factor applying to the active power of the generator
        type: type variable to classify generators
        slack: flag that sets the generator as slack if True
        reactive_capability_curve: True if both the id_q_capability_characteristic and the curve_style are present for
            the generator
        id_q_capability_characteristic: references the index of the characteristic from the
            net.q_capability_characteristic table (id_q_capability_curve column)
        curve_style: The curve style of the generator represents the relationship between active power (P) and reactive
            power (Q). It indicates whether the reactive power remains constant as the active power changes or varies
            dynamically in response to it, e.g. "straightLineYValues" and "constantYValue".
        controllable: True: p_mw, q_mvar and vm_pu limits are enforced for this generator in OPF;
            False: p_mw and vm_pu set points are enforced and *limits are ignored*.
        slack_weight: Contribution factor for distributed slack power flow calculation (active power balancing)
        vn_kv: Rated voltage of the generator for short-circuit calculation
        xdss_pu: Subtransient generator reactance for short-circuit calculation
        rdss_ohm: Subtransient generator resistance for short-circuit calculation
        cos_phi: Rated cosine phi of the generator for short-circuit calculation
        pg_percent: Rated pg (voltage control range) of the generator for short-circuit calculation
        power_station_trafo: Index of the power station transformer for short-circuit calculation
        in_service: True for in_service or False for out of service
        max_p_mw: Maximum active power injection - necessary for OPF
        min_p_mw: Minimum active power injection - necessary for OPF
        max_q_mvar: Maximum reactive power injection - necessary for OPF
        min_q_mvar: Minimum reactive power injection - necessary for OPF
        min_vm_pu: Minimum voltage magnitude. If not set, the bus voltage limit is taken - necessary for OPF.
        max_vm_pu: Maximum voltage magnitude. If not set, the bus voltage limit is taken - necessary for OPF.

    Returns:
        The unique ID of the created generator

    Example:
        >>> create_gen(net, 1, p_mw=120, vm_pu=1.02)
    """
    _check_element(net, bus)

    index = _get_index_with_check(net, "gen", index, name="generator")

    entries = {
        "name": name,
        "bus": bus,
        "p_mw": p_mw,
        "vm_pu": vm_pu,
        "sn_mva": sn_mva,
        "type": type,
        "slack": slack,
        "in_service": in_service,
        "scaling": scaling,
        "slack_weight": slack_weight,
        **kwargs,
    }
    _set_entries(net, "gen", index, True, entries=entries)

    # OPF limits
    _set_value_if_not_nan(net, index, controllable, "controllable", "gen", default_val=True)

    # id for q capability curve table
    _set_value_if_not_nan(net, index, id_q_capability_characteristic, "id_q_capability_characteristic", "gen")

    # behaviour of reactive power capability curve
    _set_value_if_not_nan(net, index, curve_style, "curve_style", "gen")

    _set_value_if_not_nan(net, index, reactive_capability_curve, "reactive_capability_curve", "gen", default_val=False)

    # P limits for OPF if controllable == True
    _set_value_if_not_nan(net, index, min_p_mw, "min_p_mw", "gen")
    _set_value_if_not_nan(net, index, max_p_mw, "max_p_mw", "gen")
    # Q limits for OPF if controllable == True
    _set_value_if_not_nan(net, index, min_q_mvar, "min_q_mvar", "gen")
    _set_value_if_not_nan(net, index, max_q_mvar, "max_q_mvar", "gen")
    # V limits for OPF if controllable == True
    _set_value_if_not_nan(net, index, max_vm_pu, "max_vm_pu", "gen", default_val=2.0)
    _set_value_if_not_nan(net, index, min_vm_pu, "min_vm_pu", "gen", default_val=0.0)

    # Short circuit calculation variables
    _set_value_if_not_nan(net, index, vn_kv, "vn_kv", "gen")
    _set_value_if_not_nan(net, index, cos_phi, "cos_phi", "gen")
    _set_value_if_not_nan(net, index, xdss_pu, "xdss_pu", "gen")
    _set_value_if_not_nan(net, index, rdss_ohm, "rdss_ohm", "gen")
    _set_value_if_not_nan(net, index, pg_percent, "pg_percent", "gen")
    _set_value_if_not_nan(net, index, power_station_trafo, "power_station_trafo", "gen")

    return index


def create_gens(
    net: pandapowerNet,
    buses: Sequence,
    p_mw: float | Iterable[float],
    vm_pu: float | Iterable[float] = 1.0,
    sn_mva: float | Iterable[float] = nan,
    name: Iterable[str] | None = None,
    index: Int | Iterable[Int] | None = None,
    max_q_mvar: float | Iterable[float] = nan,
    min_q_mvar: float | Iterable[float] = nan,
    min_p_mw: float | Iterable[float] = nan,
    max_p_mw: float | Iterable[float] = nan,
    min_vm_pu: float | Iterable[float] = nan,
    max_vm_pu: float | Iterable[float] = nan,
    scaling: float | Iterable[float] = 1.0,
    type: str | Iterable[str] = pd.NA,
    slack: bool | Iterable[bool] = False,
    id_q_capability_characteristic: Int | Iterable[Int] | None = pd.NA,
    reactive_capability_curve: bool | Iterable[bool] | None = None,
    curve_style: str | Iterable[str] | None = pd.NA,
    controllable: bool | float | Iterable[bool | float] = pd.NA,
    vn_kv: float | Iterable[float] = pd.NA,
    xdss_pu: float | Iterable[float] = pd.NA,
    rdss_ohm: float | Iterable[float] = pd.NA,
    cos_phi: float | Iterable[float] = pd.NA,
    pg_percent: float = pd.NA,
    power_station_trafo: int | float = pd.NA,
    in_service: bool = True,
    slack_weight: float = nan,
    **kwargs,
) -> npt.NDArray[Int]:
    """
    Adds generators to the specified buses network.

    Generators are always modelled as voltage controlled PV nodes, which is why the input parameter
    is active power and a voltage set point. If you want to model a generator as PQ load with fixed
    reactive power and variable voltage, please use a static generator instead.

    Parameters:
        net: The net within this generator should be created
        buses: The bus ids to which the generators are connected
        p_mw: The active power of the generator (positive for generation!)
        vm_pu: The voltage set point of the generator.
        sn_mva: Nominal power of the generator
        name: The name for this generator
        index: Force a specified ID if it is available. If None, the index one higher than the highest already existing
            index is selected.
        scaling: scaling factor which for the active power of the generator
        type: type variable to classify generators
        reactive_capability_curve: True if both the id_q_capability_characteristic and the curve_style are present in
            the generator.
        id_q_capability_characteristic: references the index of the characteristic from the lookup table
            net.q_capability_characteristic
        curve_style: The curve style of the generator represents the relationship between active power (P) and reactive
            power (Q). It indicates whether the reactive power remains constant as the active power changes or varies
            dynamically in response to it.
            e.g. "straightLineYValues" and "constantYValue"
        controllable: True: p_mw, q_mvar and vm_pu limits are enforced for this generator in OPF
            False: p_mw and vm_pu set points are enforced and *limits are ignored*.
        vn_kv: Rated voltage of the generator for short-circuit calculation
        xdss_pu: Subtransient generator reactance for short-circuit calculation
        rdss_ohm: Subtransient generator resistance for short-circuit calculation
        cos_phi: Rated cosine phi of the generator for short-circuit calculation
        pg_percent: Rated pg (voltage control range) of the generator for short-circuit calculation
        power_station_trafo: Index of the power station transformer for short-circuit calculation
        in_service: True for in_service or False for out of service
        slack_weight: Contribution factor for distributed slack power flow calculation (active power balancing)
        max_p_mw: Maximum active power injection - necessary for OPF
        min_p_mw: Minimum active power injection - necessary for OPF
        max_q_mvar: Maximum reactive power injection - necessary for OPF
        min_q_mvar: Minimum reactive power injection - necessary for OPF
        min_vm_pu: Minimum voltage magnitude. If not set the bus voltage limit is taken. - necessary for OPF.
        max_vm_pu: Maximum voltage magnitude. If not set the bus voltage limit is taken. - necessary for OPF

    Returns:
        The unique ID of the created generators

    Example:
        >>> create_gens(net, [1, 2], p_mw=[120, 100], vm_pu=[1.02, 0.99])

    """
    _check_multiple_elements(net, buses)

    index = _get_multiple_index_with_check(net, "gen", index, len(buses))

    entries = {
        "bus": buses,
        "p_mw": p_mw,
        "vm_pu": vm_pu,
        "sn_mva": sn_mva,
        "scaling": scaling,
        "in_service": in_service,
        "slack_weight": slack_weight,
        "name": name,
        "type": type,
        "slack": slack,
        "curve_style": curve_style,
        **kwargs,
    }

    _add_to_entries_if_not_nan(net, "gen", entries, index, "min_p_mw", min_p_mw)
    _add_to_entries_if_not_nan(net, "gen", entries, index, "max_p_mw", max_p_mw)
    _add_to_entries_if_not_nan(net, "gen", entries, index, "min_q_mvar", min_q_mvar)
    _add_to_entries_if_not_nan(net, "gen", entries, index, "max_q_mvar", max_q_mvar)
    _add_to_entries_if_not_nan(net, "gen", entries, index, "min_vm_pu", min_vm_pu)
    _add_to_entries_if_not_nan(net, "gen", entries, index, "max_vm_pu", max_vm_pu)
    _add_to_entries_if_not_nan(net, "gen", entries, index, "vn_kv", vn_kv)
    _add_to_entries_if_not_nan(net, "gen", entries, index, "cos_phi", cos_phi)
    _add_to_entries_if_not_nan(net, "gen", entries, index, "xdss_pu", xdss_pu)
    _add_to_entries_if_not_nan(net, "gen", entries, index, "rdss_ohm", rdss_ohm)
    _add_to_entries_if_not_nan(net, "gen", entries, index, "pg_percent", pg_percent)
    _add_to_entries_if_not_nan(
        net, "gen", entries, index, "id_q_capability_characteristic", id_q_capability_characteristic
    )
    
    if "reactive_capability_curve" in net.gen or reactive_capability_curve is not None:
        _add_to_entries_if_not_nan(
            net, "gen", entries, index, "reactive_capability_curve", reactive_capability_curve, default_val=False
        )

    _add_to_entries_if_not_nan(net, "gen", entries, index, "power_station_trafo", power_station_trafo)
    _add_to_entries_if_not_nan(net, "gen", entries, index, "controllable", controllable)

    _set_multiple_entries(net, "gen", index, entries=entries)

    return index
