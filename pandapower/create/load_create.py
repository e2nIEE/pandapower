# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging
from typing import Iterable, Sequence

from numpy import nan, bool_
import numpy.typing as npt

from pandapower import pandapowerNet
from pandapower.pp_types import Int, UnderOverExcitedType, WyeDeltaType
from pandapower.create._utils import (
    _add_to_entries_if_not_nan,
    _check_element,
    _check_multiple_elements,
    _get_index_with_check,
    _get_multiple_index_with_check,
    _set_const_percent_values,
    _set_entries,
    _set_multiple_entries,
    _set_value_if_not_nan,
)

logger = logging.getLogger(__name__)


def create_load(
    net: pandapowerNet,
    bus: Int,
    p_mw: float,
    q_mvar: float = 0,
    const_z_p_percent: float = 0,
    const_i_p_percent: float = 0,
    const_z_q_percent: float = 0,
    const_i_q_percent: float = 0,
    sn_mva: float = nan,
    name: str | None = None,
    scaling: float = 1.0,
    index: Int | None = None,
    in_service: bool = True,
    type: WyeDeltaType = "wye",
    max_p_mw: float = nan,
    min_p_mw: float = nan,
    max_q_mvar: float = nan,
    min_q_mvar: float = nan,
    controllable: bool | float = nan,
    **kwargs,
) -> Int:
    """
    Adds one load in table net["load"].

    All loads are modelled in the consumer system, meaning load is positive and generation is
    negative active power. Please pay attention to the correct signing of the reactive power as
    well.

    INPUT:
        **net** - The net within this load should be created

        **bus** (int) - The bus id to which the load is connected

        **p_mw** (float) - The active power of the load

        - positive value -> load
        - negative value -> generation

    OPTIONAL:
        **q_mvar** (float, default 0) - The reactive power of the load

        **const_z_p_percent** (float, default 0) - percentage of p_mw that will be \
            associated to constant impedance load at rated voltage

        **const_i_p_percent** (float, default 0) - percentage of p_mw that will be \
            associated to constant current load at rated voltage

        **const_z_q_percent** (float, default 0) - percentage of q_mvar that will be \
            associated to constant impedance load at rated voltage

        **const_i_q_percent** (float, default 0) - percentage of q_mvar that will be \
            associated to constant current load at rated voltage

        **sn_mva** (float, default NaN) - Nominal power of the load

        **name** (string, default None) - The name for this load

        **scaling** (float, default 1.) - An OPTIONAL scaling factor.
        Multiplies with p_mw and q_mvar.

        **type** (string, 'wye') -  type variable to classify the load: wye/delta

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **in_service** (boolean) - True for in_service or False for out of service

        **max_p_mw** (float, default NaN) - Maximum active power load - necessary for controllable \
            loads in for OPF

        **min_p_mw** (float, default NaN) - Minimum active power load - necessary for controllable \
            loads in for OPF

        **max_q_mvar** (float, default NaN) - Maximum reactive power load - necessary for \
            controllable loads in for OPF

        **min_q_mvar** (float, default NaN) - Minimum reactive power load - necessary for \
            controllable loads in OPF

        **controllable** (boolean, default NaN) - States, whether a load is controllable or not. \
            Only respected for OPF; defaults to False if "controllable" column exists in DataFrame

    OUTPUT:
        **index** (int) - The unique ID of the created element

    EXAMPLE:
        create_load(net, bus=0, p_mw=10., q_mvar=2.)

    """
    _check_element(net, bus)

    index = _get_index_with_check(net, "load", index)

    if ("const_z_percent" in kwargs) or ("const_i_percent" in kwargs):
        const_percent_values_list = [const_z_p_percent, const_i_p_percent, const_z_q_percent, const_i_q_percent]
        const_z_p_percent, const_i_p_percent, const_z_q_percent, const_i_q_percent, kwargs = _set_const_percent_values(
            const_percent_values_list, kwargs_input=kwargs
        )

    entries = {
        "name": name,
        "bus": bus,
        "p_mw": p_mw,
        "const_z_p_percent": const_z_p_percent,
        "const_i_p_percent": const_i_p_percent,
        "const_z_q_percent": const_z_q_percent,
        "const_i_q_percent": const_i_q_percent,
        "scaling": scaling,
        "q_mvar": q_mvar,
        "sn_mva": sn_mva,
        "in_service": in_service,
        "type": type,
        **kwargs,
    }
    _set_entries(net, "load", index, True, entries=entries)

    _set_value_if_not_nan(net, index, min_p_mw, "min_p_mw", "load")
    _set_value_if_not_nan(net, index, max_p_mw, "max_p_mw", "load")
    _set_value_if_not_nan(net, index, min_q_mvar, "min_q_mvar", "load")
    _set_value_if_not_nan(net, index, max_q_mvar, "max_q_mvar", "load")
    _set_value_if_not_nan(net, index, controllable, "controllable", "load", dtype=bool_, default_val=False)

    return index


def create_loads(
    net: pandapowerNet,
    buses: Sequence,
    p_mw: float | Iterable[float],
    q_mvar: float | Iterable[float] = 0,
    const_z_p_percent: float | Iterable[float] = 0,
    const_i_p_percent: float | Iterable[float] = 0,
    const_z_q_percent: float | Iterable[float] = 0,
    const_i_q_percent: float | Iterable[float] = 0,
    sn_mva: float | Iterable[float] = nan,
    name: Iterable[str] | None = None,
    scaling: float | Iterable[float] = 1.0,
    index: Int | Iterable[Int] | None = None,
    in_service: bool | Iterable[bool] = True,
    type: WyeDeltaType = "wye",
    max_p_mw: float | Iterable[float] = nan,
    min_p_mw: float | Iterable[float] = nan,
    max_q_mvar: float | Iterable[float] = nan,
    min_q_mvar: float | Iterable[float] = nan,
    controllable: bool | Iterable[bool] | float = nan,
    **kwargs,
) -> npt.NDArray[Int]:
    """
    Adds a number of loads in table net["load"].

    All loads are modelled in the consumer system, meaning load is positive and generation is
    negative active power. Please pay attention to the correct signing of the reactive power as
    well.

    INPUT:
        **net** - The net within this load should be created

        **buses** (list of int) - A list of bus ids to which the loads are connected

        **p_mw** (list of floats) - The active power of the loads

        - positive value   -> load
        - negative value  -> generation

    OPTIONAL:
        **q_mvar** (list of floats, default 0) - The reactive power of the loads

        **const_z_p_percent** (list of floats, default 0) - percentage of p_mw that will \
            be associated to constant impedance loads at rated voltage

        **const_i_p_percent** (list of floats, default 0) - percentage of p_mw that will \
            be associated to constant current load at rated voltage

        **const_z_q_percent** (list of floats, default 0) - percentage of q_mvar that will \
            be associated to constant impedance loads at rated voltage

        **const_i_q_percent** (list of floats, default 0) - percentage of q_mvar that will \
            be associated to constant current load at rated voltage

        **sn_mva** (list of floats, default None) - Nominal power of the loads

        **name** (list of strings, default None) - The name for this load

        **scaling** (list of floats, default 1.) - An OPTIONAL scaling factor to be set customly.
        Multiplies with p_mw and q_mvar.

        **type** (string, None) -  type variable to classify the load

        **index** (list of int, None) - Force a specified ID if it is available. If None, the index\
            is set to a range between one higher than the highest already existing index and the \
            length of loads that shall be created.

        **in_service** (list of boolean) - True for in_service or False for out of service

        **max_p_mw** (list of floats, default NaN) - Maximum active power load - necessary for \
            controllable loads in for OPF

        **min_p_mw** (list of floats, default NaN) - Minimum active power load - necessary for \
            controllable loads in for OPF

        **max_q_mvar** (list of floats, default NaN) - Maximum reactive power load - necessary for \
            controllable loads in for OPF

        **min_q_mvar** (list of floats, default NaN) - Minimum reactive power load - necessary for \
            controllable loads in OPF

        **controllable** (list of boolean, default NaN) - States, whether a load is controllable \
            or not. Only respected for OPF
            Defaults to False if "controllable" column exists in DataFrame

    OUTPUT:
        **index** (numpy.ndarray (int)) - The unique IDs of the created elements

    EXAMPLE:
        create_loads(net, buses=[0, 2], p_mw=[10., 5.], q_mvar=[2., 0.])

    """
    _check_multiple_elements(net, buses)

    index = _get_multiple_index_with_check(net, "load", index, len(buses))

    if ("const_z_percent" in kwargs) or ("const_i_percent" in kwargs):
        const_percent_values_list = [const_z_p_percent, const_i_p_percent, const_z_q_percent, const_i_q_percent]
        const_z_p_percent, const_i_p_percent, const_z_q_percent, const_i_q_percent, kwargs = _set_const_percent_values(
            const_percent_values_list, kwargs_input=kwargs
        )

    entries = {
        "bus": buses,
        "p_mw": p_mw,
        "q_mvar": q_mvar,
        "sn_mva": sn_mva,
        "const_z_p_percent": const_z_p_percent,
        "const_i_p_percent": const_i_p_percent,
        "const_z_q_percent": const_z_q_percent,
        "const_i_q_percent": const_i_q_percent,
        "scaling": scaling,
        "in_service": in_service,
        "name": name,
        "type": type,
        **kwargs,
    }

    _add_to_entries_if_not_nan(net, "load", entries, index, "min_p_mw", min_p_mw)
    _add_to_entries_if_not_nan(net, "load", entries, index, "max_p_mw", max_p_mw)
    _add_to_entries_if_not_nan(net, "load", entries, index, "min_q_mvar", min_q_mvar)
    _add_to_entries_if_not_nan(net, "load", entries, index, "max_q_mvar", max_q_mvar)
    _add_to_entries_if_not_nan(
        net, "load", entries, index, "controllable", controllable, dtype=bool_, default_val=False
    )
    defaults_to_fill = [("controllable", False)]

    _set_multiple_entries(net, "load", index, defaults_to_fill=defaults_to_fill, entries=entries)

    return index


def create_asymmetric_load(
    net: pandapowerNet,
    bus: Int,
    p_a_mw: float = 0,
    p_b_mw: float = 0,
    p_c_mw: float = 0,
    q_a_mvar: float = 0,
    q_b_mvar: float = 0,
    q_c_mvar: float = 0,
    sn_a_mva: float=nan,
    sn_b_mva: float=nan,
    sn_c_mva: float=nan,
    sn_mva: float = nan,
    name: str | None = None,
    scaling: float = 1.0,
    index: Int | None = None,
    in_service: bool = True,
    type: WyeDeltaType = "wye",
    **kwargs,
) -> Int:
    """
    Adds one 3 phase load in table net["asymmetric_load"].

    All loads are modelled in the consumer system, meaning load is positive and generation is
    negative active power. Please pay attention to the correct signing of the reactive power as
    well.

    INPUT:
        **net** - The net within this load should be created

        **bus** (int) - The bus id to which the load is connected

    OPTIONAL:
        **p_a_mw** (float, default 0) - The active power for Phase A load

        **p_b_mw** (float, default 0) - The active power for Phase B load

        **p_c_mw** (float, default 0) - The active power for Phase C load

        **q_a_mvar** float, default 0) - The reactive power for Phase A load

        **q_b_mvar** float, default 0) - The reactive power for Phase B load

        **q_c_mvar** (float, default 0) - The reactive power for Phase C load

        **sn_a_mva** (float, default NaN) - Nominal power for Phase A load

        **sn_b_mva** (float, default NaN) - Nominal power for Phase B load

        **sn_c_mva** (float, default NaN) - Nominal power for Phase C load

        **sn_mva** (float, default: NaN) - Nominal power of the load

        **name** (string, default: None) - The name for this load

        **scaling** (float, default: 1.) - An OPTIONAL scaling factor to be set customly
        Multiplies with p_mw and q_mvar of all phases.

        **type** (string,default: wye) -  type variable to classify three ph load: delta/wye

        **index** (int,default: None) - Force a specified ID if it is available. If None, the index\
            one higher than the highest already existing index is selected.

        **in_service** (boolean) - True for in_service or False for out of service

    OUTPUT:
        **index** (int) - The unique ID of the created element

    EXAMPLE:
        **create_asymmetric_load(net, bus=0, p_c_mw=9., q_c_mvar=1.8)**

    """
    _check_element(net, bus)

    index = _get_index_with_check(net, "asymmetric_load", index, name="3 phase asymmetric_load")

    entries = {
        "name": name,
        "bus": bus,
        "p_a_mw": p_a_mw,
        "p_b_mw": p_b_mw,
        "p_c_mw": p_c_mw,
        "scaling": scaling,
        "q_a_mvar": q_a_mvar,
        "q_b_mvar": q_b_mvar,
        "q_c_mvar": q_c_mvar,
        "sn_a_mva": sn_a_mva,
        "sn_b_mva": sn_b_mva,
        "sn_c_mva": sn_c_mva,
        "sn_mva": sn_mva,
        "in_service": in_service,
        "type": type,
        **kwargs,
    }
    _set_entries(net, "asymmetric_load", index, True, entries=entries)

    return index


# =============================================================================
# def create_impedance_load(net, bus, r_A , r_B , r_C, x_A=0, x_B=0, x_C=0,
#                      sn_mva=nan, name=None, scaling=1.,
#                     index = None, in_service=True, type=None,
#                     ):
#     """
#     Creates a constant impedance load element ABC.
#
#     INPUT:
#         **net** - The net within this constant impedance load should be created
#
#         **bus** (int) - The bus id to which the load is connected
#
#         **sn_mva** (float) - rated power of the load
#
#         **r_A** (float) - Resistance in Phase A
#         **r_B** (float) - Resistance in Phase B
#         **r_C** (float) - Resistance in Phase C
#         **x_A** (float) - Reactance in Phase A
#         **x_B** (float) - Reactance in Phase B
#         **x_C** (float) - Reactance in Phase C
#
#
#         **kwargs are passed on to the create_load function
#
#     OUTPUT:
#         **index** (int) - The unique ID of the created load
#
#     Load elements are modeled from a consumer point of view. Active power will therefore always be
#     positive, reactive power will be positive for under-excited behavior (Q absorption, decreases voltage) and
#     negative for over-excited behavior (Q injection, increases voltage)
#     """
#     if bus not in net["bus"].index.values:
#         raise UserWarning("Cannot attach to bus %s, bus does not exist" % bus)
#
#     if index is None:
#         index = get_free_id(net["asymmetric_load"])
#     if index in net["impedance_load"].index:
#         raise UserWarning("A 3 phase asymmetric_load with the id %s already exists" % index)
#
#     # store dtypes
#     dtypes = net.impedance_load.dtypes
#
#     net.impedance_load.loc[index, ["name", "bus", "r_A","r_B","r_C", "scaling",
#                       "x_A","x_B","x_C","sn_mva", "in_service", "type"]] = \
#     [name, bus, r_A,r_B,r_C, scaling,
#       x_A,x_B,x_C,sn_mva, in_service, type]
#
#     # and preserve dtypes
#     _preserve_dtypes(net.impedance_load, dtypes)
#
#     return index
#
# =============================================================================


def create_load_from_cosphi(  # no index ?
    net: pandapowerNet, bus: Int, sn_mva: float, cos_phi: float, mode: UnderOverExcitedType, **kwargs
) -> Int:
    """
    Creates a load element from rated power and power factor cos(phi).

    INPUT:
        **net** - The net within this static generator should be created

        **bus** (int) - The bus id to which the load is connected

        **sn_mva** (float) - rated power of the load

        **cos_phi** (float) - power factor cos_phi

        **mode** (str) - "underexcited" (Q absorption, decreases voltage) or "overexcited"
                         (Q injection, increases voltage)

    OPTIONAL:
        same as in create_load, keyword arguments are passed to the create_load function

    OUTPUT:
        **index** (int) - The unique ID of the created load

    Load elements are modeled from a consumer point of view. Active power will therefore always be
    positive, reactive power will be positive for underexcited behavior (Q absorption, decreases voltage) and negative
    for overexcited behavior (Q injection, increases voltage).
    """
    from pandapower.toolbox import pq_from_cosphi

    p_mw, q_mvar = pq_from_cosphi(sn_mva, cos_phi, qmode=mode, pmode="load")
    return create_load(net, bus, sn_mva=sn_mva, p_mw=p_mw, q_mvar=q_mvar, **kwargs)


def create_load_dc(
    net: pandapowerNet,
    bus_dc: Int,
    p_dc_mw: float,
    scaling: float = 1.0,
    type: str | None = None,
    index: Int | None = None,
    name: str | None = None,
    in_service: bool = True,
    controllable: bool = False,
    **kwargs,
):
    """
    Creates a dc voltage source in a dc grid with an adjustable set point
    INPUT:

        **net** (pandapowerNet) - The pandapower network in which the element is created

        **bus_dc** (int) - index of the dc bus the dc load is connected to

        **p_dc_mw** (float) - The power of the load

    OPTIONAL:
        **name** (str, None) - element name

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **in_service** (bool, True) - True for in service or False for out of service.

        **scaling** (float, default 1.) - An OPTIONAL scaling factor, is multiplied with p_dc_mw.

        **type** (str) - A string describing the type.

        **controllable** (boolean, default NaN) - States, whether a load is controllable or not. \
            Only respected for OPF; defaults to False if "controllable" column exists in DataFrame

    OUTPUT:
        **index** (int) - The unique ID of the created svc

    """
    _check_element(net, bus_dc, element="bus_dc")

    index = _get_index_with_check(net, "source_dc", index=index)

    entries = {
        "name": name,
        "bus_dc": bus_dc,
        "p_dc_mw": p_dc_mw,
        "in_service": in_service,
        "scaling": scaling,
        "type": type,
        "controllable": controllable,
        **kwargs,
    }
    _set_entries(net, "load_dc", index, True, entries=entries)

    return index
