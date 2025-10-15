# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging
from typing import Iterable, Sequence, Literal

import numpy as np
import numpy.typing as npt

from pandapower.auxiliary import pandapowerNet
from pandapower.pp_types import Int
from pandapower.create._utils import (
    _check_element,
    _check_multiple_elements,
    _get_index_with_check,
    _get_multiple_index_with_check,
    _set_entries,
    _set_multiple_entries,
    _set_value_if_not_nan
)

logger = logging.getLogger(__name__)


def create_shunt(
    net: pandapowerNet,
    bus: Int,
    q_mvar: float,
    p_mw: float = 0.,
    vn_kv: Optional[float] = None,
    step: int = 1,
    max_step: int = 1,
    name: Optional[str] = None,
    step_dependency_table: bool = False,
    id_characteristic_table: Optional[int] = None,
    in_service: bool = True,
    index: Optional[Int] = None,
    **kwargs
) -> Int:
    """
    Creates a shunt element.

    INPUT:
        **net** (pandapowerNet) - the pandapower network in which the element is created

        **bus** (int) - index of the bus the shunt is connected to

        **p_mw** (float) - shunt active power in MW at v = 1.0 p.u. per step

        **q_mvar** (float) - shunt reactive power in MVAr at v = 1.0 p.u. per step

    OPTIONAL:
        **vn_kv** (float, None) - rated voltage of the shunt. Defaults to rated voltage of connected bus, since this \
            value is mandatory for powerflow calculations. If it is set to NaN it will be replaced by the bus vn_kv \
            during power flow

        **step** (int, 1) - step of shunt with which power values are multiplied

        **max_step** (int, 1) - maximum allowed step of shunt

        **name** (str, None) - element name

        **step_dependency_table** (boolean, False) - True if shunt parameters (p_mw, q_mvar) must be adjusted \
            dependent on the step of the shunt. Requires the additional column "id_characteristic_table". \
            The function pandapower.control.shunt_characteristic_table_diagnostic can be used for sanity checks. \
            The function pandapower.control.create_shunt_characteristic_object can be used to create \
            SplineCharacteristic objects in the net.shunt_characteristic_spline table and add the additional column \
            "id_characteristic_spline" to set up the reference to the spline characteristics.

        **id_characteristic_table** (int, None) - references the index of the characteristic from the lookup table \
                                                 net.shunt_characteristic_table

        **in_service** (boolean, True) - True for in_service or False for out of service

        **index** (int, None) - force a specified ID if it is available. If None, the index one higher than the \
                                highest already existing index is selected.

    OUTPUT:
        **index** (int) - the unique ID of the created shunt

    EXAMPLE:
        create_shunt(net, 0, 20)
    """
    _check_element(net, bus)

    index = _get_index_with_check(net, "shunt", index)

    if vn_kv is None:
        vn_kv = net.bus.vn_kv.at[bus]

    entries = {"bus": bus, "name": name, "p_mw": p_mw, "q_mvar": q_mvar, "vn_kv": vn_kv, "step": step,
               "max_step": max_step, "in_service": in_service, "step_dependency_table": step_dependency_table,
               "id_characteristic_table": id_characteristic_table, **kwargs}
    _set_entries(net, "shunt", index, entries=entries)

    _set_value_if_not_nan(net, index, id_characteristic_table, "id_characteristic_table",
                          "shunt", dtype="Int64")

    return index


def create_shunts(
    net: pandapowerNet,
    buses: Sequence,
    q_mvar: float | Iterable[float],
    p_mw: float | Iterable[float] = 0.,
    vn_kv: Optional[float | Iterable[float]] = None,
    step: int | Iterable[int] = 1,
    max_step: int | Iterable[int] = 1,
    name: Optional[Iterable[str]] = None,
    step_dependency_table: bool | Iterable[bool] = False,
    id_characteristic_table: Optional[int] | Iterable[Optional[int]] = None,
    in_service: bool | Iterable[bool] = True,
    index = None,
    **kwargs
) -> npt.NDArray[np.array]:
    """
    Creates a number of shunt elements.

    INPUT:
        **net** (pandapowerNet) - The pandapower network in which the element is created

        **buses** (list of ints) - bus numbers of buses to which the shunts should be connected to

        **p_mw** (list of floats, 0) - shunts' active power in MW at v = 1.0 p.u.

        **q_mvar** (list of floats) - shunts' reactive power in MVAr at v = 1.0 p.u.

    OPTIONAL:
        **vn_kv** (list of floats, None) - rated voltage of the shunts. Defaults to rated voltage of connected bus, since this \
            value is mandatory for powerflow calculations. If it is set to NaN it will be replaced by the bus vn_kv \
            during power flow

        **step** (list of ints, 1) - step of shunts with which power values are multiplied

        **max_step** (list of ints, 1) - maximum allowed step of shunts

        **name** (list of strs, None) - element name

        **step_dependency_table** (list of booleans, False) - True if shunt parameters (p_mw, q_mvar) must be \
            adjusted dependent on the step of the shunts. Requires the additional column "id_characteristic_table". \
            The function pandapower.control.shunt_characteristic_table_diagnostic can be used for sanity checks. \
            The function pandapower.control.create_shunt_characteristic_object can be used to create \
            SplineCharacteristic objects in the net.shunt_characteristic_spline table and add the additional column \
            "id_characteristic_spline" to set up the reference to the spline characteristics.

        **id_characteristic_table** (list of ints, None) - references the index of the characteristic from the lookup \
                                                          table net.shunt_characteristic_table

        **in_service** (list of booleans, True) - True for in_service or False for out of service

        **index** (list of ints, None) - force a specified ID if it is available. If None, the \
                                         index one higher than the highest already existing index is selected.

    OUTPUT:
        **index** (list of ints) - the list of IDs of the created shunts

    EXAMPLE:
        create_shunts(net, [0, 2], [20, 30])
    """
    _check_multiple_elements(net, buses)

    index = _get_multiple_index_with_check(net, "shunt", index, len(buses))

    if vn_kv is None:
        vn_kv = net.bus.vn_kv.loc[buses]

    entries = {"bus": buses, "name": name, "p_mw": p_mw, "q_mvar": q_mvar, "vn_kv": vn_kv, "step": step,
               "max_step": max_step, "in_service": in_service, "step_dependency_table": step_dependency_table,
               "id_characteristic_table": id_characteristic_table, **kwargs}
    _set_multiple_entries(net, "shunt", index, entries=entries)

    return index


def create_shunt_as_capacitor(
    net: pandapowerNet,
    bus: Int,
    q_mvar: float,
    loss_factor: float,
    **kwargs
) -> Int:
    """
    Creates a shunt element representing a capacitor bank.

    INPUT:

        **net** (pandapowerNet) - The pandapower network in which the element is created

        **bus** (int) - index of the bus the shunt is connected to

        **q_mvar** (float) - reactive power of the capacitor bank at rated voltage

        **loss_factor** (float) - loss factor tan(delta) of the capacitor bank

    OPTIONAL:
        same as in create_shunt, keyword arguments are passed to the create_shunt function

    OUTPUT:
        **index** (int) - the unique ID of the created shunt
    """
    q_mvar = -abs(q_mvar)  # q is always negative for capacitor
    p_mw = abs(q_mvar * loss_factor)  # p is always positive for active power losses
    return create_shunt(net, bus, q_mvar=q_mvar, p_mw=p_mw, **kwargs)


def create_svc(
    net: pandapowerNet,
    bus: Int,
    x_l_ohm: float,
    x_cvar_ohm: float,
    set_vm_pu: float,
    thyristor_firing_angle_degree: float,
    name: Optional[str] = None,
    controllable: bool = True,
    in_service: bool = True,
    index: Optional[Int] = None,
    min_angle_degree: float = 90,
    max_angle_degree: float = 180,
    **kwargs
) -> Int:
    """
    Creates an SVC element - a shunt element with adjustable impedance used to control the voltage \
        at the connected bus

    Does not work if connected to "PV" bus (gen bus, ext_grid bus)

    min_angle_degree, max_angle_degree are placeholders (ignored in the Newton-Raphson power \
        flow at the moment).

    INPUT:
        **net** (pandapowerNet) - The pandapower network in which the element is created

        **bus** (int) - connection bus of the svc

        **x_l_ohm** (float) - inductive reactance of the reactor component of svc

        **x_cvar_ohm** (float) - capacitive reactance of the fixed capacitor component of svc

        **set_vm_pu** (float) - set-point for the bus voltage magnitude at the connection bus

        **thyristor_firing_angle_degree** (float) - the value of thyristor firing angle of svc (is used directly if
            controllable==False, otherwise is the starting point in the Newton-Raphson calculation)

    OPTIONAL:
        **name** (list of strs, None) - element name

        **controllable** (bool, True) - whether the element is considered as actively controlling or
            as a fixed shunt impedance

        **in_service** (bool, True) - True for in_service or False for out of service

        **index** (int, None) - Force a specified ID if it is available. If None, the
            index one higher than the highest already existing index is selected.

        **min_angle_degree** (float, 90) - minimum value of the thyristor_firing_angle_degree

        **max_angle_degree** (float, 180) - maximum value of the thyristor_firing_angle_degree

    OUTPUT:
        **index** (int) - The unique ID of the created svc

    """

    _check_element(net, bus)

    index = _get_index_with_check(net, "svc", index)

    entries = {"name": name, "bus": bus, "x_l_ohm": x_l_ohm, "x_cvar_ohm": x_cvar_ohm, "set_vm_pu": set_vm_pu,
               "thyristor_firing_angle_degree": thyristor_firing_angle_degree, "controllable": controllable,
               "in_service": in_service, "min_angle_degree": min_angle_degree, "max_angle_degree": max_angle_degree,
               **kwargs}
    _set_entries(net, "svc", index, entries=entries)

    return index


def create_ssc(
    net: pandapowerNet,
    bus: Int,
    r_ohm: float,
    x_ohm: float,
    set_vm_pu: float = 1.,
    vm_internal_pu: float = 1.,
    va_internal_degree: float = 0.,
    name: Optional[str] = None,
    controllable: bool = True,
    in_service: bool = True,
    index: Optional[Int] = None,
    **kwargs
) -> Int:
    """
    Creates an SSC element (STATCOM)- a shunt element with adjustable VSC internal voltage used to control the voltage \
        at the connected bus

    Does not work if connected to "PV" bus (gen bus, ext_grid bus)


    INPUT:
        **net** (pandapowerNet) - The pandapower network in which the element is created

        **bus** (int) - connection bus of the ssc

        **r_ohm** (float) - resistance of the coupling transformer component of ssc

        **x_ohm** (float) - reactance of the coupling transformer component of ssc

        **set_vm_pu** (float) - set-point for the bus voltage magnitude at the connection bus

        **vm_internal_pu** (float) -  The voltage magnitude of the voltage source converter VSC at the ssc component.
                                    if the amplitude of the VSC output voltage is increased above that of the ac system
                                    voltage, the VSC behaves as a capacitor and reactive power is supplied to the ac
                                    system, decreasing the output voltage below that of the ac system leads to the VSC
                                    consuming reactive power acting as reactor.(source PhD Panosyan)


        **va_internal_degree** (float) - The voltage angle of the voltage source converter VSC at the ssc component.

    OPTIONAL:
        **name** (list of strs, None) - element name

        **controllable** (bool, True) - whether the element is considered as actively controlling or
            as a fixed shunt impedance

        **in_service** (bool, True) - True for in_service or False for out of service

        **index** (int, None) - Force a specified ID if it is available. If None, the
            index one higher than the highest already existing index is selected.

    OUTPUT:
        **index** (int) - The unique ID of the created ssc

    """

    _check_element(net, bus)

    index = _get_index_with_check(net, "ssc", index)

    entries = {"name": name, "bus": bus, "r_ohm": r_ohm, "x_ohm": x_ohm, "set_vm_pu": set_vm_pu,
               "vm_internal_pu": vm_internal_pu, "va_internal_degree": va_internal_degree, "controllable": controllable,
               "in_service": in_service, **kwargs}
    _set_entries(net, "ssc", index, entries=entries)

    return index


def create_b2b_vsc(
        net: pandapowerNet,
        bus: Int,
        bus_dc_plus: Int,
        bus_dc_minus: Int,
        r_ohm: float,
        x_ohm: float,
        r_dc_ohm: float,
        pl_dc_mw: float = 0.,
        control_mode_ac: str = "vm_pu",
        control_value_ac: float = 1.,
        control_mode_dc: str = "p_mw",
        control_value_dc: float = 0.,
        name: Optional[str] = None,
        controllable: bool = True,
        in_service: bool = True,
        index: Optional[Int] = None,
        **kwargs
) -> Int:
    """
    Creates an VSC converter element - a shunt element with adjustable VSC internal voltage used to connect the \
    AC grid and the DC grid. The element implements several control modes.

    Does not work if connected to "PV" bus (gen bus, ext_grid bus)

    INPUT:
        **net** (pandapowerNet) - The pandapower network in which the element is created

        **bus** (int) - AC connection of the B2B VSC

        **bus_dc_plus** (int) - connection bus of the plus side of the B2B VSC

        **bus_dc_minus** (int) - connection bus of the minus side of the B2B VSC

        **r_ohm** (float) - resistance of the coupling transformer component of B2B VSC

        **x_ohm** (float) - reactance of the coupling transformer component of B2B VSC

        **r_dc_ohm** (float) - resistance of the internal dc resistance component of B2B VSC

        **pl_dc_mw** (float) - no-load losses of the B2B VSC on the DC side for the shunt R representing the no load losses

        **control_mode_ac** (string) - the control mode of the ac side of the VSC. it could be "vm_pu", "q_mvar" or "slack"

        **control_value_ac** (float) - the value of the controlled parameter at the ac bus in "p.u." or "MVAr"

        **control_mode_dc** (string) - the control mode of the dc side of the B2B VSC. it could be "vm_pu" or "p_mw"

        **control_value_dc** (float) - the value of the controlled parameter at the dc bus in "p.u." or "MW"

    OPTIONAL:
        **name** (list of strs, None) - element name

        **controllable** (bool, True) - whether the element is considered as actively controlling or
            as a fixed voltage source connected via shunt impedance

        **in_service** (bool, True) - True for in_service or False for out of service

        **index** (int, None) - Force a specified ID if it is available. If None, the
            index one higher than the highest already existing index is selected.

    OUTPUT:
        **index** (int) - The unique ID of the created ssc

    """

    _check_element(net, bus)
    _check_element(net, bus_dc_plus, "bus_dc")
    _check_element(net, bus_dc_minus, "bus_dc")

    index = _get_index_with_check(net, "b2b_vsc", index)

    entries = {"name": name, "bus": bus, "bus_dc_plus": bus_dc_plus, "bus_dc_minus": bus_dc_minus, "r_ohm": r_ohm,
               "x_ohm": x_ohm, "r_dc_ohm": r_dc_ohm, "pl_dc_mw": pl_dc_mw, "control_mode_ac": control_mode_ac,
               "control_value_ac": control_value_ac, "control_mode_dc": control_mode_dc,
               "control_value_dc": control_value_dc, "controllable": controllable, "in_service": in_service, **kwargs}
    _set_entries(net, "b2b_vsc", index, entries=entries)

    return index


def create_bi_vsc(
        net: pandapowerNet,
        bus: Int,
        bus_dc_plus: Int,
        bus_dc_minus: Int,
        r_ohm: float,
        x_ohm: float,
        r_dc_ohm: float,
        pl_dc_mw: float = 0.,
        control_mode_ac: str = "vm_pu",
        control_value_ac: float = 1.,
        control_mode_dc: str = "p_mw",
        control_value_dc: float = 0.,
        name: Optional[str] = None,
        controllable: bool = True,
        in_service: bool = True,
        index: Optional[Int] = None,
        **kwargs
) -> Int:
    """
    Creates an VSC converter element - a shunt element with adjustable VSC internal voltage used to connect the \
    AC grid and the DC grid. The element implements several control modes.

    Does not work if connected to "PV" bus (gen bus, ext_grid bus)

    INPUT:
        **net** (pandapowerNet) - The pandapower network in which the element is created

        **bus** (int) - connection bus of the VSC

        **bus_dc** (int) - connection bus of the VSC

        **r_ohm** (float) - resistance of the coupling transformer component of VSC

        **x_ohm** (float) - reactance of the coupling transformer component of VSC

        **r_dc_ohm** (float) - resistance of the internal dc resistance component of VSC

        **pl_dc_mw** (float) - no-load losses of the VSC on the DC side for the shunt R representing the no load losses

        **control_mode_ac** (string) - the control mode of the ac side of the VSC. it could be "vm_pu", "q_mvar" or "slack"

        **control_value_ac** (float) - the value of the controlled parameter at the ac bus in "p.u." or "MVAr"

        **control_mode_dc** (string) - the control mode of the dc side of the VSC. it could be "vm_pu" or "p_mw"

        **control_value_dc** (float) - the value of the controlled parameter at the dc bus in "p.u." or "MW"

    OPTIONAL:
        **name** (list of strs, None) - element name

        **controllable** (bool, True) - whether the element is considered as actively controlling or
            as a fixed voltage source connected via shunt impedance

        **in_service** (bool, True) - True for in_service or False for out of service

        **index** (int, None) - Force a specified ID if it is available. If None, the
            index one higher than the highest already existing index is selected.

    OUTPUT:
        **index** (int) - The unique ID of the created ssc

    """

    _check_element(net, bus)
    _check_element(net, bus_dc_plus, "bus_dc")
    _check_element(net, bus_dc_minus, "bus_dc")

    index = _get_index_with_check(net, "bi_vsc", index)

    entries = {"name": name, "bus": bus, "bus_dc_plus": bus_dc_plus, "bus_dc_minus": bus_dc_minus, "r_ohm": r_ohm,
               "x_ohm": x_ohm, "r_dc_ohm": r_dc_ohm, "pl_dc_mw": pl_dc_mw, "control_mode_ac": control_mode_ac,
               "control_value_ac": control_value_ac, "control_mode_dc": control_mode_dc,
               "control_value_dc": control_value_dc, "controllable": controllable, "in_service": in_service, **kwargs}
    _set_entries(net, "bi_vsc", index, entries=entries)

    return index


def create_vsc(
    net: pandapowerNet,
    bus: Int,
    bus_dc: Int,
    r_ohm: float,
    x_ohm: float,
    r_dc_ohm: float,
    pl_dc_mw: float = 0.,
    control_mode_ac: Literal["vm_pu", "q_mvar"] = "vm_pu",
    control_value_ac: float = 1.,
    control_mode_dc: Literal["vm_pu", "p_mw"] = "p_mw",
    control_value_dc: float = 0.,
    name: Optional[str] = None,
    controllable: bool = True,
    in_service: bool = True,
    index: Optional[Int] = None,
    ref_bus = None,
    **kwargs
) -> Int:
    """
    Creates an VSC converter element - a shunt element with adjustable VSC internal voltage used to connect the \
    AC grid and the DC grid. The element implements several control modes.

    Does not work if connected to "PV" bus (gen bus, ext_grid bus)

    INPUT:
        **net** (pandapowerNet) - The pandapower network in which the element is created

        **bus** (int) - connection bus of the VSC

        **bus_dc** (int) - connection bus of the VSC

        **r_ohm** (float) - resistance of the coupling transformer component of VSC

        **x_ohm** (float) - reactance of the coupling transformer component of VSC

        **r_dc_ohm** (float) - resistance of the internal dc resistance component of VSC

        **pl_dc_mw** (float) - no-load losses of the VSC on the DC side for the shunt R representing the no load losses

        **control_mode_ac** (string) - the control mode of the ac side of the VSC. it could be "vm_pu", "q_mvar" or "slack"

        **control_value_ac** (float) - the value of the controlled parameter at the ac bus in "p.u." or "MVAr"

        **control_mode_dc** (string) - the control mode of the dc side of the VSC. it could be "vm_pu" or "p_mw"

        **control_value_dc** (float) - the value of the controlled parameter at the dc bus in "p.u." or "MW"

    OPTIONAL:
        **name** (list of strs, None) - element name

        **controllable** (bool, True) - whether the element is considered as actively controlling or
            as a fixed voltage source connected via shunt impedance

        **in_service** (bool, True) - True for in_service or False for out of service

        **index** (int, None) - Force a specified ID if it is available. If None, the
            index one higher than the highest already existing index is selected.

    OUTPUT:
        **index** (int) - The unique ID of the created ssc

    """

    _check_element(net, bus)
    _check_element(net, bus_dc, "bus_dc")

    index = _get_index_with_check(net, "vsc", index)

    entries = {"name": name, "bus": bus, "bus_dc": bus_dc, "r_ohm": r_ohm, "x_ohm": x_ohm, "r_dc_ohm": r_dc_ohm,
               "pl_dc_mw": pl_dc_mw, "control_mode_ac": control_mode_ac, "control_value_ac": control_value_ac,
               "control_mode_dc": control_mode_dc, "control_value_dc": control_value_dc, "controllable": controllable,
               "in_service": in_service, "ref_bus": ref_bus, **kwargs}
    _set_entries(net, "vsc", index, entries=entries)

    return index
