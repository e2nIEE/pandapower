# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import numpy.typing as npt

from pandapower.auxiliary import pandapowerNet
from pandapower.pp_types import Int
from pandapower.create._utils import (
    _check_branch_element,
    _check_multiple_branch_elements,
    _get_index_with_check,
    _get_multiple_index_with_check,
    _set_entries,
    _set_multiple_entries,
    _set_value_if_not_nan,
)

logger = logging.getLogger(__name__)


def create_impedance(
    net: pandapowerNet,
    from_bus: Int,
    to_bus: Int,
    rft_pu: float,
    xft_pu: float,
    sn_mva: float,
    rtf_pu: float | None = None,
    xtf_pu: float | None = None,
    name: str | None = None,
    in_service: bool = True,
    index: Int | None = None,
    rft0_pu: float | None = None,
    xft0_pu: float | None = None,
    rtf0_pu: float | None = None,
    xtf0_pu: float | None = None,
    gf_pu: float | None = 0,
    bf_pu: float | None = 0,
    gt_pu: float | None = None,
    bt_pu: float | None = None,
    gf0_pu: float | None = None,
    bf0_pu: float | None = None,
    gt0_pu: float | None = None,
    bt0_pu: float | None = None,
    **kwargs,
) -> Int:
    """
    Creates an impedance element in per unit (pu).

    Parameters
    ----------
    net : pandapowerNet
        The pandapower grid model in which the element is created.

    from_bus : int
        The starting bus of the impedance element.

    to_bus : int
        The ending bus of the impedance element.

    rft_pu : float
        The real part of the impedance from 'from_bus' to 'to_bus' in per unit.

    xft_pu : float
        The imaginary part of the impedance from 'from_bus' to 'to_bus' in per unit.

    sn_mva : float
        The rated power of the impedance element in MVA.

    rtf_pu : float, optional
        The real part of the impedance from 'to_bus' to 'from_bus' in per unit. Defaults to `rft_pu`.

    xtf_pu : float, optional
        The imaginary part of the impedance from 'to_bus' to 'from_bus' in per unit. Defaults to `xft_pu`.

    name : str, optional
        The name of the impedance element. Default is None.

    in_service : bool, optional
        The service status of the impedance element. Default is True.

    index : int, optional
        The index of the impedance element. Default is None.

    rft0_pu : float, optional
        The zero-sequence real part of the impedance from 'from_bus' to 'to_bus' in per unit. Default is None.

    xft0_pu : float, optional
        The zero-sequence imaginary part of the impedance from 'from_bus' to 'to_bus' in per unit. Default is None.

    rtf0_pu : float, optional
        The zero-sequence real part of the impedance from 'to_bus' to 'from_bus' in per unit. Default is `rft0_pu`.

    xtf0_pu : float, optional
        The zero-sequence imaginary part of the impedance from 'to_bus' to 'from_bus' in per unit. Default is `xft0_pu`.

    gf_pu : float, optional
        Conductance at the 'from_bus' in per unit. Default is 0.

    bf_pu : float, optional
        Susceptance at the 'from_bus' in per unit. Default is 0.

    gt_pu : float, optional
        Conductance at the 'to_bus' in per unit. Defaults to `gf_pu`.

    bt_pu : float, optional
        Susceptance at the 'to_bus' in per unit. Defaults to `bf_pu`.

    gf0_pu : float, optional
        The zero-sequence conductance at the 'from_bus' in per unit. Default is None.

    bf0_pu : float, optional
        The zero-sequence susceptance at the 'from_bus' in per unit. Default is None.

    gt0_pu : float, optional
        The zero-sequence conductance at the 'to_bus' in per unit. Defaults to `gf0_pu`.

    bt0_pu : float, optional
        The zero-sequence susceptance at the 'to_bus' in per unit. Defaults to `bf0_pu`.

    kwargs : dict, optional
        Additional arguments (for additional columns in net.impedance table).

    Returns
    -------
    int
        The index of the created impedance element.

    Raises
    ------
    UserWarning
        If required impedance parameters are missing.
    """

    index = _get_index_with_check(net, "impedance", index)

    _check_branch_element(net, "Impedance", index, from_bus, to_bus)

    if (
        rft_pu is None
        or xft_pu is None
        or (rft0_pu is None and rtf0_pu is not None)
        or (xft0_pu is None and xtf0_pu is not None)
    ):
        raise UserWarning("*ft_pu parameters are missing for impedance element")

    if rtf_pu is None:
        rtf_pu = rft_pu
    if xtf_pu is None:
        xtf_pu = xft_pu
    if rft0_pu is not None and rtf0_pu is None:
        rtf0_pu = rft0_pu
    if xft0_pu is not None and xtf0_pu is None:
        xtf0_pu = xft0_pu

    if gt_pu is None:
        gt_pu = gf_pu
    if bt_pu is None:
        bt_pu = bf_pu
    if gf0_pu is not None and gt0_pu is None:
        gt0_pu = gf0_pu
    if bf0_pu is not None and bt0_pu is None:
        bt0_pu = bf0_pu

    entries = {
        "from_bus": from_bus,
        "to_bus": to_bus,
        "rft_pu": rft_pu,
        "xft_pu": xft_pu,
        "rtf_pu": rtf_pu,
        "xtf_pu": xtf_pu,
        "gf_pu": gf_pu,
        "bf_pu": bf_pu,
        "gt_pu": gt_pu,
        "bt_pu": bt_pu,
        "name": name,
        "sn_mva": sn_mva,
        "in_service": in_service,
        **kwargs,
    }
    _set_entries(net, "impedance", index, entries=entries)

    if rft0_pu is not None:
        _set_value_if_not_nan(net, index, rft0_pu, "rft0_pu", "impedance")
        _set_value_if_not_nan(net, index, xft0_pu, "xft0_pu", "impedance")
        _set_value_if_not_nan(net, index, rtf0_pu, "rtf0_pu", "impedance")
        _set_value_if_not_nan(net, index, xtf0_pu, "xtf0_pu", "impedance")

    if gf0_pu is not None:
        _set_value_if_not_nan(net, index, gf0_pu, "gf0_pu", "impedance")
        _set_value_if_not_nan(net, index, bf0_pu, "bf0_pu", "impedance")
        _set_value_if_not_nan(net, index, gt0_pu, "gt0_pu", "impedance")
        _set_value_if_not_nan(net, index, bt0_pu, "bt0_pu", "impedance")

    return index


def create_impedances(
    net: pandapowerNet,
    from_buses: Sequence,
    to_buses: Sequence,
    rft_pu: float | Iterable[float],
    xft_pu: float | Iterable[float],
    sn_mva: float | Iterable[float],
    rtf_pu: float | Iterable[float] | None = None,
    xtf_pu: float | Iterable[float] | None = None,
    name: Iterable[str] | None = None,
    in_service: bool | Iterable[str] = True,
    index: Int | Iterable[Int] | None = None,
    rft0_pu: float | Iterable[float] | None = None,
    xft0_pu: float | Iterable[float] | None = None,
    rtf0_pu: float | Iterable[float] | None = None,
    xtf0_pu: float | Iterable[float] | None = None,
    gf_pu: float | Iterable[float] | None = 0,
    bf_pu: float | Iterable[float] | None = 0,
    gt_pu: float | Iterable[float] | None = None,
    bt_pu: float | Iterable[float] | None = None,
    gf0_pu: float | Iterable[float] | None = None,
    bf0_pu: float | Iterable[float] | None = None,
    gt0_pu: float | Iterable[float] | None = None,
    bt0_pu: float | Iterable[float] | None = None,
    **kwargs,
) -> npt.NDArray[np.array]:
    """
    Creates an impedance element in per unit (pu).

    Parameters
    ----------
    net : pandapowerNet
        The pandapower grid model in which the element is created.

    from_buses : int
        The starting buses of the impedance element.

    to_buses : int
        The ending buses of the impedance element.

    rft_pu : float
        The real part of the impedance from 'from_bus' to 'to_bus' in per unit.

    xft_pu : float
        The imaginary part of the impedance from 'from_bus' to 'to_bus' in per unit.

    sn_mva : float
        The rated power of the impedance element in MVA.

    rtf_pu : float, optional
        The real part of the impedance from 'to_bus' to 'from_bus' in per unit. Defaults to `rft_pu`.

    xtf_pu : float, optional
        The imaginary part of the impedance from 'to_bus' to 'from_bus' in per unit. Defaults to `xft_pu`.

    name : str, optional
        The name of the impedance element. Default is None.

    in_service : bool, optional
        The service status of the impedance element. Default is True.

    index : int, optional
        The index of the impedance element. Default is None.

    rft0_pu : float, optional
        The zero-sequence real part of the impedance from 'from_bus' to 'to_bus' in per unit. Default is None.

    xft0_pu : float, optional
        The zero-sequence imaginary part of the impedance from 'from_bus' to 'to_bus' in per unit. Default is None.

    rtf0_pu : float, optional
        The zero-sequence real part of the impedance from 'to_bus' to 'from_bus' in per unit. Default is `rft0_pu`.

    xtf0_pu : float, optional
        The zero-sequence imaginary part of the impedance from 'to_bus' to 'from_bus' in per unit. Default is `xft0_pu`.

    gf_pu : float, optional
        Conductance at the 'from_bus' in per unit. Default is 0.

    bf_pu : float, optional
        Susceptance at the 'from_bus' in per unit. Default is 0.

    gt_pu : float, optional
        Conductance at the 'to_bus' in per unit. Defaults to `gf_pu`.

    bt_pu : float, optional
        Susceptance at the 'to_bus' in per unit. Defaults to `bf_pu`.

    gf0_pu : float, optional
        The zero-sequence conductance at the 'from_bus' in per unit. Default is None.

    bf0_pu : float, optional
        The zero-sequence susceptance at the 'from_bus' in per unit. Default is None.

    gt0_pu : float, optional
        The zero-sequence conductance at the 'to_bus' in per unit. Defaults to `gf0_pu`.

    bt0_pu : float, optional
        The zero-sequence susceptance at the 'to_bus' in per unit. Defaults to `bf0_pu`.

    kwargs : dict, optional
        Additional arguments (for additional columns in net.impedance table).

    Returns
    -------
    int
        The index of the created impedance element.

    Raises
    ------
    UserWarning
        If required impedance parameters are missing.
    """
    _check_multiple_branch_elements(net, from_buses, to_buses, "Impedances")

    index = _get_multiple_index_with_check(net, "impedance", index, len(from_buses))

    if (
        rft_pu is None
        or xft_pu is None
        or (rft0_pu is None and rtf0_pu is not None)
        or (xft0_pu is None and xtf0_pu is not None)
    ):
        raise UserWarning("*ft_pu parameters are missing for impedance element")

    if rtf_pu is None:
        rtf_pu = rft_pu
    if xtf_pu is None:
        xtf_pu = xft_pu
    if rft0_pu is not None and rtf0_pu is None:
        rtf0_pu = rft0_pu
    if xft0_pu is not None and xtf0_pu is None:
        xtf0_pu = xft0_pu

    if gt_pu is None:
        gt_pu = gf_pu
    if bt_pu is None:
        bt_pu = bf_pu
    if gf0_pu is not None and gt0_pu is None:
        gt0_pu = gf0_pu
    if bf0_pu is not None and bt0_pu is None:
        bt0_pu = bf0_pu

    entries = {
        "from_bus": from_buses,
        "to_bus": to_buses,
        "rft_pu": rft_pu,
        "xft_pu": xft_pu,
        "rtf_pu": rtf_pu,
        "xtf_pu": xtf_pu,
        "gf_pu": gf_pu,
        "bf_pu": bf_pu,
        "gt_pu": gt_pu,
        "bt_pu": bt_pu,
        "name": name,
        "sn_mva": sn_mva,
        "in_service": in_service,
        **kwargs,
    }
    _set_multiple_entries(net, "impedance", index, entries=entries)

    if rft0_pu is not None:
        _set_value_if_not_nan(net, index, rft0_pu, "rft0_pu", "impedance")
        _set_value_if_not_nan(net, index, xft0_pu, "xft0_pu", "impedance")
        _set_value_if_not_nan(net, index, rtf0_pu, "rtf0_pu", "impedance")
        _set_value_if_not_nan(net, index, xtf0_pu, "xtf0_pu", "impedance")

    if gf0_pu is not None:
        _set_value_if_not_nan(net, index, gf0_pu, "gf0_pu", "impedance")
        _set_value_if_not_nan(net, index, bf0_pu, "bf0_pu", "impedance")
        _set_value_if_not_nan(net, index, gt0_pu, "gt0_pu", "impedance")
        _set_value_if_not_nan(net, index, bt0_pu, "bt0_pu", "impedance")

    return index


def create_tcsc(
    net: pandapowerNet,
    from_bus: Int,
    to_bus: Int,
    x_l_ohm: float,
    x_cvar_ohm: float,
    set_p_to_mw: float,
    thyristor_firing_angle_degree: float,
    name: str | None = None,
    controllable: bool = True,
    in_service: bool = True,
    index: Int | None = None,
    min_angle_degree: float = 90,
    max_angle_degree: float = 180,
    **kwargs,
) -> Int:
    """
    Creates a TCSC element - series impedance compensator to control series reactance.
    The TCSC device allows controlling the active power flow through the path it is connected in.

    Multiple TCSC elements in net are possible.
    Unfortunately, TCSC is not implemented for the case when multiple TCSC elements
    have the same from_bus or the same to_bus.

    Note: in the Newton-Raphson power flow calculation, the initial voltage vector is adjusted slightly
    if the initial voltage at the from_bus is the same as at the to_bus to avoid
    some terms in J (for TCSC) becoming zero.

    min_angle_degree, max_angle_degree are placeholders (ignored in the Newton-Raphson power flow at the moment).

    Parameters:
        net: The pandapower network in which the element is created
        from_bus: starting bus of the tcsc
        to_bus: ending bus of the tcsc
        x_l_ohm: impedance of the reactor component of tcsc
        x_cvar_ohm: impedance of the fixed capacitor component of tcsc
        set_p_to_mw: set-point for the branch active power at the to_bus
        thyristor_firing_angle_degree: the value of thyristor firing angle of tcsc (is used directly if controllable==False, otherwise is the starting point in the Newton-Raphson calculation)
        name: element name
        controllable: whether the element is considered as actively controlling or as a fixed series impedance
        in_service: True for in_service or False for out of service
        index: Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.
        min_angle_degree: minimum value of the thyristor_firing_angle_degree
        max_angle_degree: maximum value of the thyristor_firing_angle_degree

    Returns:
        The ID of the created tcsc
    """
    index = _get_index_with_check(net, "tcsc", index)

    _check_branch_element(net, "TCSC", index, from_bus, to_bus)

    entries = {
        "name": name,
        "from_bus": from_bus,
        "to_bus": to_bus,
        "x_l_ohm": x_l_ohm,
        "x_cvar_ohm": x_cvar_ohm,
        "set_p_to_mw": set_p_to_mw,
        "thyristor_firing_angle_degree": thyristor_firing_angle_degree,
        "controllable": controllable,
        "in_service": in_service,
        "min_angle_degree": min_angle_degree,
        "max_angle_degree": max_angle_degree,
        **kwargs,
    }
    _set_entries(net, "tcsc", index, entries=entries)

    return index


def create_series_reactor_as_impedance(
    net: pandapowerNet,
    from_bus: Int,
    to_bus: Int,
    r_ohm: float,
    x_ohm: float,
    sn_mva: float,
    name: str | None = None,
    in_service: bool = True,
    index: int | None = None,
    r0_ohm: float | None = None,
    x0_ohm: float | None = None,
    **kwargs,
) -> Int:
    """
    Creates a series reactor as per-unit impedance
    
    Parameters:
        net: The pandapower network in which the element is created
        from_bus: starting bus of the series reactor
        to_bus: ending bus of the series reactor
        r_ohm: real part of the impedance in Ohm
        x_ohm: imaginary part of the impedance in Ohm
        sn_mva: rated power of the series reactor in MVA
        name:
        in_service:
        index:
    
    Returns:
        index of the created element
    """
    if net.bus.at[from_bus, "vn_kv"] == net.bus.at[to_bus, "vn_kv"]:
        vn_kv = net.bus.at[from_bus, "vn_kv"]
    else:
        raise UserWarning(
            "Unable to infer rated voltage vn_kv for series reactor %s due to "
            "different rated voltages of from_bus %d (%.3f p.u.) and "
            "to_bus %d (%.3f p.u.)"
            % (name, from_bus, net.bus.at[from_bus, "vn_kv"], to_bus, net.bus.at[to_bus, "vn_kv"])
        )

    base_z_ohm = vn_kv**2 / sn_mva
    rft_pu = r_ohm / base_z_ohm
    xft_pu = x_ohm / base_z_ohm
    rft0_pu = r0_ohm / base_z_ohm if r0_ohm is not None else None
    xft0_pu = x0_ohm / base_z_ohm if x0_ohm is not None else None

    index = create_impedance(
        net,
        from_bus=from_bus,
        to_bus=to_bus,
        rft_pu=rft_pu,
        xft_pu=xft_pu,
        sn_mva=sn_mva,
        name=name,
        in_service=in_service,
        index=index,
        rft0_pu=rft0_pu,
        xft0_pu=xft0_pu,
        **kwargs,
    )
    return index
