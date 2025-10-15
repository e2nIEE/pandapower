# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
from numpy import nan, any as np_any

from pandapower.auxiliary import pandapowerNet
from pandapower.pp_types import Int, SwitchElementType, SwitchType
from pandapower.create._utils import (
    _check_element,
    _check_multiple_elements,
    _get_index_with_check,
    _get_multiple_index_with_check,
    _set_entries,
    _set_multiple_entries,
)

logger = logging.getLogger(__name__)


def create_switch(
    net: pandapowerNet,
    bus: Int,
    element: Int,
    et: SwitchElementType,
    closed: bool = True,
    type: SwitchType | None = None,
    name: str | None = None,
    index: Int | None = None,
    z_ohm: float = 0,
    in_ka: float = nan,
    **kwargs,
) -> Int:
    """
    Adds a switch in the net["switch"] table.

    Switches can be either between two buses (bus-bus switch) or at the end of a line or transformer
    element (bus-element switch).

    Two buses that are connected through a closed bus-bus switches are fused in the power flow if
    the switch is closed or separated if the switch is open.

    An element that is connected to a bus through a bus-element switch is connected to the bus
    if the switch is closed or disconnected if the switch is open.

    INPUT:
        **net** (pandapowerNet) - The net within which this switch should be created

        **bus** - The bus that the switch is connected to

        **element** - index of the element: bus id if et == "b", line id if et == "l", trafo id if \
            et == "t"

        **et** - (string) element type: "l" = switch between bus and line, "t" = switch between
            bus and transformer, "t3" = switch between bus and transformer3w, "b" = switch between
            two buses

    OPTIONAL:
        **closed** (boolean, True) - switch position: False = open, True = closed

        **type** (str, None) - indicates the type of switch: "LS" = Load Switch, "CB" = \
            Circuit Breaker, "LBS" = Load Break Switch or "DS" = Disconnecting Switch

        **z_ohm** (float, 0) - indicates the resistance of the switch, which has effect only on
            bus-bus switches, if sets to 0, the buses will be fused like before, if larger than
            0 a branch will be created for the switch which has also effects on the bus mapping

        **name** (string, default None) - The name for this switch

        **in_ka** (float, default None) - maximum current that the switch can carry
            normal operating conditions without tripping

    OUTPUT:
        **sid** - The unique switch_id of the created switch

    EXAMPLE:
        create_switch(net, bus=0, element=1, et='b', type="LS", z_ohm=0.1)

        create_switch(net, bus=0, element=1, et='l')

    """
    _check_element(net, bus)
    match(et):
        case "l":
            elm_tab = "line"
            if element not in net[elm_tab].index:
                raise UserWarning("Unknown line index")
            if net[elm_tab]["from_bus"].loc[element] != bus and net[elm_tab]["to_bus"].loc[element] != bus:
                raise UserWarning(f"Line {element} not connected to bus {bus}")
        case "t":
            elm_tab = "trafo"
            if element not in net[elm_tab].index:
                raise UserWarning("Unknown bus index")
            if net[elm_tab]["hv_bus"].loc[element] != bus and net[elm_tab]["lv_bus"].loc[element] != bus:
                raise UserWarning(f"Trafo {element} not connected to bus {bus}")
        case "t3":
            elm_tab = "trafo3w"
            if element not in net[elm_tab].index:
                raise UserWarning("Unknown trafo3w index")
            if (
                net[elm_tab]["hv_bus"].loc[element] != bus
                and net[elm_tab]["mv_bus"].loc[element] != bus
                and net[elm_tab]["lv_bus"].loc[element] != bus
            ):
                raise UserWarning(f"Trafo3w {element} not connected to bus {bus}")
        case "b":
            _check_element(net, element)
        case _:
            raise UserWarning("Unknown element type")

    index = _get_index_with_check(net, "switch", index)

    entries = {
        "bus": bus,
        "element": element,
        "et": et,
        "closed": closed,
        "type": type,
        "name": name,
        "z_ohm": z_ohm,
        "in_ka": in_ka,
        **kwargs,
    }
    _set_entries(net, "switch", index, entries=entries)

    return index


def create_switches(
    net: pandapowerNet,
    buses: Sequence,
    elements: Sequence,
    et: SwitchElementType | Sequence[str],
    closed: bool = True,
    type: SwitchType | None = None,
    name: Iterable[str] | None = None,
    index: Int | Iterable[Int] | None = None,
    z_ohm: float = 0,
    in_ka: float = nan,
    **kwargs,
) -> Int:
    """
    Adds a switch in the net["switch"] table.

    Switches can be either between two buses (bus-bus switch) or at the end of a line or transformer
    element (bus-element switch).

    Two buses that are connected through a closed bus-bus switches are fused in the power flow if
    the switch is closed or separated if the switch is open.

    An element that is connected to a bus through a bus-element switch is connected to the bus
    if the switch is closed or disconnected if the switch is open.

    INPUT:
        **net** (pandapowerNet) - The net within which this switch should be created

        **buses** (list)- The bus that the switch is connected to

        **element** (list)- index of the element: bus id if et == "b", line id if et == "l", \
            trafo id if et == "t"

        **et** - (list) element type: "l" = switch between bus and line, "t" = switch between
            bus and transformer, "t3" = switch between bus and transformer3w, "b" = switch between
            two buses

    OPTIONAL:
        **closed** (boolean, True) - switch position: False = open, True = closed

        **type** (str, None) - indicates the type of switch: "LS" = Load Switch, "CB" = \
            Circuit Breaker, "LBS" = Load Break Switch or "DS" = Disconnecting Switch

        **z_ohm** (float, 0) - indicates the resistance of the switch, which has effect only on
            bus-bus switches, if sets to 0, the buses will be fused like before, if larger than
            0 a branch will be created for the switch which has also effects on the bus mapping

        **name** (list of str, default None) - The name for this switch

        **in_ka** (float, default None) - maximum current that the switch can carry
            normal operating conditions without tripping

    OUTPUT:
        **sid** - List of switch_id of the created switches

    EXAMPLE:
        create_switches(net, buses=[0, 1], element=1, et='b', type="LS", z_ohm=0.1)

        create_switches(net, buses=[0, 1], element=1, et='l')

    """
    index = _get_multiple_index_with_check(net, "switch", index, len(buses), name="Switches")
    _check_multiple_elements(net, buses)
    rel_els = ["b", "l", "t", "t3"]
    matcher = {"b": ["bus", "buses"], "l": ["line", "lines"], "t": ["trafo", "trafos"], "t3": ["trafo3w", "trafo3ws"]}
    for typ in rel_els:
        if et == typ:
            _check_multiple_elements(net, elements, *matcher[typ])
    if np.any(np.isin(et, ["b", "l", "t"])):
        mask_all = np.array([False] * len(et))
        for typ in rel_els:
            et_arr = np.array(et)
            el_arr = np.array(elements)
            mask = et_arr == typ
            mask_all |= mask
            _check_multiple_elements(net, el_arr[mask], *matcher[typ])
        not_def = ~mask_all
        if np.any(not_def):
            raise UserWarning(f"et type {et_arr[not_def]} is not implemented")
    else:
        raise UserWarning(f"et type {et} is not implemented")

    b_arr = np.array(buses)[:, None]
    el_arr = np.array(elements)
    et_arr = np.array([et] * len(buses) if isinstance(et, str) else et)
    # Ensure switches are connected correctly.
    for typ, table, joining_busses in [
        ("l", "line", ["from_bus", "to_bus"]),
        ("t", "trafo", ["hv_bus", "lv_bus"]),
        ("t3", "trafo3w", ["hv_bus", "mv_bus", "lv_bus"]),
    ]:
        el = el_arr[et_arr == typ]
        bs = net[table].loc[el, joining_busses].values
        not_connected_mask = ~np.isin(b_arr[et_arr == typ], bs)
        if np_any(not_connected_mask):
            bus_element_pairs = zip(
                el_arr[et_arr == typ][:, None][not_connected_mask].tolist(),
                b_arr[et_arr == typ][not_connected_mask].tolist(),
            )
            raise UserWarning(f"{table.capitalize()} not connected ({table} element, bus): {list(bus_element_pairs)}")

    entries = {
        "bus": buses,
        "element": elements,
        "et": et,
        "closed": closed,
        "type": type,
        "name": name,
        "z_ohm": z_ohm,
        "in_ka": in_ka,
        **kwargs,
    }

    _set_multiple_entries(net, "switch", index, entries=entries)

    return index
