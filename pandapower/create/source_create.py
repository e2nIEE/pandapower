# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging

from pandapower.auxiliary import pandapowerNet
from pandapower.pp_types import Int
from pandapower.create._utils import _check_element, _get_index_with_check, _set_entries

logger = logging.getLogger(__name__)


def create_source_dc(
    net: pandapowerNet,
    bus_dc: Int,
    vm_pu: float = 1.0,
    index: Int | None = None,
    name: str | None = None,
    in_service: bool = True,
    type: str | None = None,
    **kwargs,
):
    """
    Creates a dc voltage source in a dc grid with an adjustable set point
    
    Parameters:
        net: The pandapower network in which the element is created
        bus_dc: index of the bus the shunt is connected to
        vm_pu: set-point for the bus voltage magnitude at the connection bus
        name: element name
        index: Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.
        in_service: True for in_service or False for out of service
        type: A string describing the type.

    Returns:
        The ID of the created svc
    """
    _check_element(net, bus_dc, element="bus_dc")

    index = _get_index_with_check(net, "source_dc", index)

    entries = {"name": name, "bus_dc": bus_dc, "vm_pu": vm_pu, "in_service": in_service, "type": type, **kwargs}
    _set_entries(net, "source_dc", index, True, entries=entries)

    return index
