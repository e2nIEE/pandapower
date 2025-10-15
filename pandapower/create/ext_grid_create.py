# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging

from numpy import nan, bool_

from pandapower.auxiliary import pandapowerNet
from pandapower.pp_types import Int
from pandapower.create._utils import (
    _check_element,
    _get_index_with_check,
    _set_entries,
    _set_value_if_not_nan,
)

logger = logging.getLogger(__name__)


def create_ext_grid(
    net: pandapowerNet,
    bus: Int,
    vm_pu: float = 1.0,
    va_degree: float = 0.,
    name: str | None = None,
    in_service: bool = True,
    s_sc_max_mva: float = nan,
    s_sc_min_mva: float = nan,
    rx_max: float = nan,
    rx_min: float = nan,
    max_p_mw: float = nan,
    min_p_mw: float = nan,
    max_q_mvar: float = nan,
    min_q_mvar: float = nan,
    index: Int | None = None,
    r0x0_max: float = nan,
    x0x_max: float = nan,
    controllable: bool | float = nan,
    slack_weight: float = 1.0,
    **kwargs
) -> Int:
    """
    Creates an external grid connection.

    External grids represent the higher level power grid connection and are modelled as the slack
    bus in the power flow calculation.

    INPUT:
        **net** - pandapower network

        **bus** (int) - bus where the slack is connected

    OPTIONAL:
        **vm_pu** (float, default 1.0) - voltage at the slack node in per unit

        **va_degree** (float, default 0.) - voltage angle at the slack node in degrees*

        **name** (string, default None) - name of the external grid

        **in_service** (boolean) - True for in_service or False for out of service

        **s_sc_max_mva** (float, NaN) - maximum short circuit apparent power to calculate internal \
            impedance of ext_grid for short circuit calculations

        **s_sc_min_mva** (float, NaN) - minimum short circuit apparent power to calculate internal \
            impedance of ext_grid for short circuit calculations

        **rx_max** (float, NaN) - maximum R/X-ratio to calculate internal impedance of ext_grid \
            for short circuit calculations

        **rx_min** (float, NaN) - minimum R/X-ratio to calculate internal impedance of ext_grid \
            for short circuit calculations

        **max_p_mw** (float, NaN) - Maximum active power injection. Only respected for OPF

        **min_p_mw** (float, NaN) - Minimum active power injection. Only respected for OPF

        **max_q_mvar** (float, NaN) - Maximum reactive power injection. Only respected for OPF

        **min_q_mvar** (float, NaN) - Minimum reactive power injection. Only respected for OPF

        **r0x0_max** (float, NaN) - maximum R/X-ratio to calculate Zero sequence
            internal impedance of ext_grid

        **x0x_max** (float, NaN) - maximum X0/X-ratio to calculate Zero sequence
            internal impedance of ext_grid

        **slack_weight** (float, default 1.0) - Contribution factor for distributed slack power flow calculation \
            (active power balancing)

        **controllable** (bool, NaN) - Control of value limits

                                        - True: p_mw, q_mvar and vm_pu limits are enforced for the \
                                             ext_grid in OPF. The voltage limits set in the \
                                             ext_grid bus are enforced.

                                        - False: p_mw and vm_pu set points are enforced and *limits are\
                                              ignored*. The vm_pu set point is enforced and limits \
                                              of the bus table are ignored. Defaults to False if \
                                              "controllable" column exists in DataFrame

        \\* considered in load flow if calculate_voltage_angles = True

    EXAMPLE:
        create_ext_grid(net, 1, voltage=1.03)

        For three phase load flow

        create_ext_grid(net, 1, voltage=1.03, s_sc_max_mva=1000, rx_max=0.1, r0x0_max=0.1,\
                       x0x_max=1.0)
    """
    _check_element(net, bus)

    index = _get_index_with_check(net, "ext_grid", index, name="external grid")

    entries = {"name": name, "bus": bus, "vm_pu": vm_pu, "va_degree": va_degree, "in_service": in_service,
               "slack_weight": slack_weight, **kwargs}
    _set_entries(net, "ext_grid", index, entries=entries)

    # OPF limits
    _set_value_if_not_nan(net, index, min_p_mw, "min_p_mw", "ext_grid")
    _set_value_if_not_nan(net, index, max_p_mw, "max_p_mw", "ext_grid")
    _set_value_if_not_nan(net, index, min_q_mvar, "min_q_mvar", "ext_grid")
    _set_value_if_not_nan(net, index, max_q_mvar, "max_q_mvar", "ext_grid")
    _set_value_if_not_nan(net, index, controllable, "controllable", "ext_grid",
                          dtype=bool_, default_val=False)
    # others
    _set_value_if_not_nan(net, index, x0x_max, "x0x_max", "ext_grid")
    _set_value_if_not_nan(net, index, r0x0_max, "r0x0_max", "ext_grid")
    _set_value_if_not_nan(net, index, s_sc_max_mva, "s_sc_max_mva", "ext_grid")
    _set_value_if_not_nan(net, index, s_sc_min_mva, "s_sc_min_mva", "ext_grid")
    _set_value_if_not_nan(net, index, rx_min, "rx_min", "ext_grid")
    _set_value_if_not_nan(net, index, rx_max, "rx_max", "ext_grid")

    return index
