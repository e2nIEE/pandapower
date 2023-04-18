# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from numpy import allclose

from pandapower.auxiliary import _add_ppc_options
from pandapower.powerflow import _pd2ppc
from pandapower.opf.validate_opf_input import _check_necessary_opf_parameters

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def to_ppc(net, calculate_voltage_angles=False, trafo_model="t", switch_rx_ratio=2,
           check_connectivity=True, voltage_depend_loads=False, init="results", mode=None,
           take_slack_vm_limits=True):
    """
    This function converts a pandapower net to a pypower case file.

    INPUT:
        **net** - The pandapower net.

    OPTIONAL:
        **calculate_voltage_angles** (bool, False) - consider voltage angles in loadflow calculation

        If True, voltage angles of ext_grids and transformer shifts are considered in the
        loadflow calculation. Considering the voltage angles is only necessary in meshed
        networks that are usually found in higher networks.

        **trafo_model** (str, "t") - transformer equivalent circuit model
        pandapower provides two equivalent circuit models for the transformer:

            - "t" - transformer is modeled as equivalent with the T-model.
            - "pi" - transformer is modeled as equivalent PI-model.

            The "pi" - Model is not recommended, since it is less exact than the T-model. It is only
            recommended for validation with other software that uses the pi-model.


        **switch_rx_ratio** (float, 2) - rx_ratio of bus-bus-switches. If impedance is zero,
        buses connected by a closed bus-bus switch are fused to model an ideal bus.
        Otherwise, they are modelled as branches with resistance defined as z_ohm column in
        switch table and this parameter

        **check_connectivity** (bool, True) - Perform an extra connectivity test after the
        conversion from pandapower to PYPOWER

            If True, an extra connectivity test based on SciPy Compressed Sparse Graph Routines is
            perfomed. If check finds unsupplied buses, they are set out of service in the ppc

        **voltage_depend_loads** (bool, False) - consideration of voltage-dependent loads.
        If False, net.load.const_z_percent and net.load.const_i_percent are not considered, i.e.
        net.load.p_mw and net.load.q_mvar are considered as constant-power loads.

        **init** (str, "results") - initialization method of the converter
        pandapower ppc converter supports two methods for initializing the converter:

            - "flat" - flat start with voltage of 1.0pu and angle of 0° at all PQ-buses and 0° for PV buses as initial solution

            - "results" - voltage vector from net.res_bus is used as initial solution.

        **mode** (str, None) - mode of power flow calculation type ("pf" - power flow, "opf" -
        optimal power flow or "sc" - short circuit). "mode" influences for instance whether opf
        cost data will be converted or which slack bus voltage limits are respected. If "mode"
        is None, cost data will be respected via mode="opf" if cost data are existing.

        **take_slack_vm_limits** (bool, True) - Per default the voltage magnitude limits are assumed
        as setpoint of the slack unit (usually net.ext_grid.vm_pu). To replace that by values from
        net.bus[["min_vm_pu", "max_vm_pu"]], take_slack_vm_limits can be set to False.

    OUTPUT:
        **ppc** - The Pypower casefile for usage with pypower

    EXAMPLE:
        >>> import pandapower.converter as pc
        >>> import pandapower.networks as pn
        >>> net = pn.case9()
        >>> ppc = pc.to_ppc(net)

    """
    if (not (net["poly_cost"].empty and net["pwl_cost"].empty) and
       mode is None) or mode == "opf":
        mode = "opf"
        _check_necessary_opf_parameters(net, logger)
    else:
        mode = "pf"

    # check init values
    if init != "flat":
        if not net.res_bus.shape[0]:
            raise UserWarning("res_bus is empty. Change the input parameter 'init' to 'flat' or "
                              "add result values to allow initialization with 'results'.")
        elif len(net.bus.index.difference(net.res_bus.index)):
            raise ValueError("The res_bus indices doesn't fit to the bus indices. Change the "
                             "input parameter 'init' to flat or correct the res_bus table.")

    # select elements in service
    net["_options"] = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode=mode, switch_rx_ratio=switch_rx_ratio, init_vm_pu=init,
                     init_va_degree=init, enforce_q_lims=True,
                     recycle=None, voltage_depend_loads=voltage_depend_loads)

    if net["_options"]["voltage_depend_loads"] and not (
            allclose(net.load.const_z_percent.values, 0) and
            allclose(net.load.const_i_percent.values, 0)):
        logger.error("to_ppc() does not consider voltage depend loads. The z and i parts of "
                     "voltage depend loads are set to additional columns 13 and 14 but the p/q part"
                     " is still unchanged.")

    #  do the conversion
    _, ppci = _pd2ppc(net)
    ppci['branch'] = ppci['branch'].real
    # ppci.pop('internal')

    if not take_slack_vm_limits:
        slack_bus = min(net.ext_grid.bus.loc[net.ext_grid.in_service].tolist() + \
                        net.gen.bus.loc[net.gen.slack & net.gen.in_service].tolist())
        slack_bus_ppci_pos = net.bus.index[net.bus.in_service].get_loc(slack_bus)
        ppci["bus"][slack_bus_ppci_pos, 11] = net.bus.max_vm_pu.at[slack_bus]
        ppci["bus"][slack_bus_ppci_pos, 12] = net.bus.min_vm_pu.at[slack_bus]

    return ppci
