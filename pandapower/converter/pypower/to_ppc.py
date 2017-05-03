# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

from pandapower.auxiliary import _add_ppc_options
from pandapower.powerflow import _pd2ppc

try:
    import pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)


def to_ppc(net, calculate_voltage_angles=False, trafo_model="t", r_switch=0.0,
           check_connectivity=True, voltage_depend_loads=True, init="results"):

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
            - "pi" - transformer is modeled as equivalent PI-model. This is not recommended, since \
            it is less exact than the T-model. It is only recommended for validation with other \
            software that uses the pi-model.

        **r_switch** (float, 0.0) - resistance of bus-bus-switches. If impedance is zero, buses
        connected by a closed bus-bus switch are fused to model an ideal bus. Otherwise, they are
        modelled as branches with resistance r_switch.

        **check_connectivity** (bool, True) - Perform an extra connectivity test after the
        conversion from pandapower to PYPOWER

            If True, an extra connectivity test based on SciPy Compressed Sparse Graph Routines is
            perfomed. If check finds unsupplied buses, they are set out of service in the ppc

        **voltage_depend_loads** (bool, True) - consideration of voltage-dependent loads. If False, net.load.const_z_percent and net.load.const_i_percent are not considered, i.e. net.load.p_kw and net.load.q_kvar are considered as constant-power loads.

        **init** (str, "results") - initialization method of the converter
        pandapower ppc converter supports two methods for initializing the converter:

            - "flat"- flat start with voltage of 1.0pu and angle of 0° at all PQ-buses and 0° for PV buses as initial solution
            - "results" - voltage vector from net.res_bus is used as initial solution.


    OUTPUT:

        **ppc** - The Pypower casefile for usage with pypower

    EXAMPLE:

        import pandapower.converter as pc

        import pandapower.networks as pn

        net = pn.case9()

        ppc = pc.to_ppc(net)

    """

    # select elements in service
    net["_options"] = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="pf", copy_constraints_to_ppc=True,
                     r_switch=r_switch, init=init, enforce_q_lims=True, recycle=None,
                     voltage_depend_loads=voltage_depend_loads)
    #  do the conversion
    ppc, _ = _pd2ppc(net)
    ppc['branch'] = ppc['branch'].real
    ppc.pop('internal')
    return ppc
