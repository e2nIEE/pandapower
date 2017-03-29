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
           check_connectivity=True):
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
            it is less exact than the T-model. It is only recommended for valdiation with other \
            software that uses the pi-model.

        **r_switch** (float, 0.0) - resistance of bus-bus-switches. If impedance is zero, buses
        connected by a closed bus-bus switch are fused to model an ideal bus. Otherwise, they are
        modelled as branches with resistance r_switch.

        **check_connectivity** (bool, True) - Perform an extra connectivity test after the
        conversion from pandapower to PYPOWER

            If True, an extra connectivity test based on SciPy Compressed Sparse Graph Routines is
            perfomed. If check finds unsupplied buses, they are set out of service in the ppc

    OUTPUT:

        **ppc** - The Pypower casefile for usage with pypower

    EXAMPLE:

        import pandapower.converter as pc

        import pandapower.networks as pn

        net = pn.case9()

        ppc = pc.pp2ppc(net)

    """

    # select elements in service
    net["_options"] = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="pf", copy_constraints_to_ppc=True,
                     r_switch=r_switch, init="results", enforce_q_lims=True, recycle=None)
    #  do the conversion
    ppc, _ = _pd2ppc(net)
    ppc['branch'] = ppc['branch'].real
    ppc.pop('internal')
    return ppc
