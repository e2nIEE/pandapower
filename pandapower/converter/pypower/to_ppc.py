# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

from pandapower.powerflow import _pd2ppc, _select_is_elements
from pandapower.auxiliary import _add_ppc_options
try:
    import pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)


def to_ppc(net, calculate_voltage_angles=False, trafo_model="t", r_switch=0,
           check_connectivity=True):
    """
     This function converts a pandapower net to a pypower case file.

    INPUT:

        **net** - The pandapower net.

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
    net["_is_elems"] = _select_is_elements(net)
    #  do the conversion
    ppc, ppci = _pd2ppc(net)
    ppc['branch'] = ppc['branch'].real
    ppc.pop('internal')
    return ppc
