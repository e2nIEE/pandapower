# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

from pandas import DataFrame
import warnings

from pypower.ppoption import ppoption

from pandapower.build_opf import _pd2ppc_opf, _make_objective
from pandapower.results_opf import _extract_results_opf
from pandapower.opf import opf
from pandapower.run import _select_is_elements, reset_results
from pandapower.auxiliary import ppException


class OPFNotConverged(ppException):
    """
    Exception being raised in case loadflow did not converge.
    """
    pass


def runopp(net, cost_function="linear", verbose=False, suppress_warnings=True, **kwargs):
    """
    Runs the  Pandapower Optimal Power Flow.
    Flexibilities, constraints and cost parameters are defined in the pandapower element tables.

    Flexibilities for generators can be defined in net.sgen / net.gen.
    net.sgen.controllable / net.gen.controllable signals if a generator is controllable. If False,
    the active and reactive power are assigned as in a normal power flow. If yes, the following
    flexibilities apply:
        - net.sgen.min_p_kw / net.sgen.max_p_kw
        - net.sgen.min_q_kvar / net.sgen.max_q_kvar
        - net.gen.min_p_kw / net.gen.max_p_kw
        - net.gen.min_q_kvar / net.gen.max_q_kvar

    Network constraints can be defined for buses, lines and transformers the elements in the following columns:
        - net.bus.min_vm_pu / net.bus.max_vm_pu
        - net.line.max_loading_percent
        - net.trafo.max_loading_percent

    Costs can be assigned to generation units in the following columns:
        - net.gen.cost_per_kw
        - net.sgen.cost_per_kw
        - net.ext_grid.cost_per_kw

    How these costs are combined into a cost function depends on the cost_function parameter.

    INPUT:
        **net** - The Pandapower format network

    OPTIONAL:
        **cost_function** (str,"linear")- cost function
            - "linear" - minimizes weighted generator costs
            - "linear_minloss" - minimizes weighted generator cost and line losses

        **verbose** (bool, False) - If True, some basic information is printed

        **suppress_warnings** (bool, True) - suppress warnings in pypower

            If set to True, warnings are disabled during the loadflow. Because of the way data is
            processed in pypower, ComplexWarnings are raised during the loadflow.
            These warnings are suppressed by this option, however keep in mind all other pypower
            warnings are suppressed, too.
    """
    ppopt = ppoption(OPF_VIOLATION=1e-1, PDIPM_GRADTOL=1e-1, PDIPM_COMPTOL=1e-1,
                     PDIPM_COSTTOL=1e-1, OUT_ALL=0, VERBOSE=verbose, OPF_ALG=560)
    net["OPF_converged"] = False

    reset_results(net)
    # select elements in service (time consuming, so we do it once)
    is_elems = _select_is_elements(net)

    if "controllable" in net.sgen.columns:
        sg_is = net.sgen[(net.sgen.in_service & net.sgen.controllable) == True]
    else:
        sg_is = DataFrame()

    ppc, bus_lookup = _pd2ppc_opf(net, is_elems, sg_is)
    ppc, ppopt = _make_objective(ppc, net, is_elems, sg_is, ppopt, cost_function, **kwargs)
    net["_ppc_opf"] = ppc

    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opf(ppc, ppopt)
            if not result["success"]:
                raise OPFNotConverged("Optimal Power Flow did not converge!")

    net["_ppc_opf"] = result
    net["OPF_converged"] = True
    _extract_results_opf(net, result, is_elems, bus_lookup,  "current", True)
