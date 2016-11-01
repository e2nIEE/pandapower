# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

from .build_opf import _pd2mpc_opf, _make_objective
from .results_opf import _extract_results_opf
from .opf import opf
from pypower.ppoption import ppoption
from pandapower.run import _select_is_elements
import warnings
import pandas as pd
# TODO: Maybe rename "simple" to "maxfeedin"


def runopp(net, objectivetype="maxp", verbose=False, suppress_warnings=True):
    """ This is the first Pandapower Optimal Power Flow
    It is basically an interface to the Pypower Optimal Power Flow and can be used only with certain
    limitations. Every objective function needs to be implemented seperately. 
    Currently the following objective functions are supported:

    * "maxp"  - Maximizing the DG feed in subject to voltage and loading limits
                - Takes the constraints from the PP tables, such as min_vm_pu, max_vm_pu, min_p_kw, 
                    max_p_kw,max_q_kvar, min_q_kvar and max_loading_percent  (limits for branch currents). 
                - Supported elements are buses, lines, trafos, sgen and gen. The controllable 
                    elements can be sgen and gen
                - If a sgen is controllable, set the controllable flag for it to True
                - Uncontrollable sgens need to be constrained with max_p_kw > p_kw > min_p_kw with a 
                    very small range. e.g. p_kw is 10, so you set pmax to 10.1 kw and pmin to 9.9 kw
                - The cost for generation can be specified in the cost_per_kw and cost_per_kvar in 
                    the respective columns for gen and sgen, this is currently not used, because the 
                    costs are equal for all gens
                - uncontrollable sgens are not supported

    """
    # TODO make this kwargs???
    ppopt = ppoption(OPF_VIOLATION=1e-1, PDIPM_GRADTOL=1e-1, PDIPM_COMPTOL=1e-1,
                     PDIPM_COSTTOL=1e-1, OUT_ALL=0, VERBOSE=verbose, OPF_ALG=560)
    net["OPF_converged"] = False

    if not (net.ward.empty & net.xward.empty & net.impedance.empty):
        raise UserWarning("Ward and Xward are not yet supported in Pandapower Optimal Power Flow!")

    # select elements in service (time consuming, so we do it once)
    is_elems = _select_is_elements(net)

    if "controllable" in net.sgen.columns:
        sg_is = net.sgen[(net.sgen.in_service & net.sgen.controllable)==True]
    else:
        sg_is = pd.DataFrame()

    mpc, bus_lookup = _pd2mpc_opf(net, is_elems, sg_is)
    mpc, ppopt = _make_objective(mpc, net, is_elems, sg_is, ppopt, objectivetype)

    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opf(mpc, ppopt)

    _extract_results_opf(net, result, is_elems, bus_lookup,  "current", True)
