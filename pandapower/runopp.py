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
from .run import reset_results
from pandapower.auxiliary import ppException


class OPFNotConverged(ppException):
    """
    Exception being raised in case loadflow did not converge.
    """
    pass

def runopp(net, objectivetype="linear", verbose=False, suppress_warnings=True):
    """ Runs the  Pandapower Optimal Power Flow.
    
    INPUT:
        **net** - The Pandapower format network
        
    OPTIONAL:
        **objectivetype** (str,"linear")- either "linear" or "linear_minloss", 
        see description below
        
        **verbose** (bool, False) - If True, some basic information is printed
        
        **suppress_warnings** (bool, True) - suppress warnings in pypower
        
            If set to True, warnings are disabled during the loadflow. Because of the way data is
            processed in pypower, ComplexWarnings are raised during the loadflow. 
            These warnings are suppressed by this option, however keep in mind all other pypower 
            warnings are suppressed, too. 
    
    AVAILABLE COST FUNCTIONS:
          **linear**  
                - Minimizing the linear costs with subject to voltage and loading limits
                - Takes the constraints from the PP tables, such as min_vm_pu, max_vm_pu, min_p_kw, 
                max_p_kw,max_q_kvar, min_q_kvar and max_loading_percent(limits for branch currents). 
                - Supported elements are buses, lines, trafos, sgen and gen. The controllable elements can be sgen and gen
                - If a sgen is controllable, set the controllable flag for it to True
                - Uncontrollable gens need to be constrained with max_p_kw < p_kw < min_p_kw with a 
                very small range. e.g. p_kw is 10, so you set pmax to 10.01 kW and pmin to 9.99 kW
                
                - The cost for generation can be specified in the cost_per_kw in the respective 
                columns for gen and sgen, reactive power costs are currently not used
      
          **linear_minloss**  
                - Minimizing the linear costs and the line losses of the whole net with subject to 
                voltage and loading limits
                
                - Takes the constraints from the PP tables, such as min_vm_pu, max_vm_pu, min_p_kw, 
                max_p_kw,max_q_kvar, min_q_kvar and max_loading_percent(limits for branch currents). 
                    
                - Supported elements are buses, lines, trafos, sgen and gen. 
                The controllable elements can be sgen and gen
                    
                - If a sgen is controllable, set the controllable flag for it to True
                
                - Uncontrollable gens should to be constrained with max_p_kw > p_kw > min_p_kw with a 
                very small range. e.g. p_kw is 10, so you set pmax to 10.01 kW and pmin to 9.99 kW
                    
                - The cost for generation can be specified in the cost_per_kw in the respective 
                columns for gen and sgen, reactive power costs are currently not used
    """
    ppopt = ppoption(OPF_VIOLATION=1e-1, PDIPM_GRADTOL=1e-1, PDIPM_COMPTOL=1e-1,
                     PDIPM_COSTTOL=1e-1, OUT_ALL=0, VERBOSE=verbose, OPF_ALG=560)
    net["OPF_converged"] = False

    reset_results(net)
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
    if not net["OPF_converged"]:
        raise OPFNotConverged("Optimal Power Flow did not converge!")
