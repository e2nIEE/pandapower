__author__ = "fmeier"

from pypower.runopf import runopf
from .build_opf import _pd2mpc_opf, _make_objective
from .results_opf import _extract_results_opf
from pypower.opf import opf
from pypower.ppoption import ppoption
from pandapower.run import _pd2mpc
import numpy as np
import warnings

# TODO: check for unused imports
# TODO: Maybe rename "simple" to "maxfeedin"
# TODO: rename result tabels to e.g. res_bus_opf


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

    gen_is = net["gen"][net["gen"]["in_service"].values.astype(bool)]
    eg_is = net["ext_grid"][net["ext_grid"]["in_service"].values.astype(bool)]
    if "controllable" in net.sgen.columns:
        sg_is = net.sgen[(net.sgen.in_service & net.sgen.controllable)==True]
    else:
        sg_is = {}

    mpc, bus_lookup = _pd2mpc_opf(net, gen_is, eg_is, sg_is)
    mpc, ppopt = _make_objective(mpc, ppopt, objectivetype="maxp")

    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opf(mpc, ppopt)

    _extract_results_opf(net, result, gen_is, eg_is, bus_lookup,  "current", True)
