import warnings

from pypower.ppoption import ppoption
from pypower.idx_bus import VM
from pypower.add_userfcn import add_userfcn

from pandapower.auxiliary import ppException, _select_is_elements, _clean_up
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower_extensions.opf import opf
from pandapower.results import _copy_results_ppci_to_ppc, reset_results, \
    _extract_results_opf
from pandapower.powerflow import add_dcline_constraints, _add_auxiliary_elements


class OPFNotConverged(ppException):
    """
    Exception being raised in case optimal powerflow did not converge.
    """
    pass


def _optimal_powerflow(net, verbose, suppress_warnings, **kwargs):
    ac = net["_options"]["ac"]

    ppopt = ppoption(VERBOSE=verbose, OPF_FLOW_LIM=2, PF_DC=not ac, **kwargs)
    net["OPF_converged"] = False
    _add_auxiliary_elements(net)
    reset_results(net)
    # select elements in service (time consuming, so we do it once)
    net["_is_elems"] = _select_is_elements(net)

    ppc, ppci = _pd2ppc(net)
    if not ac:
        ppci["bus"][:, VM] = 1.0
    net["_ppc_opf"] = ppc
    if len(net.dcline) > 0:
        ppci = add_userfcn(ppci, 'formulation', add_dcline_constraints, args=net)

    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opf(ppci, ppopt)
    else:
        result = opf(ppci, ppopt)
    net["_ppc_opf"] = result

    if not result["success"]:
        raise OPFNotConverged("Optimal Power Flow did not converge!")

    # ppci doesn't contain out of service elements, but ppc does -> copy results accordingly
    mode = net["_options"]["mode"]
    result = _copy_results_ppci_to_ppc(result, ppc, mode=mode)

    net["_ppc_opf"] = result
    net["OPF_converged"] = True
    _extract_results_opf(net, result)
    _clean_up(net)
