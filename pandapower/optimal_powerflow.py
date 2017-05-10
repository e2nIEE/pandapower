# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import warnings

from pandapower.idx_bus import VM
from pypower.add_userfcn import add_userfcn
from pypower.ppoption import ppoption
from scipy.sparse import csr_matrix as sparse

from pandapower.auxiliary import ppException, _clean_up
from pandapower.pd2ppc import _pd2ppc
from pandapower.powerflow import _add_auxiliary_elements
from pandapower.opf.opf import opf
from pandapower.results import _copy_results_ppci_to_ppc, reset_results, \
    _extract_results_opf


class OPFNotConverged(ppException):
    """
    Exception being raised in case optimal powerflow did not converge.
    """
    pass


def _optimal_powerflow(net, verbose, suppress_warnings, **kwargs):
    ac = net["_options"]["ac"]

    ppopt = ppoption(VERBOSE=verbose, OPF_FLOW_LIM=2, PF_DC=not ac, **kwargs)
    net["OPF_converged"] = False
    net["converged"] = False
    _add_auxiliary_elements(net)
    reset_results(net)

    ppc, ppci = _pd2ppc(net)
    if not ac:
        ppci["bus"][:, VM] = 1.0
    net["_ppc_opf"] = ppc
    if len(net.dcline) > 0:
        ppci = add_userfcn(ppci, 'formulation', _add_dcline_constraints, args=net)

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


def _add_dcline_constraints(om, net):
    # from numpy import hstack, diag, eye, zeros
    ppc = om.get_ppc()
    ndc = len(net.dcline)  ## number of in-service DC lines
    ng = ppc['gen'].shape[0]  ## number of total gens
    Adc = sparse((ndc, ng))
    gen_lookup = net._pd2ppc_lookups["gen"]

    dcline_gens_from = net.gen.index[-2 * ndc::2]
    dcline_gens_to = net.gen.index[-2 * ndc + 1::2]
    for i, (f, t, loss) in enumerate(zip(dcline_gens_from, dcline_gens_to,
                                         net.dcline.loss_percent.values)):
        Adc[i, gen_lookup[f]] = 1. + loss * 1e-2
        Adc[i, gen_lookup[t]] = 1.

    ## constraints
    nL0 = -net.dcline.loss_kw.values * 1e-3  # absolute losses
    #    L1  = -net.dcline.loss_percent.values * 1e-2 #relative losses
    #    Adc = sparse(hstack([zeros((ndc, ng)), diag(1-L1), eye(ndc)]))

    ## add them to the model
    om = om.add_constraints('dcline', Adc, nL0, nL0, ['Pg'])