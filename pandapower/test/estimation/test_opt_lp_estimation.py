# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pytest
import sys
import pandapower as pp
import pandapower.networks as nw
from pandapower.estimation import estimate
from pandapower.estimation.util import add_virtual_meas_from_loadflow

from pandapower.estimation.algorithm.optimization import OptAlgorithm
from pandapower.estimation.algorithm.base import WLSAlgorithm
from pandapower.estimation.algorithm.lp import LPAlgorithm
from pandapower.estimation.ppc_conversion import pp2eppci
from pandapower.estimation.results import eppci2pp


def test_case9_compare_classical_wls_opt_wls():
    net = nw.case9()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net)

    # give it a warm start
    net, ppc, eppci = pp2eppci(net)
    estimation_wls = WLSAlgorithm(1e-3, 3)
    estimation_opt = OptAlgorithm(1e-6, 1000)

    eppci = estimation_wls.estimate(eppci)
    eppci = estimation_opt.estimate(eppci, estimator="wls", verbose=False)
    if not estimation_opt.successful:
        raise AssertionError("Estimation failed due to algorithm failing!")
    net = eppci2pp(net, ppc, eppci)

    net_wls = net.deepcopy()
    estimate(net_wls)

    if not (np.allclose(net_wls.res_bus_est.vm_pu.copy(), net.res_bus_est.vm_pu.copy(),
                        atol=1e-2) and
            np.allclose(net_wls.res_bus_est.va_degree.copy(), net.res_bus_est.va_degree.copy(),
                        atol=1e-2)):
        raise AssertionError("Estimation failed!")


def test_lp_scipy_lav():
    '''
    If OR-Tools is installed, run this test.
    '''
    # Set the solver
    LPAlgorithm.ortools_available = False

    net = nw.case9()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net, with_random_error=False)

    net, ppc, eppci       = pp2eppci(net)
    estimation_ortools_lp = LPAlgorithm(1e-3, 5)

    estimation_ortools = estimation_ortools_lp.estimate(eppci, with_ortools=False)

    net = eppci2pp(net, ppc, eppci)

    if not np.allclose(net.res_bus.vm_pu, net.res_bus_est.vm_pu, atol=1e-2) or \
            not np.allclose(net.res_bus.va_degree, net.res_bus_est.va_degree, atol=5e-2):
        raise AssertionError("Estimation failed!")


def test_lp_ortools_lav():
    '''
    If OR-Tools is installed, run this test.
    '''
    # Set the solver
    LPAlgorithm.ortools_available = True
    net = nw.case9()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net, with_random_error=False)

    net, ppc, eppci = pp2eppci(net)
    estimation_ortools_lp = LPAlgorithm(1e-3, 5)

    estimation_ortools = estimation_ortools_lp.estimate(eppci, with_ortools=True)

    net = eppci2pp(net, ppc, eppci)

    if not np.allclose(net.res_bus.vm_pu, net.res_bus_est.vm_pu, atol=1e-2) or \
            not np.allclose(net.res_bus.va_degree, net.res_bus_est.va_degree, atol=5e-2):
        raise AssertionError("Estimation failed!")

def test_lp_lav():
    '''
    This will test the default LP solver installed.
    If OR-Tools is installed, it will use it. Otherwise scipy is used.
    '''
    net = nw.case9()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net, p_std_dev=0.01, q_std_dev=0.01)

    estimate(net, algorithm="lp")

    if not np.allclose(net.res_bus.vm_pu, net.res_bus_est.vm_pu, atol=1e-2) or \
            not np.allclose(net.res_bus.va_degree, net.res_bus_est.va_degree, atol=5e-2):
        raise AssertionError("Estimation failed!")

def test_opt_lav():
    net = nw.case9()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net, with_random_error=False)

    net, ppc, eppci = pp2eppci(net)
    estimation_wls = WLSAlgorithm(1e-3, 5)
    estimation_opt = OptAlgorithm(1e-6, 1000)

    eppci = estimation_wls.estimate(eppci)
    eppci = estimation_opt.estimate(eppci, estimator="lav", verbose=False)

    net = eppci2pp(net, ppc, eppci)

    if not np.allclose(net.res_bus.vm_pu, net.res_bus_est.vm_pu, atol=1e-2) or \
            not np.allclose(net.res_bus.va_degree, net.res_bus_est.va_degree, atol=5e-2):
        raise AssertionError("Estimation failed!")

@pytest.mark.skipif((sys.version_info[0] == 3) & (sys.version_info[1] <= 7), 
                   reason="This test can fail under Python 3.7 depending"
                   "on the processing power of the hardware used.")
def test_ql_qc():
    net = nw.case9()
    net.sn_mva = 1.
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net, p_std_dev=0.01, q_std_dev=0.01)
    pf_vm_pu, pf_va_degree = net.res_bus.vm_pu, net.res_bus.va_degree

    #  give it a warm start
    net, ppc, eppci = pp2eppci(net)
    estimation_wls = WLSAlgorithm(1e-3, 5)
    estimation_opt = OptAlgorithm(1e-6, 3000)

    eppci = estimation_wls.estimate(eppci)

    eppci = estimation_opt.estimate(eppci, estimator="ql", a=3, verbose=False)
    if not estimation_opt.successful:
        eppci = estimation_opt.estimate(eppci, estimator="ql", a=3, opt_method="Newton-CG", verbose=False)

    if not estimation_opt.successful:
        raise AssertionError("Estimation failed due to algorithm failing!")

    net = eppci2pp(net, ppc, eppci)

    if not np.allclose(pf_vm_pu, net.res_bus_est.vm_pu, atol=1e-2) or \
            not np.allclose(pf_va_degree, net.res_bus_est.va_degree, atol=5e-2):
        raise AssertionError("Estimation failed!")

    # give it a warm start
    net, ppc, eppci = pp2eppci(net)
    estimation_wls = WLSAlgorithm(1e-6, 5)
    estimation_opt = OptAlgorithm(1e-6, 3000)

    eppci = estimation_wls.estimate(eppci)
    eppci = estimation_opt.estimate(eppci, estimator="qc", a=3, verbose=False)
    if not estimation_opt.successful:
        eppci = estimation_opt.estimate(eppci, estimator="qc", a=3, opt_method="Newton-CG", verbose=False)
    net = eppci2pp(net, ppc, eppci)

    if not np.allclose(pf_vm_pu, net.res_bus_est.vm_pu, atol=1e-2) or \
            not np.allclose(pf_va_degree, net.res_bus_est.va_degree, atol=5e-2):
        raise AssertionError("Estimation failed!")


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
