# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""
Tests for the optional HELM (Holomorphic Embedding Load flow Method) solver, provided by
the external HELMpy package and selectable via ``pp.runpp(net, algorithm='helm')``.

All tests are skipped (not failed) when HELMpy is not installed, so HELM stays optional.
The HELM results are compared against Newton-Raphson, which is treated as the reference.
"""

import numpy as np
import pytest

import pandapower as pp
import pandapower.networks as nw
from pandapower.powerflow import LoadflowNotConverged

try:
    from helmpy.core import helm  # type: ignore[import-not-found, import-untyped]
    helmpy_available = True
except ImportError:
    helmpy_available = False

pytestmark = pytest.mark.skipif(not helmpy_available, reason="HELMpy is not installed")

# HELM converges to its internal mismatch (1e-8); voltages match NR to ~1e-6, derived
# quantities (powers in MW/MVAr) to ~1e-2. Use tolerances that comfortably cover this.
VM_ATOL = 1e-5
VA_ATOL = 1e-2
P_ATOL = 1e-1


def _distributed_slack_grid():
    """case14 with user-defined slack weights on the ext_grid and the gens, so the
    distributed-slack imbalance is split by slack_weight rather than by generation.

    Tiny hand-built grids are avoided here because HELM converges in very few power-series
    terms on trivial networks, which makes the Pade approximant ill-conditioned; the
    standard MATPOWER cases give a robust, non-trivial test.
    """
    net = nw.case14()
    net.ext_grid["slack_weight"] = 2.0
    net.gen["slack_weight"] = 1.0
    return net


def _run_both(net_factory, **kwargs):
    """Run NR and HELM on fresh copies of the net and return (net_nr, net_helm)."""
    net_nr = net_factory()
    net_helm = net_factory()
    pp.runpp(net_nr, algorithm='nr', **kwargs)
    pp.runpp(net_helm, algorithm='helm', **kwargs)
    return net_nr, net_helm


def _assert_bus_parity(net_nr, net_helm):
    assert np.allclose(net_helm.res_bus.vm_pu, net_nr.res_bus.vm_pu, atol=VM_ATOL)
    assert np.allclose(net_helm.res_bus.va_degree, net_nr.res_bus.va_degree, atol=VA_ATOL)


def _assert_full_parity(net_nr, net_helm):
    _assert_bus_parity(net_nr, net_helm)
    if len(net_nr.res_gen):
        assert np.allclose(net_helm.res_gen.q_mvar, net_nr.res_gen.q_mvar, atol=P_ATOL)
        assert np.allclose(net_helm.res_gen.p_mw, net_nr.res_gen.p_mw, atol=P_ATOL)
    assert np.allclose(net_helm.res_ext_grid.p_mw, net_nr.res_ext_grid.p_mw, atol=P_ATOL)
    assert np.allclose(net_helm.res_ext_grid.q_mvar, net_nr.res_ext_grid.q_mvar, atol=P_ATOL)
    assert np.allclose(net_helm.res_line.p_from_mw, net_nr.res_line.p_from_mw, atol=P_ATOL)
    assert np.allclose(net_helm.res_line.q_from_mvar, net_nr.res_line.q_from_mvar, atol=P_ATOL)


# ---------------------------------------------------------------------------------------
# Voltage parity vs Newton-Raphson
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("case_name", ["case9", "case14", "case30", "case118"])
def test_helm_voltage_parity_standard_cases(case_name):
    net_nr, net_helm = _run_both(getattr(nw, case_name))
    assert net_helm.converged
    _assert_bus_parity(net_nr, net_helm)


# ---------------------------------------------------------------------------------------
# Full result parity (gen Q, slack injection, branch flows) via pfsoln
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("case_name", ["case30", "case118"])
def test_helm_full_result_parity(case_name):
    net_nr, net_helm = _run_both(getattr(nw, case_name))
    _assert_full_parity(net_nr, net_helm)


# ---------------------------------------------------------------------------------------
# Options coverage
# ---------------------------------------------------------------------------------------
def test_helm_enforce_q_lims():
    net_nr, net_helm = _run_both(nw.case30, enforce_q_lims=True)
    _assert_full_parity(net_nr, net_helm)


def test_helm_distributed_slack_matches_slack_weight():
    # user-defined slack_weight must be honoured -> identical to NR distributed slack
    net_nr, net_helm = _run_both(_distributed_slack_grid, distributed_slack=True)
    _assert_full_parity(net_nr, net_helm)


def test_helm_distributed_slack_without_weights_converges():
    # no slack_weight set on standard case -> HELM uses its generation-proportional default;
    # it must still converge to a sensible voltage profile.
    net = nw.case9()
    pp.runpp(net, algorithm='helm', distributed_slack=True)
    assert net.converged
    assert net.res_bus.vm_pu.between(0.8, 1.2).all()


def test_helm_non_convergence_raises():
    net = nw.case9()
    net.load.p_mw *= 100  # massively overload -> no physical solution
    with pytest.raises(LoadflowNotConverged):
        pp.runpp(net, algorithm='helm')
