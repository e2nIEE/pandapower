# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging
import os
from importlib.util import find_spec

import numpy as np
import pytest

import pandapower as pp
from pandapower.converter.matpower import from_mpc, to_mpc
from pandapower.converter.pypower import from_ppc, to_ppc
from pandapower.networks import case14, case24_ieee_rts
from pandapower.pypower.idx_brch import F_BUS, T_BUS, TAP
from pandapower.pypower.idx_bus import BASE_KV
from pandapower.results import reset_results
from pandapower.run import runpp, set_user_pf_options
from pandapower.toolbox.comparison import nets_equal
from test import test_path

matpowercaseframes_imported = find_spec("matpowercaseframes") is not None

logger = logging.getLogger(__name__)


def test_from_mpc_mat():
    case24 = case24_ieee_rts()
    set_user_pf_options(case24)
    this_folder = os.path.join(test_path, "converter")
    mat_case = os.path.join(this_folder, "case24_ieee_rts.mat")
    case24_from_mpc = from_mpc(mat_case, f_hz=60, casename_mpc_file="mpc", tap_side="hv")
    # TODO: remove after https://github.com/e2nIEE/pandapower/pull/2813:
    #  reset 3ph results (new columns in case24 but not in from_mpc, would be solved by 3ph powerflow, so not relevant)
    reset_results(case24, "pf_3ph")
    reset_results(case24_from_mpc, "pf_3ph")

    runpp(case24)
    runpp(case24_from_mpc)

    assert case24_from_mpc.converged
    assert nets_equal(case24, case24_from_mpc, check_only_results=True)


@pytest.mark.skipif(not matpowercaseframes_imported, reason="matpowercaseframes is needed to convert .m files.")
def test_from_mpc_m():
    this_folder = os.path.join(test_path, "converter")
    mat_case = os.path.join(this_folder, "case24_ieee_rts.mat")
    m_case = os.path.join(this_folder, "case24_ieee_rts.m")
    case24_mat = from_mpc(mat_case, f_hz=60, casename_mpc_file="mpc", tap_side="hv")
    case24_m = from_mpc(m_case, f_hz=60, tap_side="hv")

    runpp(case24_mat)
    runpp(case24_m)

    assert case24_m.converged
    assert nets_equal(case24_mat, case24_m)


def test_to_and_from_mpc_case14_trafos(tmp_path):
    # Regression test for https://github.com/e2nIEE/pandapower/issues/2620:
    # the two transformers at the three-winding star point bus of case14 have a tap ratio of 1
    # (no tap changer). They must be converted back to transformers, not to impedances.
    net = case14()
    runpp(net)
    mat_file = str(tmp_path / "case14_roundtrip.mat")
    to_mpc(net, mat_file)
    net2 = from_mpc(mat_file)

    assert len(net2.trafo) == len(net.trafo) == 5
    assert len(net2.impedance) == len(net.impedance) == 0
    assert len(net2.line) == len(net.line)

    cols = ["hv_bus", "lv_bus", "sn_mva", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent"]
    trafo = net.trafo[cols].sort_values(["hv_bus", "lv_bus"]).reset_index(drop=True)
    trafo2 = net2.trafo[cols].sort_values(["hv_bus", "lv_bus"]).reset_index(drop=True)
    for col in cols[:2]:
        assert (trafo[col] == trafo2[col]).all()
    for col in cols[2:]:
        # rtol is set explicitly so that the tolerance stays dominated by atol at the
        # magnitudes of these parameters (<= ~100): the observed deviation of the round
        # trip is float64 storage noise (<= 2.3e-13), far below atol
        assert np.allclose(trafo[col].astype(float), trafo2[col].astype(float), atol=1e-6, rtol=1e-9)

    runpp(net2)
    assert net2.converged
    # rtol is set explicitly for the same reason: with the default rtol=1e-5, the
    # effective tolerance for vm_pu values around 1.0 would be ~1e-5 instead of 1e-9
    # (the deviation observed for the identical power flow inputs is 0.0)
    assert np.allclose(net.res_bus.vm_pu.values, net2.res_bus.vm_pu.values, atol=1e-9, rtol=1e-9)


def test_to_and_from_mpc_impedance_and_trafo_without_tap_changer(tmp_path):
    # A transformer without a tap changer (tap ratio 1) and an impedance element both connect
    # buses of different voltage levels. Neither of them must change its element type in a
    # to_mpc/from_mpc or to_ppc/from_ppc round trip.
    net = pp.pandapowerNet(name="test_to_and_from_mpc", sn_mva=100)
    b1, b2, b3 = pp.create_buses(net, 3, vn_kv=[110, 20, 110])
    pp.create_ext_grid(net, b1, vm_pu=1.0)
    pp.create_lines_from_parameters(
        net, [b1], [b3], length_km=1.0, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=10, max_i_ka=1.0
    )
    pp.create_impedances(net, [b1], [b2], rft_pu=0.01, xft_pu=0.05, sn_mva=100)
    pp.create_transformers_from_parameters(
        net, [b1], [b2], sn_mva=100, vn_hv_kv=110, vn_lv_kv=20, vk_percent=5, vkr_percent=0.5, pfe_kw=0, i0_percent=0
    )
    # the load makes the bus voltages depend on the parameters of both parallel branches,
    # so that the power flow comparison below is sensitive to their element types as well
    pp.create_load(net, b2, p_mw=30, q_mvar=10)
    runpp(net)
    # sanity check: the load bus voltage actually deviates from the slack voltage, so the
    # power flow comparison of the converted nets is not a trivial one
    assert abs(net.res_bus.vm_pu.iat[1] - 1.0) > 1e-3

    mat_file = str(tmp_path / "roundtrip.mat")
    to_mpc(net, mat_file)
    net_from_mat = from_mpc(mat_file)
    net_from_ppc = from_ppc(to_ppc(net))

    for net2 in (net_from_mat, net_from_ppc):
        assert len(net2.line) == len(net.line) == 1
        assert len(net2.impedance) == len(net.impedance) == 1
        assert len(net2.trafo) == len(net.trafo) == 1
        assert np.isclose(net2.impedance.rft_pu.iat[0], net.impedance.rft_pu.iat[0])
        assert np.isclose(net2.impedance.xft_pu.iat[0], net.impedance.xft_pu.iat[0])
        assert np.isclose(net2.trafo.vk_percent.iat[0], net.trafo.vk_percent.iat[0])
        assert np.isclose(net2.trafo.vkr_percent.iat[0], net.trafo.vkr_percent.iat[0])
        # the converted net must be electrically equivalent to the original net
        runpp(net2)
        assert net2.converged
        assert np.allclose(net2.res_bus.vm_pu.values, net.res_bus.vm_pu.values, atol=1e-6, rtol=1e-9)


@pytest.mark.parametrize("filtered_branch", ["out_of_service", "disconnected"])
def test_to_and_from_ppc_impedance_behind_filtered_branch(filtered_branch):
    # Branches that are filtered out of the internal ppc (an out-of-service element, or
    # in-service elements on buses that are not supplied) must not shift the rows of the
    # branches behind them: the tap ratio marking in to_ppc must still hit the impedance
    # branch, so that it is converted back to an impedance element with its parameters
    # (regression test for https://github.com/e2nIEE/pandapower/issues/2620).
    net = pp.pandapowerNet(name="test_to_and_from_ppc_filtered", sn_mva=100)
    b1, b2, b3 = pp.create_buses(net, 3, vn_kv=[110, 20, 110])
    pp.create_ext_grid(net, b1, vm_pu=1.0)
    if filtered_branch == "out_of_service":
        pp.create_lines_from_parameters(
            net,
            [b1],
            [b3],
            length_km=1.0,
            r_ohm_per_km=0.1,
            x_ohm_per_km=0.1,
            c_nf_per_km=0,
            max_i_ka=1.0,
            in_service=False,
        )
    else:
        # line and buses in service, but on an island without a supply: the connectivity
        # check of the ppc conversion marks them out of service in the ppc
        b4, b5 = pp.create_buses(net, 2, vn_kv=[110, 110])
        pp.create_lines_from_parameters(
            net, [b4], [b5], length_km=1.0, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1.0
        )
    pp.create_lines_from_parameters(
        net, [b1], [b3], length_km=1.0, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1.0
    )
    pp.create_impedances(net, [b1], [b2], rft_pu=0.01, xft_pu=0.05, sn_mva=100)

    ppci = to_ppc(net, init="flat")
    # the only branch between buses of different voltage levels is the impedance branch:
    # it must be the branch that is marked with a tap ratio of 0
    branch = ppci["branch"]
    cross_voltage = (
        ppci["bus"][branch[:, F_BUS].astype(int), BASE_KV] != ppci["bus"][branch[:, T_BUS].astype(int), BASE_KV]
    )
    assert cross_voltage.sum() == 1
    assert branch[cross_voltage, TAP][0] == 0

    net2 = from_ppc(ppci)
    assert len(net2.line) == 1
    assert len(net2.trafo) == 0
    assert len(net2.impedance) == 1
    impedance = net2.impedance.iloc[0]
    assert net2.bus.vn_kv.at[impedance.from_bus] == 110
    assert net2.bus.vn_kv.at[impedance.to_bus] == 20
    assert np.isclose(impedance.rft_pu, 0.01)
    assert np.isclose(impedance.xft_pu, 0.05)


def test_to_and_from_mpc_trafo_and_impedance_behind_filtered_line(tmp_path):
    # An out-of-service line in front of a transformer and an impedance element must not
    # shift their rows in the internal ppc: in the exported matpower case, the transformer
    # branch must keep its tap ratio of 1 and the impedance branch must be marked with a
    # tap ratio of 0, so that both are converted back to their original element type, at
    # their original buses and with their original parameters (regression test for
    # https://github.com/e2nIEE/pandapower/issues/2620).
    net = pp.pandapowerNet(name="test_to_and_from_mpc_filtered", sn_mva=100)
    b1, b2, b3 = pp.create_buses(net, 3, vn_kv=[110, 20, 110])
    pp.create_ext_grid(net, b1, vm_pu=1.0)
    pp.create_lines_from_parameters(
        net,
        [b1],
        [b3],
        length_km=1.0,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.1,
        c_nf_per_km=0,
        max_i_ka=1.0,
        in_service=False,
    )
    pp.create_transformers_from_parameters(
        net, [b1], [b2], sn_mva=100, vn_hv_kv=110, vn_lv_kv=20, vk_percent=5, vkr_percent=0.5, pfe_kw=0, i0_percent=0
    )
    pp.create_impedances(net, [b1], [b2], rft_pu=0.01, xft_pu=0.05, sn_mva=100)

    mat_file = str(tmp_path / "roundtrip_filtered.mat")
    to_mpc(net, mat_file, init="flat")
    net2 = from_mpc(mat_file)

    assert len(net2.line) == 0
    assert len(net2.trafo) == 1
    assert len(net2.impedance) == 1
    trafo = net2.trafo.iloc[0]
    assert net2.bus.vn_kv.at[trafo.hv_bus] == 110
    assert net2.bus.vn_kv.at[trafo.lv_bus] == 20
    assert np.isclose(trafo.vk_percent, 5)
    assert np.isclose(trafo.vkr_percent, 0.5)
    impedance = net2.impedance.iloc[0]
    assert net2.bus.vn_kv.at[impedance.from_bus] == 110
    assert net2.bus.vn_kv.at[impedance.to_bus] == 20
    assert np.isclose(impedance.rft_pu, 0.01)
    assert np.isclose(impedance.xft_pu, 0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
