# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging
import os
import pickle
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

from pandapower.converter.pypower import from_ppc, to_ppc, validate_from_ppc
from pandapower.converter.pypower.from_ppc import _branch_to_which, _gen_to_which
from pandapower.file_io import from_json
from pandapower.networks.power_system_test_cases import case9, case24_ieee_rts
from pandapower.pypower.idx_brch import PF, PT, QF, QT
from pandapower.pypower.idx_bus import BASE_KV, PD, VA, VM
from pandapower.pypower.idx_gen import PG, QG
from pandapower.results import reset_results
from pandapower.run import runopp, runpp, set_user_pf_options
from pandapower.toolbox.comparison import nets_equal
from pandapower.toolbox.data_modification import reindex_buses
from test import test_path

if find_spec("pypower"):
    import pypower.case24_ieee_rts as c24

    pypower_installed = True
else:
    pypower_installed = False

logger = logging.getLogger(__name__)
max_diff_values1 = {
    "bus_vm_pu": 1e-6,
    "bus_va_degree": 1e-5,
    "branch_p_mw": 1e-3,
    "branch_q_mvar": 1e-3,
    "gen_p_mw": 1e-3,
    "gen_q_mvar": 1e-3,
}


def get_testgrids(foldername, filename):
    """
    This function return the ppc (or pp net) which is saved in ppc_testgrids.p to validate the
    from_ppc function via validate_from_ppc.
    """
    folder = os.path.join(test_path, "converter", foldername)
    file = os.path.join(folder, filename)
    convert = "ppc" in filename
    ppcs = from_json(file, convert=convert)
    return ppcs


def test_from_ppc_simple_against_target():
    ppc = get_testgrids("ppc_testgrids", "case2_2.json")
    net_by_ppc = from_ppc(ppc)
    net_by_code = from_json(os.path.join(test_path, "converter", "case2_2_by_code.json"))
    reindex_buses(net_by_code, dict(zip(net_by_code.bus.index, net_by_ppc.bus.index)))
    set_user_pf_options(net_by_code)  # for assertion of nets_equal
    runpp(net_by_ppc, trafo_model="pi")
    runpp(net_by_code, trafo_model="pi")

    assert type(net_by_ppc) is type(net_by_code)
    assert net_by_ppc.converged
    assert len(net_by_ppc.bus) == len(net_by_code.bus)
    assert len(net_by_ppc.impedance) == len(
        net_by_code.impedance
    )  # TODO: due to changes in the models this tests fails.
    assert len(net_by_ppc.ext_grid) == len(net_by_code.ext_grid)
    assert nets_equal(net_by_ppc, net_by_code, check_only_results=True, atol=1e-9)


@pytest.mark.xfail
def test_validate_from_ppc_simple_against_target():
    ppc = get_testgrids(
        "ppc_testgrids", "case2_2.json"
    )  # TODO: marked as xfail since it uses wrong paths to load references
    net = from_json(os.path.join(test_path, "converter", "case2_2_by_code.json"))
    assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values1)


@pytest.mark.parametrize(
    "case_name", ["case2_1", "case2_2", "case2_3", "case2_4", "case3_1", "case3_2", "case6", "case14", "case57"]
)
def test_ppc_testgrids(case_name):
    # check ppc_testgrids
    ppc = get_testgrids("ppc_testgrids", case_name + ".json")
    net = from_ppc(ppc, f_hz=60)
    assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values1)
    logger.info(f"{case_name} has been checked successfully.")


@pytest.mark.slow
@pytest.mark.parametrize(
    "case_name", ["case4gs", "case6ww", "case24_ieee_rts", "case30", "case39", "case118"]
)  # 'case300'
def test_pypower_cases(case_name):
    # check pypower cases
    ppc = get_testgrids("pypower_cases", case_name + ".json")
    net = from_ppc(ppc, f_hz=60)
    assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values1)
    logger.info(f"{case_name} has been checked successfully.")


def test_case9_not_in_matpower():
    # --- Because there is a pypower power flow failure in generator results in case9 (which is not
    # in matpower) another max_diff_values must be used to receive an successful validation
    max_diff_values2 = {
        "bus_vm_pu": 1e-6,
        "bus_va_degree": 1e-5,
        "branch_p_mw": 1e-3,
        "branch_q_mvar": 1e-3,
        "gen_p_mw": 1e3,
        "gen_q_mvar": 1e3,
    }
    ppc = get_testgrids("pypower_cases", "case9.json")
    net = from_ppc(ppc, f_hz=60)
    assert validate_from_ppc(ppc, net, max_diff_values2)


def test_to_and_from_ppc():
    net9 = case9()
    net24 = case24_ieee_rts()
    net24.trafo.iat[1, net24.trafo.columns.get_loc("tap_side")] = "hv"

    for i, net in enumerate([net24, net9]):
        # set max_loading_percent to enable line limit conversion
        net.line["max_loading_percent"] = 100
        runpp(net)
        ppc = to_ppc(net, mode="opf", take_slack_vm_limits=False)

        # correct ppc data (to_ppc() does not convert completely)
        if i == 0:
            vm_setps = pd.concat(
                [
                    pd.Series(net.ext_grid.vm_pu.values, index=net.ext_grid.bus),
                    pd.Series(net.gen.vm_pu.values, index=net.gen.bus),
                ]
            )
            ppc["gen"][-net.sgen.shape[0] :, 5] = vm_setps.loc[net.sgen.bus].values

        net2 = from_ppc(ppc, f_hz=net.f_hz, tap_side=net.trafo.tap_side.values)
        # again add max_loading_percent to enable valid comparison
        net2.line["max_loading_percent"] = 100

        # TODO: remove after https://github.com/e2nIEE/pandapower/pull/2813:
        #  reset 3ph results (new columns not in net but in ppc, would be solved by 3ph powerflow, so not relevant)
        reset_results(net, "pf_3ph")
        reset_results(net2, "pf_3ph")

        # compare loadflow results
        runpp(net)
        runpp(net2)
        assert nets_equal(net, net2, check_only_results=True, atol=1e-10)

        # compare optimal powerflow results
        if i == 1:
            runopp(net, delta=1e-16)
            runopp(net2, delta=1e-16)
            assert nets_equal(net, net2, check_only_results=True, atol=1e-10)


def test_gencost_pwl():
    case6 = get_testgrids("ppc_testgrids", "case6.json")
    case6["gencost"] = np.array(
        [
            [1, 0, 0, 2, 1, 0, 5, 7],
            [1, 0, 0, 2, 10, -10, 15, 7],
            [1, 0, 0, 2, 1, 2, 0, 8],
        ]
    )
    net = from_ppc(case6, f_hz=60)
    assert net.pwl_cost.shape[0] == 3
    assert isinstance(net.pwl_cost.points.at[0][0], list)
    assert np.isclose(net.pwl_cost.points.at[0][0][0], 1)
    assert np.isclose(net.pwl_cost.points.at[0][0][1], 5)
    assert np.isclose(net.pwl_cost.points.at[0][0][2], 7 / (5 - 1))
    assert np.isclose(net.pwl_cost.points.at[1][0][0], 10)
    assert np.isclose(net.pwl_cost.points.at[1][0][1], 15)
    assert np.isclose(net.pwl_cost.points.at[1][0][2], 17 / (15 - 10))


def test_gencost_pwl_q():
    case6 = get_testgrids("ppc_testgrids", "case6.json")
    case6["gencost"] = np.array(
        [
            [1, 0, 0, 2, 1, 0, 5, 7],
            [1, 0, 0, 2, 10, -10, 15, 7],
            [1, 0, 0, 2, 1, 2, 0, 8],
            [1, 0, 0, 2, 1, 0, 5, 7],
            [1, 0, 0, 2, 10, -10, 15, 7],
            [1, 0, 0, 2, 1, 2, 0, 8],
        ]
    )
    net = from_ppc(case6, f_hz=60)
    assert net.pwl_cost.shape[0] == 6
    assert list(net.pwl_cost.power_type) == ["p"] * 3 + ["q"] * 3
    assert isinstance(net.pwl_cost.points.at[0], list)
    assert len(net.pwl_cost.points.at[0]) == 1
    assert isinstance(net.pwl_cost.points.at[0][0], list)
    assert np.isclose(net.pwl_cost.points.at[0][0][0], 1)
    assert np.isclose(net.pwl_cost.points.at[0][0][1], 5)
    assert np.isclose(net.pwl_cost.points.at[0][0][2], 7 / (5 - 1))
    assert np.isclose(net.pwl_cost.points.at[1][0][0], 10)
    assert np.isclose(net.pwl_cost.points.at[1][0][1], 15)
    assert np.isclose(net.pwl_cost.points.at[1][0][2], 17 / (15 - 10))


def test_gencost_poly_part():
    case6 = get_testgrids("ppc_testgrids", "case6.json")
    case6["gencost"] = np.array(
        [
            [2, 0, 0, 2, 14, 0],
            [2, 0, 0, 2, 15, 1],
        ]
    )
    net = from_ppc(case6, f_hz=60)
    assert net.poly_cost.shape[0] == 2
    assert np.allclose(net.poly_cost.cp1_eur_per_mw.values, range(14, 16))
    assert np.allclose(net.poly_cost.cp0_eur.values, [0, 1])
    assert np.allclose(net.poly_cost[["cp2_eur_per_mw2", "cq0_eur", "cq2_eur_per_mvar2"]].values, 0)


def test_gencost_poly_q():
    case6 = get_testgrids("ppc_testgrids", "case6.json")
    case6["gencost"] = np.array(
        [
            [2, 0, 0, 2, 14, 0],
            [2, 0, 0, 2, 15, 1],
            [2, 0, 0, 2, 16, 0],
            [2, 0, 0, 2, 17, 0],
            [2, 0, 0, 2, 18, 0],
            [2, 0, 0, 2, 19, 0],
        ]
    )
    net = from_ppc(case6, f_hz=60)
    assert net.poly_cost.shape[0] == 3
    assert np.allclose(net.poly_cost.cp1_eur_per_mw.values, range(14, 17))
    assert np.allclose(net.poly_cost.cp0_eur.values, [0, 1, 0])
    assert np.allclose(net.poly_cost.cq1_eur_per_mvar.values, range(17, 20))
    assert np.allclose(net.poly_cost[["cp2_eur_per_mw2", "cq0_eur", "cq2_eur_per_mvar2"]].values, 0)


def test_gencost_poly_q_part():
    case6 = get_testgrids("ppc_testgrids", "case6.json")
    case6["gencost"] = np.array(
        [
            [2, 0, 0, 2, 14, 0],
            [2, 0, 0, 2, 15, 1],
            [2, 0, 0, 2, 16, 0],
            [2, 0, 0, 2, 17, 0],
            [2, 0, 0, 2, 18, 0],
        ]
    )
    net = from_ppc(case6, f_hz=60)
    assert net.poly_cost.shape[0] == 3
    assert np.allclose(net.poly_cost.cp1_eur_per_mw.values, range(14, 17))
    assert np.allclose(net.poly_cost.cp0_eur.values, [0, 1, 0])
    assert np.allclose(net.poly_cost.cq1_eur_per_mvar.values, [17, 18, 0])
    assert np.allclose(net.poly_cost[["cp2_eur_per_mw2", "cq0_eur", "cq2_eur_per_mvar2"]].values, 0)


def test_gencost_poly_pwl():
    case6 = get_testgrids("ppc_testgrids", "case6.json")
    case6["gencost"] = np.array(
        [
            [1, 0, 0, 3, 1, 0, 5, 7, 10, 10],
            [2, 0, 0, 3, 3, 15, 0, np.nan, np.nan, np.nan],
            [2, 0, 0, 2, 16, 2, np.nan, np.nan, np.nan, np.nan],
        ]
    )
    net = from_ppc(case6, f_hz=60)
    assert net.pwl_cost.shape[0] == 1
    assert isinstance(net.pwl_cost.points.at[0], list)
    assert len(net.pwl_cost.points.at[0]) == 2
    assert isinstance(net.pwl_cost.points.at[0][0], list)
    assert np.isclose(net.pwl_cost.points.at[0][0][0], 1)
    assert np.isclose(net.pwl_cost.points.at[0][0][1], 5)
    assert np.isclose(net.pwl_cost.points.at[0][0][2], 7 / (5 - 1))
    assert np.isclose(net.pwl_cost.points.at[0][1][0], 5)
    assert np.isclose(net.pwl_cost.points.at[0][1][1], 10)
    assert np.isclose(net.pwl_cost.points.at[0][1][2], (10 - 7) / (10 - 5))
    assert net.poly_cost.shape[0] == 2
    assert np.allclose(net.poly_cost.cp1_eur_per_mw.values, [15, 16])
    assert np.allclose(net.poly_cost.cp2_eur_per_mw2.values, [3, 0])
    assert np.allclose(net.poly_cost.cp0_eur.values, [0, 2])


def test_gencost_poly_pwl_part_mix():
    case6 = get_testgrids("ppc_testgrids", "case6.json")
    case6["gencost"] = np.array(
        [
            [1, 0, 0, 3, 1, 0, 5, 7, 10, 10],
            [2, 0, 0, 3, 3, 15, 0, np.nan, np.nan, np.nan],
            [2, 0, 0, 2, 16, 2, np.nan, np.nan, np.nan, np.nan],
            [2, 0, 0, 2, 16, 2, np.nan, np.nan, np.nan, np.nan],
            [1, 0, 0, 3, 1, 0, 5, 7, 10, 10],
        ]
    )
    net = from_ppc(case6, f_hz=60, check_costs=False)
    assert net.pwl_cost.shape[0] == 2
    assert list(net.pwl_cost.power_type) == ["p", "q"]
    assert net.pwl_cost.et.tolist() == ["ext_grid", "sgen"]
    assert net.poly_cost.shape[0] == 3

    try:
        net = from_ppc(case6, f_hz=60)
        error = False
    except UserWarning:
        error = True
    assert error


@pytest.mark.skipif(not pypower_installed, reason="c24 test net is taken from mandatory pypower installation")
def test_case24_from_pypower():
    net = from_ppc(c24.case24_ieee_rts())
    runopp(net)
    assert net.OPF_converged


def _bool_arr_to_positional_column_vector(arr):
    return np.arange(len(arr), dtype=np.int64)[arr].reshape((-1, 1))


def overwrite_results_data_of_ppc_pickle(file_name, grid_names):
    folder = os.path.join(test_path, "converter")
    file = os.path.join(folder, file_name)
    with open(file, "rb") as f:
        ppcs = pickle.load(f)

    for i in grid_names:
        ppc = ppcs[i]
        net = from_ppc(ppc, f_hz=60)
        runpp(net)

        # --- determine is_line - same as in from_ppc()
        is_line, to_vn_is_leq = _branch_to_which(ppc)
        line_pos = _bool_arr_to_positional_column_vector(is_line)
        tr_pos = _bool_arr_to_positional_column_vector(~is_line)
        tr_swap_pos = _bool_arr_to_positional_column_vector(~to_vn_is_leq)

        # --- determine which gen should considered as ext_grid, gen or sgen - same as in from_ppc()
        is_ext_grid, is_gen, is_sgen = _gen_to_which(ppc)
        eg_pos = _bool_arr_to_positional_column_vector(is_ext_grid)
        gen_pos = _bool_arr_to_positional_column_vector(is_gen)
        sgen_pos = _bool_arr_to_positional_column_vector(is_sgen)

        # --- overwrite res data
        ppc["bus"][:, [VM, VA]] = net.res_bus[["vm_pu", "va_degree"]].values

        ppc["gen"][eg_pos, [PG, QG]] = net.res_ext_grid[["p_mw", "q_mvar"]].values
        ppc["gen"][gen_pos, [PG, QG]] = net.res_gen[["p_mw", "q_mvar"]].values
        ppc["gen"][sgen_pos, [PG, QG]] = net.res_sgen[["p_mw", "q_mvar"]].iloc[sum(ppc["bus"][:, PD] < 0) :].values

        cols = ["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]
        ppc["branch"][line_pos, [PF, QF, PT, QT]] = net.res_line[cols].values
        cols = pd.Series(cols).str.replace("from", "hv").str.replace("to", "lv")
        new_bra_res = net.res_trafo[cols].values
        if len(tr_swap_pos):
            new_bra_res[tr_swap_pos.flatten(), :] = np.concatenate(
                (new_bra_res[tr_swap_pos, [2, 3]], new_bra_res[tr_swap_pos, [0, 1]]), axis=1
            )
        ppc["branch"][tr_pos, [PF, QF, PT, QT]] = new_bra_res

    # --- overwrite pickle
    with open(file, "wb") as handle:
        # pickle.dump(ppcs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(ppcs, handle, protocol=4)  # as long as py3.7 is supported, 4 is the highest protocol


def test_from_ppc_without_base_kv():
    # MATPOWER calculates in per-unit and does not need baseKV, so many MATPOWER case files
    # set it to 0 (e.g. the original MATPOWER case14 and case57). from_ppc must nevertheless
    # produce a runnable net (regression test for #2516: vn_kv of 0 let runpp fail with a
    # FloatingPointError). Since the ppc branch data is in per-unit, the placeholder vn_kv
    # must not change the power flow results: they must equal those of the same case
    # converted with an arbitrary constant baseKV.
    ppc = get_testgrids("ppc_testgrids", "case6.json")
    ppc["bus"][:, BASE_KV] = 110.0
    net_ref = from_ppc(ppc)
    runpp(net_ref)

    ppc = get_testgrids("ppc_testgrids", "case6.json")
    ppc["bus"][:, BASE_KV] = 0
    net = from_ppc(ppc)
    assert (net.bus.vn_kv == 1.0).all()
    runpp(net)
    assert net.converged
    assert np.allclose(net.res_bus.vm_pu, net_ref.res_bus.vm_pu, atol=1e-6)
    assert np.allclose(net.res_bus.va_degree, net_ref.res_bus.va_degree, atol=1e-6)


def test_from_ppc_without_base_kv_logs_warning(caplog):
    # the user must notice that the vn_kv of buses without a valid baseKV is only a
    # placeholder: a warning naming the affected buses is logged
    ppc = get_testgrids("ppc_testgrids", "case6.json")
    ppc["bus"][:, BASE_KV] = 0
    with caplog.at_level(logging.WARNING, logger="pandapower.converter.pypower.from_ppc"):
        from_ppc(ppc)
    assert f"{ppc['bus'].shape[0]} buses without a valid baseKV" in caplog.text
    assert "vn_kv is set to 1.0" in caplog.text

    # with all baseKV valid, no warning is logged
    caplog.clear()
    ppc = get_testgrids("ppc_testgrids", "case6.json")
    ppc["bus"][:, BASE_KV] = 110.0
    with caplog.at_level(logging.WARNING, logger="pandapower.converter.pypower.from_ppc"):
        from_ppc(ppc)
    assert "without a valid baseKV" not in caplog.text


def test_from_ppc_with_partly_missing_base_kv():
    # a bus without baseKV gets the placeholder vn_kv of 1.0; no value is taken over from
    # a connected bus, and the valid baseKV values of the other buses are kept unchanged
    ppc = get_testgrids("ppc_testgrids", "case6.json")
    net_ref = from_ppc(get_testgrids("ppc_testgrids", "case6.json"))
    ppc["bus"][0, BASE_KV] = 0
    net = from_ppc(ppc)
    assert (net.bus.vn_kv > 0).all()
    assert net.bus.vn_kv.iloc[0] == 1.0
    assert (net.bus.vn_kv.iloc[1:] == net_ref.bus.vn_kv.iloc[1:]).all()
    runpp(net)
    assert net.converged


def _chain_ppc(base_kv, branch_status=(1, 1)):
    # three buses in a chain (slack bus 1 - bus 2 - bus 3) with loads at buses 2 and 3
    bus = np.array(
        [
            [1, 3, 0, 0, 0, 0, 1, 1.0, 0, base_kv[0], 1, 1.1, 0.9],
            [2, 1, 20, 10, 0, 0, 1, 1.0, 0, base_kv[1], 1, 1.1, 0.9],
            [3, 1, 30, 15, 0, 0, 1, 1.0, 0, base_kv[2], 1, 1.1, 0.9],
        ]
    )
    gen = np.array([[1, 50, 0, 100, -100, 1.0, 100, 1, 200, 0]], dtype=float)
    branch = np.array(
        [
            [1, 2, 0.01, 0.05, 0.0, 100, 100, 100, 0, 0, branch_status[0], -360, 360],
            [2, 3, 0.01, 0.05, 0.0, 100, 100, 100, 0, 0, branch_status[1], -360, 360],
        ],
        dtype=float,
    )
    return {"version": "2", "baseMVA": 100.0, "bus": bus, "gen": gen, "branch": branch}


def test_from_ppc_with_non_finite_base_kv():
    # a NaN baseKV is invalid as well and is replaced by the placeholder 1.0
    # (regression: NaN neither compares <= 0 nor > 0, so a validity check based on such
    # comparisons alone would leak it through)
    net = from_ppc(_chain_ppc([110.0, np.nan, 0.0]))
    assert list(net.bus.vn_kv) == [110.0, 1.0, 1.0]
    runpp(net)
    assert net.converged

    # an infinite baseKV is not a usable voltage level either and must be replaced
    net = from_ppc(_chain_ppc([110.0, np.inf, 20.0]))
    assert np.isfinite(net.bus.vn_kv).all()
    assert (net.bus.vn_kv > 0).all()
    assert list(net.bus.vn_kv) == [110.0, 1.0, 20.0]

    # if no bus has a valid baseKV, 1.0 is used for all of them
    net = from_ppc(_chain_ppc([np.nan, np.nan, np.nan]))
    assert (net.bus.vn_kv == 1.0).all()
    runpp(net)
    assert net.converged


def test_from_ppc_missing_base_kv_not_taken_from_connected_bus():
    # no value is taken over from a connected bus: a bus without baseKV gets the
    # placeholder 1.0 even if its neighbours have a valid baseKV, and this does not
    # depend on the switching state of the connecting branches either
    net = from_ppc(_chain_ppc([110.0, 0.0, 20.0]))
    assert list(net.bus.vn_kv) == [110.0, 1.0, 20.0]

    net = from_ppc(_chain_ppc([110.0, 0.0, 20.0], branch_status=(0, 1)))
    assert net.bus.vn_kv.iloc[1] == 1.0


def test_from_ppc_does_not_modify_input_bus_array():
    # from_ppc must not write the filled baseKV values into the bus array of the caller:
    # ppc["bus"] is replaced by a corrected copy instead
    ppc = get_testgrids("ppc_testgrids", "case6.json")
    input_bus = ppc["bus"]
    input_bus[0, BASE_KV] = 0
    snapshot = input_bus.copy()
    net = from_ppc(ppc)
    np.testing.assert_array_equal(input_bus, snapshot)
    assert net.bus.vn_kv.iloc[0] == 1.0


def _two_level_ppc(base_kv):
    # two voltage levels: slack bus 1 (110) - bus 2 (110) - transformer with a tap ratio
    # of 0.98 - bus 3 (20) - bus 4 (20)
    bus = np.array(
        [
            [1, 3, 0, 0, 0, 0, 1, 1.05, 0, base_kv[0], 1, 1.1, 0.9],
            [2, 2, 20, 10, 0, 0, 1, 1.02, 0, base_kv[1], 1, 1.1, 0.9],
            [3, 1, 30, 15, 0, 0, 1, 1.0, 0, base_kv[2], 1, 1.1, 0.9],
            [4, 1, 25, 12, 0, 0, 1, 1.0, 0, base_kv[3], 1, 1.1, 0.9],
        ]
    )
    gen = np.array(
        [
            [1, 0, 0, 100, -100, 1.05, 100, 1, 200, 0],
            [2, 40, 0, 80, -80, 1.02, 100, 1, 120, 0],
        ],
        dtype=float,
    )
    branch = np.array(
        [
            [1, 2, 0.01, 0.05, 0.02, 100, 100, 100, 0, 0, 1, -360, 360],
            [2, 3, 0.005, 0.04, 0.0, 50, 50, 50, 0.98, 0, 1, -360, 360],
            [3, 4, 0.02, 0.06, 0.01, 60, 60, 60, 0, 0, 1, -360, 360],
        ],
        dtype=float,
    )
    return {"version": "2", "baseMVA": 100.0, "bus": bus, "gen": gen, "branch": branch}


def test_from_ppc_mixed_base_kv_with_trafo():
    # partly missing baseKV in a net with two voltage levels and a transformer with a
    # non-unit tap ratio: the placeholder values must not change the per-unit power flow
    # results at the buses, compared to the conversion with all true baseKV given
    # (the reference conversion itself matches a pypower runpf of the same data)
    net_ref = from_ppc(_two_level_ppc([110.0, 110.0, 20.0, 20.0]))
    runpp(net_ref)
    assert net_ref.converged

    # bus 4 has no baseKV and gets the placeholder 1.0 instead of its true level 20
    net = from_ppc(_two_level_ppc([110.0, 110.0, 20.0, 0.0]))
    assert net.bus.vn_kv.iloc[3] == 1.0
    runpp(net)
    assert np.allclose(net.res_bus.vm_pu, net_ref.res_bus.vm_pu, atol=1e-6)
    assert np.allclose(net.res_bus.va_degree, net_ref.res_bus.va_degree, atol=1e-6)

    # buses 3 and 4 have no baseKV and get the placeholder 1.0 instead of their true 20:
    # their vn_kv (and absolute quantities derived from it) are placeholders, but the
    # per-unit results at the buses, which only depend on the per-unit branch data and
    # the tap ratio, are unchanged
    net = from_ppc(_two_level_ppc([110.0, 110.0, 0.0, 0.0]))
    assert list(net.bus.vn_kv) == [110.0, 110.0, 1.0, 1.0]
    runpp(net)
    assert np.allclose(net.res_bus.vm_pu, net_ref.res_bus.vm_pu, atol=1e-6)
    assert np.allclose(net.res_bus.va_degree, net_ref.res_bus.va_degree, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
