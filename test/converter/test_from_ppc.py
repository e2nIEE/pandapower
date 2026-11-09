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
from pandapower.pypower.idx_brch import ANGMAX, ANGMIN, PF, PT, QF, QT
from pandapower.pypower.idx_bus import PD, VA, VM
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

        # unconstrained cases must not gain the angle limit columns in the round trip
        assert np.all(ppc["branch"][:, ANGMIN] == -360.0)
        assert np.all(ppc["branch"][:, ANGMAX] == 360.0)

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


def test_from_ppc_angle_limits():
    """ANGMIN/ANGMAX must be imported; the pypower 'no limit' spellings must become NaN."""
    ppc = get_testgrids("pypower_cases", "case9.json")
    ppc["branch"][0, ANGMIN] = -30.0
    ppc["branch"][0, ANGMAX] = 20.0
    ppc["branch"][1, ANGMIN] = 0.0  # MATPOWER convention: no limit
    ppc["branch"][1, ANGMAX] = 360.0
    ppc["branch"][2, ANGMIN] = -360.0
    ppc["branch"][2, ANGMAX] = 0.0

    net = from_ppc(ppc, f_hz=60)

    assert np.isclose(net.line.at[0, "min_angle_degree"], -30.0)
    assert np.isclose(net.line.at[0, "max_angle_degree"], 20.0)
    assert net.line["min_angle_degree"].iloc[1:].isnull().all()
    assert net.line["max_angle_degree"].iloc[1:].isnull().all()


def test_from_ppc_no_angle_columns_for_unconstrained_case():
    """A fully unconstrained case must not gain the angle columns at all."""
    ppc = get_testgrids("pypower_cases", "case9.json")
    net = from_ppc(ppc, f_hz=60)
    assert "min_angle_degree" not in net.line.columns
    assert "max_angle_degree" not in net.line.columns
    assert "min_angle_degree" not in net.trafo.columns
    assert "max_angle_degree" not in net.trafo.columns
    assert "min_angle_degree" not in net.impedance.columns
    assert "max_angle_degree" not in net.impedance.columns


def test_from_ppc_angle_limits_reversed_trafo():
    """For a trafo branch oriented lv -> hv in the ppc, the limits are swapped and negated."""
    mpc = {
        "branch": np.array(
            [
                # f_bus, t_bus, r, x, b, rateA, rateB, rateC, ratio, shift, status, angmin, angmax
                [2, 1, 0.001, 0.01, 0.0, 100.0, 0.0, 0.0, 1.0, 0.0, 1, -30.0, 10.0],
            ]
        ),
        "bus": np.array(
            [
                [1, 3, 0.0, 0.0, 0.0, 0.0, 1, 1.0, 0.0, 110.0, 1, 1.1, 0.9],
                [2, 1, 50.0, 10.0, 0.0, 0.0, 1, 1.0, 0.0, 10.0, 1, 1.1, 0.9],
            ]
        ),
        "gen": np.array([[1, 50.0, 0.0, 100.0, -100.0, 1.0, 100.0, 1, 100.0, 0.0]]),
        "gencost": np.array([[2, 0.0, 0.0, 3, 0.0, 10.0, 0.0]]),
        "version": 2,
        "baseMVA": 100.0,
    }

    net = from_ppc(mpc, f_hz=50)
    # pandapower models the trafo hv -> lv, which is the reverse ppc direction,
    # so va_lv - va_hv in [-30, 10] becomes va_hv - va_lv in [-10, 30]
    assert np.isclose(net.trafo.at[0, "min_angle_degree"], -10.0)
    assert np.isclose(net.trafo.at[0, "max_angle_degree"], 30.0)


def test_angle_limits_roundtrip():
    """from_ppc -> to_ppc -> from_ppc must preserve the limits."""
    ppc = get_testgrids("pypower_cases", "case9.json")
    ppc["branch"][0, ANGMIN] = -30.0
    ppc["branch"][0, ANGMAX] = 20.0
    net = from_ppc(ppc, f_hz=60)
    net.line["max_loading_percent"] = 100.0
    runpp(net)

    net2 = from_ppc(to_ppc(net, mode="opf"), f_hz=60)

    for col in ("min_angle_degree", "max_angle_degree"):
        assert np.allclose(net.line[col].to_numpy(), net2.line[col].to_numpy(), equal_nan=True), (
            f"{col} not preserved by the round trip"
        )


def test_from_ppc_branch_without_angle_columns():
    """A branch array that stops before ANGMIN/ANGMAX must convert unconstrained instead of raising.

    from_ppc accepts raw ppc dicts, and before the angle limits were imported only columns up to BR_STATUS
    were read. Such truncated branch arrays must keep working.
    """
    ppc = get_testgrids("pypower_cases", "case9.json")
    ppc["branch"] = ppc["branch"][:, : ANGMIN]  # drop ANGMIN and ANGMAX

    net = from_ppc(ppc, f_hz=60)

    assert len(net.line) > 0
    assert "min_angle_degree" not in net.line.columns
    assert "max_angle_degree" not in net.line.columns


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


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
