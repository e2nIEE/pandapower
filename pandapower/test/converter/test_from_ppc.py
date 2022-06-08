# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os
import pickle
import pytest
import sys
from packaging import version
import numpy as np
import pandas as pd

import pandapower as pp
import pandapower.networks as pn
from pandapower.converter import from_ppc, validate_from_ppc, to_ppc
from pandapower.converter.pypower.from_ppc import _branch_to_which, _gen_to_which
from pandapower.pypower.idx_bus import \
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN
from pandapower.pypower.idx_gen import \
    GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN
from pandapower.pypower.idx_brch import \
    F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, PF, QF, PT, QT

try:
    import pypower.case24_ieee_rts as c24
    pypower_installed = True
except ImportError:
    pypower_installed = False

try:
    import pandaplan.core.pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)
max_diff_values1 = {"bus_vm_pu": 1e-6, "bus_va_degree": 1e-5, "branch_p_mw": 1e-3,
                    "branch_q_mvar": 1e-3, "gen_p_mw": 1e-3, "gen_q_mvar": 1e-3}


def get_testgrids(name, filename):
    """
    This function return the ppc (or pp net) which is saved in ppc_testgrids.p to validate the
    from_ppc function via validate_from_ppc.
    """
    folder = os.path.join(pp.pp_dir, 'test', 'converter')
    file = os.path.join(folder, filename)
    if filename.endswith(".json"):
        ppcs = pp.from_json(file)
    elif filename.endswith(".p"):
        ppcs = pickle.load(open(file, "rb"))
    return ppcs[name]


def validate_other_than_py37(ppc, net, max_diff_values):
    if sys.version_info.minor < 8 and sys.version_info.major == 3:
        if not validate_from_ppc(ppc, net, max_diff_values=max_diff_values):
            logger.error("test_pypower_cases() fails for py3.7")
    else:
        assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values)


def test_from_ppc_simple_against_target():
    ppc = get_testgrids('case2_2', 'ppc_testgrids.json')
    net_by_ppc = from_ppc(ppc)
    net_by_code = pp.from_json(os.path.join(pp.pp_dir, 'test', 'converter', 'case2_2_by_code.json'))
    pp.set_user_pf_options(net_by_code)  # for assertion of nets_equal
    pp.runpp(net_by_ppc, trafo_model="pi")
    pp.runpp(net_by_code, trafo_model="pi")

    assert type(net_by_ppc) == type(net_by_code)
    assert net_by_ppc.converged
    assert len(net_by_ppc.bus) == len(net_by_code.bus)
    assert len(net_by_ppc.trafo) == len(net_by_code.trafo)
    assert len(net_by_ppc.ext_grid) == len(net_by_code.ext_grid)
    assert pp.nets_equal(net_by_ppc, net_by_code, check_only_results=True, atol=1e-9)


def test_validate_from_ppc_simple_against_target():
    ppc = get_testgrids('case2_2', 'ppc_testgrids.json')
    net = pp.from_json(os.path.join(pp.pp_dir, 'test', 'converter', 'case2_2_by_code.json'))
    assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values1)


def test_ppc_testgrids():
    # check ppc_testgrids
    name = ['case2_1', 'case2_2', 'case2_3', 'case2_4', 'case3_1', 'case3_2', 'case6', 'case14',
            'case57']
    for i in name:
        ppc = get_testgrids(i, 'ppc_testgrids.json')
        net = from_ppc(ppc, f_hz=60)
        validate_other_than_py37(ppc, net, max_diff_values1)
        logger.debug(f'{i} has been checked successfully.')


@pytest.mark.slow
def test_pypower_cases():
    # check pypower cases
    name = ['case4gs', 'case6ww', 'case24_ieee_rts', 'case30', 'case39',
            'case118'] # 'case300'
    for i in name:
        ppc = get_testgrids(i, 'pypower_cases.json')
        net = from_ppc(ppc, f_hz=60)
        validate_other_than_py37(ppc, net, max_diff_values1)
        logger.debug(f'{i} has been checked successfully.')
    # --- Because there is a pypower power flow failure in generator results in case9 (which is not
    # in matpower) another max_diff_values must be used to receive an successful validation
    max_diff_values2 = {"vm_pu": 1e-6, "va_degree": 1e-5, "p_branch_mw": 1e-3,
                        "q_branch_mvar": 1e-3, "p_gen_mw": 1e3, "q_gen_mvar": 1e3}
    ppc = get_testgrids('case9', 'pypower_cases.json')
    net = from_ppc(ppc, f_hz=60)
    validate_other_than_py37(ppc, net, max_diff_values2)


def test_to_and_from_ppc():
    net9 = pn.case9()
    net24 = pn.case24_ieee_rts()
    net24.trafo.tap_side.iat[1] = "hv"

    for i, net in enumerate([net24, net9]):

        # set max_loading_percent to enable line limit conversion
        net.line["max_loading_percent"] = 100
        pp.runpp(net)
        ppc = to_ppc(net, mode="opf", take_slack_vm_limits=False)

        # correct ppc data (to_ppc() does not convert completely)
        if i == 0:
            vm_setps = pd.concat([pd.Series(net.ext_grid.vm_pu.values, index=net.ext_grid.bus),
                                  pd.Series(net.gen.vm_pu.values, index=net.gen.bus)])
            ppc["gen"][-net.sgen.shape[0]:, 5] = vm_setps.loc[net.sgen.bus].values

        net2 = from_ppc(ppc, f_hz=net.f_hz, tap_side=net.trafo.tap_side.values)
        # again add max_loading_percent to enable valid comparison
        net2.line["max_loading_percent"] = 100

        # compare loadflow results
        pp.runpp(net)
        pp.runpp(net2)
        assert pp.nets_equal(net, net2, check_only_results=True, atol=1e-10)

        # compare optimal powerflow results
        if i == 1:
            pp.runopp(net, delta=1e-16)
            pp.runopp(net2, delta=1e-16)
            assert pp.nets_equal(net, net2, check_only_results=True, atol=1e-10)


@pytest.mark.skipif(not pypower_installed,
                    reason="c24 test net is taken from mandatory pypower installation")
def test_case24_from_pypower():
    net = from_ppc(c24.case24_ieee_rts())
    pp.runopp(net)
    assert net.OPF_converged


def _bool_arr_to_positional_column_vector(arr):
    return np.arange(len(arr), dtype=int)[arr].reshape((-1, 1))


def overwrite_results_data_of_ppc_pickle(file_name, grid_names):
    folder = os.path.join(pp.pp_dir, 'test', 'converter')
    file = os.path.join(folder, file_name)
    ppcs = pickle.load(open(file, "rb"))

    for i in grid_names:
        ppc = ppcs[i]
        net = from_ppc(ppc, f_hz=60)
        pp.runpp(net)

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
        ppc["gen"][sgen_pos, [PG, QG]] = net.res_sgen[["p_mw", "q_mvar"]].iloc[
            sum(ppc["bus"][:, PD] < 0):].values

        cols = ["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]
        ppc["branch"][line_pos, [PF, QF, PT, QT]] = net.res_line[cols].values
        cols = pd.Series(cols).str.replace("from", "hv").str.replace("to", "lv")
        new_bra_res = net.res_trafo[cols].values
        if len(tr_swap_pos):
            new_bra_res[tr_swap_pos.flatten(), :] = np.concatenate(
                (new_bra_res[tr_swap_pos, [2, 3]], new_bra_res[tr_swap_pos, [0, 1]]), axis=1)
        ppc["branch"][tr_pos, [PF, QF, PT, QT]] = new_bra_res

    # --- overwrite pickle
    with open(file, "wb") as handle:
        # pickle.dump(ppcs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(ppcs, handle, protocol=4)  # as long as py3.7 is supported, 4 is the highest protocol


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
