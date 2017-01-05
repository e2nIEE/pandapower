# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import pytest
from pypower import case9, case9Q

import pandapower as pp
import pandapower.test as pt
from pandapower.converter import ppc2pp, validate_ppc2pp
import pplog

logger = pplog.getLogger(__name__)

max_diff_values1 = {"vm_pu": 1e-6, "va_degree": 1e-5, "p_branch_kw": 1e-3, "q_branch_kvar": 1e-3,
                    "p_gen_kw": 1e-3, "q_gen_kvar": 1e-3}


def test_ppc2pp():
    ppc_net = pt.case2_2()
    net_by_ppc_net = ppc2pp(ppc_net)
    net_by_code = pt.case2_2_by_code()
    pp.runpp(net_by_ppc_net, trafo_model="pi")
    pp.runpp(net_by_code, trafo_model="pi")

    assert type(net_by_ppc_net) == type(net_by_code)
    assert net_by_ppc_net.converged
    assert len(net_by_ppc_net.bus) == len(net_by_code.bus)
    assert len(net_by_ppc_net.trafo) == len(net_by_code.trafo)
    assert len(net_by_ppc_net.ext_grid) == len(net_by_code.ext_grid)
    assert pp.equal_nets(net_by_ppc_net, net_by_code, check_only_results=True, tol=1.e-9)

    # check detect_trafo
    ppc_net2_4 = pt.case2_4()
    net1 = ppc2pp(ppc_net2_4, detect_trafo='vn_kv')
    net2 = ppc2pp(ppc_net2_4, detect_trafo='ratio')
    assert type(net1) == type(net_by_code)
    assert type(net2) == type(net_by_code)
    assert len(net1.trafo) == 1
    assert len(net1.line) == 0
    assert len(net2.trafo) == 0
    assert len(net2.line) == 1


def test_validate_ppc2pp():
    ppc_net = pt.case2_2()
    pp_net = pt.case2_2_by_code()
    assert validate_ppc2pp(ppc_net, pp_net, max_diff_values=max_diff_values1,
                           detect_trafo='vn_kv')


def test_cases():
    # check ppc_testfile
    name = ['case2_1', 'case2_2', 'case2_3', 'case2_4', 'case3_1', 'case3_2', 'case6', 'case39',
            'case57', 'case118']
    module = __import__('pandapower')
    sub_module = getattr(module, 'test')
    for i in name:
        my_function = getattr(sub_module, i)
        ppc_net = my_function()
        pp_net = ppc2pp(ppc_net, f_hz=60)
        assert validate_ppc2pp(ppc_net, pp_net, max_diff_values=max_diff_values1)
        logger.info('%s has been checked successfully.' % i)
    # check pypower cases
    name = ['case4gs', 'case6ww', 'case30', 'case30pwl', 'case30Q']
    for i in name:
        module = __import__('pypower.' + i)
        submodule = getattr(module, i)
        my_function = getattr(submodule, i)
        ppc_net = my_function()
        pp_net = ppc2pp(ppc_net, f_hz=60)
        assert validate_ppc2pp(ppc_net, pp_net, max_diff_values=max_diff_values1)
        logger.info('%s has been checked successfully.' % i)
    # --- Because there is a pypower power flow failure in generator results in case9 (which is not
    # in matpower) another max_diff_values must be used to receive an successful validation
    max_diff_values2 = {"vm_pu": 1e-6, "va_degree": 1e-5, "p_branch_kw": 1e-3,
                        "q_branch_kvar": 1e-3, "p_gen_kw": 1e3, "q_gen_kvar": 1e3}
    ppc_net = case9.case9()
    pp_net = ppc2pp(ppc_net, f_hz=60)
    assert validate_ppc2pp(ppc_net, pp_net, max_diff_values=max_diff_values2)
    logger.info('case9 has been checked successfully.')
    ppc_net = case9Q.case9Q()
    pp_net = ppc2pp(ppc_net, f_hz=60)
    assert validate_ppc2pp(ppc_net, pp_net, max_diff_values=max_diff_values2)
    logger.info('case9Q has been checked successfully.')


if __name__ == '__main__':
    pytest.main(["test_ppc2pp.py", "-xs"])
