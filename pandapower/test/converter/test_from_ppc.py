# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import pytest
from pypower import case9, case9Q

import pandapower as pp
import pandapower.test.converter.ppc_testgrids as testgrids
from pandapower.converter import from_ppc, validate_from_ppc
try:
    import pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)
max_diff_values1 = {"vm_pu": 1e-6, "va_degree": 1e-5, "p_branch_kw": 1e-3, "q_branch_kvar": 1e-3,
                    "p_gen_kw": 1e-3, "q_gen_kvar": 1e-3}


def test_from_ppc():
    ppc = testgrids.case2_2()
    net_by_ppc = from_ppc(ppc)
    net_by_code = testgrids.case2_2_by_code()
    pp.runpp(net_by_ppc, trafo_model="pi")
    pp.runpp(net_by_code, trafo_model="pi")

    assert type(net_by_ppc) == type(net_by_code)
    assert net_by_ppc.converged
    assert len(net_by_ppc.bus) == len(net_by_code.bus)
    assert len(net_by_ppc.trafo) == len(net_by_code.trafo)
    assert len(net_by_ppc.ext_grid) == len(net_by_code.ext_grid)
    assert pp.equal_nets(net_by_ppc, net_by_code, check_only_results=True, tol=1.e-9)

    # check detect_trafo
    ppc2_4 = testgrids.case2_4()
    net1 = from_ppc(ppc2_4, detect_trafo='vn_kv')
    net2 = from_ppc(ppc2_4, detect_trafo='ratio')
    assert type(net1) == type(net_by_code)
    assert type(net2) == type(net_by_code)
    assert len(net1.trafo) == 1
    assert len(net1.line) == 0
    assert len(net2.trafo) == 0
    assert len(net2.line) == 1


def test_validate_from_ppc():
    ppc = testgrids.case2_2()
    net = testgrids.case2_2_by_code()
    assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values1,
                             detect_trafo='vn_kv')


def test_ppc_testgrids():
    # check ppc_testgrids
    name = ['case2_1', 'case2_2', 'case2_3', 'case2_4', 'case3_1', 'case3_2', 'case6', 'case14',
            'case57', 'case118']
    for i in name:
        my_function = getattr(testgrids, i)
        ppc = my_function()
        net = from_ppc(ppc, f_hz=60)
        assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values1)
        logger.debug('%s has been checked successfully.' % i)


def test_pypower_cases():
    # check pypower cases
    name = ['case4gs', 'case6ww', 'case30', 'case30pwl', 'case30Q', 'case39']
    for i in name:
        module = __import__('pypower.' + i)
        submodule = getattr(module, i)
        my_function = getattr(submodule, i)
        ppc = my_function()
        if i == 'case39':
            net = from_ppc(ppc, f_hz=60, detect_trafo='ratio')
            assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values1,
                                     detect_trafo='ratio')
        else:
            net = from_ppc(ppc, f_hz=60)
            assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values1)
        logger.debug('%s has been checked successfully.' % i)
    # --- Because there is a pypower power flow failure in generator results in case9 (which is not
    # in matpower) another max_diff_values must be used to receive an successful validation
    max_diff_values2 = {"vm_pu": 1e-6, "va_degree": 1e-5, "p_branch_kw": 1e-3,
                        "q_branch_kvar": 1e-3, "p_gen_kw": 1e3, "q_gen_kvar": 1e3}
    ppc = case9.case9()
    net = from_ppc(ppc, f_hz=60)
    assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values2)
    logger.debug('case9 has been checked successfully.')
    ppc = case9Q.case9Q()
    net = from_ppc(ppc, f_hz=60)
    assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values2)
    logger.debug('case9Q has been checked successfully.')


if __name__ == '__main__':
    pytest.main(["-xs"])
