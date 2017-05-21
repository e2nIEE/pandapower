# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import pytest
import os
import pickle

import pandapower as pp
from pandapower.converter import from_ppc, validate_from_ppc

try:
    import pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)
max_diff_values1 = {"vm_pu": 1e-6, "va_degree": 1e-5, "p_branch_kw": 1e-3, "q_branch_kvar": 1e-3,
                    "p_gen_kw": 1e-3, "q_gen_kvar": 1e-3}


def get_testgrids(name, filename):
    """
    This function return the ppc (or pp net) which is saved in ppc_testgrids.p to validate the
    from_ppc function via validate_from_ppc.
    """
    pp_path = pp_path = os.path.split(pp.__file__)[0]
    folder = os.path.join(pp_path, 'test', 'converter')
    if name == "case2_2_by_code":
        return pp.from_pickle(os.path.join(folder, name + ".p"))
    else:
        ppcs = pickle.load(open(os.path.join(folder, filename), "rb"))
        return ppcs[name]


def test_from_ppc():
    ppc = get_testgrids('case2_2', 'ppc_testgrids.p')
    net_by_ppc = from_ppc(ppc)
    net_by_code = get_testgrids('case2_2_by_code', 'ppc_testgrids.p')
    pp.runpp(net_by_ppc, trafo_model="pi")
    pp.runpp(net_by_code, trafo_model="pi")

    assert type(net_by_ppc) == type(net_by_code)
    assert net_by_ppc.converged
    assert len(net_by_ppc.bus) == len(net_by_code.bus)
    assert len(net_by_ppc.trafo) == len(net_by_code.trafo)
    assert len(net_by_ppc.ext_grid) == len(net_by_code.ext_grid)
    assert pp.nets_equal(net_by_ppc, net_by_code, check_only_results=True, tol=1.e-9)


def test_validate_from_ppc():
    ppc = get_testgrids('case2_2', 'ppc_testgrids.p')
    net = get_testgrids('case2_2_by_code', 'ppc_testgrids.p')
    assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values1)


def test_ppc_testgrids():
    # check ppc_testgrids
    name = ['case2_1', 'case2_2', 'case2_3', 'case2_4', 'case3_1', 'case3_2', 'case6', 'case14',
            'case57']
    for i in name:
        ppc = get_testgrids(i, 'ppc_testgrids.p')
        net = from_ppc(ppc, f_hz=60)
        assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values1)
        logger.debug('%s has been checked successfully.' % i)


def test_pypower_cases():
    # check pypower cases
    name = ['case4gs', 'case6ww', 'case24_ieee_rts', 'case30', 'case39',
            'case118', 'case300']
    for i in name:
        ppc = get_testgrids(i, 'pypower_cases.p')
        net = from_ppc(ppc, f_hz=60)
        assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values1)
        logger.debug('%s has been checked successfully.' % i)
    # --- Because there is a pypower power flow failure in generator results in case9 (which is not
    # in matpower) another max_diff_values must be used to receive an successful validation
    max_diff_values2 = {"vm_pu": 1e-6, "va_degree": 1e-5, "p_branch_kw": 1e-3,
                        "q_branch_kvar": 1e-3, "p_gen_kw": 1e3, "q_gen_kvar": 1e3}
    ppc = get_testgrids('case9', 'pypower_cases.p')
    net = from_ppc(ppc, f_hz=60)
    assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values2)
    logger.debug('case9 has been checked successfully.')


if __name__ == '__main__':
    pytest.main(["-s"])