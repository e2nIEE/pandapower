# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os

import pytest

import pandapower as pp
import pandapower.networks as pn
import pandapower.toolbox
from pandapower.converter import from_mpc

try:
    import matpowercaseframes
    matpowercaseframes_imported = True
except ImportError:
    matpowercaseframes_imported = False

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def test_from_mpc_mat():
    case24 = pn.case24_ieee_rts()
    pp.set_user_pf_options(case24)
    this_folder = os.path.join(pp.pp_dir, "test", "converter")
    mat_case = os.path.join(this_folder, 'case24_ieee_rts.mat')
    case24_from_mpc = from_mpc(mat_case, f_hz=60, casename_mpc_file='mpc', tap_side="hv")

    pp.runpp(case24)
    pp.runpp(case24_from_mpc)

    assert case24_from_mpc.converged
    assert pandapower.toolbox.nets_equal(case24, case24_from_mpc, check_only_results=True)


@pytest.mark.skipif(not matpowercaseframes_imported,
                    reason="matpowercaseframes is needed to convert .m files.")
def test_from_mpc_m():
    this_folder = os.path.join(pp.pp_dir, "test", "converter")
    mat_case = os.path.join(this_folder, 'case24_ieee_rts.mat')
    m_case = os.path.join(this_folder, 'case24_ieee_rts.m')
    case24_mat = from_mpc(mat_case, f_hz=60, casename_mpc_file='mpc', tap_side="hv")
    case24_m = from_mpc(m_case, f_hz=60, tap_side="hv")

    pp.runpp(case24_mat)
    pp.runpp(case24_m)

    assert case24_m.converged
    assert pandapower.toolbox.nets_equal(case24_mat, case24_m)


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
