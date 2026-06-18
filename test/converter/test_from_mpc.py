# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import os
import logging

import pytest

from pandapower.converter.matpower import from_mpc
from pandapower.networks import case24_ieee_rts
from pandapower.run import runpp, set_user_pf_options
from pandapower.toolbox.comparison import nets_equal
from pandapower.results import reset_results

try:
    import matpowercaseframes

    matpowercaseframes_imported = True
except ImportError:
    matpowercaseframes_imported = False

from test import test_path

logger = logging.getLogger(__name__)


def test_from_mpc_mat():
    case24 = case24_ieee_rts()
    set_user_pf_options(case24)
    this_folder = os.path.join(test_path, "converter")
    mat_case = os.path.join(this_folder, 'case24_ieee_rts.mat')
    case24_from_mpc = from_mpc(mat_case, f_hz=60, casename_mpc_file='mpc', tap_side="hv")
    # TODO: remove after https://github.com/e2nIEE/pandapower/pull/2813:
    #  reset 3ph results (new columns in case24 but not in from_mpc, would be solved by 3ph powerflow, so not relevant)
    reset_results(case24, "pf_3ph")
    reset_results(case24_from_mpc, "pf_3ph")

    runpp(case24)
    runpp(case24_from_mpc)

    assert case24_from_mpc.converged
    assert nets_equal(case24, case24_from_mpc, check_only_results=True)


@pytest.mark.skipif(not matpowercaseframes_imported,
                    reason="matpowercaseframes is needed to convert .m files.")
def test_from_mpc_m():
    this_folder = os.path.join(test_path, "converter")
    mat_case = os.path.join(this_folder, 'case24_ieee_rts.mat')
    m_case = os.path.join(this_folder, 'case24_ieee_rts.m')
    case24_mat = from_mpc(mat_case, f_hz=60, casename_mpc_file='mpc', tap_side="hv")
    case24_m = from_mpc(m_case, f_hz=60, tap_side="hv")

    runpp(case24_mat)
    runpp(case24_m)

    assert case24_m.converged
    assert nets_equal(case24_mat, case24_m)


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
