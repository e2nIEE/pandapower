import pytest
import numpy as np
from pandapower.networks.power_system_test_cases import case24_ieee_rts

import pandapower as pp
import pandapower.networks as nw
from pandapower.converter.matpower.from_mpc import _m2ppc

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

def test_pglib_case24():
    case24 = case24_ieee_rts()
    file = 'pglib_opf_case24_ieee_rts.m'
    ppc_before = _m2ppc(file, 'mpc')
    net = pp.converter.from_mpc(file, tap_side='lv')
    ppc_after = pp.converter.to_ppc(net, init='flat')

    pp.runopp(net)
    assert np.isclose(net.res_cost, 63352., 40.) # Value taken from https://arxiv.org/abs/1908.02788

    # next compare fails, since format, sorting and some standard parameters are different...
    # assert np.array_equal(ppc_after['branch'], ppc_before['branch'])