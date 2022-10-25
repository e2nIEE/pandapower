# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest
import numpy as np
import pandapower as pp

import pandapower.networks
from pandapower.pypower.idx_bus import BS, SVC_FIRING_ANGLE


def facts_case_study_grid():
    net = pp.create_empty_network()
    return net


@pytest.mark.parametrize("vm_set_pu", [0.96, 1., 1.04])
def test_svc(vm_set_pu):
    net = pp.networks.case9()
    net3 = net.deepcopy()
    lidx = pp.create_load(net3, 3, 0, 0)
    pp.create_shunt(net, 3, 0, 0, 345)
    net2 = net.deepcopy()
    net.shunt["controllable"] = True
    net.shunt["set_vm_pu"] = vm_set_pu
    net.shunt["thyristor_firing_angle_degree"] = 90.
    net.shunt["svc_x_l_ohm"] = 1
    net.shunt["svc_x_cvar_ohm"] = -10
    pp.runpp(net)
    assert 90 <= net.shunt.at[0, "thyristor_firing_angle_degree"] <= 180
    assert np.isclose(net.res_bus.at[3, 'vm_pu'], net.shunt.at[0, 'set_vm_pu'], rtol=0, atol=1e-6)

    net3.load.loc[lidx, "q_mvar"] = net.res_shunt.q_mvar.at[0]
    pp.runpp(net3)

    net2.shunt.q_mvar.at[0] = -net._ppc["bus"][net._pd2ppc_lookups["bus"][net.shunt.bus.values], BS]
    pp.runpp(net2)
    assert np.isclose(net2.res_bus.at[3, 'vm_pu'], net.shunt.at[0, 'set_vm_pu'], rtol=0, atol=1e-6)
    assert np.isclose(net2.res_bus.at[3, 'q_mvar'], net.res_bus.at[3, 'q_mvar'], rtol=0, atol=1e-6)
    assert np.isclose(net2.res_shunt.at[0, 'vm_pu'], net.res_shunt.at[0, 'vm_pu'], rtol=0, atol=1e-6)
    assert np.isclose(net2.res_shunt.at[0, 'q_mvar'], net.res_shunt.at[0, 'q_mvar'], rtol=0, atol=1e-6)

    pp.runpp(net)
    assert np.allclose(net.shunt.q_mvar, -net._ppc["bus"][net._pd2ppc_lookups["bus"][net.shunt.bus.values], BS],
                       rtol=0, atol=1e-6)
    assert np.allclose(np.deg2rad(net.shunt.thyristor_firing_angle_degree),
                       net._ppc["bus"][net._pd2ppc_lookups["bus"][net.shunt.bus.values], SVC_FIRING_ANGLE],
                       rtol=0, atol=1e-6)

    net.shunt.controllable = False
    pp.runpp(net)
    assert np.isclose(net.res_bus.at[3, 'vm_pu'], net.shunt.at[0, 'set_vm_pu'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[3, 'q_mvar'], net2.res_bus.at[3, 'q_mvar'], rtol=0, atol=1e-5)
    assert np.isclose(net.res_shunt.at[0, 'vm_pu'], net2.res_shunt.at[0, 'vm_pu'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_shunt.at[0, 'q_mvar'], net2.res_shunt.at[0, 'q_mvar'], rtol=0, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
