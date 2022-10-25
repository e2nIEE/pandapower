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

    b1 = pp.create_bus(net, name="B1", vn_kv=18)
    b2 = pp.create_bus(net, name="B2", vn_kv=16.5)
    b3 = pp.create_bus(net, name="B3", vn_kv=230)
    b4 = pp.create_bus(net, name="B4", vn_kv=230)
    b5 =pp.create_bus(net, name="B5", vn_kv=230)
    b6 =pp.create_bus(net, name="B6", vn_kv=230)
    b7 =pp.create_bus(net, name="B7", vn_kv=230)
    b8 =pp.create_bus(net, name="B8", vn_kv=230)

    pp.create_ext_grid(net,bus=b1,vm_pu=1,va_degree=0)

    pp.create_line_from_parameters(net,name="L1",from_bus=b3,to_bus=b4,length_km=30, r_ohm_per_km=0.049, x_ohm_per_km=0.136,g_us_per_km=0,c_nf_per_km=142,max_i_ka=1)
    pp.create_line_from_parameters(net,name="L2",from_bus=b3,to_bus=b4,length_km=30, r_ohm_per_km=0.049, x_ohm_per_km=0.136,g_us_per_km=0,c_nf_per_km=142,max_i_ka=1)
    pp.create_line_from_parameters(net,name="L3",from_bus=b4,to_bus=b5,length_km=100, r_ohm_per_km=0.081, x_ohm_per_km=0.312,g_us_per_km=0,c_nf_per_km=11,max_i_ka=1)
    pp.create_line_from_parameters(net,name="L4",from_bus=b4,to_bus=b6,length_km=100, r_ohm_per_km=0.081, x_ohm_per_km=0.312,g_us_per_km=0,c_nf_per_km=11,max_i_ka=1)
    pp.create_line_from_parameters(net,name="L5",from_bus=b5,to_bus=b7,length_km=220, r_ohm_per_km=0.081, x_ohm_per_km=0.312,g_us_per_km=0,c_nf_per_km=11,max_i_ka=1)
    pp.create_line_from_parameters(net,name="L6",from_bus=b6,to_bus=b8,length_km=140, r_ohm_per_km=0.081, x_ohm_per_km=0.312,g_us_per_km=0,c_nf_per_km=11,max_i_ka=1)
    pp.create_line_from_parameters(net,name="L7",from_bus=b5,to_bus=b6,length_km=180, r_ohm_per_km=0.081, x_ohm_per_km=0.312,g_us_per_km=0,c_nf_per_km=11,max_i_ka=1)
    pp.create_line_from_parameters(net,name="L8",from_bus=b7,to_bus=b8,length_km=180, r_ohm_per_km=0.081, x_ohm_per_km=0.312,g_us_per_km=0,c_nf_per_km=11,max_i_ka=1)

   # pp.create_line_from_parameters(net,name="L9",from_bus=3,to_bus=4,length_km=100, r_ohm_per_km=0.312, x_ohm_per_km=0.312,g_us_per_km=0,c_nf_per_km=11)

    pp.create_transformer_from_parameters(net, name="trafo1",hv_bus=b8,lv_bus=b1,sn_mva=192,vn_hv_kv=230,vn_lv_kv=18,vkr_percent=0,vector_group="Yy0",pfe_kw=0,vk_percent=12,i0_percent=0)
    pp.create_transformer_from_parameters(net, name="trafo2",hv_bus=b3,lv_bus=b2,sn_mva=500,vn_hv_kv=230,vn_lv_kv=16.5,vkr_percent=0,vector_group="Yy0",pfe_kw=0,vk_percent=16,i0_percent=0)

    pp.create_gen(net,bus=b2,p_mw=500,vm_pu=1)
    # pp.create_sgen(net,bus = 2, p_mw=500,name="WT")
    #
    pp.create_load(net,bus=b4,p_mw=130,q_mvar=50)
    pp.create_load(net,bus=b5,p_mw=120,q_mvar=50)
    pp.create_load(net,bus=b6,p_mw=80,q_mvar=25)
    pp.create_load(net,bus=b7,p_mw=50,q_mvar=25)

    # pp.create_load(net,bus=4,p_mw=0,q_mvar=0)
    # pp.create_load(net,bus=5,p_mw=0,q_mvar=0)
    # pp.create_load(net,bus=6,p_mw=0,q_mvar=0)
    # pp.create_load(net,bus=7,p_mw=0,q_mvar=0)

    return net
import matplotlib.pyplot

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
