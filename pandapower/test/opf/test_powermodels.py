# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:59:36 2018

@author: thurner
"""
import pytest
import pandapower as pp
from pandapower.test.consistency_checks import consistency_checks
import numpy as np

try:
    from julia import Main
    julia_installed = True
except ImportError:
    julia_installed = False

@pytest.fixture
def net_3w_trafo_opf():

    net = pp.create_empty_network()

    #create buses
    bus1 = pp.create_bus(net, vn_kv=220.)
    bus2 = pp.create_bus(net, vn_kv=110.)
    bus3 = pp.create_bus(net, vn_kv=110.)
    bus4 = pp.create_bus(net, vn_kv=110.)
    bus5 = pp.create_bus(net, vn_kv=110.)

    pp.create_bus(net, vn_kv=110., in_service=False)

    #create 220/110 kV transformer
    pp.create_transformer3w_from_parameters(net, bus1, bus2, bus5, vn_hv_kv=220, vn_mv_kv=110,
                                            vn_lv_kv=110, vsc_hv_percent=10., vsc_mv_percent=10.,
                                            vsc_lv_percent=10., vscr_hv_percent=0.5,
                                            vscr_mv_percent=0.5, vscr_lv_percent=0.5, pfe_kw=100.,
                                            i0_percent=0.1, shift_mv_degree=0, shift_lv_degree=0,
                                            sn_hv_kva=100e3, sn_mv_kva=50e3, sn_lv_kva=50e3)

    #create 110 kV lines
    pp.create_line(net, bus2, bus3, length_km=70., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus3, bus4, length_km=50., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus4, bus2, length_km=40., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus4, bus5, length_km=30., std_type='149-AL1/24-ST1A 110.0')

    #create loads
    pp.create_load(net, bus2, p_kw=60e3, controllable = False)
    pp.create_load(net, bus3, p_kw=70e3, controllable = False)
    pp.create_sgen(net, bus3, p_kw=10e3, controllable=False)

    #create generators
    pp.create_ext_grid(net, bus1, min_p_kw=0, max_p_kw=1e6, max_q_kvar=10, min_q_kvar=0)
    pp.create_gen(net, bus3, p_kw=80*1e3, min_p_kw=0, max_p_kw=80e3, vm_pu=1.01)
    pp.create_gen(net, bus4, p_kw=80*1e3, min_p_kw=0, max_p_kw=80e3, vm_pu=1.01)
    net.gen["controllable"] = False
    return net

@pytest.mark.skipif(julia_installed==False, reason="requires julia installation")
def test_compare_pwl_and_poly(net_3w_trafo_opf):
    net = net_3w_trafo_opf
    pp.create_pwl_cost(net, 0, 'ext_grid', [(0, 0), (1e6, 1e6)])
    pp.create_pwl_cost(net, 0, 'gen', [(0, 0), (80e3, 240e3)])
    pp.create_pwl_cost(net, 1, 'gen', [(0, 0), (100e3, 200e3)])

    pp.runpm(net)
    consistency_checks(net)

    p_gen = net.res_gen.p_kw.values
    q_gen = net.res_gen.q_kvar.values
    vm_bus = net.res_bus.vm_pu.values
    va_bus = net.res_bus.va_degree.values

    net.pwl_cost.drop(net.pwl_cost.index, inplace=True)

    pp.create_poly_cost(net, 0, 'ext_grid', cp1_eur_per_kw=1.)
    pp.create_poly_cost(net, 0, 'gen', cp1_eur_per_kw=3.)
    pp.create_poly_cost(net, 1, 'gen', cp1_eur_per_kw=2.)

    pp.runpm(net)
    consistency_checks(net)

    np.allclose(p_gen, net.res_gen.p_kw.values)
    np.allclose(q_gen, net.res_gen.q_kvar.values)
    np.allclose(vm_bus, net.res_bus.vm_pu.values)
    np.allclose(va_bus, net.res_bus.va_degree.values)


@pytest.mark.skipif(julia_installed==False, reason="requires julia installation")
def test_pwl():
    net = pp.create_empty_network()

    #create buses
    bus1 = pp.create_bus(net, vn_kv=110., min_vm_pu=0.9, max_vm_pu=1.1)
    bus2 = pp.create_bus(net, vn_kv=110., min_vm_pu=0.9, max_vm_pu=1.1)
    bus3 = pp.create_bus(net, vn_kv=110., min_vm_pu=0.9, max_vm_pu=1.1)

    #create 110 kV lines
    pp.create_line(net, bus1, bus2, length_km=50., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus2, bus3, length_km=50., std_type='149-AL1/24-ST1A 110.0')

    #create loads
    pp.create_load(net, bus2, p_kw=80e3, controllable = False)

    #create generators
    g1 = pp.create_gen(net, bus1, p_kw=80*1e3, min_p_kw=0, max_p_kw=80e3, vm_pu=1.01, slack=True)
    g2 = pp.create_gen(net, bus3, p_kw=80*1e3, min_p_kw=0, max_p_kw=80e3, vm_pu=1.01)
    net.gen["controllable"] = False

    pp.create_pwl_cost(net, g1, 'gen', [(0, 0), (20, 40), (80, 340)])
    pp.create_pwl_cost(net, g2, 'gen', [(0, 0), (20, 40), (80, 340)])

    pp.runpm(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_kw.iloc[0], net.res_gen.p_kw.iloc[1])
    assert np.isclose(net.res_gen.q_kvar.iloc[0], net.res_gen.q_kvar.iloc[1])

    net.pwl_cost.drop(net.pwl_cost.index, inplace=True)
    g3 = pp.create_gen(net, bus1, p_kw=80*1e3, min_p_kw=0, max_p_kw=80e3, vm_pu=1.01, slack=True)


    pp.create_pwl_cost(net, g1, 'gen', [(0, 0), (0.2, 0.2), (0.4, 1.8)])
    pp.create_pwl_cost(net, g2, 'gen', [(0, 0), (0.3, 0.6), (0.4, 2)])
    pp.create_pwl_cost(net, g3, 'gen', [(0, 0), (0.1, 0.3)])

    net.load.p_kw = 0.1e3
    pp.runpm(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_kw.at[g2], 0)
    assert np.isclose(net.res_gen.p_kw.at[g3], 0)
    assert np.isclose(net.res_cost, net.res_gen.p_kw.at[g1]*1e-3)

    net.load.p_kw = 0.3e3
    pp.runpm(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_kw.at[g3], 0)
    assert np.isclose(net.res_gen.p_kw.at[g1], 200)
    assert np.isclose(net.res_cost, net.res_gen.p_kw.at[g1]*1e-3 + net.res_gen.p_kw.at[g2]*2e-3)

    net.load.p_kw = 0.5e3
    pp.runpm(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_kw.at[g1], 200)
    assert np.isclose(net.res_gen.p_kw.at[g2], 300)
    assert np.isclose(net.res_cost, net.res_gen.p_kw.at[g1]*1e-3 + net.res_gen.p_kw.at[g2]*2e-3 + \
                                    net.res_gen.p_kw.at[g3]*3e-3)

if __name__ == '__main__':
    pytest.main([__file__])