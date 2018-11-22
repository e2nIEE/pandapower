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
    pp.create_pwl_cost(net, 0, 'ext_grid', [(0, 1e6, 1)])
    pp.create_pwl_cost(net, 0, 'gen', [(0, 30e3, 3), (30e3, 80e3, 3)])
    pp.create_pwl_cost(net, 1, 'gen', [(0, 100e3, 2)])

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
#    net.gen["controllable"] = False

    pp.create_pwl_cost(net, g1, 'gen', [(0, 2e3, 2), (2e3, 80e3, 5)])
    pp.create_pwl_cost(net, g2, 'gen', [(0, 2e3, 2), (2e3, 80e3, 5)])

#    pp.runpm(net)
#    consistency_checks(net, rtol=1e-3)
#    assert np.isclose(net.res_gen.p_kw.iloc[0], net.res_gen.p_kw.iloc[1])
#    assert np.isclose(net.res_gen.q_kvar.iloc[0], net.res_gen.q_kvar.iloc[1])

    net.pwl_cost.drop(net.pwl_cost.index, inplace=True)
    g3 = pp.create_gen(net, bus1, p_kw=80*1e3, min_p_kw=0, max_p_kw=80e3, vm_pu=1.01)


    pp.create_pwl_cost(net, g1, 'gen', [(0, 2e3, 1.), (2e3, 803, 8.)])
    pp.create_pwl_cost(net, g2, 'gen', [(0, 3e3, 2.), (3e3, 803, 14)])
    pp.create_pwl_cost(net, g3, 'gen', [(0, 1e3, 3.), (1e3, 803, 10.)])

    net.load.p_kw = 1e3
    pp.runpm(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_kw.at[g2], 0)
    assert np.isclose(net.res_gen.p_kw.at[g3], 0)
    assert np.isclose(net.res_cost, net.res_gen.p_kw.at[g1])

    net.load.p_kw = 3e3
    pp.runpm(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_kw.at[g3], 0)
    assert np.isclose(net.res_gen.p_kw.at[g1], 2e3)
    assert np.isclose(net.res_cost, net.res_gen.p_kw.at[g1] + net.res_gen.p_kw.at[g2]*2)

    net.load.p_kw = 5e3
    pp.runpm(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_kw.at[g1], 2e3)
    assert np.isclose(net.res_gen.p_kw.at[g2], 3e3)
    assert np.isclose(net.res_cost, net.res_gen.p_kw.at[g1] + net.res_gen.p_kw.at[g2]*2 + \
                                    net.res_gen.p_kw.at[g3]*3)

@pytest.mark.skipif(julia_installed==False, reason="requires julia installation")
def test_without_ext_grid():
    net = pp.create_empty_network()

    #create buses
    bus1 = pp.create_bus(net, vn_kv=220., geodata=(5,9))
    bus2 = pp.create_bus(net, vn_kv=110., geodata=(6,10))
    bus3 = pp.create_bus(net, vn_kv=110., geodata=(10,9))
    bus4 = pp.create_bus(net, vn_kv=110., geodata=(8,8))
    bus5 = pp.create_bus(net, vn_kv=110., geodata=(6,8))

    #create 220/110/110 kV 3W-transformer
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
    pp.create_load(net, bus2, p_kw=60e3)
    pp.create_load(net, bus3, p_kw=70e3)
    pp.create_load(net, bus4, p_kw=10e3)

    #create generators
    g1 = pp.create_gen(net, bus1, p_kw=40e3, min_p_kw=0, max_p_kw=200e3, slack=True)
    pp.create_poly_cost(net, g1, 'gen', cp1_eur_per_kw=1)

    g2 = pp.create_gen(net, bus3, p_kw=40*1e3, min_p_kw=0, max_p_kw=200e3, vm_pu=1.01)
    pp.create_poly_cost(net, g2, 'gen', cp1_eur_per_kw=2)

    g3 = pp.create_gen(net, bus4, p_kw=50*1e3, min_p_kw=0, max_p_kw=200e3, vm_pu=1.01)
    pp.create_poly_cost(net, g3, 'gen', cp1_eur_per_kw=3)

    pp.runpm(net)

    assert net.res_gen.p_kw.iloc[g1]
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_kw.at[g2], 0)
    assert np.isclose(net.res_gen.p_kw.at[g3], 0)
    assert np.isclose(net.res_cost, net.res_gen.p_kw.at[g1])

    net.trafo3w["max_loading_percent"] = 50
    pp.runpm(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_trafo3w.loading_percent.values[0], 50)
    assert np.isclose(net.res_cost, net.res_gen.p_kw.at[g1] + 2*net.res_gen.p_kw.at[g2])



if __name__ == '__main__':
    pytest.main([__file__])