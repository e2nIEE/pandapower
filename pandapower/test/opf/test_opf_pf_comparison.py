# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:48:47 2020

@author: jwiemer
"""

import pytest
import numpy as np
import pandapower as pp

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def SimpleGrid():
    # Create a Simple Example Network with 50 Hz
    net = pp.create_empty_network(f_hz=50.0)

    b1 = pp.create_bus(net, vn_kv=110, name='b1_hv', type='n')
    b2 = pp.create_bus(net, vn_kv=110, name='b10', type='n')

    pp.create_line(net, b1, b2, length_km=1, std_type='243-AL1/39-ST1A 110.0', name='l3')

    # Chose the load to match nominal current
    p_ac = 110 * 0.645 * np.sqrt(3)
    pp.create_load(net, b2, sn_mva=p_ac, q_mvar=5, p_mw=p_ac, name="load_b8", const_i_percent=0)

    pp.create_sgen(net, b2, sn_mva=p_ac*1.05, q_mvar=50, p_mw=p_ac, name="sgen_b8")

    pp.create_ext_grid(net, 0, vm_pu=1, va_degree=0, s_sc_max_mva=20 * 110 * np.sqrt(3), rx_max=0.1)
    return net


def test_OPF_PF_Comparison():
    net = SimpleGrid()

    # OPF usage
    net.line['max_loading_percent'] = 100

    # Load
    net.load["controllable"] = False

    # Sgen
    net.sgen["controllable"] = True

    net.sgen['max_p_mw'] = net.sgen.p_mw
    net.sgen['min_p_mw'] = 0

    net.sgen['max_q_mvar'] = 50
    net.sgen['min_q_mvar'] = net.sgen.q_mvar

    # Bus
    net.bus['max_vm_pu'] = 1.05
    net.bus['min_vm_pu'] = 0.95

    # Maximizing Generation
    for i in net.sgen.index:
        pp.create_poly_cost(net, i, 'sgen', cp1_eur_per_mw=-1)

    pp.runopp(net)
    loading_result_opf = net.res_line.loading_percent

    # Transfer OPF Solution
    net.sgen.p_mw = net.res_sgen.p_mw
    net.sgen.q_mvar = net.res_sgen.q_mvar

    net.load.p_mw = net.res_load.p_mw
    net.load.q_mvar = net.res_load.q_mvar

    # Calculate Powerflow in order to compare the results
    pp.runpp(net)
    loading_result_pf = net.res_line.loading_percent

    assert np.isclose(loading_result_pf, loading_result_opf)


if __name__ == '__main__':
    pytest.main(['-xs', __file__])
