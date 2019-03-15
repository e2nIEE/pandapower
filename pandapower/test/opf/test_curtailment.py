# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest
from numpy import array, allclose

import pandapower as pp

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def test_minimize_active_power_curtailment():
    net = pp.create_empty_network()

    # create buses
    bus1 = pp.create_bus(net, vn_kv=220.)
    bus2 = pp.create_bus(net, vn_kv=110.)
    bus3 = pp.create_bus(net, vn_kv=110.)
    bus4 = pp.create_bus(net, vn_kv=110.)

    # create 220/110 kV transformer
    pp.create_transformer(net, bus1, bus2, std_type="100 MVA 220/110 kV")

    # create 110 kV lines
    pp.create_line(net, bus2, bus3, length_km=70., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus3, bus4, length_km=50., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus4, bus2, length_km=40., std_type='149-AL1/24-ST1A 110.0')

    # create loads
    pp.create_load(net, bus2, p_mw=60, controllable=False)
    pp.create_load(net, bus3, p_mw=70, controllable=False)
    pp.create_load(net, bus4, p_mw=10, controllable=False)

    # create generators
    pp.create_ext_grid(net, bus1)
    pp.create_gen(net, bus3, p_mw=80, max_p_mw=80, min_p_mw=0, vm_pu=1.01,
                  controllable=True)
    pp.create_gen(net, bus4, p_mw=0.1, max_p_mw=100, min_p_mw=0, vm_pu=1.01,
                  controllable=True)

    net.trafo["max_loading_percent"] = 50
    net.line["max_loading_percent"] = 50

    net.bus["min_vm_pu"] = 1.0
    net.bus["max_vm_pu"] = 1.02

    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=-0.01)
    pp.create_poly_cost(net, 1, "gen", cp1_eur_per_mw=-0.01)
    pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=0)



    pp.runopp(net, calculate_voltage_angles=True)
    assert net["OPF_converged"]
    assert allclose(net.res_bus.vm_pu.values, array([1., 1.00000149,  1.01998544,  1.01999628]),
                    atol=1e-5)
    assert allclose(net.res_bus.va_degree.values, array([0., -0.7055226, 0.85974768, 2.24584537]),
                    atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
