# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import pytest
from numpy import array, allclose

import pandapower as pp

try:
    import pplog as logging
except:
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
    pp.create_load(net, bus2, p_kw=60e3, controllable=False)
    pp.create_load(net, bus3, p_kw=70e3, controllable=False)
    pp.create_load(net, bus4, p_kw=10e3, controllable=False)

    # create generators
    pp.create_ext_grid(net, bus1)
    pp.create_gen(net, bus3, p_kw=-80 * 1e3, min_p_kw=-80e3, max_p_kw=0, vm_pu=1.01,
                  controllable=True)
    pp.create_gen(net, bus4, p_kw=-100 * 1e3, min_p_kw=-100e3, max_p_kw=0, vm_pu=1.01,
                  controllable=True)

    net.trafo["max_loading_percent"] = 50
    net.line["max_loading_percent"] = 50

    net.bus["min_vm_pu"] = 1.0
    net.bus["max_vm_pu"] = 1.02

    pp.create_polynomial_cost(net, 0, "gen", array([-1e-5, 0]))
    pp.create_polynomial_cost(net, 1, "gen", array([-1e-5, 0]))
    pp.create_polynomial_cost(net, 0, "ext_grid", array([0, 0]))



    pp.runopp(net, calculate_voltage_angles=True)
    assert net["OPF_converged"]
    assert allclose(net.res_bus.vm_pu.values, array([1., 1.00000149,  1.01998544,  1.01999628]),
                    atol=1e-5)
    assert allclose(net.res_bus.va_degree.values, array([0., -0.7055226, 0.85974768, 2.24584537]),
                    atol=1e-5)


if __name__ == "__main__":
    pytest.main(["test_curtailment.py", "-xs"])
