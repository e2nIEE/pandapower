# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pytest

from pandapower.run import runpp
from pandapower.networks.power_system_test_cases import case14
from pandapower.create import create_empty_network, create_bus, create_ext_grid, create_sgen, create_load, \
    create_line, create_transformer3w
from pandapower.estimation import estimate
from pandapower.estimation.util import add_virtual_pmu_meas_from_loadflow

def run_se_verify(net):
    estimate(net, algorithm="wls")
    if not np.allclose(net.res_bus.vm_pu, net.res_bus_est.vm_pu, atol=1e-2) or\
       not np.allclose(net.res_bus.va_degree, net.res_bus_est.va_degree, atol=1e-1):
        raise AssertionError("Estimation failed!")


def test_pmu_case14():
    net = case14()

    runpp(net)
    add_virtual_pmu_meas_from_loadflow(net)

    run_se_verify(net)


def test_pmu_with_trafo3w():
    net = create_empty_network()

    bus_slack = create_bus(net, vn_kv=110)
    create_ext_grid(net, bus=bus_slack)

    bus_20_1 = create_bus(net, vn_kv=20, name="b")
    create_sgen(net, bus=bus_20_1, p_mw=0.03, q_mvar=0.02)

    bus_10_1 = create_bus(net, vn_kv=10)
    create_sgen(net, bus=bus_10_1, p_mw=0.02, q_mvar=0.02)

    bus_10_2 = create_bus(net, vn_kv=10)
    create_load(net, bus=bus_10_2, p_mw=0.06, q_mvar=0.01)
    create_line(net, from_bus=bus_10_1, to_bus=bus_10_2, std_type="149-AL1/24-ST1A 10.0", length_km=2)

    create_transformer3w(net, bus_slack, bus_20_1, bus_10_1, std_type="63/25/38 MVA 110/20/10 kV")

    runpp(net)
    add_virtual_pmu_meas_from_loadflow(net, with_random_error=False)

    run_se_verify(net)


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
