# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

from pandapower.create import (
    create_bus,
    create_empty_network,
    create_ext_grid,
    create_gen,
    create_line,
    create_load,
    create_sgen,
    create_transformer
)
from pandapower.test.loadflow.result_test_network_generator import result_test_network_generator


@pytest.fixture(scope="session")
def simple_network():
    net = create_empty_network()
    b1 = create_bus(net, name="bus1", vn_kv=10.)
    create_ext_grid(net, b1)
    b2 = create_bus(net, name="bus2", geodata=(1, 2), vn_kv=.4)
    b3 = create_bus(net, name="bus3", geodata=(1, 3), vn_kv=.4)
    b4 = create_bus(net, name="bus4", vn_kv=10.)
    create_transformer(net, b4, b2,
                          std_type="0.25 MVA 10/0.4 kV",
                          name=None, in_service=True, index=None)
    create_line(net, b2, b3, 1, name="line1",
                   std_type="NAYY 4x150 SE",
                   geodata=np.array([[1, 2], [3, 4]]))
    create_line(net, b1, b4, 1, name="line2",
                   std_type="NAYY 4x150 SE")
    create_load(net, b2, p_mw=0.01, q_mvar=0, name="load1")
    create_load(net, b3, p_mw=0.04, q_mvar=0.002, name="load2")
    create_gen(net, 3, q_mvar=0.020, vm_pu=1.0)
    create_sgen(net, 2, p_mw=0.050, sn_mva=0.1)
    return net


@pytest.fixture(scope="session")
def result_test_network():
    from pandapower import runpp

    # gets the last element of the generator
    for net in result_test_network_generator():
        pass
    runpp(net, trafo_model="t", trafo_loading="current")
    return net


def pytest_generate_tests(metafunc):
    if "result_test_networks" in metafunc.fixturenames:
        net = result_test_network_generator()
        metafunc.parametrize("result_test_networks", net, ids=lambda n: n.last_added_case)


if __name__ == '__main__':
    pytest.main([__file__, "-x"])
