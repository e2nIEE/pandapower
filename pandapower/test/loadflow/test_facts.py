# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandapower as pp
import pandapower.networks
from pandapower.control import ContinuousTapControl
from pandapower.pypower.idx_bus import PD, GS, VM
from pandapower.pypower.idx_brch import PF
import pytest
from pandapower.test.toolbox import assert_res_equal


def test_svc():
    net = pp.networks.case9()
    pp.create_shunt(net, 3, 0, 0, 345)
    net.shunt["set_vm_pu"] = 1.02
    net.shunt["svc_firing_angle"] = 45.
    pp.runpp(net, lightsim2grid=False, max_iteration=10)
    assert np.isclose(net.res_bus.at[3, 'vm_pu'], net.shunt.at[0, 'set_vm_pu'], rtol=0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
