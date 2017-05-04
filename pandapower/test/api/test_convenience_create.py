# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandapower as pp
import pytest


def test_convenience_create_functions():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110.)
    b2 = pp.create_bus(net, 110.)
    pp.create_ext_grid(net, b1)
    pp.create_line_from_parameters(net, b1, b2, length_km=20., r_ohm_per_km=0.0487,
                                   x_ohm_per_km=0.1382301, c_nf_per_km=160., max_i_ka=0.664)

    l0 = pp.create_load_from_cosphi(net, b2, 10e3, 0.95, "ind", name="load")
    pp.runpp(net)
    assert net.load.p_kw.at[l0] == 9.5e3
    assert net.load.q_kvar.at[l0] > 0
    assert np.sqrt(net.load.p_kw.at[l0]**2 +  net.load.q_kvar.at[l0]**2) == 10e3
    assert np.isclose(net.res_bus.vm_pu.at[b2], 0.99990833838)
    assert net.load.name.at[l0] == "load"

    sh0 = pp.create_shunt_as_condensator(net, b2, 10e3, loss_factor=0.01, name="shunt")
    pp.runpp(net)
    assert np.isclose(net.res_shunt.q_kvar.at[sh0], -10,043934174e3)
    assert np.isclose(net.res_shunt.p_kw.at[sh0], 100.43933665)
    assert np.isclose(net.res_bus.vm_pu.at[b2], 1.0021942964)
    assert net.shunt.name.at[sh0] == "shunt"

    sg0 = pp.create_sgen_from_cosphi(net, b2, 5e3, 0.95, "cap", name="sgen")
    assert np.sqrt(net.sgen.p_kw.at[sg0]**2 +  net.sgen.q_kvar.at[sg0]**2) == 5e3
    assert net.sgen.p_kw.at[sg0] == -4.75e3
    assert net.sgen.q_kvar.at[sg0] < 0
    assert np.isclose(net.res_bus.vm_pu.at[b2], 1.0021942964)
    assert net.sgen.name.at[sg0] == "sgen"

if __name__ == '__main__':
    pytest.main(["test_convenience_create.py"])

