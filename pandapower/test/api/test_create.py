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
    pp.runpp(net, init="flat")
    assert net.load.p_kw.at[l0] == 9.5e3
    assert net.load.q_kvar.at[l0] > 0
    assert np.sqrt(net.load.p_kw.at[l0]**2 +  net.load.q_kvar.at[l0]**2) == 10e3
    assert np.isclose(net.res_bus.vm_pu.at[b2], 0.99990833838)
    assert net.load.name.at[l0] == "load"

    sh0 = pp.create_shunt_as_capacitor(net, b2, 10e3, loss_factor=0.01, name="shunt")
    pp.runpp(net, init="flat")
    assert np.isclose(net.res_shunt.q_kvar.at[sh0], -10,043934174e3)
    assert np.isclose(net.res_shunt.p_kw.at[sh0], 100.43933665)
    assert np.isclose(net.res_bus.vm_pu.at[b2], 1.0021942964)
    assert net.shunt.name.at[sh0] == "shunt"

    sg0 = pp.create_sgen_from_cosphi(net, b2, 5e3, 0.95, "cap", name="sgen")
    pp.runpp(net, init="flat")
    assert np.sqrt(net.sgen.p_kw.at[sg0]**2 +  net.sgen.q_kvar.at[sg0]**2) == 5e3
    assert net.sgen.p_kw.at[sg0] == -4.75e3
    assert net.sgen.q_kvar.at[sg0] < 0
    assert np.isclose(net.res_bus.vm_pu.at[b2], 1.0029376578)
    assert net.sgen.name.at[sg0] == "sgen"

def test_nonexistent_bus():
    from functools import partial
    net = pp.create_empty_network()
    create_functions = [partial(pp.create_load, net=net, p_kw=0, q_kvar=0, bus=0, index=0),
                        partial(pp.create_sgen, net=net, p_kw=0, q_kvar=0, bus=0, index=0),
                        partial(pp.create_dcline, net, from_bus=0, to_bus=1, p_kw=100,
                                loss_percent=0, loss_kw=10., vm_from_pu=1., vm_to_pu=1., index=0),
                        partial(pp.create_gen, net=net, p_kw=0, bus=0, index=0),
                        partial(pp.create_ward, net, 0, 0, 0, 0, 0, index=0),
                        partial(pp.create_xward, net, 0, 0, 0, 0, 0, 1, 1, 1, index=0),
                        partial(pp.create_shunt, net=net, q_kvar=0, bus=0, index=0),
                        partial(pp.create_ext_grid, net=net, bus=1, index=0),
                        partial(pp.create_line, net=net, from_bus=0, to_bus=1, length_km=1.,
                                std_type="NAYY 4x50 SE", index=0),
                        partial(pp.create_line_from_parameters, net=net, from_bus=0, to_bus=1,
                                length_km=1., r_ohm_per_km=0.1, x_ohm_per_km=0.1, max_i_ka=0.4,
                                c_nf_per_km=10, index=1),
                        partial(pp.create_transformer, net=net, hv_bus=0, lv_bus=1,
                                std_type="63 MVA 110/20 kV", index=0),
                        partial(pp.create_transformer3w, net=net, hv_bus=0, lv_bus=1, mv_bus=2,
                                std_type="63/25/38 MVA 110/20/10 kV", index=0),
                        partial(pp.create_transformer3w_from_parameters, net=net, hv_bus=0,
                                lv_bus=1, mv_bus=2, i0_percent = 0.89, pfe_kw= 35,
                                vn_hv_kv= 110, vn_lv_kv= 10, vn_mv_kv= 20,  sn_hv_kva= 63000,
                                sn_lv_kva = 38000, sn_mv_kva = 25000, vsc_hv_percent= 10.4,
                                vsc_lv_percent= 10.4, vsc_mv_percent= 10.4, vscr_hv_percent= 0.28,
                                 vscr_lv_percent= 0.35, vscr_mv_percent= 0.32, index=1),
                        partial(pp.create_transformer_from_parameters, net=net, hv_bus=0, lv_bus=1,
                                sn_kva=600, vn_hv_kv=20., vn_lv_kv=0.4, vsc_percent=10,
                                vscr_percent=0.1, pfe_kw=0, i0_percent=0, index=1),
                        partial(pp.create_impedance, net=net, from_bus=0, to_bus=1,
                                rft_pu=0.1, xft_pu=0.1, sn_kva=600, index=0),
                        partial(pp.create_switch, net, bus=0, element=1, et="b", index=0)]
    for func in create_functions:
        with pytest.raises(Exception): #exception has to be raised since bus doesn't exist
            func()
    pp.create_bus(net, 0.4)
    pp.create_bus(net, 0.4)
    pp.create_bus(net, 0.4)
    for func in create_functions:
        func() #buses exist, element can be created
        with pytest.raises(Exception): #exception is raised because index already exists
            func()

if __name__ == '__main__':
    test_convenience_create_functions()
#    pytest.main(["test_create.py"])

