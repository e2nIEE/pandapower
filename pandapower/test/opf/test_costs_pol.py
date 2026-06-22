# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pytest

from pandapower.create import create_empty_network, create_bus, create_gen, create_ext_grid, create_load, \
    create_line_from_parameters, create_poly_cost, create_sgen
from pandapower.converter.pypower import to_ppc
from pandapower.pypower.idx_cost import COST, NCOST
from pandapower.run import runopp

import logging


def _create_controllable_load_and_sgen_cost_net():
    vm_max = 1.05
    vm_min = 0.95

    net = create_empty_network()
    create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    create_ext_grid(net, 0, controllable=False)
    create_line_from_parameters(net, 0, 1, 1, name="line2", r_ohm_per_km=0.1,
                                c_nf_per_km=0.0, max_i_ka=1.0, x_ohm_per_km=0.1,
                                max_loading_percent=100)
    load = create_load(net, 1, p_mw=0.5, q_mvar=0.1, controllable=True,
                       min_p_mw=0.1, max_p_mw=1.0, min_q_mvar=-0.5, max_q_mvar=0.5)
    sgen = create_sgen(net, 1, p_mw=0.5, q_mvar=0.1, controllable=True,
                       min_p_mw=0.1, max_p_mw=1.0, min_q_mvar=-0.5, max_q_mvar=0.5)

    create_poly_cost(net, load, "load", cp0_eur=3, cp1_eur_per_mw=1, cp2_eur_per_mw2=2)
    create_poly_cost(net, sgen, "sgen", cp0_eur=30, cp1_eur_per_mw=10, cp2_eur_per_mw2=20)

    return net, load, sgen


def test_cost_pol_gen():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """
    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = create_empty_network()
    create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    create_gen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.15, max_q_mvar=0.05,
               min_q_mvar=-0.05)
    create_ext_grid(net, 0)
    create_load(net, 1, p_mw=0.02, controllable=False)
    create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                max_loading_percent=100 * 690)

    create_poly_cost(net, 0, "gen", cp1_eur_per_mw=1)

    runopp(net)

    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, net.res_gen.p_mw.values)

    net.poly_cost.at[0, "cp1_eur_per_mw"] = 0
    net.poly_cost.at[0, "cp2_eur_per_mw2"] = 1
    # run OPF
    runopp(net)

    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, net.res_gen.p_mw.values ** 2)


def test_cost_pol_all_elements():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """
    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = create_empty_network()
    create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    create_gen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.15, max_q_mvar=0.05,
               min_q_mvar=-0.05)
    create_sgen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.15, max_q_mvar=0.05,
                min_q_mvar=-0.05)
    create_ext_grid(net, 0)
    create_load(net, 1, p_mw=0.02, controllable=False)
    create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                max_loading_percent=100 * 690)

    create_poly_cost(net, 0, "gen", cp1_eur_per_mw=1)
    create_poly_cost(net, 0, "sgen", cp1_eur_per_mw=1)
    # run OPF
    runopp(net)

    assert net["OPF_converged"]
    assert abs(net.res_cost - (net.res_gen.p_mw.values + net.res_sgen.p_mw.values)) < 1e-2

    net.poly_cost.at[0, "cp1_eur_per_mw"] = 0
    net.poly_cost.at[0, "cp2_eur_per_mw2"] = 1

    runopp(net)

    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, net.res_gen.p_mw.values ** 2 + net.res_sgen.p_mw.values)


def test_controllable_load_polynomial_cost_signs():
    """Controllable loads use the opposite active-power sign in PYPOWER."""
    net, load, sgen = _create_controllable_load_and_sgen_cost_net()
    ppci = to_ppc(net, mode="opf", init="flat", calculate_voltage_angles=False)
    load_gen = net._pd2ppc_lookups["load_controllable"][load]
    sgen_gen = net._pd2ppc_lookups["sgen_controllable"][sgen]

    assert ppci["gencost"][load_gen, NCOST] == 3
    assert ppci["gencost"][sgen_gen, NCOST] == 3
    assert np.allclose(ppci["gencost"][load_gen, COST:COST + 3], [2, -1, 3])
    assert np.allclose(ppci["gencost"][sgen_gen, COST:COST + 3], [20, 10, 30])

    net.poly_cost.loc[:, "cp2_eur_per_mw2"] = 0
    ppci = to_ppc(net, mode="opf", init="flat", calculate_voltage_angles=False)

    assert ppci["gencost"][load_gen, NCOST] == 2
    assert ppci["gencost"][sgen_gen, NCOST] == 2
    assert np.allclose(ppci["gencost"][load_gen, COST:COST + 2], [-1, 3])
    assert np.allclose(ppci["gencost"][sgen_gen, COST:COST + 2], [10, 30])


def test_cost_pol_q():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """
    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = create_empty_network()
    create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    create_sgen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.15, max_q_mvar=0.05,
                min_q_mvar=-0.05)
    create_ext_grid(net, 0)
    create_load(net, 1, p_mw=0.02, controllable=False)
    create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                max_loading_percent=100 * 690)

    create_poly_cost(net, 0, "sgen", cp1_eur_per_mw=0, cq1_eur_per_mvar=-1)
    # run OPF
    runopp(net)

    assert net["OPF_converged"]
    assert abs(net.res_cost + net.res_sgen.q_mvar.values) < 1e-2

    net.poly_cost.at[0, "cq1_eur_per_mvar"] = 0
    net.poly_cost.at[0, "cq2_eur_per_mvar2"] = 1
    #    net.poly_cost.at[0, "c"] = np.array([[1, 0, 0]])
    # run OPF
    runopp(net)

    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, net.res_sgen.q_mvar.values ** 2)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel("DEBUG")
    pytest.main([__file__, "-xs"])
