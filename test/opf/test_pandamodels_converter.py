# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import json
import os
from copy import deepcopy
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

import test
from pandapower.converter.pandamodels import convert_pp_to_pm
from pandapower.converter.pandamodels.from_pm import read_pm_results_to_net
from pandapower.create import create_poly_cost, create_storage
from pandapower.networks.cigre_networks import create_cigre_network_mv
from pandapower.opf.pm_storage import add_storage_opf_settings, read_pm_storage_results
from pandapower.pd2ppc import _pd2ppc
from pandapower.run import runopp
from test.opf.test_basic import (
    net_3w_trafo_opf,  # noqa: F401
    simple_opf_test_net,  # noqa: F401
)

julia_installed = find_spec("juliacall") is not None


def test_pp_to_pm_conversion(net_3w_trafo_opf):
    # tests if the conversion to power models works
    net = deepcopy(net_3w_trafo_opf)
    convert_pp_to_pm(net)
    convert_pp_to_pm(net, opf_flow_lim="I")


def test_pm_to_pp_conversion(simple_opf_test_net):
    # this tests checks if the runopp results are the same as the ones from powermodels.
    # Results are read from a result file containing the simple_opf_test_net

    net = simple_opf_test_net
    create_poly_cost(net, 0, "gen", cp1_eur_per_mw=100)

    # get pandapower opf results
    runopp(net, delta=1e-13)
    va_degree = deepcopy(net.res_bus.va_degree)
    vm_pu = deepcopy(net.res_bus.vm_pu)

    # get previously calculated power models results
    pm_res_file = os.path.join(os.path.abspath(os.path.dirname(test.__file__)), "test_files", "pm_example_res.json")

    with open(pm_res_file, "r") as fp:
        result_pm = json.load(fp)
    net._options["correct_pm_network_data"] = True
    ppc, ppci = _pd2ppc(net)
    read_pm_results_to_net(net, ppc, ppci, result_pm)
    assert np.allclose(net.res_bus.vm_pu, vm_pu, atol=1e-4)
    assert np.allclose(net.res_bus.va_degree, va_degree, atol=1e-2, rtol=1e-2)


def test_obj_factors(net_3w_trafo_opf):
    net = deepcopy(net_3w_trafo_opf)
    net["obj_factors"] = [0.9, 0.1]
    pm = convert_pp_to_pm(net)
    assert pm["user_defined_params"]["obj_factors"]["fac_1"] == 0.9
    assert pm["user_defined_params"]["obj_factors"]["fac_2"] == 0.1
    assert pm["user_defined_params"]["gen_and_controllable_sgen"]["2"] == 2
    assert pm["user_defined_params"]["gen_and_controllable_sgen"]["3"] == 3


def test_to_pm_angle_limits(net_3w_trafo_opf):
    """The branch angle limits of the net must reach the PowerModels data structure."""
    net = deepcopy(net_3w_trafo_opf)
    create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=100)
    net.line["min_angle_degree"] = -30.0
    net.line["max_angle_degree"] = 20.0
    net.trafo3w["min_angle_degree"] = -5.0
    net.trafo3w["max_angle_degree"] = 5.0

    pm = convert_pp_to_pm(net, correct_pm_network_data=False)

    line_limits = {
        (round(branch["angmin"], 9), round(branch["angmax"], 9))
        for branch in pm["branch"].values()
        if not branch["transformer"]
    }
    trafo_limits = {
        (round(branch["angmin"], 9), round(branch["angmax"], 9))
        for branch in pm["branch"].values()
        if branch["transformer"]
    }
    # MATPOWER units: angles in degrees, PandaModels.jl converts them to rad
    assert not pm["per_unit"]
    assert line_limits == {(-30.0, 20.0)}
    assert trafo_limits == {(-5.0, 5.0)}


def test_storage_opf_settings():
    """
    A controllable storage is optimized by the PowerModels storage model only, its ppc generator (with
    unlimited power) must be deactivated (PandaModels.jl#124). Non-controllable storages stay fixed loads.
    """
    net = create_cigre_network_mv("pv_wind")
    net.switch["closed"] = True
    s0 = create_storage(net, 10, p_mw=0.5, max_e_mwh=0.2, soc_percent=0.0, q_mvar=0.0, controllable=True)
    s1 = create_storage(net, 11, p_mw=0.1, max_e_mwh=1.0, soc_percent=50.0, q_mvar=0.0, controllable=False)
    s2 = create_storage(
        net,
        12,
        p_mw=0.0,
        max_e_mwh=1.0,
        soc_percent=50.0,
        q_mvar=0.0,
        max_p_mw=0.3,
        min_p_mw=-0.2,
        max_q_mvar=0.1,
        min_q_mvar=-0.05,
        sn_mva=0.4,
        controllable=True,
    )
    pm = convert_pp_to_pm(net)
    net._options["n_time_steps"] = 1
    net._options["time_elapsed"] = 1.0
    add_storage_opf_settings(net, net._ppc_opf, pm)

    storage_gens = {str(net._pd2pm_lookups["storage_controllable"][s]) for s in [s0, s2]}
    assert len(storage_gens) == 2
    for idx, gen in pm["gen"].items():
        assert gen["gen_status"] == (0 if idx in storage_gens else 1)

    # only the controllable storages are optimized, the other one keeps its fixed power as a load
    assert set(pm["storage"]) == {str(s0 + 1), str(s2 + 1)}
    assert pm["load"][str(net._pd2pm_lookups["storage"][s1])]["pd"] == 0.1

    # without power limits the storage power is limited by p_mw / q_mvar
    st = pm["storage"][str(s0 + 1)]
    assert (st["charge_rating"], st["discharge_rating"], st["thermal_rating"]) == (0.5, 0.5, 0.5)
    assert (st["qmin"], st["qmax"]) == (0.0, 0.0)
    st = pm["storage"][str(s2 + 1)]
    assert (st["charge_rating"], st["discharge_rating"], st["thermal_rating"]) == (0.3, 0.2, 0.4)
    assert (st["qmin"], st["qmax"]) == (-0.05, 0.1)


def test_storage_opf_settings_limits(caplog):
    net = create_cigre_network_mv("pv_wind")
    net.switch["closed"] = True
    # discharging set point without limits -> the limits are |p_mw| and |q_mvar|
    s0 = create_storage(net, 10, p_mw=-0.5, max_e_mwh=1.0, soc_percent=50.0, q_mvar=-0.1, controllable=True)
    create_poly_cost(net, s0, "storage", cp1_eur_per_mw=10)
    pm = convert_pp_to_pm(net)
    net._options["n_time_steps"] = 1
    net._options["time_elapsed"] = 1.0
    with caplog.at_level("WARNING", logger="pandapower.opf.pm_storage"):
        add_storage_opf_settings(net, net._ppc_opf, pm)
    # the storage model has no costs
    assert "poly_cost of the storages [0] are ignored" in caplog.text
    st = pm["storage"][str(s0 + 1)]
    assert (st["charge_rating"], st["discharge_rating"]) == (0.5, 0.5)
    assert (st["qmin"], st["qmax"]) == (-0.1, 0.1)

    # a minimum charging power cannot be modeled
    net.storage["min_p_mw"] = 0.1
    net.storage["max_p_mw"] = 0.5
    pm = convert_pp_to_pm(net)
    net._options["n_time_steps"] = 1
    net._options["time_elapsed"] = 1.0
    with pytest.raises(ValueError, match="min_p_mw"):
        add_storage_opf_settings(net, net._ppc_opf, pm)


def test_read_pm_storage_results_without_soc():
    net = create_cigre_network_mv("pv_wind")
    create_storage(net, 10, p_mw=0.1, max_e_mwh=1.0, controllable=False)
    p = pd.DataFrame({0: [0.1, 0.1]}, index=[3, 4])
    net.res_ts_opt = {"res_storage.p_mw": p, "res_storage.q_mvar": p * 0}
    res = read_pm_storage_results(net)[0]
    assert list(res.columns) == ["p_mw", "q_mvar", "soc_mwh", "soc_percent"]
    assert list(res.index) == [3, 4]
    assert np.allclose(res.p_mw, 0.1)
    assert res.soc_mwh.isna().all() and res.soc_percent.isna().all()


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
