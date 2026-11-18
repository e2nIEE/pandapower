# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy

import numpy as np
import pandas as pd
import pytest

from pandapower.control import ConstControl
from pandapower.networks.create_examples import example_multivoltage
from pandapower.networks.power_system_test_cases import (
    case14,
    case30,
    case118,
    case300,
    case1354pegase,
    case2869pegase,
)
from pandapower.run import rundcpp
from pandapower.sensitivity.lodf import run_lodf, verify_dc_n1_with_lodf
from pandapower.sensitivity.psdf import run_psdf
from pandapower.sensitivity.ptdf import run_ptdf, run_ptdf_dc_profile
from pandapower.timeseries import DFData, OutputWriter, run_timeseries


@pytest.fixture(
    params=[
        (case30, 10),
        (case118, 10),
        (case300, 100),
        (case1354pegase, 100),
        (case2869pegase, 100),
    ]
)
def net_in(request):
    case_func, lodf_line = request.param
    net = case_func()
    return net, lodf_line


@pytest.fixture
def profiles():
    net = case118()
    net.line.in_service.at[5] = False
    net.gen.slack.loc[[1, 5]] = True
    profiles = {}
    num_calc = 100
    # Create random profile for test
    for ele_type in ("load", "sgen", "gen"):
        if not net[ele_type].empty:
            profiles[(ele_type, "p_mw")] = pd.DataFrame(
                data=np.tile(net[ele_type]["p_mw"].to_numpy(), (num_calc, 1)),
                index=np.arange(num_calc),
                columns=net[ele_type].index.to_numpy(),
            )
            profiles[(ele_type, "p_mw")] *= np.random.rand(*profiles[(ele_type, "p_mw")].shape)
    return profiles


# TODO: source_bus is hardcoded to 1000 which makes no sense in context of the test networks.
# def test_ptdf(net_in: pandapowerNet):
#     ptdf_matrix = run_PTDF(net_in, using_sparse_solver=True)
#     ptdf_perturb = run_PTDF(net_in, source_bus=1000, perturb=True)
#     ptdf_comp_df = pd.DataFrame(
#         data={"matrix": ptdf_matrix["line"].loc[:, 1000], "perturb": ptdf_perturb["line"].loc[:, 1000]}
#     )
#     ptdf_comp_df["delta"] = ptdf_comp_df["matrix"] - ptdf_comp_df["perturb"]
#     assert np.allclose(ptdf_comp_df["matrix"].to_numpy(), ptdf_comp_df["perturb"].to_numpy())


def test_lodf(net_in):
    net, lodf_line = net_in
    outage_branch = lodf_line
    lodf_matrix = run_lodf(
        net, outage_branch_type="line", outage_branch_ix=outage_branch, perturb=False, random_verify=False
    )
    lodf_perturb = run_lodf(net, outage_branch_type="line", outage_branch_ix=outage_branch, perturb=True)
    lodf_comp_df = pd.DataFrame(
        data={
            "matrix": lodf_matrix[("line", "line")].loc[:, outage_branch],
            "perturb": lodf_perturb[("line", "line")].loc[:, outage_branch],
        }
    )
    lodf_comp_df["delta"] = lodf_comp_df["matrix"] - lodf_comp_df["perturb"]
    assert np.allclose(lodf_comp_df["matrix"].to_numpy(), lodf_comp_df["perturb"].to_numpy())


def test_random_outage_of_element():
    # Example distributed slacks
    net0 = case118()
    net0.line.in_service.iat[5] = False
    net0.gen.slack.iloc[[1, 5]] = True
    net1 = case118()
    net1.gen.slack.iloc[[2, 10, 20]] = True
    net1.bus.index += 118

    for ele_type in ("gen", "sgen", "load", "ext_grid"):
        net1[ele_type].bus += 118

    net1.line.from_bus += 118
    net1.line.to_bus += 118
    net1.trafo.hv_bus += 118
    net1.trafo.lv_bus += 118

    net = copy.deepcopy(net0)
    net.bus = pd.concat([net0.bus, net1.bus])
    for ele_type in ("gen", "sgen", "load", "ext_grid", "line", "trafo"):
        net[ele_type] = pd.concat([net0[ele_type], net1[ele_type]], ignore_index=True)

    rundcpp(net)


def test_trafo3w():
    # Example net with trafo3w
    net = example_multivoltage()
    run_ptdf(net)
    run_lodf(net, outage_branch_type="line")


def _dc_timeseries_reference(net, profiles, res_table, res_column):
    # exact reference: one DC power flow per time step with run_timeseries
    net = copy.deepcopy(net)
    for (ele_type, variable), profile in profiles.items():
        ConstControl(
            net,
            element=ele_type,
            variable=variable,
            element_index=profile.columns,
            data_source=DFData(profile),
            profile_name=profile.columns,
        )
    time_steps = next(iter(profiles.values())).index
    ow = OutputWriter(net, time_steps=time_steps, output_path=None)
    ow.log_variable(res_table, res_column)
    run_timeseries(net, time_steps=time_steps, run=rundcpp, verbose=False)
    return ow.output[f"{res_table}.{res_column}"]


def _assert_profile_matches_timeseries(net, profiles, result_side=0):
    res = run_ptdf_dc_profile(net, profiles, result_side=result_side)
    res_column = "p_from_mw" if result_side == 0 else "p_to_mw"
    reference = _dc_timeseries_reference(net, profiles, "res_line", res_column)
    assert np.allclose(res[("res_line", res_column)].to_numpy(), reference.to_numpy(), atol=1e-8)


def test_profile_multiple_elements(profiles):
    # Example run profile of multiple element types
    net = case118()
    _assert_profile_matches_timeseries(net, profiles)


def test_run_selected_elements(profiles):
    # Example run profile simulation of only selected elements
    net = case118()
    profiles_partial = {}
    num_calc = 100
    load_ix = [2, 3, 5]
    profiles_partial[("load", "p_mw")] = pd.DataFrame(
        data=np.tile(net["load"]["p_mw"].loc[load_ix].to_numpy(), (num_calc, 1)),
        index=np.arange(num_calc),
        columns=load_ix,
    )
    profiles_partial[("load", "p_mw")] *= np.random.rand(*profiles_partial[("load", "p_mw")].shape)

    _assert_profile_matches_timeseries(net, profiles_partial)
    _assert_profile_matches_timeseries(net, profiles, result_side=1)
    verify_dc_n1_with_lodf(net, outage_branch_type="line")


def test_psdf_direct_vs_perturb():
    for net_func in (case14, example_multivoltage):
        psdf_direct = run_psdf(net_func(), phase_shift_branch_type="trafo", distributed_slack=False)
        psdf_perturb = run_psdf(net_func(), phase_shift_branch_type="trafo", distributed_slack=False, perturb=True)
        assert sorted(psdf_direct) == sorted(psdf_perturb)
        for key, data in psdf_direct.items():
            assert np.allclose(data, psdf_perturb[key], equal_nan=True, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
