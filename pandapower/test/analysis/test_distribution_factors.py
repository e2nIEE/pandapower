import numpy as np
import pandas as pd
import pytest
import copy
from pandapower import pandapowerNet
from pandapower.run import rundcpp
from pandapower.analysis.PTDF import run_PTDF, verify_dc_profile_with_PTDF
from pandapower.analysis.LODF import run_LODF, verify_dc_n1_with_LODF
from pandapower.analysis.sensitivity_dc import run_dc_profile
from pandapower.networks.power_system_test_cases import (
    case30,
    case118,
    case_illinois200,
    case300,
    case1354pegase,
    case2869pegase,
    case6470rte,
    case9241pegase,
)
from pandapower.networks.create_examples import example_multivoltage


@pytest.fixture(
    params=[case30, case118, case_illinois200, case300, case1354pegase, case2869pegase, case6470rte, case9241pegase]
)
def net_in(request):
    net = request.param()
    return net


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


def test_ptdf(net_in: pandapowerNet):
    ptdf_matrix = run_PTDF(net_in, using_sparse_solver=True)
    ptdf_perturb = run_PTDF(net_in, source_bus=1000, perturb=True)
    ptdf_comp_df = pd.DataFrame(
        data={"matrix": ptdf_matrix["line"].loc[:, 1000], "perturb": ptdf_perturb["line"].loc[:, 1000]}
    )
    ptdf_comp_df["delta"] = ptdf_comp_df["matrix"] - ptdf_comp_df["perturb"]
    assert np.allclose(ptdf_comp_df["matrix"].to_numpy(), ptdf_comp_df["perturb"].to_numpy())


def test_lodf(net_in: pandapowerNet):
    lodf_matrix = run_LODF(net_in, outage_branch_type="line", outage_branch_ix=100, perturb=False, random_verify=False)
    lodf_perturb = run_LODF(net_in, outage_branch_type="line", outage_branch_ix=100, perturb=True)
    lodf_comp_df = pd.DataFrame(
        data={
            "matrix": lodf_matrix[("line", "line")].loc[:, 100],
            "perturb": lodf_perturb[("line", "line")].loc[:, 100],
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
    ptdf_t3w = run_PTDF(net)
    lodf_t3w = run_LODF(net, outage_branch_type="line")


def test_profile_multiple_elements(profiles):
    # Example run profile of multiple element types
    net = case118()
    res_profiles_ptdf = run_dc_profile(net, profiles=profiles)
    res_profiles_full = run_dc_profile(net, profiles=profiles, extra_data_points=[("bus", "va_degree")])
    verify_dc_profile_with_PTDF(net, profiles)


def test_run_selected_elements(profiles):
    # Example run profile simulation of only selected elements
    net = case118()
    profiles_partial = dict()
    num_calc = 100
    load_ix = [2, 3, 5]
    profiles_partial[("load", "p_mw")] = pd.DataFrame(
        data=np.tile(net["load"]["p_mw"].loc[load_ix].to_numpy(), (num_calc, 1)),
        index=np.arange(num_calc),
        columns=load_ix,
    )
    profiles_partial[("load", "p_mw")] *= np.random.rand(*profiles_partial[("load", "p_mw")].shape)

    res_profiles_partial = run_dc_profile(net, profiles_partial)
    verify_dc_profile_with_PTDF(net, profiles=profiles, result_side=1)
    verify_dc_n1_with_LODF(net, outage_branch_type="line")


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
