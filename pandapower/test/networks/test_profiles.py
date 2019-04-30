import pytest
from copy import deepcopy
import numpy as np
import pandas as pd

import pandapower.networks as nw


def test_dismantle_dict_values():
    test_keys = ["renewables", "load", "powerplants", "storage"]
    test_dict = dict.fromkeys(test_keys, 2)
    test_dict["load"] = dict.fromkeys(["p", "q"], 3)

    assert nw.dismantle_dict_values_to_deep_list(test_dict) == [[3, 3], 2, 2, 2]
    assert nw.dismantle_dict_values_to_list(test_dict) == [3, 3, 2, 2, 2]


def _testnet_with_profiles():
    # get and manipulate net
    net = nw.create_cigre_network_mv(with_der="all")
    in_net_profiles = pd.DataFrame([[0.1, 0.2, np.nan], [0.3, 0.2, 0.6]], columns=[
        "in_net1", "in_net2", "input1"])
    net["profiles"] = {"load": deepcopy(in_net_profiles),
                       "powerplants": in_net_profiles,
                       "renewables": in_net_profiles,
                       "storage": in_net_profiles}
    net["profiles"]["load"].columns += "_pload"
    in_net_profiles_qload = deepcopy(in_net_profiles)
    in_net_profiles_qload.columns += "_qload"
    net["profiles"]["load"] = pd.concat([net["profiles"]["load"], in_net_profiles_qload], axis=1)
    np.random.seed(1)
    for elm, high in zip(["load", "sgen", "storage"], [2, 3, 3]):
        net[elm]["profile"] = np.array(["in_net1", "in_net2", "input1"])[np.random.randint(
            0, high, net[elm].shape[0])]
    net["loadcases"] = pd.DataFrame([["hL", 1.0, 1., 0., 0., 0,  1.035],
                                     ["n1", 1.0, 1., 0., 0., 0,  1.035],
                                     ["hW", 1.0, 1., 1.00, 0.80, 1,  1.035],
                                     ["hPV",  1.0, 1., 0.85, 0.95, 1,  1.035],
                                     ["lW", 0.1, 0.122543, 1.00, 0.80, 1,  1.015],
                                     ["lPV",  0.1, 0.122543, 0.85, 0.95, 1,  1.015]],
                                    columns=["Study Case", "pload", "qload", "Wind_p", "PV_p",
                                             "RES_p", "Slack_vm"])
    net["loadcases"].set_index("Study Case", inplace=True)
    return net, deepcopy(in_net_profiles)


def test_missing_profiles():
    net, in_net_profiles = _testnet_with_profiles()

    assert not nw.profiles_are_missing(net, return_as_bool=True)
    av_prof = nw.get_available_profiles(net, "load", p_or_q="q", continue_on_missing=False)
    assert not len(set(in_net_profiles.columns) - set(av_prof))
    miss_prof = nw.get_missing_profiles(net, "load", p_or_q="p")
    assert isinstance(miss_prof, set)
    assert not len(miss_prof)

    # remove availabel profile
    del net.profiles["renewables"]["input1"]
    assert nw.profiles_are_missing(net, return_as_bool=True)
    av_prof = nw.get_available_profiles(net, "renewables", p_or_q=None, continue_on_missing=True)
    assert len(set(in_net_profiles.columns) - set(av_prof)) == 1
    miss_prof = nw.profiles_are_missing(net, return_as_bool=False)
    for key, val in miss_prof.items():
        if key == "load":
            for key2, val2 in miss_prof[key].items():
                assert not len(val2)
        elif key != "renewables":
            assert not len(val)
        else:
            assert val == set(["input1"])


def test_get_absolute_profiles_from_relative_profiles():
    net, in_net_profiles = _testnet_with_profiles()

    # test get_absolute_profiles_from_relative_profiles()
    for col in ["p_mw", "q_mvar"]:
        output_profiles = nw.get_absolute_profiles_from_relative_profiles(
            net, "load", col)
        assert output_profiles.shape[0] == in_net_profiles.shape[0]
        assert output_profiles.shape[1] == net.load.shape[0]
        idx = [0, 16, 17]
        excpected_output_values = net["profiles"]["load"].values[:, [1, 0, 1]] * \
            net.load[col].loc[idx].values
        assert np.allclose(output_profiles.loc[:, idx].values, excpected_output_values)

    output_profiles = nw.get_absolute_profiles_from_relative_profiles(
        net, "sgen", "p_mw")
    assert output_profiles.shape[0] == in_net_profiles.shape[0]
    assert output_profiles.shape[1] == net.sgen.shape[0]
    idx = [1, 3, 0]
    excpected_output_values = net["profiles"]["renewables"].fillna(0.123).values*net.sgen[
        "p_mw"].loc[idx].values
    assert np.allclose(output_profiles.loc[:, idx].fillna(0.123*2e-2).values,
                       excpected_output_values)

    # --- test get_absolute_values
    abs_val = nw.get_absolute_values(net, profiles_instead_of_study_cases=True)
    expected_keys = [("load", "p_mw"), ("load", "q_mvar"), ("sgen", "p_mw"), ("gen", "p_mw"),
                     ("storage", "p_mw")]
    assert len(abs_val.keys()) == len(expected_keys)
    for key, val in abs_val.items():
        assert key in expected_keys
        assert val.shape == (2, net[key[0]].shape[0])

    abs_val = nw.get_absolute_values(net, profiles_instead_of_study_cases=False)
    expected_keys = [("load", "p_mw"), ("load", "q_mvar"), ("sgen", "p_mw"), ("ext_grid", "vm_pu")]
    assert len(abs_val.keys()) == len(expected_keys)
    for key, val in abs_val.items():
        assert key in expected_keys
        assert val.shape == (7, net[key[0]].shape[0])


if __name__ == '__main__':
    if 0:
        pytest.main(["test_profiles.py", "-xs"])
    else:
#        test_dismantle_dict_values()
#        test_missing_profiles()
#        test_get_absolute_profiles_from_relative_profiles()
        pass
