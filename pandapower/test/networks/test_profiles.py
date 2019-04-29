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


def test_get_absolute_profiles_from_relative_profiles():
    # get and manipulate net
    net = nw.create_cigre_network_mv(with_der="all")
    in_net_profiles = pd.DataFrame([[0.1, 0.2, 0.5], [0.3, 0.2, 0.6]], columns=[
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

    # run get_absolute_profiles_from_relative_profiles() and test
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
    excpected_output_values = net["profiles"]["renewables"].values*net.sgen["p_mw"].loc[idx].values
    assert np.allclose(output_profiles.loc[:, idx].values, excpected_output_values)


if __name__ == '__main__':
    if 0:
        pytest.main(["test_profiles.py", "-xs"])
    else:
#        test_dismantle_dict_values()
#        test_get_absolute_profiles_from_relative_profiles()
        pass
