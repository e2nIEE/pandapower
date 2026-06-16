# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.network import pandapowerNet
from pandapower.create.trafo_characteristic_create import create_trafo_characteristic


def test_create_trafo_characteristic():
    net = pandapowerNet(name="test_create_trafo_characteristic")
    values = {
        "step": [1, 2, 3],
        "voltage_ratio": [0.98, 0.99, 1.00],
        "angle_deg": [30.0, 31.5, 33.0],
        "vk_percent": [10.1, 10.2, 10.3],
        "vkr_percent": [2.0, 2.1, 2.2],
    }

    create_trafo_characteristic(net, values)

    assert net.trafo_characteristic_table.loc[
               (0, 1), ["voltage_ratio", "angle_deg", "vk_percent", "vkr_percent"]].to_dict() == {
               "voltage_ratio": 0.98,
               "angle_deg": 30.0,
               "vk_percent": 10.1,
               "vkr_percent": 2.0,
           }
    assert net.trafo_characteristic_table.loc[
               (0, 2), ["voltage_ratio", "angle_deg", "vk_percent", "vkr_percent"]].to_dict() == {
               "voltage_ratio": 0.99,
               "angle_deg": 31.5,
               "vk_percent": 10.2,
               "vkr_percent": 2.1,
           }
    assert net.trafo_characteristic_table.loc[
               (0, 3), ["voltage_ratio", "angle_deg", "vk_percent", "vkr_percent"]].to_dict() == {
               "voltage_ratio": 1.0,
               "angle_deg": 33.0,
               "vk_percent": 10.3,
               "vkr_percent": 2.2,
           }
    assert net.trafo_characteristic_table.loc[:, "vk_hv_percent"].isna().all()
    assert net.trafo_characteristic_table.loc[:, "vkr_hv_percent"].isna().all()
    assert net.trafo_characteristic_table.loc[:, "vk_mv_percent"].isna().all()
    assert net.trafo_characteristic_table.loc[:, "vkr_mv_percent"].isna().all()
    assert net.trafo_characteristic_table.loc[:, "vk_lv_percent"].isna().all()
    assert net.trafo_characteristic_table.loc[:, "vkr_lv_percent"].isna().all()
