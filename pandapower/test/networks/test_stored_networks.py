# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest

import pandapower as pp
import pandapower.networks as pn


@pytest.mark.parametrize("stored_network_function_name", [
    "mv_oberrhein",
    "case4gs",
    "case5",
    "case6ww",
    "case9",
    "case11_iwamoto",
    "case14",
    "case24_ieee_rts",
    "case30",
    "case_ieee30",
    "case33bw",
    "case39",
    "case57",
    "case89pegase",
    "case118",
    "case145",
    "caseillinois200",
    "case300",
    "case1354pegase",
    "case1888rte",
    "case2848rte",
    "case2869pegase",
    "case3120sp",
    "case6470rte",
    "case6495rte",
    "case6515rte",
    "case9241pegase",
    "GBreducednetwork",
    "GBnetwork",
    "iceland",
    "lv_schutterwald",
    "ieee_european_lv_asymmetric",
])
def test_missing_element_parameters_in_stored_networks(stored_network_function_name):
    stored_network_function = getattr(pn, stored_network_function_name)
    net = stored_network_function()
    _test_missing_element_parameters_in_stored_networks(
        net, net_name=stored_network_function_name)


def _test_missing_element_parameters_in_stored_networks(net, net_name=None):
    """
    This test is expected to pass in most cases. However, if the networks stored as (json) files
    were already updated after the last release, the format_version was updated, too. If further
    updates are made but not intigrated to the (json) files, the loaded networks will not have the
    updates, even if these updates should be catched by the update_format() function. This is
    because update_format() won't do corrections to loaded data of the most recent format_version.
    To fix failing tests, the networks must be stored again including the most recent updates.
    """
    if net_name is None:
        net_name = net.name
    empty_net = pp.create_empty_network()

    et_with_different_columns = list()
    for et in pp.pp_elements():
        if len(empty_net[et].columns.difference(net[et].columns)):
            et_with_different_columns.append(et)
    if len(et_with_different_columns):
        raise AssertionError(
            f"In {net_name}, these element types misses some columns: {et_with_different_columns}")
    assert not len(et_with_different_columns)


def test_missing_element_parameters_in_mv_oberrhein_substations():
    net = pn.mv_oberrhein(include_substations=True)
    _test_missing_element_parameters_in_stored_networks(net, net_name="mv_oberrhein")


def test_missing_element_parameters_in_eu_lv_asymmetric_off_peak1():
    net = pn.ieee_european_lv_asymmetric(scenario="off_peak_1")
    _test_missing_element_parameters_in_stored_networks(
        net, net_name="ieee_european_lv_asymmetric")


def test_missing_element_parameters_in_eu_lv_asymmetric_off_peak1440():
    net = pn.ieee_european_lv_asymmetric(scenario="off_peak_1440")
    _test_missing_element_parameters_in_stored_networks(
        net, net_name="ieee_european_lv_asymmetric")


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
