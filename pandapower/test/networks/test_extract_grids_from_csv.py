import pytest
import os
from copy import deepcopy
import numpy as np
import pandas as pd
import time

import pandapower as pp
from pandapower import pp_dir
import pandapower.converter as cv
import pandapower.networks as nw
from pandapower.networks.simbench.extract_simbench_grids_from_csv import \
    _get_extracted_csv_data_from_dict
try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

__author__ = "smeinecke"


def set_scalings(net, scaling):
    net.load.scaling = scaling
    net.sgen.scaling = scaling
    net.gen.scaling = scaling


def test_get_bus_bus_switch_indices_from_csv():
    node_table = pd.DataFrame([["Bus 1", "Type 4"],
                               ["Bus 3", "Type 1"],
                               ["Bus 4", "auxiliary"],
                               ["Bus 5", "auxiliary"]], columns=["id", "type"])
    switch_table = pd.DataFrame([["Sw 2", "Bus 1", "Bus 3"],
                                 ["Sw 3", "Bus 1", "Bus 4"],
                                 ["Sw 7", "Bus 1", "Bus 5"],
                                 ["Sw 8", "Bus 4", "Bus 3"],
                                 ["Sw 4", "Bus 4", "Bus 1"],
                                 ["Sw 9", "Bus 3", "Bus 1"],
                                 ["Sw 5", "Bus 5", "Bus 4"],
                                 ["Sw 1", "Bus 1", "Bus 2"]],
                                columns=["id", "nodeA", "nodeB"])
    try:
        nw.get_bus_bus_switch_indices_from_csv(switch_table, node_table)
        bool_ = False
    except ValueError:
        bool_ = True
    assert bool_

    switch_table.drop(switch_table.index[-1], inplace=True)
    try:
        nw.get_bus_bus_switch_indices_from_csv(switch_table, node_table)
        bool_ = False
    except ValueError:
        bool_ = True
    assert bool_

    switch_table.drop(switch_table.index[-1], inplace=True)
    assert nw.get_bus_bus_switch_indices_from_csv(switch_table, node_table) == [0, 5]


def test_get_relevant_subnets():
    input_path = nw.all_grids_csv_path(0)

    def subnets_stay_equal(sb_code_info, hv_subnet, lv_subnets):
        new_hv_subnet, new_lv_subnets = nw.get_relevant_subnets(sb_code_info, input_path=input_path)
        assert hv_subnet == new_hv_subnet
        assert lv_subnets == new_lv_subnets

    hv_subnet_list = ["HV1", "HV2"]
    mv_subnet_list = ["MV%i.%i" % (i, j) for j in [101, 201] for i in range(1, 5)]
    lv_subnet_list = ["LV%i.%i" % (i, j) for i in range(1, 7) for j in [101, 201, 301, 401]]
    unexpected_lv_subnet_list = ['LV5.101', 'LV6.101', 'LV1.301', 'LV2.301', 'LV1.401']
    for u in unexpected_lv_subnet_list:
        lv_subnet_list.remove(u)

    sb_code_params = [1, 'EHV', 'HVMVLV', 'mixed', 'all', '1', True]
    hv_subnets, lv_subnets = nw.get_relevant_subnets(sb_code_params, input_path=input_path)
    assert sorted(hv_subnets) == sorted(["EHV1"]+hv_subnet_list+mv_subnet_list[:4])
    assert pd.Series(hv_subnet_list+mv_subnet_list+lv_subnet_list).isin(lv_subnets).all()

    subnets_stay_equal(nw.complete_grid_sb_code(1), hv_subnets, lv_subnets)

    hv_subnet, lv_subnets = nw.get_relevant_subnets('1-complete_data-mixed-all-1-sw',
                                                    input_path=input_path)
    assert hv_subnet == "complete_data"
    assert lv_subnets == ""

    sb_code_params = [1, 'EHV', 'HV', 'mixed', 'all', '0', False]
    hv_subnet, lv_subnets = nw.get_relevant_subnets(sb_code_params, input_path=input_path)
    assert hv_subnet == "EHV1"
    assert lv_subnets == ["HV1", "HV2"]

    sb_code_params = [1, 'EHV', 'HV', 'mixed', 2, '0', False]
    hv_subnet, lv_subnets = nw.get_relevant_subnets(sb_code_params, input_path=input_path)
    assert hv_subnet == "EHV1"
    assert lv_subnets == ["HV2"]

    sb_code_params = [1, 'EHV', '', 'mixed', '', '0', True]
    hv_subnet, lv_subnets = nw.get_relevant_subnets(sb_code_params, input_path=input_path)
    assert hv_subnet == "EHV1"
    assert lv_subnets == []

    sb_code_params = [1, 'HV', 'MV', 'urban', 'all', '0', False]
    hv_subnet, lv_subnets = nw.get_relevant_subnets(sb_code_params, input_path=input_path)
    assert hv_subnet == "HV2"
    assert 30 > len(lv_subnets) > 8
    assert pd.Series(["MV1.201", "MV1.205", "MV2.203", "MV3.202", "MV4.203"]).isin(lv_subnets).all()

    sb_code_params = [1, 'HV', '', 'mixed', '', '1', True]
    hv_subnet, lv_subnets = nw.get_relevant_subnets(sb_code_params, input_path=input_path)
    assert hv_subnet == "HV1"
    assert lv_subnets == []

    sb_code_params = [1, 'MV', 'LV', 'semiurb', '3.209', '0', False]
    hv_subnet, lv_subnets = nw.get_relevant_subnets(sb_code_params, input_path=input_path)
    assert hv_subnet == "MV2.101"
    assert lv_subnets == ["LV3.209"]

    sb_code_params = [1, 'MV', 'LV', 'semiurb', '2.211', '1', True]
    hv_subnet, lv_subnets = nw.get_relevant_subnets(sb_code_params, input_path=input_path)
    assert hv_subnet == "MV2.101"
    assert lv_subnets == ["LV2.211"]

    hv_subnet, lv_subnets = nw.get_relevant_subnets('1-MV-rural--0-no_sw', input_path=input_path)
    assert hv_subnet == "MV1.101"
    assert lv_subnets == []

    sb_code_params = [1, 'LV', '', 'urban6', '', '0', False]
    hv_subnet, lv_subnets = nw.get_relevant_subnets(sb_code_params, input_path=input_path)
    assert hv_subnet == "LV6.201"
    assert lv_subnets == []

    hv_subnet, lv_subnets = nw.get_relevant_subnets('1-LV-semiurb5--0-sw', input_path=input_path)
    assert hv_subnet == "LV5.201"
    assert lv_subnets == []

    hv_subnet, lv_subnets = nw.get_relevant_subnets('1-LV-semiurb4--0-no_sw', input_path=input_path)
    assert hv_subnet == "LV4.101"
    assert lv_subnets == []


def csv_data_to_test_extracting():
    test_network_path = os.path.join(pp_dir, "test", "converter", "test_network")
    csv_data = cv.read_csv_data(test_network_path, sep=";")
    tested_tables = ["Node", "Load", "ExternalNet", "Switch", "Coordinates", "Measurement"]
    csv_data = {tt: csv_data[tt] for tt in tested_tables}
    csv_data["Node"]["subnet"] = ["EHV1_Feeder1"]*2 + ["EHV2"]*2 + ["EHV1_HV1"]*3 + [
        "HV1_Feder%i" % i for i in range(4)] + ["HV1_MV3.101"]*2 + ["MV3.101"] + ["EHV1_Feeder1"]*2
    csv_data["Load"]["subnet"] = ["EHV1_Load1", "EHV2_Load6", "EHV1_HV1_eq", "EHV1_HV1_load",
                                  "HV1_MV3.101_eq", "MV3.101_HV1_eq"]
    csv_data["ExternalNet"]["subnet"] = ["EHV1_boundary", "HV1_EHV1_eq", "MV3.101_HV1_eq",
                                         "MV3.101_bound", "EHV1_HV1_eq"]
    csv_data["Switch"].loc[3] = csv_data["Switch"].loc[0]
    csv_data["Switch"].loc[3, ["id", "nodeA", "nodeB"]] = ["BusBus2", "Bus 2", "Bus 4"]
    csv_data["Switch"]["subnet"] = ["EHV1_HV1", "EHV1_HV1", "HV1_MV3.101", "HV1_MV3.101",
        "EHV1_HV1"]
    csv_data["Coordinates"]["subnet"] = list(csv_data["Node"]["subnet"][1:12]) + ["MV3.101"]
    csv_data["Measurement"] = pd.concat([csv_data["Measurement"], csv_data["Measurement"],
                                        csv_data["Measurement"]], ignore_index=True)
    csv_data["Measurement"]["name"] = ["Messung %i" % i for i in range(1, 7)]
    csv_data["Measurement"]["subnet"] = ["HV1", "HV1_MV3.101", "EHV1_HV1"]*2
    return csv_data


def assert_csv_data_shape(csv_data, n_node, n_load, n_ext, n_sw, n_meas, print_instead=False):
    for tablename, n_expected in zip(
      ["Node", "Load", "ExternalNet", "Switch", "Coordinates", "Measurement"],
      [n_node, n_load, n_ext, n_sw, len(csv_data["Node"]["coordID"].unique()), n_meas]):
        if not print_instead:
            assert csv_data[tablename].shape[0] == n_expected
        else:
            if csv_data[tablename].shape[0] != n_expected:
                print("%s number (is / should be): %i / %i" % (
                    tablename, csv_data[tablename].shape[0], n_expected))


def test_get_extracted_csv_data_from_dict(print_instead=False):
    input_path = nw.all_grids_csv_path(0)

    csv_data = csv_data_to_test_extracting()

    data1 = _get_extracted_csv_data_from_dict(csv_data, ("EHV1", []))
    if print_instead:
        print("data1")
    assert_csv_data_shape(data1, 7, 3, 2, 3, 2, print_instead)

    data2 = _get_extracted_csv_data_from_dict(csv_data, ("EHV2", []))
    if print_instead:
        print("\ndata2")
    assert_csv_data_shape(data2, 2, 1, 0, 0, 0, print_instead)

    data3 = _get_extracted_csv_data_from_dict(csv_data, ("HV1", []))
    if print_instead:
        print("\ndata3")
    assert_csv_data_shape(data3, 9, 1, 1, 3, 5, print_instead)

    data4 = _get_extracted_csv_data_from_dict(csv_data, ("MV3.101", []))
    if print_instead:
        print("\ndata4")
    assert_csv_data_shape(data4, 3, 1, 2, 1, 1, print_instead)

    data5 = _get_extracted_csv_data_from_dict(csv_data, ("HV1", ["MV3.101"]))
    if print_instead:
        print("\ndata5")
    assert_csv_data_shape(data5, 10, 0, 2, 3, 4, print_instead)

    data6 = _get_extracted_csv_data_from_dict(csv_data, ("EHV1", ["HV1"]))
    if print_instead:
        print("\ndata6")
    assert_csv_data_shape(data6, 13, 2, 1, 5, 5, print_instead)

    hv_subnet, lv_subnets = nw.get_relevant_subnets('1-complete_data-mixed-all-2-sw',
                                                    input_path=input_path)
    data7 = _get_extracted_csv_data_from_dict(csv_data, (hv_subnet, lv_subnets))
    if print_instead:
        print("\nCheck equal DataFrames for lv_subnets=['all'].")
    for key, df7 in data7.items():
        to_assert = csv_data[key].equals(df7)
        if not print_instead:
            assert to_assert
        else:
            print(key, to_assert)

    sb_code_params = [1, 'EHV', 'HVMVLV', 'mixed', 'all', '1', True]
    hv_subnets, lv_subnets = nw.get_relevant_subnets(sb_code_params, input_path=input_path)
    data8 = _get_extracted_csv_data_from_dict(csv_data, (hv_subnets, lv_subnets))
    assert_csv_data_shape(data8, 14, 1, 2, 5, 4, print_instead)


def _net_for_testing():
    net = pp.create_empty_network()
    pp.create_buses(net, 17, 10, name=["Bus %i" % i for i in range(17)])
    pp.create_buses(net, 2, 0.4, name=["Bus %i" % i for i in range(17, 19)])
    pp.create_switch(net, 0, 1, "b", closed=True)
    pp.create_switch(net, 2, 3, "b", closed=False)
    pp.create_switch(net, 4, 5, "b", closed=False)
    pp.create_switch(net, 5, 6, "b", closed=True)

    pp.create_line(net, 7, 8, 1, 'NAYY 4x50 SE', name="Line 0")
    pp.create_switch(net, 7, 0, "l", closed=True)
    pp.create_switch(net, 8, 0, "l", closed=True)

    pp.create_line(net, 9, 10, 1, 'NAYY 4x50 SE', name="Line 1")
    pp.create_switch(net, 9, 1, "l", closed=False)
    pp.create_switch(net, 10, 1, "l", closed=True)

    pp.create_line(net, 11, 12, 1, 'NAYY 4x50 SE', name="Line 2")
    pp.create_switch(net, 11, 2, "l", closed=True)
    pp.create_switch(net, 12, 2, "l", closed=False)

    pp.create_line(net, 13, 14, 1, 'NAYY 4x50 SE', name="Line 3")
    pp.create_switch(net, 13, 3, "l", closed=False)
    pp.create_switch(net, 14, 3, "l", closed=False)

    pp.create_transformer(net, 15, 17, std_type="0.4 MVA 10/0.4 kV", name="Trafo 0")
    pp.create_switch(net, 15, 0, "t", closed=True)
    pp.create_switch(net, 17, 0, "t", closed=False)

    pp.create_transformer(net, 16, 18, std_type="0.4 MVA 10/0.4 kV", name="Trafo 1")
    pp.create_switch(net, 16, 1, "t", closed=False)
    pp.create_switch(net, 18, 1, "t", closed=True)

    return net


def test_generate_no_sw_variant():
    net_orig = _net_for_testing()
    net = deepcopy(net_orig)
    nw.generate_no_sw_variant(net)
    assert (net.bus.name.values == net_orig.bus.name.loc[net.bus.index.difference(
            [1, 6])].values).all()
    assert pp.dataframes_equal(net.line, net_orig.line)
    assert pp.dataframes_equal(net.trafo, net_orig.trafo)
    assert pp.dataframes_equal(net.switch, net_orig.switch.loc[[6, 9, 10, 11, 13, 14]])


def _test_net_validity(net, sb_code_params, shortened, input_path=None):
    """ This function is to test validity of a simbench net. """

    # --- deduce values from sb_code_params to test extracted csv data
    # lv_net_extent: 0-no lv_net    1-one lv_net    2-all lv_nets
    lv_net_extent = int(bool(len(sb_code_params[2])))
    if bool(lv_net_extent) and sb_code_params[4] == "all":
        lv_net_extent += 1
    # net_factor: how many lower voltage grids are expected to be connected
    if shortened:
        net_factor = 8
    else:
        if sb_code_params[1] == "HV":
            net_factor = 10
        else:
            net_factor = 50

    # --- test data existence
    # buses
    expected_buses = {0: 12, 1: 80, 2: net_factor*12}[lv_net_extent] if sb_code_params[1] != "EHV" \
        else {0: 6, 1: 65, 2: 125}[lv_net_extent]
    assert net.bus.shape[0] > expected_buses

    # ext_grid
    assert bool(net.ext_grid.shape[0])

    # switches
    if sb_code_params[6]:
        if int(sb_code_params[5]) > 0:
            if net.switch.shape[0] <= net.line.shape[0]*2-2:
                logger.info("There are %i switches, but %i " % (
                            net.switch.shape[0], net.line.shape[0]) +
                            "lines -> some lines are not surrounded by switches.")
        else:
            assert net.switch.shape[0] > net.line.shape[0]*2-2
    else:
        assert not net.switch.closed.any()
        assert (net.switch.et != "b").all()

    # all buses supplied
    if sb_code_params[1] != "complete_data":
        unsup_buses = pp.unsupplied_buses(net, respect_switches=False)
        if len(unsup_buses):
            logger.error("There are %i unsupplied buses." % len(unsup_buses))
            if len(unsup_buses) < 10:
                logger.error("These are: " + str(net.bus.name.loc[unsup_buses]))
        assert not len(unsup_buses)

    # lines
    assert net.line.shape[0] >= net.bus.shape[0]-net.trafo.shape[0]-(net.switch.et == "b").sum() - \
        2*net.trafo3w.shape[0]-net.impedance.shape[0]-net.dcline.shape[0]-net.ext_grid.shape[0]

    # trafos
    if sb_code_params[1] == "EHV":
        expected_trafos = {0: 0, 1: 2, 2: 8}[lv_net_extent]
    elif sb_code_params[1] == "HV":
        expected_trafos = {0: 2, 1: 4, 2: net_factor*2}[lv_net_extent]
    elif sb_code_params[1] == "MV":
        expected_trafos = {0: 2, 1: 3, 2: net_factor*1}[lv_net_extent]
    elif sb_code_params[1] == "LV":
        expected_trafos = {0: 1}[lv_net_extent]
    elif sb_code_params[1] == "complete_data":
        expected_trafos = 200
    assert net.trafo.shape[0] >= expected_trafos

    # load
    expected_loads = {0: 10, 1: net_factor, 2: net_factor*10}[lv_net_extent] if \
        sb_code_params[1] != "EHV" else {0: 3, 1: 53, 2: 113}[lv_net_extent]
    assert net.load.shape[0] > expected_loads

    # sgen
    if sb_code_params[1] == "LV":
        expected_sgen = {0: 0}[lv_net_extent]
    elif sb_code_params[2] == "LV":
        expected_sgen = {1: 50, 2: 50+net_factor*1}[lv_net_extent]
    else:
        expected_sgen = expected_loads
    assert net.sgen.shape[0] > expected_sgen

    # measurement
    if pd.Series(["HV", "MV"]).isin([sb_code_params[1], sb_code_params[2]]).any():
        assert net.measurement.shape[0] > 1

    # bus_geodata
    assert net.bus.shape[0] == net.bus_geodata.shape[0]

    # --- test data content
    # substation
    for elm in ["bus", "trafo", "trafo3w", "switch"]:
        mentioned_substations = pd.Series(net[elm].substation.unique()).dropna()
        if not mentioned_substations.isin(net.substation.name.values).all():
            raise AssertionError(str(list(mentioned_substations.loc[~mentioned_substations.isin(
                net.substation.name.values)].values)) +
                " from element '%s' misses in net.substation" % elm)

    # check subnet
    input_path = input_path if input_path is not None else nw.all_grids_csv_path(sb_code_params[5])
    hv_subnet, lv_subnets = nw.get_relevant_subnets(sb_code_params, input_path=input_path)
    allowed_elms_missing_subnet = ["gen", "dcline", "trafo3w", "impedance", "measurement",
                                   "shunt", "storage", "ward", "xward"]
    if not sb_code_params[6]:
        allowed_elms_missing_subnet += ["switch"]

    if sb_code_params[1] != "complete_data":
        hv_subnets = cv.ensure_iterability(hv_subnet)
        for elm in pp.pp_elements():
            if "subnet" not in net[elm].columns or not bool(net[elm].shape[0]):
                assert elm in allowed_elms_missing_subnet
            else:  # subnet is in net[elm].columns and there are one or more elements
                subnet_split = net[elm].subnet.str.split("_", expand=True)
                subnet_ok = set()
                subnet_ok |= set(subnet_split.index[subnet_split[0].isin(hv_subnets+lv_subnets)])
                if elm in ["bus", "measurement", "switch"]:
                    if 1 in subnet_split.columns:
                        subnet_ok |= set(subnet_split.index[subnet_split[1].isin(hv_subnets)])
                assert len(subnet_ok) == net[elm].shape[0]

    # check profile existing
    assert not nw.profiles_are_missing(net)

    # --- check profiles and loadflow
    check_loadflow = sb_code_params[1] != "complete_data"
    check_loadflow &= sb_code_params[2] != "HVMVLV"
    check_loadflow &= not ((sb_code_params[1] == "MV") & (sb_code_params[3] == "urban") &
                           (sb_code_params[4] == "all") & (sb_code_params[5] != "0"))
    if check_loadflow:
        try:
            pp.runpp(net)
            converged = net.converged
        except:
            sb_code = nw.get_simbench_code_from_parameters(sb_code_params)
            logger.error("Loadflow not converged with %s" % sb_code)
            converged = False
        assert converged


@pytest.mark.slow
def test_get_simbench_net(sb_codes=None, n=8, scenarios=None, input_path=None):
    """ Test nets exctracted from csv (shortened) folder. 'shortened' and 'test' must be set properly
        in extract_simbench_grids_from_csv.py
        If sb_codes is None, randomn simbench codes are tested in the number of n.
        If in input_path is None, no input_path will be given to get_simbench_net()
    """
    shortened = input_path is not None and "shortened" in input_path
    scenarios = scenarios if scenarios is not None else [0, 1, 2]
    if not sb_codes:
        sb_codes = []
        for scenario in scenarios:
            sb_codes += nw.collect_all_simbench_codes(scenario=scenario, shortened=shortened)
        first = 2  # always test the first 2 (complete_data and complete_grid)
        np.random.seed(int(time.time()))
        sb_codes = [sb_codes[i] for i in np.append(np.random.randint(first, len(sb_codes)-first, n),
                                                   range(first))]
    else:
        sb_codes = cv.ensure_iterability(sb_codes)

    for sb_code_info in sb_codes:
        sb_code, sb_code_params = nw.get_simbench_code_and_parameters(sb_code_info)
        logger.info("Get SimBench net '%s'" % sb_code)
        net = nw.get_simbench_net(sb_code, input_path=input_path)
        logger.info("Now test validity...")
        _test_net_validity(net, sb_code_params, shortened, input_path=input_path)


if __name__ == '__main__':
    if 0:
        pytest.main([__file__, "-xs"])
    else:
#        test_get_bus_bus_switch_indices_from_csv()
#        test_get_relevant_subnets()
#        test_get_extracted_csv_data_from_dict(print_instead=True)
#        test_generate_no_sw_variant()
#        test_get_simbench_net()
#        test_get_simbench_net(sb_codes=nw.collect_all_simbench_codes())

        pass