import pytest
import os
from copy import deepcopy
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
from pandapower.plotting import create_generic_coordinates

try:
    import pplog as logging
except ImportError:
    import logging

try:
    import igraph
    igraph_installed = True
except ImportError:
    igraph_installed = False

from pandapower import pp_dir
from pandapower.converter import csv2pp, csv_data2pp, pp2csv, pp2csv_data, \
    convert_parallel_branches, read_csv_data, ensure_full_column_data_existence, \
    avoid_duplicates_in_column, merge_busbar_coordinates

logger = logging.getLogger(__name__)

simbench_converter_test_path = os.path.join(pp_dir, "test", "converter")
test_network_path = os.path.join(simbench_converter_test_path, "test_network")
test_output_folder_path = os.path.join(simbench_converter_test_path, "test_network_output_folder")

__author__ = 'smeinecke'


def test_convert_parallel_branches():
    # create test grid
    net = pp.create_empty_network()
    pp.create_bus(net, 110)
    pp.create_buses(net, 4, 20)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 4, 1e3, 4e2)
    pp.create_transformer(net, 0, 1, "40 MVA 110/20 kV", name="sd", parallel=3)
    pp.create_switch(net, 1, 0, "t", name="dfjk")
    pp.create_line(net, 1, 2, 1.11, "94-AL1/15-ST1A 20.0", name="sdh", parallel=2)
    pp.create_switch(net, 2, 0, "l", name="dfsdf")
    pp.create_line(net, 2, 3, 1.11, "94-AL1/15-ST1A 20.0", name="swed", parallel=1)
    pp.create_line(net, 3, 4, 1.11, "94-AL1/15-ST1A 20.0", name="sdhj", parallel=3)
    pp.create_switch(net, 3, 2, "l", name="dfdfg")
    pp.create_switch(net, 4, 2, "l", False, name="dfhgj")
    # check test grid
    assert net.trafo.shape[0] == 1
    assert net.line.shape[0] == 3
    assert net.switch.shape[0] == 4

    convert_parallel_branches(net)
    # test parallelisation
    assert net.trafo.shape[0] == 3
    assert net.line.shape[0] == 6
    assert net.switch.shape[0] == 11

    net1 = deepcopy(net)
    net1.switch.closed.loc[4] = False
    convert_parallel_branches(net, multiple_entries=False)
    convert_parallel_branches(net1, multiple_entries=False)
    # test sum up of parallels
    assert net.trafo.shape[0] == 1
    assert net.line.shape[0] == 3
    assert net.switch.shape[0] == 4
    assert net1.trafo.shape[0] == 1
    assert net1.line.shape[0] == 4
    assert net1.switch.shape[0] == 5


@pytest.mark.skipif(igraph_installed==False, reason="requires igraph")
def test_test_network():
    net = csv2pp(test_network_path, no_generic_coord=True)

    # test min/max ratio
    for elm in pp.pp_elements(bus=False, branch_elements=False, other_elements=False):
        if "min_p_mw" in net[elm].columns and "max_p_mw" in net[elm].columns:
            isnull = net[elm][["min_p_mw", "max_p_mw"]].isnull().any(1)
            assert (net[elm].min_p_mw[~isnull] <= net[elm].max_p_mw[~isnull]).all()
        if "min_q_mvar" in net[elm].columns and "max_q_mvar" in net[elm].columns:
            isnull = net[elm][["min_q_mvar", "max_q_mvar"]].isnull().any(1)
            assert (net[elm].min_q_mvar[~isnull] <= net[elm].max_q_mvar[~isnull]).all()

    pp2csv(net, test_output_folder_path, export_pp_std_types=False, drop_inactive_elements=False)

    # --- test equality of exported csv data and given csv data
    csv_orig = read_csv_data(test_network_path, ";")
    csv_exported = read_csv_data(test_output_folder_path, ";")

    all_eq = True
    for tablename in csv_orig.keys():
        if "TransformerType" == tablename:
            print()
        try:
            eq = pp.dataframes_equal(csv_orig[tablename], csv_exported[tablename])
            if not eq:
                logger.error("csv_orig['%s'] and csv_exported['%s'] differ." % (tablename,
                                                                                tablename))
        except ValueError:
            eq = False
            logger.error("dataframes_equal did not work for %s." % tablename)
        all_eq &= eq
    assert all_eq


@pytest.mark.skipif(igraph_installed==False, reason="requires igraph")
def test_example_simple():
    net = nw.example_simple()

    # --- fix scaling
    net.load["scaling"] = 1.

    # --- add some additional data
    net.bus["subnet"] = ["net%i" % i for i in net.bus.index]
    pp.create_measurement(net, "i", "trafo", np.nan, np.nan, 0, "hv", name="1")
    pp.create_measurement(net, "i", "line", np.nan, np.nan, 1, "to", name="2")
    pp.create_measurement(net, "v", "bus", np.nan, np.nan, 0, name="3")

    net.shunt["max_step"] = np.nan
    stor = pp.create_storage(net, 6, 0.01, 0.1, -0.002, 0.05, 80, name="sda", min_p_mw=-0.01,
                             max_p_mw=0.008, min_q_mvar=-0.01, max_q_mvar=0.005)
    net.storage.loc[stor, "efficiency_percent"] = 90
    net.storage.loc[stor, "self-discharge_percent_per_day"] = 0.3
    pp.create_dcline(net, 4, 6, 0.01, 0.1, 1e-3, 1.0, 1.01, name="df", min_q_from_mvar=-0.01)
    pp.runpp(net)
    to_drop = pp.create_bus(net, 7, "to_drop")

    # --- add names to elements
    for i in pp.pp_elements():
        net[i] = ensure_full_column_data_existence(net, i, 'name')
        avoid_duplicates_in_column(net, i, 'name')

    # --- create geodata
    create_generic_coordinates(net)
    merge_busbar_coordinates(net)

    # --- convert
    csv_data = pp2csv_data(net, export_pp_std_types=True, drop_inactive_elements=True)
    net_from_csv_data = csv_data2pp(csv_data)

    # --- adjust net appearance
    pp.drop_buses(net, [to_drop])
    del net["OPF_converged"]
    del net_from_csv_data["substation"]
    del net_from_csv_data["profiles"]
    for key in net.keys():
        if isinstance(net[key], pd.DataFrame):
            # drop unequal columns
            dummy_columns = net[key].columns
            extra_columns = net_from_csv_data[key].columns.difference(dummy_columns)
            net_from_csv_data[key].drop(columns=extra_columns, inplace=True)
            # drop result table rows
            if "res_" in key:
                if not key == "res_bus":
                    net[key].drop(net[key].index, inplace=True)
                else:
                    net[key].loc[:, ["p_mw", "q_mvar"]] = np.nan
            # adjust dtypes
            if net[key].shape[0]:
                try:
                    net_from_csv_data[key] = net_from_csv_data[key].astype(dtype=dict(net[
                        key].dtypes))
                except:
                    logger.error("dtype adjustment of %s failed." % key)

    eq = pp.nets_equal(net, net_from_csv_data, check_only_results=False, exclude_elms=None)
    assert eq


if __name__ == "__main__":
    if 0:
        pytest.main([__file__, "-xs"])
    else:
#        test_convert_parallel_branches()
#        test_test_network()
#        test_example_simple()
        pass
