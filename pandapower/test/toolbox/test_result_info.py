# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pytest

from pandapower.create import (
    create_empty_network, create_buses, create_ext_grid, create_gen, create_load, create_sgen,
    create_storage, create_dcline, create_line, create_transformer, create_bus, create_poly_cost, create_pwl_cost
)
from pandapower.run import runpp
from pandapower.networks import create_cigre_network_lv, case9
from pandapower.toolbox.comparison import dataframes_equal
from pandapower.toolbox.result_info import res_power_columns, opf_task, overloaded_lines, violated_buses, clear_result_tables


def test_opf_task():
    net = create_empty_network()
    create_buses(net, 6, [10, 10, 10, 0.4, 7, 7],
                    min_vm_pu=[0.9, 0.9, 0.88, 0.9, np.nan, np.nan])
    idx_ext_grid = 1
    create_ext_grid(net, 0, max_q_mvar=80, min_p_mw=0, index=idx_ext_grid)
    create_gen(net, 1, 10, min_q_mvar=-50, max_q_mvar=-10, min_p_mw=0, max_p_mw=60)
    create_gen(net, 2, 8)
    create_gen(net, 3, 5)
    create_load(net, 3, 120, max_p_mw=8)
    create_sgen(net, 1, 8, min_q_mvar=-50, max_q_mvar=-10, controllable=False)
    create_sgen(net, 2, 8)
    create_storage(net, 3, 2, 100, min_q_mvar=-10, max_q_mvar=-50, min_p_mw=0, max_p_mw=60,
                      controllable=True)
    create_dcline(net, 4, 5, 0.3, 1e-4, 1e-2, 1.01, 1.02, min_q_from_mvar=-10,
                     min_q_to_mvar=-10)
    create_line(net, 3, 4, 5, "122-AL1/20-ST1A 10.0", max_loading_percent=50)
    create_transformer(net, 2, 3, "0.25 MVA 10/0.4 kV")

    # --- run and check opf_task()
    out1 = opf_task(net, keep=True)
    assert out1["flexibilities_without_costs"] == "all"
    assert sorted(out1["flexibilities"].keys()) == [i1 + i2 for i1 in ["P", "Q"] for i2 in [
        "dcline", "ext_grid", "gen", "storage"]]
    for key, df in out1["flexibilities"].items():
        assert df.shape[0]
        if "gen" in key:
            assert df.shape[0] > 1
    assert out1["flexibilities"]["Pext_grid"].loc[0, "index"] == [1]
    assert np.isnan(out1["flexibilities"]["Pext_grid"].loc[0, "max"])
    assert out1["flexibilities"]["Pext_grid"].loc[0, "min"] == 0
    assert np.isnan(out1["flexibilities"]["Qext_grid"].loc[0, "min"])
    assert out1["flexibilities"]["Qext_grid"].loc[0, "max"] == 80
    assert sorted(out1["network_constraints"].keys()) == ["LOADINGline", "VMbus"]
    assert out1["network_constraints"]["VMbus"].shape[0] == 3

    # check delta_pq
    net.gen.loc[0, "min_p_mw"] = net.gen.loc[0, "max_p_mw"] - 1e-5
    out2 = opf_task(net, delta_pq=1e-3, keep=True)
    assert out2["flexibilities"]["Pgen"].shape[0] == 1

    net.gen.loc[0, "min_p_mw"] = net.gen.loc[0, "max_p_mw"] - 1e-1
    out1["flexibilities"]["Pgen"].loc[0, "min"] = out1["flexibilities"]["Pgen"].loc[
                                                      0, "max"] - 1e-1
    out3 = opf_task(net, delta_pq=1e-3, keep=True)
    for key in out3["flexibilities"]:
        assert dataframes_equal(out3["flexibilities"][key], out1["flexibilities"][key])

    # check costs
    create_poly_cost(net, idx_ext_grid, "ext_grid", 2)
    create_poly_cost(net, 1, "gen", 1.7)
    create_poly_cost(net, 0, "dcline", 2)
    create_pwl_cost(net, 2, "gen", [[-1e9, 1, 3.1], [1, 1e9, 0.5]], power_type="q")
    out4 = opf_task(net)
    for dict_key in ["flexibilities", "network_constraints"]:
        for key in out4[dict_key]:
            assert dataframes_equal(out4[dict_key][key], out1[dict_key][key])
    assert isinstance(out4["flexibilities_without_costs"], dict)
    expected_elm_without_cost = ["gen", "storage"]
    assert sorted(out4["flexibilities_without_costs"].keys()) == expected_elm_without_cost
    for elm in expected_elm_without_cost:
        assert len(out4["flexibilities_without_costs"][elm]) == 1


def test_overloaded_lines():
    net = create_empty_network()

    bus0 = create_bus(net, vn_kv=.4)
    bus1 = create_bus(net, vn_kv=.4)

    create_ext_grid(net, bus0)

    line0 = create_line(net, bus0, bus1, length_km=1, std_type="NAYY 4x50 SE")
    line1 = create_line(net, bus0, bus1, length_km=1, std_type="NA2XS2Y 1x95 RM/25 12/20 kV")
    line2 = create_line(net, bus0, bus1, length_km=1, std_type="15-AL1/3-ST1A 0.4")
    create_line(net, bus0, bus1, length_km=10, std_type="149-AL1/24-ST1A 10.0")

    create_load(net, bus1, p_mw=0.2, q_mvar=0.05)

    runpp(net)
    # test the overloaded lines by default value of max_load=100
    overloaded_lines_ = overloaded_lines(net, max_load=100)

    assert set(overloaded_lines_) == {line0, line1}

    # test the overloaded lines by a self defined value of max_load=50
    overloaded_lines_ = overloaded_lines(net, max_load=50)

    assert set(overloaded_lines_) == {line0, line1, line2}


def test_violated_buses():
    net = create_cigre_network_lv()

    runpp(net)

    # set the range of vm.pu
    min_vm_pu = 0.92
    max_vm_pu = 1.1

    # print out the list of violated_bus's index
    violated_bus = violated_buses(net, min_vm_pu, max_vm_pu)

    assert set(violated_bus) == set(net["bus"].index[[16, 35, 36, 40]])


def test_clear_result_tables():
    net = case9()
    runpp(net)
    elms_to_check = ["bus", "line", "load"]
    for elm in elms_to_check:
        assert net["res_%s" % elm].shape[0]
    clear_result_tables(net)
    for elm in elms_to_check:
        assert not net["res_%s" % elm].shape[0]


def test_res_power_columns():
    assert res_power_columns("gen") == ["p_mw", "q_mvar"]
    assert res_power_columns("line") == res_power_columns("line", side="from") == \
           res_power_columns("line", side=0) == ["p_from_mw", "q_from_mvar"]
    assert res_power_columns("line", side="all") == [
        "p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]
    assert res_power_columns("trafo3w", side="all") == [
        "p_hv_mw", "q_hv_mvar", "p_mv_mw", "q_mv_mvar", "p_lv_mw", "q_lv_mvar"]


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])