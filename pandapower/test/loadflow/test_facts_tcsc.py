import copy

import numpy as np
import pytest

from pandapower.create import (
    create_impedance,
    create_buses,
    create_tcsc,
    create_bus,
    create_empty_network,
    create_line_from_parameters,
    create_load,
    create_ext_grid,
)

from pandapower.run import runpp
from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.test.loadflow.test_facts import (
    copy_with_impedance,
    facts_case_study_grid,
    compare_tcsc_impedance,
)


def add_tcsc_to_line(net, xl, xc, set_p_mw, from_bus, line, side="from_bus"):
    aux = create_bus(net, net.bus.at[from_bus, "vn_kv"], "aux")
    net.line.loc[line, side] = aux

    idx = create_tcsc(net, from_bus, aux, xl, xc, set_p_mw, 100, controllable=True)
    return idx


def test_tcsc_simple():
    net = create_empty_network()
    create_buses(net, 2, 110)
    create_ext_grid(net, 0)
    # create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
    create_load(net, 1, 100, 25)
    create_tcsc(net, 0, 1, 1, -10, -100, 140, controllable=False)

    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net.impedance.index)

    net.tcsc.controllable = True
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net.impedance.index)


def test_tcsc_simple1():
    net = create_empty_network()
    create_buses(net, 3, 110)
    create_ext_grid(net, 0)
    create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 1, 100, 0.0487, 0.13823, 160, 0.664)
    create_load(net, 1, 100, 25)
    create_tcsc(net, 0, 2, 1, -10, 6, 144, controllable=False)

    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net.impedance.index)

    net.tcsc.controllable = True
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net.impedance.index)


def test_tcsc_simple2():
    net = create_empty_network()
    create_buses(net, 3, 110)
    create_ext_grid(net, 0)
    # create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
    create_load(net, 1, 40, 25)
    create_load(net, 2, 60, 25)
    create_tcsc(net, 0, 1, 1, -10, -40, 140, controllable=False)
    create_tcsc(net, 0, 2, 1, -10, -60, 140, controllable=False)

    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    net.tcsc.at[0, "controllable"] = True
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    net.tcsc.at[1, "controllable"] = True
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)


def test_tcsc_simple3():
    baseMVA = 100  # MVA
    baseV = 110  # kV
    baseI = baseMVA / (baseV * np.sqrt(3))
    baseZ = baseV**2 / baseMVA
    xl = 0.2
    xc = -20
    # plot_z(baseZ, xl, xc)

    # (0)-------------(1)-----------------(3)->
    #                  |--(TCSC)--(2)------|

    net = create_empty_network(sn_mva=baseMVA)
    create_buses(net, 4, baseV)
    create_ext_grid(net, 0)
    create_line_from_parameters(net, 0, 1, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 3, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 20, 0.0487, 0.13823, 160, 0.664)

    create_load(net, 3, 100, 40)

    create_tcsc(
        net,
        1,
        2,
        xl,
        xc,
        5,
        170,
        "Test",
        controllable=True,
        min_angle_degree=90,
        max_angle_degree=180,
    )

    runpp_with_consistency_checks(net, init="dc")

    net_ref = copy_with_impedance(net)
    runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    # todo:
    #  test with distributed slack
    #  test results by comparing impedance result to formula; p, q, i by comparing to line results; vm, va by comparing to bus results


def test_tcsc_simple3_slack():
    baseMVA = 100  # MVA
    baseV = 110  # kV
    baseI = baseMVA / (baseV * np.sqrt(3))
    baseZ = baseV**2 / baseMVA
    xl = 0.2
    xc = -20
    # plot_z(baseZ, xl, xc)

    # (0)-------------(1)-----------------(3)->
    #  |-----(TCSC)--(2)-------------------|

    net = create_empty_network(sn_mva=baseMVA)
    create_buses(net, 4, baseV)
    create_ext_grid(net, 0)
    create_line_from_parameters(net, 0, 1, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 3, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 20, 0.0487, 0.13823, 160, 0.664)

    create_load(net, 2, 100, 40)

    # plotting.simple_plot(net)
    create_tcsc(
        net,
        2,
        0,
        xl,
        xc,
        5,
        170,
        "Test",
        controllable=True,
        min_angle_degree=90,
        max_angle_degree=180,
    )

    runpp_with_consistency_checks(net, init="dc")

    net_ref = copy_with_impedance(net)
    runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    # todo:
    #  test with distributed slack
    #  test results by comparing impedance result to formula; p, q, i by comparing to line results; vm, va by comparing to bus results


def test_compare_to_impedance():
    baseMVA = 100  # MVA
    baseV = 110  # kV
    baseI = baseMVA / (baseV * np.sqrt(3))
    baseZ = baseV**2 / baseMVA
    xl = 0.2
    xc = -20
    # plot_z(baseZ, xl, xc)

    # (0)-------------(1)-----------------(3)->
    #                  |--(TCSC)--(2)------|

    net = create_empty_network(sn_mva=baseMVA)
    create_buses(net, 4, baseV)
    create_ext_grid(net, 0)
    create_line_from_parameters(net, 0, 1, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 3, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 20, 0.0487, 0.13823, 160, 0.664)

    create_load(net, 3, 100, 40)

    net_ref = copy.deepcopy(net)

    create_tcsc(
        net,
        1,
        2,
        xl,
        xc,
        -20,
        170,
        "Test",
        controllable=True,
        min_angle_degree=90,
        max_angle_degree=180,
    )

    runpp_with_consistency_checks(net, init="dc")

    create_impedance(net_ref, 1, 2, 0, net.res_tcsc.x_ohm.at[0] / baseZ, baseMVA)

    runpp(net_ref)

    # compare when controllable
    compare_tcsc_impedance(net, net_ref, 0, 0)
    assert np.allclose(
        net._ppc["internal"]["J"].toarray()[:-1, :-1],
        net_ref._ppc["internal"]["J"].toarray(),
        rtol=0,
        atol=5e-5,
    )
    assert np.allclose(
        net._ppc["internal"]["Ybus"].toarray(),
        net_ref._ppc["internal"]["Ybus"].toarray(),
        rtol=0,
        atol=1e-6,
    )

    # compare when not controllable
    net.tcsc.thyristor_firing_angle_degree = net.res_tcsc.thyristor_firing_angle_degree
    net.tcsc.controllable = False
    runpp_with_consistency_checks(net, init="dc")

    compare_tcsc_impedance(net, net_ref, 0, 0)
    assert np.allclose(
        net._ppc["internal"]["J"].toarray(),
        net_ref._ppc["internal"]["J"].toarray(),
        rtol=0,
        atol=5e-5,
    )
    assert np.allclose(
        net._ppc["internal"]["Ybus"].toarray(),
        net_ref._ppc["internal"]["Ybus"].toarray(),
        rtol=0,
        atol=1e-6,
    )


def test_tcsc_case_study():
    net = facts_case_study_grid()
    baseMVA = net.sn_mva
    baseV = 230
    baseZ = baseV**2 / baseMVA
    xl = 0.2
    xc = -20
    # plot_z(baseZ, xl, xc)
    f = net.bus.loc[net.bus.name == "B4"].index.values[0]
    t = net.bus.loc[net.bus.name == "B6"].index.values[0]
    aux = create_bus(net, 230, "aux")
    l = net.line.loc[(net.line.from_bus == f) & (net.line.to_bus == t)].index.values[0]
    net.line.loc[l, "from_bus"] = aux

    net_ref = copy.deepcopy(net)

    create_tcsc(net, f, aux, xl, xc, -100, 100, controllable=True)
    runpp(net, init="dc")

    create_impedance(net_ref, f, aux, 0, net.res_tcsc.at[0, "x_ohm"] / baseZ, baseMVA)
    runpp(net_ref)

    compare_tcsc_impedance(net, net_ref, 0, 0)
    assert np.allclose(
        net._ppc["internal"]["Ybus"].toarray(),
        net_ref._ppc["internal"]["Ybus"].toarray(),
        rtol=0,
        atol=1e-6,
    )


def test_tcsc_simple5():
    net = create_empty_network(sn_mva=100)
    create_buses(net, 4, 110)
    create_ext_grid(net, 0)
    create_line_from_parameters(net, 0, 1, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 3, 20, 0.0487, 0.13823, 160, 0.664)
    create_load(net, 3, 100, 25)

    create_tcsc(net, 2, 3, 1, -10, -20, 90)
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, 0, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
