import copy
from itertools import product

import numpy as np
import pytest
from pandapower.create import create_impedance, create_shunts, create_buses, create_gens, create_svc, create_tcsc, \
    create_bus, create_empty_network, create_line_from_parameters, create_load, create_ext_grid, \
    create_transformer_from_parameters, create_gen, create_ssc
from pandapower.run import runpp
from pandapower.test.consistency_checks import runpp_with_consistency_checks


def _many_tcsc_test_net():
    #                  |--(TCSC)--(4)------|
    # (0)-------------(1)-----------------(3)--------(6)
    #                  |-(5)-(TCSC)--(2)---|#
    # unfortunately, TCSC is not implemented for the case when multiple TCSC elements
    # have the same from_bus or to_bus
    baseMVA = 100
    baseV = 110
    baseZ = baseV ** 2 / baseMVA
    xl = 0.2
    xc = -15

    net = create_empty_network(sn_mva=baseMVA)
    create_buses(net, 7, baseV)
    create_ext_grid(net, 0)
    create_line_from_parameters(net, 0, 1, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 3, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 4, 3, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 5, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 3, 6, 20, 0.0487, 0.13823, 160, 0.664)

    create_load(net, 3, 100, 40)

    create_tcsc(net, 5, 2, xl, xc, -10, 160, "Test", controllable=True, min_angle_degree=90,
                max_angle_degree=180)
    create_tcsc(net, 1, 4, xl, xc, -5, 160, "Test", controllable=True, min_angle_degree=90,
                max_angle_degree=180)

    create_svc(net, 3, 1, -10, 1.01, 144)
    create_svc(net, 2, 1, -10, 1., 144)

    create_ssc(net, 6, 0, 5, 1, controllable=True, in_service=True)
    return net


def compare_tcsc_impedance(net, net_ref, idx_tcsc, idx_impedance):
    backup_q = net_ref.res_bus.loc[net.ssc.bus.values, "q_mvar"].copy()
    net_ref.res_bus.loc[net.ssc.bus.values, "q_mvar"] += net_ref.res_impedance.loc[
        net_ref.impedance.query("name=='ssc'").index, "q_from_mvar"].values
    bus_idx = net.bus.index.values
    for col in ("vm_pu", "va_degree", "p_mw", "q_mvar"):
        assert np.allclose(net.res_bus[col], net_ref.res_bus.loc[bus_idx, col], rtol=0, atol=1e-6)
    assert np.allclose(net.res_ext_grid.p_mw, net_ref.res_ext_grid.p_mw, rtol=0, atol=1e-6)
    assert np.allclose(net.res_ext_grid.q_mvar, net_ref.res_ext_grid.q_mvar, rtol=0, atol=1e-6)

    for col in "p_from_mw", "q_from_mvar", "p_to_mw", 'q_to_mvar':
        assert np.allclose(net.res_tcsc.loc[idx_tcsc, col], net_ref.res_impedance.loc[idx_impedance, col],
                           rtol=0, atol=1e-6)
    assert np.allclose(net._ppc["internal"]["Ybus"].toarray(), net_ref._ppc["internal"]["Ybus"].toarray(), rtol=0,
                       atol=1e-6)
    net_ref.res_bus.loc[net.ssc.bus.values, "q_mvar"] = backup_q


def compare_ssc_impedance_gen(net, net_ref, element="ssc"):
    backup_q = net_ref.res_bus.loc[net[element].bus.values, "q_mvar"].copy()
    ### comparing the original buses in net and net_ref(witout the auxilary buses)
    net_ref.res_bus.loc[net[element].bus.values, "q_mvar"] += \
        net_ref.res_impedance.loc[net_ref.impedance.query(f"name=='{element}'").index, "q_from_mvar"].values
    assert np.allclose(np.abs(net._ppc["internal"]["V"][net.bus.index]),
                       np.abs(net_ref._ppc["internal"]["V"][net.bus.index]), rtol=0, atol=1e-6)

    for col in net.res_bus.columns:
        assert np.allclose(net.res_bus[col][net.bus.index], net_ref.res_bus[col][net.bus.index], rtol=0, atol=1e-6)

    ### compare the internal bus at ssc and the auxilary bus at net_ref
    in_service = net[element].in_service.values
    for i, j in zip(['vm_internal_pu', 'va_internal_degree'], ['vm_pu', 'va_degree']):
        assert np.allclose(net[f"res_{element}"][i][in_service],
                           net_ref.res_bus[j][len(net.bus):].values[in_service], rtol=0, atol=1e-6)

    assert np.allclose(np.abs(net._ppc["internal"]["V"][len(net.bus):]),
                       net_ref.res_bus.vm_pu[len(net.bus):][in_service], rtol=0, atol=1e-6)
    assert np.allclose(np.angle(net._ppc["internal"]["V"][len(net.bus):], deg=True),
                       net_ref.res_bus.va_degree[len(net.bus):][in_service], rtol=0, atol=1e-6)

    # compare ext_grid_result
    for col in net.res_ext_grid.columns:
        assert np.allclose(net.res_ext_grid[col][net.ext_grid.index], net_ref.res_ext_grid[col][net.ext_grid.index],
                           rtol=0, atol=1e-6)

    # compare line results
    ###
    if "res_line" in net:
        for col in net.res_line.columns:
            assert np.allclose(net.res_line[col][net.line.index], net_ref.res_line[col][net.line.index], rtol=0, atol=1e-6)

    assert np.allclose(net._ppc["internal"]["Ybus"].toarray(), net_ref._ppc["internal"]["Ybus"].toarray(), rtol=0,
                       atol=1e-6)
    net_ref.res_bus.loc[net[element].bus.values, "q_mvar"] = backup_q



def copy_with_impedance(net):
    baseMVA = net.sn_mva  # MVA
    baseV = net.bus.vn_kv.values  # kV
    baseI = baseMVA / (baseV * np.sqrt(3))
    baseZ = baseV ** 2 / baseMVA

    net_ref = copy.deepcopy(net)
    for i in net.tcsc.index.values:
        create_impedance(net_ref, net.tcsc.from_bus.at[i], net.tcsc.to_bus.at[i], 0,
                         net.res_tcsc.x_ohm.at[i] / baseZ[net.tcsc.to_bus.at[i]], baseMVA,
                         in_service=net.tcsc.in_service.at[i], name="tcsc")
    net_ref.tcsc.in_service = False

    if len(net.svc) > 0:
        # create_loads(net_ref, net.svc.bus.values, 0, net.res_svc.q_mvar.values, in_service=net.svc.in_service.values)
        # create shunts because of Ybus comparison
        q = np.square(net.bus.loc[net.svc.bus.values, 'vn_kv']) / net.res_svc.x_ohm.values
        create_shunts(net_ref, net.svc.bus.values, q.fillna(0), in_service=net.svc.in_service.values, name="svc")
        net_ref.svc.in_service = False

    if len(net.ssc) > 0:
        # create shunts because of Ybus comparison
        in_service = net.ssc.in_service.values
        ssc_bus = net.ssc.bus.values
        aux_bus = create_buses(net_ref, len(net.ssc), net.bus.loc[ssc_bus, "vn_kv"].values)
        for fb, tb, r, x, i in zip(ssc_bus, aux_bus, net.ssc.r_ohm.values / baseZ[ssc_bus],
                                   net.ssc.x_ohm.values / baseZ[ssc_bus], in_service):
            create_impedance(net_ref, fb, tb, r, x, baseMVA, name="ssc", in_service=i)
        if len(net.res_ssc) > 0:
            vm_pu = net.res_ssc.vm_internal_pu.fillna(1)
        else:
            vm_pu = net.ssc.set_vm_pu.fillna(1)
        create_gens(net_ref, aux_bus, 0, vm_pu, in_service=in_service)
        net_ref.ssc.in_service = False

    if len(net.vsc) > 0:
        # create shunts because of Ybus comparison
        in_service = net.vsc.in_service.values
        vsc_bus = net.vsc.bus.values
        aux_bus = create_buses(net_ref, len(net.vsc), net.bus.loc[vsc_bus, "vn_kv"].values)
        for fb, tb, r, x, i in zip(vsc_bus, aux_bus, net.vsc.r_ohm.values / baseZ[vsc_bus],
                                   net.vsc.x_ohm.values / baseZ[vsc_bus], in_service):
            create_impedance(net_ref, fb, tb, r, x, baseMVA, name="vsc", in_service=i)
        g = create_gens(net_ref, aux_bus, 0, net.res_vsc.vm_internal_pu.fillna(1), in_service=in_service)
        vsc_pv = net.vsc.control_mode_ac.values == "vm_pu"
        net_ref.gen.loc[g[vsc_pv], "vm_pu"] = net.vsc.loc[vsc_pv, "control_value_ac"].values
        net_ref.vsc.in_service = False
        net_ref.bus_dc.loc[net.vsc.bus_dc.values, "in_service"] = False

    return net_ref


def test_multiple_facts():
    #                  |--(TCSC)--(4)------|
    # (0)-------------(1)-----------------(3)--------(6)
    #                  |-(5)-(TCSC)--(2)---|#
    # unfortunately, TCSC is not implemented for the case when multiple TCSC elements
    # have the same from_bus or to_bus
    baseMVA = 100
    baseV = 110
    baseZ = baseV ** 2 / baseMVA
    xl = 0.2
    xc = -15

    net = create_empty_network(sn_mva=baseMVA)
    create_buses(net, 7, baseV)
    create_ext_grid(net, 0)
    create_line_from_parameters(net, 0, 1, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 3, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 4, 3, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 5, 20, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 3, 6, 20, 0.0487, 0.13823, 160, 0.664)

    create_load(net, 3, 100, 40)

    create_tcsc(net, 5, 2, xl, xc, -10, 140, "Test", controllable=True, min_angle_degree=90, max_angle_degree=180)
    create_tcsc(net, 1, 4, xl, xc, -5, 140, "Test", controllable=True, min_angle_degree=90, max_angle_degree=180)

    runpp_with_consistency_checks(net)

    # net = _many_tcsc_test_net()

    net.tcsc.loc[1, "thyristor_firing_angle_degree"] = net.res_tcsc.loc[1, "thyristor_firing_angle_degree"]
    net.tcsc.loc[1, "controllable"] = False
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.query("name=='tcsc'").index)

    create_svc(net, 3, 1, -10, 1.01, 90)
    runpp_with_consistency_checks(net)

    net.svc.at[0, "thyristor_firing_angle_degree"] = net.res_svc.loc[0, "thyristor_firing_angle_degree"]
    net.svc.controllable = False
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.query("name=='tcsc'").index)
    compare_tcsc_impedance(net, net_ref, net.svc.index, net_ref.impedance.query("name=='svc'").index)


@pytest.mark.parametrize("svc_status", list(product([True, False], repeat=2)))
@pytest.mark.parametrize("tcsc_status", list(product([True, False], repeat=2)))
@pytest.mark.parametrize("ssc_status", list(product([True, False], repeat=1)))
def test_multiple_facts_combinations(svc_status, tcsc_status, ssc_status):
    net = _many_tcsc_test_net()

    net.svc.controllable = svc_status
    net.tcsc.controllable = tcsc_status
    net.ssc.in_service = ssc_status
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.query("name=='tcsc'").index)
    compare_ssc_impedance_gen(net, net_ref)

    net.svc.controllable = True
    net.tcsc.controllable = True
    net.svc.in_service = svc_status
    net.tcsc.in_service = tcsc_status

    # create_ssc(net, 6, 0, 5, 1,controllable=True,in_service=True)

    runpp_with_consistency_checks(net)

    net_ref = copy_with_impedance(net)
    runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.query("name=='tcsc'").index)
    compare_ssc_impedance_gen(net, net_ref)


def test_svc_tcsc_case_study():
    net = facts_case_study_grid()
    baseMVA = net.sn_mva
    baseV = 230
    baseZ = baseV ** 2 / baseMVA
    xl = 0.2
    xc = -20
    # plot_z(baseZ, xl, xc)
    f = net.bus.loc[net.bus.name == "B4"].index.values[0]
    t = net.bus.loc[net.bus.name == "B6"].index.values[0]
    aux = create_bus(net, 230, "aux")
    l = net.line.loc[(net.line.from_bus == f) & (net.line.to_bus == t)].index.values[0]
    net.line.loc[l, "from_bus"] = aux

    create_tcsc(net, f, aux, xl, xc, -100, 100, controllable=True)

    create_svc(net, net.bus.loc[net.bus.name == "B7"].index.values[0], 1, -10, 1., 90)

    runpp(net, init="dc")

    net_ref = copy_with_impedance(net)
    runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    net.gen.slack_weight = 1
    runpp(net, distributed_slack=True, init="dc")
    net_ref = copy_with_impedance(net)
    runpp(net_ref, distributed_slack=True)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)


def facts_case_study_grid():
    net = create_empty_network()

    b1 = create_bus(net, name="B1", vn_kv=18)
    b2 = create_bus(net, name="B2", vn_kv=16.5)
    b3 = create_bus(net, name="B3", vn_kv=230)
    b4 = create_bus(net, name="B4", vn_kv=230)
    b5 = create_bus(net, name="B5", vn_kv=230)
    b6 = create_bus(net, name="B6", vn_kv=230)
    b7 = create_bus(net, name="B7", vn_kv=230)
    b8 = create_bus(net, name="B8", vn_kv=230)

    create_ext_grid(net, bus=b1, vm_pu=1, va_degree=0)

    create_line_from_parameters(net, name="L1", from_bus=b3, to_bus=b4, length_km=30, r_ohm_per_km=0.049,
                                x_ohm_per_km=0.136, g_us_per_km=0, c_nf_per_km=142, max_i_ka=1.5)
    create_line_from_parameters(net, name="L2", from_bus=b3, to_bus=b4, length_km=30, r_ohm_per_km=0.049,
                                x_ohm_per_km=0.136, g_us_per_km=0, c_nf_per_km=142, max_i_ka=1.5)
    create_line_from_parameters(net, name="L3", from_bus=b4, to_bus=b5, length_km=100, r_ohm_per_km=0.081,
                                x_ohm_per_km=0.312, g_us_per_km=0, c_nf_per_km=11, max_i_ka=1.5)
    create_line_from_parameters(net, name="L4", from_bus=b4, to_bus=b6, length_km=100, r_ohm_per_km=0.081,
                                x_ohm_per_km=0.312, g_us_per_km=0, c_nf_per_km=11, max_i_ka=1.5)
    create_line_from_parameters(net, name="L5", from_bus=b5, to_bus=b7, length_km=220, r_ohm_per_km=0.081,
                                x_ohm_per_km=0.312, g_us_per_km=0, c_nf_per_km=11, max_i_ka=1.5)
    create_line_from_parameters(net, name="L6", from_bus=b6, to_bus=b8, length_km=140, r_ohm_per_km=0.081,
                                x_ohm_per_km=0.312, g_us_per_km=0, c_nf_per_km=11, max_i_ka=1.5)
    create_line_from_parameters(net, name="L7", from_bus=b5, to_bus=b6, length_km=180, r_ohm_per_km=0.081,
                                x_ohm_per_km=0.312, g_us_per_km=0, c_nf_per_km=11, max_i_ka=1.5)
    create_line_from_parameters(net, name="L8", from_bus=b7, to_bus=b8, length_km=180, r_ohm_per_km=0.081,
                                x_ohm_per_km=0.312, g_us_per_km=0, c_nf_per_km=11, max_i_ka=1.5)

    # create_line_from_parameters(net,name="L9",from_bus=3,to_bus=4,length_km=100, r_ohm_per_km=0.312,
    # x_ohm_per_km=0.312,g_us_per_km=0,c_nf_per_km=11)

    create_transformer_from_parameters(net, name="trafo1", hv_bus=b8, lv_bus=b1, sn_mva=192, vn_hv_kv=230,
                                       vn_lv_kv=18, vkr_percent=0, vector_group="Yy0", pfe_kw=0, vk_percent=12,
                                       i0_percent=0)
    create_transformer_from_parameters(net, name="trafo2", hv_bus=b3, lv_bus=b2, sn_mva=500, vn_hv_kv=230,
                                       vn_lv_kv=16.5, vkr_percent=0, vector_group="Yy0", pfe_kw=0, vk_percent=16,
                                       i0_percent=0)

    create_gen(net, bus=b2, p_mw=500, vm_pu=1)
    # create_sgen(net,bus = 2, p_mw=500,name="WT")
    #
    create_load(net, bus=b4, p_mw=130, q_mvar=50)
    create_load(net, bus=b5, p_mw=120, q_mvar=50)
    create_load(net, bus=b6, p_mw=80, q_mvar=25)
    create_load(net, bus=b7, p_mw=50, q_mvar=25)

    return net


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
