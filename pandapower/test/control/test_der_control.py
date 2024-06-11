# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import os
from copy import deepcopy
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.control.controller.DERController as DERModels

try:
    from pandaplan.core import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def simple_test_net():
    net = pp.create_empty_network()
    pp.create_buses(net, 2, vn_kv=20)
    pp.create_ext_grid(net, 0)
    pp.create_sgen(net, 1, p_mw=2., sn_mva=3, name="DER1")
    pp.create_line(net, 0, 1, length_km=0.1, std_type="NAYY 4x50 SE")
    return net


def simple_test_net2():
    net = simple_test_net()
    bus = pp.create_bus(net, vn_kv=20)
    pp.create_line(net, 0, bus, 0.1, std_type="NAYY 4x50 SE")
    pp.create_sgen(net, bus, 2., sn_mva=3., name="DER2")
    return net


def test_qofv():
    """ Test basic QV curve behaviour of QModelCosphiVCurve and QModelQVCurve. """

    net = simple_test_net()
    p = net.sgen.p_mw.at[0]
    sn = net.sgen.sn_mva.at[0]

    qofv_cosphi = DERModels.QModelCosphiVCurve({
        "vm_points_pu": (0, 0.96, 1., 1.04),
        "cosphi_points": (0.9, 0.9, 1, -0.9)})
    qofv_q = DERModels.QModelQVCurve({
        "vm_points_pu": (0, 0.96, 1., 1.04),
        "q_points_pu": (0.4843221*p/sn, 0.4843221*p/sn, 0., -0.4843221*p/sn)})

    # the following, applied pqv_area has no influence in this test (vm near 1, p > 0.2 -> no
    # limitation). The functionality is not tested here. It is only tested that using it produces
    # no errors
    pp.control.DERController(net, 0, q_model=qofv_cosphi, pqv_area=DERModels.PQVArea4120V2())
    pp.control.DERController(net, 0, q_model=qofv_q, pqv_area=DERModels.PQVArea4120V2())

    pp.runpp(net)
    # check that vm difference to ext_grid is low
    assert  0.995 <= net.res_bus.vm_pu.at[1] <= 1.005
    assert np.isclose(net.res_sgen.q_mvar.at[0], 0)


    # --- run control -> nearly no q injection since vm is nearly 1.0
    # pf without controller
    net.controller.in_service = [True, False]
    pp.runpp(net, run_control=True)
    assert 0.991 <= pp.cosphi_from_pq(-net.res_sgen.p_mw.at[0], -net.res_sgen.q_mvar.at[0])[0]
    # pf with 2nd controller (should have same result)
    net.controller.in_service = [False, True]
    pp.runpp(net, run_control=True)
    assert 0.995 <= pp.cosphi_from_pq(-net.res_sgen.p_mw.at[0], -net.res_sgen.q_mvar.at[0])[0]


    # --- run control -> q injection is positive with cosphi=0.9 since vm is nearly 1.05
    net.ext_grid.vm_pu = 1.05
    net.sgen.q_mvar = 0.

    # pf without controller
    pp.runpp(net)
    vmb4 = net.res_bus.vm_pu.at[1]

    # pf with 1st controller
    net.controller.in_service = [True, False]
    pp.runpp(net, run_control=True)
    assert net.res_bus.vm_pu.at[1] < vmb4
    assert net.res_sgen.q_mvar.at[0] < 0
    cosphi_expected = 0.9
    q_expected = ((net.res_sgen.p_mw.at[0]/cosphi_expected)**2 - net.res_sgen.p_mw.at[0]**2)**0.5
    assert np.isclose(net.res_sgen.q_mvar.at[0], -q_expected, atol=1e-5)

    # pf with 2nd controller (should have same result)
    net.controller.in_service = [False, True]
    pp.runpp(net, run_control=True)
    assert net.res_bus.vm_pu.at[1] < vmb4
    assert np.isclose(net.res_sgen.q_mvar.at[0], -q_expected, atol=1e-5)


    # --- run control -> q injection is negative with cosphi=0.9 since vm is nearly 0.93
    net.ext_grid.vm_pu = 0.93
    net.sgen.q_mvar = 0.

    # pf without controller
    pp.runpp(net)
    vmb4 = net.res_bus.vm_pu.at[1]

    # pf with 1st controller
    net.controller.in_service = [True, False]
    pp.runpp(net, run_control=True)
    assert net.res_bus.vm_pu.at[1] > vmb4
    assert net.res_sgen.q_mvar.at[0] > 0
    cosphi_expected = 0.9
    q_expected = ((net.res_sgen.p_mw.at[0]/cosphi_expected)**2 - net.res_sgen.p_mw.at[0]**2)**0.5
    assert np.isclose(net.res_sgen.q_mvar.at[0], q_expected, atol=1e-5)

    # pf with 2nd controller (should have same result)
    net.controller.in_service = [False, True]
    pp.runpp(net, run_control=True)
    assert net.res_bus.vm_pu.at[1] > vmb4
    assert np.isclose(net.res_sgen.q_mvar.at[0], q_expected, atol=1e-5)


    # --- run control -> q injection is negative with cosphi is nearly 0.95 since vm is nearly 0.98
    net.ext_grid.vm_pu = 0.98
    net.sgen.q_mvar = 0.

    # pf without controller
    pp.runpp(net)
    vmb4 = net.res_bus.vm_pu.at[1]

    # pf with 1st controller
    net.controller.in_service = [True, False]
    pp.runpp(net, run_control=True)
    assert net.res_bus.vm_pu.at[1] > vmb4
    assert net.res_sgen.q_mvar.at[0] > 0
    assert 0.95 < pp.cosphi_from_pq(-net.res_sgen.p_mw.at[0], -net.res_sgen.q_mvar.at[0])[0] < 0.96

    # pf with 2nd controller (should have different result because slope is not given by cosphi
    # points but by q points)
    net.sgen.q_mvar = 0.
    net.controller.in_service = [False, True]
    pp.runpp(net, run_control=True)
    assert net.res_bus.vm_pu.at[1] > vmb4
    assert 0.2*p < net.res_sgen.q_mvar.at[0] < 0.243*p


def test_cosphi_of_p_timeseries():
    """ Test basic QModelCosphiPCurve behaviour (-> q_prio=False). """

    net = simple_test_net()
    sn = net.sgen.sn_mva.at[0]
    ts_data = pd.DataFrame({"P_0": list(range(-50, -1360, -100))+[-1400, -1425, -1450, -1475]})
    ds = pp.timeseries.DFData(ts_data)

    # Create, add output and set outputwriter
    ow = pp.timeseries.OutputWriter(net)
    ow.log_variable("res_sgen", "p_mw")
    ow.log_variable("res_sgen", "q_mvar")

    DER_no_q = pp.control.DERController(
        net, 0, data_source=ds, p_profile="P_0", profile_scale=-2e-3,
        q_model=DERModels.QModelCosphiPCurve({
            'p_points_pu': (0, 0.5, 1),
            'cosphi_points': (1, 1, 1)}))

    DER_no_q2 = pp.control.DERController(
        net, 0, data_source=ds, p_profile="P_0", profile_scale=-2e-3)

    DER_ue = pp.control.DERController(
        net, 0, data_source=ds, p_profile="P_0", profile_scale=-2e-3,
        q_model=DERModels.QModelCosphiPCurve({
            'p_points_pu': (0, 0.5, 1),
            'cosphi_points': (1, 1, -0.95)}))

    DER_ue2 = pp.control.DERController(
        net, 0, data_source=ds, p_profile="P_0", profile_scale=-2e-3,
        q_model=DERModels.QModelCosphiPCurve({
            'p_points_pu': (0, 0.2, 0.25, 0.3, 0.5, 1),
            'cosphi_points': (1, 1, 0.975, 1, 1, -0.95)}))

    DER_oe = pp.control.DERController(
        net, 0, data_source=ds, p_profile="P_0", profile_scale=-2e-3,
        q_model=DERModels.QModelCosphiPCurve({
            'p_points_pu': (0, 0.5, 1),
            'cosphi_points': (1, 1, 0.95)}))

    # Run timeseries
    net.controller["in_service"] = False
    net.controller.in_service.at[DER_ue.index] = True
    pp.timeseries.run_timeseries(net, time_steps=range(len(ts_data)))
    res_ue = deepcopy(ow.output)
    net.controller["in_service"] = False
    net.controller.in_service.at[DER_ue2.index] = True
    pp.timeseries.run_timeseries(net, time_steps=range(len(ts_data)))
    res_ue2 = deepcopy(ow.output)
    net.controller["in_service"] = False
    net.controller.in_service.at[DER_oe.index] = True
    pp.timeseries.run_timeseries(net, time_steps=range(len(ts_data)))
    res_oe = deepcopy(ow.output)
    net.controller["in_service"] = False
    net.controller.in_service.at[DER_no_q.index] = True
    pp.timeseries.run_timeseries(net, time_steps=range(len(ts_data)))
    res_no_q = deepcopy(ow.output)
    net.controller["in_service"] = False
    net.controller.in_service.at[DER_no_q2.index] = True
    pp.timeseries.run_timeseries(net, time_steps=range(len(ts_data)))
    res_no_q2 = deepcopy(ow.output)

    if False:  # plot cosphi course
        import matplotlib.pyplot as plt

        res_to_plot = {
            "no_q": res_no_q,
            "no_q2": res_no_q2,
            "ue": res_ue,
            "ue2": res_ue2,
            "oe": res_oe,
        }
        colors = "bgrcmyk"
        fig = plt.figure(figsize=(9, 5))
        ax = fig.gca()
        for i_key, (key, res) in enumerate(res_to_plot.items()):
            cosphi_pos_neg = pp.toolbox.cosphi_pos_neg_from_pq(
                res["res_sgen.p_mw"], res["res_sgen.q_mvar"])
            cosphi_pos_neg[np.isnan(cosphi_pos_neg[0])] = 1
            cosphi_pos = pp.toolbox.cosphi_to_pos(cosphi_pos_neg)
            x = res["res_sgen.p_mw"].values.flatten()/net.sgen.sn_mva.at[0]
            plt.plot(x, cosphi_pos, label=key, c=colors[i_key], marker="+")
        yticks = ax.get_yticks()
        yticks_signed = deepcopy(yticks)
        yticks_signed[yticks > 1] -= 2
        yticks_signed = np.round(yticks_signed, 3)
        ax.set_yticks(yticks, yticks_signed)
        plt.xlabel('p/sn')
        plt.ylabel('cosphi (negative=underexcited)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # check results
    assert np.allclose(res_no_q["res_sgen.q_mvar"].values, 0, atol=1e-5)
    assert np.allclose(res_no_q2["res_sgen.q_mvar"].values, 0, atol=1e-5)
    assert (res_ue["res_bus.vm_pu"][1] <= res_no_q["res_bus.vm_pu"][1] + 1e-8).all()
    assert (res_oe["res_bus.vm_pu"][1] + 1e-8 >= res_no_q["res_bus.vm_pu"][1]).all()
    assert (res_ue["res_sgen.q_mvar"][0] <= 1e-5).all()
    assert np.allclose(res_ue["res_sgen.q_mvar"][0],
                      -res_oe["res_sgen.q_mvar"][0], atol=1e-5)

    # diff between ue and ue2
    should_be_same = ((ts_data["P_0"]*-2e-3/sn <= 0.2) | (ts_data["P_0"]*-2e-3/sn >= 0.3)).values
    assert np.allclose(res_ue["res_sgen.q_mvar"].values[should_be_same, 0],
                       res_ue2["res_sgen.q_mvar"].values[should_be_same, 0], atol=1e-5)
    assert np.allclose(res_ue["res_sgen.q_mvar"].values[~should_be_same, 0], 0, atol=1e-5)
    assert np.all(res_ue2["res_sgen.q_mvar"].values[~should_be_same, 0] > -1e-5)


def test_QModels_with_2Dim_timeseries():
    def define_outputwriters(nets):
        ows = list()
        for net in nets:
            ow = pp.timeseries.OutputWriter(net)
            ow.log_variable("res_sgen", "p_mw")
            ow.log_variable("res_sgen", "q_mvar")
            ows.append(ow)
        return tuple(ows)

    ts_data = pd.DataFrame({"P_DER1": [2., 2.5], "P_DER2": [1.75, 2.25]})
    ds = pp.timeseries.DFData(ts_data)

    net0 = simple_test_net2()
    net1 = simple_test_net2()
    net2 = simple_test_net2()
    net3 = simple_test_net2()

    pp.control.ConstControl(net0, element="sgen", variable="p_mw", element_index=[0, 1],
                           data_source=ds, profile_name=["P_DER1", "P_DER2"])
    net0.sgen["q_mvar"] = 0.1 * net0.sgen["sn_mva"]
    pp.control.DERController(
        net1, [0, 1], data_source=ds, p_profile=["P_DER1", "P_DER2"],
        q_model=DERModels.QModelConstQ(0.1),
        pqv_area=DERModels.PQArea4105(1),
        )
    pp.control.DERController(
        net2, [0, 1], data_source=ds, profile_from_name=True,
        q_model=DERModels.QModelCosphiP(0.98),
        pqv_area=DERModels.PQArea4105(2),
        )
    pp.control.DERController(
        net3, [0, 1], data_source=ds, profile_from_name=True,
        q_model=DERModels.QModelCosphiPQ(0.98),
        pqv_area=DERModels.PQVArea4130V3(380.),
        )

    ows = define_outputwriters([net0, net1, net2, net3])
    ow0, ow1, ow2, ow3 = ows

    pp.timeseries.run_timeseries(net0, time_steps=[0, 1])
    pp.timeseries.run_timeseries(net1, time_steps=[0, 1])
    pp.timeseries.run_timeseries(net2, time_steps=[0, 1])
    pp.timeseries.run_timeseries(net3, time_steps=[0, 1])

    # --- check results
    # ow0 and ow1 are equal
    pd.testing.assert_frame_equal(ow0.output["res_bus.vm_pu"], ow1.output["res_bus.vm_pu"])
    pd.testing.assert_frame_equal(ow0.output["res_sgen.q_mvar"], ow1.output["res_sgen.q_mvar"],
                                  atol=1e-4)
    # p values are taken from input data (no PQV area is falsifies the input)
    for ow in ows:
        pd.testing.assert_frame_equal(pd.DataFrame(ts_data.values), ow.output["res_sgen.p_mw"],
                                      atol=1e-4)
    # q of ow2 is as expected
    p = ow2.output["res_sgen.p_mw"].values.reshape((-1,))
    q = ow2.output["res_sgen.q_mvar"].values.reshape((-1,))
    assert np.allclose(np.sin(np.arccos(0.98))*p, q, atol=1e-5)
    # cosphi of ow3 is correct
    p = ow3.output["res_sgen.p_mw"].values.reshape((-1,))
    q = ow3.output["res_sgen.q_mvar"].values.reshape((-1,))
    assert np.allclose(pp.toolbox.cosphi_pos_neg_from_pq(p, q), 0.98, atol=1e-6)


if __name__ == '__main__':
    if 0:
        pytest.main(['-xs', __file__])
    elif 1:
        test_qofv()
        test_cosphi_of_p_timeseries()
        test_QModels_with_2Dim_timeseries()
    pass
