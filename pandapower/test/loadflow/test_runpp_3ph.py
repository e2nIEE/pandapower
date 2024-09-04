# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:06:25 2018
Tests 3 phase power flow algorithm
@author: sghosh
"""
import pandapower as pp
import numpy as np
import pytest

import pandapower.toolbox
from pandapower.test.loadflow.PF_Results import get_PF_Results
from pandapower.test.consistency_checks import runpp_3ph_with_consistency_checks, runpp_with_consistency_checks
import os


@pytest.fixture
def net():
    v_base = 110              # 110kV Base Voltage
    k_va_base = 100         # 100 MVA
#    I_base = (kVA_base/V_base) * 1e-3           # in kA
    net = pp.create_empty_network(sn_mva=k_va_base)
    pp.create_bus(net, vn_kv=v_base, index=1)
    pp.create_bus(net, vn_kv=v_base, index=5)
    pp.create_ext_grid(net, bus=1, vm_pu=1.0, s_sc_max_mva=5000, rx_max=0.1,
                       r0x0_max=0.1, x0x_max=1.0)
    pp.create_std_type(net, {"r0_ohm_per_km": 0.0848, "x0_ohm_per_km": 0.4649556,
                             "c0_nf_per_km": 230.6, "g0_us_per_km": 0, "max_i_ka": 0.963, "r_ohm_per_km": 0.0212,
                             "x_ohm_per_km": 0.1162389, "c_nf_per_km":  230},
                       "example_type")
    pp.create_line(net, from_bus=1, to_bus=5, length_km=50.0, std_type="example_type")

    pp.create_asymmetric_load(net, 5, p_a_mw=50, q_a_mvar=50, p_b_mw=10, q_b_mvar=15,
                              p_c_mw=10, q_c_mvar=5)
    return net


def check_it(net):
    bus_pp = np.abs(net.res_bus_3ph[['vm_a_pu', 'vm_b_pu', 'vm_c_pu']]
                    [~np.isnan(net.res_bus_3ph.vm_a_pu)].values)
    bus_pf = np.abs(np.array([[0.96742893, 1.01302766, 1.019784],
                             [0.74957533, 1.09137945, 1.05124282]]))
    assert np.max(np.abs(bus_pp - bus_pf)) < 1.1e-6
    line_pp = np.abs(net.res_line_3ph[~np.isnan(net.res_line_3ph.i_a_from_ka)]
                     [['i_a_from_ka', 'i_a_to_ka', 'i_b_from_ka', 'i_b_to_ka',
                       'i_c_from_ka', 'i_c_to_ka',
                       'p_a_from_mw', 'p_a_to_mw', 'q_a_from_mvar', 'q_a_to_mvar',
                       'p_b_from_mw', 'p_b_to_mw', 'q_b_from_mvar', 'q_b_to_mvar',
                       'p_c_from_mw', 'p_c_to_mw', 'q_c_from_mvar', 'q_c_to_mvar',
                       'loading_a_percent', 'loading_b_percent', 'loading_c_percent',
                       'loading_percent'
                       ]].values)
    line_pf = np.abs(np.array(
                        [[1.34212045, 1.48537916, 0.13715552, 0.26009611,
                          0.22838401, 0.1674634,
                          55.70772301, (-49.999992954), 60.797262682, (-49.999959283),
                          8.7799379802, (-9.9999996625), (-0.88093549983), (-15.000000238),
                          9.3739293122, (-10.000000161), (-11.441663679), (-4.9999997418),
                          154.2452,  27.00894,  23.71589,
                          154.2452]]))
    assert np.max(np.abs(line_pp - line_pf)) < 1.1e-4


def test_2bus_network(net):
    # -o---o
    pp.add_zero_impedance_parameters(net)
    runpp_3ph_with_consistency_checks(net)
    assert net['converged']
    check_it(net)


def test_2bus_network_single_isolated_busses(net):
    # -o---o o x
    pp.create_bus(net, vn_kv=110)
    pp.create_bus(net, vn_kv=110, in_service=False)
    pp.add_zero_impedance_parameters(net)
    runpp_3ph_with_consistency_checks(net)
    assert net['converged']
    check_it(net)


def test_2bus_network_isolated_net_part(net):
    # -o---o o---o
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=50.0, std_type="example_type")
    pp.create_asymmetric_load(net, b2, p_a_mw=50, q_a_mvar=50, p_b_mw=10, q_b_mvar=15,
                              p_c_mw=10, q_c_mvar=5)
    pp.add_zero_impedance_parameters(net)
    runpp_3ph_with_consistency_checks(net)
    assert net['converged']
    check_it(net)


def test_2bus_network_singel_oos_bus(net):
    # -o---x---o
    b1 = pp.create_bus(net, vn_kv=110)
    net.bus.loc[5, "in_service"] = False
    pp.create_line(net, from_bus=5, to_bus=b1, length_km=10.0, std_type="example_type")
    pp.create_asymmetric_load(net, b1, p_a_mw=-5, q_a_mvar=5, p_b_mw=-1, q_b_mvar=1.5,
                              p_c_mw=-1, q_c_mvar=.5)
    pp.add_zero_impedance_parameters(net)
    runpp_3ph_with_consistency_checks(net)
    assert net['converged']


def test_out_serv_load(net):
    # <-x--o------o
    pp.add_zero_impedance_parameters(net)
    runpp_3ph_with_consistency_checks(net)
    assert net['converged']
    check_it(net)
    pp.create_asymmetric_load(net, 5, p_a_mw=50, q_a_mvar=100, p_b_mw=29, q_b_mvar=38,
                              p_c_mw=10, q_c_mvar=5, in_service=False)
    runpp_3ph_with_consistency_checks(net)
    assert net['converged']
    check_it(net)


def test_2bus_network_one_of_two_ext_grids_oos(net):
    """ Out of service ext_grids should not affect the unbalanced powerflow"""
    pp.create_ext_grid(net, bus=1, vm_pu=1.0, s_sc_max_mva=5000, rx_max=0.1,
                       r0x0_max=0.1, x0x_max=1.0, in_service=False)
    pp.runpp_3ph(net)
    assert net['converged']


@pytest.mark.parametrize("init", ["auto", "results", "flat", "dc"])
@pytest.mark.parametrize("recycle", [None, {"bus_pq": True, "Ybus": True}, {"bus_pq": True, "Ybus": False}])
def test_4bus_network(init, recycle):
    v_base = 110                     # 110kV Base Voltage
    mva_base = 100                      # 100 MVA
    net = pp.create_empty_network(sn_mva=mva_base)
    # =============================================================================
    # Main Program
    # =============================================================================
    busn = pp.create_bus(net, vn_kv=v_base, name="busn")
    busk = pp.create_bus(net, vn_kv=v_base, name="busk")
    busm = pp.create_bus(net, vn_kv=v_base, name="busm")
    busp = pp.create_bus(net, vn_kv=v_base, name="busp")
    pp.create_ext_grid(net, bus=busn, vm_pu=1.0, name="Grid Connection", s_sc_max_mva=5000,
                       rx_max=0.1, r0x0_max=0.1, x0x_max=1.0)
    pp.create_std_type(net, {"r0_ohm_per_km": .154, "x0_ohm_per_km": 0.5277876,
                             "c0_nf_per_km": 170.4, "g0_us_per_km": 0, "max_i_ka": 0.741,
                             "r_ohm_per_km": .0385, "x_ohm_per_km": 0.1319469,
                             "c_nf_per_km": 170}, "example_type3")
    pp.create_line(net, from_bus=busn, to_bus=busm, length_km=1.0, std_type="example_type3")
    pp.create_line(net, from_bus=busn, to_bus=busp, length_km=1.0, std_type="example_type3")
    pp.create_line_from_parameters(net, from_bus=busn, to_bus=busk, length_km=1.0, r0_ohm_per_km=.1005,
                                   x0_ohm_per_km=0.4900884, c0_nf_per_km=200.5, max_i_ka=0.89,
                                   r_ohm_per_km=.0251, x_ohm_per_km=0.1225221, c_nf_per_km=210)
    pp.create_line_from_parameters(net, from_bus=busk, to_bus=busm, length_km=1.0,
                                   r0_ohm_per_km=0.0848, x0_ohm_per_km=0.4649556, c0_nf_per_km=230.6,
                                   max_i_ka=0.963, r_ohm_per_km=0.0212, x_ohm_per_km=0.1162389, c_nf_per_km=230)
    pp.create_line_from_parameters(net, from_bus=busk, to_bus=busp, length_km=1.0, r0_ohm_per_km=.3048,
                                   x0_ohm_per_km=0.6031856, c0_nf_per_km=140.3, max_i_ka=0.531,
                                   r_ohm_per_km=.0762, x_ohm_per_km=0.1507964, c_nf_per_km=140)
    pp.add_zero_impedance_parameters(net)

    pp.create_asymmetric_load(net, busk, p_a_mw=50, q_a_mvar=20, p_b_mw=80, q_b_mvar=60,
                              p_c_mw=20, q_c_mvar=5)
    pp.create_asymmetric_load(net, busm, p_a_mw=50, q_a_mvar=50, p_b_mw=10, q_b_mvar=15,
                              p_c_mw=10, q_c_mvar=5)
    pp.create_asymmetric_load(net, busp, p_a_mw=50, q_a_mvar=20, p_b_mw=60, q_b_mvar=20,
                              p_c_mw=10, q_c_mvar=5)
    runpp_3ph_with_consistency_checks(net, init=init, recycle=recycle)
    runpp_3ph_with_consistency_checks(net, init=init, recycle=recycle)
    assert net['converged']

    bus_pp = np.abs(net.res_bus_3ph[['vm_a_pu', 'vm_b_pu', 'vm_c_pu']]
                    [~np.isnan(net.res_bus_3ph.vm_a_pu)].values)
    bus_pf = np.abs(np.array([[0.98085729, 0.97711997, 1.04353786],
                             [0.97828577, 0.97534651, 1.04470864],
                             [0.97774307, 0.97648197, 1.04421233],
                             [0.9780892, 0.97586805, 1.04471106]]))
    assert np.max(np.abs(bus_pp - bus_pf)) < 1e-8

    line_pp = np.abs(net.res_line_3ph[
            ['i_a_from_ka',  'i_b_from_ka', 'i_c_from_ka',
             'i_a_to_ka', 'i_b_to_ka', 'i_c_to_ka',
             'p_a_from_mw', 'p_b_from_mw', 'p_c_from_mw',
             'q_a_from_mvar', 'q_b_from_mvar', 'q_c_from_mvar',
             'p_a_to_mw', 'p_b_to_mw', 'p_c_to_mw',
             'q_a_to_mvar', 'q_b_to_mvar', 'q_c_to_mvar',
             'loading_a_percent', 'loading_b_percent', 'loading_c_percent',
             'loading_percent']].values)
    line_pf = np.abs(np.array(
            [[0.98898804851	,	0.68943734	,	0.19848961	,
              0.99093993	,	0.69146384	,	0.19966503	,
              49.87434308	,	33.86579548	,	12.44659879	,
              36.16562613	,	26.14426519	,	4.25746428	,
              -49.75842138	,	-33.90236497	,	-12.45155362	,
              -36.19862688	,	-26.25675246	,	-4.50384238	,
              133.730100000000	,	93.314960000000	,	26.945350000000	,
              133.730100000000],
             [0.87075816277	,	1.03463205	,	0.19072622	,
              0.87210779	,	1.03599167	,	0.19188991	,
              49.59359423	,	58.53676842	,	11.97553941	,
              21.96967200	,	26.37559958	,	4.04458873	,
              -49.47110289	,	-58.55284705	,	-11.98669516	,
              -22.07474008	,	-26.34476811	,	-4.29078447	,
              117.693400000000	,	139.809900000000	,	25.896070000000	,
              139.809900000000],
             [0.95760407055	,	1.14786582	,	0.24829126	,
              0.95975383	,	1.15028040	,	0.24975553	,
              50.87938854	,	57.53628873	,	15.54470531	,
              31.13888557	,	41.99378843	,	5.39758513	,
              -50.76249094	,	-57.56374777	,	-15.56099267	,
              -31.28560646	,	-41.99056453	,	-5.69609575	,
              107.837500000000	,	129.245000000000	,	28.062420000000	,
              129.245000000000],
             [0.21780921494 	,	0.42795803	,	0.03706412	,
              0.22229619	,	0.42603286	,	0.03771703	,
              0.23292404	,	-23.88471674	,	-2.45255095	,
              13.53037092	,	-11.49972060	,	0.17971665	,
              -0.24157862	,	23.90236497	,	2.45155361	,
              -13.80137312	,	11.25675247	,	-0.49615762	,
              23.083720000000	,	44.440090000000	,	3.916618000000	,
              44.440090000000],
             [0.03712221482	,	0.10766244	,	0.03093505	,
              0.03446871	,	0.10500386	,	0.03179428	,
              0.52956690	,	1.44846452	,	-1.98645639	,
              -2.24476446	,	-6.50971485	,	0.51637910	,
              -0.52889712	,	-1.44715295	,	1.98669515	,
              2.07474008	,	6.34476812	,	-0.70921554	,
              6.991001000000	,	20.275410000000	,	5.987624000000	,
              20.275410000000]]))
    assert np.max(np.abs(line_pp - line_pf)) < 1e-4


def test_3ph_bus_mapping_order():
    net = pp.create_empty_network()
    b2 = pp.create_bus(net, vn_kv=0.4, index=4)
    pp.create_bus(net, vn_kv=0.4, in_service=False, index=3)
    b1 = pp.create_bus(net, vn_kv=0.4, index=7)

    pp.create_ext_grid(net, b1, vm_pu=1.0, s_sc_max_mva=10, rx_max=0.1)
    net.ext_grid["x0x_max"] = 1.
    net.ext_grid["r0x0_max"] = 0.1
    pp.create_std_type(net, {"r_ohm_per_km": 0.1013, "x_ohm_per_km": 0.06911504,
                             "c_nf_per_km": 690, "g_us_per_km": 0, "max_i_ka": 0.44,
                             "c0_nf_per_km": 312.4, "g0_us_per_km": 0, "r0_ohm_per_km": 0.4053,
                             "x0_ohm_per_km": 0.2764602}, "N2XRY 3x185sm 0.6/1kV")

    pp.create_line(net, b1, b2, 1.0, std_type="N2XRY 3x185sm 0.6/1kV", index=4)
    pp.create_line(net, b1, b2, 1.0, std_type="N2XRY 3x185sm 0.6/1kV", index=3, in_service=False)
    pp.create_line(net, b1, b2, 1.0, std_type="N2XRY 3x185sm 0.6/1kV", index=7)
    pp.add_zero_impedance_parameters(net)
    pp.create_load(net, b2, p_mw=0.030, q_mvar=0.030)
    pp.runpp(net)
    runpp_3ph_with_consistency_checks(net)
    assert net['converged']

    assert np.allclose(net.res_bus_3ph.vm_a_pu.values, net.res_bus.vm_pu.values, equal_nan=True)
    assert net.res_bus_3ph.index.tolist() == net.res_bus.index.tolist()

    assert net.res_line_3ph.index.tolist() == net.res_line.index.tolist()
    assert np.allclose(net.res_line.p_from_mw, net.res_line_3ph.p_a_from_mw +
                       net.res_line_3ph.p_b_from_mw +
                       net.res_line_3ph.p_c_from_mw)
    assert np.allclose(net.res_line.loading_percent, net.res_line_3ph.loading_a_percent)


def test_3ph_two_bus_line_powerfactory():
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, vn_kv=0.4)
    b2 = pp.create_bus(net, vn_kv=0.4)

    pp.create_ext_grid(net, b1, vm_pu=1.0, s_sc_max_mva=10, rx_max=0.1)
    net.ext_grid["x0x_max"] = 1.
    net.ext_grid["r0x0_max"] = 0.1
    pp.create_std_type(net, {"r_ohm_per_km": 0.1013, "x_ohm_per_km": 0.06911504,
                             "c_nf_per_km": 690, "g_us_per_km": 0, "max_i_ka": 0.44,
                             "c0_nf_per_km": 312.4, "g0_us_per_km": 0, "r0_ohm_per_km": 0.4053,
                             "x0_ohm_per_km": 0.2764602}, "N2XRY 3x185sm 0.6/1kV")

    pp.create_line(net, b1, b2, 0.4, std_type="N2XRY 3x185sm 0.6/1kV")
    pp.add_zero_impedance_parameters(net)
    pp.create_load(net, b2, p_mw=0.010, q_mvar=0.010)
    pp.create_asymmetric_load(net, b2, p_a_mw=0.020, q_a_mvar=0.010, p_b_mw=0.015, q_b_mvar=0.005, p_c_mw=0.025,
                              q_c_mvar=0.010)
    runpp_3ph_with_consistency_checks(net)
    assert net['converged']

    bus_pp = np.abs(net.res_bus_3ph[['vm_a_pu', 'vm_b_pu', 'vm_c_pu']].values)
    bus_pf = np.abs(np.array([[0.99939853552, 1.0013885141, 0.99921580141],
                             [0.97401782343, 0.98945593737, 0.96329605983]]))

    assert np.max(np.abs(bus_pp-bus_pf)) < 4e-6

    line_pp = np.abs(net.res_line_3ph[
            ['i_a_from_ka', 'i_b_from_ka', 'i_c_from_ka',
             'i_a_to_ka', 'i_b_to_ka', 'i_c_to_ka',
             'p_a_from_mw', 'p_b_from_mw', 'p_c_from_mw',
             'q_a_from_mvar', 'q_b_from_mvar', 'q_c_from_mvar',
             'p_a_to_mw', 'p_b_to_mw', 'p_c_to_mw',
             'q_a_to_mvar', 'q_b_to_mvar', 'q_c_to_mvar']].values)
    line_pf = np.abs(np.array(
            [[0.11946088987	,	0.08812337783	,	0.14074226065	,
             0.1194708224	,	0.088131567331	,	0.14075063601	,
             0.023810539354	,	0.01855791658	,	0.029375192747	,
             0.013901720672	,	0.008421814704	,	0.013852398586	,
             -0.023333142958	,	-0.018333405987	,	-0.028331643666	,
             -0.013332756527	,	-0.008333413919	,	-0.013332422725	]]))
    assert np.max(np.abs(line_pp - line_pf)) < 1e-5

    line_load_pp = np.abs(net.res_line_3ph[
            ['loading_a_percent', 'loading_b_percent', 'loading_c_percent',
             'loading_percent']].values)
    line_load_pf = np.abs(np.array(
                          [[27.1525	,	20.0299	,	31.98878	,
                            31.98878]]))
    assert np.max(np.abs(line_load_pp - line_load_pf)) < 1e-2


def check_bus_voltages(net, result, trafo_vector_group):
    res_vm_pu = []
    ordered_bus_table = net.bus.sort_values(by='name').reset_index(drop=True)
    for i in range(len(ordered_bus_table)):
        index = net.bus[net.bus.name == ordered_bus_table['name'][i]].index[0]
        res_vm_pu.append(net.res_bus_3ph.vm_a_pu[index])
        res_vm_pu.append(net.res_bus_3ph.vm_b_pu[index])
        res_vm_pu.append(net.res_bus_3ph.vm_c_pu[index])

    # max_tol = 0
    # for tol in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    #     if np.allclose(result, res_vm_pu, atol=tol):
    #         max_tol = tol
    # print('Voltage Magnitude for %s is within tolerance of %s' % (trafo_vector_group, max_tol))

    tolerances = {'YNyn': 1e-05,
                  'Dyn': 1e-05,
                  'Yzn': 1e-03}

    assert np.allclose(result, res_vm_pu, atol=tolerances[trafo_vector_group], rtol=0), f"Incorrect results for {trafo_vector_group}"


def check_line_currents(net, result, trafo_vector_group):
    res_line_i_ka = []
    ordered_line_table = net.line.sort_values(by='name').reset_index(drop=True)
    for i in range(len(ordered_line_table)):
        index = net.line[net.line.name == ordered_line_table['name'][i]].index[0]
        res_line_i_ka.append(net.res_line_3ph.i_a_from_ka[index])
        res_line_i_ka.append(net.res_line_3ph.i_b_from_ka[index])
        res_line_i_ka.append(net.res_line_3ph.i_c_from_ka[index])
        res_line_i_ka.append(net.res_line_3ph.i_a_to_ka[index])
        res_line_i_ka.append(net.res_line_3ph.i_b_to_ka[index])
        res_line_i_ka.append(net.res_line_3ph.i_c_to_ka[index])

    # max_tol = 0
    # for tol in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    #     if np.allclose(result, res_line_i_ka, atol=tol):
    #         max_tol = tol
    # print('Line current for %s is within tolerance of %s' % (trafo_vector_group, max_tol))

    tolerances = {'YNyn': 1e-07,
                  'Dyn': 1e-07,
                  'Yzn': 1e-04}

    assert np.allclose(result, res_line_i_ka, atol=tolerances[trafo_vector_group])
    if not np.allclose(result, res_line_i_ka, atol=tolerances[trafo_vector_group]):
        raise ValueError("Incorrect results for vector group %s" % trafo_vector_group, res_line_i_ka, result)


def check_trafo_currents(net, result, trafo_vector_group):
    res_trafo_i_ka = []
    ordered_trafo_table = net.trafo.sort_values(by='name').reset_index(drop=True)
    for i in range(len(ordered_trafo_table)):
        index = net.trafo[net.trafo.name == ordered_trafo_table['name'][i]].index[0]
        res_trafo_i_ka.append(net.res_trafo_3ph.i_a_hv_ka[index])
        res_trafo_i_ka.append(net.res_trafo_3ph.i_b_hv_ka[index])
        res_trafo_i_ka.append(net.res_trafo_3ph.i_c_hv_ka[index])
        res_trafo_i_ka.append(net.res_trafo_3ph.i_a_lv_ka[index])
        res_trafo_i_ka.append(net.res_trafo_3ph.i_b_lv_ka[index])
        res_trafo_i_ka.append(net.res_trafo_3ph.i_c_lv_ka[index])

    # max_tol = 0
    # for tol in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    #     if np.allclose(result, res_trafo_i_ka, atol=tol):
    #         max_tol = tol
    # print('Trafo current for %s is within tolerance of %s' % (trafo_vector_group, max_tol))

    tolerances = {'YNyn': 1e-03,
                  'Dyn': 1e-03,
                  'Yzn': 1e-03}

    assert np.allclose(result, res_trafo_i_ka, atol=tolerances[trafo_vector_group])
    if not np.allclose(result, res_trafo_i_ka, atol=tolerances[trafo_vector_group]):
        raise ValueError("Incorrect results for vector group %s" % trafo_vector_group, res_trafo_i_ka, result)


def check_results(net, trafo_vector_group, results):
    check_bus_voltages(net, results[0], trafo_vector_group)
    check_line_currents(net, results[1], trafo_vector_group)
    check_trafo_currents(net, results[2], trafo_vector_group)


def make_nw(net, bushv, tap_ps, case, vector_group):
    b1 = pp.create_bus(net, bushv, zone=vector_group, index=pp.get_free_id(net.bus))
    b2 = pp.create_bus(net, 0.4, zone=vector_group)
    b3 = pp.create_bus(net, 0.4, zone=vector_group)
    pp.create_ext_grid(net, b1, s_sc_max_mva=10000,
                       rx_max=0.1, r0x0_max=0.1, x0x_max=1.0)
    pp.create_transformer_from_parameters(net, hv_bus=b1, lv_bus=b2,
                                          sn_mva=1.6, vn_hv_kv=10,
                                          vn_lv_kv=0.4, vk_percent=6,
                                          vkr_percent=0.78125, pfe_kw=2.7,
                                          i0_percent=0.16875, shift_degree=0,
                                          tap_side='lv', tap_neutral=0,
                                          tap_min=-2, tap_max=2,
                                          tap_step_degree=0,
                                          tap_step_percent=2.5,
                                          tap_phase_shifter=False,
                                          vk0_percent=6, vkr0_percent=0.78125,
                                          mag0_percent=100, mag0_rx=0.,
                                          si0_hv_partial=0.9, vector_group=vector_group,
                                          parallel=1, tap_pos=tap_ps,
                                          index=pp.get_free_id(net.trafo)+1)
    pp.create_line_from_parameters(net, b2, b3, length_km=0.5, r_ohm_per_km=0.1941, x_ohm_per_km=0.07476991,
                                   c_nf_per_km=1160., max_i_ka=0.421,
                                   endtemp_degree=70.0, r0_ohm_per_km=0.7766,
                                   x0_ohm_per_km=0.2990796,
                                   c0_nf_per_km=496.2,
                                   index=pp.get_free_id(net.line)+1)
    if case == "bal_wye":
        # Symmetric Load
        pp.create_load(net, b3, 0.08, 0.012, type='wye')
    elif case == "delta_wye":
        # Unsymmetric Light Load
        pp.create_asymmetric_load(net, b3, p_a_mw=0.0044, q_a_mvar=0.0013, p_b_mw=0.0044, q_b_mvar=0.0013,
                                  p_c_mw=0.0032, q_c_mvar=0.0013, type='wye')
        pp.create_asymmetric_load(net, b3, p_a_mw=0.0300, q_a_mvar=0.0048, p_b_mw=0.0280, q_b_mvar=0.0036,
                                  p_c_mw=0.027, q_c_mvar=0.0043, type='delta')

    elif case == "wye":
        # Unsymmetric Heavy Load
        pp.create_asymmetric_load(net, b3, p_a_mw=0.0300, q_a_mvar=0.0048, p_b_mw=0.0280, q_b_mvar=0.0036,
                                  p_c_mw=0.027, q_c_mvar=0.0043, type=case)
    elif case == "delta":
        pp.create_asymmetric_load(net, b3, p_a_mw=0.0300, q_a_mvar=0.0048, p_b_mw=0.0280, q_b_mvar=0.0036,
                                  p_c_mw=0.027, q_c_mvar=0.0043, type=case)

#    pp.add_zero_impedance_parameters(net) Not required here since added through parameters


def test_trafo_asym():
    nw_dir = os.path.abspath(os.path.join(pp.pp_dir, "test/loadflow"))
    # only 3 vector groups are supported in the 3ph power flow
    for trafo_vector_group in ["YNyn", "Dyn", "Yzn"]:
        net = pp.from_json(nw_dir + '/runpp_3ph Validation.json')
        net['trafo'].vector_group = trafo_vector_group
        runpp_3ph_with_consistency_checks(net)
        assert net['converged']
        check_results(net, trafo_vector_group, get_PF_Results(trafo_vector_group))


def test_2trafos():
    net = pp.create_empty_network()
    make_nw(net, 10., 0., "wye", "YNyn")
    make_nw(net, 10., 0., "wye", "YNyn")
    runpp_3ph_with_consistency_checks(net)
    assert net['converged']
    assert np.allclose(net.res_ext_grid_3ph.iloc[0].values, net.res_ext_grid_3ph.iloc[1].values)


def test_3ph_isolated_nodes():
    v_base = 110  # 110kV Base Voltage
    mva_base = 100  # 100 MVA
    net = pp.create_empty_network(sn_mva=mva_base)

    busn = pp.create_bus(net, vn_kv=v_base, name="busn", index=1)
    pp.create_bus(net, vn_kv=20., in_service=True, index=2, name="busx")
    busk = pp.create_bus(net, vn_kv=v_base, name="busk", index=5)
    busl = pp.create_bus(net, vn_kv=v_base, name="busl", index=6)
    pp.create_bus(net, vn_kv=20., in_service=False, index=3)
    busy = pp.create_bus(net, vn_kv=20., in_service=True, index=0, name="busy")

    pp.create_ext_grid(net, bus=busn, vm_pu=1.0, name="Grid Connection",
                       s_sc_max_mva=5000, rx_max=0.1)
    net.ext_grid["r0x0_max"] = 0.1
    net.ext_grid["x0x_max"] = 1.0
    pp.create_std_type(net, {"r0_ohm_per_km": 0.0848, "x0_ohm_per_km": 0.4649556,
                             "c0_nf_per_km": 230.6, "g0_us_per_km": 0, "max_i_ka": 0.963,
                             "r_ohm_per_km": 0.0212, "x_ohm_per_km": 0.1162389,
                             "c_nf_per_km": 230}, "example_type")
    # Loads on supplied buses
    pp.create_asymmetric_load(net, busk, p_a_mw=50, q_a_mvar=50, p_b_mw=10, q_b_mvar=15,
                              p_c_mw=10, q_c_mvar=5)
    pp.create_load(net, bus=busl, p_mw=7, q_mvar=0.070, name="Load 1")
    # Loads on unsupplied buses
    pp.create_load(net, bus=busy, p_mw=70, q_mvar=70, name="Load Y")
    pp.create_line(net, from_bus=busn, to_bus=busk, length_km=50.0, std_type="example_type")
    pp.create_line(net, from_bus=busl, to_bus=busk, length_km=50.0, std_type="example_type")
    pp.add_zero_impedance_parameters(net)
    runpp_3ph_with_consistency_checks(net)
    assert net['converged']
    assert np.allclose(net.res_bus_3ph.T[[0, 2, 3]].T[["vm_a_pu", "va_a_degree", "vm_b_pu",
                       "va_b_degree", "vm_c_pu", "va_c_degree"]], np.nan, equal_nan=True)
    assert np.allclose(net.res_bus_3ph.T[[0, 2, 3]].T[["p_a_mw", "q_a_mvar", "p_b_mw", "q_b_mvar",
                       "p_c_mw", "q_c_mvar"]], 0.0)


def test_balanced_power_flow_with_unbalanced_loads_and_sgens():
    net = pp.create_empty_network(sn_mva=100)
    make_nw(net, 10, 0, "wye", "Dyn")
    pp.create_asymmetric_sgen(net, 1, p_a_mw=0.01, p_b_mw=0.02, scaling=0.8)
    runpp_with_consistency_checks(net)

    vm_pu = net.res_bus.vm_pu

    net.asymmetric_load.in_service = False
    pp.create_load(net, bus=net.asymmetric_load.bus.iloc[0],
                   scaling=net.asymmetric_load.scaling.iloc[0],
                   p_mw=net.asymmetric_load.loc[0, ["p_a_mw", "p_b_mw", "p_c_mw"]].sum(),
                   q_mvar=net.asymmetric_load.loc[0, ["q_a_mvar", "q_b_mvar", "q_c_mvar"]].sum()
                   )
    runpp_with_consistency_checks(net)
    assert net.res_bus.vm_pu.equals(vm_pu)

    net.asymmetric_sgen.in_service = False
    pp.create_sgen(net, bus=net.asymmetric_sgen.bus.iloc[0],
                   scaling=net.asymmetric_sgen.scaling.iloc[0],
                   p_mw=net.asymmetric_sgen.loc[0, ["p_a_mw", "p_b_mw", "p_c_mw"]].sum(),
                   q_mvar=net.asymmetric_sgen.loc[0, ["q_a_mvar", "q_b_mvar", "q_c_mvar"]].sum()
                   )
    runpp_with_consistency_checks(net)
    assert net.res_bus.vm_pu.equals(vm_pu)


def test_3ph_with_impedance():
    nw_dir = os.path.abspath(os.path.join(pp.pp_dir, "test/loadflow"))
    net = pp.from_json(nw_dir + '/runpp_3ph Validation.json')
    net.line.c_nf_per_km = 0.
    net.line.g_nf_per_km = 0.
    net.line["c0_nf_per_km"] = 0.
    net.line["g0_us_per_km"] = 0.
    net_imp = net.deepcopy()
    pp.replace_line_by_impedance(net_imp, net.line.index, 100)
    pp.runpp_3ph(net)
    pp.runpp_3ph(net_imp)
    assert pandapower.toolbox.dataframes_equal(net.res_bus_3ph, net_imp.res_bus_3ph)


def test_shunt_3ph():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 20.0)
    b2 = pp.create_bus(net, 20.0)
    pp.create_ext_grid(net, b1, s_sc_max_mva=1000, rx_max=0.1, x0x_max=1.0, r0x0_max=0.1)
    pp.create_line_from_parameters(net, b1, b2, length_km=1, r_ohm_per_km=1, x_ohm_per_km=1, c_nf_per_km=1,
                                   r0_ohm_per_km=1, x0_ohm_per_km=1, c0_nf_per_km=1, max_i_ka=1)
    pp.create_shunt(net, in_service=True, bus=b2, p_mw=1, q_mvar=1)
    pp.runpp_3ph(net)

    bus_cols = ["vm_a_pu", "va_a_degree", "vm_b_pu", "va_b_degree", "vm_c_pu", "va_c_degree"]
    bus_pp = np.abs(net.res_bus_3ph[bus_cols].values)
    bus_expected = np.abs(np.array([[1.0, 0.0, 1.0, -120.0, 1.0, 120.0],
                                    [0.9950250311521571, -8.955222949441059e-06, 0.995025031152157, -120.00000895522294,
                                     0.995025031152157, 119.99999104477705]]))

    assert np.max(np.abs(bus_pp - bus_expected)) < 1e-5

    line_cols = ["p_a_from_mw", "q_a_from_mvar", "p_b_from_mw", "q_b_from_mvar", "q_c_from_mvar", "p_a_to_mw",
                 "q_a_to_mvar", "p_b_to_mw", "q_b_to_mvar", "p_c_to_mw", "q_c_to_mvar"]
    line_power_pp = np.abs(net.res_line_3ph[line_cols].values)
    line_power_expected = np.abs(np.array(
        [[0.33167495789351165, 0.33163327786948577, 0.33167495789351165, 0.33163327786948565, 0.33163327786948565,
          -0.3300249368894127, -0.3300249368948133, -0.3300249368894127, -0.3300249368948133, -0.3300249368894127,
          -0.3300249368948134]]))
    assert np.max(np.abs(line_power_pp - line_power_expected)) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

