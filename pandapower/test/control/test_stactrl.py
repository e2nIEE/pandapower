# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import os
import numpy as np
from pandapower.control.controller.station_control import BinarySearchControl, DroopControl
from pandapower.create import create_empty_network, create_bus, create_buses, create_ext_grid, create_transformer, \
    create_load, create_line, create_sgen
from pandapower.run import runpp
from pandapower.file_io import from_json
from pandapower import pp_dir

try:
    from pandaplan.core import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def simple_test_net():
    net = create_empty_network()
    create_bus(net, 110)
    create_buses(net, 2, 20)
    create_ext_grid(net, 0)
    create_transformer(net, 0, 1, "63 MVA 110/20 kV")
    create_load(net, 1, 3, 0.1)
    create_sgen(net, 2, p_mw=2., sn_mva=10, name="sgen1")
    create_line(net, 1, 2, length_km=0.1, std_type="NAYY 4x50 SE")
    return net

def distribution_test_net():
    net = create_empty_network()
    create_bus(net, 110, index = 0)
    create_buses(net, 4, 20)
    create_ext_grid(net, 0)
    create_transformer(net, 0, 1, "63 MVA 110/20 kV")
    create_transformer(net, 0, 3, std_type='63 MVA 110/20 kV')
    create_load(net, 1, 3, 5)
    create_load(net, 3, 3)
    create_sgen(net, 2, p_mw=2, sn_mva=10, name="sgen1")
    create_sgen(net, 4, p_mw=1, sn_mva=5, name='sgen2')
    create_sgen(net, 4,1, sn_mva=5, name = 'sgen3')
    create_line(net, 1, 2, length_km=0.1, std_type="NAYY 4x50 SE")
    create_line(net, 3, 4, length_km=0.2, std_type= 'NAYY 4x50 SE')
    return net


def test_volt_ctrl():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, name="BSC1", ctrl_in_service=True,
                                   output_element="sgen", output_variable="q_mvar", output_element_index=[0],
                                   output_element_in_service=[True], output_values_distribution=[1],
                                   input_element="res_bus", input_variable="vm_pu", input_element_index=[1],
                                   set_point=1.02, voltage_ctrl=True, tol=tol)
    runpp(net, run_control=False)
    assert (abs(net.res_bus.loc[1, "vm_pu"] - 0.999648) < tol)
    runpp(net, run_control=True)
    assert (abs(net.res_bus.loc[1, "vm_pu"] - 1.02) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)


def test_volt_ctrl_droop():
    net = simple_test_net()
    tol = 1e-6
    bsc = BinarySearchControl(net, ctrl_in_service=True,
                                         output_element="sgen", output_variable="q_mvar", output_element_index=[0],
                                         output_element_in_service=[True], output_values_distribution=['rel_P'],
                                         input_element="res_trafo", input_variable="q_hv_mvar", input_element_index=[0],
                                         set_point=1.02, voltage_ctrl=True, bus_idx=1, tol=tol)
    DroopControl(net, name="DC1", q_droop_mvar=40, bus_idx=1,
                            vm_set_pu=1.02, controller_idx=bsc.index, voltage_ctrl=True)
    runpp(net, run_control=False)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - 0.999648) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - (1.02 + net.res_trafo.loc[0, "q_hv_mvar"] / 40)) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].modus == 'V_ctrl')#test correct modus
    assert(net.controller.at[1, 'object'].modus == 'V_ctrl')  # test correct modus
    assert(net.controller.at[1, 'object'].controller_idx == 0)  # test droop controller linkage


def test_qctrl():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
                                   output_element_index=[0], output_element_in_service=[True],
                                   output_values_distribution='rel_rated_S', input_element="res_line",
                                   damping_factor=0.9, input_variable=["q_to_mvar"],
                                   input_element_index=0, set_point=1, voltage_ctrl=False, tol=1e-6)
    runpp(net, run_control=False)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (-6.092016e-12)) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - 1.0) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].modus == 'Q_ctrl')  # test correct modus


def test_qctrl_droop():
    net = simple_test_net()
    tol = 1e-6
    net.load.loc[0, "p_mw"] = 60  # create voltage drop at bus 1
    bsc = BinarySearchControl(net, name="BSC1", ctrl_in_service=True,
                                         output_element="sgen", output_variable="q_mvar", output_element_index=[0],
                                         output_element_in_service=[True], output_values_distribution='set_Q',
                                         input_element="res_line", damping_factor=0.9, input_variable=["q_to_mvar"],
                                         input_element_index=0, set_point=1, voltage_ctrl=False, tol=1e-6)
    DroopControl(net, name="DC1", q_droop_mvar=40, bus_idx=1,
                            vm_set_pu=1, vm_set_ub=1.005, vm_set_lb=0.995,
                            controller_idx=bsc.index, voltage_ctrl=False)
    runpp(net, run_control=False)
    assert (abs(net.res_line.loc[0, "q_to_mvar"] - (-1e-13)) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (1 + (0.995 - net.res_bus.loc[1, "vm_pu"]) * 40)) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].modus == 'Q_ctrl')  # test correct modus
    assert(net.controller.at[1, 'object'].modus == 'Q_ctrl')  # test correct modus
    assert(net.controller.at[1, 'object'].controller_idx == 0)  # test droop controller linkage


def test_station_ctrl_pf_import():
    path = os.path.join(pp_dir, 'test', 'control', 'testfiles', 'station_ctrl_test.json')
    net = from_json(path)
    tol = 1e-6
    runpp(net, run_control=True)
    print("\n")
    print("--------------------------------------")
    print("Scenario 1 - Constant Q")
    print("Controlled line 0 to, expected constQ = -0.86 MVar for q_from_mvar and constQ = 0.5 MVar for q_to_mvar: \n",
          net.res_line.loc[0, "q_from_mvar"], "\t", net.res_line.loc[0, "q_to_mvar"])
    print("Controlled line 1 to, expected constQ = -0.86 MVar for q_from_mvar and constQ = 0.5 MVar for q_to_mvar: \n",
          net.res_line.loc[1, "q_from_mvar"], "\t", net.res_line.loc[1, "q_to_mvar"])
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - 0.5) < tol)
    assert(abs(net.res_line.loc[1, "q_to_mvar"] - 0.5) < tol)
    assert(net.controller.at[1, 'object'].modus == 'Q_ctrl')  # test correct modus
    print("--------------------------------------")
    print("Scenario 2 - Constant V, droop 40 MVar/pu")
    print("Input Measurement line 4 q_from_mvar and q_to_mvar, expected: \n -0.6215 MVar \t 0.2442 MVar \n",
          net.res_line.loc[4, "q_from_mvar"], "\t", net.res_line.loc[4, "q_to_mvar"])
    print("Input Measurement line 5 q_from_mvar and q_to_mvar, expected:\n -0.6215 MVar \t 0.2442 MVar \n",
          net.res_line.loc[5, "q_from_mvar"], "\t", net.res_line.loc[5, "q_to_mvar"])
    print("Controlled bus, initial set point 1.01 pu and 40 MVar/pu, vm_pu, \n expected: "
          "2 * 0.2442 MVar / 40 MVar/pu + 1.01 pu = 1.02221: \n", net.res_bus.loc[62, "vm_pu"])
    assert(abs(net.res_bus.loc[62, "vm_pu"] - (1.01 + ((net.res_line.loc[4, "q_to_mvar"] +
                                             net.res_line.loc[5, "q_to_mvar"]) /
                                            40))) < tol)  # still not close enough, increased tolerance
    assert(net.controller.at[2, 'object'].modus == 'V_ctrl')  # test correct modus
    assert(net.controller.at[3, 'object'].modus == 'V_ctrl')  # test correct modus
    assert(net.controller.at[3, 'object'].controller_idx == 2)  # test droop controller linkage
    print("--------------------------------------")
    print("Scenario 3 - Constant V")
    print("Controlled bus, set point = 1.03 pu, vm_pu: ", net.res_bus.loc[84, "vm_pu"])
    assert(abs(net.res_bus.loc[84, "vm_pu"] - 1.03) < tol)
    assert(net.controller.at[0, 'object'].modus == 'V_ctrl')  # test correct modus
    print("--------------------------------------")
    print("Scenario 4 - Q(U) - droop 40 MVar/pu")
    print("Input Measurement vm_pu: ", net.res_bus.loc[103, "vm_pu"])
    print("Controlled Transformer Q, lower voltage band 0.999 pu, initial set point 1 MVar and 40 MVar/pu, q_hv_mvar, "
          "expected: \n -(1 MVar + (0.999 pu  - 0.995846 pu) * 40 MVar/pu) = -1.126176: \n",
          net.res_trafo.loc[3, "q_hv_mvar"])
    assert(abs(net.res_trafo.loc[3, "q_hv_mvar"] - (-(1 + (0.999 - net.res_bus.loc[103, "vm_pu"]) * 40))) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[4, 'object'].modus == 'Q_ctrl')  # test correct modus
    assert(net.controller.at[5, 'object'].modus == 'Q_ctrl')  # test correct modus
    assert(net.controller.at[5, 'object'].controller_idx == 4) #test droop controller linkage

### Testing after rework of station controller###

def test_volt_ctrl_new():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, ctrl_in_service=True,
                                   output_element="sgen", output_variable="q_mvar", output_element_index=0,
                                   output_element_in_service=True, output_values_distribution='rel_P',
                                   output_distribution_values = 2,
                                   input_element="res_bus", input_variable="vm_pu", input_element_index=1,
                                   set_point=1.02, modus='V_ctrl', tol=tol, bus_idx = 1)
    runpp(net, run_control=False)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - 0.999648) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - 1.02) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].modus == 'V_ctrl')  # test correct modus


def test_volt_ctrl_droop_new():
    net = simple_test_net()
    tol = 1e-6
    bsc = BinarySearchControl(net, ctrl_in_service=True,
                                         output_element="sgen", output_variable="q_mvar", output_element_index=0,
                                         output_element_in_service=True, output_values_distribution='rel_rated_S',
                                         input_element="res_bus", input_variable="vm_pu", input_element_index=1,
                                         set_point=1.02, modus = 'V_ctrl', tol=tol)
    DroopControl(net, q_droop_mvar=40, controller_idx=bsc.index, modus = 'V_ctrl', input_element_q_meas='res_trafo',
                 input_variable_q_meas='q_hv_mvar', input_element_index_q_meas=0)
    runpp(net, run_control=False)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - 0.999648) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - (1.02 + net.res_trafo.loc[0, "q_hv_mvar"] / 40)) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].modus == 'V_ctrl')  # test correct modus
    assert(net.controller.at[1, 'object'].modus == 'V_ctrl')  # test correct modus
    assert(net.controller.at[1, 'object'].controller_idx == 0)  # test droop controller linkage


def test_qctrl_new():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
                                   output_element_index=0, output_element_in_service=True,
                                   output_values_distribution='set_Q', input_element="res_line",
                                   damping_factor=0.9, input_variable=["q_to_mvar"], output_distribution_values= [0.2, 0.3],
                                   input_element_index=0, set_point=1, modus = 'Q_ctrl', tol=1e-6)
    runpp(net, run_control=False)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (-6.092016e-12)) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - 1.0) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].modus == 'Q_ctrl')  # test correct modus


def test_qctrl_droop_new():
    net = simple_test_net()
    tol = 1e-6
    net.load.loc[0, "p_mw"] = 60  # create voltage drop at bus 1
    bsc = BinarySearchControl(net, ctrl_in_service=True,
                                         output_element="sgen", output_variable="q_mvar", output_element_index=0,
                                         output_element_in_service=True, output_values_distribution='max_Q',
                                         input_element="res_line", damping_factor=0.9, input_variable="q_to_mvar",
                                         input_element_index=0, set_point=1, modus = 'Q_ctrl', tol=1e-6)
    DroopControl(net, q_droop_mvar=40, bus_idx=1,
                            vm_set_pu=1, vm_set_ub=1.005, vm_set_lb=0.995,
                            controller_idx=bsc.index, modus = 'Q_ctrl')
    runpp(net, run_control=False)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (-7.094325e-13)) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (1 + (0.995 - net.res_bus.loc[1, "vm_pu"]) * 40)) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].modus == 'Q_ctrl')  # test correct modus
    assert(net.controller.at[1, 'object'].modus == 'Q_ctrl')  # test correct modus
    assert(net.controller.at[1, 'object'].controller_idx == 0)  # test droop controller linkage


def test_pf_control_cap():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, ctrl_in_service=True, output_element='sgen', output_variable='q_mvar',
                                         output_element_index=0, output_values_distribution='rel_V_pu',
                                         input_element='res_line', output_element_in_service=True,
                                         damping_factor = 0.9, input_variable='q_to_mvar', input_element_index=0,
                                         set_point = 0.7, tol = 1e-6, modus = 'PF_ctrl_cap',
                                         output_distribution_values=[1, 0.9, 1.1])
    runpp(net, run_control=False)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"] / net.res_line.loc[0, 'p_to_mw']) + 0.7953988 - np.arccos(0.7)) < tol)
    runpp(net, run_control = True)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"]/net.res_line.loc[0, 'p_to_mw']) - - np.arccos(0.7)) < tol)#negative cause capacitive
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].modus == 'PF_ctrl')  # test correct modus


def test_pf_control_ind():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, ctrl_in_service=True, output_element='sgen', output_variable='q_mvar',
                                         output_element_index=0, output_values_distribution='max_Q',
                                         input_element='res_line', output_element_in_service=True,
                                         damping_factor = 0.9, input_variable='q_to_mvar', input_element_index=0,
                                         set_point = 0.7, tol = 1e-6, modus = 'PF_ctrl_ind',
                                         output_distribution_values=[1, 0.9, 1.1])
    runpp(net, run_control=False)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"] / net.res_line.loc[0, 'p_to_mw']) + 0.7953988 - np.arccos(0.7)) < tol)
    runpp(net, run_control = True)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"]/net.res_line.loc[0, 'p_to_mw']) - np.arccos(0.7)) < tol)#positive = inductive
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].modus == 'PF_ctrl')  # test correct modus


def test_pf_control_droop_q():
    net = simple_test_net()
    tol = 1e-6
    bsc = BinarySearchControl(net, ctrl_in_service=True,
                                         output_element="sgen", output_variable="q_mvar", output_element_index=0,
                                         output_element_in_service=True, output_values_distribution='max_Q',
                                         input_element="res_line", damping_factor=0.9, input_variable="q_to_mvar",
                                         input_element_index=0, set_point=1, modus='PF_ctrl', tol=1e-6)
    DroopControl(net, bus_idx=1, pf_overexcited= 0.5, pf_underexcited= 0.9,
                            vm_set_pu=1, vm_set_ub=3, vm_set_lb=1,
                            controller_idx=bsc.index, modus='PF_ctrl_P')
    runpp(net, run_control=False)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"] / net.res_line.loc[0, 'p_to_mw']) - 0) < tol)
    runpp(net, run_control=True) #Phi at point = 1.0471975521726393, Q at point = 3.4641016229480206 P at point = 1.999999999976119
    m = ((1 - 0.9) + (1 - 0.5)) / (3 - 1)  # getting function #m = 0.3
    b = -(1 - 0.9) - m * 1 #-0.4
    droop_set_point = 1 - (m * net.res_line.loc[0, 'p_to_mw'] + b) #should be 0.8 #reactance positive cause droop set point > 1
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"] / net.res_line.loc[0, 'p_to_mw']) - np.arccos(droop_set_point)) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].modus == 'PF_ctrl')  # test correct modus
    assert(net.controller.at[1, 'object'].modus == 'PF_ctrl')  # test correct modus
    assert(net.controller.at[1, 'object'].controller_idx == 0)  # test droop controller linkage


def test_pf_control_droop_v():
    net = simple_test_net()
    tol = 1e-6
    bsc = BinarySearchControl(net, ctrl_in_service=True,
                                         output_element="sgen", output_variable="q_mvar", output_element_index=0,
                                         output_element_in_service=True, output_values_distribution='rel_V_pu',
                                         input_element="res_line", damping_factor=0.9, input_variable="q_to_mvar",
                                         input_element_index=0, modus='PF_ctrl', tol=1e-6, set_point=0.5)
    DroopControl(net, bus_idx=1, pf_overexcited=-0.3, pf_underexcited=0.7,
                            vm_set_pu=1, vm_set_ub=0.6, vm_set_lb=2.2,
                            controller_idx=bsc.index, modus='PF_ctrl_V')
    runpp(net, run_control=False)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"] / net.res_line.loc[0, 'p_to_mw']) - 0) < tol)
    runpp(net, run_control=True) #V at bus 1 is 1.038866702987955, Phi at point is 1.0439447685158276
    m = ((1 - 0.7) + (1 - -0.3)) / (0.6 - 2.2)  # getting function #m = -1
    b = (1 - -0.3) - m * 0.6 #b = 1.9
    droop_set_point = 1 - (m * net.res_bus.loc[1, 'vm_pu'] + b)  # should be 0.1388667029878976 #reactance positive cause droop set point > 1
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"] / net.res_line.loc[0, 'p_to_mw']) - np.arccos(droop_set_point)) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].modus == 'PF_ctrl')  # test correct modus
    assert(net.controller.at[1, 'object'].modus == 'PF_ctrl')  # test correct modus
    assert(net.controller.at[1, 'object'].controller_idx == 0)  # test droop controller linkage


def test_tan_phi_control():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, ctrl_in_service= True, output_element='sgen', output_variable='q_mvar',
                         output_element_index= 0, output_element_in_service= True, output_values_distribution='rel_P',
                         input_element='res_trafo', input_variable='q_lv_mvar', input_element_index=0, modus='tan(phi)_ctrl',
                                         tol = 1e-6, set_point=2)
    runpp(net, run_control=False)
    assert(abs(net.res_trafo.loc[0, "q_lv_mvar"] / net.res_trafo.loc[0, 'p_lv_mw'] - 0.097382) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_trafo.loc[0, "q_lv_mvar"] / net.res_trafo.loc[0, 'p_lv_mw'] - 2) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].modus == 'tan(phi)_ctrl')  # test correct modus


def test_station_ctrl_pf_import_new():
    path = os.path.join(pp_dir, 'test', 'control', 'testfiles', 'station_ctrl_test_new.json')
    net = from_json(path)
    tol = 1e-6
    runpp(net, run_control=True)
    print("\n")
    print("--------------------------------------")
    print("Scenario 1 - Constant Q")
    print("Controlled line 0, constQ = -0.86 MVar for q_from_mvar and constQ = 0.5 MVar for q_to_mvar: \n",
          net.res_line.loc[0, "q_from_mvar"], "\t", net.res_line.loc[0, "q_to_mvar"])
    print("Controlled line 15, constQ = -0.86 MVar for q_from_mvar and constQ = 0.5 MVar for q_to_mvar: \n",
          net.res_line.loc[15, "q_from_mvar"], "\t", net.res_line.loc[15, "q_to_mvar"])
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - 0.5) < tol)
    assert(abs(net.res_line.loc[15, "q_to_mvar"] - 0.5) < tol)
    assert(net.controller.at[0, 'object'].modus == 'Q_ctrl')  # test correct modus
    print("--------------------------------------")
    print("Scenario 2 - Constant V, droop 40 MVar/pu")
    print("Input Measurement q_from_mvar and q_to_mvar, expected: \n -0.6215 MVar \t 0.2442 MVar \n",
          net.res_line.loc[3, "q_from_mvar"], "\t", net.res_line.loc[3, "q_to_mvar"])
    print("Input Measurement q_from_mvar and q_to_mvar, expected:\n -0.6215 MVar \t 0.2442 MVar \n",
          net.res_line.loc[4, "q_from_mvar"], "\t", net.res_line.loc[4, "q_to_mvar"])
    print("Controlled bus, initial set point 1.01 pu and 40 MVar/pu, vm_pu, \n expected: "
          "2 * 0.2442 MVar / 40 MVar/pu + 1.01 pu = 1.02221: \n", net.res_bus.loc[86, "vm_pu"])
    assert(abs(net.res_bus.loc[86, "vm_pu"] - (1.01 + ((net.res_line.loc[3, "q_to_mvar"] +
                                             net.res_line.loc[4, "q_to_mvar"])) / 40)) < tol)
    assert(net.controller.at[3, 'object'].modus == 'V_ctrl')  # test correct modus
    assert(net.controller.at[4, 'object'].modus == 'V_ctrl')  # test correct modus
    assert(net.controller.at[4, 'object'].controller_idx == 3)  # test droop controller linkage
    print("--------------------------------------")
    print("Scenario 3 - Constant V")
    print("Controlled bus, set point = 1.03 pu \n vm_pu: ", net.res_bus.loc[108, "vm_pu"])
    assert(abs(net.res_bus.loc[108, "vm_pu"] - 1.03) < tol)
    assert(net.controller.at[5, 'object'].modus == 'V_ctrl')  # test correct modus
    print("--------------------------------------")
    print("Scenario 4 - Q(U) - droop 40 MVar/pu")
    print("Input Measurement vm_pu: ", net.res_bus.loc[127, "vm_pu"])
    print("Controlled Transformer Q, lower voltage band 0.999 pu, initial set point 1 MVar and 40 MVar/pu, q_hv_mvar, "
          "expected: \n -(1 MVar + (0.999 pu  - 0.99585 pu) * 40 MVar/pu)= -1.12618 MVar: \n",
          net.res_trafo.loc[3, "q_hv_mvar"])
    assert(abs(net.res_trafo.loc[3, "q_hv_mvar"] - (-(1 + (0.999 - net.res_bus.loc[127, "vm_pu"]) * 40))) < tol)
    assert(net.controller.at[1, 'object'].modus == 'Q_ctrl')  # test correct modus
    assert(net.controller.at[2, 'object'].modus == 'Q_ctrl')  # test correct modus
    assert(net.controller.at[2, 'object'].controller_idx == 1)  # test droop controller linkage
    print("------------------------------------- ")
    print("Scenario 5 - Constant Power factor")
    print("Controlled line 16 to, expected const PF = 0.93 for Phi_from and const PF = 1 for Phi_to: \n",
          np.cos(np.arctan(net.res_line.loc[16, 'q_from_mvar'] / net.res_line.loc[16, 'p_from_mw'])), "\t",
          np.cos(np.arctan(net.res_line.loc[16, 'q_to_mvar'] / net.res_line.loc[16, 'p_to_mw'])))
    print("Controlled line 17 to, expected const PF = 0.93 for Phi_from and const PF = 1 for Phi_to: \n",
          np.cos(np.arctan(net.res_line.loc[17, 'q_from_mvar'] / net.res_line.loc[17, 'p_from_mw'])), "\t",
          np.cos(np.arctan(net.res_line.loc[17, 'q_to_mvar'] / net.res_line.loc[17, 'p_to_mw'])))
    assert(abs(np.arctan(net.res_line.loc[16, "q_to_mvar"]/net.res_line.loc[16, 'p_to_mw']) - np.arccos(1)) < tol) #positive reactance because inductive
    assert(abs(np.arctan(net.res_line.loc[17, "q_to_mvar"]/net.res_line.loc[17, 'p_to_mw']) - np.arccos(1)) < tol)
    assert(net.controller.at[6, 'object'].modus == 'PF_ctrl')  # test correct modus
    print("------------------------------------- ")
    print("Scenario 6 - PF_Phi(V)")
    print("controlled line 26 to, expected PF_phi(V) = 0.83 Phi_from, Phi_to: \n",
          np.cos(np.arctan(net.res_line.loc[26, 'q_from_mvar'] / net.res_line.loc[26, 'p_from_mw'])), "\t",
          np.cos(np.arctan(net.res_line.loc[26, 'q_to_mvar'] / net.res_line.loc[26, 'p_to_mw'])))
    print("Controlled line 27 to, expected PF_phi(V) = 0.83 Phi_from and Phi_to: \n",
          np.cos(np.arctan(net.res_line.loc[27, 'q_from_mvar'] / net.res_line.loc[27, 'p_from_mw'])), "\t",
          np.cos(np.arctan(net.res_line.loc[27, 'q_to_mvar'] / net.res_line.loc[27, 'p_to_mw'])))
    m = ((1 - 1) + (1 - 0.54)) / (1.05 - 0.95)  # getting function m = 4.6
    b = -(1 - 1) - m * 0.95 #b = -4.37
    droop_set_point = 1 - (m * net.res_bus.loc[ #should be 0.83
        192, 'vm_pu'] + b) #reactance positive cause droop set point > 1
    assert(abs(np.arctan(net.res_line.loc[26, "q_to_mvar"] / net.res_line.loc[26, 'p_to_mw']) - np.arccos(
        droop_set_point)) < tol)
    assert(abs(np.arctan(net.res_line.loc[27, "q_to_mvar"] / net.res_line.loc[27, 'p_to_mw']) - np.arccos(
        droop_set_point)) < tol)
    assert(net.controller.at[8, 'object'].modus == 'PF_ctrl')  # test correct modus
    assert(net.controller.at[9, 'object'].modus == 'PF_ctrl')  # test correct modus
    assert(net.controller.at[9, 'object'].p_cosphi == False) #correct modus
    assert(net.controller.at[9, 'object'].controller_idx == 8)  # test droop controller linkage
    print("------------------------------------- ")
    print("Scenario 7 - PF_Phi(P)")
    print("Controlled line 24 to, expected const PF = 0.65 for  Phi_from and Phi_to: \n",
          np.cos(np.arctan(net.res_line.loc[24, 'q_from_mvar'] / net.res_line.loc[24, 'p_from_mw'])), "\t",
          np.cos(np.arctan(net.res_line.loc[24, 'q_to_mvar'] / net.res_line.loc[24, 'p_to_mw'])))
    print("Controlled line 25 to, expected const PF = 0.65 for Phi_from and Phi_to: \n",
          np.cos(np.arctan(net.res_line.loc[25, 'q_from_mvar'] / net.res_line.loc[25, 'p_from_mw'])), "\t",
          np.cos(np.arctan(net.res_line.loc[25, 'q_to_mvar'] / net.res_line.loc[25, 'p_to_mw'])))
    droop_set_point = net.controller.at[11, 'object'].pf_over #should be 0.65 and positive reactance, because overexcited
    assert(abs(np.arctan(net.res_line.loc[24, "q_to_mvar"] / net.res_line.loc[24, 'p_to_mw']) - np.arccos(
        droop_set_point)) < tol)
    assert(abs(np.arctan(net.res_line.loc[25, "q_to_mvar"] / net.res_line.loc[25, 'p_to_mw']) - np.arccos(
        droop_set_point)) < tol)
    assert(net.controller.at[10, 'object'].modus == 'PF_ctrl')  # test correct modus
    assert(net.controller.at[11, 'object'].modus == 'PF_ctrl')  # test correct modus
    assert(net.controller.at[11, 'object'].p_cosphi == True)  # correct modus
    assert(net.controller.at[11, 'object'].controller_idx == 10)  # test droop controller linkage
    print("------------------------------------- ")
    print("Scenario 8 - Tan(Phi)")
    print("Controlled line 20 to, expected tan(phi) = 0.376 for tan(phi)_from and tan(phi) = 0 for tan(phi)_to: \n",
          net.res_line.loc[20, "q_from_mvar"] / net.res_line.loc[20, "p_from_mw"],
          "\t", net.res_line.loc[20, 'q_to_mvar'] / net.res_line.loc[20, 'p_to_mw'])
    print("Controlled line 21 to, expected tan(phi) = 0.376 for tan(phi)_from and tan(phi) = 0 for tan(phi)_to: \n",
          net.res_line.loc[21, "q_from_mvar"] / net.res_line.loc[21, "p_from_mw"],
          "\t", net.res_line.loc[21, 'q_to_mvar'] / net.res_line.loc[21, 'p_to_mw'])
    assert(abs(net.res_line.loc[20, "q_to_mvar"] / net.res_line.loc[20, 'p_to_mw'] - 0) < tol)
    assert(abs(net.res_line.loc[21, "q_to_mvar"] / net.res_line.loc[21, 'p_to_mw']  - 0) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[7, 'object'].modus == 'tan(phi)_ctrl')  # test correct modus

### Testing the distributions###

def test_q_relative_to_p_dist():
    net = distribution_test_net()
    tol = 1e-6
    BinarySearchControl(net, True, 'sgen', 'q_mvar',
                        [0,1], [True, True], 'res_bus',
                        'vm_pu', 4, 1, 'rel_P',
                        None, 'V_ctrl', 1e-6)
    runpp(net, run_control = False)
    assert(net.sgen.at[0, 'q_mvar'] == net.sgen.at[1, 'q_mvar'])
    runpp(net, run_control = True)
    assert(net.sgen.at[0, 'q_mvar'] != net.sgen.at[1, 'q_mvar'])
    assert(abs(net.sgen.at[0, 'q_mvar']/(net.sgen.at[0, 'q_mvar'] + net.sgen.at[1, 'q_mvar']) - net.sgen.at[0, 'p_mw']/(
        net.sgen.at[0, 'p_mw'] + net.sgen.at[1, 'p_mw'])) < tol)
    assert(abs(net.sgen.at[1, 'q_mvar'] / (net.sgen.at[0, 'q_mvar'] + net.sgen.at[1, 'q_mvar'])-net.sgen.at[1, 'p_mw']/(
        net.sgen.at[0, 'p_mw'] + net.sgen.at[1, 'p_mw'])) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].modus == 'V_ctrl')  # test correct modus


def test_q_relative_to_rated_s_dist(): #rated p is not implemented and defaults to 50 MVar => 50/50
    net = distribution_test_net()
    BinarySearchControl(net, True, 'sgen', 'q_mvar',
                        [0,1], [True, True], 'res_line',
                        'q_to_mvar', 0, 4, 'rel_rated_S',
                        None, 'Q_ctrl', 1e-6)
    runpp(net, run_control = False)
    assert(net.sgen.at[0, 'q_mvar'] == net.sgen.at[1, 'q_mvar']) #distribution is 50/50
    assert((net.sgen.at[0, 'q_mvar'] + 1) / net.sgen.at[0, 'sn_mva'] != #plus one because Q_sgen is 0
           (net.sgen.at[1, 'q_mvar'] + 1) / net.sgen.at[1, 'sn_mva']) #should not be equal, because unregulated
    runpp(net, run_control = True)
    assert (net.sgen.at[0, 'q_mvar'] != net.sgen.at[1, 'q_mvar']) #not equal anymore, but the relative values are equal
    assert(net.sgen.at[0, 'q_mvar'] != 0 and net.sgen.at[1, 'q_mvar'] != 0) #prove that not 0 divided by values
    assert(net.sgen.at[0, 'q_mvar'] / net.sgen.at[0, 'sn_mva'] == net.sgen.at[1, 'q_mvar'] / net.sgen.at[1, 'sn_mva'])
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].modus == 'Q_ctrl')  # test correct modus

def test_set_q_dist():
    net = distribution_test_net()
    tol = 1e-6
    BinarySearchControl(net, True, 'sgen', 'q_mvar',
                        [0, 1], [True, True], 'res_line',
                        'q_to_mvar', 0, 0.6, 'set_Q',
                        [0.5, 0.8], 'PF_ctrl_ind', 1e-6)
    runpp(net, run_control=False)
    assert(net.sgen.at[0, 'q_mvar'] == net.sgen.at[1, 'q_mvar'])
    runpp(net, run_control=True)
    assert(net.sgen.at[0, 'q_mvar'] != net.sgen.at[1, 'q_mvar'])
    assert(abs(net.sgen.at[0, 'q_mvar'] / (net.sgen.at[0, 'q_mvar'] + net.sgen.at[1, 'q_mvar']) - 0.5 / (0.5 + 0.8)) < tol)
    assert(abs(net.sgen.at[1, 'q_mvar'] / (net.sgen.at[0, 'q_mvar'] + net.sgen.at[1, 'q_mvar']) - 0.8 / (0.5 + 0.8)) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].modus == 'PF_ctrl')  # test correct modus

def test_max_q():
    net = distribution_test_net()
    BinarySearchControl(net, True, 'sgen', 'q_mvar',
                        [0,1], [True, True], 'res_line',
                        'q_to_mvar', 0, 0.2, 'max_Q',
                        None, 'PF_ctrl_cap', 1e-6)
    runpp(net, run_control = False)
    assert(net.sgen.at[0, 'q_mvar'] == net.sgen.at[1, 'q_mvar'])
    runpp(net, run_control = True)
    assert(net.sgen.at[0, 'q_mvar'] == net.sgen.at[1, 'q_mvar'])#check internal error handling
    net = distribution_test_net() #recall net to test other functions
    net.sgen.at[0, 'min_q_mvar'] = -20 #setting necessary parameters
    net.sgen.at[0, 'max_q_mvar'] = 50
    net.sgen.at[1, 'min_q_mvar'] = -7
    net.sgen.at[1, 'max_q_mvar'] = 20
    BinarySearchControl(net, True, 'sgen', 'q_mvar',
                        [0, 1], [True, True], 'res_line',
                        'q_to_mvar', 0, 0.5, 'max_Q',
                        None, 'PF_ctrl_cap',  1e-6)
    runpp(net, run_control = True)
    assert(net.sgen.at[0, 'q_mvar'] != net.sgen.at[1, 'q_mvar'])
    net = distribution_test_net()#testing generators at limit
    net.sgen.at[0, 'min_q_mvar'] = -20 #lowest
    net.sgen.at[0, 'max_q_mvar'] = 0 #least high
    net.sgen.at[1, 'min_q_mvar'] = -7 #second lowest
    net.sgen.at[1, 'max_q_mvar'] = 20 #highest
    net.sgen.at[2, 'min_q_mvar'] = -6 #second highest
    net.sgen.at[2, 'max_q_mvar'] = 4 # least low
    idx_neg, idx_pos = [2, 1, 0], [0, 2, 1] #correct orders
    BinarySearchControl(net, True, 'sgen', 'q_mvar',
                        [0, 1, 2], [True, True, True], 'res_line',
                        'q_to_mvar', 0, 0.2, 'max_Q',
                        None, 'PF_ctrl_cap',  1e-6)
    runpp(net, run_control = True)
    #checking if control worked
    assert(net.sgen.at[0, 'q_mvar'] != net.sgen.at[1, 'q_mvar'] != net.sgen.at[2, 'q_mvar'])
    #checking if Q output order coincides with set Q limits
    all_sgens = np.array([abs(net.sgen.at[0, 'q_mvar']), abs(net.sgen.at[1, 'q_mvar']), abs(net.sgen.at[2, 'q_mvar'])])
    idx = np.argsort(all_sgens)
    assert(np.array_equal(idx, idx_pos) or np.array_equal(idx,idx_neg)) #correct order for set values + and -
    assert(abs(net.sgen.at[idx_neg[0], 'q_mvar']) < abs(net.sgen.at[idx_neg[1], 'q_mvar']) < #redundant
           abs(net.sgen.at[idx_neg[2], 'q_mvar']) or abs(net.sgen.at[idx_pos[0], 'q_mvar']) <
           abs(net.sgen.at[idx_pos[1], 'q_mvar']) < abs(net.sgen.at[idx_pos[2], 'q_mvar']))
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].modus == 'PF_ctrl')  # test correct modus



def test_rel_v_pu():
    net = distribution_test_net()
    tol = 0.02 #voltage adaption is not very precise
    BinarySearchControl(net, True, 'sgen', 'q_mvar',
                        [0,1], [True, True], 'res_line',
                        'q_to_mvar', 0, 0.5, 'rel_V_pu',
                        [[0.98, 0.95, 1.1], [0.89, 0.8, 1.3]], 'tan(phi)_ctrl', 1e-6)
    runpp(net, run_control = False)
    assert(net.sgen.at[0, 'q_mvar'] == net.sgen.at[1, 'q_mvar'] == net.sgen.at[2, 'q_mvar']) #sgens are the same
    assert(abs(net.res_bus.at[net.sgen.at[0, 'bus'], 'vm_pu'] + net.res_bus.at[net.sgen.at[1, 'bus'], 'vm_pu']
           - 0.98 - 0.89) > tol) #uncontrolled buses are not at V set points
    with pytest.raises(NotImplementedError):
        runpp(net, run_control=True) #test if sgens at same busbar are detected
    net = distribution_test_net()
    BinarySearchControl(net, True, 'sgen', 'q_mvar',
                        [0, 1], [True, True], 'res_line',
                        'q_to_mvar', 0, 0.5, 'rel_V_pu',
                        [[0.98, 0.95, 1.1], [0.89, 0.8, 1.3]], 'tan(phi)_ctrl', 1e-6)
    net.sgen.drop(2, inplace=True) #delete interfering sgen
    runpp(net, run_control= True)
    assert(net.sgen.at[0, 'q_mvar'] != net.sgen.at[1, 'q_mvar']) #now controlled sgens
    assert(abs(net.res_bus.at[net.sgen.at[0, 'bus'], 'vm_pu'] + net.res_bus.at[net.sgen.at[1, 'bus'], 'vm_pu']
                - 0.98 - 0.89) < tol) #now within set points
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].modus == 'tan(phi)_ctrl')  # test correct modus

def test_station_ctrl_pf_import_distributions():#test comparability between PF and pp
    path = os.path.join(pp_dir, 'test', 'control', 'testfiles', 'station_ctrl_test_distributions.json')
    net = from_json(path)
    tol = 5e-6
    tol_v = 2e-3 #smaller tolerance for voltage set point adaptation rel_V_pu and max_Q
    runpp(net, run_control=True)
    assert(all(abs(np.array(net.sgen.loc[net.controller.at[0, 'object'].output_element_index, 'q_mvar']) -
                [0.06333, 0.33249]) < tol)) #set_Q
    assert(all(abs(np.array(net.sgen.loc[net.controller.at[1, 'object'].output_element_index, 'q_mvar']) -
                [0.63910, 0.35675]) < tol)) #rel_rated_S
    assert(all(abs(np.array(net.sgen.loc[net.controller.at[2, 'object'].output_element_index, 'q_mvar']) -
                [0.62056, 1.24112]) < tol)) #rel_P
    assert(all(abs(np.array(net.sgen.loc[net.controller.at[3, 'object'].output_element_index, 'q_mvar']) -
                [6.77276, -9.79795, -0.89898]) < tol_v)) #max_Q
    assert(all(abs(np.array(net.sgen.loc[net.controller.at[4, 'object'].output_element_index, 'q_mvar']) -
                [-31.23760, 1]) < tol_v)) #rel_V_pu Q_vals
    assert(all(abs(np.array(net.res_bus.loc[net.sgen.loc[net.controller.at[4, 'object'].output_element_index].bus, 'vm_pu']) -
                [0.89847, 0.98847 ]) < tol_v)) #rel_V_pu busbar voltage
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].modus == 'Q_ctrl')  # test correct modus
    assert(net.controller.at[1, 'object'].modus == 'tan(phi)_ctrl')  # test correct modus
    assert(net.controller.at[2, 'object'].modus == 'PF_ctrl')  # test correct modus
    assert(net.controller.at[3, 'object'].modus == 'PF_ctrl')  # test correct modus
    assert(net.controller.at[4, 'object'].modus == 'tan(phi)_ctrl')  # test correct modus


if __name__ == '__main__':
    pytest.main(['-s', __file__])