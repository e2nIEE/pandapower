# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import os
import logging
import numpy as np
from pandapower.control.controller.station_control import BinarySearchControl, DroopControl
from pandapower.create import create_empty_network, create_bus, create_buses, create_ext_grid, create_transformer, \
    create_load, create_line, create_sgen, create_impedance
from pandapower.run import runpp
from pandapower.file_io import from_json
from pandapower import pp_dir

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
    bsc = BinarySearchControl(net, name="BSC1", ctrl_in_service=True,
                                         output_element="sgen", output_variable="q_mvar", output_element_index=[0],
                                         output_element_in_service=[True], output_values_distribution=[1],
                                         input_element="res_trafo", input_variable="q_hv_mvar", input_element_index=[0],
                                         set_point=1.02, voltage_ctrl=True, bus_idx=1, tol=tol)
    DroopControl(net, name="DC1", q_droop_mvar=40, bus_idx=1,
                            vm_set_pu=1.02, controller_idx=bsc.index, voltage_ctrl=True, tol = tol)
    runpp(net, run_control=False)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - 0.999648) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - (1.02 + net.res_trafo.loc[0, "q_hv_mvar"] / 40)) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].control_modus == 'V_ctrl')#test correct control_modus
    assert(net.controller.at[1, 'object'].voltage_ctrl is True)  # test correct control_modus
    assert(net.controller.at[1, 'object'].controller_idx == 0)  # test droop controller linkage


def test_qctrl():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
                                   output_element_index=[0], output_element_in_service=[True],
                                   output_values_distribution=[1], input_element="res_line",
                                   damping_factor=0.9, input_variable=["q_to_mvar"],
                                   input_element_index=0, set_point=1, voltage_ctrl=False, tol=1e-6)
    runpp(net, run_control=False)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (-6.092016e-12)) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - 1.0) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].control_modus == 'Q_ctrl')  # test correct control_modus


def test_qctrl_Imp_Input():
    net = simple_test_net()
    tol = 1e-6
    create_impedance(net, 1, 2, sn_mva=1, rft_pu=0.01, xft_pu=0.01, rtf_pu=0.01, xtf_pu=0.01)
    BinarySearchControl(net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
                                   output_element_index=[0], output_element_in_service=[True],
                                   output_values_distribution=[1], input_element="res_impedance",
                                   damping_factor=0.9, input_variable=["q_to_mvar"],
                                   input_element_index=0, set_point=1, voltage_ctrl=False, tol=1e-6)
    runpp(net, run_control=False)
    assert (abs(net.res_impedance.loc[0, "q_to_mvar"] - 0.01373636) < tol)
    runpp(net, run_control=True)
    assert (abs(net.res_impedance.loc[0, "q_to_mvar"] - 1.0) < tol)


def test_qctrl_droop():
    net = simple_test_net()
    tol = 1e-6
    net.load.loc[0, "p_mw"] = 60  # create voltage drop at bus 1
    bsc = BinarySearchControl(net, name="BSC1", ctrl_in_service=True,
                                         output_element="sgen", output_variable="q_mvar", output_element_index=[0],
                                         output_element_in_service=[True], output_values_distribution=[1],
                                         input_element="res_line", damping_factor=0.9, input_variable=["q_to_mvar"],
                                         input_inverted=True, input_element_index=0, set_point=1, voltage_ctrl=False, tol=1e-6)
    DroopControl(net, name="DC1", q_droop_mvar=40, bus_idx=1,
                            vm_set_pu=1, vm_set_ub=1.005, vm_set_lb=0.995,
                            controller_idx=bsc.index, voltage_ctrl=False)
    runpp(net, run_control=False)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (-7.094325e-13)) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (1 + (0.995 - net.res_bus.loc[1, "vm_pu"]) * 40)) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].control_modus == 'Q_ctrl')  # test correct control_modus
    assert(net.controller.at[1, 'object'].voltage_ctrl is False)  # test correct control_modus
    assert(net.controller.at[1, 'object'].controller_idx == 0)  # test droop controller linkage

def test_qlimits_qctrl():
    net = simple_test_net()
    tol = 1e-6
    net.sgen['min_q_mvar'] = -0.5
    net.sgen['max_q_mvar'] = 0.5

    BinarySearchControl(net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
                                   output_element_index=[0], output_element_in_service=[True],
                                   output_values_distribution=[1], input_element="res_line", damping_factor=0.9,
                                   input_variable=["q_to_mvar"], input_element_index=0, set_point=1,
                                   voltage_ctrl=False, tol=1e-6)
    runpp(net, run_control=True, enforce_q_lims=True)
    assert (abs(net.res_sgen.loc[0, "q_mvar"] - 0.5) < tol)

    net = simple_test_net()
    tol = 1e-6
    net.sgen['min_q_mvar'] = -0.5
    net.sgen['max_q_mvar'] = 0.5

    create_load(net, bus=net.sgen.loc[0, 'bus'], p_mw=0, q_mvar=-2)
    BinarySearchControl(net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
                                   output_element_index=[0], output_element_in_service=[True],
                                   output_values_distribution=[1], input_element="res_line", damping_factor=0.9,
                                   input_variable=["q_to_mvar"], input_element_index=0, set_point=1,
                                   voltage_ctrl=False, tol=1e-6)
    runpp(net, run_control=True, enforce_q_lims=True)
    assert (abs(net.res_sgen.loc[0, "q_mvar"] + 0.5) < tol)

def test_qlimits_voltctrl():
    net = simple_test_net()
    tol = 1e-6
    net.sgen['min_q_mvar'] = -0.7
    net.sgen['max_q_mvar'] = 0.7

    BinarySearchControl(net, name="BSC1", ctrl_in_service=True,
                                   output_element="sgen", output_variable="q_mvar", output_element_index=[0],
                                   output_element_in_service=[True], output_values_distribution=[1],
                                   input_element="res_bus", input_variable="vm_pu", input_element_index=[1],
                                   set_point=1.02, voltage_ctrl=True, tol=tol)
    runpp(net, run_control=True, enforce_q_lims=True)
    assert (abs(net.res_sgen.loc[0, "q_mvar"] - 0.7) < tol)

    net = simple_test_net()
    tol = 1e-6
    net.sgen['min_q_mvar'] = -0.7
    net.sgen['max_q_mvar'] = 0.7
    BinarySearchControl(net, name="BSC1", ctrl_in_service=True,
                                   output_element="sgen", output_variable="q_mvar", output_element_index=[0],
                                   output_element_in_service=[True], output_values_distribution=[1],
                                   input_element="res_bus", input_variable="vm_pu", input_element_index=[1],
                                   set_point=.98, voltage_ctrl=True, tol=tol)
    runpp(net, run_control=True, enforce_q_lims=True)
    assert (abs(net.res_sgen.loc[0, "q_mvar"] + 0.7) < tol)
    net.sgen.min_q_mvar = -0.8 # tests change of min_q_mvar afterwards
    runpp(net, run_control=True, enforce_q_lims=True)
    assert (abs(net.res_sgen.loc[0, "q_mvar"] + 0.8) < tol)

def test_station_ctrl_pf_import():
    path = os.path.join(pp_dir, 'test', 'control', 'testfiles', 'stactrl_test.json')
    net = from_json(path)
    tol = 1e-6
    runpp(net, run_control=True)
    print("\n")
    print("--------------------------------------")
    print("Scenario 1 - Constant Q")
    print("Controlled line 0 to, expected constQ = -0.86 MVar for q_from_mvar and constQ = 0.5 MVar for q_to_mvar: \n",
          net.res_line.loc[0, "q_from_mvar"], "\t", net.res_line.loc[0, "q_to_mvar"])
    print("Controlled line 1 to, expected constQ = -0.86 MVar for q_from_mvar and constQ = 0.5 MVar for q_to_mvar: \n",
          net.res_line.loc[2, "q_from_mvar"], "\t", net.res_line.loc[2, "q_to_mvar"])
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - 0.5) < tol)
    assert(abs(net.res_line.loc[2, "q_to_mvar"] - 0.5) < tol)
    assert(net.controller.at[0, 'object'].control_modus == 'Q_ctrl')  # test correct control_modus
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
    assert(net.controller.at[4, 'object'].voltage_ctrl is True)  # test correct droop control_modus
    assert(net.controller.at[3, 'object'].control_modus == 'V_ctrl')  # test correct control_modus
    assert(net.controller.at[4, 'object'].controller_idx == 3)  # test droop controller linkage
    print("--------------------------------------")
    print("Scenario 3 - Constant V")
    print("Controlled bus, set point = 1.03 pu, vm_pu: ", net.res_bus.loc[84, "vm_pu"])
    assert(abs(net.res_bus.loc[84, "vm_pu"] - 1.03) < tol)
    assert(net.controller.at[5, 'object'].control_modus == 'V_ctrl')  # test correct control_modus
    print("--------------------------------------")
    print("Scenario 4 - Q(U) - droop 40 MVar/pu")
    print("Input Measurement vm_pu: ", net.res_bus.loc[91, "vm_pu"])
    print("Controlled Transformer Q, lower voltage band 0.999 pu, initial set point 1 MVar and 40 MVar/pu, q_hv_mvar, "
          "expected: \n -(1 MVar + (0.999 pu  - 0.99585 pu) * 40 MVar/pu)= -1.12618: \n",
          net.res_trafo.loc[3, "q_hv_mvar"])
    assert(abs(net.res_trafo.loc[3, "q_hv_mvar"] - (1 + (0.999 - net.res_bus.loc[91, "vm_pu"]) * 40)) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[2, 'object'].voltage_ctrl is False)  # test correct droop control_modus
    assert(net.controller.at[1, 'object'].control_modus == 'Q_ctrl')  # test correct control_modus
    assert(net.controller.at[2, 'object'].controller_idx == 1) #test droop controller linkage

### Testing after rework of station controller###

def test_volt_ctrl_new():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, ctrl_in_service=True,
                                   output_element="sgen", output_variable="q_mvar", output_element_index=0,
                                   output_element_in_service=True, output_values_distribution=1,
                                   output_distribution_values = 2,
                                   input_element="res_bus", input_variable="vm_pu", input_element_index=1,
                                   set_point=1.02,control_modus='V_ctrl', tol=tol, bus_idx = 1)
    runpp(net, run_control=False)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - 0.999648) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - 1.02) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].control_modus == 'V_ctrl')  # test correct control_modus


def test_volt_ctrl_droop_new():
    net = simple_test_net()
    tol = 1e-6
    bsc = BinarySearchControl(net, ctrl_in_service=True,
                                         output_element="sgen", output_variable="q_mvar", output_element_index=0,
                                         output_element_in_service=True, output_values_distribution=1,
                                         input_element="res_trafo", input_variable="q_hv_mvar", input_element_index=0,
                                         set_point=1.02,control_modus = 'V_ctrl', tol=tol, bus_idx =1)
    DroopControl(net, q_droop_mvar=40, controller_idx=bsc.index, voltage_ctrl = True, input_element_q_meas='res_trafo',
                 input_variable_q_meas='q_hv_mvar', bus_idx=1, vm_set_pu_bsc = 1.02, tol = tol)
    runpp(net, run_control=False)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - 0.999648) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - (1.02 + net.res_trafo.loc[0, "q_hv_mvar"] / 40)) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].control_modus == 'V_ctrl')  # test correct control_modus
    assert(net.controller.at[1, 'object'].voltage_ctrl is True)  # test correct control_modus
    assert(net.controller.at[1, 'object'].controller_idx == 0)  # test droop controller linkage


def test_qctrl_new():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
                                   output_element_index=0, output_element_in_service=True,
                                   output_values_distribution=1, input_element="res_line",
                                   damping_factor=0.9, input_variable=["q_to_mvar"], output_distribution_values= [0.2, 0.3],
                                   input_element_index=0, set_point=1,control_modus = 'Q_ctrl', tol=1e-6)
    runpp(net, run_control=False)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (-6.092016e-12)) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - 1.0) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].control_modus == 'Q_ctrl')  # test correct control_modus


def test_qctrl_droop_new():
    net = simple_test_net()
    tol = 1e-6
    net.load.loc[0, "p_mw"] = 60  # create voltage drop at bus 1
    bsc = BinarySearchControl(net, ctrl_in_service=True,
                                         output_element="sgen", output_variable="q_mvar", output_element_index=0,
                                         output_element_in_service=True, output_values_distribution=1,
                                         input_element="res_line", damping_factor=0.9, input_variable="q_to_mvar",
                                         input_element_index=0, set_point=1,control_modus = 'Q_ctrl', tol=1e-6)
    DroopControl(net, q_droop_mvar=40, bus_idx=1,
                 vm_set_pu=1, vm_set_ub=1.005, vm_set_lb=0.995,
                 controller_idx=bsc.index, voltage_ctrl=False)
    runpp(net, run_control=False)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (-7.094325e-13)) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (1 + (0.995 - net.res_bus.loc[1, "vm_pu"]) * 40)) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].control_modus == 'Q_ctrl')  # test correct control_modus
    assert(net.controller.at[1, 'object'].voltage_ctrl == False)  # test correct control_modus
    assert(net.controller.at[1, 'object'].controller_idx == 0)  # test droop controller linkage

def test_pf_control_cap():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, ctrl_in_service=True, output_element='sgen', output_variable='q_mvar',
                                         output_element_index=0, output_values_distribution=1,
                                         input_element='res_line', output_element_in_service=True,
                                         damping_factor = 0.9, input_variable='q_to_mvar', input_element_index=0,
                                         set_point = 0.7, tol = 1e-6,control_modus = 'PF_ctrl_cap',
                                         output_distribution_values=[1, 0.9, 1.1])
    runpp(net, run_control=False)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"] / net.res_line.loc[0, 'p_to_mw']) + 0.7953988 - np.arccos(0.7)) < tol)
    runpp(net, run_control = True)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"]/net.res_line.loc[0, 'p_to_mw']) - - np.arccos(0.7)) < tol)#negative cause capacitive
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].control_modus == 'PF_ctrl')  # test correct control_modus


def test_pf_control_ind():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, ctrl_in_service=True, output_element='sgen', output_variable='q_mvar',
                                         output_element_index=0, output_values_distribution=1,
                                         input_element='res_line', output_element_in_service=True,
                                         damping_factor = 0.9, input_variable='q_to_mvar', input_element_index=0,
                                         set_point = 0.7, tol = 1e-6,control_modus = 'PF_ctrl_ind',
                                         output_distribution_values=[1, 0.9, 1.1])
    runpp(net, run_control=False)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"] / net.res_line.loc[0, 'p_to_mw']) + 0.7953988 - np.arccos(0.7)) < tol)
    runpp(net, run_control = True)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"]/net.res_line.loc[0, 'p_to_mw']) - np.arccos(0.7)) < tol)#positive = inductive
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].control_modus == 'PF_ctrl')  # test correct control_modus

def test_tan_phi_control():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, ctrl_in_service= True, output_element='sgen', output_variable='q_mvar',
                         output_element_index= 0, output_element_in_service= True, output_values_distribution=1,
                         input_element='res_trafo', input_variable='q_lv_mvar', input_element_index=0, control_modus='tan(phi)_ctrl',
                                         tol = 1e-6, set_point=2)
    runpp(net, run_control=False)
    assert(abs(net.res_trafo.loc[0, "q_lv_mvar"] / net.res_trafo.loc[0, 'p_lv_mw'] - 0.097382) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_trafo.loc[0, "q_lv_mvar"] / net.res_trafo.loc[0, 'p_lv_mw'] - 2) < tol)
    assert(all(net.controller.at[i, 'object'].converged) for i in net.controller.index)
    assert(net.controller.at[0, 'object'].control_modus == 'tan(phi)_ctrl')  # test correct control_modus

if __name__ == '__main__':
    pytest.main(['-s', __file__])
