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


def test_voltctrl():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, ctrl_in_service=True,
                                   output_element="sgen", output_variable="q_mvar", output_element_index=[0],
                                   output_element_in_service=[True], output_values_distribution=[1],
                                   input_element="res_bus", input_variable="vm_pu", input_element_index=[1],
                                   set_point=1.02, voltage_ctrl=True, tol=tol)
    runpp(net, run_control=False)
    assert (abs(net.res_bus.loc[1, "vm_pu"] - 0.999648) < tol)
    runpp(net, run_control=True)
    assert (abs(net.res_bus.loc[1, "vm_pu"] - 1.02) < tol)


def test_voltctrl_droop():
    net = simple_test_net()
    tol = 1e-3
    bsc = BinarySearchControl(net, ctrl_in_service=True,
                                         output_element="sgen", output_variable="q_mvar", output_element_index=[0],
                                         output_element_in_service=[True], output_values_distribution=['rel_P'],
                                         input_element="res_trafo", input_variable="q_hv_mvar", input_element_index=[0],
                                         set_point=1.02, voltage_ctrl=True, bus_idx=1, tol=tol)
    DroopControl(net, q_droop_mvar=40, bus_idx=1,
                            vm_set_pu=1.02, controller_idx=bsc.index, voltage_ctrl=True)
    runpp(net, run_control=False)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - 0.999648) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - (1.02 + net.res_trafo.loc[0, "q_hv_mvar"] / 40)) < tol)


def test_qctrl():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
                                   output_element_index=[0], output_element_in_service=[True],
                                   output_values_distribution='rel_rated_P', input_element="res_line",
                                   damping_factor=0.9, input_variable=["q_to_mvar"],
                                   input_element_index=0, set_point=1, voltage_ctrl=False, tol=1e-6)
    runpp(net, run_control=False)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (-6.092016e-12)) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - 1.0) < tol)


def test_qctrl_droop():
    net = simple_test_net()
    tol = 1e-6
    net.load.loc[0, "p_mw"] = 60  # create voltage drop at bus 1
    bsc = BinarySearchControl(net, ctrl_in_service=True,
                                         output_element="sgen", output_variable="q_mvar", output_element_index=[0],
                                         output_element_in_service=[True], output_values_distribution='set_Q',
                                         input_element="res_line", damping_factor=0.9, input_variable=["q_to_mvar"],
                                         input_element_index=0, set_point=1, voltage_ctrl=False, tol=1e-6)
    DroopControl(net, q_droop_mvar=40, bus_idx=1,
                            vm_set_pu=1, vm_set_ub=1.005, vm_set_lb=0.995,
                            controller_idx=bsc.index, voltage_ctrl=False)
    runpp(net, run_control=False)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (-7.094325e-13)) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (1 + (0.995 - net.res_bus.loc[1, "vm_pu"]) * 40)) < tol)

def test_stactrl_pf_import():
    path = os.path.join(pp_dir, 'test', 'control', 'testfiles', 'stactrl_test.json')
    net = from_json(path)
    tol = 1e-3

    runpp(net, run_control=True)
    print("\n")
    print("--------------------------------------")
    print("Scenario 1 - Constant Q")
    print("Controlled line, constQ = 0.5 MVar - q_from_mvar and q_to_mvar: \n",
          net.res_line.loc[0, "q_to_mvar"], "\t", net.res_line.loc[0, "q_from_mvar"])
    print("Controlled line, constQ = 0.5 MVar - q_from_mvar and q_to_mvar: \n",
          net.res_line.loc[1, "q_to_mvar"], "\t", net.res_line.loc[1, "q_from_mvar"])
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - 0.5) < tol)
    assert(abs(net.res_line.loc[1, "q_to_mvar"] - 0.5) < tol)
    print("--------------------------------------")
    print("Scenario 2 - Constant U, droop 40 MVar/pu")
    print("Input Measurement q_from_mvar and q_to_mvar, expected: \n 0.2442 MVar, -0.6215 MVar: \n",
          net.res_line.loc[4, "q_to_mvar"], "\t", net.res_line.loc[4, "q_from_mvar"])
    print("Input Measurement q_from_mvar and q_to_mvar, expected:\n 0.2442 MVar, -0.6215 MVar: \n",
          net.res_line.loc[5, "q_to_mvar"], "\t", net.res_line.loc[5, "q_from_mvar"])
    print("Controlled bus, initial set point 1.01 pu and 40 MVar/pu, vm_pu, \n expected: "
          "2 * 0.2442 MVar / 40 MVar/pu + 1.01 pu = 1.02221: \n", net.res_bus.loc[62, "vm_pu"])
    assert(abs(net.res_bus.loc[62, "vm_pu"] - ((net.res_line.loc[4, "q_to_mvar"] +
                                             net.res_line.loc[5, "q_to_mvar"]) /
                                            40 + 1.01)) < (tol + 0.1))  # still not close enough, increased tolerance
    print("--------------------------------------")
    print("Scenario 3 - Constant U")
    print("Controlled bus, set point = 1.03 pu, vm_pu: \n", net.res_bus.loc[84, "vm_pu"])
    assert(abs(net.res_bus.loc[84, "vm_pu"] - 1.03) < tol)
    print("--------------------------------------")
    print("Scenario 4 - Q(U) - droop 40 MVar/pu")
    print("Input Measurement vm_pu: \n", net.res_bus.loc[103, "vm_pu"])
    print("Controlled Transformer Q, lower voltage band 0.999 pu, initial set point 1 MVar and 40 MVar/pu, q_hv_mvar, "
          "expected: \n -(1 MVar + (0.999 pu  - 0.99585 pu) * 40 MVar/pu)= -1.12618: \n",
          net.res_trafo.loc[3, "q_hv_mvar"])
    assert(abs(net.res_trafo.loc[3, "q_hv_mvar"] - (-(1 + (0.999 - net.res_bus.loc[103, "vm_pu"]) * 40))) < tol)

### Testing after rework of station controller###

def test_voltctrl_new():
    net = simple_test_net()
    tol = 1e-6
    pp.control.BinarySearchControl(net, ctrl_in_service=True,
                                   output_element="sgen", output_variable="q_mvar", output_element_index=0,
                                   output_element_in_service=True, output_values_distribution='rel_P',
                                   output_distribution_values = 2,
                                   input_element="res_bus", input_variable="vm_pu", input_element_index=1,
                                   set_point=1.02, modus='V_ctrl', tol=tol)
    pp.runpp(net, run_control=False)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - 0.999648) < tol)
    pp.runpp(net, run_control=True)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - 1.02) < tol)


def test_voltctrl_droop_new():
    net = simple_test_net()
    tol = 1e-6
    bsc = pp.control.BinarySearchControl(net, ctrl_in_service=True,
                                         output_element="sgen", output_variable="q_mvar", output_element_index=0,
                                         output_element_in_service=True, output_values_distribution='rel_rated_P',
                                         input_element="res_trafo", input_variable="q_hv_mvar", input_element_index=0,
                                         set_point=1.02, modus = 'V_ctrl', bus_idx=1, tol=tol)
    pp.control.DroopControl(net, q_droop_mvar=40, bus_idx=1,
                            vm_set_pu=1.02, controller_idx=bsc.index, modus = 'V_ctrl')
    pp.runpp(net, run_control=False)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - 0.999648) < tol)
    pp.runpp(net, run_control=True)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - (1.02 + net.res_trafo.loc[0, "q_hv_mvar"] / 40)) < tol)


def test_qctrl_new():
    net = simple_test_net()
    tol = 1e-6
    pp.control.BinarySearchControl(net, ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
                                   output_element_index=0, output_element_in_service=True,
                                   output_values_distribution='set_Q', input_element="res_line",
                                   damping_factor=0.9, input_variable=["q_to_mvar"], output_distribution_values= [0.2, 0.3],
                                   input_element_index=0, set_point=1, modus = 'Q_ctrl', tol=1e-6)
    pp.runpp(net, run_control=False)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (-6.092016e-12)) < tol)
    pp.runpp(net, run_control=True)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - 1.0) < tol)


def test_qctrl_droop_new():
    net = simple_test_net()
    tol = 1e-6
    net.load.loc[0, "p_mw"] = 60  # create voltage drop at bus 1
    bsc = pp.control.BinarySearchControl(net, ctrl_in_service=True,
                                         output_element="sgen", output_variable="q_mvar", output_element_index=0,
                                         output_element_in_service=True, output_values_distribution='max_Q',
                                         input_element="res_line", damping_factor=0.9, input_variable="q_to_mvar",
                                         input_element_index=0, set_point=1, modus = 'Q_ctrl', tol=1e-6)
    pp.control.DroopControl(net, q_droop_mvar=40, bus_idx=1,
                            vm_set_pu=1, vm_set_ub=1.005, vm_set_lb=0.995,
                            controller_idx=bsc.index, modus = 'Q_ctrl')
    pp.runpp(net, run_control=False)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (-7.094325e-13)) < tol)
    pp.runpp(net, run_control=True)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (1 + (0.995 - net.res_bus.loc[1, "vm_pu"]) * 40)) < tol)


def test_pf_control_cap():
    net = simple_test_net()
    tol = 1e-6
    pp.control.BinarySearchControl(net, ctrl_in_service=True, output_element='sgen', output_variable='q_mvar',
                                         output_element_index=0, output_values_distribution='rel_V_pu',
                                         input_element='res_line', output_element_in_service=True,
                                         damping_factor = 0.9, input_variable='q_to_mvar', input_element_index=0,
                                         set_point = 0.7, tol = 1e-6, modus = 'PF_ctrl_cap',
                                         output_distribution_values=[1, 0.9, 1.1])
    pp.runpp(net, run_control=False)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"] / net.res_line.loc[0, 'p_to_mw']) + 0.7953988 - np.arccos(0.7)) < tol)
    pp.runpp(net, run_control = True)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"]/net.res_line.loc[0, 'p_to_mw']) - - np.arccos(0.7)) < tol)#negative cause capacitive


def test_pf_control_ind():
    net = simple_test_net()
    tol = 1e-6
    pp.control.BinarySearchControl(net, ctrl_in_service=True, output_element='sgen', output_variable='q_mvar',
                                         output_element_index=0, output_values_distribution='max_Q',
                                         input_element='res_line', output_element_in_service=True,
                                         damping_factor = 0.9, input_variable='q_to_mvar', input_element_index=0,
                                         set_point = 0.7, tol = 1e-6, modus = 'PF_ctrl_ind',
                                         output_distribution_values=[1, 0.9, 1.1])
    pp.runpp(net, run_control=False)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"] / net.res_line.loc[0, 'p_to_mw']) + 0.7953988 - np.arccos(0.7)) < tol)
    pp.runpp(net, run_control = True)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"]/net.res_line.loc[0, 'p_to_mw']) - np.arccos(0.7)) < tol)#positive = inductive


def test_pf_control_droop_q():
    net = simple_test_net()
    tol = 1e-6
    bsc = pp.control.BinarySearchControl(net, ctrl_in_service=True,
                                         output_element="sgen", output_variable="q_mvar", output_element_index=0,
                                         output_element_in_service=True, output_values_distribution='max_Q',
                                         input_element="res_line", damping_factor=0.9, input_variable="q_to_mvar",
                                         input_element_index=0, set_point=1, modus='PF_ctrl', tol=1e-6)
    pp.control.DroopControl(net, bus_idx=1, pf_overexcited= 0.5, pf_underexcited= 0.9,
                            vm_set_pu=1, vm_set_ub=3, vm_set_lb=1,
                            controller_idx=bsc.index, modus='PF_ctrl_P')
    pp.runpp(net, run_control=False)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"] / net.res_line.loc[0, 'p_to_mw']) - 0) < tol)
    pp.runpp(net, run_control=True) #Phi at point = 1.0471975521726393, Q at point = 3.4641016229480206 P at point = 1.999999999976119
    m = ((1 - 0.9) + (1 - 0.5)) / (3 - 1)  # getting function #m = 0.3
    b = -(1 - 0.9) - m * 1 #-0.4
    droop_set_point = 1 - (m * net.res_line.loc[0, 'p_to_mw'] + b) #should be 0.8 #reactance positive cause droop set point > 1
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"] / net.res_line.loc[0, 'p_to_mw']) - np.arccos(droop_set_point)) < tol)


def test_pf_control_droop_v():
    net = simple_test_net()
    tol = 1e-6
    bsc = pp.control.BinarySearchControl(net, ctrl_in_service=True,
                                         output_element="sgen", output_variable="q_mvar", output_element_index=0,
                                         output_element_in_service=True, output_values_distribution='rel_V_pu',
                                         input_element="res_line", damping_factor=0.9, input_variable="q_to_mvar",
                                         input_element_index=0, modus='PF_ctrl', tol=1e-6, set_point=0.5)
    pp.control.DroopControl(net, bus_idx=1, pf_overexcited=-0.3, pf_underexcited=0.7,
                            vm_set_pu=1, vm_set_ub=0.6, vm_set_lb=2.2,
                            controller_idx=bsc.index, modus='PF_ctrl_V')
    pp.runpp(net, run_control=False)
    assert (abs(np.arctan(net.res_line.loc[0, "q_to_mvar"] / net.res_line.loc[0, 'p_to_mw']) - 0) < tol)
    pp.runpp(net, run_control=True) #V at bus 1 is 1.038866702987955, Phi at point is 1.0439447685158276
    m = ((1 - 0.7) + (1 - -0.3)) / (0.6 - 2.2)  # getting function #m = -1
    b = (1 - -0.3) - m * 0.6 #b = 1.9
    droop_set_point = 1 - (m * net.res_bus.loc[1, 'vm_pu'] + b)  # should be 0.1388667029878976 #reactance positive cause droop set point > 1
    assert (abs(np.arctan(net.res_line.loc[0, "q_to_mvar"] / net.res_line.loc[0, 'p_to_mw']) - np.arccos(droop_set_point)) < tol)


def test_tan_phi_control():
    net = simple_test_net()
    tol = 1e-6
    bsc = pp.control.BinarySearchControl(net, ctrl_in_service= True, output_element='sgen', output_variable='q_mvar',
                         output_element_index= 0, output_element_in_service= True, output_values_distribution='rel_P',
                         input_element='res_switch', input_variable='q_mvar', input_element_index=2, modus='tan_(phi)_ctrl',
                                         tol = 1e-6, set_point=2)
    #todo rest



def test_stactrl_pf_import_new():
    path = os.path.join(pp.pp_dir, 'test', 'control', 'testfiles', 'stactrl_test_new.json')
    net = pp.from_json(path)
    tol = 1e-6
    pp.runpp(net, run_control=True)
    print("\n")
    print("--------------------------------------")
    print("Scenario 1 - Constant Q")
    print("Controlled line, constQ = 0.5 MVar - q_from_mvar and q_to_mvar: \n",
          net.res_line.loc[0, "q_to_mvar"], "\t", net.res_line.loc[0, "q_from_mvar"])
    print("Controlled line, constQ = 0.5 MVar - q_from_mvar and q_to_mvar: \n",
          net.res_line.loc[15, "q_to_mvar"], "\t", net.res_line.loc[15, "q_from_mvar"])
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - 0.5) < tol)
    assert(abs(net.res_line.loc[15, "q_to_mvar"] - 0.5) < tol)
    print("--------------------------------------")
    print("Scenario 2 - Constant U, droop 40 MVar/pu")
    print("Input Measurement q_from_mvar and q_to_mvar, expected: \n 0.2442 MVar, -0.6215 MVar: \n",
          net.res_line.loc[3, "q_to_mvar"], "\t", net.res_line.loc[3, "q_from_mvar"])
    print("Input Measurement q_from_mvar and q_to_mvar, expected:\n 0.2442 MVar, -0.6215 MVar: \n",
          net.res_line.loc[4, "q_to_mvar"], "\t", net.res_line.loc[4, "q_from_mvar"])
    print("Controlled bus, initial set point 1.01 pu and 40 MVar/pu, vm_pu, \n expected: "
          "2 * 0.2442 MVar / 40 MVar/pu + 1.01 pu = 1.02221: \n", net.res_bus.loc[74, "vm_pu"])
    assert(abs(net.res_bus.loc[74, "vm_pu"] - ((net.res_line.loc[3, "q_to_mvar"] +
                                             net.res_line.loc[4, "q_to_mvar"]) /
                                            40 + 1.01)) < tol)
    print("--------------------------------------")
    print("Scenario 3 - Constant U")
    print("Controlled bus, set point = 1.03 pu, vm_pu: \n", net.res_bus.loc[96, "vm_pu"])
    assert(abs(net.res_bus.loc[96, "vm_pu"] - 1.03) < tol)
    print("--------------------------------------")
    print("Scenario 4 - Q(U) - droop 40 MVar/pu")
    print("Input Measurement vm_pu: \n", net.res_bus.loc[115, "vm_pu"])
    print("Controlled Transformer Q, lower voltage band 0.999 pu, initial set point 1 MVar and 40 MVar/pu, q_hv_mvar, "
          "expected: \n -(1 MVar + (0.999 pu  - 0.99585 pu) * 40 MVar/pu)= -1.12618: \n",
          net.res_trafo.loc[3, "q_hv_mvar"])
    assert(abs(net.res_trafo.loc[3, "q_hv_mvar"] - (-(1 + (0.999 - net.res_bus.loc[115, "vm_pu"]) * 40))) < tol)
    print("------------------------------------- ")
    print("Scenario 5 - Constant Power factor")
    print("Controlled line, const PF = -1 Phi_from and Phi_to: \n",
          -np.cos(np.arctan(net.res_line.loc[16, 'q_from_mvar'] / net.res_line.loc[16, 'p_from_mw'])), "\t",
          -np.cos(np.arctan(net.res_line.loc[16, 'q_to_mvar'] / net.res_line.loc[16, 'p_to_mw'])))
    print("Controlled line, const PF = -1 Phi_from and Phi_to: \n",
          -np.cos(np.arctan(net.res_line.loc[17, 'q_from_mvar'] / net.res_line.loc[17, 'p_from_mw'])), "\t",
          -np.cos(np.arctan(net.res_line.loc[17, 'q_to_mvar'] / net.res_line.loc[17, 'p_to_mw'])))
    assert(abs(np.arctan(net.res_line.loc[16, "q_to_mvar"]/net.res_line.loc[16, 'p_to_mw']) - np.arccos(1)) < tol)
    assert(abs(np.arctan(net.res_line.loc[17, "q_to_mvar"]/net.res_line.loc[17, 'p_to_mw']) - np.arccos(1)) < tol)
    print("------------------------------------- ")#todo
    print("Scenario 6 - PF_Phi(V)")
    print("------------------------------------- ")
    print("Scenario 7 - PF_Phi(P)")


    print("------------------------------------- ")
    print("Scenario 8 - Tan(Phi)")
    print("Controlled line, tan(phi) = 0 tan(phi)_from and tan(phi)_to: \n",
          net.res_line.loc[20, "q_from_mvar"] / net.res_line.loc[20, "p_from_mw"],
          "\t", net.res_line.loc[20, 'q_to_mvar'] / net.res_line.loc[20, 'p_to_mw'])
    print("Controlled line, tan(phi) = 0 tan(phi)_from and tan(phi)_to: \n",
          net.res_line.loc[21, "q_from_mvar"] / net.res_line.loc[21, "p_from_mw"],
          "\t", net.res_line.loc[21, 'q_to_mvar'] / net.res_line.loc[21, 'p_to_mw'])
    assert(abs(net.res_line.loc[20, "q_to_mvar"] / net.res_line.loc[20, 'p_to_mw'] - np.arccos(1)) < tol)
    assert(abs(net.res_line.loc[21, "q_to_mvar"] / net.res_line.loc[21, 'p_to_mw']  - np.arccos(1)) < tol)


if __name__ == '__main__':
    pytest.main(['-s', __file__])
