# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import os
import logging

from pandapower.control.controller.station_control import BinarySearchControl, DroopControl
from pandapower.create import create_empty_network, create_bus, create_buses, create_ext_grid, create_transformer, \
    create_load, create_line, create_sgen, create_impedance
from pandapower.run import runpp
from pandapower.file_io import from_json
from pandapower import pp_dir
from pandapower.control.util.auxiliary import create_q_capability_characteristics_object

from numpy import linspace, float64

from pandas import DataFrame

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
    BinarySearchControl(net, name="BSC1", ctrl_in_service=True,
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
    bsc = BinarySearchControl(net, name="BSC1", ctrl_in_service=True,
                                         output_element="sgen", output_variable="q_mvar", output_element_index=[0],
                                         output_element_in_service=[True], output_values_distribution=[1],
                                         input_element="res_trafo", input_variable="q_hv_mvar", input_element_index=[0],
                                         set_point=1.02, voltage_ctrl=True, bus_idx=1, tol=tol)
    DroopControl(net, name="DC1", q_droop_mvar=40, bus_idx=1,
                            vm_set_pu=1.02, controller_idx=bsc.index, voltage_ctrl=True)
    runpp(net, run_control=False)
    assert (abs(net.res_bus.loc[1, "vm_pu"] - 0.999648) < tol)
    runpp(net, run_control=True)
    assert(net.controller.object[0].converged == True and net.controller.object[1].converged == True)
    assert (abs(net.res_bus.loc[1, "vm_pu"] - (1.02 + net.res_trafo.loc[0, "q_hv_mvar"] / 40)) < tol)


def test_qctrl():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
                                   output_element_index=[0], output_element_in_service=[True],
                                   output_values_distribution=[1], input_element="res_line",
                                   damping_factor=0.9, input_variable=["q_to_mvar"],
                                   input_element_index=0, set_point=1, voltage_ctrl=False, tol=1e-6)
    runpp(net, run_control=False)
    assert (abs(net.res_line.loc[0, "q_to_mvar"] - (-6.092016e-12)) < tol)
    runpp(net, run_control=True)
    assert (abs(net.res_line.loc[0, "q_to_mvar"] - 1.0) < tol)


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
                              input_element="res_line", damping_factor=0.9, input_variable=["q_from_mvar"],
                              input_inverted=True, input_element_index=0, set_point=1, voltage_ctrl=False, tol=1e-6)
    DroopControl(net, name="DC1", q_droop_mvar=40, bus_idx=1,
                            vm_set_pu=1, vm_set_ub=1.005, vm_set_lb=0.995,
                            controller_idx=bsc.index, voltage_ctrl=False)
    runpp(net, run_control=False)
    assert (abs(net.res_line.loc[0, "q_to_mvar"] - (-1e-13)) < tol)
    runpp(net, run_control=True)
    assert (net.controller.object[0].converged == True and net.controller.object[1].converged == True)
    assert (abs(net.controller.object[0].input_sign[0] * net.res_line.loc[0, "q_from_mvar"] - (
                net.controller.object[1].q_set_mvar_bsc + (net.res_bus.loc[1, "vm_pu"] - 0.995) * 40)) < tol)

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

def test_qlimits_with_capability_curve():
    for v in linspace(start=0.98, stop=1.02, num=5, dtype=float64):
        for p in linspace(start=-2.5, stop=2.5, num=10, dtype=float64):
            net = simple_test_net()
            create_sgen(net, 2, p_mw=0., sn_mva=0, name="sgen2")
            tol = 1e-6
            # create q characteristics table
            net["q_capability_curve_table"] = DataFrame(
                {'id_q_capability_curve': [0, 0, 0, 0, 0],
                'p_mw': [-2.0, -1.0, 0.0, 1.0, 2.0],
                'q_min_mvar': [-0.1, -0.1, -0.1, -0.1, -0.1],
                'q_max_mvar': [0.1, 0.1, 0.1, 0.1, 0.1]})

            net.sgen.id_q_capability_characteristic.at[0] = 0
            net.sgen['curve_style'] = "straightLineYValues"
            create_q_capability_characteristics_object(net)

            BinarySearchControl(net, name="BSC1", ctrl_in_service=True,
                                output_element="sgen", output_variable="q_mvar", output_element_index=[0],
                                output_element_in_service=[True], output_values_distribution=[1],
                                input_element="res_bus", input_variable="vm_pu", input_element_index=[1],
                                set_point=v, voltage_ctrl=True, tol=tol)
            net.sgen.loc[0, 'p_mw'] = p
            runpp(net, run_control=True, enforce_q_lims=True)
            assert -0.1 <= net.res_sgen.loc[0, 'q_mvar'] <= 0.1

    net = simple_test_net() # test once more when there is no reactive power capability curve
    net["q_capability_curve_table"] = DataFrame(
        {'id_q_capability_curve': [0, 0, 0, 0, 0],
        'p_mw': [-2.0, -1.0, 0.0, 1.0, 2.0],
        'q_min_mvar': [-0.1, -0.1, -0.1, -0.1, -0.1],
        'q_max_mvar': [0.1, 0.1, 0.1, 0.1, 0.1]})

    net.sgen.id_q_capability_characteristic.at[0] = 0
    net.sgen['curve_style'] = "straightLineYValues"
    create_q_capability_characteristics_object(net)
    net.sgen.drop(columns=['reactive_capability_curve'], inplace=True)
    BinarySearchControl(net, name="BSC1", ctrl_in_service=True,
                        output_element="sgen", output_variable="q_mvar", output_element_index=[0],
                        output_element_in_service=[True], output_values_distribution=[1],
                        input_element="res_bus", input_variable="vm_pu", input_element_index=[1],
                        set_point=0.98, voltage_ctrl=True, tol=tol)
    runpp(net, run_control=True, enforce_q_lims=True)
    assert abs(net.res_sgen.loc[0, 'q_mvar'] + 6.7373132) < tol



def test_stactrl_pf_import():
    path = os.path.join(pp_dir, 'test', 'control', 'testfiles', 'stactrl_test.json')
    net = from_json(path)

    tol = 1e-3

    # Decrease controllers tolerance
    net.controller.object[2].tol = 0.00001

    runpp(net, run_control=True)
    logger.info("Scenario 1 - Constant Q")
    logger.info("Controlled line, constQ = 0.5 MVar - q_from_mvar and q_to_mvar: \n",
          net.res_line.loc[0, "q_to_mvar"], "\t", net.res_line.loc[0, "q_from_mvar"])
    logger.info("Controlled line, constQ = 0.5 MVar - q_from_mvar and q_to_mvar: \n",
          net.res_line.loc[2, "q_to_mvar"], "\t", net.res_line.loc[1, "q_from_mvar"])
    assert (net.res_line.loc[0, "q_to_mvar"] - 0.5 < tol)
    assert (net.res_line.loc[2, "q_to_mvar"] - 0.5 < tol)
    logger.info("Scenario 2 - Constant U, droop 40 MVar/pu")
    logger.info("Input Measurement q_from_mvar and q_to_mvar, expected: \n 0.2442 MVar, -0.6215 MVar: \n",
          net.res_line.loc[4, "q_to_mvar"], "\t", net.res_line.loc[4, "q_from_mvar"])
    logger.info("Input Measurement q_from_mvar and q_to_mvar, expected:\n 0.2442 MVar, -0.6215 MVar: \n",
          net.res_line.loc[5, "q_to_mvar"], "\t", net.res_line.loc[5, "q_from_mvar"])
    logger.info("Controlled bus, initial set point 1.01 pu and 40 MVar/pu, vm_pu, \n expected: "
        "2 * 0.2442 MVar / 40 MVar/pu + 1.01 pu = 1.02221: \n", net.res_bus.loc[62, "vm_pu"])
    assert (net.res_bus.loc[62, "vm_pu"] - (1.01 + (net.res_line.loc[4, "q_to_mvar"] + net.res_line.loc[5, "q_to_mvar"])
                                            / net.controller.object[4].q_droop_mvar) < tol)
    logger.info("Scenario 3 - Constant U")
    logger.info("Controlled bus, set point = 1.03 pu, vm_pu: \n", net.res_bus.loc[84, "vm_pu"])
    assert (net.res_bus.loc[84, "vm_pu"] - 1.03 < tol)
    logger.info("Scenario 4 - Q(U) - droop 40 MVar/pu")
    logger.info("Input Measurement vm_pu: \n", net.res_bus.loc[103, "vm_pu"])
    logger.info(
        "Controlled Transformer Q, lower voltage band 0.999 pu, initial set point 1 MVar and 40 MVar/pu, "
        "q_hv_mvar, expected: \n 1 MVar + (0.995778 - 0.999 pu) * 40 MVar/pu) = 0.87112 MVar: \n",
        net.res_trafo.loc[3, "q_hv_mvar"])
    assert (net.res_trafo.loc[3, "q_hv_mvar"] - (1 + (net.res_bus.loc[91, "vm_pu"] - 0.999)
                                                   * net.controller.object[2].q_droop_mvar) < tol)


if __name__ == '__main__':
    pytest.main(['-s', __file__])
