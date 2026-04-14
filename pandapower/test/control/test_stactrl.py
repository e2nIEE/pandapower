# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
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

###test legacy support###
def test_volt_ctrl():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(
        net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar", tol=tol,
        output_element_index=[0], output_element_in_service=[True], distribution_method=[1], voltage_ctrl=True,
        input_element="res_bus", input_variable="vm_pu", input_element_index=[1], set_point=1.02
    )
    runpp(net, run_control=False)
    assert (abs(net.res_bus.loc[1, "vm_pu"] - 0.999648) < tol)
    runpp(net, run_control=True)
    assert (abs(net.res_bus.loc[1, "vm_pu"] - 1.02) < tol)
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'V_ctrl')
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))


def test_volt_ctrl_droop():
    net = simple_test_net()
    tol = 1e-6
    bsc = BinarySearchControl(net, name="BSC1", ctrl_in_service=True,
                              output_element="sgen", output_variable="q_mvar", output_element_index=[0],
                              output_element_in_service=[True], distribution_method=['rel_P'],
                              input_element="res_trafo", input_variable="q_hv_mvar", input_element_index=[0],
                              set_point=1.02, voltage_ctrl=True, bus_idx=1, tol=tol)
    DroopControl(net, name="DC1", q_droop_mvar=40, bus_idx=1,
                            vm_set_pu=1.02, controller_idx=bsc.index, voltage_ctrl=True, tol = tol)
    runpp(net, run_control=False)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - 0.999648) < tol)
    runpp(net, run_control=True)
    assert(net.controller.object[0].converged == True and net.controller.object[1].converged == True)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - (1.02 + net.res_trafo.loc[0, "q_hv_mvar"] / 40)) < tol)
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'V_ctrl_Q_droop')#test correct control_modus
    assert(getattr(net.controller.at[1, 'object'].control_modus, 'value', None) == 'V_ctrl_Q_droop')  # test correct control_modus
    assert(net.controller.at[1, 'object'].controller_idx == 0)  # test droop controller linkage


def test_qctrl():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
                        output_element_index=[0], output_element_in_service=[True],
                        distribution_method='rel_rated_S', input_element="res_line",
                        damping_factor=0.9, input_variable=["q_to_mvar"],
                        input_element_index=0, set_point=1, voltage_ctrl=False, tol=1e-6)
    runpp(net, run_control=False)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (-6.092016e-12)) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - 1.0) < tol)
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'Q_ctrl')# test correct control_modus


def test_qctrl_imp_input():
    net = simple_test_net()
    tol = 1e-6
    create_impedance(net, 1, 2, sn_mva=1, rft_pu=0.01, xft_pu=0.01, rtf_pu=0.01, xtf_pu=0.01)
    BinarySearchControl(
        net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar", damping_factor=0.9,
        output_element_index=[0], output_element_in_service=[True], output_values_distribution=[1], voltage_ctrl=False,
        input_element="res_impedance", input_variable="q_to_mvar", input_element_index=0, set_point=1, tol=1e-6
    )
    runpp(net, run_control=False)
    assert (abs(net.res_impedance.loc[0, "q_to_mvar"] - 0.01373636) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_impedance.loc[0, "q_to_mvar"] - 1.0) < tol)
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'Q_ctrl')
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))


def test_qctrl_droop():
    net = simple_test_net()
    tol = 1e-6
    net.load.loc[0, "p_mw"] = 60  # create voltage drop at bus 1
    bsc = BinarySearchControl(net, name="BSC1", ctrl_in_service=True,
                              output_element="sgen", output_variable="q_mvar", output_element_index=[0],
                              output_element_in_service=[True], distribution_method='set_Q',
                              input_element="res_line", damping_factor=0.9, input_variable=["q_from_mvar"],
                              input_inverted=True, input_element_index=0, set_point=1, voltage_ctrl=False, tol=1e-6)
    DroopControl(net, name="DC1", q_droop_mvar=40, bus_idx=1,
                            vm_set_pu=1, vm_set_ub=1.005, vm_set_lb=0.995,
                            controller_idx=bsc.index, voltage_ctrl=False)
    runpp(net, run_control=False)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (-1e-13)) < tol)
    runpp(net, run_control=True)
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(abs(net.controller.object[0].input_sign[0] * net.res_line.loc[0, "q_from_mvar"] - (
            net.controller.object[1].q_set_mvar_bsc + (0.995 - net.res_bus.loc[1, "vm_pu"]) * 40)) < tol)
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'Q_ctrl_V_droop')#test correct control_modus
    assert(getattr(net.controller.at[1, 'object'].control_modus, 'value', None) == 'Q_ctrl_V_droop')   # test correct control_modus
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
    assert(abs(net.res_sgen.loc[0, "q_mvar"] - 0.5) < tol)
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'Q_ctrl')
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
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'Q_ctrl')
    assert(abs(net.res_sgen.loc[0, "q_mvar"] + 0.5) < tol)



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
    assert(abs(net.res_sgen.loc[0, "q_mvar"] - 0.7) < tol)
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'V_ctrl')
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
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
    assert(abs(net.res_sgen.loc[0, "q_mvar"] + 0.7) < tol)
    net.sgen.min_q_mvar = -0.8 # tests change of min_q_mvar afterward
    runpp(net, run_control=True, enforce_q_lims=True)
    assert(abs(net.res_sgen.loc[0, "q_mvar"] + 0.8) < tol)
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'V_ctrl')
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))

@pytest.mark.parametrize("v", linspace(start=0.98, stop=1.02, num=5, dtype=float64))
@pytest.mark.parametrize("p", linspace(start=-2.5, stop=2.5, num=10, dtype=float64))
def test_qlimits_with_capability_curve(v, p):
    net = simple_test_net()
    tol = 1e-6
    create_sgen(net, 2, p_mw=0., sn_mva=0, name="sgen2")
    # create q characteristics table
    net["q_capability_curve_table"] = DataFrame(
        {'id_q_capability_curve': [0, 0, 0, 0, 0],
        'p_mw': [-2.0, -1.0, 0.0, 1.0, 2.0],
        'q_min_mvar': [-0.1, -0.1, -0.1, -0.1, -0.1],
        'q_max_mvar': [0.1, 0.1, 0.1, 0.1, 0.1]})
    net.sgen.at[0, "id_q_capability_characteristic"] = 0
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
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'V_ctrl')
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))


def test_qlimits_with_capability_curve_no_reactive_power():
    # test once more when there is no reactive power capability curve
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, name="BSC1", ctrl_in_service=True,
                        output_element="sgen", output_variable="q_mvar", output_element_index=[0],
                        output_element_in_service=[True], output_values_distribution=[1],
                        input_element="res_bus", input_variable="vm_pu", input_element_index=[1],
                        set_point=0.98, voltage_ctrl=True, tol=tol)
    runpp(net, run_control=True, enforce_q_lims=True)
    assert(abs(net.res_sgen.loc[0, 'q_mvar'] + 6.7373132) < tol)
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'V_ctrl')
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))


def test_stactrl_pf_import():
    path = os.path.join(pp_dir, 'test', 'control', 'testfiles', 'stactrl_test.json')
    net = from_json(path)
    tol = 1e-6
    runpp(net, run_control=True)
    assert all(net.controller.object[i].converged for i in net.controller.index)

    logger.info("")
    logger.info("--------------------------------------")
    logger.info("Scenario 1 - Constant Q")
    logger.info(
        "Controlled line 0 to, expected constQ = -0.86 MVar for q_from_mvar and constQ = 0.5 MVar for q_to_mvar:\n%s\t%s",
        net.res_line.loc[0, "q_from_mvar"], net.res_line.loc[0, "q_to_mvar"]
    )
    logger.info(
        "Controlled line 1 to, expected constQ = -0.86 MVar for q_from_mvar and constQ = 0.5 MVar for q_to_mvar:\n%s\t%s",
        net.res_line.loc[2, "q_from_mvar"], net.res_line.loc[2, "q_to_mvar"]
    )
    assert abs(net.res_line.loc[0, "q_to_mvar"] - 0.5) < tol
    assert abs(net.res_line.loc[2, "q_to_mvar"] - 0.5) < tol
    assert getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'Q_ctrl'  # test correct control_modus

    logger.info("--------------------------------------")
    logger.info("Scenario 2 - Constant V, droop 40 MVar/pu")
    logger.info(
        "Input Measurement line 4 q_from_mvar and q_to_mvar, expected:\n-0.6215 MVar\t0.2442 MVar\n%s\t%s",
        net.res_line.loc[4, "q_from_mvar"], net.res_line.loc[4, "q_to_mvar"]
    )
    logger.info(
        "Input Measurement line 5 q_from_mvar and q_to_mvar, expected:\n-0.6215 MVar\t0.2442 MVar\n%s\t%s",
        net.res_line.loc[5, "q_from_mvar"], net.res_line.loc[5, "q_to_mvar"]
    )
    logger.info(
        "Controlled bus, initial set point 1.01 pu and 40 MVar/pu, vm_pu,\n"
        "expected: 2 * 0.2442 MVar / 40 MVar/pu + 1.01 pu = 1.02221:\n%s",
        net.res_bus.loc[62, "vm_pu"]
    )
    assert abs(
        net.res_bus.loc[62, "vm_pu"] - (
            1.01 + (
                (net.res_line.loc[4, "q_to_mvar"] + net.res_line.loc[5, "q_to_mvar"])
                / net.controller.object[4].q_droop_mvar
            )
        )
    ) < tol
    assert getattr(net.controller.at[4, 'object'].control_modus, 'value', None) == 'V_ctrl_Q_droop'  # test correct droop control_modus
    assert getattr(net.controller.at[3, 'object'].control_modus, 'value', None) == 'V_ctrl_Q_droop'  # test correct control_modus
    assert net.controller.at[4, 'object'].controller_idx == 3  # test droop controller linkage

    logger.info("--------------------------------------")
    logger.info("Scenario 3 - Constant V")
    logger.info("Controlled bus, set point = 1.03 pu, vm_pu: %s", net.res_bus.loc[84, "vm_pu"])
    assert abs(net.res_bus.loc[84, "vm_pu"] - 1.03) < tol
    assert getattr(net.controller.at[5, 'object'].control_modus, 'value', None) == 'V_ctrl'  # test correct control_modus

    logger.info("--------------------------------------")
    logger.info("Scenario 4 - Q(U) - droop 40 MVar/pu")
    logger.info("Input Measurement vm_pu: %s", net.res_bus.loc[91, "vm_pu"])
    logger.info(
        "Controlled Transformer Q, lower voltage band 0.999 pu, initial set point 1 MVar and 40 MVar/pu, q_hv_mvar,\n"
        "expected:\n-(1 MVar + (0.999 pu - 0.99585 pu) * 40 MVar/pu) = -1.12618:\n%s",
        net.res_trafo.loc[3, "q_hv_mvar"]
    )
    assert abs(
        net.res_trafo.loc[3, "q_hv_mvar"] - (
            -(1 + (0.999 - net.res_bus.loc[91, "vm_pu"]) * net.controller.object[2].q_droop_mvar)
        )
    ) < tol
    assert getattr(net.controller.at[2, 'object'].control_modus, 'value', None) == 'Q_ctrl_V_droop'  # test correct droop control_modus
    assert getattr(net.controller.at[1, 'object'].control_modus, 'value', None) == 'Q_ctrl_V_droop'  # test correct control_modus
    assert net.controller.at[2, 'object'].controller_idx == 1  # test droop controller linkage

### Testing after rework of station controller###

def test_volt_ctrl_new():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, ctrl_in_service=True,
                        output_element="sgen", output_variable="q_mvar", output_element_index=0,
                        output_element_in_service=True, distribution_method='rel_P',
                        output_values_distribution= 2,
                        input_element="res_bus", input_variable="vm_pu", input_element_index=1,
                        set_point=1.02, control_modus='V_ctrl', tol=tol, bus_idx = 1)
    runpp(net, run_control=False)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - 0.999648) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - 1.02) < tol)
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'V_ctrl')# test correct control_modus


def test_volt_ctrl_droop_new():
    net = simple_test_net()
    tol = 1e-6
    bsc = BinarySearchControl(net, ctrl_in_service=True,
                                         output_element="sgen", output_variable="q_mvar", output_element_index=0,
                                         output_element_in_service=True, distribution_method='rel_rated_S',
                                         input_element="res_trafo", input_variable="q_hv_mvar", input_element_index=0,
                                         set_point=1.02,control_modus = 'V_ctrl_Q_droop', tol=tol, bus_idx =1)
    DroopControl(net, q_droop_mvar=40, controller_idx=bsc.index, control_modus = "V_ctrl_Q_droop", input_element_q_meas='res_trafo',
                 input_variable_q_meas='q_hv_mvar', bus_idx=1, vm_set_pu_bsc = 1.02, tol = tol)
    runpp(net, run_control=False)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - 0.999648) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_bus.loc[1, "vm_pu"] - (1.02 + net.res_trafo.loc[0, "q_hv_mvar"] / 40)) < tol)
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'V_ctrl_Q_droop')# test correct control_modus
    assert(getattr(net.controller.at[1, 'object'].control_modus, 'value', None) == 'V_ctrl_Q_droop')# test correct control_modus
    assert(net.controller.at[1, 'object'].controller_idx == 0)  # test droop controller linkage


def test_qctrl_new():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
                        output_element_index=0, output_element_in_service=True,
                        distribution_method='set_Q', input_element="res_line",
                        damping_factor=0.9, input_variable=["q_to_mvar"], output_values_distribution= [0.2, 0.3],
                        input_element_index=0, set_point=1, control_modus = 'Q_ctrl', tol=1e-6)
    runpp(net, run_control=False)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (-6.092016e-12)) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - 1.0) < tol)
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'Q_ctrl')#test correct control_modus


def test_qctrl_droop_new():
    net = simple_test_net()
    tol = 1e-6
    net.load.loc[0, "p_mw"] = 60  # create voltage drop at bus 1
    bsc = BinarySearchControl(net, ctrl_in_service=True,
                              output_element="sgen", output_variable="q_mvar", output_element_index=0,
                              output_element_in_service=True, distribution_method='max_Q',
                              input_element="res_line", damping_factor=0.9, input_variable="q_to_mvar",
                              input_element_index=0, set_point=1, control_modus = 'Q_ctrl_V_droop', tol=1e-6)
    DroopControl(net, q_droop_mvar=40, bus_idx=1,
                 vm_set_pu=1, vm_set_ub=1.005, vm_set_lb=0.995,
                 controller_idx=bsc.index, control_modus='Q_ctrl_V_droop')
    runpp(net, run_control=False)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (-7.094325e-13)) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_line.loc[0, "q_to_mvar"] - (1 + (0.995 - net.res_bus.loc[1, "vm_pu"]) * 40)) < tol)
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'Q_ctrl_V_droop')# test correct control_modus
    assert(getattr(net.controller.at[1, 'object'].control_modus, 'value', None) == 'Q_ctrl_V_droop')   # test correct control_modus
    assert(net.controller.at[1, 'object'].controller_idx == 0)  # test droop controller linkage

def test_pf_control_cap():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, ctrl_in_service=True, output_element='sgen', output_variable='q_mvar',
                        output_element_index=0, distribution_method='rel_V_pu',
                        input_element='res_line', output_element_in_service=True,
                        damping_factor = 0.9, input_variable='q_to_mvar', input_element_index=0,
                        set_point = 0.7, tol = 1e-6, control_modus = 'PF_ctrl_cap',
                        output_values_distribution=[1, 0.9, 1.1])
    runpp(net, run_control=False)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"] / net.res_line.loc[0, 'p_to_mw']) + 0.7953988 - np.arccos(0.7)) < tol)
    runpp(net, run_control = True)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"]/net.res_line.loc[0, 'p_to_mw']) - - np.arccos(0.7)) < tol)#negative cause capacitive
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'PF_ctrl_cap')# test correct control_modus


def test_pf_control_ind():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, ctrl_in_service=True, output_element='sgen', output_variable='q_mvar',
                        output_element_index=0, distribution_method='max_Q',
                        input_element='res_line', output_element_in_service=True,
                        damping_factor = 0.9, input_variable='q_to_mvar', input_element_index=0,
                        set_point = 0.7, tol = 1e-6, control_modus = 'PF_ctrl_ind',
                        output_values_distribution=[1, 0.9, 1.1])
    runpp(net, run_control=False)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"] / net.res_line.loc[0, 'p_to_mw']) + 0.7953988 - np.arccos(0.7)) < tol)
    runpp(net, run_control = True)
    assert(abs(np.arctan(net.res_line.loc[0, "q_to_mvar"]/net.res_line.loc[0, 'p_to_mw']) - np.arccos(0.7)) < tol)#positive means inductive
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'PF_ctrl_ind')# test correct control_modus

def test_tan_phi_control():
    net = simple_test_net()
    tol = 1e-6
    BinarySearchControl(net, ctrl_in_service= True, output_element='sgen', output_variable='q_mvar',
                        output_element_index= 0, output_element_in_service= True, distribution_method='rel_P',
                        input_element='res_trafo', input_variable='q_lv_mvar', input_element_index=0, control_modus='tan_phi_ctrl',
                        tol = 1e-6, set_point=2)
    runpp(net, run_control=False)
    assert(abs(net.res_trafo.loc[0, "q_lv_mvar"] / net.res_trafo.loc[0, 'p_lv_mw'] - 0.097382) < tol)
    runpp(net, run_control=True)
    assert(abs(net.res_trafo.loc[0, "q_lv_mvar"] / net.res_trafo.loc[0, 'p_lv_mw'] - 2) < tol)
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'tan_phi_ctrl')   # test correct control_modus

def test_station_ctrl_pf_import_new():
    path = os.path.join(pp_dir, 'test', 'control', 'testfiles', 'station_ctrl_test_new.json')
    net = from_json(path)
    tol = 1e-6
    runpp(net, run_control=True)
    assert all(net.controller.object[i].converged for i in net.controller.index)

    logger.info("")
    logger.info("--------------------------------------")
    logger.info("Scenario 1 - Constant Q")
    logger.info(
        "Controlled line 0, constQ = -0.86 MVar for q_from_mvar and constQ = 0.5 MVar for q_to_mvar:\n%s\t%s",
        net.res_line.loc[0, "q_from_mvar"], net.res_line.loc[0, "q_to_mvar"]
    )
    logger.info(
        "Controlled line 15, constQ = -0.86 MVar for q_from_mvar and constQ = 0.5 MVar for q_to_mvar:\n%s\t%s",
        net.res_line.loc[15, "q_from_mvar"], net.res_line.loc[15, "q_to_mvar"]
    )
    assert abs(net.res_line.loc[0, "q_to_mvar"] - 0.5) < tol
    assert abs(net.res_line.loc[15, "q_to_mvar"] - 0.5) < tol
    assert getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'Q_ctrl'  # test correct control_modus

    logger.info("--------------------------------------")
    logger.info("Scenario 2 - Constant V, droop 40 MVar/pu")
    logger.info(
        "Input Measurement q_from_mvar and q_to_mvar, expected:\n-0.6215 MVar\t0.2442 MVar\n%s\t%s",
        net.res_line.loc[3, "q_from_mvar"], net.res_line.loc[3, "q_to_mvar"]
    )
    logger.info(
        "Input Measurement q_from_mvar and q_to_mvar, expected:\n-0.6215 MVar\t0.2442 MVar\n%s\t%s",
        net.res_line.loc[4, "q_from_mvar"], net.res_line.loc[4, "q_to_mvar"]
    )
    logger.info(
        "Controlled bus, initial set point 1.01 pu and 40 MVar/pu, vm_pu,\n"
        "expected: 2 * 0.2442 MVar / 40 MVar/pu + 1.01 pu = 1.02221:\n%s",
        net.res_bus.loc[86, "vm_pu"]
    )
    assert abs(
        net.res_bus.loc[77, "vm_pu"] - (
            1.01 + (net.res_line.loc[3, "q_to_mvar"] + net.res_line.loc[4, "q_to_mvar"]) / 40
        )
    ) < tol
    assert getattr(net.controller.at[6, 'object'].control_modus, 'value', None) == 'V_ctrl_Q_droop'  # test correct control_modus
    assert getattr(net.controller.at[7, 'object'].control_modus, 'value', None) == 'V_ctrl_Q_droop'  # test correct control_modus
    assert net.controller.at[7, 'object'].controller_idx == 6  # test droop controller linkage

    logger.info("--------------------------------------")
    logger.info("Scenario 3 - Constant V")
    logger.info("Controlled bus, set point = 1.03 pu\nvm_pu: %s", net.res_bus.loc[99, "vm_pu"])
    assert abs(net.res_bus.loc[99, "vm_pu"] - 1.03) < tol
    assert getattr(net.controller.at[5, 'object'].control_modus, 'value', None) == 'V_ctrl'  # test correct control_modus

    logger.info("--------------------------------------")
    logger.info("Scenario 4 - Q(U) - droop 40 MVar/pu")
    logger.info("Input Measurement vm_pu: %s", net.res_bus.loc[127, "vm_pu"])
    logger.info(
        "Controlled Transformer Q, lower voltage band 0.999 pu, initial set point 1 MVar and 40 MVar/pu, q_hv_mvar,\n"
        "expected:\n-(1 MVar + (0.999 pu - 0.99585 pu) * 40 MVar/pu) = -1.12618 MVar:\n%s",
        net.res_trafo.loc[3, "q_hv_mvar"]
    )
    assert abs(
        net.res_trafo.loc[3, "q_hv_mvar"] - (-(1 + (0.999 - net.res_bus.loc[127, "vm_pu"]) * 40))
    ) < tol
    assert getattr(net.controller.at[1, 'object'].control_modus, 'value', None) == 'Q_ctrl_V_droop'  # test correct control_modus
    assert getattr(net.controller.at[2, 'object'].control_modus, 'value', None) == 'Q_ctrl_V_droop'  # test correct control_modus
    assert net.controller.at[2, 'object'].controller_idx == 1  # test droop controller linkage

    logger.info("------------------------------------- ")
    logger.info("Scenario 5 - Constant Power factor")
    logger.info(
        "Controlled line 16 to, expected const PF = 0.93 for Phi_from and const PF = 1 for Phi_to:\n%s\t%s",
        np.cos(np.arctan(net.res_line.loc[16, 'q_from_mvar'] / net.res_line.loc[16, 'p_from_mw'])),
        np.cos(np.arctan(net.res_line.loc[16, 'q_to_mvar'] / net.res_line.loc[16, 'p_to_mw']))
    )
    logger.info(
        "Controlled line 17 to, expected const PF = 0.93 for Phi_from and const PF = 1 for Phi_to:\n%s\t%s",
        np.cos(np.arctan(net.res_line.loc[17, 'q_from_mvar'] / net.res_line.loc[17, 'p_from_mw'])),
        np.cos(np.arctan(net.res_line.loc[17, 'q_to_mvar'] / net.res_line.loc[17, 'p_to_mw']))
    )
    assert abs(np.arctan(net.res_line.loc[16, "q_to_mvar"] / net.res_line.loc[16, 'p_to_mw']) - np.arccos(1)) < tol  # positive reactance because inductive
    assert abs(np.arctan(net.res_line.loc[17, "q_to_mvar"] / net.res_line.loc[17, 'p_to_mw']) - np.arccos(1)) < tol
    assert getattr(net.controller.at[3, 'object'].control_modus, 'value', None) == 'PF_ctrl_ind'  # test correct control_modus

    logger.info("------------------------------------- ")
    logger.info("Scenario 8 - Tan(Phi)")
    logger.info(
        "Controlled line 20 to, expected tan(phi) = 0.376 for tan(phi)_from and tan(phi) = 0 for tan(phi)_to:\n%s\t%s",
        net.res_line.loc[20, "q_from_mvar"] / net.res_line.loc[20, "p_from_mw"],
        net.res_line.loc[20, 'q_to_mvar'] / net.res_line.loc[20, 'p_to_mw']
    )
    logger.info(
        "Controlled line 21 to, expected tan(phi) = 0.376 for tan(phi)_from and tan(phi) = 0 for tan_phi_to:\n%s\t%s",
        net.res_line.loc[21, "q_from_mvar"] / net.res_line.loc[21, "p_from_mw"],
        net.res_line.loc[21, 'q_to_mvar'] / net.res_line.loc[21, 'p_to_mw']
    )
    assert abs(net.res_line.loc[20, "q_to_mvar"] / net.res_line.loc[20, 'p_to_mw'] - 0) < tol
    assert abs(net.res_line.loc[21, "q_to_mvar"] / net.res_line.loc[21, 'p_to_mw'] - 0) < tol
    assert getattr(net.controller.at[4, 'object'].control_modus, 'value', None) == 'tan_phi_ctrl'  # test correct control_modus

### Test Q distributions###

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
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(net.sgen.at[0, 'q_mvar'] != net.sgen.at[1, 'q_mvar'])
    assert(abs(net.sgen.at[0, 'q_mvar']/(net.sgen.at[0, 'q_mvar'] + net.sgen.at[1, 'q_mvar']) - net.sgen.at[0, 'p_mw']/(
        net.sgen.at[0, 'p_mw'] + net.sgen.at[1, 'p_mw'])) < tol)
    assert(abs(net.sgen.at[1, 'q_mvar'] / (net.sgen.at[0, 'q_mvar'] + net.sgen.at[1, 'q_mvar'])-net.sgen.at[1, 'p_mw']/(
        net.sgen.at[0, 'p_mw'] + net.sgen.at[1, 'p_mw'])) < tol)
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'V_ctrl')  # test correct control_modus
    assert(getattr(net.controller.at[0, 'object'].distribution_method, 'value', None) == 'rel_P')

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
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'Q_ctrl')  # test correct control_modus
    assert(getattr(net.controller.at[0, 'object'].distribution_method, 'value', None) == 'rel_rated_S')

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
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'PF_ctrl_ind')  # test correct control_modus
    assert(getattr(net.controller.at[0, 'object'].distribution_method, 'value', None) == 'set_Q')

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
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'PF_ctrl_cap')  # test correct control_modus
    assert (getattr(net.controller.at[0, 'object'].distribution_method, 'value', None) == 'max_Q')

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
                        [[0.98, 0.95, 1.1], [0.89, 0.8, 1.3]], 'tan_phi_ctrl', 1e-6)
    net.sgen.drop(2, inplace=True) #delete interfering sgen
    runpp(net, run_control= True)
    assert(net.sgen.at[0, 'q_mvar'] != net.sgen.at[1, 'q_mvar']) #now controlled sgens
    assert(abs(net.res_bus.at[net.sgen.at[0, 'bus'], 'vm_pu'] + net.res_bus.at[net.sgen.at[1, 'bus'], 'vm_pu']
                - 0.98 - 0.89) < tol) #now at set points
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'tan_phi_ctrl')  # test correct control_modus
    assert(getattr(net.controller.at[0, 'object'].distribution_method, 'value', None) == 'rel_V_pu')

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
    assert(all(net.controller.object[i].converged == True for i in net.controller.index))
    assert(getattr(net.controller.at[0, 'object'].control_modus, 'value', None) == 'Q_ctrl')  # test correct control_modus
    assert(getattr(net.controller.at[1, 'object'].control_modus, 'value', None) == 'tan_phi_ctrl')  # test correct control_modus
    assert(getattr(net.controller.at[2, 'object'].control_modus, 'value', None) == 'PF_ctrl_ind')  # test correct control_modus
    assert(getattr(net.controller.at[3, 'object'].control_modus, 'value', None) == 'PF_ctrl_cap')  # test correct control_modus
    assert(getattr(net.controller.at[4, 'object'].control_modus, 'value', None) == 'tan_phi_ctrl')  # test correct control_modus
    assert(getattr(net.controller.at[0, 'object'].distribution_method, 'value', None) == 'set_Q')
    assert(getattr(net.controller.at[1, 'object'].distribution_method, 'value', None) == 'rel_rated_S')
    assert(getattr(net.controller.at[2, 'object'].distribution_method, 'value', None) == 'rel_P')
    assert(getattr(net.controller.at[3, 'object'].distribution_method, 'value', None) == 'max_Q')
    assert(getattr(net.controller.at[4, 'object'].distribution_method, 'value', None) == 'rel_V_pu')


#todo test distributions with enabled q_lims

if __name__ == '__main__':
    pytest.main(['-s', __file__])