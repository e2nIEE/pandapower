# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""
Characterization tests for the station controller (BinarySearchControl, DroopControl,
VDroopControl_local).

The RECORDS dict below pins the converged results and the number of powerflow calls of every
scenario, recorded on the pre-refactor implementation (develop, 2026-07). Refactor steps of the
station controller must reproduce these values within VALUE_ATOL and must not need more powerflow
calls than the recorded baseline (fewer is allowed and expected).

To (re-)record after an intentional behavior change, run:
    python pandapower/test/control/test_stactrl_characterization.py --record
and paste the printed dict into RECORDS.
"""

import os
import logging

import numpy as np
import pytest

from pandapower import pp_dir
from pandapower.control.controller.station_control import (
    BinarySearchControl, DroopControl, VDroopControl_local)
from pandapower.control.run_control import run_control
from pandapower.create import (
    create_empty_network, create_bus, create_buses, create_ext_grid, create_transformer,
    create_load, create_line, create_line_from_parameters, create_sgen, create_gen,
    create_impedance, create_shunt)
from pandapower.file_io import from_json
from pandapower.run import runpp

logger = logging.getLogger(__name__)

VALUE_ATOL = 5e-6
# station outputs that merely realize a target (sgen/gen setpoints, shunt steps) depend on
# which iterate first satisfies |diff| < tol; they get a looser tolerance than the
# controlled quantities themselves
OUTPUT_ATOL = 5e-3
PREREFACTOR_JSON = os.path.join(pp_dir, 'test', 'control', 'testfiles',
                                'stactrl_prerefactor_v1.json')


def _atol_for(label):
    if label.startswith(("q_sgen", "q_gen", "shunt_step")):
        return OUTPUT_ATOL
    return VALUE_ATOL


class CountingRunpp:
    """runpp wrapper passed to run_control(net, run=...) that counts powerflow calls."""

    def __init__(self):
        self.count = 0

    def __call__(self, net, **kwargs):
        kwargs.pop("run", None)
        self.count += 1
        runpp(net, **kwargs)


def _simple_test_net():
    # same net as test_stactrl.simple_test_net
    net = create_empty_network()
    create_bus(net, 110)
    create_buses(net, 2, 20)
    create_ext_grid(net, 0)
    create_transformer(net, 0, 1, "63 MVA 110/20 kV")
    create_load(net, 1, 3, 0.1)
    create_sgen(net, 2, p_mw=2., sn_mva=10, name="sgen1")
    create_line(net, 1, 2, length_km=0.1, std_type="NAYY 4x50 SE")
    return net


def _multi_feeder_net(n_feeders):
    """110 kV slack bus with n identical 20 kV feeders (trafo, load, line, sgen each)."""
    net = create_empty_network()
    hv = create_bus(net, 110)
    create_ext_grid(net, hv)
    for _ in range(n_feeders):
        mv = create_bus(net, 20)
        lv = create_bus(net, 20)
        create_transformer(net, hv, mv, "63 MVA 110/20 kV")
        create_load(net, mv, 3, 0.1)
        create_sgen(net, lv, p_mw=2., sn_mva=10)
        create_line(net, mv, lv, length_km=0.1, std_type="NAYY 4x50 SE")
    return net


# ----------------------------------------------------------------------------------------------
# scenario builders: each returns (net, run_kwargs, extract) where extract(net) -> {label: value}
# ----------------------------------------------------------------------------------------------

def build_v_ctrl():
    net = _simple_test_net()
    BinarySearchControl(
        net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[0], output_element_in_service=[True], output_values_distribution=[1],
        input_element="res_bus", input_variable="vm_pu", input_element_index=[1],
        set_point=1.02, control_modus="V_ctrl", tol=1e-6)
    return net, {}, lambda net: {
        "vm_bus1": net.res_bus.vm_pu.at[1],
        "q_sgen0": net.res_sgen.q_mvar.at[0]}


def build_v_ctrl_two_sgens():
    net = _simple_test_net()
    create_sgen(net, 2, p_mw=1., sn_mva=10, name="sgen2")
    BinarySearchControl(
        net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[0, 1], output_element_in_service=[True, True],
        output_values_distribution=[0.6, 0.4], input_element="res_bus", input_variable="vm_pu",
        input_element_index=[1], set_point=1.02, control_modus="V_ctrl", tol=1e-6)
    return net, {}, lambda net: {
        "vm_bus1": net.res_bus.vm_pu.at[1],
        "q_sgen0": net.res_sgen.q_mvar.at[0],
        "q_sgen1": net.res_sgen.q_mvar.at[1]}


def build_q_ctrl_line():
    net = _simple_test_net()
    BinarySearchControl(
        net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[0], output_element_in_service=[True], output_values_distribution=[1],
        input_element="res_line", damping_factor=0.9, input_variable=["q_to_mvar"],
        input_element_index=0, set_point=1, control_modus="Q_ctrl", tol=1e-6)
    return net, {}, lambda net: {
        "q_to_line0": net.res_line.q_to_mvar.at[0],
        "q_sgen0": net.res_sgen.q_mvar.at[0]}


def build_q_ctrl_impedance():
    net = _simple_test_net()
    create_impedance(net, 1, 2, sn_mva=1, rft_pu=0.01, xft_pu=0.01, rtf_pu=0.01, xtf_pu=0.01)
    BinarySearchControl(
        net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[0], output_element_in_service=[True], output_values_distribution=[1],
        input_element="res_impedance", damping_factor=0.9, input_variable="q_to_mvar",
        input_element_index=0, set_point=1, control_modus="Q_ctrl", tol=1e-6)
    return net, {}, lambda net: {
        "q_to_imp0": net.res_impedance.q_to_mvar.at[0],
        "q_sgen0": net.res_sgen.q_mvar.at[0]}


def build_q_ctrl_inverted():
    net = _simple_test_net()
    BinarySearchControl(
        net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[0], output_element_in_service=[True], output_values_distribution=[1],
        input_element="res_line", damping_factor=0.9, input_variable=["q_from_mvar"],
        input_inverted=True, input_element_index=0, set_point=1, control_modus="Q_ctrl", tol=1e-6)
    return net, {}, lambda net: {
        "q_from_line0": net.res_line.q_from_mvar.at[0],
        "q_sgen0": net.res_sgen.q_mvar.at[0]}


def build_pf_ctrl_cap():
    net = _simple_test_net()
    BinarySearchControl(
        net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=0, output_element_in_service=True, output_values_distribution=1,
        input_element="res_line", damping_factor=0.9, input_variable="q_to_mvar",
        input_element_index=0, set_point=0.7, control_modus="PF_ctrl_cap", tol=1e-6)
    return net, {}, lambda net: {
        "phi_to_line0": np.arctan(net.res_line.q_to_mvar.at[0] / net.res_line.p_to_mw.at[0]),
        "q_sgen0": net.res_sgen.q_mvar.at[0]}


def build_pf_ctrl_ind():
    net = _simple_test_net()
    BinarySearchControl(
        net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=0, output_element_in_service=True, output_values_distribution=1,
        input_element="res_line", damping_factor=0.9, input_variable="q_to_mvar",
        input_element_index=0, set_point=0.7, control_modus="PF_ctrl_ind", tol=1e-6)
    return net, {}, lambda net: {
        "phi_to_line0": np.arctan(net.res_line.q_to_mvar.at[0] / net.res_line.p_to_mw.at[0]),
        "q_sgen0": net.res_sgen.q_mvar.at[0]}


def build_tan_phi_ctrl():
    net = _simple_test_net()
    BinarySearchControl(
        net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=0, output_element_in_service=True, output_values_distribution=1,
        input_element="res_trafo", input_variable="q_lv_mvar", input_element_index=0,
        set_point=2, control_modus="tan_phi_ctrl", tol=1e-6)
    return net, {}, lambda net: {
        "tan_phi_trafo0": net.res_trafo.q_lv_mvar.at[0] / net.res_trafo.p_lv_mw.at[0],
        "q_sgen0": net.res_sgen.q_mvar.at[0]}


def build_v_ctrl_q_droop():
    net = _simple_test_net()
    bsc = BinarySearchControl(
        net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[0], output_element_in_service=[True], output_values_distribution=[1],
        input_element="res_trafo", input_variable="q_hv_mvar", input_element_index=[0],
        set_point=1.02, control_modus="V_ctrl_Q_droop", bus_idx=1, tol=1e-6)
    DroopControl(net, name="DC1", q_droop_mvar=40, bus_idx=1, vm_set_pu=1.02,
                 controller_idx=bsc.index, control_modus="V_ctrl_Q_droop", tol=1e-6)
    return net, {}, lambda net: {
        "vm_bus1": net.res_bus.vm_pu.at[1],
        "q_hv_trafo0": net.res_trafo.q_hv_mvar.at[0],
        "q_sgen0": net.res_sgen.q_mvar.at[0]}


def build_q_ctrl_v_droop_deadband():
    net = _simple_test_net()
    net.load.loc[0, "p_mw"] = 60  # create voltage drop at bus 1
    bsc = BinarySearchControl(
        net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[0], output_element_in_service=[True], output_values_distribution=[1],
        input_element="res_line", damping_factor=0.9, input_variable=["q_from_mvar"],
        input_inverted=True, input_element_index=0, set_point=1, control_modus="Q_ctrl_V_droop",
        tol=1e-6)
    DroopControl(net, name="DC1", q_droop_mvar=40, bus_idx=1, vm_set_pu=1, vm_set_ub=1.005,
                 vm_set_lb=0.995, controller_idx=bsc.index, control_modus="Q_ctrl_V_droop")
    return net, {}, lambda net: {
        "vm_bus1": net.res_bus.vm_pu.at[1],
        "q_from_line0": net.res_line.q_from_mvar.at[0],
        "q_sgen0": net.res_sgen.q_mvar.at[0]}


def build_v_droop_local():
    # gen voltage setpoint controlled so that the gen's Q output follows a local V droop,
    # mirroring the PowerFactory converter pattern (pp_import_functions.py ~2216)
    net = _simple_test_net()
    # start away from the droop fixed point so the controllers actually iterate
    create_gen(net, 2, p_mw=2., vm_pu=1.02, sn_mva=10)
    bsc = BinarySearchControl(
        net, name="BSC1", ctrl_in_service=True, output_element="gen", output_variable="vm_pu",
        output_element_index=[0], output_element_in_service=[True], output_values_distribution=[1],
        input_element="res_gen", input_variable="q_mvar", input_element_index=[0],
        input_inverted=[False], set_point=1.0, control_modus="V_ctrl_Q_droop_local", bus_idx=2,
        tol=1e-5)
    VDroopControl_local(net, q_droop_mvar=20, controller_idx=bsc.index, bus_idx=2,
                        control_modus="V_ctrl_Q_droop_local", q_set_mvar=0.5, vm_set_pu_bsc=1.0,
                        tol=1e-5)
    return net, {}, lambda net: {
        "vm_bus2": net.res_bus.vm_pu.at[2],
        "q_gen0": net.res_gen.q_mvar.at[0]}


def build_qlims_q_ctrl():
    net = _simple_test_net()
    net.sgen['min_q_mvar'] = -0.5
    net.sgen['max_q_mvar'] = 0.5
    BinarySearchControl(
        net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[0], output_element_in_service=[True], output_values_distribution=[1],
        input_element="res_line", damping_factor=0.9, input_variable=["q_to_mvar"],
        input_element_index=0, set_point=1, control_modus="Q_ctrl", tol=1e-6)
    return net, {"enforce_q_lims": True}, lambda net: {
        "q_sgen0": net.res_sgen.q_mvar.at[0]}


def build_qlims_v_ctrl():
    net = _simple_test_net()
    net.sgen['min_q_mvar'] = -0.7
    net.sgen['max_q_mvar'] = 0.7
    BinarySearchControl(
        net, name="BSC1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[0], output_element_in_service=[True], output_values_distribution=[1],
        input_element="res_bus", input_variable="vm_pu", input_element_index=[1],
        set_point=1.02, control_modus="V_ctrl", tol=1e-6)
    return net, {"enforce_q_lims": True}, lambda net: {
        "vm_bus1": net.res_bus.vm_pu.at[1],
        "q_sgen0": net.res_sgen.q_mvar.at[0]}


def build_shunt_step():
    net = create_empty_network()
    b = create_buses(net, 2, 110)
    create_ext_grid(net, b[0])
    create_line_from_parameters(net, from_bus=b[0], to_bus=b[1], length_km=50,
                                r_ohm_per_km=0.1021, x_ohm_per_km=0.1570796, max_i_ka=0.461,
                                c_nf_per_km=130)
    create_shunt(net, bus=b[1], q_mvar=-50, p_mw=0, step=1, max_step=5)
    BinarySearchControl(
        net, ctrl_in_service=True, output_element='shunt', output_variable='step',
        output_element_index=[0], output_element_in_service=[True], output_values_distribution=[1],
        input_element='res_bus', input_variable='vm_pu', input_element_index=[1],
        set_point=1.08, control_modus="V_ctrl", tol=1e-6)
    return net, {}, lambda net: {
        "vm_bus1": net.res_bus.vm_pu.at[1],
        "shunt_step": net.shunt.step.at[0]}


def build_three_controllers():
    # three stations on one HV bus: interacting controllers of different modi
    net = _multi_feeder_net(3)
    # feeder k: buses (1+2k, 2+2k), trafo k, line k, sgen k
    BinarySearchControl(
        net, name="V", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[0], output_element_in_service=[True], output_values_distribution=[1],
        input_element="res_bus", input_variable="vm_pu", input_element_index=[1],
        set_point=1.02, control_modus="V_ctrl", tol=1e-6)
    BinarySearchControl(
        net, name="Q", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[1], output_element_in_service=[True], output_values_distribution=[1],
        input_element="res_line", damping_factor=0.9, input_variable=["q_to_mvar"],
        input_element_index=1, set_point=0.5, control_modus="Q_ctrl", tol=1e-6)
    BinarySearchControl(
        net, name="T", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[2], output_element_in_service=[True], output_values_distribution=[1],
        input_element="res_trafo", input_variable="q_lv_mvar", input_element_index=2,
        set_point=0.5, control_modus="tan_phi_ctrl", tol=1e-6)
    return net, {}, lambda net: {
        "vm_bus1": net.res_bus.vm_pu.at[1],
        "q_to_line1": net.res_line.q_to_mvar.at[1],
        "tan_phi_trafo2": net.res_trafo.q_lv_mvar.at[2] / net.res_trafo.p_lv_mw.at[2],
        "q_sgen0": net.res_sgen.q_mvar.at[0],
        "q_sgen1": net.res_sgen.q_mvar.at[1],
        "q_sgen2": net.res_sgen.q_mvar.at[2]}


SCENARIOS = {
    "v_ctrl": build_v_ctrl,
    "v_ctrl_two_sgens": build_v_ctrl_two_sgens,
    "q_ctrl_line": build_q_ctrl_line,
    "q_ctrl_impedance": build_q_ctrl_impedance,
    "q_ctrl_inverted": build_q_ctrl_inverted,
    "pf_ctrl_cap": build_pf_ctrl_cap,
    "pf_ctrl_ind": build_pf_ctrl_ind,
    "tan_phi_ctrl": build_tan_phi_ctrl,
    "v_ctrl_q_droop": build_v_ctrl_q_droop,
    "q_ctrl_v_droop_deadband": build_q_ctrl_v_droop_deadband,
    "v_droop_local": build_v_droop_local,
    "qlims_q_ctrl": build_qlims_q_ctrl,
    "qlims_v_ctrl": build_qlims_v_ctrl,
    "shunt_step": build_shunt_step,
    "three_controllers": build_three_controllers,
}

# recorded on pre-refactor develop (0c6a1f0a6, 2026-07) -- see module docstring
RECORDS = {
    'v_ctrl': {
        'runpp_count': 5,
        'values': {
            'vm_bus1': 1.019999996915152,
            'q_sgen0': 7.265806884389418,
        },
    },
    'v_ctrl_two_sgens': {
        'runpp_count': 5,
        'values': {
            'vm_bus1': 1.0199999969589186,
            'q_sgen0': 4.347800920511612,
            'q_sgen1': 2.8985339470077416,
        },
    },
    'q_ctrl_line': {
        'runpp_count': 3,
        'values': {
            'q_to_line0': 0.9999999983680944,
            'q_sgen0': 0.9999999983679086,
        },
    },
    'q_ctrl_impedance': {
        'runpp_count': 4,
        'values': {
            'q_to_imp0': 0.9999996694339104,
            'q_sgen0': 109.23659617195352,
        },
    },
    'q_ctrl_inverted': {
        'runpp_count': 4,
        'values': {
            'q_from_line0': -0.999999999577785,
            'q_sgen0': 0.997450108031786,
        },
    },
    'pf_ctrl_cap': {
        'runpp_count': 3,
        'values': {
            'phi_to_line0': -0.7953988279821941,
            'q_sgen0': -2.040408113453427,
        },
    },
    'pf_ctrl_ind': {
        'runpp_count': 3,
        'values': {
            'phi_to_line0': 0.7953988307386679,
            'q_sgen0': 2.040408124707689,
        },
    },
    'tan_phi_ctrl': {
        'runpp_count': 4,
        'values': {
            'tan_phi_trafo0': 1.9999993561180684,
            'q_sgen0': -1.9049229990555525,
        },
    },
    'v_ctrl_q_droop': {
        'runpp_count': 5,
        'values': {
            'vm_bus1': 1.0020245144045044,
            'q_hv_trafo0': -0.7190194237899732,
            'q_sgen0': 0.8332080621680016,
        },
    },
    'q_ctrl_v_droop_deadband': {
        'runpp_count': 5,
        'values': {
            'vm_bus1': 0.9863775299392599,
            'q_from_line0': -1.34489880242843,
            'q_sgen0': 1.342454080724194,
        },
    },
    'v_droop_local': {
        'runpp_count': 6,
        'values': {
            'vm_bus2': 1.0017284837979823,
            'q_gen0': 0.4654303789138794,
        },
    },
    'qlims_q_ctrl': {
        'runpp_count': 3,
        'values': {
            'q_sgen0': 0.5,
        },
    },
    'qlims_v_ctrl': {
        'runpp_count': 3,
        'values': {
            'vm_bus1': 1.0016454077928925,
            'q_sgen0': 0.7,
        },
    },
    'shunt_step': {
        'runpp_count': 5,
        'values': {
            'vm_bus1': 1.0799999784869583,
            'shunt_step': 2.0752751205395854,
        },
    },
    'three_controllers': {
        'runpp_count': 5,
        'values': {
            'vm_bus1': 1.019999996915152,
            'q_to_line1': 0.49999999918549787,
            'tan_phi_trafo2': 0.4999999997653505,
            'q_sgen0': 7.265806884389418,
            'q_sgen1': 0.4999999991854217,
            'q_sgen2': -0.4028800134651635,
        },
    },
    'prerefactor_json': {
        'runpp_count': 6,
        'keys': [
            ('res_bus', 1, 'vm_pu'),
            ('res_line', 1, 'q_to_mvar'),
            ('res_line', 2, 'q_to_mvar'),
            ('res_line', 2, 'p_to_mw'),
            ('res_trafo', 3, 'q_lv_mvar'),
            ('res_trafo', 3, 'p_lv_mw'),
            ('res_bus', 9, 'vm_pu'),
            ('res_trafo', 4, 'q_hv_mvar'),
            ('res_bus', 12, 'vm_pu'),
            ('res_gen', 0, 'q_mvar'),
            ('res_sgen', 0, 'q_mvar'),
            ('res_sgen', 1, 'q_mvar'),
            ('res_sgen', 2, 'q_mvar'),
            ('res_sgen', 3, 'q_mvar'),
            ('res_sgen', 4, 'q_mvar'),
        ],
        'values': [
            1.019999996915152,
            0.5000000005905079,
            0.9686442123565855,
            2.000000000000376,
            -0.5003347375516176,
            -1.000669475572584,
            1.0020245144045037,
            -0.719019423789689,
            1.0017284837979825,
            0.4654303789138794,
            7.26580688439267,
            0.5000000005886843,
            0.9686442123544164,
            -0.4028800134649547,
            0.8332080621679098,
        ],
    },
}


def run_scenario(name):
    net, run_kwargs, extract = SCENARIOS[name]()
    counter = CountingRunpp()
    run_control(net, run=counter, **run_kwargs)
    assert all(net.controller.object[i].converged for i in net.controller.index)
    return counter.count, extract(net)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_characterization(name):
    if name not in RECORDS:
        pytest.skip(f"no record for scenario {name} -- record with --record first")
    runpp_count, values = run_scenario(name)
    expected = RECORDS[name]
    assert runpp_count <= expected["runpp_count"], (
        f"{name}: {runpp_count} powerflow calls, baseline is {expected['runpp_count']}")
    for label, expected_value in expected["values"].items():
        assert values[label] == pytest.approx(expected_value, abs=_atol_for(label)), (
            f"{name}: {label} = {values[label]}, recorded {expected_value}")


def test_load_prerefactor_json_and_solve():
    """A net saved with the pre-refactor implementation must load and solve unchanged.

    Guards attribute-level backward compatibility: from_json restores controller __dict__
    without calling __init__.
    """
    if not os.path.isfile(PREREFACTOR_JSON):
        pytest.skip("frozen pre-refactor fixture not generated yet")
    net = from_json(PREREFACTOR_JSON)
    counter = CountingRunpp()
    run_control(net, run=counter)
    assert all(net.controller.object[i].converged for i in net.controller.index)
    if "prerefactor_json" in RECORDS:
        expected = RECORDS["prerefactor_json"]
        assert counter.count <= expected["runpp_count"]
        for (element, index, column), expected_value in zip(
                expected["keys"], expected["values"]):
            if element in ("res_sgen", "res_gen"):
                atol = OUTPUT_ATOL
            elif element == "res_bus":
                atol = VALUE_ATOL
            else:
                atol = 1e-4  # branch measurements shift slightly with the realized outputs
            assert net[element].at[index, column] == pytest.approx(
                expected_value, abs=atol), (element, index, column)


def test_json_roundtrip_after_refactor(tmp_path):
    """Controllers keep working after to_json/from_json and derived state is not serialized."""
    from pandapower.file_io import to_json

    net, run_kwargs, extract = SCENARIOS["v_ctrl_two_sgens"]()
    run_control(net, **run_kwargs)  # populates derived state (_vstate etc.) on the controller
    ctrl = net.controller.object.at[0]
    serialized = ctrl.to_dict()
    assert "_vstate" not in serialized
    assert "_linked_droop_objs" not in serialized

    json_file = os.path.join(tmp_path, "roundtrip.json")
    to_json(net, json_file)
    net2 = from_json(json_file)
    counter = CountingRunpp()
    run_control(net2, run=counter, **run_kwargs)
    assert all(net2.controller.object[i].converged for i in net2.controller.index)
    reference, restored = extract(net), extract(net2)
    for label in reference:
        assert restored[label] == pytest.approx(reference[label], abs=VALUE_ATOL), label


def test_out_of_service_output_element_midrun():
    """in_service changes between two run_control calls must be picked up (derived positional
    state is rebuilt in initialize_control)."""
    net, run_kwargs, extract = SCENARIOS["v_ctrl_two_sgens"]()
    run_control(net, **run_kwargs)
    assert net.res_sgen.q_mvar.at[1] != 0.0
    net.sgen.at[1, "in_service"] = False
    run_control(net, **run_kwargs)
    assert all(net.controller.object[i].converged for i in net.controller.index)
    # remaining sgen carries the whole station, target voltage still reached; the
    # out-of-service sgen contributes nothing (its stale setpoint is not written/applied)
    assert net.res_bus.vm_pu.at[1] == pytest.approx(1.02, abs=1e-5)
    assert net.res_sgen.q_mvar.at[1] == 0.0


def _record():
    records = {}
    for name in SCENARIOS:
        try:
            runpp_count, values = run_scenario(name)
        except Exception as err:  # noqa: BLE001 - report scenario failures during recording
            print(f"# scenario {name} FAILED: {err!r}")
            continue
        records[name] = {"runpp_count": runpp_count, "values": values}
    print("RECORDS = {")
    for name, rec in records.items():
        print(f"    {name!r}: {{")
        print(f"        'runpp_count': {rec['runpp_count']},")
        print("        'values': {")
        for label, value in rec["values"].items():
            print(f"            {label!r}: {float(value)!r},")
        print("        },")
        print("    },")
    print("}")


if __name__ == '__main__':
    import sys
    if "--record" in sys.argv:
        _record()
    else:
        pytest.main(['-s', __file__])
