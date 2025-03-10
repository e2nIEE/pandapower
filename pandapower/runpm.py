# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.


import os
from pandapower.auxiliary import _add_ppc_options, _add_opf_options
from pandapower.converter.pandamodels.from_pm import read_ots_results, read_tnep_results
from pandapower.opf.pm_storage import add_storage_opf_settings, read_pm_storage_results
from pandapower.opf.run_pandamodels import _runpm


def runpm(net, julia_file=None, pp_to_pm_callback=None, calculate_voltage_angles=True,
          trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
          correct_pm_network_data=True, silence=True, pm_model="ACPPowerModel", pm_solver="ipopt",
          pm_mip_solver="cbc", pm_nl_solver="ipopt", pm_time_limits=None, pm_log_level=0,
          delete_buffer_file=True, pm_file_path = None, opf_flow_lim="S", pm_tol=1e-8,
          pdm_dev_mode=False, **kwargs):  # pragma: no cover
    """
    Runs  optimal power flow from PowerModels.jl via PandaModels.jl

    Flexibilities, constraints and cost parameters are defined in the pandapower element tables.

    Flexibilities can be defined in net.sgen / net.gen /net.load
    net.sgen.controllable if a static generator is controllable. If False,
    the active and reactive power are assigned as in a normal power flow. If True, the following
    flexibilities apply:

        - net.sgen.min_p_mw / net.sgen.max_p_mw
        - net.sgen.min_q_mvar / net.sgen.max_q_mvar
        - net.load.min_p_mw / net.load.max_p_mw
        - net.load.min_q_mvar / net.load.max_q_mvar
        - net.gen.min_p_mw / net.gen.max_p_mw
        - net.gen.min_q_mvar / net.gen.max_q_mvar
        - net.ext_grid.min_p_mw / net.ext_grid.max_p_mw
        - net.ext_grid.min_q_mvar / net.ext_grid.max_q_mvar
        - net.dcline.min_q_to_mvar / net.dcline.max_q_to_mvar / net.dcline.min_q_from_mvar / net.dcline.max_q_from_mvar

    Controllable loads behave just like controllable static generators. It must be stated if they are controllable.
    Otherwise, they are not respected as flexibilities.
    Dc lines are controllable per default

    Network constraints can be defined for buses, lines and transformers the elements in the following columns:

        - net.bus.min_vm_pu / net.bus.max_vm_pu
        - net.line.max_loading_percent
        - net.trafo.max_loading_percent
        - net.trafo3w.max_loading_percent

    How these costs are combined into a cost function depends on the cost_function parameter.

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **julia_file** (str, None) - path to a custom julia optimization file

        **pp_to_pm_callback** (function, None) - callback function to add data to the PowerModels data structure

        **correct_pm_network_data** (bool, True) - checks if network data is correct. If not tries to correct it

        **silence** (bool, True) - Suppresses information and warning messages output by PowerModels

        **pm_model** (str, "ACPPowerModel") - The PowerModels.jl model to use

        **pm_solver** (str, "ipopt") - The "main" power models solver

        **pm_mip_solver** (str, "cbc") - The mixed integer solver (when "main" solver == juniper)

        **pm_nl_solver** (str, "ipopt") - The nonlinear solver (when "main" solver == juniper)

        **pm_time_limits** (Dict, None) - Time limits in seconds for power models interface. To be set as a dict like {"pm_time_limit": 300., "pm_nl_time_limit": 300., "pm_mip_time_limit": 300.}

        **pm_log_level** (int, 0) - solver log level in power models

        **delete_buffer_file** (Bool, True) - If True, the .json file used by powermodels will be deleted after optimization.

        **pm_file_path** (str, None) - Specifiy the filename, under which the .json file for powermodels is stored. If you want to keep the file after optimization, you should also set delete_buffer_file to False!

        **opf_flow_lim** (str, "I") - Quantity to limit for branch flow constraints, in line with matpower's
                "opf.flowlim" parameter:

                    "S" - apparent power flow (limit in MVA),
                    "I" - current magnitude (limit in MVA at 1 p.u. voltage)

        **pm_tol** (float, 1e-8) - default desired convergence tolerance for solver to use.

        **pdm_dev_mode** (bool, False) - If True, the develope mode of PdM is called.
    """
    ac = True if "DC" not in pm_model else False
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=ac, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file="run_powermodels_opf", pm_solver=pm_solver, pm_model=pm_model,
                     correct_pm_network_data=correct_pm_network_data, silence=silence, pm_mip_solver=pm_mip_solver,
                     pm_nl_solver=pm_nl_solver, pm_time_limits=pm_time_limits, pm_log_level=pm_log_level,
                     opf_flow_lim=opf_flow_lim, pm_tol=pm_tol)

    _runpm(net, delete_buffer_file=delete_buffer_file, pm_file_path=pm_file_path, pdm_dev_mode=pdm_dev_mode)


def runpm_dc_opf(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
                 trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
                 correct_pm_network_data=True, silence=True, pm_model="DCPPowerModel", pm_solver="ipopt",
                 pm_time_limits=None, pm_log_level=0, delete_buffer_file=True, pm_file_path = None,
                 pm_tol=1e-8, pdm_dev_mode=False, **kwargs):
    """
    Runs linearized optimal power flow from PowerModels.jl via PandaModels.jl
    """
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=False, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file="run_powermodels_opf",
                     correct_pm_network_data=correct_pm_network_data, silence=silence, pm_model=pm_model, pm_solver=pm_solver,
                     pm_time_limits=pm_time_limits, pm_log_level=pm_log_level, opf_flow_lim="S", pm_tol=pm_tol)

    _runpm(net, delete_buffer_file=delete_buffer_file, pm_file_path=pm_file_path, pdm_dev_mode=pdm_dev_mode)


def runpm_ac_opf(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
                 trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
                 pm_solver="ipopt", correct_pm_network_data=True, silence=True,
                 pm_time_limits=None, pm_log_level=0, pm_file_path=None, delete_buffer_file=True,
                 opf_flow_lim="S", pm_tol=1e-8, pdm_dev_mode=False, **kwargs):
    """
    Runs non-linear optimal power flow from PowerModels.jl via PandaModels.jl
    """
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file="run_powermodels_opf", pm_model="ACPPowerModel", pm_solver=pm_solver,
                     correct_pm_network_data=correct_pm_network_data, silence=silence, pm_time_limits=pm_time_limits,
                     pm_log_level=pm_log_level, opf_flow_lim=opf_flow_lim, pm_tol=pm_tol)

    _runpm(net, delete_buffer_file=delete_buffer_file, pm_file_path=pm_file_path, pdm_dev_mode=pdm_dev_mode)



def runpm_tnep(net, julia_file=None, pp_to_pm_callback=None, calculate_voltage_angles=True,
               trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
               pm_model="ACPPowerModel", pm_solver="juniper", correct_pm_network_data=True, silence=True,
               pm_nl_solver="ipopt", pm_mip_solver="cbc", pm_time_limits=None, pm_log_level=0,
               delete_buffer_file=True, pm_file_path=None, opf_flow_lim="S", pm_tol=1e-8,
               pdm_dev_mode=False, **kwargs):
    """
    Runs transmission network extension planning (tnep) optimization from PowerModels.jl via PandaModels.jl
    """
    ac = True if "DC" not in pm_model else False
    if pm_solver is None:
        if pm_model == "DCPPowerModel":
            pm_solver = "ipopt"
        else:
            pm_solver = "juniper"

    if "ne_line" not in net:
        raise ValueError("ne_line DataFrame missing in net. Please define to run tnep")
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=ac, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file="run_powermodels_tnep", pm_model=pm_model, pm_solver=pm_solver,
                     correct_pm_network_data=correct_pm_network_data, silence=silence, pm_nl_solver=pm_nl_solver,
                     pm_mip_solver=pm_mip_solver, pm_time_limits=pm_time_limits, pm_log_level=pm_log_level,
                     opf_flow_lim=opf_flow_lim, pm_tol=pm_tol)

    _runpm(net, delete_buffer_file=delete_buffer_file, pm_file_path=pm_file_path, pdm_dev_mode=pdm_dev_mode)
    read_tnep_results(net)


def runpm_ots(net, julia_file=None, pp_to_pm_callback=None, calculate_voltage_angles=True,
              trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
              pm_model="DCPPowerModel", pm_solver="juniper", pm_nl_solver="ipopt",
              pm_mip_solver="cbc", correct_pm_network_data=True, silence=True, pm_time_limits=None,
              pm_log_level=0, delete_buffer_file=True, pm_file_path=None, opf_flow_lim="S", pm_tol=1e-8,
              pdm_dev_mode=False, **kwargs):
    """
    Runs optimal transmission switching (OTS) optimization from PowerModels.jl via PandaModels.jl
    """
    ac = True if "DC" not in pm_model else False
    if pm_solver is None:
        pm_solver = "juniper"

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=ac, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file="run_powermodels_ots", pm_model=pm_model, pm_solver=pm_solver,
                     correct_pm_network_data=correct_pm_network_data, silence=silence, pm_mip_solver=pm_mip_solver,
                     pm_nl_solver=pm_nl_solver, pm_time_limits=pm_time_limits, pm_log_level=pm_log_level,
                     opf_flow_lim="S", pm_tol=pm_tol)

    _runpm(net, delete_buffer_file=delete_buffer_file, pm_file_path=pm_file_path, pdm_dev_mode=pdm_dev_mode)
    read_ots_results(net)

def runpm_storage_opf(net, from_time_step, to_time_step, calculate_voltage_angles=True,
                      trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
                      n_timesteps=24, time_elapsed=1., correct_pm_network_data=True, silence=True,
                      pm_solver="juniper", pm_mip_solver="cbc", pm_nl_solver="ipopt",
                      pm_model="ACPPowerModel", pm_time_limits=None, pm_log_level=0,
                      opf_flow_lim="S", charge_efficiency=1., discharge_efficiency=1.,
                      standby_loss=1e-8, p_loss=1e-8, q_loss=1e-8, pm_tol=1e-4, pdm_dev_mode=False,
                      delete_buffer_file=True, pm_file_path = None, **kwargs):
    """
    Runs a non-linear power system optimization with storages and time series using PandaModels.jl.
    """
    kwargs["from_time_step"] = from_time_step
    kwargs["to_time_step"] = to_time_step

    ac = True if "DC" not in pm_model else False
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=ac, init="flat", numba=True,
                     pp_to_pm_callback=add_storage_opf_settings, julia_file="run_powermodels_multi_storage",
                     correct_pm_network_data=correct_pm_network_data, silence=silence, pm_model=pm_model,
                     pm_solver=pm_solver, pm_time_limits=pm_time_limits,
                     pm_log_level=pm_log_level, opf_flow_lim=opf_flow_lim, pm_tol=pm_tol, pdm_dev_mode=pdm_dev_mode)

    net._options["n_time_steps"] = to_time_step - from_time_step
    net._options["time_elapsed"] = time_elapsed

    _runpm(net, delete_buffer_file=delete_buffer_file, pm_file_path=pm_file_path, pdm_dev_mode=pdm_dev_mode, **kwargs)



def runpm_vstab(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
                trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
                pm_model="ACPPowerModel", pm_solver="ipopt", correct_pm_network_data=True, silence=True,
                pm_time_limits=None, pm_log_level=0, pm_file_path = None, delete_buffer_file=True,
                opf_flow_lim="S", pm_tol=1e-8, pdm_dev_mode=False, **kwargs):
    """
    Runs non-linear problem for voltage deviation minimization from PandaModels.jl.
    """
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file="run_pandamodels_vstab", pm_model=pm_model, pm_solver=pm_solver,
                     correct_pm_network_data=correct_pm_network_data, silence=silence, pm_time_limits=pm_time_limits,
                     pm_log_level=pm_log_level, opf_flow_lim=opf_flow_lim, pm_tol=pm_tol)

    _runpm(net, delete_buffer_file=delete_buffer_file, pm_file_path=pm_file_path, pdm_dev_mode=pdm_dev_mode)


def runpm_multi_vstab(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
                      trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
                      pm_model="ACPPowerModel", pm_solver="ipopt", correct_pm_network_data=True, silence=True,
                      pm_time_limits=None, pm_log_level=0, pm_file_path = None, delete_buffer_file=True,
                      opf_flow_lim="S", pm_tol=1e-8, pdm_dev_mode=False, **kwargs):
    """
    Runs non-linear problem for time-series voltage deviation minimization from PandaModels.jl.
    """
    try:
        assert isinstance(kwargs["from_time_step"], int)
        assert isinstance(kwargs["from_time_step"], int)
    except (KeyError, AssertionError):
        raise ValueError ("For time series optimization, you need define 'from_time_step' and 'to_time_step', e.g.," +
                          " 'from_time_step = 0', 'to_time_step = 15'.")
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file="run_pandamodels_multi_vstab", pm_model=pm_model, pm_solver=pm_solver,
                     correct_pm_network_data=correct_pm_network_data, silence=silence, pm_time_limits=pm_time_limits,
                     pm_log_level=pm_log_level, opf_flow_lim=opf_flow_lim, pm_tol=pm_tol)

    _runpm(net, delete_buffer_file=delete_buffer_file, pm_file_path=pm_file_path, pdm_dev_mode=pdm_dev_mode, **kwargs)


def runpm_qflex(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
                trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
                pm_model="ACPPowerModel", pm_solver="ipopt", correct_pm_network_data=True, silence=True,
                pm_time_limits=None, pm_log_level=0, pm_file_path = None, delete_buffer_file=True,
                opf_flow_lim="S", pm_tol=1e-8, pdm_dev_mode=False, **kwargs):
    """
    Runs non-linear optimization for maintaining q-setpoint.
    """
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file="run_pandamodels_qflex", pm_model=pm_model, pm_solver=pm_solver,
                     correct_pm_network_data=correct_pm_network_data, silence=silence, pm_time_limits=pm_time_limits,
                     pm_log_level=pm_log_level, opf_flow_lim=opf_flow_lim, pm_tol=pm_tol)

    _runpm(net, delete_buffer_file=delete_buffer_file, pm_file_path=pm_file_path, pdm_dev_mode=pdm_dev_mode)


def runpm_multi_qflex(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
                      trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
                      pm_model="ACPPowerModel", pm_solver="ipopt", correct_pm_network_data=True, silence=True,
                      pm_time_limits=None, pm_log_level=0, pm_file_path = None, delete_buffer_file=True,
                      opf_flow_lim="S", pm_tol=1e-8, pdm_dev_mode=False, **kwargs):
    """
    Runs non-linear optimization for maintaining q-setpoint over a time-series.
    """
    try:
        assert isinstance(kwargs["from_time_step"], int)
        assert isinstance(kwargs["from_time_step"], int)
    except (KeyError, AssertionError):
        raise ValueError ("For time series optimization, you need define 'from_time_step' and 'to_time_step', e.g.," +
                          " 'from_time_step = 0', 'to_time_step = 15'.")
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file="run_pandamodels_multi_qflex", pm_model=pm_model, pm_solver=pm_solver,
                     correct_pm_network_data=correct_pm_network_data, silence=silence, pm_time_limits=pm_time_limits,
                     pm_log_level=pm_log_level, opf_flow_lim=opf_flow_lim, pm_tol=pm_tol)

    _runpm(net, delete_buffer_file=delete_buffer_file, pm_file_path=pm_file_path, pdm_dev_mode=pdm_dev_mode, **kwargs)


def runpm_ploss(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
                trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
                pm_model="ACPPowerModel", pm_solver="ipopt", correct_pm_network_data=True, silence=True,
                pm_time_limits=None, pm_log_level=0, pm_file_path = None, delete_buffer_file=True,
                opf_flow_lim="S", pm_tol=1e-8, pdm_dev_mode=False, **kwargs):
    """
    Runs non-linear optimization for active power loss reduction.
    """
    for elm in ["line", "trafo"]:
        if "pm_param/target_branch" in net[elm].columns:
            net[elm]["pm_param/side"] = None
            net[elm]["pm_param/side"][net[elm]["pm_param/target_branch"]==True] = "from"

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file="run_pandamodels_ploss", pm_model=pm_model, pm_solver=pm_solver,
                     correct_pm_network_data=correct_pm_network_data, silence=silence, pm_time_limits=pm_time_limits,
                     pm_log_level=pm_log_level, opf_flow_lim=opf_flow_lim, pm_tol=pm_tol)

    _runpm(net, delete_buffer_file=delete_buffer_file, pm_file_path=pm_file_path, pdm_dev_mode=pdm_dev_mode)


def runpm_loading(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
                trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
                pm_model="ACPPowerModel", pm_solver="ipopt", correct_pm_network_data=True, silence=True,
                pm_time_limits=None, pm_log_level=0, pm_file_path = None, delete_buffer_file=True,
                opf_flow_lim="S", pm_tol=1e-8, pdm_dev_mode=False, **kwargs):
    """
    Runs non-linear optimization for active power loss reduction.
    """
    for elm in ["line", "trafo"]:
        if "pm_param/target_branch" in net[elm].columns:
            net[elm]["pm_param/side"] = None
            net[elm]["pm_param/side"][net[elm]["pm_param/target_branch"]==True] = "from"

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file="run_pandamodels_loading", pm_model=pm_model, pm_solver=pm_solver,
                     correct_pm_network_data=correct_pm_network_data, silence=silence, pm_time_limits=pm_time_limits,
                     pm_log_level=pm_log_level, opf_flow_lim=opf_flow_lim, pm_tol=pm_tol)

    _runpm(net, delete_buffer_file=delete_buffer_file, pm_file_path=pm_file_path, pdm_dev_mode=pdm_dev_mode)


def runpm_pf(net, julia_file=None, pp_to_pm_callback=None, calculate_voltage_angles=True,
             trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
             correct_pm_network_data=True, silence=True, pm_model="ACPPowerModel", pm_solver="ipopt",
             pm_mip_solver="cbc", pm_nl_solver="ipopt", pm_time_limits=None, pm_log_level=0,
             delete_buffer_file=True, pm_file_path = None, opf_flow_lim="S", pm_tol=1e-8,
             pdm_dev_mode=False, **kwargs):  # pragma: no cover
    """
    Runs power flow from PowerModels.jl via PandaModels.jl
    """
    ac = True if "DC" not in pm_model else False
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=ac, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file="run_powermodels_pf", pm_solver=pm_solver, pm_model=pm_model,
                     correct_pm_network_data=correct_pm_network_data, silence=silence, pm_mip_solver=pm_mip_solver,
                     pm_nl_solver=pm_nl_solver, pm_time_limits=pm_time_limits, pm_log_level=pm_log_level,
                     opf_flow_lim=opf_flow_lim, pm_tol=pm_tol)

    _runpm(net, delete_buffer_file=delete_buffer_file, pm_file_path=pm_file_path, pdm_dev_mode=pdm_dev_mode)