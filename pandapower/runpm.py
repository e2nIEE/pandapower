# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

from pandapower import pp_dir
from pandapower.auxiliary import _add_ppc_options, _add_opf_options
from pandapower.opf.pm_conversion import _runpm
from pandapower.opf.pm_ots import read_ots_results
from pandapower.opf.pm_storage import add_storage_opf_settings, read_pm_storage_results
from pandapower.opf.pm_tnep import read_tnep_results


def runpm(net, julia_file, pp_to_pm_callback=None, calculate_voltage_angles=True,
          trafo_model="t", delta=0, trafo3w_losses="hv", check_connectivity=True):  # pragma: no cover
    """
    Runs a power system optimization using PowerModels.jl. with a custom julia file.
    
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

     """
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file=julia_file)
    _runpm(net)


def runpm_dc_opf(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
                 trafo_model="t", delta=0, trafo3w_losses="hv", check_connectivity=True):  # pragma: no cover
    """
    Runs a linearized power system optimization using PowerModels.jl.
    
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
        **pp_to_pm_callback** (function, None) - callback function to add data to the PowerModels data structure

     """
    julia_file = os.path.join(pp_dir, "opf", 'run_powermodels_dc.jl')

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file=julia_file)
    _runpm(net)


def runpm_ac_opf(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
                 trafo_model="t", delta=0, trafo3w_losses="hv", check_connectivity=True):  # pragma: no cover
    """
    Runs a non-linear power system optimization using PowerModels.jl.

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

        **pp_to_pm_callback** (function, None) - callback function to add data to the PowerModels data structure

     """
    julia_file = os.path.join(pp_dir, "opf", 'run_powermodels_ac.jl')

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file=julia_file)
    _runpm(net)


def runpm_tnep(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
               trafo_model="t", delta=0, trafo3w_losses="hv", check_connectivity=True):  # pragma: no cover
    """
    Runs a non-linear transmission network extension planning (tnep) optimization using PowerModels.jl.

    see above

     """
    julia_file = os.path.join(pp_dir, "opf", 'run_powermodels_tnep.jl')

    if "ne_line" not in net:
        raise ValueError("ne_line DataFrame missing in net. Please define to run tnep")
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file=julia_file)
    _runpm(net)
    read_tnep_results(net)


def runpm_ots(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
              trafo_model="t", delta=0, trafo3w_losses="hv", check_connectivity=True):  # pragma: no cover
    """
    Runs a non-linear optimal transmission switching (OTS) optimization using PowerModels.jl.

    see above

     """
    julia_file = os.path.join(pp_dir, "opf", 'run_powermodels_ots.jl')

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file=julia_file)
    _runpm(net)
    read_ots_results(net)


def runpm_storage_opf(net, calculate_voltage_angles=True,
                      trafo_model="t", delta=0, trafo3w_losses="hv", check_connectivity=True,
                      n_timesteps=24, time_elapsed=1.0):  # pragma: no cover
    """
    Runs a non-linear power system optimization with storages and time series using PowerModels.jl.


    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **n_timesteps** (int, 24) - number of time steps to optimize

        **time_elapsed** (float, 1.0) - time elapsed between time steps (1.0 = 1 hour)

     """
    julia_file = os.path.join(pp_dir, "opf", 'run_powermodels_mn_storage.jl')

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
                     pp_to_pm_callback=add_storage_opf_settings, julia_file=julia_file)

    net._options["n_time_steps"] = n_timesteps
    net._options["time_elapsed"] = time_elapsed

    _runpm(net)
    storage_results = read_pm_storage_results(net)
    return storage_results
