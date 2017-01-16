# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import warnings

from pypower.ppoption import ppoption
from pypower.idx_bus import VM

from pandapower.pypower_extensions.runpf import _runpf
from pandapower.auxiliary import ppException, _select_is_elements, _clean_up
from pandapower.pd2ppc import _pd2ppc, _update_ppc
from pandapower.pypower_extensions.opf import opf
from pandapower.results import _extract_results, _copy_results_ppci_to_ppc, reset_results, \
    _extract_results_opf


class LoadflowNotConverged(ppException):
    """
    Exception being raised in case loadflow did not converge.
    """
    pass


class OPFNotConverged(ppException):
    """
    Exception being raised in case optimal powerflow did not converge.
    """
    pass


def runpp(net, init="flat", calculate_voltage_angles=False, tolerance_kva=1e-5, trafo_model="t"
          , trafo_loading="current", enforce_q_lims=False, numba=True, recycle=None, **kwargs):
    """
    Runs PANDAPOWER AC Flow

    Note: May raise pandapower.api.run["load"]flowNotConverged

    INPUT:
        **net** - The Pandapower format network

    Optional:

        **init** (str, "flat") - initialization method of the loadflow
        Pandapower supports three methods for initializing the loadflow:

            - "flat"- flat start with voltage of 1.0pu and angle of 0° at all buses as initial solution
            - "dc" - initial DC loadflow before the AC loadflow. The results of the DC loadflow are used as initial solution for the AC loadflow.
            - "results" - voltage vector of last loadflow from net.res_bus is used as initial solution. This can be useful to accelerate convergence in iterative loadflows like time series calculations.

        **calculate_voltage_angles** (bool, False) - consider voltage angles in loadflow calculation

            If True, voltage angles are considered in the  loadflow calculation. In some cases with
            large differences in voltage angles (for example in case of transformers with high
            voltage shift), the difference between starting and end angle value is very large.
            In this case, the loadflow might be slow or it might not converge at all. That is why 
            the possibility of neglecting the voltage angles of transformers and ext_grids is
            provided to allow and/or accelarate convergence for networks where calculation of 
            voltage angles is not necessary. Note that if calculate_voltage_angles is True the
            loadflow is initialized with a DC power flow (init = "dc")

            The default value is False because pandapower was developed for distribution networks.
            Please be aware that this parameter has to be set to True in meshed network for correct
            results!

        **tolerance_kva** (float, 1e-5) - loadflow termination condition referring to P / Q mismatch of node power in kva

        **trafo_model** (str, "t")  - transformer equivalent circuit model
        Pandapower provides two equivalent circuit models for the transformer:

            - "t" - transformer is modelled as equivalent with the T-model. This is consistent with PowerFactory and is also more accurate than the PI-model. We recommend using this transformer model.
            - "pi" - transformer is modelled as equivalent PI-model. This is consistent with Sincal, but the method is questionable since the transformer is physically T-shaped. We therefore recommend the use of the T-model. 

        **trafo_loading** (str, "current") - mode of calculation for transformer loading

            Transformer loading can be calculated relative to the rated current or the rated power. In both cases the overall transformer loading is defined as the maximum loading on the two sides of the transformer.

            - "current"- transformer loading is given as ratio of current flow and rated current of the transformer. This is the recommended setting, since thermal as well as magnetic effects in the transformer depend on the current.
            - "power" - transformer loading is given as ratio of apparent power flow to the rated apparent power of the transformer. 

        **enforce_q_lims** (bool, False) - respect generator reactive power limits

            If True, the reactive power limits in net.gen.max_q_kvar/min_q_kvar are respected in the
            loadflow. This is done by running a second loadflow if reactive power limits are
            violated at any generator, so that the runtime for the loadflow will increase if reactive
            power has to be curtailed.

        **numba** (bool, True) - Usage numba JIT compiler

            If set to True, the numba JIT compiler is used to generate matrices for the powerflow. Massive
            speed improvements are likely.

        **recycle** (dict, none) - Reuse of internal powerflow variables

            Contains a dict with the following parameters:
            is_elems: If True in service elements are not filtered again and are taken from the last result in net["_is_elems"]
            ppc: If True the ppc (PYPOWER case file) is taken from net["_ppc"] and gets updated instead of regenerated entirely
            Ybus: If True the admittance matrix (Ybus, Yf, Yt) is taken from ppc["internal"] and not regenerated

        ****kwargs** - options to use for PYPOWER.runpf
    """
    ac = True
    # recycle parameters
    if recycle == None:
        recycle = dict(is_elems=False, ppc=False, Ybus=False)

    _runpppf(net, init, ac, calculate_voltage_angles, tolerance_kva, trafo_model,
             trafo_loading, enforce_q_lims, numba, recycle, **kwargs)


def rundcpp(net, trafo_model="t", trafo_loading="current", suppress_warnings=True, recycle=None, 
            **kwargs):
    """
    Runs PANDAPOWER DC Flow

    Note: May raise pandapower.api.run["load"]flowNotConverged

    INPUT:
        **net** - The Pandapower format network

    Optional:

        **trafo_model** (str, "t")  - transformer equivalent circuit model
        Pandapower provides two equivalent circuit models for the transformer:

            - "t" - transformer is modelled as equivalent with the T-model. This is consistent with PowerFactory and is also more accurate than the PI-model. We recommend using this transformer model.
            - "pi" - transformer is modelled as equivalent PI-model. This is consistent with Sincal, but the method is questionable since the transformer is physically T-shaped. We therefore recommend the use of the T-model. 

        **trafo_loading** (str, "current") - mode of calculation for transformer loading

            Transformer loading can be calculated relative to the rated current or the rated power. In both cases the overall transformer loading is defined as the maximum loading on the two sides of the transformer.

            - "current"- transformer loading is given as ratio of current flow and rated current of the transformer. This is the recommended setting, since thermal as well as magnetic effects in the transformer depend on the current.
            - "power" - transformer loading is given as ratio of apparent power flow to the rated apparent power of the transformer. 

        **suppress_warnings** (bool, True) - suppress warnings in pypower

            If set to True, warnings are disabled during the loadflow. Because of the way data is
            processed in pypower, ComplexWarnings are raised during the loadflow. These warnings are
            suppressed by this option, however keep in mind all other pypower warnings are also suppressed.

        **numba** (bool, True) - Usage numba JIT compiler

            If set to True, the numba JIT compiler is used to generate matrices for the powerflow. Massive
            speed improvements are likely.

        **recycle** (dict, none) - Reuse of internal powerflow variables

            Contains a dict with the following parameters:
            is_elems: If True in service elements are not filtered again and are taken from the last result in net["_is_elems"]
            ppc: If True the ppc (PYPOWER case file) is taken from net["_ppc"] and gets updated instead of regenerated entirely
            Ybus: If True the admittance matrix (Ybus, Yf, Yt) is taken from ppc["internal"] and not regenerated

        ****kwargs** - options to use for PYPOWER.runpf
    """
    ac = False
    # the following parameters have no effect if ac = False
    calculate_voltage_angles = True
    enforce_q_lims = False
    init = ''
    tolerance_kva = 1e-5
    numba = True
    if recycle == None:
        recycle = dict(is_elems=False, ppc=False, Ybus=False)

    _runpppf(net, init, ac, calculate_voltage_angles, tolerance_kva, trafo_model,
             trafo_loading, enforce_q_lims, numba, recycle, **kwargs)


def _runpppf(net, init, ac, calculate_voltage_angles, tolerance_kva, trafo_model,
             trafo_loading, enforce_q_lims, numba, recycle, **kwargs):
    """
    Gets called by runpp or rundcpp with different arguments.
    """

    net["converged"] = False
    if (ac and not init == "results") or not ac:
        reset_results(net)

    # select elements in service (time consuming, so we do it once)
    is_elems = _select_is_elements(net, recycle)

    if recycle["ppc"] and "_ppc" in net and net["_ppc"] is not None and "_bus_lookup" in net:
        # update the ppc from last cycle
        ppc, ppci, bus_lookup = _update_ppc(net, is_elems, recycle, calculate_voltage_angles, enforce_q_lims,
                                            trafo_model)
    else:
        # convert pandapower net to ppc
        ppc, ppci, bus_lookup = _pd2ppc(net, is_elems, calculate_voltage_angles, enforce_q_lims,
                                        trafo_model, init_results=(init == "results"))

    # store variables
    net["_ppc"] = ppc
    net["_bus_lookup"] = bus_lookup
    net["_is_elems"] = is_elems

    if not "VERBOSE" in kwargs:
        kwargs["VERBOSE"] = 0

    # run the powerflow
    result = _runpf(ppci, init, ac, numba, recycle, ppopt=ppoption(ENFORCE_Q_LIMS=enforce_q_lims,
                                                                   PF_TOL=tolerance_kva * 1e-3, **kwargs))[0]

    # ppci doesn't contain out of service elements, but ppc does -> copy results accordingly
    result = _copy_results_ppci_to_ppc(result, ppc, bus_lookup)

    # raise if PF was not successful. If DC -> success is always 1
    if result["success"] != 1:
        raise LoadflowNotConverged("Loadflow did not converge!")
    else:
        net["_ppc"] = result
        net["converged"] = True

    _extract_results(net, result, is_elems, bus_lookup, trafo_loading, ac)
    _clean_up(net)


def runopp(net, cost_function="linear", verbose=False, suppress_warnings=True, **kwargs):
    """
    Runs the  Pandapower Optimal Power Flow.
    Flexibilities, constraints and cost parameters are defined in the pandapower element tables.

    Flexibilities for generators can be defined in net.sgen / net.gen.
    net.sgen.controllable / net.gen.controllable signals if a generator is controllable. If False,
    the active and reactive power are assigned as in a normal power flow. If yes, the following
    flexibilities apply:
        - net.sgen.min_p_kw / net.sgen.max_p_kw
        - net.sgen.min_q_kvar / net.sgen.max_q_kvar
        - net.gen.min_p_kw / net.gen.max_p_kw
        - net.gen.min_q_kvar / net.gen.max_q_kvar

    Network constraints can be defined for buses, lines and transformers the elements in the following columns:
        - net.bus.min_vm_pu / net.bus.max_vm_pu
        - net.line.max_loading_percent
        - net.trafo.max_loading_percent

    Costs can be assigned to generation units in the following columns:
        - net.gen.cost_per_kw
        - net.sgen.cost_per_kw
        - net.ext_grid.cost_per_kw

    How these costs are combined into a cost function depends on the cost_function parameter.

    INPUT:
        **net** - The Pandapower format network

    OPTIONAL:
        **cost_function** (str,"linear")- cost function
            - "linear" - minimizes weighted generator costs
            - "linear_minloss" - minimizes weighted generator cost and line losses

        **verbose** (bool, False) - If True, some basic information is printed

        **suppress_warnings** (bool, True) - suppress warnings in pypower

            If set to True, warnings are disabled during the loadflow. Because of the way data is
            processed in pypower, ComplexWarnings are raised during the loadflow.
            These warnings are suppressed by this option, however keep in mind all other pypower
            warnings are suppressed, too.
    """
    _runopp(net, verbose, suppress_warnings, cost_function, True, **kwargs)


def rundcopp(net, cost_function="linear", verbose=False, suppress_warnings=True, **kwargs):
    """
    Runs the  Pandapower Optimal Power Flow.
    Flexibilities, constraints and cost parameters are defined in the pandapower element tables.

    Flexibilities for generators can be defined in net.sgen / net.gen.
    net.sgen.controllable / net.gen.controllable signals if a generator is controllable. If False,
    the active and reactive power are assigned as in a normal power flow. If yes, the following
    flexibilities apply:
        - net.sgen.min_p_kw / net.sgen.max_p_kw
        - net.sgen.min_q_kvar / net.sgen.max_q_kvar
        - net.gen.min_p_kw / net.gen.max_p_kw
        - net.gen.min_q_kvar / net.gen.max_q_kvar

    Network constraints can be defined for buses, lines and transformers the elements in the following columns:
        - net.line.max_loading_percent
        - net.trafo.max_loading_percent

    Costs can be assigned to generation units in the following columns:
        - net.gen.cost_per_kw
        - net.sgen.cost_per_kw
        - net.ext_grid.cost_per_kw

    How these costs are combined into a cost function depends on the cost_function parameter.

    INPUT:
        **net** - The Pandapower format network

    OPTIONAL:
        **cost_function** (str,"linear")- cost function
            - "linear" - minimizes weighted generator costs
            - "linear_minloss" - minimizes weighted generator cost and line losses

        **verbose** (bool, False) - If True, some basic information is printed

        **suppress_warnings** (bool, True) - suppress warnings in pypower

            If set to True, warnings are disabled during the loadflow. Because of the way data is
            processed in pypower, ComplexWarnings are raised during the loadflow.
            These warnings are suppressed by this option, however keep in mind all other pypower
            warnings are suppressed, too.
    """
    _runopp(net, verbose, suppress_warnings, cost_function, False, **kwargs)


def _runopp(net, verbose, suppress_warnings, cost_function, ac=True, **kwargs):
    ppopt = ppoption(VERBOSE=verbose, OPF_FLOW_LIM=2, PF_DC=not ac, **kwargs)
    net["OPF_converged"] = False

    reset_results(net)
    # select elements in service (time consuming, so we do it once)
    is_elems = _select_is_elements(net)

    ppc, ppci, bus_lookup = _pd2ppc(net, is_elems, copy_constraints_to_ppc=True, trafo_model="t",
                                    opf=True, cost_function=cost_function, 
                                    calculate_voltage_angles=False)
    if not ac:
        ppci["bus"][:, VM] = 1.0
    net["_ppc_opf"] = ppc
    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opf(ppci, ppopt)
    else:
        result = opf(ppci, ppopt)
    net["_ppc_opf"] = result

    if not result["success"]:
        raise OPFNotConverged("Optimal Power Flow did not converge!")

    # ppci doesn't contain out of service elements, but ppc does -> copy results accordingly
    result = _copy_results_ppci_to_ppc(result, ppc, bus_lookup)

    net["_ppc_opf"] = result
    net["OPF_converged"] = True
    _extract_results_opf(net, result, is_elems, bus_lookup, "current", True, ac)
    _clean_up(net)
