# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import warnings

from pypower.ppoption import ppoption
from pypower.idx_bus import VM
from pypower.add_userfcn import add_userfcn

from pandapower.pypower_extensions.runpf import _runpf
from pandapower.auxiliary import ppException, _select_is_elements, _clean_up, _add_pf_options,\
                                _get_voltage_level, _add_ppc_options, _add_opf_options
from pandapower.pd2ppc import _pd2ppc, _update_ppc
from pandapower.pypower_extensions.opf import opf
from pandapower.results import _extract_results, _copy_results_ppci_to_ppc, reset_results, \
    _extract_results_opf
from pandapower.create import create_gen

from pandapower.run_bfswpf import _run_bfswpf

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


class AlgorithmUnknown(ppException):
    """
    Exception being raised in case optimal powerflow did not converge.
    """
    pass


def runpp(net, init="auto", calculate_voltage_angles="auto", tolerance_kva=1e-5, trafo_model="t",
          trafo_loading="current", enforce_q_lims=False, numba=True, recycle=None,
          check_connectivity=True, r_switch=0.0, algorithm='nr', max_iteration=10, **kwargs):
    """
    Runs PANDAPOWER AC Flow

    Note: May raise pandapower.api.run["load"]flowNotConverged

    INPUT:
        **net** - The Pandapower format network

    OPTIONAL:
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
            provided to allow and/or accelerate convergence for networks where calculation of
            voltage angles is not necessary. Note that if calculate_voltage_angles is True the
            loadflow is initialized with a DC power flow (init = "dc")

            The default value is False because pandapower was developed for distribution networks.
            Please be aware that this parameter has to be set to True in meshed network for correct
            results!

        **tolerance_kva** (float, 1e-5) - loadflow termination condition referring to P / Q mismatch of node power in kva

        **trafo_model** (str, "t")  - transformer equivalent circuit model
        Pandapower provides two equivalent circuit models for the transformer:

            - "t" - transformer is modeled as equivalent with the T-model. This is consistent with PowerFactory and is also more accurate than the PI-model. We recommend using this transformer model.
            - "pi" - transformer is modeled as equivalent PI-model. This is consistent with Sincal, but the method is questionable since the transformer is physically T-shaped. We therefore recommend the use of the T-model.

        **trafo_loading** (str, "current") - mode of calculation for transformer loading

            Transformer loading can be calculated relative to the rated current or the rated power. In both cases the overall transformer loading is defined as the maximum loading on the two sides of the transformer.

            - "current"- transformer loading is given as ratio of current flow and rated current of the transformer. This is the recommended setting, since thermal as well as magnetic effects in the transformer depend on the current.
            - "power" - transformer loading is given as ratio of apparent power flow to the rated apparent power of the transformer.

        **enforce_q_lims** (bool, False) - respect generator reactive power limits

            If True, the reactive power limits in net.gen.max_q_kvar/min_q_kvar are respected in the
            loadflow. This is done by running a second loadflow if reactive power limits are
            violated at any generator, so that the runtime for the loadflow will increase if reactive
            power has to be curtailed.

        **numba** (bool, True) - Activation of numba JIT compiler in the newton solver

            If set to True, the numba JIT compiler is used to generate matrices for the powerflow. Massive
            speed improvements are likely.

        **recycle** (dict, none) - Reuse of internal powerflow variables for time series calculation

            Contains a dict with the following parameters:
            is_elems: If True in service elements are not filtered again and are taken from the last result in net["_is_elems"]
            ppc: If True the ppc (PYPOWER case file) is taken from net["_ppc"] and gets updated instead of reconstructed entirely
            Ybus: If True the admittance matrix (Ybus, Yf, Yt) is taken from ppc["internal"] and not reconstructed

        **check_connectivity** (bool, False) - Perform an extra connectivity test after the conversion from pandapower to PYPOWER

            If true, an extra connectivity test based on SciPy Compressed Sparse Graph Routines is perfomed.
            If check finds unsupplied buses, they are put out of service in the PYPOWER matrix

        ****kwargs** - options to use for PYPOWER.runpf
    """
    ac = True
    mode = "pf"
    copy_constraints_to_ppc = False
    if calculate_voltage_angles == "auto":
        voltage_level = _get_voltage_level(net)
        calculate_voltage_angles = True if voltage_level > 70 else False
    if init == "auto":
        init = "dc" if calculate_voltage_angles else "flat"
    # recycle parameters
    if recycle == None:
        recycle = dict(is_elems=False, ppc=False, Ybus=False)
    # init options
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles, 
                             trafo_model=trafo_model, check_connectivity=check_connectivity,
                             mode=mode, copy_constraints_to_ppc=copy_constraints_to_ppc,
                             r_switch=r_switch, init=init, enforce_q_lims=enforce_q_lims)
    _add_pf_options(net, tolerance_kva=tolerance_kva, trafo_loading=trafo_loading,
                     numba=numba, recycle=recycle, ac=ac, 
                    algorithm=algorithm, max_iteration=max_iteration)
    _runpppf(net, **kwargs)


def rundcpp(net, trafo_model="t", trafo_loading="current", recycle=None, check_connectivity=True,
            r_switch = 0.0, **kwargs):
    """
    Runs PANDAPOWER DC Flow

    Note: May raise pandapower.api.run["load"]flowNotConverged

    INPUT:
        **net** - The Pandapower format network

    OPTIONAL:
        **trafo_model** (str, "t")  - transformer equivalent circuit model
        Pandapower provides two equivalent circuit models for the transformer:

            - "t" - transformer is modeled as equivalent with the T-model. This is consistent with PowerFactory and is also more accurate than the PI-model. We recommend using this transformer model.
            - "pi" - transformer is modeled as equivalent PI-model. This is consistent with Sincal, but the method is questionable since the transformer is physically T-shaped. We therefore recommend the use of the T-model.

        **trafo_loading** (str, "current") - mode of calculation for transformer loading

            Transformer loading can be calculated relative to the rated current or the rated power. In both cases the overall transformer loading is defined as the maximum loading on the two sides of the transformer.

            - "current"- transformer loading is given as ratio of current flow and rated current of the transformer. This is the recommended setting, since thermal as well as magnetic effects in the transformer depend on the current.
            - "power" - transformer loading is given as ratio of apparent power flow to the rated apparent power of the transformer.

        **suppress_warnings** (bool, True) - suppress warnings in pypower

            If set to True, warnings are disabled during the loadflow. Because of the way data is
            processed in pypower, ComplexWarnings are raised during the loadflow. These warnings are
            suppressed by this option, however keep in mind all other pypower warnings are also suppressed.

        **numba** (bool, True) - Activation of numba JIT compiler in the newton solver

            If set to True, the numba JIT compiler is used to generate matrices for the powerflow. Massive
            speed improvements are likely.

        **recycle** (dict, none) - Reuse of internal powerflow variables for time series calculation

            Contains a dict with the following parameters:
            is_elems: If True in service elements are not filtered again and are taken from the last result in net["_is_elems"]
            ppc: If True the ppc (PYPOWER case file) is taken from net["_ppc"] and gets updated instead of reconstructed entirely
            Ybus: If True the admittance matrix (Ybus, Yf, Yt) is taken from ppc["internal"] and not reconstructed

        **check_connectivity** (bool, False) - Perform an extra connectivity test after the conversion from pandapower to PYPOWER

            If true, an extra connectivity test based on SciPy Compressed Sparse Graph Routines is perfomed.
            If check finds unsupplied buses, they are put out of service in the PYPOWER matrix

        ****kwargs** - options to use for PYPOWER.runpf
    """
    ac = False
    numba = True
    mode = "pf"
    init = 'flat'

    # the following parameters have no effect if ac = False
    calculate_voltage_angles = True
    copy_constraints_to_ppc = False
    enforce_q_lims = False
    algorithm = None
    max_iteration = None
    tolerance_kva = None
    
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles, 
                             trafo_model=trafo_model, check_connectivity=check_connectivity,
                             mode=mode, copy_constraints_to_ppc=copy_constraints_to_ppc,
                             r_switch=r_switch, init=init)
    _add_pf_options(net, tolerance_kva=tolerance_kva, trafo_loading=trafo_loading,
                    enforce_q_lims=enforce_q_lims, numba=numba, recycle=recycle, ac=ac, 
                    algorithm=algorithm, max_iteration=max_iteration)

    _runpppf(net, **kwargs)


def _runpppf(net, **kwargs):
    """
    Gets called by runpp or rundcpp with different arguments.
    """

    # get infos from options
    init = net["_options"]["init"]
    ac = net["_options"]["ac"]
    recycle = net["_options"]["recycle"]
    mode = net["_options"]["mode"]
    algorithm = net["_options"]["algorithm"]

    net["converged"] = False
    _add_auxiliary_elements(net)

    if (ac and not init == "results") or not ac:
        reset_results(net)

    # select elements in service (time consuming, so we do it once)
    net["_is_elems"] = _select_is_elements(net, recycle)

    if recycle["ppc"] and "_ppc" in net and net["_ppc"] is not None and "_pd2ppc_lookups" in net:
        # update the ppc from last cycle
        ppc, ppci = _update_ppc(net, recycle)
    else:
        # convert pandapower net to ppc
        ppc, ppci = _pd2ppc(net)

    # store variables
    net["_ppc"] = ppc

    if not "VERBOSE" in kwargs:
        kwargs["VERBOSE"] = 0

    # ----- run the powerflow -----
    if algorithm == 'bfsw':  # forward/backward sweep power flow algorithm
        result = _run_bfswpf(ppci, net["_options"], **kwargs)[0]

    elif algorithm in ['nr', 'fdBX', 'fdXB', 'gs']:  # algorithms existing within pypower
        result = _runpf(ppci, net["_options"], **kwargs)[0]

    else:
        raise AlgorithmUnknown("Algorithm {0} is unknown!".format(algorithm))

    # ppci doesn't contain out of service elements, but ppc does -> copy results accordingly
    result = _copy_results_ppci_to_ppc(result, ppc, mode)

    # raise if PF was not successful. If DC -> success is always 1
    if result["success"] != 1:
        raise LoadflowNotConverged("Power Flow did not converge!")
    else:
        net["_ppc"] = result
        net["converged"] = True

    _extract_results(net, result)
    _clean_up(net)


def runopp(net, verbose=False, calculate_voltage_angles=False, check_connectivity=True,
           suppress_warnings=True, r_switch=0.0, **kwargs):
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
        - net.dcline.min_q_to_kvar / net.dcline.max_q_to_kvar / net.dcline.min_q_from_kvar / net.dcline.max_q_from_kvar

    Network constraints can be defined for buses, lines and transformers the elements in the following columns:
        - net.bus.min_vm_pu / net.bus.max_vm_pu
        - net.line.max_loading_percent
        - net.trafo.max_loading_percent
        - net.trafo3w.max_loading_percent

    How these costs are combined into a cost function depends on the cost_function parameter.

    INPUT:
        **net** - The Pandapower format network

    OPTIONAL:
        **verbose** (bool, False) - If True, some basic information is printed

        **suppress_warnings** (bool, True) - suppress warnings in pypower

            If set to True, warnings are disabled during the loadflow. Because of the way data is
            processed in pypower, ComplexWarnings are raised during the loadflow.
            These warnings are suppressed by this option, however keep in mind all other pypower
            warnings are suppressed, too.
    """
    mode = "opf"
    ac = True
    copy_constraints_to_ppc = True
    trafo_model = "t"
    trafo_loading = 'current'
    init = "flat"
    enforce_q_lims = True

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles, 
                             trafo_model=trafo_model, check_connectivity=check_connectivity,
                             mode=mode, copy_constraints_to_ppc=copy_constraints_to_ppc,
                             r_switch=r_switch, init=init, enforce_q_lims=enforce_q_lims)
    _add_opf_options(net, trafo_loading=trafo_loading, ac=ac)
    _runopp(net, verbose, suppress_warnings, **kwargs)


def rundcopp(net, verbose=False, check_connectivity=True, suppress_warnings=True, r_switch=0.0,
             **kwargs):
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
        - net.dcline.min_q_to_kvar / net.dcline.max_q_to_kvar / net.dcline.min_q_from_kvar / net.dcline.max_q_from_kvar

    Network constraints can be defined for buses, lines and transformers the elements in the following columns:
        - net.line.max_loading_percent
        - net.trafo.max_loading_percent
        - net.trafo3w.max_loading_percent

    INPUT:
        **net** - The Pandapower format network

    OPTIONAL:
        **verbose** (bool, False) - If True, some basic information is printed

        **suppress_warnings** (bool, True) - suppress warnings in pypower

            If set to True, warnings are disabled during the loadflow. Because of the way data is
            processed in pypower, ComplexWarnings are raised during the loadflow.
            These warnings are suppressed by this option, however keep in mind all other pypower
            warnings are suppressed, too.
    """
    mode = "opf"
    ac = False
    init = "flat"
    copy_constraints_to_ppc = True
    trafo_model = "t"
    trafo_loading = 'current'
    calculate_voltage_angles = True
    enforce_q_lims = True

    net._options = {}   
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles, 
                             trafo_model=trafo_model, check_connectivity=check_connectivity,
                             mode=mode, copy_constraints_to_ppc=copy_constraints_to_ppc,
                             r_switch=r_switch, init=init, enforce_q_lims=enforce_q_lims)
    _add_opf_options(net, trafo_loading=trafo_loading, ac=ac)
    _runopp(net, verbose, suppress_warnings, **kwargs)


def _runopp(net, verbose, suppress_warnings, **kwargs):
    ac = net["_options"]["ac"]

    ppopt = ppoption(VERBOSE=verbose, OPF_FLOW_LIM=2, PF_DC=not ac, **kwargs)
    net["OPF_converged"] = False
    _add_auxiliary_elements(net)
    reset_results(net)
    # select elements in service (time consuming, so we do it once)
    net["_is_elems"] = _select_is_elements(net)

    ppc, ppci = _pd2ppc(net)
    if not ac:
        ppci["bus"][:, VM] = 1.0
    net["_ppc_opf"] = ppc
    if len(net.dcline) > 0:
        ppci = add_userfcn(ppci, 'formulation', add_dcline_constraints, args=net)

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
    mode = net["_options"]["mode"]
    result = _copy_results_ppci_to_ppc(result, ppc, mode=mode)

    net["_ppc_opf"] = result
    net["OPF_converged"] = True
    _extract_results_opf(net, result)
    _clean_up(net)



def _add_auxiliary_elements(net):
    # TODO: include directly in pd2ppc so that buses are only in ppc, not in pandapower
    if len(net["trafo3w"]) > 0:
        _create_trafo3w_buses(net)
    if len(net.dcline) > 0:
        _add_dcline_gens(net)
    if len(net["xward"]) > 0:
        _create_xward_buses(net)


def _create_xward_buses(net):
    from pandapower.create import create_buses
    init = net["_options"]["init"]

    init_results = init == "results"
    main_buses = net.bus.loc[net.xward.bus.values]
    bid = create_buses(net, nr_buses=len(main_buses),
                       vn_kv=main_buses.vn_kv.values,
                       in_service=net["xward"]["in_service"].values)
    net.xward["ad_bus"] = bid
    if init_results:
        # TODO: this is probably slow, but the whole auxiliary bus creation should be included in
        #      pd2ppc anyways. LT
        for hv_bus, aux_bus in zip(main_buses.index, bid):
            net.res_bus.loc[aux_bus] = net.res_bus.loc[hv_bus].values


def _create_trafo3w_buses(net):
    from pandapower.create import create_buses
    init = net["_options"]["init"]

    init_results = init == "results"
    hv_buses = net.bus.loc[net.trafo3w.hv_bus.values]
    bid = create_buses(net, nr_buses=len(net["trafo3w"]),
                       vn_kv=hv_buses.vn_kv.values,
                       in_service=net.trafo3w.in_service.values)
    net.trafo3w["ad_bus"] = bid
    if init_results:
        # TODO: this is probably slow, but the whole auxiliary bus creation should be included in
        #      pd2ppc anyways. LT
        for hv_bus, aux_bus in zip(hv_buses.index, bid):
            net.res_bus.loc[aux_bus] = net.res_bus.loc[hv_bus].values


def _add_dcline_gens(net):
    for _, dctab in net.dcline.iterrows():
        pfrom = dctab.p_kw
        pto = - (pfrom * (1 - dctab.loss_percent / 100) - dctab.loss_kw)
        pmax = dctab.max_p_kw
        create_gen(net, bus=dctab.to_bus, p_kw=pto, vm_pu=dctab.vm_to_pu,
                   min_p_kw=-pmax, max_p_kw=0.,
                   max_q_kvar=dctab.max_q_to_kvar, min_q_kvar=dctab.min_q_to_kvar,
                   in_service=dctab.in_service)
        create_gen(net, bus=dctab.from_bus, p_kw=pfrom, vm_pu=dctab.vm_from_pu,
                   min_p_kw=0, max_p_kw=pmax,
                   max_q_kvar=dctab.max_q_from_kvar, min_q_kvar=dctab.min_q_from_kvar,
                   in_service=dctab.in_service)


def add_dcline_constraints(om, net):
    # from numpy import hstack, diag, eye, zeros
    from scipy.sparse import csr_matrix as sparse
    ppc = om.get_ppc()
    ndc = len(net.dcline)  ## number of in-service DC lines
    ng = ppc['gen'].shape[0]  ## number of total gens
    Adc = sparse((ndc, ng))
    gen_lookup = net._pd2ppc_lookups["gen"]

    dcline_gens_from = net.gen.index[-2 * ndc::2]
    dcline_gens_to = net.gen.index[-2 * ndc + 1::2]
    for i, (f, t, loss) in enumerate(zip(dcline_gens_from, dcline_gens_to,
                                         net.dcline.loss_percent.values)):
        Adc[i, gen_lookup[f]] = 1. + loss * 1e-2
        Adc[i, gen_lookup[t]] = 1.

    ## constraints
    nL0 = -net.dcline.loss_kw.values * 1e-3  # absolute losses
    #    L1  = -net.dcline.loss_percent.values * 1e-2 #relative losses
    #    Adc = sparse(hstack([zeros((ndc, ng)), diag(1-L1), eye(ndc)]))

    ## add them to the model
    om = om.add_constraints('dcline', Adc, nL0, nL0, ['Pg'])
