# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

from numpy import where

from pandapower.auxiliary import _add_pf_options, _add_ppc_options, _add_opf_options
from pandapower.powerflow import _powerflow
from pandapower.optimal_powerflow import _optimal_powerflow


def runpp(net, algorithm='nr', calculate_voltage_angles="auto", init="auto", max_iteration="auto",
          tolerance_kva=1e-5, trafo_model="t", trafo_loading="current", enforce_q_lims=False,
          numba=True, recycle=None, check_connectivity=True, r_switch=0.0, **kwargs):
    """
    Runs PANDAPOWER AC Flow

    Note: May raise pandapower.api.run["load"]flowNotConverged

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **algorithm** (str, "nr"): algorithm that is used to solve the power flow problem.
                
            The following algorithms are available:
                
                - "nr" newton-raphson (pypower implementation with numba accelerations)
                - "bfsw" backward/forward sweep (specially suited for radial and weakly-meshed networks)
                - "gs" gauss-seidel (pypower implementation)
                - "fdbx" (pypower implementation)
                - "fdxb"(pypower implementation)
                
        **calculate_voltage_angles** (bool, "auto") - consider voltage angles in loadflow calculation

            If True, voltage angles of ext_grids and transformer shifts are considered in the 
            loadflow calculation. Considering the voltage angles is only necessary in meshed 
            networks that are usually found in higher networks. Thats why calculate_voltage_angles
            in "auto" mode defaults to:
                
                - True, if the network voltage level is above 70 kV
                - False otherwise
                
            The network voltage level is defined as the maximum rated voltage in the network that
            is connected to a line.            

        **init** (str, "auto") - initialization method of the loadflow
        pandapower supports four methods for initializing the loadflow:

            - "flat"- flat start with voltage of 1.0pu and angle of 0° at all buses as initial solution
            - "dc" - initial DC loadflow before the AC loadflow. The results of the DC loadflow are used as initial solution for the AC loadflow.
            - "results" - voltage vector of last loadflow from net.res_bus is used as initial solution. This can be useful to accelerate convergence in iterative loadflows like time series calculations.
            
        Considering the voltage angles might lead to non-convergence of the power flow in flat start.
        That is why in "auto" mode, init defaults to "dc" if calculate_voltage_angles is True or "flat" otherwise

        **max_iteration** (int, "auto"): maximum number of iterations carried out in the power flow algorithm.
                
            In "auto" mode, the default value depends on the power flow solver:
                
                - 10 for "nr"
                - 100 for "bfsw"
                - 1000 for "gs"
                - 30 for "fdbx" 
                - 30 for "fdxb" 
        
        **tolerance_kva** (float, 1e-5) - loadflow termination condition referring to P / Q mismatch of node power in kva

        **trafo_model** (str, "t")  - transformer equivalent circuit model
        pandapower provides two equivalent circuit models for the transformer:

            - "t" - transformer is modeled as equivalent with the T-model. 
            - "pi" - transformer is modeled as equivalent PI-model. This is not recommended, since it is less exact than the T-model. It is only recommended for valdiation with other software that uses the pi-model.

        **trafo_loading** (str, "current") - mode of calculation for transformer loading

            Transformer loading can be calculated relative to the rated current or the rated power. In both cases the overall transformer loading is defined as the maximum loading on the two sides of the transformer.

            - "current"- transformer loading is given as ratio of current flow and rated current of the transformer. This is the recommended setting, since thermal as well as magnetic effects in the transformer depend on the current.
            - "power" - transformer loading is given as ratio of apparent power flow to the rated apparent power of the transformer.

        **enforce_q_lims** (bool, True) - respect generator reactive power limits

            If True, the reactive power limits in net.gen.max_q_kvar/min_q_kvar are respected in the
            loadflow. This is done by running a second loadflow if reactive power limits are
            violated at any generator, so that the runtime for the loadflow will increase if reactive
            power has to be curtailed.

        **numba** (bool, True) - Activation of numba JIT compiler in the newton solver

            If set to True, the numba JIT compiler is used to generate matrices for the powerflow,
            which leads to significant speed improvements.

        **recycle** (dict, none) - Reuse of internal powerflow variables for time series calculation

            Contains a dict with the following parameters:
            _is_elements: If True in service elements are not filtered again and are taken from the last result in net["_is_elements"]
            ppc: If True the ppc is taken from net["_ppc"] and gets updated instead of reconstructed entirely
            Ybus: If True the admittance matrix (Ybus, Yf, Yt) is taken from ppc["internal"] and not reconstructed

        **check_connectivity** (bool, False) - Perform an extra connectivity test after the conversion from pandapower to PYPOWER

            If true, an extra connectivity test based on SciPy Compressed Sparse Graph Routines is perfomed.
            If check finds unsupplied buses, they are set out of service in the ppc
            
        **r_switch** (float, 0.0) - resistance of bus-bus-switches. If impedance is zero, buses connected by a closed bus-bus switch are fused to model an ideal bus. Otherwise, they are modelled as branches with resistance r_switch.
       
        ****kwargs** - options to use for PYPOWER.runpf
    """
    ac = True
    mode = "pf"
    copy_constraints_to_ppc = False
    if calculate_voltage_angles == "auto":
        calculate_voltage_angles = False
        hv_buses = where(net.bus.vn_kv.values > 70)[0]
        if len(hv_buses) > 0:
            line_buses = net.line[["from_bus", "to_bus"]].values.flatten()
            if len(set(net.bus.index[hv_buses]) & set(line_buses)) > 0:
                calculate_voltage_angles = True
    if init == "auto":
        init = "dc" if calculate_voltage_angles else "flat"
    default_max_iteration = {"nr": 10, "bfsw": 100, "gs": 10000, "fdxb": 30, "fdbx": 30}
    if max_iteration == "auto":
        max_iteration = default_max_iteration[algorithm]

    # init options
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode=mode, copy_constraints_to_ppc=copy_constraints_to_ppc,
                     r_switch=r_switch, init=init, enforce_q_lims=enforce_q_lims,
                     recycle=recycle)
    _add_pf_options(net, tolerance_kva=tolerance_kva, trafo_loading=trafo_loading,
                    numba=numba, ac=ac, algorithm=algorithm, max_iteration=max_iteration)
    _powerflow(net, **kwargs)


def rundcpp(net, trafo_model="t", trafo_loading="current", recycle=None, check_connectivity=True,
            r_switch=0.0, **kwargs):
    """
    Runs PANDAPOWER DC Flow

    Note: May raise pandapower.api.run["load"]flowNotConverged

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **trafo_model** (str, "t")  - transformer equivalent circuit model
        pandapower provides two equivalent circuit models for the transformer:

            - "t" - transformer is modeled as equivalent with the T-model. This is consistent with PowerFactory and is also more accurate than the PI-model. We recommend using this transformer model.
            - "pi" - transformer is modeled as equivalent PI-model. This is consistent with Sincal, but the method is questionable since the transformer is physically T-shaped. We therefore recommend the use of the T-model.

        **trafo_loading** (str, "current") - mode of calculation for transformer loading

            Transformer loading can be calculated relative to the rated current or the rated power. In both cases the overall transformer loading is defined as the maximum loading on the two sides of the transformer.

            - "current"- transformer loading is given as ratio of current flow and rated current of the transformer. This is the recommended setting, since thermal as well as magnetic effects in the transformer depend on the current.
            - "power" - transformer loading is given as ratio of apparent power flow to the rated apparent power of the transformer.

        **recycle** (dict, none) - Reuse of internal powerflow variables for time series calculation

            Contains a dict with the following parameters:
            _is_elements: If True in service elements are not filtered again and are taken from the last result in net["_is_elements"]
            ppc: If True the ppc (PYPOWER case file) is taken from net["_ppc"] and gets updated instead of reconstructed entirely
            Ybus: If True the admittance matrix (Ybus, Yf, Yt) is taken from ppc["internal"] and not reconstructed

        **check_connectivity** (bool, False) - Perform an extra connectivity test after the conversion from pandapower to PYPOWER

            If true, an extra connectivity test based on SciPy Compressed Sparse Graph Routines is perfomed.
            If check finds unsupplied buses, they are put out of service in the PYPOWER matrix

        **r_switch** (float, 0.0) - resistance of bus-bus-switches. If impedance is zero, buses connected by a closed bus-bus switch are fused to model an ideal bus. Otherwise, they are modelled as branches with resistance r_switch

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
                     r_switch=r_switch, init=init, enforce_q_lims=enforce_q_lims, recycle=recycle)
    _add_pf_options(net, tolerance_kva=tolerance_kva, trafo_loading=trafo_loading,
                    numba=numba, ac=ac, algorithm=algorithm, max_iteration=max_iteration)

    _powerflow(net, **kwargs)


def runopp(net, verbose=False, calculate_voltage_angles=False, check_connectivity=True,
           suppress_warnings=True, r_switch=0.0, **kwargs):
    """
    Runs the  pandapower Optimal Power Flow.
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
        **net** - The pandapower format network

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
    recycle = dict(_is_elements=False, ppc=False, Ybus=False)

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode=mode, copy_constraints_to_ppc=copy_constraints_to_ppc,
                     r_switch=r_switch, init=init, enforce_q_lims=enforce_q_lims, recycle=recycle)
    _add_opf_options(net, trafo_loading=trafo_loading, ac=ac)
    _optimal_powerflow(net, verbose, suppress_warnings, **kwargs)


def rundcopp(net, verbose=False, check_connectivity=True, suppress_warnings=True, r_switch=0.0,
             **kwargs):
    """
    Runs the  pandapower Optimal Power Flow.
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
        **net** - The pandapower format network

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
    recycle = dict(_is_elements=False, ppc=False, Ybus=False)

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode=mode, copy_constraints_to_ppc=copy_constraints_to_ppc,
                     r_switch=r_switch, init=init, enforce_q_lims=enforce_q_lims, recycle=recycle)
    _add_opf_options(net, trafo_loading=trafo_loading, ac=ac)
    _optimal_powerflow(net, verbose, suppress_warnings, **kwargs)
