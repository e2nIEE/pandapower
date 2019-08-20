# -*- coding: utf-8 -*-
"""
# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

@author: sghosh (Intern : Feb 2018-July 2018)
@author: Alexander Prostejovsky (alepros), Technical University of Denmark
"""
import copy
from time import time
import numpy as np
from numpy import flatnonzero as find, pi, exp
from pandapower.pypower.pfsoln import pfsoln
try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)
from pandapower.pypower.idx_gen import PG, QG, GEN_BUS
from pandapower.pd2ppc import _pd2ppc
from pandapower.pd2ppc_zero import _pd2ppc_zero
from pandapower.pypower.makeYbus import makeYbus
from pandapower.pypower.idx_bus import GS, BS, PD , QD,BASE_KV
from pandapower.auxiliary import _sum_by_group, _check_if_numba_is_installed,\
    _check_bus_index_and_print_warning_if_high,\
    _check_gen_index_and_print_warning_if_high, \
    _add_pf_options, _add_ppc_options, _clean_up, sequence_to_phase, \
    phase_to_sequence,Y_phase_to_sequence, X012_to_X0, X012_to_X2, \
    I1_from_V012, S_from_VI_elementwise, V1_from_ppc, V_from_I,\
    combine_X012, I0_from_V012, I2_from_V012, ppException
from pandapower.pf.run_newton_raphson_pf import _run_newton_raphson_pf
from pandapower.build_bus import _add_ext_grid_sc_impedance
from pandapower.pypower.bustypes import bustypes
from pandapower.run import _passed_runpp_parameters
from pandapower.pypower.idx_bus import VM, VA
from pandapower.pypower.idx_gen import GEN_BUS, GEN_STATUS, VG
from pandapower.results import _copy_results_ppci_to_ppc, _extract_results_3ph\
, reset_results

class LoadflowNotConverged(ppException):
    """
    Exception being raised in case loadflow did not converge.
    """
    pass

def _get_pf_variables_from_ppci(ppci):
    """
    Used for getting values for pfsoln function in one convinient function
    """
    ## default arguments
    if ppci is None:
        ValueError('ppci is empty')
    # ppopt = ppoption(ppopt)

    # get data for calc
    baseMVA, bus, gen, branch = \
        ppci["baseMVA"], ppci["bus"], ppci["gen"], ppci["branch"]

    ## get bus index lists of each type of bus
    ref, pv, pq = bustypes(bus, gen)

    ## generator info
    on = find(gen[:, GEN_STATUS] > 0)  ## which generators are on?
    gbus = gen[on, GEN_BUS].astype(int)  ## what buses are they at?

    ## initial state
    # V0    = ones(bus.shape[0])            ## flat start
    V0 = bus[:, VM] * exp(1j * pi / 180 * bus[:, VA])
    V0[gbus] = gen[on, VG] / abs(V0[gbus]) * V0[gbus]
    ref_gens = ppci["internal"]["ref_gens"]
    return baseMVA, bus, gen, branch, ref, pv, pq, on, gbus, V0,ref_gens
def _store_results_from_pf_in_ppci(ppci, bus, gen, branch):
    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch
    return ppci
# =============================================================================
# Mapping load for positive sequence loads
# =============================================================================

def _load_mapping(net,ppci1):
    """
    Takes three phase P, Q values from PQ elements
    sums them up for each bus
    maps them in ppc bus order and forms Sabc matrix
    
    
    """
    bus_lookup = net["_pd2ppc_lookups"]["bus"]    
    params=dict()
    phases = ['A','B','C']
    load_types = ['wye','delta']
    load_elements = ['load','asymmetric_load','sgen','asymmetric_sgen']
#                    ,'impedance_load']
    
# =============================================================================
#        Loop to initialize and feed Sabc wye and delta values 
# =============================================================================
    for phase in phases:
        for typ in load_types:
            params['S'+phase+typ] = (ppci1["bus"][:, PD]+\
                                     ppci1["bus"][:, QD]*1j)*0
            params['p'+phase+typ] = np.array([])#p values from loads/sgens
            params['q'+phase+typ] = np.array([])#q values from loads/sgens
            params['P'+phase+typ] = np.array([])#Aggreagated Active Power 
            params['Q'+phase+typ] = np.array([])#Aggreagated reactive Power
            params['b'+phase+typ] = np.array([], dtype=int)#bus map for phases
            params['b'+typ] = np.array([], dtype=int)#aggregated bus map(Sabc)  
            params['Y012'+typ] = np.array([], dtype=np.complex)
            params['b_imp'+typ] = np.array([], dtype=int)
            for element in load_elements:
                sign = -1 if element.endswith("sgen") else 1
                elm = net[element]
                elm_is = net["_is_elements"][element] 
                elm["in_service"] = elm_is
                elm = elm[elm["in_service"] == True]
                elm_typ = elm[elm["type"] == typ]
                if len(elm_typ) > 0 :
                    vl = (elm_typ["in_service"].values * elm_typ["scaling"]\
                         .values.T)[elm_typ["in_service"].values]
                    if element =='load' or element == 'sgen':
                        params['p'+phase+typ] = np.hstack([params['p'+phase+typ]\
                                            , elm_typ["p_mw"].values/3 * vl*sign])
                        params['q'+phase+typ] = np.hstack([params['q'+phase+typ]\
                                            , elm_typ["q_mvar"].values/3 * vl*sign])
                        params['b'+typ] = np.hstack([params['b'+typ], \
                                              elm_typ["bus"].values])                    
# =============================================================================
#TODO: Uncomment if constant impedance loads are required in the future:
#    auxiliary,results and runpp_3ph
#                     elif element == 'impedance_load':
#                         arange_load = np.arange(len(elm_typ))
#                         params['y'+phase+typ] = np.hstack([\
#                              params['y'+phase+typ], \
#                              (1/(elm_typ["r_"+phase].values\
#                                  +elm_typ["x_"+phase].values * 1j))\
#                                  .astype(complex) * vl]) 
#                         params['b_imp'+typ]= np.hstack([params['b_imp'+typ]\
#                                                        , elm_typ["bus"].values])
# =============================================================================
                    elif element.startswith('asymmetric'):
                        params['p'+phase+typ] =np.hstack([params['p'+phase+typ]\
                                    , elm_typ['p_'+phase+'_mw'].values * vl*sign])
                        params['q'+phase+typ] =np.hstack([params['q'+phase+typ]\
                                    , elm_typ['q_'+phase+'_mvar'].values * vl*sign])
                        params['b'+typ] = np.hstack([params['b'+typ], \
                                      elm_typ["bus"].values])
           # Mapping constant power loads to buses    
            if params['b'+typ].size:
                   params['b'+phase+typ] = bus_lookup[params['b'+typ]]
                   
                   params['b'+phase+typ], params['P'+phase+typ], \
                   params['Q'+phase+typ]\
                   \
                   = _sum_by_group(params['b'+phase+typ]\
                         , params['p'+phase+typ], params['q'+phase+typ] * 1j)
                   params['S'+phase+typ][params['b'+phase+typ]]= \
                   (params['P'+phase+typ] + params['Q'+phase+typ])
# =============================================================================
#TODO: Uncomment if constant impedance loads are required in the future 
#           # Mapping constant impedance loads to buses
# #            params['b_imp'+typ] = np.hstack([params['b_imp'+typ],\
# #                                               elm_typ["bus"].values])
# #            Y_abc = {yph:np.zeros([3, 3],dtype=np.complex) for yph in params['b'+typ]}
# #            Y_012 = {ysq:np.zeros([3, 3],dtype=np.complex) for ysq in params['b'+typ]}
# #           # Making a dictionary with bus number and Yabc-Y012 conversion
# #            for b_imp in params['b_imp'+typ]:
# #                for load in arange_load:                                
# #                    if phase =='A':
# #                        Y_abc[b_imp][0,0] = params['y'+phase+typ][load]
# #                    elif phase == 'B':
# #                        Y_abc[b_imp][1,1] = params['y'+phase+typ][load]
# #                    elif phase == 'C':
# #                        Y_abc[b_imp][2,2] = params['y'+phase+typ][load]
# #                    Y_012[b_imp] = Y_phase_to_sequence(Y_abc[b_imp])          
# =============================================================================
    #last return varaible left for constant impedance loads
    return np.vstack([params['S'+phase+'delta']] for phase in phases),\
    np.vstack([params['S'+phase+'wye']] for phase in phases),{}
# 
# =============================================================================



# =============================================================================
# 3 phase algorithm function
# =============================================================================
def runpp_3ph(net, calculate_voltage_angles="auto", init="auto", 
              max_iteration="auto", tolerance_mva=1e-8, trafo_model="t", 
              trafo_loading="current", enforce_q_lims=False, numba=True, 
              recycle=None, check_connectivity=True, switch_rx_ratio=2.0,
              delta_q=0,v_debug =False, **kwargs):
    """
 runpp_3ph: Performs Unbalanced/Asymmetric/Three Phase Load flow
 
    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **algorithm** (str, "nr") - algorithm that is used to solve the power 
        flow problem.

            The following algorithms are available:

                - "nr" Newton-Raphson (pypower implementation with numba 
                accelerations)
                
                Used only for positive sequence network
                
                Zero and Negative sequence networks use Current Injection method
                
                Vnew = Y.inv * Ispecified ( from Sabc/Vabc old)
                
                Icalculated = Y * Vnew
                

        **calculate_voltage_angles** (bool, "auto") - consider voltage angles 
        in loadflow calculation

            If True, voltage angles of ext_grids and transformer shifts are 
            considered in the loadflow calculation. Considering the voltage 
            angles is only necessary in meshed networks that are usually 
            found in higher voltage levels. calculate_voltage_angles
            in "auto" mode defaults to:

                - True, if the network voltage level is above 70 kV
                - False otherwise

            The network voltage level is defined as the maximum rated voltage 
            of any bus in the network that is connected to a line.


        **max_iteration** (int, "auto") - maximum number of iterations carried
        out in the power flow algorithm.

            In "auto" mode, the default value depends on the power flow solver:

                - 10 for "nr"

            For three phase calculations, its extended to 3 * max_iteration

        **tolerance_mva** (float, 1e-8) - loadflow termination condition 
        referring to P / Q mismatch of node power in MVA

        **trafo_model** (str, "t") ('pi' Yet to be implemented in 3 Phase load flow) - transformer equivalent circuit model
        pandapower provides two equivalent circuit models for the transformer:

            - "t" - transformer is modeled as equivalent with the T-model.
            - "pi" - transformer is modeled as equivalent PI-model. This is 
            not recommended, since it is less exact than the T-model. 
            It is only recommended for valdiation with other software 
            that uses the pi-model.
        
        But, for three phase analysis we have considered only T model, which 
        is internally converted to pi model
        
        **trafo_loading** (str, "current") (Not tested with 3 Phase load flow) - mode of calculation for 
        transformer loading

            Transformer loading can be calculated relative to the rated 
            current or the rated power. In both cases the overall transformer 
            loading is defined as the maximum loading on the two sides of 
            the transformer.

            - "current"- transformer loading is given as ratio of current 
            flow and rated current of the transformer. This is the recommended 
            setting, since thermal as well as magnetic effects in the 
            transformer depend on the current.
            - "power" - transformer loading is given as ratio of apparent 
            power flow to the rated apparent power of the transformer.

        **enforce_q_lims** (bool, False) (Not tested with 3 Phase load flow) - respect generator reactive power 
        limits

            If True, the reactive power limits in net.gen.max_q_mvar/min_q_mvar
            are respected in the loadflow. This is done by running a second 
            loadflow if reactive power limits are violated at any generator, 
            so that the runtime for the loadflow will increase if reactive
            power has to be curtailed.

            Note: enforce_q_lims only works if algorithm="nr"!


        **check_connectivity** (bool, True) - Perform an extra connectivity 
        test after the conversion from pandapower to PYPOWER

            If True, an extra connectivity test based on SciPy Compressed 
            Sparse Graph Routines is perfomed. If check finds unsupplied buses,
            they are set out of service in the ppc

        **voltage_depend_loads** (bool, True)(Not tested with 3 Phase load flow)  - consideration of 
        voltage-dependent loads. If False, net.load.const_z_percent and 
        net.load.const_i_percent are not considered, i.e. net.load.p_mw and 
        net.load.q_mvar are considered as constant-power loads.

        **consider_line_temperature** (bool, False) (Not tested with 3 Phase load flow) - adjustment of line 
        impedance based on provided line temperature. If True, net.line must 
        contain a column "temperature_degree_celsius". The temperature 
        dependency coefficient alpha must be provided in the net.line.alpha
            column, otherwise the default value of 0.004 is used


        **KWARGS:

        **numba** (bool, True) - Activation of numba JIT compiler in the 
        newton solver

            If set to True, the numba JIT compiler is used to generate
            matrices for the powerflow, which leads to significant speed 
            improvements.

        **switch_rx_ratio** (float, 2) (Not tested with 3 Phase load flow)  - rx_ratio of bus-bus-switches. 
        If impedance is zero, buses connected by a closed bus-bus switch 
        are fused to model an ideal bus. Otherwise, they are modelled 
        as branches with resistance defined as z_ohm column in switch 
        table and this parameter

        **delta_q** (Not tested with 3 Phase load flow) - Reactive power tolerance for option "enforce_q_lims" 
        in kvar - helps convergence in some cases.

        **trafo3w_losses** (Not tested with 3 Phase load flow) - defines where open loop losses of three-winding
        transformers are considered. Valid options are "hv", "mv", "lv" 
        for HV/MV/LV side or "star" for the star point.

        **v_debug** (bool, False) (Not tested with 3 Phase load flow) - if True, voltage values in each 
        newton-raphson iteration are logged in the ppc

        **init_vm_pu** (string/float/array/Series, None) (Not tested with 3 Phase load flow) - Allows to define 
        initialization specifically for voltage magnitudes. 
        Only works with init == "auto"!

            - "auto": all buses are initialized with the mean value of all 
            voltage controlled elements in the grid
            - "flat" for flat start from 1.0
            - "results": voltage magnitude vector is taken from result table
            - a float with which all voltage magnitudes are initialized
            - an iterable with a voltage magnitude value for each bus
            (length and order has to match with the buses in net.bus)
            - a pandas Series with a voltage magnitude value for each bus
            (indexes have to match the indexes in net.bus)

         **init_va_degree** (string/float/array/Series, None) (Not tested with 3 Phase load flow)-
        Allows to define initialization specifically for voltage angles. 
        Only works with init == "auto"!

            - "auto": voltage angles are initialized from DC power flow 
            if angles are calculated or as 0 otherwise
            - "dc": voltage angles are initialized from DC power flow
            - "flat" for flat start from 0
            - "results": voltage angle vector is taken from result table
            - a float with which all voltage angles are initialized
            - an iterable with a voltage angle value for each bus (length 
            and order has to match with the buses in net.bus)
            - a pandas Series with a voltage angle value for each bus (indexes 
            have to match the indexes in net.bus)

        **recycle** (dict, none)(Not tested with 3 Phase load flow) - Reuse of internal powerflow variables for 
        time series calculation(Not tested with 3 Phase load flow) 

            Contains a dict with the following parameters:
            _is_elements: If True in service elements are not filtered again
            and are taken from the last result in net["_is_elements"]
            ppc: If True the ppc is taken from net["_ppc"] and gets updated 
            instead of reconstructed entirely
            Ybus: If True the admittance matrix (Ybus, Yf, Yt) is taken from 
            ppc["internal"] and not reconstructed

        **neglect_open_switch_branches** (bool, False) (Not tested with 3 Phase load flow) - If True no auxiliary 
        buses are created for branches when switches are opened at the branch.
        Instead branches are set out of service

    Return values:
    ---------------
    **count(int)** No of iterations taken to reach convergence
    
    **V012_it(complex)**   - Sequence voltages                          
    
    **I012_it(complex)**   - Sequence currents 

    See Also:
    ----------
    pp.add_zero_impedance_parameters(net): 
    To add zero sequence parameters into network from the standard type

    Examples:
    ----------
    >>> from pandapower.pf.runpp_3ph import runpp_3ph

    >>> runpp_3ph(net)

    Notes:
    --------
    - Three phase load flow uses Sequence Frame for power flow solution. 
    - Three phase system is modelled with earth return. 
    - PH-E load type is called as wye since Neutral and Earth are considered same
    - This solver has proved successful only for Earthed transformers (i.e Dyn,Yyn,YNyn & Yzn vector groups) 
    """
    # =============================================================================
    # pandapower settings
    # =============================================================================
    overrule_options = {}
    if "user_pf_options" in net.keys() and len(net.user_pf_options) > 0:
        passed_parameters = _passed_runpp_parameters(locals())
        overrule_options = {key: val for key, val in net.user_pf_options.items()
                            if key not in passed_parameters.keys()}
    if numba:
        numba = _check_if_numba_is_installed(numba)

    ac = True
    mode = "pf_3ph"  # TODO: Make valid modes (pf, pf_3ph, se, etc.) available in seperate file (similar to idx_bus.py)
#    v_debug = kwargs.get("v_debug", False)
    copy_constraints_to_ppc = False
    if calculate_voltage_angles == "auto":
        calculate_voltage_angles = False
        hv_buses = np.where(net.bus.vn_kv.values > 70)[0] # Todo: Where does that number come from?
        if len(hv_buses) > 0:
            line_buses = net.line[["from_bus", "to_bus"]].values.flatten()
            if len(set(net.bus.index[hv_buses]) & set(line_buses)) > 0:
                calculate_voltage_angles = True
    if init == "results" and len(net.res_bus) == 0:
        init = "auto"
    if init == "auto":
        init = "dc" if calculate_voltage_angles else "flat"   
    default_max_iteration = {"nr": 10, "bfsw": 10, "gs": 10000, "fdxb": 30,\
                             "fdbx": 30}
    if max_iteration == "auto":
        max_iteration = default_max_iteration["nr"]
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=\
                     check_connectivity, mode=mode,switch_rx_ratio=\
                     switch_rx_ratio, init_vm_pu=init, init_va_degree=init,\
                     enforce_q_lims=enforce_q_lims, recycle=recycle, \
                     voltage_depend_loads=False, delta=delta_q)
    _add_pf_options(net, tolerance_mva=tolerance_mva, trafo_loading=trafo_loading,
                    numba=numba, ac=ac, algorithm="nr", max_iteration=\
                    max_iteration,v_debug=v_debug)
    net._options.update(overrule_options)
    _check_bus_index_and_print_warning_if_high(net)
    _check_gen_index_and_print_warning_if_high(net)
    reset_results(net, balanced=False)
    # =========================================================================
    # pd2ppc conversion
    # =========================================================================
    net["_is_elements"] = None
    _, ppci1 = _pd2ppc(net, 1)

    _, ppci2 = _pd2ppc(net, 2)
    gs_eg,bs_eg = _add_ext_grid_sc_impedance(net, ppci2)

    _, ppci0 = _pd2ppc(net, 0)
    
    _,       bus0, gen0, branch0,      _,      _,      _, _, _,\
    V00, ref_gens = _get_pf_variables_from_ppci(ppci0)
    baseMVA, bus1, gen1, branch1, sl_bus, pv_bus, pq_bus, _, _, \
    V01, ref_gens = _get_pf_variables_from_ppci(ppci1)
    _,       bus2, gen2, branch2,      _,      _,      _, _, _, \
    V02, ref_gens = _get_pf_variables_from_ppci(ppci2)

# =============================================================================
#     P Q values aggragated and summed up for each bus to make Sabc matrix 
#     Sabc for wye connections ; Sabc_delta for delta connection
# =============================================================================
    Sabc_delta, Sabc, _ = _load_mapping(net,ppci1)
    # =========================================================================
    # Construct Sequence Frame Bus admittance matrices Ybus
    # =========================================================================
    
    ppci0, ppci1, ppci2, Y0_pu, Y1_pu, Y2_pu, Y0_f, Y1_f, Y2_f,\
    Y0_t, Y1_t, Y2_t = _get_Y_bus(ppci0, ppci1, ppci2, recycle, makeYbus)
 
# =============================================================================
#     if net.impedance_load.r_A.any() : 
#         Y0_pu= Y0_pu.todense()
#         Y2_pu= Y2_pu.todense()
#         for bus,Y in Y012.items(): 
#             bus_lookup = net["_pd2ppc_lookups"]["bus"]
#             b= bus_lookup[bus]
#             baseY =  net.sn_mva/( np.square(ppci1["bus"][b, BASE_KV]) )
#             ppci1["bus"][b, GS] = Y[1,1].conjugate().real
#             # Y11 assigned to ppc1 load bus
#             ppci1["bus"][b, BS] = Y[1,1].conjugate().imag  
#             Y0_pu[b,b]+=Y[0,0]/baseY
#             Y2_pu[b,b]+=Y[2,2]/baseY
#     ppci0, ppci1, ppci2, _, Y1_pu, _, _, Y1_f, _,_, Y1_t, _ =\
#    _get_Y_bus(ppci0, ppci1, ppci2, recycle, makeYbus)
# TODO: Uncomment if constant impedance loads are required in the future
# =============================================================================


    # =========================================================================
    # Initial voltage values
    # =========================================================================
    nb = ppci1["bus"].shape[0]
    V012_it = np.concatenate(
        (
            np.array(np.zeros((1, nb), dtype=np.complex128))
            , np.array(np.ones((1, nb), dtype=np.complex128))
            , np.array(np.zeros((1, nb), dtype=np.complex128))
        )
        , axis=0
    )
    #For Delta transformation:
    #Voltage changed from line-earth to line-line using V_T
    #Sabc/Vabc will now give line-line currents. This is converted to line-earth 
    #current using I-T
    V_T=np.matrix([[1,-1,0],
                   [0,1,-1],
                   [-1,0,1]]) 
    I_T=np.matrix([[1,0,-1],
                   [-1,1,0],
                   [0,-1,1]]) 
    Vabc_it = sequence_to_phase(V012_it)
    
    # =========================================================================
    #             Iteration using Power mismatch criterion
    # =========================================================================

    count = 0
    S_mismatch = np.array([[True], [True]], dtype=bool)
    t0 = time()
    while (S_mismatch > tolerance_mva).any() and count < 3*max_iteration :
        # =====================================================================
        #     Voltages and Current transformation for PQ and Slack bus
        # =====================================================================
        Sabc_pu = -np.divide(Sabc, ppci1["baseMVA"])
        Sabc_delta_pu = -np.divide(Sabc_delta, ppci1["baseMVA"])
        
        Iabc_it_wye = (np.divide(Sabc_pu, Vabc_it)).conjugate()
        Iabc_it_delta = np.matmul(I_T,(np.divide(Sabc_delta_pu, np.matmul\
                                                 (V_T,Vabc_it))).conjugate())
        
        #For buses with both delta and wye loads we need to sum of their currents
        # to sum up the currents
        Iabc_it=Iabc_it_wye+Iabc_it_delta
        I012_it = phase_to_sequence(Iabc_it)
        V1_for_S1 = V012_it[1, :]
        I1_for_S1 = -I012_it[1, :]
        V0_pu_it = X012_to_X0(V012_it)  
        V2_pu_it = X012_to_X2(V012_it)
        
        I0_pu_it = X012_to_X0(I012_it)    
        I2_pu_it = X012_to_X2(I012_it)
        
        
# =============================================================================
#         if net.impedance_load.r_A.any() :
#             for bus,Y in Y012.items():
#                 bus_lookup = net["_pd2ppc_lookups"]["bus"]
#                 b= bus_lookup[bus]
#                 lv = bus_lookup[int(net.trafo.lv_bus.values)]
#                 I0_pu_it[b] = Y[0,1] * V1_for_S1[b] + Y[0,2]*V2_pu_it[b]
#                 I1_for_S1[b]= Y[1,0] * V0_pu_it[b] + Y[1,2]*V2_pu_it[b]
#                 I2_pu_it[b] = Y[2,0] * V0_pu_it[b] + Y[2,1]*V1_for_S1[b]
#                 
# =============================================================================

        
        I1_for_S1 = -I012_it[1, :]
        S1 = np.multiply(V1_for_S1, I1_for_S1.conjugate())
        # =============================================================================
        # Current used to find S1 Positive sequence power
        # =============================================================================

        ppci1["bus"][pq_bus, PD] = np.real(S1[pq_bus]) * ppci1["baseMVA"]
        ppci1["bus"][pq_bus, QD] = np.imag(S1[pq_bus]) * ppci1["baseMVA"]
        # =============================================================================
        # Conduct Positive sequence power flow
        # =============================================================================
        _run_newton_raphson_pf(ppci1, net._options)
        # =============================================================================
        # Conduct Negative and Zero sequence power flow
        # =============================================================================
        V0_pu_it = V_from_I(Y0_pu, I0_pu_it)
        V2_pu_it = V_from_I(Y2_pu, I2_pu_it)
        # =============================================================================
        #    Evaluate Positive Sequence Power Mismatch     
        # =============================================================================
        I1_from_V_it = I1_from_V012(V012_it, Y1_pu).flatten()
        s_from_voltage = S_from_VI_elementwise(V1_for_S1, I1_from_V_it)
        V1_pu_it = V1_from_ppc(ppci1)

        V012_new = combine_X012(V0_pu_it, V1_pu_it, V2_pu_it)

        S_mismatch = np.abs(np.abs(S1[pq_bus]) - np.abs(s_from_voltage[pq_bus]))
        V012_it = V012_new
        Vabc_it = sequence_to_phase(V012_it)
        count += 1
    et = time() - t0
    success = (count < 3 * max_iteration)
    for ppc in [ppci0,ppci1,ppci2]:
        ppc["et"] = et
        ppc["success"] = success
    
    # TODO: Add reference to paper to explain the following steps
# =============================================================================
# #    No longer needed. Change in accuracy is not so much
#    Can be added if required
# =============================================================================
#    ref, pv, pq = bustypes(ppci0["bus"], ppci0["gen"])
#    ppci0["bus"][ref, GS] -= gs_eg
#    ppci0["bus"][ref, BS] -= bs_eg
#
#    Y0_pu, Y0_f, Y0_t = makeYbus(ppci0["baseMVA"], ppci0["bus"], ppci0["branch"])

    ## update data matrices with solution
    # TODO: Add reference to paper to explain the choice of Y1 over Y2 in the negative sequence
    bus0, gen0, branch0 = pfsoln(baseMVA, bus0, gen0, branch0, Y0_pu, Y0_f, Y0_t, V012_it[0, :].flatten(), sl_bus, ref_gens)
    bus1, gen1, branch1 = pfsoln(baseMVA, bus1, gen1, branch1, Y1_pu, Y1_f, Y1_t, V012_it[1, :].flatten(), sl_bus, ref_gens)
    bus2, gen2, branch2 = pfsoln(baseMVA, bus2, gen2, branch2, Y2_pu, Y1_f, Y1_t, V012_it[2, :].flatten(), sl_bus, ref_gens)
    ppci0 = _store_results_from_pf_in_ppci(ppci0, bus0, gen0, branch0)
    ppci1 = _store_results_from_pf_in_ppci(ppci1, bus1, gen1, branch1)
    ppci2 = _store_results_from_pf_in_ppci(ppci2, bus2, gen2, branch2)   
    ppci0["internal"]["Ybus"] = Y0_pu
    ppci1["internal"]["Ybus"] = Y1_pu
    ppci2["internal"]["Ybus"] = Y2_pu    
    ppci0["internal"]["Yf"] = Y0_f
    ppci1["internal"]["Yf"] = Y1_f
    ppci2["internal"]["Yf"] = Y2_f   
    ppci0["internal"]["Yt"] = Y0_t
    ppci1["internal"]["Yt"] = Y1_t
    ppci2["internal"]["Yt"] = Y2_t    
    I012_res = _current_from_voltage_results(Y0_pu, Y1_pu,Y2_pu,V012_new)
    S012_res = S_from_VI_elementwise(V012_new,I012_res) * ppci1["baseMVA"]    
    eg_is_mask = net["_is_elements"]['ext_grid']
    ext_grid_lookup = net["_pd2ppc_lookups"]["ext_grid"]
    eg_is_idx = net["ext_grid"].index.values[eg_is_mask]
    eg_idx_ppc = ext_grid_lookup[eg_is_idx]
    """ # 2 ext_grids Fix: Instead of the generator index, bus indices of the generators are used"""
    eg_bus_idx_ppc = np.real(ppci1["gen"][eg_idx_ppc, GEN_BUS]).astype(int)
    
    ppci0["gen"][eg_idx_ppc, PG] = S012_res[0,eg_bus_idx_ppc].real
    ppci1["gen"][eg_idx_ppc, PG] = S012_res[1,eg_bus_idx_ppc].real
    ppci2["gen"][eg_idx_ppc, PG] = S012_res[2,eg_bus_idx_ppc].real
    ppci0["gen"][eg_idx_ppc, QG] = S012_res[0,eg_bus_idx_ppc].imag
    ppci1["gen"][eg_idx_ppc, QG] = S012_res[1,eg_bus_idx_ppc].imag
    ppci2["gen"][eg_idx_ppc, QG] = S012_res[2,eg_bus_idx_ppc].imag
    
    ppc0 = net["_ppc0"]
    ppc1 = net["_ppc1"]
    ppc2 = net["_ppc2"]

    # ppci doesn't contain out of service elements, but ppc does -> copy results accordingly
    ppc0 = _copy_results_ppci_to_ppc(ppci0, ppc0, mode=mode)
    ppc1 = _copy_results_ppci_to_ppc(ppci1, ppc1, mode=mode)
    ppc2 = _copy_results_ppci_to_ppc(ppci2, ppc2, mode=mode)

    _extract_results_3ph(net, ppc0, ppc1, ppc2)

    #    Raise error if PF was not successful. If DC -> success is always 1

    if ppci0["success"] != True:
        net["converged"] = False
        _clean_up(net, res=False)
        raise LoadflowNotConverged("Power Flow {0} did not converge after\
                                {1} iterations!".format("nr", 3*max_iteration))
    else:
        net["converged"] = True

    _clean_up(net)

    return count, V012_it, I012_res


#def _phase_from_sequence_results(ppci0, Y1_pu, V012_pu,gs_eg,bs_eg):
def _current_from_voltage_results(Y0_pu, Y1_pu,Y2_pu, V012_pu):
#    ref, pv, pq = bustypes(ppci0["bus"], ppci0["gen"])
#    ppci0["bus"][ref, GS] -= gs_eg
#    ppci0["bus"][ref, BS] -= bs_eg
#
#    Y0_pu, _, _ = makeYbus(ppci0["baseMVA"], ppci0["bus"], ppci0["branch"])
    I012_pu = combine_X012(I0_from_V012(V012_pu, Y0_pu),
                            I1_from_V012(V012_pu, Y1_pu),
                            I2_from_V012(V012_pu, Y2_pu)) # Change it to Y1 to remove ext_grid impedance
    return I012_pu


def _get_Y_bus(ppci0, ppci1, ppci2, recycle, makeYbus):
    if recycle is not None and recycle["Ybus"] and ppci0["internal"]["Ybus"].size and ppci1["internal"]["Ybus"].size and \
            ppci2["internal"]["Ybus"].size:
        Y0_bus, Y0_f, Y0_t = ppci0["internal"]['Ybus'], ppci0["internal"]['Yf'], ppci0["internal"]['Yt']
        Y1_bus, Y1_f, Y1_t = ppci1["internal"]['Ybus'], ppci1["internal"]['Yf'], ppci1["internal"]['Yt']
        Y2_bus, Y2_f, Y2_t = ppci2["internal"]['Ybus'], ppci2["internal"]['Yf'], ppci2["internal"]['Yt']
    else:
        ## build admittance matrices
        Y0_bus, Y0_f, Y0_t = makeYbus(ppci0["baseMVA"], ppci0["bus"], ppci0["branch"])
        Y1_bus, Y1_f, Y1_t = makeYbus(ppci1["baseMVA"], ppci1["bus"], ppci1["branch"])
        Y2_bus, Y2_f, Y2_t = makeYbus(ppci2["baseMVA"], ppci2["bus"], ppci2["branch"])
        if recycle is not None and recycle["Ybus"]:
            ppci0["internal"]['Ybus'], ppci0["internal"]['Yf'], ppci0["internal"]['Yt'] = Y0_bus, Y0_f, Y0_t
            ppci1["internal"]['Ybus'], ppci1["internal"]['Yf'], ppci1["internal"]['Yt'] = Y1_bus, Y1_f, Y1_t
            ppci2["internal"]['Ybus'], ppci2["internal"]['Yf'], ppci2["internal"]['Yt'] = Y2_bus, Y2_f, Y2_t

    return ppci0, ppci1, ppci2, Y0_bus, Y1_bus, Y2_bus, Y0_f, Y1_f, Y2_f, Y0_t, Y1_t, Y2_t
