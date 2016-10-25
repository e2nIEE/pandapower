# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import numpy as np
import warnings
import copy

import pypower.ppoption as ppopt

from pandapower.runpf import _runpf
from pandapower.auxiliary import ppException
from pandapower.results import _extract_results
from pandapower.build_branch import _build_branch_mpc, _switch_branches, _branches_with_oos_buses
from pandapower.build_bus import _build_bus_mpc, _calc_loads_and_add_on_mpc, \
                                 _calc_shunts_and_add_on_mpc
from pandapower.build_gen import _build_gen_mpc

class LoadflowNotConverged(ppException):
    """
    Exception being raised in case loadflow did not converge.
    """
    pass

def runpp(net, init="flat", calculate_voltage_angles=False, tolerance_kva=1e-5, trafo_model="t",
          trafo_loading="current", enforce_q_lims=False, suppress_warnings=True, **kwargs):
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

        **suppress_warnings** (bool, True) - suppress warnings in pypower
        
            If set to True, warnings are disabled during the loadflow. Because of the way data is
            processed in pypower, ComplexWarnings are raised during the loadflow. These warnings are
            suppressed by this option, however keep in mind all other pypower warnings are also suppressed.
        
        ****kwargs** - options to use for PYPOWER.runpf
    """
    ac = True

    _runpppf(net, init, ac, calculate_voltage_angles, tolerance_kva, trafo_model,
             trafo_loading, enforce_q_lims, suppress_warnings, **kwargs)


def rundcpp(net, trafo_model="t", trafo_loading="current", suppress_warnings=True, **kwargs):
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
        
        ****kwargs** - options to use for PYPOWER.runpf
    """
    ac = False
    # the following parameters have no effect if ac = False
    calculate_voltage_angles = True
    enforce_q_lims = False
    init = ''
    tolerance_kva = 1e-5

    _runpppf(net, init, ac, calculate_voltage_angles, tolerance_kva, trafo_model,
             trafo_loading, enforce_q_lims, suppress_warnings, **kwargs)

def _runpppf(net, init, ac, calculate_voltage_angles, tolerance_kva, trafo_model,
          trafo_loading, enforce_q_lims, suppress_warnings, **kwargs):
    """
    Gets called by runpp or rundcpp with different arguments.
    """

    net["converged"] = False
    if (ac and not init == "results") or not ac:
        reset_results(net)

    # select elements in service (time consuming, so we do it once)
    is_elems = _select_is_elements(net)

    # convert pandapower net to mpc
    mpc, ppc, bus_lookup = _pd2mpc(net, is_elems, calculate_voltage_angles, enforce_q_lims,
                                   trafo_model, init_results=(init == "results"))
    net["_mpc_last_cycle"] = mpc
    if not "VERBOSE" in kwargs:
        kwargs["VERBOSE"] = 0

    # run the powerflow with or without warnings. If init='dc', AC PF will be initialized with DC voltages
    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _runpf(ppc, init, ac, ppopt=ppopt.ppoption(ENFORCE_Q_LIMS=enforce_q_lims,
                                                                     PF_TOL=tolerance_kva * 1e-3, **kwargs))[0]

    else:
        result = _runpf(ppc, init, ac, ppopt=ppopt.ppoption(ENFORCE_Q_LIMS=enforce_q_lims,
                                                                 PF_TOL=tolerance_kva * 1e-3, **kwargs))[0]

    # ppc doesn't contain out of service elements, but mpc does -> copy results accordingly
    result = _copy_results_ppc_to_mpc(result, mpc)

    # raise if PF was not successful. If DC -> success is always 1
    if result["success"] != 1:
        raise LoadflowNotConverged("Loadflow did not converge!")
    else:
        net["_mpc_last_cycle"] = result
        net["converged"] = True

    _extract_results(net, result, is_elems, bus_lookup, trafo_loading, ac)
    _clean_up(net)

def reset_results(net):
    net["res_bus"] = copy.copy(net["_empty_res_bus"])
    net["res_ext_grid"] = copy.copy(net["_empty_res_ext_grid"])
    net["res_line"] = copy.copy(net["_empty_res_line"])
    net["res_load"] = copy.copy(net["_empty_res_load"])
    net["res_sgen"] = copy.copy(net["_empty_res_sgen"])
    net["res_trafo"] = copy.copy(net["_empty_res_trafo"])
    net["res_trafo3w"] = copy.copy(net["_empty_res_trafo3w"])
    net["res_shunt"] = copy.copy(net["_empty_res_shunt"])
    net["res_impedance"] = copy.copy(net["_empty_res_impedance"])
    net["res_gen"] = copy.copy(net["_empty_res_gen"])
    net["res_ward"] = copy.copy(net["_empty_res_ward"])
    net["res_xward"] = copy.copy(net["_empty_res_xward"])

def _select_is_elements(net):
    """
    Selects certain "in_service" elements from net.
    This is quite time consuming so it is done once at the beginning


    @param net: Pandapower Network
    @return: is_elems Certain in service elements
    """
    is_elems = {
        'gen' : net["gen"][net["gen"]["in_service"].values.astype(bool)]
        ,'eg' : net["ext_grid"][net["ext_grid"]["in_service"].values.astype(bool)]
        ,'bus' : net["bus"][net["bus"]["in_service"].values.astype(bool)]
        ,'line' : net["line"][net["line"]["in_service"].values.astype(bool)]
    }

    # check if gen is also at an in service bus
    gen_is = np.in1d(net["gen"].bus.values, is_elems['bus'].index) \
              & net["gen"]["in_service"].values.astype(bool)
    is_elems['gen'] = net['gen'][gen_is]

    return is_elems

def _copy_results_ppc_to_mpc(result, mpc):
    '''
    result contains results for all in service elements
    mpc shall get the results for in- and out of service elements
    -> results must be copied

    mpc and ppc are structured as follows:

          [in_service elements]
    mpc = [out_of_service elements]

    result = [in_service elements]

    @author: fschaefer

    @param result:
    @param mpc:
    @return:
    '''

    # copy the results for bus, gen and branch
    # busses are sorted (REF, PV, PQ, NONE) -> results are the first 3 types
    mpc['bus'][:len(result['bus'])] = result['bus']
    # in service branches and gens are taken from 'order'
    mpc['branch'][result['branch_is']] = result['branch']
    mpc['gen'][result['gen_is']] = result['gen']

    mpc['success'] = result['success']
    mpc['et'] = result['et']

    result = mpc
    return result


def _pd2mpc(net, is_elems, calculate_voltage_angles=False, enforce_q_lims=False,
            trafo_model="pi", init_results=False):
    """
    Converter Flow:
        1. Create an empty pypower datatructure
        2. Calculate loads and write the bus matrix
        3. Build the gen (Infeeder)- Matrix
        4. Calculate the line parameter and the transformer parameter,
           and fill it in the branch matrix.
           Order: 1st: Line values, 2nd: Trafo values


    INPUT:
        **net** - The Pandapower format network
        **is_elems** - In service elements from the network (see _select_is_elements())


    RETURN:
        **mpc** - The simple matpower format network. Which consists of:
                  mpc = {
                        "baseMVA": 1., *float*
                        "version": 2,  *int*
                        "bus": np.array([], dtype=float),
                        "branch": np.array([], dtype=np.complex128),
                        "gen": np.array([], dtype=float)
        **ppc** - The "internal" pypower format network for PF calculations
        **bus_lookup** - Lookup Pandapower -> mpc / ppc indices
    """

    # init empty mpc
    mpc = {"baseMVA": 1.,
           "version": 2,
           "bus": np.array([], dtype=float),
           "branch": np.array([], dtype=np.complex128),
           "gen": np.array([], dtype=float)}
    # generate mpc['bus'] and the bus lookup
    bus_lookup = _build_bus_mpc(net, mpc, is_elems, init_results)
    # generate mpc['gen'] and fills mpc['bus'] with generator values (PV, REF nodes)
    _build_gen_mpc(net, mpc, is_elems, bus_lookup, enforce_q_lims, calculate_voltage_angles)
    # generate mpc['branch'] and directly generates branch values
    _build_branch_mpc(net, mpc, is_elems, bus_lookup, calculate_voltage_angles, trafo_model)
    # adds P and Q for loads / sgens in mpc['bus'] (PQ nodes)
    _calc_loads_and_add_on_mpc(net, mpc, is_elems, bus_lookup)
    # adds P and Q for shunts, wards and xwards (to PQ nodes)
    _calc_shunts_and_add_on_mpc(net, mpc, is_elems, bus_lookup)
    # adds auxilary buses for open switches at branches
    _switch_branches(net, mpc, is_elems, bus_lookup)
    # add auxilary buses for out of service buses at in service lines.
    # Also sets lines out of service if they are connected to two out of service buses
    _branches_with_oos_buses(net, mpc, is_elems, bus_lookup)
    # sets buses out of service, which aren't connected to branches / REF buses
    _set_isolated_buses_out_of_service(net, mpc)
    # generates "internal" ppc format (for powerflow calc) from "external" mpc format and updates the bus lookup
    # Note: Also reorders buses and gens in mpc
    ppc, bus_lookup = _mpc2ppc(mpc, bus_lookup)

    # add lookup with indices before any busses were fused
    bus_lookup["before_fuse"] = dict(zip(net["bus"].index.values, np.arange(len(net["bus"].index.values))))

    return mpc, ppc, bus_lookup


def _mpc2ppc(mpc, bus_lookup):
    from numpy import array, zeros

    from scipy.sparse import csr_matrix as sparse

    from pypower.idx_bus import NONE, BUS_I, BUS_TYPE
    from pypower.idx_gen import GEN_BUS, GEN_STATUS
    from pypower.idx_brch import F_BUS, T_BUS, BR_STATUS, QT
    from pypower.idx_area import PRICE_REF_BUS

    from pypower.run_userfcn import run_userfcn

    # init ppc
    ppc = {"baseMVA": 1.,
           "version": 2,
           "bus": np.array([], dtype=float),
           "branch": np.array([], dtype=np.complex128),
           "gen": np.array([], dtype=float),
           "branch_is": np.array([], dtype=bool),
           "gen_is": np.array([], dtype=bool)}

    ## BUS Sorting and lookup
    # sort busses in descending order of column 1 (namely: 4 (OOS), 3 (REF), 2 (PV), 1 (PQ))
    mpcBuses = mpc["bus"]
    mpc['bus'] = mpcBuses[mpcBuses[:, BUS_TYPE].argsort(axis=0)[::-1][:],]
    # get OOS busses and place them at the end of the bus array (so that: 3 (REF), 2 (PV), 1 (PQ), 4 (OOS))
    oos_busses = mpc['bus'][:, BUS_TYPE] == NONE
    # there are no OOS busses in the ppc
    ppc['bus'] = mpc['bus'][~oos_busses]
    # in mpc the OOS busses are included and at the end of the array
    mpc['bus'] = np.r_[mpc['bus'][~oos_busses], mpc['bus'][oos_busses]]
    # generate bus_lookup_mpc_ppc (mpc -> ppc lookup)
    mpc_former_order = (mpc['bus'][:, BUS_I]).astype(int)
    arangedBuses = np.arange(len(mpcBuses))

    # lookup mpc former order -> consecutive order
    e2i = zeros( len(mpcBuses) )
    e2i[mpc_former_order] = arangedBuses

    # save consecutive indices in mpc and ppc
    mpc['bus'][:, BUS_I] = arangedBuses
    ppc['bus'][:, BUS_I] = mpc['bus'][:len(ppc['bus']), BUS_I]

    # update bus_lookup (pandapower -> ppc internal)
    bus_lookup = {key: e2i[val] for (key, val) in bus_lookup.items()}

    ## sizes
    nb = mpc["bus"].shape[0]
    ng = mpc["gen"].shape[0]

    if 'areas' in mpc:
        if len(mpc["areas"]) == 0:  ## if areas field is empty
            del mpc['areas']  ## delete it (so it's ignored)

    # bus types
    bt = mpc["bus"][:, BUS_TYPE]

    ## update branch, gen and areas bus numbering
    mpc['gen'][:, GEN_BUS] = \
        e2i[np.real(mpc["gen"][:, GEN_BUS]).astype(int)].copy()
    mpc["branch"][:, F_BUS] = \
        e2i[np.real(mpc["branch"][:, F_BUS]).astype(int)].copy()
    mpc["branch"][:, T_BUS] = \
        e2i[np.real(mpc["branch"][:, T_BUS]).astype(int)].copy()

    #Note: The "update branch, gen and areas bus numbering" does the same as this:
    # mpc['gen'][:, GEN_BUS] = get_indices(mpc['gen'][:, GEN_BUS], bus_lookup_mpc_ppc)
    # mpc["branch"][:, F_BUS] = get_indices(mpc["branch"][:, F_BUS], bus_lookup_mpc_ppc)
    # mpc["branch"][:, T_BUS] = get_indices( mpc["branch"][:, T_BUS], bus_lookup_mpc_ppc)
    # but faster...

    if 'areas' in mpc:
        mpc["areas"][:, PRICE_REF_BUS] = \
            e2i[np.real(mpc["areas"][:, PRICE_REF_BUS]).astype(int)].copy()

    ## reorder gens in order of increasing bus number
    mpc['gen'] = mpc['gen'][mpc['gen'][:, GEN_BUS].argsort(),]

    ## determine which buses, branches, gens are connected and
    ## in-service
    # n2i = sparse((range(nb), (mpc["bus"][:, BUS_I], zeros(nb))),
    #              shape=(maxBus + 1, 1))
    # n2i = array(n2i.todense().flatten())[0, :]  # as 1D array
    n2i = mpc["bus"][:, BUS_I].astype(int)
    bs = (bt != NONE)  ## bus status

    gs = ((mpc["gen"][:, GEN_STATUS] > 0) &  ## gen status
          bs[n2i[np.real(mpc["gen"][:, GEN_BUS]).astype(int)]])
    ppc["gen_is"] = gs

    brs = (np.real(mpc["branch"][:, BR_STATUS]).astype(int) &  ## branch status
           bs[n2i[np.real(mpc["branch"][:, F_BUS]).astype(int)]] &
           bs[n2i[np.real(mpc["branch"][:, T_BUS]).astype(int)]]).astype(bool)
    ppc["branch_is"] = brs

    if 'areas' in mpc:
        ar = bs[n2i[mpc["areas"][:, PRICE_REF_BUS].astype(int)]]
        # delete out of service areas
        ppc["areas"] = mpc["areas"][ar]

    ## select in service elements from mpc and put them in ppc
    ppc["branch"] = mpc["branch"][brs]
    ppc["gen"] = mpc["gen"][gs]

    ## execute userfcn callbacks for 'ext2int' stage
    if 'userfcn' in ppc:
        ppc = run_userfcn(ppc['userfcn'], 'ext2int', ppc)

    return ppc, bus_lookup


def _set_isolated_buses_out_of_service(net, mpc):
    # set disconnected buses out of service
    # first check if buses are connected to branches
    disco = np.setxor1d(mpc["bus"][:, 0].astype(int),
                        mpc["branch"][mpc["branch"][:, 10] == 1, :2].real.astype(int).flatten())

    # but also check if they may be the only connection to an ext_grid
    disco = np.setdiff1d(disco, mpc['bus'][mpc['bus'][:,1] == 3, :1].real.astype(int))
    mpc["bus"][disco, 1] = 4


def _clean_up(net):
    if len(net["trafo3w"]) > 0:
        buses_3w = net.trafo3w["ad_bus"].values
        net["res_bus"].drop(buses_3w, inplace=True)
        net["bus"].drop(buses_3w, inplace=True)
        net["trafo3w"].drop(["ad_bus"], axis=1, inplace=True)
        
    if len(net["xward"]) > 0:
        xward_buses = net["xward"]["ad_bus"].values
        net["bus"].drop(xward_buses, inplace=True)
        net["res_bus"].drop(xward_buses, inplace=True)
        net["xward"].drop(["ad_bus"], axis=1, inplace=True)