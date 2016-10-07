# -*- coding: utf-8 -*-

from __future__ import absolute_import
__author__ = 'tdess, lthurner, scheidler'

import numpy as np
import pandas as pd
import warnings
import copy

import pypower.runpf as runpf
import pypower.rundcpf as rundcpf
import pypower.ppoption as ppopt

from pandapower.auxiliary import HpException, get_indices
from pandapower.results import _extract_results
from pandapower.build_branch import _build_branch_mpc, _switch_branches
from pandapower.build_bus import _build_bus_mpc, _calc_loads_and_add_on_mpc, _calc_shunts_and_add_on_mpc
from pandapower.build_gen import _build_gen_mpc

class LoadflowNotConverged(HpException):
    """
    Exception being raised in case loadflow did not converge.
    """
    pass

def runpp(net, init="flat", calculate_voltage_angles=False, tolerance_kva=1e-5, trafo_model="t",
          trafo_loading="current", enforce_q_lims=False, suppress_warnings=True, **kwargs):
    """
    Runs PANDAPOWER Flow

    Note: May raise pandapower.api.run["load"]flowNotConverged

    INPUT:
        **net** - The Pandapower format network

    Optional:
    
        **init** (str, "flat") - initialization method of the loadflow
        Pandapower supports three methods for initializing the loadflow:

            - "flat"- flat start with voltage of 1.0pu and angle of 0° at all busses as initial solution
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
    net["converged"] = False
    if not init == "results":
        reset_results(net)

    # select elements in service (time consuming, so we do it once)
    gen_is = net["gen"][net["gen"]["in_service"].values.astype(bool)]
    eg_is = net["ext_grid"][net["ext_grid"]["in_service"].values.astype(bool)]

    # convert pandapower net to mpc
    mpc, bus_lookup = _pd2mpc(net, gen_is, eg_is, calculate_voltage_angles, enforce_q_lims,
                              trafo_model, init == "results")
    net["_mpc_last_cycle"] = mpc
    if not "VERBOSE" in kwargs:
        kwargs["VERBOSE"] = 0
    # initalize voltage
    if init == "dc":
        if suppress_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mpc = rundcpf.rundcpf(mpc, ppopt=ppopt.ppoption(**kwargs))[0]
        else:
            mpc = rundcpf.rundcpf(mpc, ppopt=ppopt.ppoption(ENFORCE_Q_LIMS=enforce_q_lims,
                                                            **kwargs))[0]

    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = runpf.runpf(mpc,ppopt=ppopt.ppoption(ENFORCE_Q_LIMS=enforce_q_lims,
                                                       PF_TOL=tolerance_kva*1e-3, **kwargs))[0]
            
    else:
        result = runpf.runpf(mpc, ppopt=ppopt.ppoption(ENFORCE_Q_LIMS=enforce_q_lims,
                                                       PF_TOL=tolerance_kva*1e-3, **kwargs))[0]
                                    
    if result["success"] != 1:
        raise LoadflowNotConverged("Loadflow did not converge!")
    else:
        net["_mpc_last_cycle"] = result
        net["converged"] = True

    _extract_results(net, result, gen_is, eg_is, bus_lookup, trafo_loading, True)
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
           

def _pd2mpc(net, gen_is, eg_is, calculate_voltage_angles=False, enforce_q_lims=False,
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


    RETURN:
        **mpc** - The simple matpower format network. Which consists of:
                  mpc = {
                        "baseMVA": 1., *float*
                        "version": 2,  *int*
                        "bus": np.array([], dtype=float),
                        "branch": np.array([], dtype=np.complex128),
                        "gen": np.array([], dtype=float)
    """
    mpc = {"baseMVA": 1.,
           "version": 2,
           "bus": np.array([], dtype=float),
           "branch": np.array([], dtype=np.complex128),
           "gen": np.array([], dtype=float)}
    bus_lookup = _build_bus_mpc(net, mpc, calculate_voltage_angles, gen_is, eg_is, init_results)
    _build_gen_mpc(net, mpc,  gen_is, eg_is, bus_lookup, enforce_q_lims, calculate_voltage_angles)
    _build_branch_mpc(net, mpc, bus_lookup, calculate_voltage_angles, trafo_model)
    _calc_loads_and_add_on_mpc(net, mpc, bus_lookup)
    _calc_shunts_and_add_on_mpc(net, mpc, bus_lookup)
    _switch_branches(net, mpc, bus_lookup)
    _set_out_of_service(net, mpc)

    # l = mpc["bus"][:, 0]  # The bus index ...
    # print(all(l[i] <= l[i+1] for i in range(len(l)-1)),  # is ordered ...
    #       all(l[i]+1 == l[i+1] for i in range(len(l)-1)),  # is consecutive ...
    #       l[0] == 0)  # and starts with zero.
    # Note: we do fully achieve this part of the ext2int functionality, however we still need it
    # since it internally drops buses that are out of service (aka BUS_TYPE == 4) and reorders the
    # gens according to increasing bus index they are connected to.

    return mpc, bus_lookup

def _set_out_of_service(net, mpc):
    # set disconnected busses out of service
    # first check if busses are connected to branches
    disco = np.setxor1d(mpc["bus"][:, 0].astype(int),
                        mpc["branch"][mpc["branch"][:, 10] == 1, :2].real.astype(int).flatten())

    # but also check if they may be the only connection to an ext_grid
    disco = np.setdiff1d(disco, mpc["gen"][mpc["gen"][:, 7] == 1, :1].real.astype(int))
    mpc["bus"][disco, 1] = 4


def _clean_up(net):
    if len(net["trafo3w"]) > 0:
        busses_3w = net.trafo3w["ad_bus"].values
        net["res_bus"].drop(busses_3w, inplace=True)
        net["bus"].drop(busses_3w, inplace=True)
        net["trafo3w"].drop(["ad_bus"], axis=1, inplace=True)
        
    if len(net["xward"]) > 0:
        xward_busses = net["xward"]["ad_bus"].values
        net["bus"].drop(xward_busses, inplace=True)
        net["res_bus"].drop(xward_busses, inplace=True)
        net["xward"].drop(["ad_bus"], axis=1, inplace=True)