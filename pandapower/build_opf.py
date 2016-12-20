# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import copy
import numpy as np
from numpy import array, ones, zeros, complex128, nan_to_num, hstack, abs, isnan, float64
from pandapower.build_branch import _build_branch_ppc, _switch_branches, _branches_with_oos_buses
from pandapower.build_bus import _build_bus_ppc, _calc_shunts_and_add_on_ppc
from pandapower.run import _set_isolated_buses_out_of_service, _ppc2ppci
from pypower.idx_bus import BUS_TYPE, PD, QD
from pypower.idx_gen import QMIN, QMAX, PMIN, PMAX, GEN_STATUS, GEN_BUS, PG, VG, QG
from pypower.idx_bus import PV, REF, VA, VM, PQ, VMAX, VMIN
from pypower.idx_brch import T_BUS, F_BUS
import pypower.ppoption as ppoption
import numpy.core.numeric as ncn
from pandapower.auxiliary import _sum_by_group
from scipy import sparse
import warnings


def _pd2ppc_opf(net, is_elems, sg_is):
    """ we need to put the sgens into the gen table instead of the bsu table
    so we need to change _pd2ppc a little to get the ppc we need for the OPF
    """

    ppc = {"baseMVA": 1.,
           "version": 2,
           "bus": array([], dtype=float),
           "branch": array([], dtype=complex128),
           "gen": array([], dtype=float),
           "gencost": array([], dtype=float)
           , "internal": {
                  "Ybus": np.array([], dtype=np.complex128)
                  , "Yf": np.array([], dtype=np.complex128)
                  , "Yt": np.array([], dtype=np.complex128)
                  , "branch_is": np.array([], dtype=bool)
                  , "gen_is": np.array([], dtype=bool)
                  }
           }

    # init empty ppci
    ppci = copy.deepcopy(ppc)

    calculate_voltage_angles = False
    trafo_model = "t"

    # get in service elements
    eg_is = is_elems['ext_grid']
    gen_is = is_elems['gen']

    bus_lookup = _build_bus_ppc(net, ppc, is_elems, set_opf_constraints=True)
    _build_gen_opf(net, ppc,  gen_is, eg_is, bus_lookup, calculate_voltage_angles, sg_is)
    _build_branch_ppc(net, ppc, is_elems, bus_lookup, calculate_voltage_angles, trafo_model,
                      set_opf_constraints=True)
    _calc_shunts_and_add_on_ppc(net, ppc, is_elems, bus_lookup)
    _calc_loads_and_add_opf(net, ppc, bus_lookup)
    _switch_branches(net, ppc, is_elems, bus_lookup)
    _branches_with_oos_buses(net, ppc, is_elems, bus_lookup)
    _set_isolated_buses_out_of_service(net, ppc)
    # generates "internal" ppci format (for powerflow calc) from "external" ppc format and updates the bus lookup
    # Note: Also reorders buses and gens in ppc
    ppci, bus_lookup = _ppc2ppci(ppc, ppci, bus_lookup)

    return ppc, ppci, bus_lookup


def _make_objective(ppc, net, is_elems, sg_is, ppopt, objectivetype="linear", lambda_opf=1000, **kwargs):
    """
    Implementaton of diverse objective functions for the OPF of the Form C{N}, C{fparm},
    C{H} and C{Cw}

    INPUT:
        **ppc** - Matpower case of the net

        **ppopt** -

    OPTIONAL:

        **objectivetype** (string, "linear") - string with name of objective function

            - **"linear"** - Linear costs of the form  :math:`I\\cdot P_G`. :math:`P_G` represents
              the active power values of the generators. Target of this objectivefunction is to
              maximize the generator output.
              This then basically is this:

                  .. math::
                      max\{P_G\}

            - **"linear_minloss"** - Quadratic costs of the form
              :math:`I\\cdot P_G - dV_m^T Y_L dV_m`.
              :math:`P_G` represents the active power values of the generators,
              :math:`dV_m` the voltage drop for each line and :math:`Y_L` the line admittance
              matrix.
              Target of this objectivefunction is to maximize the generator output but minimize the
              linelosses.
              This then basically is this:

                  .. math::
                      max\{P_G - dVm^TY_{L}dVm\}

            .. note:: Both objective functions have the following constraints:

                .. math::
                    V_{m,min} < &V_m < V_{m,max}\\\\
                    P_{G,min} < &P_G < P_{G,max}\\\\
                    Q_{G,min} < &Q_G < Q_{G,max}\\\\
                    I < &I_{max}

        **net** (attrdict, None) - Pandapower network

    """
    ng = len(ppc["gen"])  # -
    nref = sum(ppc["bus"][:, BUS_TYPE] == REF)
    gen_is = is_elems['gen']
    eg_is = is_elems['ext_grid']

    if gen_is.empty:
        gen_cost_per_kw = array([])
    elif "cost_per_kw" in gen_is.columns:
        gen_cost_per_kw = gen_is.cost_per_kw
    else:
        gen_cost_per_kw = ones(len(gen_is))

    if sg_is.empty:
        sgen_cost_per_kw = array([])
    elif "cost_per_kw" in sg_is.columns:
        sgen_cost_per_kw = sg_is.cost_per_kw
    else:
        sgen_cost_per_kw = ones(len(sg_is))

    if "cost_per_kw" not in eg_is.columns:
        eg_cost_per_kw = ones(len(eg_is))
    else:
        eg_cost_per_kw = eg_is.cost_per_kw

    if objectivetype == "linear":

        ppc["gencost"] = zeros((ng, 8), dtype=float)
        ppc["gencost"][:nref, :] = array([1, 0, 0, 2, 0, 0, 1, 1]) # initializing gencost array for eg
        ppc["gencost"][:nref, 7] = nan_to_num(eg_cost_per_kw*1e3)
        ppc["gencost"][nref:ng, :] = array([1, 0, 0, 2, 0, 0, 1, 1])  # initializing gencost array
        ppc["gencost"][nref:ng, 7] = nan_to_num(hstack([sgen_cost_per_kw*1e3, gen_cost_per_kw*1e3]))

#        ppc["gencost"][:nref, :] = array([1, 0, 0, 2, 0, 1, 1, 1])
#        ppc["gencost"][nref:ng, :] = array([1, 0, 0, 2, 0, 1, 1, 0])  # initializing gencost array
#        ppc["gencost"][nref:ng, 5] = nan_to_num(hstack([sgen_cost_per_kw, gen_cost_per_kw]))
#        p = p_timestep/-1e3
#        p[p == 0] = 1e-6
#        ppc["gencost"][nref:ng, 6] = array(p)

        ppopt = ppoption.ppoption(ppopt, OPF_FLOW_LIM=2, OPF_VIOLATION=1e-1, OUT_LIM_LINE=2,
                                  PDIPM_GRADTOL=1e-10, PDIPM_COMPTOL=1e-10, PDIPM_COSTTOL=1e-10)

    if objectivetype == "linear_minloss":

        ppc["gencost"] = zeros((ng, 8), dtype=float)
        ppc["gencost"][:nref, :] = array([1, 0, 0, 2, 0, 0, 1, 1]) # initializing gencost array for eg
        ppc["gencost"][:nref, 7] = nan_to_num(eg_cost_per_kw*1e3)
        ppc["gencost"][nref:ng, :] = array([1, 0, 0, 2, 0, 0, 1, 1])  # initializing gencost array
        ppc["gencost"][nref:ng, 7] = nan_to_num(hstack([sgen_cost_per_kw*1e3, gen_cost_per_kw*1e3]))

#        ppc["gencost"][:nref, :] = array([1, 0, 0, 2, 0, 1, 1, 1])
#        ppc["gencost"][nref:ng, :] = array([1, 0, 0, 2, 0, 1, 1, 1])
        #ppc["gencost"][nref:ng, 5] = nan_to_num(hstack([sgen_cost_per_kw, gen_cost_per_kw]))
#        p = p_timestep/-1e3
#        p[p == 0] = 1e-6
#        ppc["gencost"][nref:ng, 6] = array(p)

        #print(ppc["gencost"][nref:ng, 6])

        ppopt = ppoption.ppoption(ppopt, OPF_FLOW_LIM=2, OPF_VIOLATION=1e-1, OUT_LIM_LINE=2,
                                  PDIPM_GRADTOL=1e-10, PDIPM_COMPTOL=1e-10, PDIPM_COSTTOL=1e-10)

        # Get additional counts
        nb = len(ppc["bus"])
        nl = len(ppc["branch"])
        dim = 2 * nb + 2 * ng + 2 * nl

#        print("nb %s" % nb)
#        print("ng %s" % ng)
#        print("nl %s" % nl)

        # Get branch admitance matrices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Ybus, Yf, Yt = makeYbus.makeYbus(1, ppc["bus"], ppc["branch"])

        #########################
        # Additional z variables
        #########################

        # z_k = u_i - u_j
        # z_m = alpha_i - alpha_j
        # with i,j = start and endbus from lines

        # Epsilon for z constraints
        eps = 0

        # z constraints upper and lower bounds
        l = ones(2*nl)*eps
        u = ones(2*nl)*-eps

        # Initialize A and H matrix
        H = sparse.csr_matrix((dim, dim), dtype=float)
        A = sparse.csr_matrix((2*nl, dim), dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(nl):
                bus_f = int(ppc["branch"][i, F_BUS].real)
                bus_t = int(ppc["branch"][i, T_BUS].real)
                # minimization of potential difference between two buses
                H[dim-2*nl+i, dim-2*nl+i] = abs(Ybus[bus_f, bus_t]) * lambda_opf  # weigthing of minloss
                A[i, nb+bus_f] = 1
                A[i, nb+bus_t] = -1
                A[i, dim-2*nl+i] = 1
                # minimization of angle between two buses
                H[dim-nl+i, dim-nl+i] = 800 * lambda_opf  # weigthing of angles
                A[nl+i, bus_f] = 1
                A[nl+i, bus_t] = -1
                A[nl+i, dim-nl+i] = 1

        # Linear transformation for new omega-vector
        N = sparse.lil_matrix((dim, dim), dtype=float)
        for i in range(dim - 2*nl, dim):
            N[i, i] = 1.0

        # Cw = 0, no linear costs in additional costfunction
        Cw = zeros(dim, dtype=float)
        # d = 1
        d = ones((dim, 1), dtype=float)
        # r = 0
        r = zeros((dim, 1), dtype=float)
        # k = 0
        k = zeros((dim, 1), dtype=float)
        # m = 1
        m = ones((dim, 1), dtype=float)

        # Set ppc matrices
        ppc["H"] = H
        ppc["Cw"] = Cw
        ppc["N"] = N
        ppc["A"] = A
        ppc["l"] = l
        ppc["u"] = u
        ppc["fparm"] = hstack((d, r, k, m))

    return ppc, ppopt


def _build_gen_opf(net, ppc, gen_is, eg_is, bus_lookup, calculate_voltage_angles, sg_is):
    '''
    Takes the empty ppc network and fills it with the gen values. The gen
    datatype will be float afterwards.

    **INPUT**:
        **net** -The Pandapower format network

        **ppc** - The PYPOWER format network to fill in values
    '''
    eg_end = len(eg_is)
    gen_end = eg_end + len(gen_is)
    sg_end = gen_end + len(sg_is)

    q_lim_default = 1e9  # which is 1000 TW - should be enough for distribution grids.
    p_lim_default = 1e9

    # initialize generator matrix
    ppc["gen"] = zeros(shape=(sg_end, 21), dtype=float)
    ppc["gen"][:] = array([0, 0, 0, q_lim_default, -q_lim_default, 1., 1., 1, p_lim_default,
                              -p_lim_default, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # add sgens first so pv bus types won't be overwritten
    if sg_end > gen_end:
        ppc["gen"][gen_end:sg_end, GEN_BUS] = bus_lookup[sg_is["bus"].values]
        ppc["gen"][gen_end:sg_end, PG] = - sg_is["p_kw"].values * 1e-3 * sg_is["scaling"].values
        ppc["gen"][gen_end:sg_end, QG] = sg_is["q_kvar"].values

        # set bus values for generator buses
        gen_buses = bus_lookup[sg_is["bus"].values]
        ppc["bus"][gen_buses, BUS_TYPE] = PQ

        # set constraints for PV generators
        if "min_q_kvar" in sg_is.columns:
            ppc["gen"][gen_end:sg_end, QMAX] = -sg_is["min_q_kvar"].values * 1e-3
            qmax = ppc["gen"][gen_end:sg_end, [QMIN]]
            ncn.copyto(qmax, -q_lim_default, where=isnan(qmax))
            ppc["gen"][gen_end:sg_end, [QMIN]] = qmax

        if "max_q_kvar" in sg_is.columns:
            ppc["gen"][gen_end:sg_end, QMIN] = -sg_is["max_q_kvar"].values * 1e-3
            qmin = ppc["gen"][gen_end:sg_end, [QMAX]]
            ncn.copyto(qmin, q_lim_default, where=isnan(qmin))
            ppc["gen"][gen_end:sg_end, [QMAX]] = qmin

        if "min_p_kw" in sg_is.columns:
            ppc["gen"][gen_end:sg_end, PMIN] = -sg_is["min_p_kw"].values * 1e-3
            pmax = ppc["gen"][gen_end:sg_end, [PMIN]]
            ncn.copyto(pmax, -p_lim_default, where=isnan(pmax))
            ppc["gen"][gen_end:sg_end, [PMIN]] = pmax
        if "max_p_kw" in sg_is.columns:
            ppc["gen"][gen_end:sg_end, PMAX] = -sg_is["max_p_kw"].values * 1e-3
            min_p_kw = ppc["gen"][gen_end:sg_end, [PMAX]]
            ncn.copyto(min_p_kw, p_lim_default, where=isnan(min_p_kw))
            ppc["gen"][gen_end:sg_end, [PMAX]] = min_p_kw

    # add ext grid / slack data
    ppc["gen"][:eg_end, GEN_BUS] = bus_lookup[eg_is["bus"].values]
    ppc["gen"][:eg_end, VG] = eg_is["vm_pu"].values
    ppc["gen"][:eg_end, GEN_STATUS] = eg_is["in_service"].values

    # set bus values for external grid buses
    eg_buses = bus_lookup[eg_is["bus"].values]
    if calculate_voltage_angles:
        ppc["bus"][eg_buses, VA] = eg_is["va_degree"].values
    ppc["bus"][eg_buses, BUS_TYPE] = REF
    ppc["bus"][eg_buses, VM] = eg_is["vm_pu"].values

    # REF busses don't have flexible voltages by definition:
    ppc["bus"][eg_buses, VMAX] = ppc["bus"][ppc["bus"][:, BUS_TYPE] == REF, VM]
    ppc["bus"][eg_buses, VMIN] = ppc["bus"][ppc["bus"][:, BUS_TYPE] == REF, VM]

    # add generator / pv data
    if gen_end > eg_end:
        ppc["gen"][eg_end:gen_end, GEN_BUS] = bus_lookup[gen_is["bus"].values]
        ppc["gen"][eg_end:gen_end, PG] = - gen_is["p_kw"].values * 1e-3 * gen_is["scaling"].values
        ppc["gen"][eg_end:gen_end, VG] = gen_is["vm_pu"].values

        # set bus values for generator buses
        gen_buses = bus_lookup[gen_is["bus"].values]
        ppc["bus"][gen_buses, BUS_TYPE] = PV
        ppc["bus"][gen_buses, VM] = gen_is["vm_pu"].values

        # set constraints for PV generators
        ppc["gen"][eg_end:gen_end, QMIN] = -gen_is["max_q_kvar"].values * 1e-3
        ppc["gen"][eg_end:gen_end, QMAX] = -gen_is["min_q_kvar"].values * 1e-3
        ppc["gen"][eg_end:gen_end, PMIN] = -gen_is["min_p_kw"].values * 1e-3
        ppc["gen"][eg_end:gen_end, PMAX] = -gen_is["max_p_kw"].values * 1e-3

        qmin = ppc["gen"][eg_end:gen_end, [QMIN]]
        ncn.copyto(qmin, -q_lim_default, where=isnan(qmin))
        ppc["gen"][eg_end:gen_end, [QMIN]] = qmin

        qmax = ppc["gen"][eg_end:gen_end, [QMAX]]
        ncn.copyto(qmax, q_lim_default, where=isnan(qmax))
        ppc["gen"][eg_end:gen_end, [QMAX]] = qmax

        min_p_kw = ppc["gen"][eg_end:gen_end, [PMIN]]
        ncn.copyto(min_p_kw, -p_lim_default, where=isnan(min_p_kw))
        ppc["gen"][eg_end:gen_end, [PMIN]] = min_p_kw

        pmax = ppc["gen"][eg_end:gen_end, [PMAX]]
        ncn.copyto(pmax, p_lim_default, where=isnan(pmax))
        ppc["gen"][eg_end:gen_end, [PMAX]] = pmax


def _calc_loads_and_add_opf(net, ppc, bus_lookup):
    """ we need to exclude controllable sgens from the bus table
    """

    l = net["load"]
    vl = l["in_service"].values * l["scaling"].values.T / float64(1000.)
    lp = l["p_kw"].values * vl
    lq = l["q_kvar"].values * vl

    sgen = net["sgen"]
    if not sgen.empty:
        vl = (sgen["in_service"].values & ~sgen["controllable"]) * sgen["scaling"].values.T / \
            float64(1000.)
        sp = sgen["p_kw"].values * vl
        sq = sgen["q_kvar"].values * vl
    else:
        sp = []
        sq = []

    b = bus_lookup[hstack([l["bus"].values, sgen["bus"].values])]
    b, vp, vq = _sum_by_group(b, hstack([lp, sp]), hstack([lq, sq]))

    ppc["bus"][b, PD] = vp
    ppc["bus"][b, QD] = vq
