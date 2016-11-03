# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandapower as pp
from pandapower.build_branch import _build_branch_mpc, _switch_branches, _branches_with_oos_buses
from pandapower.build_bus import _build_bus_mpc, _calc_shunts_and_add_on_mpc
from pandapower.run import _set_isolated_buses_out_of_service
from pypower.idx_bus import BUS_TYPE, PD, QD
from pypower.idx_gen import QMIN, QMAX, PMIN, PMAX, GEN_STATUS, GEN_BUS, PG, VG, QG
from pypower.idx_bus import PV, REF, VA, VM, PQ, VMAX, VMIN
from pypower.idx_brch import RATE_A, T_BUS, F_BUS
import pypower.ppoption as ppoption
import numpy.core.numeric as ncn
from pandapower.auxiliary import _sum_by_group
from scipy import sparse
from pypower import makeYbus
import warnings

def _pd2mpc_opf(net, is_elems, sg_is):
    """ we need to put the sgens into the gen table instead of the bsu table 
    so we need to change _pd2mpc a little to get the mpc we need for the OPF
    """

    mpc = {"baseMVA": 1.,
           "version": 2,
           "bus": np.array([], dtype=float),
           "branch": np.array([], dtype=np.complex128),
           "gen": np.array([], dtype=float),
           "gencost": np.array([], dtype=float)}

    calculate_voltage_angles = False
    trafo_model = "t"

    # get in service elements
    eg_is = is_elems['eg']
    gen_is = is_elems['gen']

    bus_lookup = _build_bus_mpc(net, mpc, is_elems, set_opf_constraints=True)
    _build_gen_opf(net, mpc,  gen_is, eg_is, bus_lookup, calculate_voltage_angles, sg_is)
    _build_branch_mpc(net, mpc, is_elems, bus_lookup, calculate_voltage_angles, trafo_model, \
                      set_opf_constraints=True)
    _calc_shunts_and_add_on_mpc(net, mpc, is_elems, bus_lookup)
    _calc_loads_and_add_opf(net, mpc, bus_lookup)
    _switch_branches(net, mpc, is_elems, bus_lookup)
    _branches_with_oos_buses(net, mpc, is_elems, bus_lookup)
    _set_isolated_buses_out_of_service(net, mpc)
    bus_lookup["before_fuse"] = dict(zip(net["bus"].index.values, \
                                    np.arange(len(net["bus"].index.values))))

    return mpc, bus_lookup


def _make_objective(mpc, net, is_elems, sg_is, ppopt, objectivetype="maxp"):
    """ 
    Implementaton of diverse objective functions for the OPF of the Form C{N}, C{fparm},
    C{H} and C{Cw}

    INPUT:
        **mpc** - Matpower case of the net

        **ppopt** -
        
    OPTIONAL:

        **objectivetype** (string, "maxp") - string with name of objective function

            - **"maxp"** - Linear costs of the form  :math:`I\\cdot P_G`. :math:`P_G` represents the
              active power values of the generators. Target of this objectivefunction is to maximize
              the generator output.
              This then basically is this:

                  .. math::
                      max\{P_G\}

            - **"minlossmaxp"** - Quadratic costs of the form  :math:`I\\cdot P_G - dV_m^T Y_L dV_m`.
              :math:`P_G` represents the active power values of the generators,
              :math:`dV_m` the voltage drop for each line and :math:`Y_L` the line admittance matrix.
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
    ng = len(mpc["gen"])  # -
    nref = sum(mpc["bus"][:, BUS_TYPE] == REF)
    gen_is = is_elems['gen']
    eg_is = is_elems['eg']

    if not gen_is.empty:
        gen_cost_per_kw = gen_is.cost_per_kw
    else:
        gen_cost_per_kw = np.array([])
        
    if  not sg_is.empty:
        sgen_cost_per_kw = sg_is.cost_per_kw
    else:
        sgen_cost_per_kw = np.array([])

    if objectivetype == "linear":

        mpc["gencost"] = np.zeros((ng, 8), dtype=float)
        if hasattr(net.ext_grid, "cost_per_kw"):
            mpc["gencost"][:nref, :] = np.array([1, 0, 0, 2, 0, 0, 100, net.ext_grid.cost_per_kw])
            
        else:
            mpc["gencost"][:nref, :] = np.array([1, 0, 0, 2, 0, 0, 100, 0])# no costs for ext_grid
        mpc["gencost"][nref:ng, :] = np.array([1, 0, 0, 2, 0, 0, 100, 0]) # initializing gencost array
        mpc["gencost"][nref:ng, 7] = np.nan_to_num(np.hstack([sgen_cost_per_kw, gen_cost_per_kw]))

        ppopt = ppoption.ppoption(ppopt, OPF_FLOW_LIM=2, OPF_VIOLATION=1e-1, OUT_LIM_LINE=2,
                                  PDIPM_GRADTOL=1e-10, PDIPM_COMPTOL=1e-10, PDIPM_COSTTOL=1e-10)

    if objectivetype == "linear_minloss":

        mpc["gencost"] = np.zeros((ng, 8), dtype=float)
        if hasattr(net.ext_grid, "cost_per_kw"):
            mpc["gencost"][:nref, :] = np.array([1, 0, 0, 2, 0, 0, 100, net.ext_grid.cost_per_kw])
        else:
            mpc["gencost"][:nref, :] = np.array([1, 0, 0, 2, 0, 0, 100, 0])# no costs for ext_grid
            
        mpc["gencost"][nref:ng, :] = np.array([1, 0, 0, 2, 0, 100, 100, 0])

        # Set gencosts for sgens
        mpc["gencost"][nref:ng, 5] = net.sgen.cost_per_kw

        ppopt = ppoption.ppoption(ppopt, OPF_FLOW_LIM=2, OPF_VIOLATION=1e-1, OUT_LIM_LINE=2,
                                  PDIPM_GRADTOL=1e-10, PDIPM_COMPTOL=1e-10, PDIPM_COSTTOL=1e-10)

        # Get additional counts
        nb = len(mpc["bus"])
        nl = len(mpc["branch"])
        dim = 2*nb+2*ng+nl

        # Get branch admitance matrices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Ybus, Yf, Yt = makeYbus.makeYbus(1, mpc["bus"], mpc["branch"])

        #########################
        # Additional z variables
        #########################

        # z_k = u_i - u_j
        # with i,j = start and endbus from lines

        # Epsilon for z constraints
        eps = 1e-3

        # z constraints upper and lower bounds
        l = np.ones(nl) * -eps
        u = np.ones(nl) * eps

        # Initialzie A and H matrix
        H = sparse.csr_matrix((dim, dim), dtype=float)
        A = sparse.csr_matrix((nl, dim), dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(nl):
                bus_f = int(mpc["branch"][i, F_BUS].real)
                bus_t = int(mpc["branch"][i, T_BUS].real)
                H[dim-nl+i, dim-nl+i] = np.abs(Ybus[bus_f, bus_t]) * 1000 # weigthing of minloss
                A[i, nb+bus_f] = 1
                A[i, nb+bus_t] = -1
                A[i, dim-nl+i] = 1

        # Linear transformation for new omega-vector
        N = sparse.csr_matrix((dim, dim), dtype=float)
        for i in range(dim-nl, dim):
            N[i, i] = 1.0

        # Cw = 0, no linear costs in additional costfunction
        Cw = np.zeros(dim, dtype=float)
        # d = 1
        d = np.ones((dim, 1), dtype=float)
        # r = 0
        r = np.zeros((dim, 1), dtype=float)
        # k = 0
        k = np.zeros((dim, 1), dtype=float)
        # m = 1
        m = np.ones((dim, 1), dtype=float)

        # Set mpc matrices
        mpc["H"] = H
        mpc["Cw"] = Cw
        mpc["N"] = N
        mpc["A"] = A
        mpc["l"] = l
        mpc["u"] = u
        mpc["fparm"] = np.hstack((d, r, k, m))

    return mpc, ppopt


def _build_gen_opf(net, mpc, gen_is, eg_is, bus_lookup, calculate_voltage_angles, sg_is):
    '''
    Takes the empty mpc network and fills it with the gen values. The gen
    datatype will be float afterwards.

    **INPUT**:
        **net** -The Pandapower format network

        **mpc** - The PYPOWER format network to fill in values
    '''
    eg_end = len(eg_is)
    gen_end = eg_end + len(gen_is)
    sg_end = gen_end + len(sg_is)

    q_lim_default = 1e9  # which is 1000 TW - should be enough for distribution grids.
    p_lim_default = 1e9

    # initialize generator matrix
    mpc["gen"] = np.zeros(shape=(sg_end, 21), dtype=float)
    mpc["gen"][:] = np.array([0, 0, 0, q_lim_default, -q_lim_default, 1., 1., 1, p_lim_default,
                              -p_lim_default, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # add sgens first so pv bus types won't be overwritten
    if sg_end > gen_end:
        mpc["gen"][gen_end:sg_end, GEN_BUS] = pp.get_indices(sg_is["bus"].values, bus_lookup)
        mpc["gen"][gen_end:sg_end, PG] = - sg_is["p_kw"].values * 1e-3 * sg_is["scaling"].values
        mpc["gen"][gen_end:sg_end, QG] = sg_is["q_kvar"].values

        # set bus values for generator buses
        sg_buses = pp.get_indices(sg_is["bus"].values, bus_lookup)
        gen_buses = pp.get_indices(sg_is["bus"].values, bus_lookup)
        mpc["bus"][gen_buses, BUS_TYPE] = PQ

        # set constraints for PV generators
        if "min_q_kvar" in sg_is.columns:
            mpc["gen"][gen_end:sg_end, QMAX] = -sg_is["min_q_kvar"].values * 1e-3
            qmax = mpc["gen"][gen_end:sg_end, [QMIN]]
            ncn.copyto(qmax, -q_lim_default, where=np.isnan(qmax))
            mpc["gen"][gen_end:sg_end, [QMIN]] = qmax

        if "max_q_kvar" in sg_is.columns:
            mpc["gen"][gen_end:sg_end, QMIN] = -sg_is["max_q_kvar"].values * 1e-3
            qmin = mpc["gen"][gen_end:sg_end, [QMAX]]
            ncn.copyto(qmin, q_lim_default, where=np.isnan(qmin))
            mpc["gen"][gen_end:sg_end, [QMAX]] = qmin

        if "min_p_kw" in sg_is.columns:
            mpc["gen"][gen_end:sg_end, PMIN] = -sg_is["min_p_kw"].values * 1e-3
            pmax = mpc["gen"][gen_end:sg_end, [PMIN]]
            ncn.copyto(pmax, -p_lim_default, where=np.isnan(pmax))
            mpc["gen"][gen_end:sg_end, [PMIN]] = pmax
        if "max_p_kw" in sg_is.columns:
            mpc["gen"][gen_end:sg_end, PMAX] = -sg_is["max_p_kw"].values * 1e-3
            min_p_kw = mpc["gen"][gen_end:sg_end, [PMAX]]
            ncn.copyto(min_p_kw, p_lim_default, where=np.isnan(min_p_kw))
            mpc["gen"][gen_end:sg_end, [PMAX]] = min_p_kw

    # add ext grid / slack data
    mpc["gen"][:eg_end, GEN_BUS] = pp.get_indices(eg_is["bus"].values, bus_lookup)
    mpc["gen"][:eg_end, VG] = eg_is["vm_pu"].values
    mpc["gen"][:eg_end, GEN_STATUS] = eg_is["in_service"].values

    # set bus values for external grid buses
    eg_buses = pp.get_indices(eg_is["bus"].values, bus_lookup)
    if calculate_voltage_angles:
        mpc["bus"][eg_buses, VA] = eg_is["va_degree"].values
    mpc["bus"][eg_buses, BUS_TYPE] = REF

    # REF busses don't have flexible voltages by definition:
    mpc["bus"][eg_buses, VMAX] = mpc["bus"][mpc["bus"][:, BUS_TYPE] == REF, VM]
    mpc["bus"][eg_buses, VMIN] = mpc["bus"][mpc["bus"][:, BUS_TYPE] == REF, VM]
    
    # add generator / pv data
    if gen_end > eg_end:
        mpc["gen"][eg_end:gen_end, GEN_BUS] = pp.get_indices(gen_is["bus"].values, bus_lookup)
        mpc["gen"][eg_end:gen_end, PG] = - gen_is["p_kw"].values * 1e-3 * gen_is["scaling"].values
        mpc["gen"][eg_end:gen_end, VG] = gen_is["vm_pu"].values

        # set bus values for generator buses
        gen_buses = pp.get_indices(gen_is["bus"].values, bus_lookup)
        mpc["bus"][gen_buses, BUS_TYPE] = PV
        mpc["bus"][gen_buses, VM] = gen_is["vm_pu"].values

        # set constraints for PV generators
        mpc["gen"][eg_end:gen_end, QMIN] = -gen_is["max_q_kvar"].values * 1e-3
        mpc["gen"][eg_end:gen_end, QMAX] = -gen_is["min_q_kvar"].values * 1e-3
        mpc["gen"][eg_end:gen_end, PMIN] = -gen_is["min_p_kw"].values * 1e-3
        mpc["gen"][eg_end:gen_end, PMAX] = -gen_is["max_p_kw"].values * 1e-3

        qmin = mpc["gen"][eg_end:gen_end, [QMIN]]
        ncn.copyto(qmin, -q_lim_default, where=np.isnan(qmin))
        mpc["gen"][eg_end:gen_end, [QMIN]] = qmin

        qmax = mpc["gen"][eg_end:gen_end, [QMAX]]
        ncn.copyto(qmax, q_lim_default, where=np.isnan(qmax))
        mpc["gen"][eg_end:gen_end, [QMAX]] = qmax

        min_p_kw = mpc["gen"][eg_end:gen_end, [PMIN]]
        ncn.copyto(min_p_kw, -p_lim_default, where=np.isnan(min_p_kw))
        mpc["gen"][eg_end:gen_end, [PMIN]] = min_p_kw

        pmax = mpc["gen"][eg_end:gen_end, [PMAX]]
        ncn.copyto(pmax, p_lim_default, where=np.isnan(pmax))
        mpc["gen"][eg_end:gen_end, [PMAX]] = pmax


def _calc_loads_and_add_opf(net, mpc, bus_lookup):
    """ we need to exclude controllable sgens from the bus table
    """

    l = net["load"]
    vl = l["in_service"].values * l["scaling"].values.T / np.float64(1000.)
    lp = l["p_kw"].values * vl
    lq = l["q_kvar"].values * vl

    sgen = net["sgen"]
    if not sgen.empty:
        vl = (sgen["in_service"].values & ~sgen["controllable"]) * sgen["scaling"].values.T / \
            np.float64(1000.)
        sp = sgen["p_kw"].values * vl
        sq = sgen["q_kvar"].values * vl
    else:
        sp = []
        sq = []

    b = pp.get_indices(np.hstack([l["bus"].values, sgen["bus"].values]
                                 ), bus_lookup)
    b, vp, vq = _sum_by_group(b, np.hstack([lp, sp]), np.hstack([lq, sq]))

    mpc["bus"][b, PD] = vp
    mpc["bus"][b, QD] = vq
