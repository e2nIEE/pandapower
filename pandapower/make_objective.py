# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import numpy as np
import warnings

from pypower.idx_brch import T_BUS, F_BUS
from pypower.idx_bus import BUS_TYPE, REF
from pandas import DataFrame
from scipy import sparse

from pandapower.pypower_extensions.makeYbus_pypower import makeYbus


def _make_objective(ppc, net, is_elems, objectivetype="linear", lambda_opf=1000):
    """
    Implementaton of diverse objective functions for the OPF of the Form C{N}, C{fparm},
    C{H} and C{Cw}

    INPUT:
        **ppc** - Matpower case of the net

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
    if len(net.dcline) > 0:
        gen_is = net.gen[net.gen.in_service==True]
    else:
        gen_is = is_elems['gen']
    eg_is = is_elems['ext_grid']
    sg_is = net.sgen[(net.sgen.in_service & net.sgen.controllable) == True] \
        if "controllable" in net.sgen.columns else DataFrame()

    if gen_is.empty:
        gen_cost_per_kw = np.array([])
    elif "cost_per_kw" in gen_is.columns:
        gen_cost_per_kw = gen_is.cost_per_kw
    else:
        gen_cost_per_kw = np.ones(len(gen_is))

    if sg_is.empty:
        sgen_cost_per_kw = np.array([])
    elif "cost_per_kw" in sg_is.columns:
        sgen_cost_per_kw = sg_is.cost_per_kw
    else:
        sgen_cost_per_kw = np.ones(len(sg_is))

    if "cost_per_kw" not in eg_is.columns:
        eg_cost_per_kw = np.ones(len(eg_is))
    else:
        eg_cost_per_kw = eg_is.cost_per_kw

    if objectivetype == "linear":

        ppc["gencost"] = np.zeros((ng, 8), dtype=float)
        ppc["gencost"][:nref, :] = np.array([1, 0, 0, 2, 0, 0, 1, 1]) # initializing gencost array for eg
        ppc["gencost"][:nref, 7] = np.nan_to_num(eg_cost_per_kw*1e3)
        ppc["gencost"][nref:ng, :] = np.array([1, 0, 0, 2, 0, 0, 1, 1])  # initializing gencost array
        ppc["gencost"][nref:ng, 7] = np.nan_to_num(np.hstack([sgen_cost_per_kw*1e3,
                                                              gen_cost_per_kw*1e3]))

#        ppc["gencost"][:nref, :] = array([1, 0, 0, 2, 0, 1, 1, 1])
#        ppc["gencost"][nref:ng, :] = array([1, 0, 0, 2, 0, 1, 1, 0])  # initializing gencost array
#        ppc["gencost"][nref:ng, 5] = nan_to_num(hstack([sgen_cost_per_kw, gen_cost_per_kw]))
#        p = p_timestep/-1e3
#        p[p == 0] = 1e-6
#        ppc["gencost"][nref:ng, 6] = array(p)


    if objectivetype == "linear_minloss":

        ppc["gencost"] = np.zeros((ng, 8), dtype=float)
        ppc["gencost"][:nref, :] = np.array([1, 0, 0, 2, 0, 0, 1, 1]) # initializing gencost array for eg
        ppc["gencost"][:nref, 7] = np.nan_to_num(eg_cost_per_kw*1e3)
        ppc["gencost"][nref:ng, :] = np.array([1, 0, 0, 2, 0, 0, 1, 1])  # initializing gencost array
        ppc["gencost"][nref:ng, 7] = np.nan_to_num(np.hstack([sgen_cost_per_kw*1e3,
                                                             gen_cost_per_kw*1e3]))

#        ppc["gencost"][:nref, :] = array([1, 0, 0, 2, 0, 1, 1, 1])
#        ppc["gencost"][nref:ng, :] = array([1, 0, 0, 2, 0, 1, 1, 1])
        #ppc["gencost"][nref:ng, 5] = nan_to_num(hstack([sgen_cost_per_kw, gen_cost_per_kw]))
#        p = p_timestep/-1e3
#        p[p == 0] = 1e-6
#        ppc["gencost"][nref:ng, 6] = array(p)

        #print(ppc["gencost"][nref:ng, 6])

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
            Ybus, Yf, Yt = makeYbus(1, ppc["bus"], ppc["branch"])

        #########################
        # Additional z variables
        #########################

        # z_k = u_i - u_j
        # z_m = alpha_i - alpha_j
        # with i,j = start and endbus from lines

        # Epsilon for z constraints
        eps = 0

        # z constraints upper and lower bounds
        l = np.ones(2*nl)*eps
        u = np.ones(2*nl)*-eps

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
        Cw = np.zeros(dim, dtype=float)
        # d = 1
        d = np.ones((dim, 1), dtype=float)
        # r = 0
        r = np.zeros((dim, 1), dtype=float)
        # k = 0
        k = np.zeros((dim, 1), dtype=float)
        # m = 1
        m = np.ones((dim, 1), dtype=float)

        # Set ppc matrices
        ppc["H"] = H
        ppc["Cw"] = Cw
        ppc["N"] = N
        ppc["A"] = A
        ppc["l"] = l
        ppc["u"] = u
        ppc["fparm"] = np.hstack((d, r, k, m))

    return ppc
