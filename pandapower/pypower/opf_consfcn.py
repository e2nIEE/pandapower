# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Evaluates nonlinear constraints and their Jacobian for OPF.
"""

from numpy import zeros, ones, conj, exp, r_, Inf, arange

from scipy.sparse import lil_matrix, vstack, hstack, csr_matrix as sparse

from pypower.idx_gen import GEN_BUS, PG, QG
from pypower.idx_brch import F_BUS, T_BUS, RATE_A

from pypower.makeSbus import makeSbus
from pypower.dSbus_dV import dSbus_dV
from pypower.dIbr_dV import dIbr_dV
from pypower.dSbr_dV import dSbr_dV
from pypower.dAbr_dV import dAbr_dV


def opf_consfcn(x, om, Ybus, Yf, Yt, ppopt, il=None, *args):
    """Evaluates nonlinear constraints and their Jacobian for OPF.

    Constraint evaluation function for AC optimal power flow, suitable
    for use with L{pips}. Computes constraint vectors and their gradients.

    @param x: optimization vector
    @param om: OPF model object
    @param Ybus: bus admittance matrix
    @param Yf: admittance matrix for "from" end of constrained branches
    @param Yt: admittance matrix for "to" end of constrained branches
    @param ppopt: PYPOWER options vector
    @param il: (optional) vector of branch indices corresponding to
    branches with flow limits (all others are assumed to be
    unconstrained). The default is C{range(nl)} (all branches).
    C{Yf} and C{Yt} contain only the rows corresponding to C{il}.

    @return: C{h} - vector of inequality constraint values (flow limits)
    limit^2 - flow^2, where the flow can be apparent power real power or
    current, depending on value of C{OPF_FLOW_LIM} in C{ppopt} (only for
    constrained lines). C{g} - vector of equality constraint values (power
    balances). C{dh} - (optional) inequality constraint gradients, column
    j is gradient of h(j). C{dg} - (optional) equality constraint gradients.

    @see: L{opf_costfcn}, L{opf_hessfcn}

    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Ray Zimmerman (PSERC Cornell)
    """
    ##----- initialize -----

    ## unpack data
    ppc = om.get_ppc()
    baseMVA, bus, gen, branch = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]
    vv, _, _, _ = om.get_idx()

    ## problem dimensions
    nb = bus.shape[0]          ## number of buses
    nl = branch.shape[0]       ## number of branches
    ng = gen.shape[0]          ## number of dispatchable injections
    nxyz = len(x)              ## total number of control vars of all types

    ## set default constrained lines
    if il is None:
        il = arange(nl)         ## all lines have limits by default
    nl2 = len(il)              ## number of constrained lines

    ## grab Pg & Qg
    Pg = x[vv["i1"]["Pg"]:vv["iN"]["Pg"]]  ## active generation in p.u.
    Qg = x[vv["i1"]["Qg"]:vv["iN"]["Qg"]]  ## reactive generation in p.u.

    ## put Pg & Qg back in gen
    gen[:, PG] = Pg * baseMVA  ## active generation in MW
    gen[:, QG] = Qg * baseMVA  ## reactive generation in MVAr

    ## rebuild Sbus
    Sbus = makeSbus(baseMVA, bus, gen) ## net injected power in p.u.

    ## ----- evaluate constraints -----
    ## reconstruct V
    Va = x[vv["i1"]["Va"]:vv["iN"]["Va"]]
    Vm = x[vv["i1"]["Vm"]:vv["iN"]["Vm"]]
    V = Vm * exp(1j * Va)

    ## evaluate power flow equations
    mis = V * conj(Ybus * V) - Sbus

    ##----- evaluate constraint function values -----
    ## first, the equality constraints (power flow)
    g = r_[ mis.real,            ## active power mismatch for all buses
            mis.imag ]           ## reactive power mismatch for all buses

    ## then, the inequality constraints (branch flow limits)
    if nl2 > 0:
        flow_max = (branch[il, RATE_A] / baseMVA)**2
        flow_max[flow_max == 0] = Inf
        if ppopt['OPF_FLOW_LIM'] == 2:       ## current magnitude limit, |I|
            If = Yf * V
            It = Yt * V
            h = r_[ If * conj(If) - flow_max,     ## branch I limits (from bus)
                    It * conj(It) - flow_max ].real    ## branch I limits (to bus)
        else:
            ## compute branch power flows
            ## complex power injected at "from" bus (p.u.)
            Sf = V[ branch[il, F_BUS].astype(int) ] * conj(Yf * V)
            ## complex power injected at "to" bus (p.u.)
            St = V[ branch[il, T_BUS].astype(int) ] * conj(Yt * V)
            if ppopt['OPF_FLOW_LIM'] == 1:   ## active power limit, P (Pan Wei)
                h = r_[ Sf.real**2 - flow_max,   ## branch P limits (from bus)
                        St.real**2 - flow_max ]  ## branch P limits (to bus)
            else:                ## apparent power limit, |S|
                h = r_[ Sf * conj(Sf) - flow_max, ## branch S limits (from bus)
                        St * conj(St) - flow_max ].real  ## branch S limits (to bus)
    else:
        h = zeros((0,1))

    ##----- evaluate partials of constraints -----
    ## index ranges
    iVa = arange(vv["i1"]["Va"], vv["iN"]["Va"])
    iVm = arange(vv["i1"]["Vm"], vv["iN"]["Vm"])
    iPg = arange(vv["i1"]["Pg"], vv["iN"]["Pg"])
    iQg = arange(vv["i1"]["Qg"], vv["iN"]["Qg"])
    iVaVmPgQg = r_[iVa, iVm, iPg, iQg].T

    ## compute partials of injected bus powers
    dSbus_dVm, dSbus_dVa = dSbus_dV(Ybus, V)           ## w.r.t. V
    ## Pbus w.r.t. Pg, Qbus w.r.t. Qg
    neg_Cg = sparse((-ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))

    ## construct Jacobian of equality constraints (power flow) and transpose it
    dg = lil_matrix((2 * nb, nxyz))
    blank = sparse((nb, ng))
    dg[:, iVaVmPgQg] = vstack([
            ## P mismatch w.r.t Va, Vm, Pg, Qg
            hstack([dSbus_dVa.real, dSbus_dVm.real, neg_Cg, blank]),
            ## Q mismatch w.r.t Va, Vm, Pg, Qg
            hstack([dSbus_dVa.imag, dSbus_dVm.imag, blank, neg_Cg])
        ], "csr")
    dg = dg.T

    if nl2 > 0:
        ## compute partials of Flows w.r.t. V
        if ppopt['OPF_FLOW_LIM'] == 2:     ## current
            dFf_dVa, dFf_dVm, dFt_dVa, dFt_dVm, Ff, Ft = \
                    dIbr_dV(branch[il, :], Yf, Yt, V)
        else:                  ## power
            dFf_dVa, dFf_dVm, dFt_dVa, dFt_dVm, Ff, Ft = \
                    dSbr_dV(branch[il, :], Yf, Yt, V)
        if ppopt['OPF_FLOW_LIM'] == 1:     ## real part of flow (active power)
            dFf_dVa = dFf_dVa.real
            dFf_dVm = dFf_dVm.real
            dFt_dVa = dFt_dVa.real
            dFt_dVm = dFt_dVm.real
            Ff = Ff.real
            Ft = Ft.real

        ## squared magnitude of flow (of complex power or current, or real power)
        df_dVa, df_dVm, dt_dVa, dt_dVm = \
                dAbr_dV(dFf_dVa, dFf_dVm, dFt_dVa, dFt_dVm, Ff, Ft)

        ## construct Jacobian of inequality constraints (branch limits)
        ## and transpose it.
        dh = lil_matrix((2 * nl2, nxyz))
        dh[:, r_[iVa, iVm].T] = vstack([
                hstack([df_dVa, df_dVm]),    ## "from" flow limit
                hstack([dt_dVa, dt_dVm])     ## "to" flow limit
            ], "csr")
        dh = dh.T
    else:
        dh = None

    return h, g, dh, dg
