# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Solves AC optimal power flow using IPOPT.
"""

from numpy import ones, zeros, shape, Inf, pi, exp, conj, r_, arange
from numpy import flatnonzero as find

from scipy.sparse import issparse, tril, vstack, hstack, csr_matrix as sparse
from scipy.sparse import eye as speye

from pypower.idx_bus import BUS_TYPE, REF, VM, VA, MU_VMAX, MU_VMIN, LAM_P, LAM_Q
from pypower.idx_brch import F_BUS, T_BUS, RATE_A, PF, QF, PT, QT, MU_SF, MU_ST
from pypower.idx_gen import GEN_BUS, PG, QG, VG, MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN
from pypower.idx_cost import MODEL, PW_LINEAR, NCOST

from pypower.makeYbus import makeYbus
from pypower.opf_costfcn import opf_costfcn
from pypower.opf_consfcn import opf_consfcn
from pypower.opf_hessfcn import opf_hessfcn
from pypower.util import sub2ind
from pypower.ipopt_options import ipopt_options


def ipoptopf_solver(om, ppopt):
    """Solves AC optimal power flow using IPOPT.

    Inputs are an OPF model object and a PYPOWER options vector.

    Outputs are a C{results} dict, C{success} flag and C{raw} output dict.

    C{results} is a PYPOWER case dict (ppc) with the usual C{baseMVA}, C{bus}
    C{branch}, C{gen}, C{gencost} fields, along with the following additional
    fields:
        - C{order}      see 'help ext2int' for details of this field
        - C{x}          final value of optimization variables (internal order)
        - C{f}          final objective function value
        - C{mu}         shadow prices on ...
            - C{var}
                - C{l}  lower bounds on variables
                - C{u}  upper bounds on variables
            - C{nln}
                - C{l}  lower bounds on nonlinear constraints
                - C{u}  upper bounds on nonlinear constraints
            - C{lin}
                - C{l}  lower bounds on linear constraints
                - C{u}  upper bounds on linear constraints

    C{success} is C{True} if solver converged successfully, C{False} otherwise

    C{raw} is a raw output dict in form returned by MINOS
        - C{xr}     final value of optimization variables
        - C{pimul}  constraint multipliers
        - C{info}   solver specific termination code
        - C{output} solver specific output information

    @see: L{opf}, L{pips}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    """
    import pyipopt

    ## unpack data
    ppc = om.get_ppc()
    baseMVA, bus, gen, branch, gencost = \
        ppc['baseMVA'], ppc['bus'], ppc['gen'], ppc['branch'], ppc['gencost']
    vv, _, nn, _ = om.get_idx()

    ## problem dimensions
    nb = shape(bus)[0]          ## number of buses
    ng = shape(gen)[0]          ## number of gens
    nl = shape(branch)[0]       ## number of branches
    ny = om.getN('var', 'y')    ## number of piece-wise linear costs

    ## linear constraints
    A, l, u = om.linear_constraints()

    ## bounds on optimization vars
    _, xmin, xmax = om.getv()

    ## build admittance matrices
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

    ## try to select an interior initial point
    ll = xmin.copy(); uu = xmax.copy()
    ll[xmin == -Inf] = -2e19   ## replace Inf with numerical proxies
    uu[xmax ==  Inf] =  2e19
    x0 = (ll + uu) / 2
    Varefs = bus[bus[:, BUS_TYPE] == REF, VA] * (pi / 180)
    x0[vv['i1']['Va']:vv['iN']['Va']] = Varefs[0]  ## angles set to first reference angle
    if ny > 0:
        ipwl = find(gencost[:, MODEL] == PW_LINEAR)
#        PQ = r_[gen[:, PMAX], gen[:, QMAX]]
#        c = totcost(gencost[ipwl, :], PQ[ipwl])
        ## largest y-value in CCV data
        c = gencost.flatten('F')[sub2ind(shape(gencost), ipwl, NCOST + 2 * gencost[ipwl, NCOST])]
        x0[vv['i1']['y']:vv['iN']['y']] = max(c) + 0.1 * abs(max(c))
#        x0[vv['i1']['y']:vv['iN']['y']) = c + 0.1 * abs(c)

    ## find branches with flow limits
    il = find((branch[:, RATE_A] != 0) & (branch[:, RATE_A] < 1e10))
    nl2 = len(il)           ## number of constrained lines

    ##-----  run opf  -----
    ## build Jacobian and Hessian structure
    if A is not None and issparse(A):
        nA = A.shape[0]                ## number of original linear constraints
    else:
        nA = 0
    nx = len(x0)
    f = branch[:, F_BUS]                           ## list of "from" buses
    t = branch[:, T_BUS]                           ## list of "to" buses
    Cf = sparse((ones(nl), (arange(nl), f)), (nl, nb))      ## connection matrix for line & from buses
    Ct = sparse((ones(nl), (arange(nl), t)), (nl, nb))      ## connection matrix for line & to buses
    Cl = Cf + Ct
    Cb = Cl.T * Cl + speye(nb, nb)
    Cl2 = Cl[il, :]
    Cg = sparse((ones(ng), (gen[:, GEN_BUS], arange(ng))), (nb, ng))
    nz = nx - 2 * (nb + ng)
    nxtra = nx - 2 * nb
    if nz > 0:
        Js = vstack([
            hstack([Cb,      Cb,      Cg,              sparse((nb, ng)),   sparse((nb,  nz))]),
            hstack([Cb,      Cb,      sparse((nb, ng)),   Cg,              sparse((nb,  nz))]),
            hstack([Cl2,     Cl2,     sparse((nl2, 2 * ng)),               sparse((nl2, nz))]),
            hstack([Cl2,     Cl2,     sparse((nl2, 2 * ng)),               sparse((nl2, nz))])
        ], 'coo')
    else:
        Js = vstack([
            hstack([Cb,      Cb,      Cg,              sparse((nb, ng))]),
            hstack([Cb,      Cb,      sparse((nb, ng)),   Cg,          ]),
            hstack([Cl2,     Cl2,     sparse((nl2, 2 * ng)),           ]),
            hstack([Cl2,     Cl2,     sparse((nl2, 2 * ng)),           ])
        ], 'coo')

    if A is not None and issparse(A):
        Js = vstack([Js, A], 'coo')

    f, _, d2f = opf_costfcn(x0, om, True)
    Hs = tril(d2f + vstack([
        hstack([Cb,  Cb,  sparse((nb, nxtra))]),
        hstack([Cb,  Cb,  sparse((nb, nxtra))]),
        sparse((nxtra, nx))
    ]), format='coo')

    ## set options struct for IPOPT
#    options = {}
#    options['ipopt'] = ipopt_options([], ppopt)

    ## extra data to pass to functions
    userdata = {
        'om':       om,
        'Ybus':     Ybus,
        'Yf':       Yf[il, :],
        'Yt':       Yt[il, :],
        'ppopt':    ppopt,
        'il':       il,
        'A':        A,
        'nA':       nA,
        'neqnln':   2 * nb,
        'niqnln':   2 * nl2,
        'Js':       Js,
        'Hs':       Hs
    }

    ## check Jacobian and Hessian structure
    #xr                  = rand(x0.shape)
    #lmbda               = rand( 2 * nb + 2 * nl2)
    #Js1 = eval_jac_g(x, flag, userdata) #(xr, options.auxdata)
    #Hs1  = eval_h(xr, 1, lmbda, userdata)
    #i1, j1, s = find(Js)
    #i2, j2, s = find(Js1)
    #if (len(i1) != len(i2)) | (norm(i1 - i2) != 0) | (norm(j1 - j2) != 0):
    #    raise ValueError, 'something''s wrong with the Jacobian structure'
    #
    #i1, j1, s = find(Hs)
    #i2, j2, s = find(Hs1)
    #if (len(i1) != len(i2)) | (norm(i1 - i2) != 0) | (norm(j1 - j2) != 0):
    #    raise ValueError, 'something''s wrong with the Hessian structure'

    ## define variable and constraint bounds
    # n is the number of variables
    n = x0.shape[0]
    # xl is the lower bound of x as bounded constraints
    xl = xmin
    # xu is the upper bound of x as bounded constraints
    xu = xmax

    neqnln = 2 * nb
    niqnln = 2 * nl2

    # number of constraints
    m = neqnln + niqnln + nA
    # lower bound of constraint
    gl = r_[zeros(neqnln), -Inf * ones(niqnln), l]
    # upper bound of constraints
    gu = r_[zeros(neqnln),       zeros(niqnln), u]

    # number of nonzeros in Jacobi matrix
    nnzj = Js.nnz
    # number of non-zeros in Hessian matrix, you can set it to 0
    nnzh = Hs.nnz

    eval_hessian = True
    if eval_hessian:
        hessian = lambda x, lagrange, obj_factor, flag, user_data=None: \
                eval_h(x, lagrange, obj_factor, flag, userdata)

        nlp = pyipopt.create(n, xl, xu, m, gl, gu, nnzj, nnzh,
                             eval_f, eval_grad_f, eval_g, eval_jac_g, hessian)
    else:
        nnzh = 0
        nlp = pyipopt.create(n, xl, xu, m, gl, gu, nnzj, nnzh,
                             eval_f, eval_grad_f, eval_g, eval_jac_g)

    nlp.int_option('print_level', 5)
    nlp.num_option('tol', 1.0000e-12)
    nlp.int_option('max_iter', 250)
    nlp.num_option('dual_inf_tol', 0.10000)
    nlp.num_option('constr_viol_tol', 1.0000e-06)
    nlp.num_option('compl_inf_tol', 1.0000e-05)
    nlp.num_option('acceptable_tol', 1.0000e-08)
    nlp.num_option('acceptable_constr_viol_tol', 1.0000e-04)
    nlp.num_option('acceptable_compl_inf_tol', 0.0010000)
    nlp.str_option('mu_strategy', 'adaptive')

    iter = 0
    def intermediate_callback(algmod, iter_count, obj_value, inf_pr, inf_du,
            mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials,
            user_data=None):
        iter = iter_count
        return True

    nlp.set_intermediate_callback(intermediate_callback)

    ## run the optimization
    # returns final solution x, upper and lower bound for multiplier, final
    # objective function obj and the return status of ipopt
    x, zl, zu, obj, status, zg = nlp.solve(x0, m, userdata)

    info = {'x': x, 'zl': zl, 'zu': zu, 'obj': obj, 'status': status, 'lmbda': zg}

    nlp.close()

    success = (status == 0) | (status == 1)

    output = {'iterations': iter}

    f, _ = opf_costfcn(x, om)

    ## update solution data
    Va = x[vv['i1']['Va']:vv['iN']['Va']]
    Vm = x[vv['i1']['Vm']:vv['iN']['Vm']]
    Pg = x[vv['i1']['Pg']:vv['iN']['Pg']]
    Qg = x[vv['i1']['Qg']:vv['iN']['Qg']]
    V = Vm * exp(1j * Va)

    ##-----  calculate return values  -----
    ## update voltages & generator outputs
    bus[:, VA] = Va * 180 / pi
    bus[:, VM] = Vm
    gen[:, PG] = Pg * baseMVA
    gen[:, QG] = Qg * baseMVA
    gen[:, VG] = Vm[gen[:, GEN_BUS].astype(int)]

    ## compute branch flows
    f_br = branch[:, F_BUS].astype(int)
    t_br = branch[:, T_BUS].astype(int)
    Sf = V[f_br] * conj(Yf * V)  ## cplx pwr at "from" bus, p.u.
    St = V[t_br] * conj(Yt * V)  ## cplx pwr at "to" bus, p.u.
    branch[:, PF] = Sf.real * baseMVA
    branch[:, QF] = Sf.imag * baseMVA
    branch[:, PT] = St.real * baseMVA
    branch[:, QT] = St.imag * baseMVA

    ## line constraint is actually on square of limit
    ## so we must fix multipliers
    muSf = zeros(nl)
    muSt = zeros(nl)
    if len(il) > 0:
        muSf[il] = 2 * info['lmbda'][2 * nb +       arange(nl2)] * branch[il, RATE_A] / baseMVA
        muSt[il] = 2 * info['lmbda'][2 * nb + nl2 + arange(nl2)] * branch[il, RATE_A] / baseMVA

    ## update Lagrange multipliers
    bus[:, MU_VMAX]  = info['zu'][vv['i1']['Vm']:vv['iN']['Vm']]
    bus[:, MU_VMIN]  = info['zl'][vv['i1']['Vm']:vv['iN']['Vm']]
    gen[:, MU_PMAX]  = info['zu'][vv['i1']['Pg']:vv['iN']['Pg']] / baseMVA
    gen[:, MU_PMIN]  = info['zl'][vv['i1']['Pg']:vv['iN']['Pg']] / baseMVA
    gen[:, MU_QMAX]  = info['zu'][vv['i1']['Qg']:vv['iN']['Qg']] / baseMVA
    gen[:, MU_QMIN]  = info['zl'][vv['i1']['Qg']:vv['iN']['Qg']] / baseMVA
    bus[:, LAM_P]    = info['lmbda'][nn['i1']['Pmis']:nn['iN']['Pmis']] / baseMVA
    bus[:, LAM_Q]    = info['lmbda'][nn['i1']['Qmis']:nn['iN']['Qmis']] / baseMVA
    branch[:, MU_SF] = muSf / baseMVA
    branch[:, MU_ST] = muSt / baseMVA

    ## package up results
    nlnN = om.getN('nln')

    ## extract multipliers for nonlinear constraints
    kl = find(info['lmbda'][:2 * nb] < 0)
    ku = find(info['lmbda'][:2 * nb] > 0)
    nl_mu_l = zeros(nlnN)
    nl_mu_u = r_[zeros(2 * nb), muSf, muSt]
    nl_mu_l[kl] = -info['lmbda'][kl]
    nl_mu_u[ku] =  info['lmbda'][ku]

    ## extract multipliers for linear constraints
    lam_lin = info['lmbda'][2 * nb + 2 * nl2 + arange(nA)]   ## lmbda for linear constraints
    kl = find(lam_lin < 0)                     ## lower bound binding
    ku = find(lam_lin > 0)                     ## upper bound binding
    mu_l = zeros(nA)
    mu_l[kl] = -lam_lin[kl]
    mu_u = zeros(nA)
    mu_u[ku] = lam_lin[ku]

    mu = {
      'var': {'l': info['zl'], 'u': info['zu']},
      'nln': {'l': nl_mu_l, 'u': nl_mu_u}, \
      'lin': {'l': mu_l, 'u': mu_u}
    }

    results = ppc
    results['bus'], results['branch'], results['gen'], \
        results['om'], results['x'], results['mu'], results['f'] = \
            bus, branch, gen, om, x, mu, f

    pimul = r_[
        results['mu']['nln']['l'] - results['mu']['nln']['u'],
        results['mu']['lin']['l'] - results['mu']['lin']['u'],
        -ones(ny > 0),
        results['mu']['var']['l'] - results['mu']['var']['u']
    ]
    raw = {'xr': x, 'pimul': pimul, 'info': info['status'], 'output': output}

    return results, success, raw


def eval_f(x, user_data=None):
    """Calculates the objective value.

    @param x: input vector
    """
    om = user_data['om']
    f,  _ = opf_costfcn(x, om)
    return f


def eval_grad_f(x, user_data=None):
    """Calculates gradient for objective function.
    """
    om = user_data['om']
    _, df = opf_costfcn(x, om)
    return df


def eval_g(x, user_data=None):
    """Calculates the constraint values and returns an array.
    """
    om    = user_data['om']
    Ybus  = user_data['Ybus']
    Yf    = user_data['Yf']
    Yt    = user_data['Yt']
    ppopt = user_data['ppopt']
    il    = user_data['il']
    A     = user_data['A']

    hn, gn, _, _ = opf_consfcn(x, om, Ybus, Yf, Yt, ppopt, il)

    if A is not None and issparse(A):
        c = r_[gn, hn, A * x]
    else:
        c = r_[gn, hn]
    return c


def eval_jac_g(x, flag, user_data=None):
    """Calculates the Jacobi matrix.

    If the flag is true, returns a tuple (row, col) to indicate the
    sparse Jacobi matrix's structure.
    If the flag is false, returns the values of the Jacobi matrix
    with length nnzj.
    """
    Js = user_data['Js']
    if flag:
        return (Js.row, Js.col)
    else:
        om    = user_data['om']
        Ybus  = user_data['Ybus']
        Yf    = user_data['Yf']
        Yt    = user_data['Yt']
        ppopt = user_data['ppopt']
        il    = user_data['il']
        A     = user_data['A']

        _, _, dhn, dgn = opf_consfcn(x, om, Ybus, Yf, Yt, ppopt, il)

        if A is not None and issparse(A):
            J = vstack([dgn.T, dhn.T, A], 'coo')
        else:
            J = vstack([dgn.T, dhn.T], 'coo')

        ## FIXME: Extend PyIPOPT to handle changes in sparsity structure
        nnzj = Js.nnz
        Jd = zeros(nnzj)
        Jc = J.tocsc()
        for i in range(nnzj):
            Jd[i] = Jc[Js.row[i], Js.col[i]]

        return Jd


def eval_h(x, lagrange, obj_factor, flag, user_data=None):
    """Calculates the Hessian matrix (optional).

    If omitted, set nnzh to 0 and Ipopt will use approximated Hessian
    which will make the convergence slower.
    """
    Hs = user_data['Hs']
    if flag:
        return (Hs.row, Hs.col)
    else:
        neqnln = user_data['neqnln']
        niqnln = user_data['niqnln']
        om     = user_data['om']
        Ybus   = user_data['Ybus']
        Yf     = user_data['Yf']
        Yt     = user_data['Yt']
        ppopt  = user_data['ppopt']
        il     = user_data['il']

        lam = {}
        lam['eqnonlin']   = lagrange[:neqnln]
        lam['ineqnonlin'] = lagrange[arange(niqnln) + neqnln]

        H = opf_hessfcn(x, lam, om, Ybus, Yf, Yt, ppopt, il, obj_factor)

        Hl = tril(H, format='csc')

        ## FIXME: Extend PyIPOPT to handle changes in sparsity structure
        nnzh = Hs.nnz
        Hd = zeros(nnzh)
        for i in range(nnzh):
            Hd[i] = Hl[Hs.row[i], Hs.col[i]]

        return Hd
