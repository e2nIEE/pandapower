__author__ = "fmeier"

import numpy as np


def _pd2mpc_opf():
    """ we need to put the sgens into the gen table instead of the bsu table so we need to change 
    _pd2mpc a little to get the mpc we need for the OPF
    """
    pass


def add_voltageconstraints(net, busconstraints):
    """ Adds a column to pandapower.bus 

    * net - the actual pandapower network
    * busconstraints - nx1 array containing upper limit for bus 
        voltage, can also be 1x1 float if all limits shall be the same
    """
    pass


def add_genconstraints(net, pconstraints):
    """ Adds a column to pandapower.gen 

    * net - the actual pandapower network
    * pconstraints - nx1 array containing upper limit for generator active power  
        can also be 1x1 float if all limits shall be the same
    * qconstraints - nx1 array containing upper limit for generator reactive power  
        can also be 1x1 float if all limits shall be the same

    """
    pass


def _make_constraints(net, mpc):
    """ copying the constraint vectors to mpc 
    l <= A*[x z] <= u
    """

#    l, u dürfen keine int sein

    A = np.array([])
    l = None
    u = None

    if len(A) > 0:
        mpc["A"] = A
        mpc["l"] = l
        mpc["u"] = u
    return A, l, u


def _make_objective(mpc, objectivetype="simple"):
    """ Implementaton of diverse objective functions for the OPF of the Form C{N}, C{fparm}, C{H} and C{Cw}

    * mpc . Matpower case of the net
    * objectivetype - string with name of objective function


    ** "simple" - Quadratic costs of the form (1/2)*x'*H*x + Cw * x. x represents 
                    the voltage angles, voltage magnitude, active and reactive power values of the generators. 
                    All weighting matrices are unit matrices.
    """
    ng = len(mpc["gen"])
    nb = len(mpc["bus"])

    from pypower.idx_cost import MODEL, NCOST, PW_LINEAR, COST, POLYNOMIAL
    from scipy import sparse

    mpc["gencost"] = np.ones((ng, 5)) * 2

    if objectivetype == "simple":

        N = np.eye(2 * (nb + ng))
        N = sparse.csc_matrix(N, dtype=np.int8)
        fparm = np.array([])
        H = sparse.csc_matrix(np.zeros((2 * (nb + ng), 2 * (nb + ng))))
        Cw_va = np.zeros((nb))
        Cw_vm = np.zeros((nb))
        Cw_pg = np.ones((ng))
        Cw_qg = np.zeros((ng))

        Cw = np.concatenate((Cw_va, Cw_vm, Cw_pg, Cw_qg), axis=0)


#        mpc["gencost"]=np.ones([5])

    return N, fparm, H, Cw


"""" Citation of Pypower docs:
    The optional user parameters for user constraints (C{A, l, u}), user costs
    (C{N, fparm, H, Cw}), user variable initializer (C{z0}), and user variable
    limits (C{zl, zu}) can also be specified as fields in a case dict,
    either passed in directly or defined in a case file referenced by name.

    When specified, C{A, l, u} represent additional linear constraints on the
    optimization variables, C{l <= A*[x z] <= u}. If the user specifies an C{A}
    matrix that has more columns than the number of "C{x}" (OPF) variables,
    then there are extra linearly constrained "C{z}" variables. For an
    explanation of the formulation used and instructions for forming the
    C{A} matrix, see the MATPOWER manual.

    A generalized cost on all variables can be applied if input arguments
    C{N}, C{fparm}, C{H} and C{Cw} are specified. First, a linear transformation
    of the optimization variables is defined by means of C{r = N * [x z]}.
    Then, to each element of C{r} a function is applied as encoded in the
    C{fparm} matrix (see MATPOWER manual). If the resulting vector is named
    C{w}, then C{H} and C{Cw} define a quadratic cost on w:
    C{(1/2)*w'*H*w + Cw * w}. C{H} and C{N} should be sparse matrices and C{H}
    should also be symmetric.
"""
