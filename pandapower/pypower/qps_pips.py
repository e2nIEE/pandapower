# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Uses the Python Interior Point Solver (PIPS) to solve QP (quadratic
programming) problems.
"""

from numpy import Inf, ones, zeros, dot

from scipy.sparse import csr_matrix as sparse

from pandapower.pypower.pips import pips


def qps_pips(H, c, A, l, u, xmin=None, xmax=None, x0=None, opt=None):
    """Uses the Python Interior Point Solver (PIPS) to solve the following
    QP (quadratic programming) problem::

            min 1/2 x'*H*x + C'*x
             x

    subject to::

            l <= A*x <= u       (linear constraints)
            xmin <= x <= xmax   (variable bounds)

    Note the calling syntax is almost identical to that of QUADPROG from
    MathWorks' Optimization Toolbox. The main difference is that the linear
    constraints are specified with C{A}, C{L}, C{U} instead of C{A}, C{B},
    C{Aeq}, C{Beq}.

    Example from U{http://www.uc.edu/sashtml/iml/chap8/sect12.htm}:

        >>> from numpy import array, zeros, Inf
        >>> from scipy.sparse import csr_matrix
        >>> H = csr_matrix(array([[1003.1,  4.3,     6.3,     5.9],
        ...                       [4.3,     2.2,     2.1,     3.9],
        ...                       [6.3,     2.1,     3.5,     4.8],
        ...                       [5.9,     3.9,     4.8,     10 ]]))
        >>> c = zeros(4)
        >>> A = csr_matrix(array([[1,       1,       1,       1   ],
        ...                       [0.17,    0.11,    0.10,    0.18]]))
        >>> l = array([1, 0.10])
        >>> u = array([1, Inf])
        >>> xmin = zeros(4)
        >>> xmax = None
        >>> x0 = array([1, 0, 0, 1])
        >>> solution = qps_pips(H, c, A, l, u, xmin, xmax, x0)
        >>> round(solution["f"], 11) == 1.09666678128
        True
        >>> solution["converged"]
        True
        >>> solution["output"]["iterations"]
        10

    All parameters are optional except C{H}, C{c}, C{A} and C{l} or C{u}.
    @param H: Quadratic cost coefficients.
    @type H: csr_matrix
    @param c: vector of linear cost coefficients
    @type c: array
    @param A: Optional linear constraints.
    @type A: csr_matrix
    @param l: Optional linear constraints. Default values are M{-Inf}.
    @type l: array
    @param u: Optional linear constraints. Default values are M{Inf}.
    @type u: array
    @param xmin: Optional lower bounds on the M{x} variables, defaults are
                 M{-Inf}.
    @type xmin: array
    @param xmax: Optional upper bounds on the M{x} variables, defaults are
                 M{Inf}.
    @type xmax: array
    @param x0: Starting value of optimization vector M{x}.
    @type x0: array
    @param opt: optional options dictionary with the following keys, all of
                which are also optional (default values shown in parentheses)
                  - C{verbose} (False) - Controls level of progress output
                    displayed
                  - C{feastol} (1e-6) - termination tolerance for feasibility
                    condition
                  - C{gradtol} (1e-6) - termination tolerance for gradient
                    condition
                  - C{comptol} (1e-6) - termination tolerance for
                    complementarity condition
                  - C{costtol} (1e-6) - termination tolerance for cost
                    condition
                  - C{max_it} (150) - maximum number of iterations
                  - C{step_control} (False) - set to True to enable step-size
                    control
                  - C{max_red} (20) - maximum number of step-size reductions if
                    step-control is on
                  - C{cost_mult} (1.0) - cost multiplier used to scale the
                    objective function for improved conditioning. Note: The
                    same value must also be passed to the Hessian evaluation
                    function so that it can appropriately scale the objective
                    function term in the Hessian of the Lagrangian.
    @type opt: dict

    @rtype: dict
    @return: The solution dictionary has the following keys:
               - C{x} - solution vector
               - C{f} - final objective function value
               - C{converged} - exit status
                   - True = first order optimality conditions satisfied
                   - False = maximum number of iterations reached
                   - None = numerically failed
               - C{output} - output dictionary with keys:
                   - C{iterations} - number of iterations performed
                   - C{hist} - dictionary of arrays with trajectories of the
                     following: feascond, gradcond, coppcond, costcond, gamma,
                     stepsize, obj, alphap, alphad
                   - C{message} - exit message
               - C{lmbda} - dictionary containing the Langrange and Kuhn-Tucker
                 multipliers on the constraints, with keys:
                   - C{eqnonlin} - nonlinear equality constraints
                   - C{ineqnonlin} - nonlinear inequality constraints
                   - C{mu_l} - lower (left-hand) limit on linear constraints
                   - C{mu_u} - upper (right-hand) limit on linear constraints
                   - C{lower} - lower bound on optimization variables
                   - C{upper} - upper bound on optimization variables

    @see: L{pips}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    if isinstance(H, dict):
        p = H
    else:
        p = {'H': H, 'c': c, 'A': A, 'l': l, 'u': u}
        if xmin is not None: p['xmin'] = xmin
        if xmax is not None: p['xmax'] = xmax
        if x0 is not None: p['x0'] = x0
        if opt is not None: p['opt'] = opt

    if 'H' not in p or p['H'] == None:#p['H'].nnz == 0:
        if p['A'] is None or p['A'].nnz == 0 and \
           'xmin' not in p and \
           'xmax' not in p:
#           'xmin' not in p or len(p['xmin']) == 0 and \
#           'xmax' not in p or len(p['xmax']) == 0:
            print('qps_pips: LP problem must include constraints or variable bounds')
            return
        else:
            if p['A'] is not None and p['A'].nnz >= 0:
                nx = p['A'].shape[1]
            elif 'xmin' in p and len(p['xmin']) > 0:
                nx = p['xmin'].shape[0]
            elif 'xmax' in p and len(p['xmax']) > 0:
                nx = p['xmax'].shape[0]
        p['H'] = sparse((nx, nx))
    else:
        nx = p['H'].shape[0]

    p['xmin'] = -Inf * ones(nx) if 'xmin' not in p else p['xmin']
    p['xmax'] =  Inf * ones(nx) if 'xmax' not in p else p['xmax']

    p['c'] = zeros(nx) if p['c'] is None else p['c']

    p['x0'] = zeros(nx) if 'x0' not in p else p['x0']

    def qp_f(x, return_hessian=False):
        f = 0.5 * dot(x * p['H'], x) + dot(p['c'], x)
        df = p['H'] * x + p['c']
        if not return_hessian:
            return f, df
        d2f = p['H']
        return f, df, d2f

    p['f_fcn'] = qp_f

    sol = pips(p)

    return sol["x"], sol["f"], sol["eflag"], sol["output"], sol["lmbda"]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
