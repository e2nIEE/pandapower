# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Used to set and retrieve a PYPOWER options vector.
"""

from numpy import Inf


PF_OPTIONS = [
    ('pf_alg', 1, '''power flow algorithm:
1 - Newton's method,
2 - Fast-Decoupled (XB version),
3 - Fast-Decoupled (BX version),
4 - Gauss Seidel'''),

    ('pf_tol', 1e-8, 'termination tolerance on per unit P & Q mismatch'),

    ('pf_max_it', 10, 'maximum number of iterations for Newton\'s method'),

    ('pf_max_it_fd', 30, 'maximum number of iterations for fast '
     'decoupled method'),

    ('pf_max_it_gs', 1000, 'maximum number of iterations for '
     'Gauss-Seidel method'),

    ('enforce_q_lims', False, 'enforce gen reactive power limits, at '
     'expense of |V|'),

    ('pf_dc', False, '''use DC power flow formulation, for power flow and OPF:
False - use AC formulation & corresponding algorithm opts,
True  - use DC formulation, ignore AC algorithm options''')
]

OPF_OPTIONS = [
    ('opf_alg', 0, '''algorithm to use for OPF:
0 - choose best default solver available in the
following order, 500, 540, 520 then 100/200
Otherwise the first digit specifies the problem
formulation and the second specifies the solver,
as follows, (see the User's Manual for more details)
500 - generalized formulation, MINOS,
540 - generalized formulation, MIPS
primal/dual interior point method,
545 - generalized formulation (except CCV), SC-MIPS
step-controlled primal/dual interior point method'''),

#    ('opf_poly2pwl_pts', 10, 'number of evaluation points to use when '
#     'converting from polynomial to piece-wise linear costs)'),

    ('opf_violation', 5e-6, 'constraint violation tolerance'),

    ('opf_flow_lim', 0, '''qty to limit for branch flow constraints:
0 - apparent power flow (limit in MVA),
1 - active power flow (limit in MW),
2 - current magnitude (limit in MVA at 1 p.u. voltage'''),

    ('opf_ignore_ang_lim', False, 'ignore angle difference limits for '
     'branches even if specified'),

    ('opf_alg_dc', 0, '''solver to use for DC OPF:
0 - choose default solver based on availability in the
following order, 600, 500, 200.
200 - PIPS, Python Interior Point Solver
primal/dual interior point method,
250 - PIPS-sc, step-controlled variant of PIPS
400 - IPOPT, requires pyipopt interface to IPOPT solver
available from: https://projects.coin-or.org/Ipopt/
500 - CPLEX, requires Python interface to CPLEX solver
600 - MOSEK, requires Python interface to MOSEK solver
available from: http://www.mosek.com/
700 - GUROBI, requires Python interface to Gurobi optimizer
available from: http://www.gurobi.com/''')
]

OUTPUT_OPTIONS = [
    ('verbose', 1, '''amount of progress info printed:
0 - print no progress info,
1 - print a little progress info,
2 - print a lot of progress info,
3 - print all progress info'''),

    ('out_all', -1, '''controls printing of results:
-1 - individual flags control what prints,
0 - don't print anything
    (overrides individual flags),
1 - print everything
    (overrides individual flags)'''),

    ('out_sys_sum', True, 'print system summary'),

    ('out_area_sum', False, 'print area summaries'),

    ('out_bus', True, 'print bus detail'),

    ('out_branch', True, 'print branch detail'),

    ('out_gen', False, '''print generator detail
(OUT_BUS also includes gen info)'''),

    ('out_all_lim', -1, '''control constraint info output:
-1 - individual flags control what constraint info prints,
0 - no constraint info (overrides individual flags),
1 - binding constraint info (overrides individual flags),
2 - all constraint info (overrides individual flags)'''),

    ('out_v_lim', 1, '''control output of voltage limit info:
0 - don't print,
1 - print binding constraints only,
2 - print all constraints
(same options for OUT_LINE_LIM, OUT_PG_LIM, OUT_QG_LIM)'''),

    ('out_line_lim', 1, 'control output of line limit info'),

    ('out_pg_lim', 1, 'control output of gen P limit info'),

    ('out_qg_lim', 1, 'control output of gen Q limit info'),

#    ('out_raw', False, 'print raw data'),

    ('return_raw_der', 0, '''return constraint and derivative info
in results['raw'] (in keys g, dg, df, d2f))''')
]

PDIPM_OPTIONS = [
    ('pdipm_feastol', 0, '''feasibility (equality) tolerance
for Primal-Dual Interior Points Methods, set
to value of OPF_VIOLATION by default'''),
    ('pdipm_gradtol', 1e-6, '''gradient tolerance for
Primal-Dual Interior Points Methods'''),
    ('pdipm_comptol', 1e-6, '''complementary condition (inequality)
tolerance for Primal-Dual Interior Points Methods'''),
    ('pdipm_costtol', 1e-6, '''optimality tolerance for
Primal-Dual Interior Points Methods'''),
    ('pdipm_max_it',  150, '''maximum number of iterations for
Primal-Dual Interior Points Methods'''),
    ('scpdipm_red_it', 20, '''maximum number of reductions per iteration
for Step-Control Primal-Dual Interior Points Methods''')
]

GUROBI_OPTIONS = [
    ('grb_method', 1, '''solution algorithm (Method)
0 - primal simplex
1 - dual simplex
2 - barrier
3 - concurrent (LP only)
4 - deterministic concurrent (LP only)
'''),
    ('grb_timelimit', Inf, 'maximum time allowed for solver (TimeLimit)'),
('grb_threads', 0, '(auto) maximum number of threads to use (Threads)'),
('grb_opt', 0, 'See gurobi_options() for details')
]


def ppoption(ppopt=None, **kw_args):
    """Used to set and retrieve a PYPOWER options vector.

    C{opt = ppoption()} returns the default options vector

    C{opt = ppoption(NAME1=VALUE1, NAME2=VALUE2, ...)} returns the default
    options vector with new values for the specified options, NAME# is the
    name of an option, and VALUE# is the new value.

    C{opt = ppoption(OPT, NAME1=VALUE1, NAME2=VALUE2, ...)} same as above
    except it uses the options vector OPT as a base instead of the default
    options vector.

    Examples::
        opt = ppoption(PF_ALG=2, PF_TOL=1e-4);
        opt = ppoption(opt, OPF_ALG=565, VERBOSE=2)

    @author: Ray Zimmerman (PSERC Cornell)
    """

    default_ppopt = {}

    options = PF_OPTIONS + OPF_OPTIONS + OUTPUT_OPTIONS + PDIPM_OPTIONS

    for name, default, _ in options:
        default_ppopt[name.upper()] = default

    ppopt = default_ppopt if ppopt == None else ppopt.copy()

    ppopt.update(kw_args)

    return ppopt
