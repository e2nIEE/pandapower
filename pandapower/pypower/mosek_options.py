# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Sets options for MOSEK.
"""

try:
    from pymosek import mosekopt
except ImportError:
#    print "MOSEK not available"
    pass

from pandapower.pypower._compat import PY2
from pandapower.pypower.util import feval


if not PY2:
    basestring = str


def mosek_options(overrides=None, ppopt=None):
    """Sets options for MOSEK.

    Inputs are all optional, second argument must be either a string
    (C{fname}) or a dict (C{ppopt}):

        - C{overrides}
            - dict containing values to override the defaults
            - C{fname} name of user-supplied function called after default
            options are set to modify them. Calling syntax is::
                modified_opt = fname(default_opt)
        - C{ppopt} PYPOWER options vector, uses the following entries:
            - C{OPF_VIOLATION} used to set opt.MSK_DPAR_INTPNT_TOL_PFEAS
            - C{VERBOSE} not currently used here
            - C{MOSEK_LP_ALG} - used to set opt.MSK_IPAR_OPTIMIZER
            - C{MOSEK_MAX_IT} used to set opt.MSK_IPAR_INTPNT_MAX_ITERATIONS
            - C{MOSEK_GAP_TOL} used to set opt.MSK_DPAR_INTPNT_TOL_REL_GAP
            - C{MOSEK_MAX_TIME} used to set opt.MSK_DPAR_OPTIMIZER_MAX_TIME
            - C{MOSEK_NUM_THREADS} used to set opt.MSK_IPAR_INTPNT_NUM_THREADS
            - C{MOSEK_OPT} user option file, if ppopt['MOSEK_OPT'] is non-zero
            it is appended to 'mosek_user_options_' to form
            the name of a user-supplied function used as C{fname}
            described above, except with calling syntax::
                modified_opt = fname(default_opt, ppopt)

    Output is a param dict to pass to MOSEKOPT.

    Example:

    If PPOPT['MOSEK_OPT'] = 3, then after setting the default MOSEK options,
    L{mosek_options} will execute the following user-defined function
    to allow option overrides::

        opt = mosek_user_options_3(opt, ppopt)

    The contents of mosek_user_options_3.py, could be something like::

        def mosek_user_options_3(opt, ppopt):
            opt = {}
            opt.MSK_DPAR_INTPNT_TOL_DFEAS   = 1e-9
            opt.MSK_IPAR_SIM_MAX_ITERATIONS = 5000000
            return opt

    See the Parameters reference in Appix E of "The MOSEK
    optimization toolbox for MATLAB manaul" for
    details on the available options.

    U{http://www.mosek.com/documentation/}

    @see: C{mosekopt}, L{ppoption}.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ##-----  initialization and arg handling  -----
    ## defaults
    verbose = 2
    gaptol  = 0
    fname   = ''

    ## get symbolic constant names
    r, res = mosekopt('symbcon echo(0)')
    sc = res['symbcon']

    ## second argument
    if ppopt == None:
        if isinstance(ppopt, basestring):        ## 2nd arg is FNAME (string)
            fname = ppopt
            have_ppopt = False
        else:                    ## 2nd arg is ppopt (MATPOWER options vector)
            have_ppopt = True
            verbose = ppopt['VERBOSE']
            if ppopt['MOSEK_OPT']:
                fname = 'mosek_user_options_#d' # ppopt['MOSEK_OPT']
    else:
        have_ppopt = False

    opt = {}
    ##-----  set default options for MOSEK  -----
    ## solution algorithm
    if have_ppopt:
        alg = ppopt['MOSEK_LP_ALG']
        if alg == sc['MSK_OPTIMIZER_FREE'] or \
            alg == sc['MSK_OPTIMIZER_INTPNT'] or \
            alg == sc['MSK_OPTIMIZER_PRIMAL_SIMPLEX'] or \
            alg == sc['MSK_OPTIMIZER_DUAL_SIMPLEX'] or \
            alg == sc['MSK_OPTIMIZER_PRIMAL_DUAL_SIMPLEX'] or \
            alg == sc['MSK_OPTIMIZER_FREE_SIMPLEX'] or \
            alg == sc['MSK_OPTIMIZER_CONCURRENT']:
                opt['MSK_IPAR_OPTIMIZER'] = alg
        else:
            opt['MSK_IPAR_OPTIMIZER'] = sc['MSK_OPTIMIZER_FREE'];

        ## (make default OPF_VIOLATION correspond to default MSK_DPAR_INTPNT_TOL_PFEAS)
        opt['MSK_DPAR_INTPNT_TOL_PFEAS'] = ppopt['OPF_VIOLATION'] / 500
        if ppopt['MOSEK_MAX_IT']:
            opt['MSK_IPAR_INTPNT_MAX_ITERATIONS'] = ppopt['MOSEK_MAX_IT']

        if ppopt['MOSEK_GAP_TOL']:
            opt['MSK_DPAR_INTPNT_TOL_REL_GAP'] = ppopt['MOSEK_GAP_TOL']

        if ppopt['MOSEK_MAX_TIME']:
            opt['MSK_DPAR_OPTIMIZER_MAX_TIME'] = ppopt['MOSEK_MAX_TIME']

        if ppopt['MOSEK_NUM_THREADS']:
            opt['MSK_IPAR_INTPNT_NUM_THREADS'] = ppopt['MOSEK_NUM_THREADS']
    else:
        opt['MSK_IPAR_OPTIMIZER'] = sc['MSK_OPTIMIZER_FREE']

    # opt['MSK_DPAR_INTPNT_TOL_PFEAS'] = 1e-8       ## primal feasibility tol
    # opt['MSK_DPAR_INTPNT_TOL_DFEAS'] = 1e-8       ## dual feasibility tol
    # opt['MSK_DPAR_INTPNT_TOL_MU_RED'] = 1e-16     ## relative complementarity gap tol
    # opt['MSK_DPAR_INTPNT_TOL_REL_GAP'] = 1e-8     ## relative gap termination tol
    # opt['MSK_IPAR_INTPNT_MAX_ITERATIONS'] = 400   ## max iterations for int point
    # opt['MSK_IPAR_SIM_MAX_ITERATIONS'] = 10000000 ## max iterations for simplex
    # opt['MSK_DPAR_OPTIMIZER_MAX_TIME'] = -1       ## max time allowed (< 0 --> Inf)
    # opt['MSK_IPAR_INTPNT_NUM_THREADS'] = 1        ## number of threads
    # opt['MSK_IPAR_PRESOLVE_USE'] = sc['MSK_PRESOLVE_MODE_OFF']

    # if verbose == 0:
    #     opt['MSK_IPAR_LOG'] = 0
    #

    ##-----  call user function to modify defaults  -----
    if len(fname) > 0:
        if have_ppopt:
            opt = feval(fname, opt, ppopt)
        else:
            opt = feval(fname, opt)

    ##-----  apply overrides  -----
    if overrides is not None:
        names = overrides.keys()
        for k in range(len(names)):
            opt[names[k]] = overrides[names[k]]
