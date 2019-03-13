# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from numpy import Inf
from pandapower.pypower._compat import PY2
from pandapower.pypower.util import feval


if not PY2:
    basestring = str


def gurobi_options(overrides=None, ppopt=None):
    """Sets options for GUROBI.

    Sets the values for the options dict normally passed to GUROBI_MEX.

    Inputs are all optional, second argument must be either a string
    (fname) or a vector (ppopt):

        overrides - dict containing values to override the defaults
        fname - name of user-supplied function called after default
            options are set to modify them. Calling syntax is:
                modified_opt = fname(default_opt)
        ppopt - PYPOWER options vector, uses the following entries:
            OPF_VIOLATION (16)  - used to set opt.FeasibilityTol
            VERBOSE (31)        - used to set opt.DisplayInterval, opt.Display
            GRB_METHOD (121)    - used to set opt.Method
            GRB_TIMELIMIT (122) - used to set opt.TimeLimit (seconds)
            GRB_THREADS (123)   - used to set opt.Threads
            GRB_OPT (124)       - user option file, if PPOPT(124) is non-zero
                it is appended to 'gurobi_user_options_' to form the name of a
                user-supplied function used as C{fname} described above, except
                with calling syntax:
                    modified_opt = fname(default_opt, mpopt)

    Output is an options struct to pass to GUROBI_MEX.

    Example:

    If ppopt['GRB_OPT'] = 3, then after setting the default GUROBI options,
    GUROBI_OPTIONS will execute the following user-defined function
    to allow option overrides:

        opt = gurobi_user_options_3(opt, ppopt)

    The contents of gurobi_user_options_3.py, could be something like:

        def gurobi_user_options_3(opt, ppopt):
            opt = {}
            opt['OptimalityTol']   = 1e-9
            opt['IterationLimit']  = 3000
            opt['BarIterLimit']    = 200
            opt['Crossover']       = 0
            opt['Presolve']        = 0
            return opt

    For details on the available options, see the "Parameters" section
    of the "Gurobi Optimizer Reference Manual" at:

        http://www.gurobi.com/doc/45/refman/

    @see: L{gurobi_mex}, L{ppoption}.
    """
    ##-----  initialization and arg handling  -----
    ## defaults
    verbose = True
    fname   = ''

    ## second argument
    if ppopt != None:
        if isinstance(ppopt, basestring):        ## 2nd arg is FNAME (string)
            fname = ppopt
            have_ppopt = False
        else:                    ## 2nd arg is MPOPT (MATPOWER options vector)
            have_ppopt = True
            verbose = ppopt['VERBOSE']
            if ppopt['GRB_OPT']:
                fname = 'gurobi_user_options_%d', ppopt['GRB_OPT']
    else:
        have_ppopt = False

    ##-----  set default options for CPLEX  -----
    opt = {}
    # opt['OptimalityTol'] = 1e-6
    ## -1 - auto, 0 - no, 1 - conserv, 2 - aggressive=
    # opt['Presolve'] = -1
    # opt['LogFile'] = 'qps_gurobi.log'
    if have_ppopt:
        ## (make default OPF_VIOLATION correspond to default FeasibilityTol)
        opt['FeasibilityTol']  = ppopt['OPF_VIOLATION'] / 5
        opt['Method']          = ppopt['GRB_METHOD']
        opt['TimeLimit']       = ppopt['GRB_TIMELIMIT']
        opt['Threads']         = ppopt['GRB_THREADS']
    else:
        opt['Method']          = 1            ## dual simplex

    opt['Display'] = min(verbose, 3)
    if verbose:
        opt['DisplayInterval'] = 1
    else:
        opt['DisplayInterval'] = Inf

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

    return opt
