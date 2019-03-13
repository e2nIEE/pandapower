# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Sets options for CPLEX.
"""
try:
    from cplex import cplexoptimset
except ImportError:
#    print "CPLEX not available"
    pass

from pypower._compat import PY2
from pypower.util import feval


if not PY2:
    basestring = str


def cplex_options(overrides=None, ppopt=None):
    """Sets options for CPLEX.

    Sets the values for the options dict normally passed to
    C{cplexoptimset}.

    Inputs are all optional, second argument must be either a string
    (C{fname}) or a dict (C{ppopt}):

    Output is an options dict to pass to C{cplexoptimset}.

    Example:

    If C{ppopt['CPLEX_OPT'] = 3}, then after setting the default CPLEX options,
    CPLEX_OPTIONS will execute the following user-defined function
    to allow option overrides::

        opt = cplex_user_options_3(opt, ppopt)

    The contents of cplex_user_options_3.py, could be something like::

        def cplex_user_options_3(opt, ppopt):
            opt = {}
            opt['threads']          = 2
            opt['simplex']['refactor'] = 1
            opt['timelimit']        = 10000
            return opt

    For details on the available options, see the I{"Parameters Reference
    Manual"} section of the CPLEX documentation at:
    U{http://publib.boulder.ibm.com/infocenter/cosinfoc/v12r2/}

    @param overrides:
      - dict containing values to override the defaults
      - fname: name of user-supplied function called after default
        options are set to modify them. Calling syntax is::

            modified_opt = fname(default_opt)

    @param ppopt: PYPOWER options vector, uses the following entries:
      - OPF_VIOLATION - used to set opt.simplex.tolerances.feasibility
      - VERBOSE - used to set opt.barrier.display,
        opt.conflict.display, opt.mip.display, opt.sifting.display,
        opt.simplex.display, opt.tune.display
      - CPLEX_LPMETHOD - used to set opt.lpmethod
      - CPLEX_QPMETHOD - used to set opt.qpmethod
      - CPLEX_OPT      - user option file, if ppopt['CPLEX_OPT'] is
        non-zero it is appended to 'cplex_user_options_' to form
        the name of a user-supplied function used as C{fname}
        described above, except with calling syntax::

            modified_opt = fname(default_opt, ppopt)

    @see: C{cplexlp}, C{cplexqp}, L{ppoption}.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ##-----  initialization and arg handling  -----
    ## defaults
    verbose = 1
    feastol = 1e-6
    fname   = ''

    ## second argument
    if ppopt != None:
        if isinstance(ppopt, basestring):        ## 2nd arg is FNAME (string)
            fname = ppopt
            have_ppopt = False
        else:                    ## 2nd arg is ppopt (MATPOWER options vector)
            have_ppopt = True
            ## (make default OPF_VIOLATION correspond to default CPLEX feastol)
            feastol = ppopt['OPF_VIOLATION'] / 5
            verbose = ppopt['VERBOSE']
            lpmethod = ppopt['CPLEX_LPMETHOD']
            qpmethod = ppopt['CPLEX_QPMETHOD']
            if ppopt['CPLEX_OPT']:
                fname = 'cplex_user_options_#d' % ppopt['CPLEX_OPT']
    else:
        have_ppopt = False

    ##-----  set default options for CPLEX  -----
    opt = cplexoptimset('cplex')
    opt['simplex']['tolerances']['feasibility'] = feastol

    ## printing
    vrb = max([0, verbose - 1])
    opt['barrier']['display']   = vrb
    opt['conflict']['display']  = vrb
    opt['mip']['display']       = vrb
    opt['sifting']['display']   = vrb
    opt['simplex']['display']   = vrb
    opt['tune']['display']      = vrb

    ## solution algorithm
    if have_ppopt:
        opt['lpmethod'] = lpmethod
        opt['qpmethod'] = qpmethod
    #else:
    #    opt['lpmethod'] = 2
    #    opt['qpmethod'] = 2

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
            if isinstance(overrides[names[k]], dict):
                names2 = overrides[names[k]].keys()
                for k2 in range(len(names2)):
                    if isinstance(overrides[names[k]][names2[k2]], dict):
                        names3 = overrides[names[k]][names2[k2]].keys()
                        for k3 in range(len(names3)):
                            opt[names[k]][names2[k2]][names3[k3]] = overrides[names[k]][names2[k2]][names3[k3]]
                    else:
                        opt[names[k]][names2[k2]] = overrides[names[k]][names2[k2]]
            else:
                opt[names[k]] = overrides[names[k]]

    return opt


#--------------------------  Default Options Struct  --------------------------
# as returned by ...
#   >> opt = cplexoptimset('cplex')
#
#   opt =
#       advance:        1
#       barrier:        [1x1 struct]
#           algorithm:      0
#           colnonzeros:    0
#           convergetol:    1.0000e-08
#           crossover:      0
#           display:        1
#           limits:         [1x1 struct]
#               corrections:    -1
#               growth:         1.0000e+12
#               iteration:      9.2234e+18
#               objrange:       1.0000e+20
#           ordering:       0
#           qcpconvergetol: 1.0000e-07
#           startalg:       1
#       clocktype:      2
#       conflict:       [1x1 struct]
#           display:        1
#       diagnostics:    'off'
#       emphasis:       [1x1 struct]
#           memory:         0
#           mip:            0
#           numerical:      0
#       exportmodel:    ''
#       feasopt:        [1x1 struct]
#           mode:           0
#           tolerance:      1.0000e-06
#       lpmethod:       0
#       mip:            [1x1 struct]
#           cuts:           [1x1 struct]
#               cliques:        0
#               covers:         0
#               disjunctive:    0
#               flowcovers:     0
#               gomory:         0
#               gubcovers:      0
#               implied:        0
#               mcfcut:         0
#               mircut:         0
#               pathcut:        0
#               zerohalfcut:    0
#           display:        2
#           interval:       0
#           limits:         [1x1 struct]
#               aggforcut:      3
#               auxrootthreads: 0
#               cutpasses:      0
#               cutsfactor:     4
#               eachcutlimit:   2.1000e+09
#               gomorycand:     200
#               gomorypass:     0
#               nodes:          9.2234e+18
#               polishtime:     0
#               populate:       20
#               probetime:      1.0000e+75
#               repairtries:    0
#               solutions:      9.2234e+18
#               strongcand:     10
#               strongit:       0
#               submipnodelim:  500
#               treememory:     1.0000e+75
#           ordertype:      0
#           polishafter:    [1x1 struct]
#               absmipgap:      0
#               mipgap:         0
#               nodes:          9.2234e+18
#               solutions:      9.2234e+18
#               time:           1.0000e+75
#           pool:           [1x1 struct]
#               absgap:         1.0000e+75
#               capacity:       2.1000e+09
#               intensity:      0
#               relgap:         1.0000e+75
#               replace:        0
#           strategy:       [1x1 struct]
#               backtrack:      0.9999
#               bbinterval:     7
#               branch:         0
#               dive:           0
#               file:           1
#               fpheur:         0
#               heuristicfreq:  0
#               kappastats:     0
#               lbheur:         0
#               miqcpstrat:     0
#               nodeselect:     1
#               order:          1
#               presolvenode:   0
#               probe:          0
#               rinsheur:       0
#               search:         0
#               startalgorithm: 0
#               subalgorithm:   0
#               variableselect: 0
#           tolerances:     [1x1 struct]
#               absmipgap:      1.0000e-06
#               integrality:    1.0000e-05
#               lowercutoff:    -1.0000e+75
#               mipgap:         1.0000e-04
#               objdifference:  0
#               relobjdifference: 0
#               uppercutoff:    1.0000e+75
#       output:         [1x1 struct]
#           clonelog:       1
#           intsolfileprefix: ''
#           mpslong:        1
#           writelevel:     0
#       parallel:       0
#       preprocessing:  [1x1 struct]
#           aggregator:     -1
#           boundstrength:  -1
#           coeffreduce:    -1
#           depency:     -1
#           dual:           0
#           fill:           10
#           linear:         1
#           numpass:        -1
#           presolve:       1
#           qpmakepsd:      1
#           reduce:         3
#           relax:          -1
#           repeatpresolve: -1
#           symmetry:       -1
#       qpmethod:       0
#       read:           [1x1 struct]
#       apiencoding:    ''
#           constraints:    30000
#           datacheck:      0
#           fileencoding:   'ISO-8859-1'
#           nonzeros:       250000
#           qpnonzeros:     5000
#           scale:          0
#           variables:      60000
#       sifting:        [1x1 struct]
#           algorithm:      0
#           display:        1
#           iterations:     9.2234e+18
#       simplex:        [1x1 struct]
#           crash:          1
#           dgradient:      0
#           display:        1
#           limits:         [1x1 struct]
#               iterations:     9.2234e+18
#               lowerobj:       -1.0000e+75
#               perturbation:   0
#               singularity:    10
#               upperobj:       1.0000e+75
#           perturbation:   [1x1 struct]
#               indicator:      0
#               constant:       1.0000e-06
#           pgradient:      0
#           pricing:        0
#           refactor:       0
#           tolerances:     [1x1 struct]
#               feasibility:    1.0000e-06
#               markowitz:      0.0100
#               optimality:     1.0000e-06
#       solutiontarget: 0
#       threads:        0
#       timelimit:      1.0000e+75
#       tune:           [1x1 struct]
#           display:        1
#           measure:        1
#           repeat:         1
#           timelimit:      10000
#       workdir:        '.'
#       workmem:        128
