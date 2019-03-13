# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Sets options for IPOPT.
"""

from pandapower.pypower._compat import PY2
from pandapower.pypower.util import feval


if not PY2:
    basestring = str


def ipopt_options(overrides=None, ppopt=None):
    """Sets options for IPOPT.

    Sets the values for the options.ipopt dict normally passed to
    IPOPT.

    Inputs are all optional, second argument must be either a string
    (C{fname}) or a dict (C{ppopt}):

        - C{overrides}
            - dict containing values to override the defaults
            - C{fname} name of user-supplied function called after default
            options are set to modify them. Calling syntax is::
                modified_opt = fname(default_opt)
        - C{ppopt} PYPOWER options vector, uses the following entries:
            - C{OPF_VIOLATION} used to set opt['constr_viol_tol']
            - C{VERBOSE}       used to opt['print_level']
            - C{IPOPT_OPT}     user option file, if ppopt['IPOPT_OPT'] is
            non-zero it is appended to 'ipopt_user_options_' to form
            the name of a user-supplied function used as C{fname}
            described above, except with calling syntax::
                modified_opt = fname(default_opt ppopt)

    Output is an options.ipopt dict to pass to IPOPT.

    Example: If ppopt['IPOPT_OPT'] = 3, then after setting the default IPOPT
    options, L{ipopt_options} will execute the following user-defined function
    to allow option overrides::

        opt = ipopt_user_options_3(opt, ppopt);

    The contents of ipopt_user_options_3.py, could be something like::

        def ipopt_user_options_3(opt, ppopt):
            opt = {}
            opt['nlp_scaling_method'] = 'none'
            opt['max_iter']           = 500
            opt['derivative_test']    = 'first-order'
            return opt

    See the options reference section in the IPOPT documentation for
    details on the available options.

    U{http://www.coin-or.org/Ipopt/documentation/}

    @see: C{pyipopt}, L{ppoption}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ##-----  initialization and arg handling  -----
    ## defaults
    verbose = 2
    fname   = ''

    ## second argument
    if ppopt != None:
        if isinstance(ppopt, basestring):        ## 2nd arg is FNAME (string)
            fname = ppopt
            have_ppopt = False
        else:                    ## 2nd arg is ppopt (MATPOWER options vector)
            have_ppopt = True
            verbose = ppopt['VERBOSE']
            if ppopt['IPOPT_OPT']:
                fname = 'ipopt_user_options_#d' % ppopt['IPOPT_OPT']
    else:
        have_ppopt = False

    opt = {}
    ##-----  set default options for IPOPT  -----
    ## printing
    if verbose:
        opt['print_level'] = min([12, verbose * 2 + 1])
    else:
        opt['print_level'] = 0

    ## convergence
    opt['tol']             = 1e-8                 ## default 1e-8
    opt['max_iter']        = 250                  ## default 3000
    opt['dual_inf_tol']    = 0.1                  ## default 1
    if have_ppopt:
        opt['constr_viol_tol'] = ppopt[16]        ## default 1e-4
        opt['acceptable_constr_viol_tol'] = ppopt[16] * 100   ## default 1e-2
    opt['compl_inf_tol']   = 1e-5                 ## default 1e-4
    opt['acceptable_tol']  = 1e-8                 ## default 1e-6
    # opt['acceptable_iter'] = 15                   ## default 15
    # opt['acceptable_dual_inf_tol']     = 1e+10    ## default 1e+10
    opt['acceptable_compl_inf_tol']    = 1e-3     ## default 1e-2
    # opt['acceptable_obj_change_tol']   = 1e+20    ## default 1e+20
    # opt['diverging_iterates_tol']      = 1e+20    ## default 1e+20

    ## NLP scaling
    # opt['nlp_scaling_method']  = 'none'           ## default 'gradient-based'

    ## NLP
    # opt['fixed_variable_treatment']    = 'make_constraint'    ## default 'make_parameter'
    # opt['honor_original_bounds']       = 'no'                 ## default 'yes'
    # opt['check_derivatives_for_naninf'] = 'yes'               ## default 'no'

    ## initialization
    # opt['least_square_init_primal']    = 'yes'        ## default 'no'
    # opt['least_square_init_duals']     = 'yes'        ## default 'no'

    ## barrier parameter update
    opt['mu_strategy']                 = 'adaptive'   ## default 'monotone'

    ## linear solver
    # opt['linear_solver']   = 'ma27'
    # opt['linear_solver']   = 'ma57'
    # opt['linear_solver']   = 'pardiso'
    # opt['linear_solver']   = 'wsmp'
    # opt['linear_solver']   = 'mumps'          ## default 'mumps'
    # opt['linear_solver']   = 'custom'
    # opt['linear_scaling_on_demand']    = 'no' ## default 'yes'

    ## step calculation
    # opt['mehrotra_algorithm']      = 'yes'    ## default 'no'
    # opt['fast_step_computation']   = 'yes'    ## default 'no'

    ## restoration phase
    # opt['expect_infeasible_problem']   = 'yes'    ## default 'no'

    ## derivative checker
    # opt['derivative_test']         = 'second-order'   ## default 'none'

    ## hessian approximation
    # opt['hessian_approximation']   = 'limited-memory' ## default 'exact'

    ## ma57 options
    #opt['ma57_pre_alloc'] = 3
    #opt['ma57_pivot_order'] = 4

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


#--------------------------  Options Documentation  --------------------------
# (as printed by IPOPT 3.8)
# ### Output ###
#
# print_level                            0 <= (          5) <= 12
#    Output verbosity level.
#      Sets the default verbosity level for console output. The larger this
#      value the more detailed is the output.
#
# output_file                   ("")
#    File name of desired output file (leave unset for no file output).
#      NOTE: This option only works when read from the ipopt.opt options file!
#      An output file with this name will be written (leave unset for no file
#      output).  The verbosity level is by default set to "print_level", but can
#      be overridden with "file_print_level".  The file name is changed to use
#      only small letters.
#    Possible values:
#     - *                       [Any acceptable standard file name]
#
# file_print_level                       0 <= (          5) <= 12
#    Verbosity level for output file.
#      NOTE: This option only works when read from the ipopt.opt options file!
#      Determines the verbosity level for the file specified by "output_file".
#      By default it is the same as "print_level".
#
# print_user_options            ("no")
#    Print all options set by the user.
#      If selected, the algorithm will print the list of all options set by the
#      user including their values and whether they have been used.  In some
#      cases this information might be incorrect, due to the internal program
#      flow.
#    Possible values:
#     - no                      [don't print options]
#     - yes                     [print options]
#
# print_options_documentation   ("no")
#    Switch to print all algorithmic options.
#      If selected, the algorithm will print the list of all available
#      algorithmic options with some documentation before solving the
#      optimization problem.
#    Possible values:
#     - no                      [don't print list]
#     - yes                     [print list]
#
# print_timing_statistics       ("no")
#    Switch to print timing statistics.
#      If selected, the program will print the CPU usage (user time) for
#      selected tasks.
#    Possible values:
#     - no                      [don't print statistics]
#     - yes                     [print all timing statistics]
#
# option_file_name              ("")
#    File name of options file (to overwrite default).
#      By default, the name of the Ipopt options file is "ipopt.opt" - or
#      something else if specified in the IpoptApplication::Initialize call. If
#      this option is set by SetStringValue BEFORE the options file is read, it
#      specifies the name of the options file.  It does not make any sense to
#      specify this option within the options file.
#    Possible values:
#     - *                       [Any acceptable standard file name]
#
# replace_bounds                ("no")
#    Indicates if all variable bounds should be replaced by inequality
#    constraints
#      This option must be set for the inexact algorithm
#    Possible values:
#     - no                      [leave bounds on variables]
#     - yes                     [replace variable bounds by inequality
#                                constraints]
#
# skip_finalize_solution_call   ("no")
#    Indicates if call to NLP::FinalizeSolution after optimization should be
#    suppressed
#      In some Ipopt applications, the user might want to call the
#      FinalizeSolution method separately.  Setting this option to "yes" will
#      cause the IpoptApplication object to suppress the default call to that
#      method.
#    Possible values:
#     - no                      [call FinalizeSolution]
#     - yes                     [do not call FinalizeSolution]
#
# print_info_string             ("no")
#    Enables printing of additional info string at end of iteration output.
#      This string contains some insider information about the current iteration.
#    Possible values:
#     - no                      [don't print string]
#     - yes                     [print string at end of each iteration output]
#
#
#
# ### Convergence ###
#
# tol                                    0 <  (      1e-08) <  +inf
#    Desired convergence tolerance (relative).
#      Determines the convergence tolerance for the algorithm.  The algorithm
#      terminates successfully, if the (scaled) NLP error becomes smaller than
#      this value, and if the (absolute) criteria according to "dual_inf_tol",
#      "primal_inf_tol", and "cmpl_inf_tol" are met.  (This is epsilon_tol in
#      Eqn. (6) in implementation paper).  See also "acceptable_tol" as a second
#      termination criterion.  Note, some other algorithmic features also use
#      this quantity to determine thresholds etc.
#
# s_max                                  0 <  (        100) <  +inf
#    Scaling threshold for the NLP error.
#      (See paragraph after Eqn. (6) in the implementation paper.)
#
# max_iter                               0 <= (       3000) <  +inf
#    Maximum number of iterations.
#      The algorithm terminates with an error message if the number of
#      iterations exceeded this number.
#
# max_cpu_time                           0 <  (      1e+06) <  +inf
#    Maximum number of CPU seconds.
#      A limit on CPU seconds that Ipopt can use to solve one problem.  If
#      during the convergence check this limit is exceeded, Ipopt will terminate
#      with a corresponding error message.
#
# dual_inf_tol                           0 <  (          1) <  +inf
#    Desired threshold for the dual infeasibility.
#      Absolute tolerance on the dual infeasibility. Successful termination
#      requires that the max-norm of the (unscaled) dual infeasibility is less
#      than this threshold.
#
# constr_viol_tol                        0 <  (     0.0001) <  +inf
#    Desired threshold for the constraint violation.
#      Absolute tolerance on the constraint violation. Successful termination
#      requires that the max-norm of the (unscaled) constraint violation is less
#      than this threshold.
#
# compl_inf_tol                          0 <  (     0.0001) <  +inf
#    Desired threshold for the complementarity conditions.
#      Absolute tolerance on the complementarity. Successful termination
#      requires that the max-norm of the (unscaled) complementarity is less than
#      this threshold.
#
# acceptable_tol                         0 <  (      1e-06) <  +inf
#    "Acceptable" convergence tolerance (relative).
#      Determines which (scaled) overall optimality error is considered to be
#      "acceptable." There are two levels of termination criteria.  If the usual
#      "desired" tolerances (see tol, dual_inf_tol etc) are satisfied at an
#      iteration, the algorithm immediately terminates with a success message.
#      On the other hand, if the algorithm encounters "acceptable_iter" many
#      iterations in a row that are considered "acceptable", it will terminate
#      before the desired convergence tolerance is met. This is useful in cases
#      where the algorithm might not be able to achieve the "desired" level of
#      accuracy.
#
# acceptable_iter                        0 <= (         15) <  +inf
#    Number of "acceptable" iterates before triggering termination.
#      If the algorithm encounters this many successive "acceptable" iterates
#      (see "acceptable_tol"), it terminates, assuming that the problem has been
#      solved to best possible accuracy given round-off.  If it is set to zero,
#      this heuristic is disabled.
#
# acceptable_dual_inf_tol                0 <  (      1e+10) <  +inf
#    "Acceptance" threshold for the dual infeasibility.
#      Absolute tolerance on the dual infeasibility. "Acceptable" termination
#      requires that the (max-norm of the unscaled) dual infeasibility is less
#      than this threshold; see also acceptable_tol.
#
# acceptable_constr_viol_tol             0 <  (       0.01) <  +inf
#    "Acceptance" threshold for the constraint violation.
#      Absolute tolerance on the constraint violation. "Acceptable" termination
#      requires that the max-norm of the (unscaled) constraint violation is less
#      than this threshold; see also acceptable_tol.
#
# acceptable_compl_inf_tol               0 <  (       0.01) <  +inf
#    "Acceptance" threshold for the complementarity conditions.
#      Absolute tolerance on the complementarity. "Acceptable" termination
#      requires that the max-norm of the (unscaled) complementarity is less than
#      this threshold; see also acceptable_tol.
#
# acceptable_obj_change_tol              0 <= (      1e+20) <  +inf
#    "Acceptance" stopping criterion based on objective function change.
#      If the relative change of the objective function (scaled by
#      Max(1,|f(x)|)) is less than this value, this part of the acceptable
#      tolerance termination is satisfied; see also acceptable_tol.  This is
#      useful for the quasi-Newton option, which has trouble to bring down the
#      dual infeasibility.
#
# diverging_iterates_tol                 0 <  (      1e+20) <  +inf
#    Threshold for maximal value of primal iterates.
#      If any component of the primal iterates exceeded this value (in absolute
#      terms), the optimization is aborted with the exit message that the
#      iterates seem to be diverging.
#
#
#
# ### NLP Scaling ###
#
# nlp_scaling_method            ("gradient-based")
#    Select the technique used for scaling the NLP.
#      Selects the technique used for scaling the problem internally before it
#      is solved. For user-scaling, the parameters come from the NLP. If you are
#      using AMPL, they can be specified through suffixes ("scaling_factor")
#    Possible values:
#     - none                    [no problem scaling will be performed]
#     - user-scaling            [scaling parameters will come from the user]
#     - gradient-based          [scale the problem so the maximum gradient at
#                                the starting point is scaling_max_gradient]
#     - equilibration-based     [scale the problem so that first derivatives are
#                                of order 1 at random points (only available
#                                with MC19)]
#
# obj_scaling_factor                  -inf <  (          1) <  +inf
#    Scaling factor for the objective function.
#      This option sets a scaling factor for the objective function. The scaling
#      is seen internally by Ipopt but the unscaled objective is reported in the
#      console output. If additional scaling parameters are computed (e.g.
#      user-scaling or gradient-based), both factors are multiplied. If this
#      value is chosen to be negative, Ipopt will maximize the objective
#      function instead of minimizing it.
#
# nlp_scaling_max_gradient               0 <  (        100) <  +inf
#    Maximum gradient after NLP scaling.
#      This is the gradient scaling cut-off. If the maximum gradient is above
#      this value, then gradient based scaling will be performed. Scaling
#      parameters are calculated to scale the maximum gradient back to this
#      value. (This is g_max in Section 3.8 of the implementation paper.) Note:
#      This option is only used if "nlp_scaling_method" is chosen as
#      "gradient-based".
#
# nlp_scaling_obj_target_gradient         0 <= (          0) <  +inf
#    Target value for objective function gradient size.
#      If a positive number is chosen, the scaling factor the objective function
#      is computed so that the gradient has the max norm of the given size at
#      the starting point.  This overrides nlp_scaling_max_gradient for the
#      objective function.
#
# nlp_scaling_constr_target_gradient         0 <= (          0) <  +inf
#    Target value for constraint function gradient size.
#      If a positive number is chosen, the scaling factor the constraint
#      functions is computed so that the gradient has the max norm of the given
#      size at the starting point.  This overrides nlp_scaling_max_gradient for
#      the constraint functions.
#
#
#
# ### NLP ###
#
# nlp_lower_bound_inf                 -inf <  (     -1e+19) <  +inf
#    any bound less or equal this value will be considered -inf (i.e. not lower
#    bounded).
#
# nlp_upper_bound_inf                 -inf <  (      1e+19) <  +inf
#    any bound greater or this value will be considered +inf (i.e. not upper
#    bounded).
#
# fixed_variable_treatment      ("make_parameter")
#    Determines how fixed variables should be handled.
#      The main difference between those options is that the starting point in
#      the "make_constraint" case still has the fixed variables at their given
#      values, whereas in the case "make_parameter" the functions are always
#      evaluated with the fixed values for those variables.  Also, for
#      "relax_bounds", the fixing bound constraints are relaxed (according to"
#      bound_relax_factor"). For both "make_constraints" and "relax_bounds",
#      bound multipliers are computed for the fixed variables.
#    Possible values:
#     - make_parameter          [Remove fixed variable from optimization
#                                variables]
#     - make_constraint         [Add equality constraints fixing variables]
#     - relax_bounds            [Relax fixing bound constraints]
#
# dependency_detector           ("none")
#    Indicates which linear solver should be used to detect linearly dependent
#    equality constraints.
#      The default and available choices depend on how Ipopt has been compiled.
#      This is experimental and does not work well.
#    Possible values:
#     - none                    [don't check; no extra work at beginning]
#     - mumps                   [use MUMPS]
#     - wsmp                    [use WSMP]
#     - ma28                    [use MA28]
#
# dependency_detection_with_rhs ("no")
#    Indicates if the right hand sides of the constraints should be considered
#    during dependency detection
#    Possible values:
#     - no                      [only look at gradients]
#     - yes                     [also consider right hand side]
#
# num_linear_variables                   0 <= (          0) <  +inf
#    Number of linear variables
#      When the Hessian is approximated, it is assumed that the first
#      num_linear_variables variables are linear.  The Hessian is then not
#      approximated in this space.  If the get_number_of_nonlinear_variables
#      method in the TNLP is implemented, this option is ignored.
#
# kappa_d                                0 <= (      1e-05) <  +inf
#    Weight for linear damping term (to handle one-sided bounds).
#      (see Section 3.7 in implementation paper.)
#
# bound_relax_factor                     0 <= (      1e-08) <  +inf
#    Factor for initial relaxation of the bounds.
#      Before start of the optimization, the bounds given by the user are
#      relaxed.  This option sets the factor for this relaxation.  If it is set
#      to zero, then then bounds relaxation is disabled. (See Eqn.(35) in
#      implementation paper.)
#
# honor_original_bounds         ("yes")
#    Indicates whether final points should be projected into original bounds.
#      Ipopt might relax the bounds during the optimization (see, e.g., option
#      "bound_relax_factor").  This option determines whether the final point
#      should be projected back into the user-provide original bounds after the
#      optimization.
#    Possible values:
#     - no                      [Leave final point unchanged]
#     - yes                     [Project final point back into original bounds]
#
# check_derivatives_for_naninf  ("no")
#    Indicates whether it is desired to check for Nan/Inf in derivative matrices
#      Activating this option will cause an error if an invalid number is
#      detected in the constraint Jacobians or the Lagrangian Hessian.  If this
#      is not activated, the test is skipped, and the algorithm might proceed
#      with invalid numbers and fail.
#    Possible values:
#     - no                      [Don't check (faster).]
#     - yes                     [Check Jacobians and Hessian for Nan and Inf.]
#
# jac_c_constant                ("no")
#    Indicates whether all equality constraints are linear
#      Activating this option will cause Ipopt to ask for the Jacobian of the
#      equality constraints only once from the NLP and reuse this information
#      later.
#    Possible values:
#     - no                      [Don't assume that all equality constraints are
#                                linear]
#     - yes                     [Assume that equality constraints Jacobian are
#                                constant]
#
# jac_d_constant                ("no")
#    Indicates whether all inequality constraints are linear
#      Activating this option will cause Ipopt to ask for the Jacobian of the
#      inequality constraints only once from the NLP and reuse this information
#      later.
#    Possible values:
#     - no                      [Don't assume that all inequality constraints
#                                are linear]
#     - yes                     [Assume that equality constraints Jacobian are
#                                constant]
#
# hessian_constant              ("no")
#    Indicates whether the problem is a quadratic problem
#      Activating this option will cause Ipopt to ask for the Hessian of the
#      Lagrangian function only once from the NLP and reuse this information
#      later.
#    Possible values:
#     - no                      [Assume that Hessian changes]
#     - yes                     [Assume that Hessian is constant]
#
#
#
# ### Initialization ###
#
# bound_push                             0 <  (       0.01) <  +inf
#    Desired minimum absolute distance from the initial point to bound.
#      Determines how much the initial point might have to be modified in order
#      to be sufficiently inside the bounds (together with "bound_frac").  (This
#      is kappa_1 in Section 3.6 of implementation paper.)
#
# bound_frac                             0 <  (       0.01) <= 0.5
#    Desired minimum relative distance from the initial point to bound.
#      Determines how much the initial point might have to be modified in order
#      to be sufficiently inside the bounds (together with "bound_push").  (This
#      is kappa_2 in Section 3.6 of implementation paper.)
#
# slack_bound_push                       0 <  (       0.01) <  +inf
#    Desired minimum absolute distance from the initial slack to bound.
#      Determines how much the initial slack variables might have to be modified
#      in order to be sufficiently inside the inequality bounds (together with
#      "slack_bound_frac").  (This is kappa_1 in Section 3.6 of implementation
#      paper.)
#
# slack_bound_frac                       0 <  (       0.01) <= 0.5
#    Desired minimum relative distance from the initial slack to bound.
#      Determines how much the initial slack variables might have to be modified
#      in order to be sufficiently inside the inequality bounds (together with
#      "slack_bound_push").  (This is kappa_2 in Section 3.6 of implementation
#      paper.)
#
# constr_mult_init_max                   0 <= (       1000) <  +inf
#    Maximum allowed least-square guess of constraint multipliers.
#      Determines how large the initial least-square guesses of the constraint
#      multipliers are allowed to be (in max-norm). If the guess is larger than
#      this value, it is discarded and all constraint multipliers are set to
#      zero.  This options is also used when initializing the restoration phase.
#      By default, "resto.constr_mult_init_max" (the one used in
#      RestoIterateInitializer) is set to zero.
#
# bound_mult_init_val                    0 <  (          1) <  +inf
#    Initial value for the bound multipliers.
#      All dual variables corresponding to bound constraints are initialized to
#      this value.
#
# bound_mult_init_method        ("constant")
#    Initialization method for bound multipliers
#      This option defines how the iterates for the bound multipliers are
#      initialized.  If "constant" is chosen, then all bound multipliers are
#      initialized to the value of "bound_mult_init_val".  If "mu-based" is
#      chosen, the each value is initialized to the the value of "mu_init"
#      divided by the corresponding slack variable.  This latter option might be
#      useful if the starting point is close to the optimal solution.
#    Possible values:
#     - constant                [set all bound multipliers to the value of
#                                bound_mult_init_val]
#     - mu-based                [initialize to mu_init/x_slack]
#
# least_square_init_primal      ("no")
#    Least square initialization of the primal variables
#      If set to yes, Ipopt ignores the user provided point and solves a least
#      square problem for the primal variables (x and s), to fit the linearized
#      equality and inequality constraints.  This might be useful if the user
#      doesn't know anything about the starting point, or for solving an LP or
#      QP.
#    Possible values:
#     - no                      [take user-provided point]
#     - yes                     [overwrite user-provided point with least-square
#                                estimates]
#
# least_square_init_duals       ("no")
#    Least square initialization of all dual variables
#      If set to yes, Ipopt tries to compute least-square multipliers
#      (considering ALL dual variables).  If successful, the bound multipliers
#      are possibly corrected to be at least bound_mult_init_val. This might be
#      useful if the user doesn't know anything about the starting point, or for
#      solving an LP or QP.  This overwrites option "bound_mult_init_method".
#    Possible values:
#     - no                      [use bound_mult_init_val and least-square
#                                equality constraint multipliers]
#     - yes                     [overwrite user-provided point with least-square
#                                estimates]
#
#
#
# ### Barrier Parameter Update ###
#
# mu_max_fact                            0 <  (       1000) <  +inf
#    Factor for initialization of maximum value for barrier parameter.
#      This option determines the upper bound on the barrier parameter.  This
#      upper bound is computed as the average complementarity at the initial
#      point times the value of this option. (Only used if option "mu_strategy"
#      is chosen as "adaptive".)
#
# mu_max                                 0 <  (     100000) <  +inf
#    Maximum value for barrier parameter.
#      This option specifies an upper bound on the barrier parameter in the
#      adaptive mu selection mode.  If this option is set, it overwrites the
#      effect of mu_max_fact. (Only used if option "mu_strategy" is chosen as
#      "adaptive".)
#
# mu_min                                 0 <  (      1e-11) <  +inf
#    Minimum value for barrier parameter.
#      This option specifies the lower bound on the barrier parameter in the
#      adaptive mu selection mode. By default, it is set to the minimum of 1e-11
#      and min("tol","compl_inf_tol")/("barrier_tol_factor"+1), which should be
#      a reasonable value. (Only used if option "mu_strategy" is chosen as
#      "adaptive".)
#
# adaptive_mu_globalization     ("obj-constr-filter")
#    Globalization strategy for the adaptive mu selection mode.
#      To achieve global convergence of the adaptive version, the algorithm has
#      to switch to the monotone mode (Fiacco-McCormick approach) when
#      convergence does not seem to appear.  This option sets the criterion used
#      to decide when to do this switch. (Only used if option "mu_strategy" is
#      chosen as "adaptive".)
#    Possible values:
#     - kkt-error               [nonmonotone decrease of kkt-error]
#     - obj-constr-filter       [2-dim filter for objective and constraint
#                                violation]
#     - never-monotone-mode     [disables globalization]
#
# adaptive_mu_kkterror_red_iters         0 <= (          4) <  +inf
#    Maximum number of iterations requiring sufficient progress.
#      For the "kkt-error" based globalization strategy, sufficient progress
#      must be made for "adaptive_mu_kkterror_red_iters" iterations. If this
#      number of iterations is exceeded, the globalization strategy switches to
#      the monotone mode.
#
# adaptive_mu_kkterror_red_fact          0 <  (     0.9999) <  1
#    Sufficient decrease factor for "kkt-error" globalization strategy.
#      For the "kkt-error" based globalization strategy, the error must decrease
#      by this factor to be deemed sufficient decrease.
#
# filter_margin_fact                     0 <  (      1e-05) <  1
#    Factor determining width of margin for obj-constr-filter adaptive
#    globalization strategy.
#      When using the adaptive globalization strategy, "obj-constr-filter",
#      sufficient progress for a filter entry is defined as follows: (new obj) <
#      (filter obj) - filter_margin_fact*(new constr-viol) OR (new constr-viol)
#      < (filter constr-viol) - filter_margin_fact*(new constr-viol).  For the
#      description of the "kkt-error-filter" option see "filter_max_margin".
#
# filter_max_margin                      0 <  (          1) <  +inf
#    Maximum width of margin in obj-constr-filter adaptive globalization
#    strategy.
#
# adaptive_mu_restore_previous_iterate("no")
#    Indicates if the previous iterate should be restored if the monotone mode
#    is entered.
#      When the globalization strategy for the adaptive barrier algorithm
#      switches to the monotone mode, it can either start from the most recent
#      iterate (no), or from the last iterate that was accepted (yes).
#    Possible values:
#     - no                      [don't restore accepted iterate]
#     - yes                     [restore accepted iterate]
#
# adaptive_mu_monotone_init_factor         0 <  (        0.8) <  +inf
#    Determines the initial value of the barrier parameter when switching to the
#    monotone mode.
#      When the globalization strategy for the adaptive barrier algorithm
#      switches to the monotone mode and fixed_mu_oracle is chosen as
#      "average_compl", the barrier parameter is set to the current average
#      complementarity times the value of "adaptive_mu_monotone_init_factor".
#
# adaptive_mu_kkt_norm_type     ("2-norm-squared")
#    Norm used for the KKT error in the adaptive mu globalization strategies.
#      When computing the KKT error for the globalization strategies, the norm
#      to be used is specified with this option. Note, this options is also used
#      in the QualityFunctionMuOracle.
#    Possible values:
#     - 1-norm                  [use the 1-norm (abs sum)]
#     - 2-norm-squared          [use the 2-norm squared (sum of squares)]
#     - max-norm                [use the infinity norm (max)]
#     - 2-norm                  [use 2-norm]
#
# mu_strategy                   ("monotone")
#    Update strategy for barrier parameter.
#      Determines which barrier parameter update strategy is to be used.
#    Possible values:
#     - monotone                [use the monotone (Fiacco-McCormick) strategy]
#     - adaptive                [use the adaptive update strategy]
#
# mu_oracle                     ("quality-function")
#    Oracle for a new barrier parameter in the adaptive strategy.
#      Determines how a new barrier parameter is computed in each "free-mode"
#      iteration of the adaptive barrier parameter strategy. (Only considered if
#      "adaptive" is selected for option "mu_strategy").
#    Possible values:
#     - probing                 [Mehrotra's probing heuristic]
#     - loqo                    [LOQO's centrality rule]
#     - quality-function        [minimize a quality function]
#
# fixed_mu_oracle               ("average_compl")
#    Oracle for the barrier parameter when switching to fixed mode.
#      Determines how the first value of the barrier parameter should be
#      computed when switching to the "monotone mode" in the adaptive strategy.
#      (Only considered if "adaptive" is selected for option "mu_strategy".)
#    Possible values:
#     - probing                 [Mehrotra's probing heuristic]
#     - loqo                    [LOQO's centrality rule]
#     - quality-function        [minimize a quality function]
#     - average_compl           [base on current average complementarity]
#
# mu_init                                0 <  (        0.1) <  +inf
#    Initial value for the barrier parameter.
#      This option determines the initial value for the barrier parameter (mu).
#      It is only relevant in the monotone, Fiacco-McCormick version of the
#      algorithm. (i.e., if "mu_strategy" is chosen as "monotone")
#
# barrier_tol_factor                     0 <  (         10) <  +inf
#    Factor for mu in barrier stop test.
#      The convergence tolerance for each barrier problem in the monotone mode
#      is the value of the barrier parameter times "barrier_tol_factor". This
#      option is also used in the adaptive mu strategy during the monotone mode.
#      (This is kappa_epsilon in implementation paper).
#
# mu_linear_decrease_factor              0 <  (        0.2) <  1
#    Determines linear decrease rate of barrier parameter.
#      For the Fiacco-McCormick update procedure the new barrier parameter mu is
#      obtained by taking the minimum of mu*"mu_linear_decrease_factor" and
#      mu^"superlinear_decrease_power".  (This is kappa_mu in implementation
#      paper.) This option is also used in the adaptive mu strategy during the
#      monotone mode.
#
# mu_superlinear_decrease_power          1 <  (        1.5) <  2
#    Determines superlinear decrease rate of barrier parameter.
#      For the Fiacco-McCormick update procedure the new barrier parameter mu is
#      obtained by taking the minimum of mu*"mu_linear_decrease_factor" and
#      mu^"superlinear_decrease_power".  (This is theta_mu in implementation
#      paper.) This option is also used in the adaptive mu strategy during the
#      monotone mode.
#
# mu_allow_fast_monotone_decrease("yes")
#    Allow skipping of barrier problem if barrier test is already met.
#      If set to "no", the algorithm enforces at least one iteration per barrier
#      problem, even if the barrier test is already met for the updated barrier
#      parameter.
#    Possible values:
#     - no                      [Take at least one iteration per barrier problem]
#     - yes                     [Allow fast decrease of mu if barrier test it met]
#
# tau_min                                0 <  (       0.99) <  1
#    Lower bound on fraction-to-the-boundary parameter tau.
#      (This is tau_min in the implementation paper.)  This option is also used
#      in the adaptive mu strategy during the monotone mode.
#
# sigma_max                              0 <  (        100) <  +inf
#    Maximum value of the centering parameter.
#      This is the upper bound for the centering parameter chosen by the quality
#      function based barrier parameter update. (Only used if option "mu_oracle"
#      is set to "quality-function".)
#
# sigma_min                              0 <= (      1e-06) <  +inf
#    Minimum value of the centering parameter.
#      This is the lower bound for the centering parameter chosen by the quality
#      function based barrier parameter update. (Only used if option "mu_oracle"
#      is set to "quality-function".)
#
# quality_function_norm_type    ("2-norm-squared")
#    Norm used for components of the quality function.
#      (Only used if option "mu_oracle" is set to "quality-function".)
#    Possible values:
#     - 1-norm                  [use the 1-norm (abs sum)]
#     - 2-norm-squared          [use the 2-norm squared (sum of squares)]
#     - max-norm                [use the infinity norm (max)]
#     - 2-norm                  [use 2-norm]
#
# quality_function_centrality   ("none")
#    The penalty term for centrality that is included in quality function.
#      This determines whether a term is added to the quality function to
#      penalize deviation from centrality with respect to complementarity.  The
#      complementarity measure here is the xi in the Loqo update rule. (Only
#      used if option "mu_oracle" is set to "quality-function".)
#    Possible values:
#     - none                    [no penalty term is added]
#     - log                     [complementarity * the log of the centrality
#                                measure]
#     - reciprocal              [complementarity * the reciprocal of the
#                                centrality measure]
#     - cubed-reciprocal        [complementarity * the reciprocal of the
#                                centrality measure cubed]
#
# quality_function_balancing_term("none")
#    The balancing term included in the quality function for centrality.
#      This determines whether a term is added to the quality function that
#      penalizes situations where the complementarity is much smaller than dual
#      and primal infeasibilities. (Only used if option "mu_oracle" is set to
#      "quality-function".)
#    Possible values:
#     - none                    [no balancing term is added]
#     - cubic                   [Max(0,Max(dual_inf,primal_inf)-compl)^3]
#
# quality_function_max_section_steps         0 <= (          8) <  +inf
#    Maximum number of search steps during direct search procedure determining
#    the optimal centering parameter.
#      The golden section search is performed for the quality function based mu
#      oracle. (Only used if option "mu_oracle" is set to "quality-function".)
#
# quality_function_section_sigma_tol         0 <= (       0.01) <  1
#    Tolerance for the section search procedure determining the optimal
#    centering parameter (in sigma space).
#      The golden section search is performed for the quality function based mu
#      oracle. (Only used if option "mu_oracle" is set to "quality-function".)
#
# quality_function_section_qf_tol         0 <= (          0) <  1
#    Tolerance for the golden section search procedure determining the optimal
#    centering parameter (in the function value space).
#      The golden section search is performed for the quality function based mu
#      oracle. (Only used if option "mu_oracle" is set to "quality-function".)
#
#
#
# ### Line Search ###
#
# alpha_red_factor                       0 <  (        0.5) <  1
#    Fractional reduction of the trial step size in the backtracking line search.
#      At every step of the backtracking line search, the trial step size is
#      reduced by this factor.
#
# accept_every_trial_step       ("no")
#    Always accept the first trial step.
#      Setting this option to "yes" essentially disables the line search and
#      makes the algorithm take aggressive steps, without global convergence
#      guarantees.
#    Possible values:
#     - no                      [don't arbitrarily accept the full step]
#     - yes                     [always accept the full step]
#
# accept_after_max_steps                -1 <= (         -1) <  +inf
#    Accept a trial point after maximal this number of steps.
#      Even if it does not satisfy line search conditions.
#
# alpha_for_y                   ("primal")
#    Method to determine the step size for constraint multipliers.
#      This option determines how the step size (alpha_y) will be calculated
#      when updating the constraint multipliers.
#    Possible values:
#     - primal                  [use primal step size]
#     - bound-mult              [use step size for the bound multipliers (good
#                                for LPs)]
#     - min                     [use the min of primal and bound multipliers]
#     - max                     [use the max of primal and bound multipliers]
#     - full                    [take a full step of size one]
#     - min-dual-infeas         [choose step size minimizing new dual
#                                infeasibility]
#     - safer-min-dual-infeas   [like "min_dual_infeas", but safeguarded by
#                                "min" and "max"]
#     - primal-and-full         [use the primal step size, and full step if
#                                delta_x <= alpha_for_y_tol]
#     - dual-and-full           [use the dual step size, and full step if
#                                delta_x <= alpha_for_y_tol]
#     - acceptor                [Call LSAcceptor to get step size for y]
#
# alpha_for_y_tol                        0 <= (         10) <  +inf
#    Tolerance for switching to full equality multiplier steps.
#      This is only relevant if "alpha_for_y" is chosen "primal-and-full" or
#      "dual-and-full".  The step size for the equality constraint multipliers
#      is taken to be one if the max-norm of the primal step is less than this
#      tolerance.
#
# tiny_step_tol                          0 <= (2.22045e-15) <  +inf
#    Tolerance for detecting numerically insignificant steps.
#      If the search direction in the primal variables (x and s) is, in relative
#      terms for each component, less than this value, the algorithm accepts the
#      full step without line search.  If this happens repeatedly, the algorithm
#      will terminate with a corresponding exit message. The default value is 10
#      times machine precision.
#
# tiny_step_y_tol                        0 <= (       0.01) <  +inf
#    Tolerance for quitting because of numerically insignificant steps.
#      If the search direction in the primal variables (x and s) is, in relative
#      terms for each component, repeatedly less than tiny_step_tol, and the
#      step in the y variables is smaller than this threshold, the algorithm
#      will terminate.
#
# watchdog_shortened_iter_trigger         0 <= (         10) <  +inf
#    Number of shortened iterations that trigger the watchdog.
#      If the number of successive iterations in which the backtracking line
#      search did not accept the first trial point exceeds this number, the
#      watchdog procedure is activated.  Choosing "0" here disables the watchdog
#      procedure.
#
# watchdog_trial_iter_max                1 <= (          3) <  +inf
#    Maximum number of watchdog iterations.
#      This option determines the number of trial iterations allowed before the
#      watchdog procedure is aborted and the algorithm returns to the stored
#      point.
#
# theta_max_fact                         0 <  (      10000) <  +inf
#    Determines upper bound for constraint violation in the filter.
#      The algorithmic parameter theta_max is determined as theta_max_fact times
#      the maximum of 1 and the constraint violation at initial point.  Any
#      point with a constraint violation larger than theta_max is unacceptable
#      to the filter (see Eqn. (21) in the implementation paper).
#
# theta_min_fact                         0 <  (     0.0001) <  +inf
#    Determines constraint violation threshold in the switching rule.
#      The algorithmic parameter theta_min is determined as theta_min_fact times
#      the maximum of 1 and the constraint violation at initial point.  The
#      switching rules treats an iteration as an h-type iteration whenever the
#      current constraint violation is larger than theta_min (see paragraph
#      before Eqn. (19) in the implementation paper).
#
# eta_phi                                0 <  (      1e-08) <  0.5
#    Relaxation factor in the Armijo condition.
#      (See Eqn. (20) in the implementation paper)
#
# delta                                  0 <  (          1) <  +inf
#    Multiplier for constraint violation in the switching rule.
#      (See Eqn. (19) in the implementation paper.)
#
# s_phi                                  1 <  (        2.3) <  +inf
#    Exponent for linear barrier function model in the switching rule.
#      (See Eqn. (19) in the implementation paper.)
#
# s_theta                                1 <  (        1.1) <  +inf
#    Exponent for current constraint violation in the switching rule.
#      (See Eqn. (19) in the implementation paper.)
#
# gamma_phi                              0 <  (      1e-08) <  1
#    Relaxation factor in the filter margin for the barrier function.
#      (See Eqn. (18a) in the implementation paper.)
#
# gamma_theta                            0 <  (      1e-05) <  1
#    Relaxation factor in the filter margin for the constraint violation.
#      (See Eqn. (18b) in the implementation paper.)
#
# alpha_min_frac                         0 <  (       0.05) <  1
#    Safety factor for the minimal step size (before switching to restoration
#    phase).
#      (This is gamma_alpha in Eqn. (20) in the implementation paper.)
#
# max_soc                                0 <= (          4) <  +inf
#    Maximum number of second order correction trial steps at each iteration.
#      Choosing 0 disables the second order corrections. (This is p^{max} of
#      Step A-5.9 of Algorithm A in the implementation paper.)
#
# kappa_soc                              0 <  (       0.99) <  +inf
#    Factor in the sufficient reduction rule for second order correction.
#      This option determines how much a second order correction step must
#      reduce the constraint violation so that further correction steps are
#      attempted.  (See Step A-5.9 of Algorithm A in the implementation paper.)
#
# obj_max_inc                            1 <  (          5) <  +inf
#    Determines the upper bound on the acceptable increase of barrier objective
#    function.
#      Trial points are rejected if they lead to an increase in the barrier
#      objective function by more than obj_max_inc orders of magnitude.
#
# max_filter_resets                      0 <= (          5) <  +inf
#    Maximal allowed number of filter resets
#      A positive number enables a heuristic that resets the filter, whenever in
#      more than "filter_reset_trigger" successive iterations the last rejected
#      trial steps size was rejected because of the filter.  This option
#      determine the maximal number of resets that are allowed to take place.
#
# filter_reset_trigger                   1 <= (          5) <  +inf
#    Number of iterations that trigger the filter reset.
#      If the filter reset heuristic is active and the number of successive
#      iterations in which the last rejected trial step size was rejected
#      because of the filter, the filter is reset.
#
# corrector_type                ("none")
#    The type of corrector steps that should be taken (unsupported!).
#      If "mu_strategy" is "adaptive", this option determines what kind of
#      corrector steps should be tried.
#    Possible values:
#     - none                    [no corrector]
#     - affine                  [corrector step towards mu=0]
#     - primal-dual             [corrector step towards current mu]
#
# skip_corr_if_neg_curv         ("yes")
#    Skip the corrector step in negative curvature iteration (unsupported!).
#      The corrector step is not tried if negative curvature has been
#      encountered during the computation of the search direction in the current
#      iteration. This option is only used if "mu_strategy" is "adaptive".
#    Possible values:
#     - no                      [don't skip]
#     - yes                     [skip]
#
# skip_corr_in_monotone_mode    ("yes")
#    Skip the corrector step during monotone barrier parameter mode
#    (unsupported!).
#      The corrector step is not tried if the algorithm is currently in the
#      monotone mode (see also option "barrier_strategy").This option is only
#      used if "mu_strategy" is "adaptive".
#    Possible values:
#     - no                      [don't skip]
#     - yes                     [skip]
#
# corrector_compl_avrg_red_fact          0 <  (          1) <  +inf
#    Complementarity tolerance factor for accepting corrector step
#    (unsupported!).
#      This option determines the factor by which complementarity is allowed to
#      increase for a corrector step to be accepted.
#
# nu_init                                0 <  (      1e-06) <  +inf
#    Initial value of the penalty parameter.
#
# nu_inc                                 0 <  (     0.0001) <  +inf
#    Increment of the penalty parameter.
#
# rho                                    0 <  (        0.1) <  1
#    Value in penalty parameter update formula.
#
# kappa_sigma                            0 <  (      1e+10) <  +inf
#    Factor limiting the deviation of dual variables from primal estimates.
#      If the dual variables deviate from their primal estimates, a correction
#      is performed. (See Eqn. (16) in the implementation paper.) Setting the
#      value to less than 1 disables the correction.
#
# recalc_y                      ("no")
#    Tells the algorithm to recalculate the equality and inequality multipliers
#    as least square estimates.
#      This asks the algorithm to recompute the multipliers, whenever the
#      current infeasibility is less than recalc_y_feas_tol. Choosing yes might
#      be helpful in the quasi-Newton option.  However, each recalculation
#      requires an extra factorization of the linear system.  If a limited
#      memory quasi-Newton option is chosen, this is used by default.
#    Possible values:
#     - no                      [use the Newton step to update the multipliers]
#     - yes                     [use least-square multiplier estimates]
#
# recalc_y_feas_tol                      0 <  (      1e-06) <  +inf
#    Feasibility threshold for recomputation of multipliers.
#      If recalc_y is chosen and the current infeasibility is less than this
#      value, then the multipliers are recomputed.
#
# slack_move                             0 <= (1.81899e-12) <  +inf
#    Correction size for very small slacks.
#      Due to numerical issues or the lack of an interior, the slack variables
#      might become very small.  If a slack becomes very small compared to
#      machine precision, the corresponding bound is moved slightly.  This
#      parameter determines how large the move should be.  Its default value is
#      mach_eps^{3/4}.  (See also end of Section 3.5 in implementation paper -
#      but actual implementation might be somewhat different.)
#
#
#
# ### Warm Start ###
#
# warm_start_init_point         ("no")
#    Warm-start for initial point
#      Indicates whether this optimization should use a warm start
#      initialization, where values of primal and dual variables are given
#      (e.g., from a previous optimization of a related problem.)
#    Possible values:
#     - no                      [do not use the warm start initialization]
#     - yes                     [use the warm start initialization]
#
# warm_start_same_structure     ("no")
#    Indicates whether a problem with a structure identical to the previous one
#    is to be solved.
#      If "yes" is chosen, then the algorithm assumes that an NLP is now to be
#      solved, whose structure is identical to one that already was considered
#      (with the same NLP object).
#    Possible values:
#     - no                      [Assume this is a new problem.]
#     - yes                     [Assume this is problem has known structure]
#
# warm_start_bound_push                  0 <  (      0.001) <  +inf
#    same as bound_push for the regular initializer.
#
# warm_start_bound_frac                  0 <  (      0.001) <= 0.5
#    same as bound_frac for the regular initializer.
#
# warm_start_slack_bound_push            0 <  (      0.001) <  +inf
#    same as slack_bound_push for the regular initializer.
#
# warm_start_slack_bound_frac            0 <  (      0.001) <= 0.5
#    same as slack_bound_frac for the regular initializer.
#
# warm_start_mult_bound_push             0 <  (      0.001) <  +inf
#    same as mult_bound_push for the regular initializer.
#
# warm_start_mult_init_max            -inf <  (      1e+06) <  +inf
#    Maximum initial value for the equality multipliers.
#
# warm_start_entire_iterate     ("no")
#    Tells algorithm whether to use the GetWarmStartIterate method in the NLP.
#    Possible values:
#     - no                      [call GetStartingPoint in the NLP]
#     - yes                     [call GetWarmStartIterate in the NLP]
#
#
#
# ### Linear Solver ###
#
# linear_solver                 ("mumps")
#    Linear solver used for step computations.
#      Determines which linear algebra package is to be used for the solution of
#      the augmented linear system (for obtaining the search directions). Note,
#      the code must have been compiled with the linear solver you want to
#      choose. Depending on your Ipopt installation, not all options are
#      available.
#    Possible values:
#     - ma27                    [use the Harwell routine MA27]
#     - ma57                    [use the Harwell routine MA57]
#     - pardiso                 [use the Pardiso package]
#     - wsmp                    [use WSMP package]
#     - mumps                   [use MUMPS package]
#     - custom                  [use custom linear solver]
#
# linear_system_scaling         ("none")
#    Method for scaling the linear system.
#      Determines the method used to compute symmetric scaling factors for the
#      augmented system (see also the "linear_scaling_on_demand" option).  This
#      scaling is independent of the NLP problem scaling.  By default, MC19 is
#      only used if MA27 or MA57 are selected as linear solvers. This option is
#      only available if Ipopt has been compiled with MC19.
#    Possible values:
#     - none                    [no scaling will be performed]
#     - mc19                    [use the Harwell routine MC19]
#
# linear_scaling_on_demand      ("yes")
#    Flag indicating that linear scaling is only done if it seems required.
#      This option is only important if a linear scaling method (e.g., mc19) is
#      used.  If you choose "no", then the scaling factors are computed for
#      every linear system from the start.  This can be quite expensive.
#      Choosing "yes" means that the algorithm will start the scaling method
#      only when the solutions to the linear system seem not good, and then use
#      it until the end.
#    Possible values:
#     - no                      [Always scale the linear system.]
#     - yes                     [Start using linear system scaling if solutions
#                                seem not good.]
#
#
#
# ### Step Calculation ###
#
# mehrotra_algorithm            ("no")
#    Indicates if we want to do Mehrotra's algorithm.
#      If set to yes, Ipopt runs as Mehrotra's predictor-corrector algorithm.
#      This works usually very well for LPs and convex QPs.  This automatically
#      disables the line search, and chooses the (unglobalized) adaptive mu
#      strategy with the "probing" oracle, and uses "corrector_type=affine"
#      without any safeguards; you should not set any of those options
#      explicitly in addition.  Also, unless otherwise specified, the values of
#      "bound_push", "bound_frac", and "bound_mult_init_val" are set more
#      aggressive, and sets "alpha_for_y=bound_mult".
#    Possible values:
#     - no                      [Do the usual Ipopt algorithm.]
#     - yes                     [Do Mehrotra's predictor-corrector algorithm.]
#
# fast_step_computation         ("no")
#    Indicates if the linear system should be solved quickly.
#      If set to yes, the algorithm assumes that the linear system that is
#      solved to obtain the search direction, is solved sufficiently well. In
#      that case, no residuals are computed, and the computation of the search
#      direction is a little faster.
#    Possible values:
#     - no                      [Verify solution of linear system by computing
#                                residuals.]
#     - yes                     [Trust that linear systems are solved well.]
#
# min_refinement_steps                   0 <= (          1) <  +inf
#    Minimum number of iterative refinement steps per linear system solve.
#      Iterative refinement (on the full unsymmetric system) is performed for
#      each right hand side.  This option determines the minimum number of
#      iterative refinements (i.e. at least "min_refinement_steps" iterative
#      refinement steps are enforced per right hand side.)
#
# max_refinement_steps                   0 <= (         10) <  +inf
#    Maximum number of iterative refinement steps per linear system solve.
#      Iterative refinement (on the full unsymmetric system) is performed for
#      each right hand side.  This option determines the maximum number of
#      iterative refinement steps.
#
# residual_ratio_max                     0 <  (      1e-10) <  +inf
#    Iterative refinement tolerance
#      Iterative refinement is performed until the residual test ratio is less
#      than this tolerance (or until "max_refinement_steps" refinement steps are
#      performed).
#
# residual_ratio_singular                0 <  (      1e-05) <  +inf
#    Threshold for declaring linear system singular after failed iterative
#    refinement.
#      If the residual test ratio is larger than this value after failed
#      iterative refinement, the algorithm pretends that the linear system is
#      singular.
#
# residual_improvement_factor            0 <  (          1) <  +inf
#    Minimal required reduction of residual test ratio in iterative refinement.
#      If the improvement of the residual test ratio made by one iterative
#      refinement step is not better than this factor, iterative refinement is
#      aborted.
#
# neg_curv_test_tol                      0 <  (          0) <  +inf
#    Tolerance for heuristic to ignore wrong inertia.
#      If positive, incorrect inertia in the augmented system is ignored, and we
#      test if the direction is a direction of positive curvature.  This
#      tolerance determines when the direction is considered to be sufficiently
#      positive.
#
# max_hessian_perturbation               0 <  (      1e+20) <  +inf
#    Maximum value of regularization parameter for handling negative curvature.
#      In order to guarantee that the search directions are indeed proper
#      descent directions, Ipopt requires that the inertia of the (augmented)
#      linear system for the step computation has the correct number of negative
#      and positive eigenvalues. The idea is that this guides the algorithm away
#      from maximizers and makes Ipopt more likely converge to first order
#      optimal points that are minimizers. If the inertia is not correct, a
#      multiple of the identity matrix is added to the Hessian of the Lagrangian
#      in the augmented system. This parameter gives the maximum value of the
#      regularization parameter. If a regularization of that size is not enough,
#      the algorithm skips this iteration and goes to the restoration phase.
#      (This is delta_w^max in the implementation paper.)
#
# min_hessian_perturbation               0 <= (      1e-20) <  +inf
#    Smallest perturbation of the Hessian block.
#      The size of the perturbation of the Hessian block is never selected
#      smaller than this value, unless no perturbation is necessary. (This is
#      delta_w^min in implementation paper.)
#
# perturb_inc_fact_first                 1 <  (        100) <  +inf
#    Increase factor for x-s perturbation for very first perturbation.
#      The factor by which the perturbation is increased when a trial value was
#      not sufficient - this value is used for the computation of the very first
#      perturbation and allows a different value for for the first perturbation
#      than that used for the remaining perturbations. (This is bar_kappa_w^+ in
#      the implementation paper.)
#
# perturb_inc_fact                       1 <  (          8) <  +inf
#    Increase factor for x-s perturbation.
#      The factor by which the perturbation is increased when a trial value was
#      not sufficient - this value is used for the computation of all
#      perturbations except for the first. (This is kappa_w^+ in the
#      implementation paper.)
#
# perturb_dec_fact                       0 <  (   0.333333) <  1
#    Decrease factor for x-s perturbation.
#      The factor by which the perturbation is decreased when a trial value is
#      deduced from the size of the most recent successful perturbation. (This
#      is kappa_w^- in the implementation paper.)
#
# first_hessian_perturbation             0 <  (     0.0001) <  +inf
#    Size of first x-s perturbation tried.
#      The first value tried for the x-s perturbation in the inertia correction
#      scheme.(This is delta_0 in the implementation paper.)
#
# jacobian_regularization_value          0 <= (      1e-08) <  +inf
#    Size of the regularization for rank-deficient constraint Jacobians.
#      (This is bar delta_c in the implementation paper.)
#
# jacobian_regularization_exponent         0 <= (       0.25) <  +inf
#    Exponent for mu in the regularization for rank-deficient constraint
#    Jacobians.
#      (This is kappa_c in the implementation paper.)
#
# perturb_always_cd             ("no")
#    Active permanent perturbation of constraint linearization.
#      This options makes the delta_c and delta_d perturbation be used for the
#      computation of every search direction.  Usually, it is only used when the
#      iteration matrix is singular.
#    Possible values:
#     - no                      [perturbation only used when required]
#     - yes                     [always use perturbation]
#
#
#
# ### Restoration Phase ###
#
# expect_infeasible_problem     ("no")
#    Enable heuristics to quickly detect an infeasible problem.
#      This options is meant to activate heuristics that may speed up the
#      infeasibility determination if you expect that there is a good chance for
#      the problem to be infeasible.  In the filter line search procedure, the
#      restoration phase is called more quickly than usually, and more reduction
#      in the constraint violation is enforced before the restoration phase is
#      left. If the problem is square, this option is enabled automatically.
#    Possible values:
#     - no                      [the problem probably be feasible]
#     - yes                     [the problem has a good chance to be infeasible]
#
# expect_infeasible_problem_ctol         0 <= (      0.001) <  +inf
#    Threshold for disabling "expect_infeasible_problem" option.
#      If the constraint violation becomes smaller than this threshold, the
#      "expect_infeasible_problem" heuristics in the filter line search are
#      disabled. If the problem is square, this options is set to 0.
#
# expect_infeasible_problem_ytol         0 <  (      1e+08) <  +inf
#    Multiplier threshold for activating "expect_infeasible_problem" option.
#      If the max norm of the constraint multipliers becomes larger than this
#      value and "expect_infeasible_problem" is chosen, then the restoration
#      phase is entered.
#
# start_with_resto              ("no")
#    Tells algorithm to switch to restoration phase in first iteration.
#      Setting this option to "yes" forces the algorithm to switch to the
#      feasibility restoration phase in the first iteration. If the initial
#      point is feasible, the algorithm will abort with a failure.
#    Possible values:
#     - no                      [don't force start in restoration phase]
#     - yes                     [force start in restoration phase]
#
# soft_resto_pderror_reduction_factor         0 <= (     0.9999) <  +inf
#    Required reduction in primal-dual error in the soft restoration phase.
#      The soft restoration phase attempts to reduce the primal-dual error with
#      regular steps. If the damped primal-dual step (damped only to satisfy the
#      fraction-to-the-boundary rule) is not decreasing the primal-dual error by
#      at least this factor, then the regular restoration phase is called.
#      Choosing "0" here disables the soft restoration phase.
#
# max_soft_resto_iters                   0 <= (         10) <  +inf
#    Maximum number of iterations performed successively in soft restoration
#    phase.
#      If the soft restoration phase is performed for more than so many
#      iterations in a row, the regular restoration phase is called.
#
# required_infeasibility_reduction         0 <= (        0.9) <  1
#    Required reduction of infeasibility before leaving restoration phase.
#      The restoration phase algorithm is performed, until a point is found that
#      is acceptable to the filter and the infeasibility has been reduced by at
#      least the fraction given by this option.
#
# max_resto_iter                         0 <= (    3000000) <  +inf
#    Maximum number of successive iterations in restoration phase.
#      The algorithm terminates with an error message if the number of
#      iterations successively taken in the restoration phase exceeds this
#      number.
#
# evaluate_orig_obj_at_resto_trial("yes")
#    Determines if the original objective function should be evaluated at
#    restoration phase trial points.
#      Setting this option to "yes" makes the restoration phase algorithm
#      evaluate the objective function of the original problem at every trial
#      point encountered during the restoration phase, even if this value is not
#      required.  In this way, it is guaranteed that the original objective
#      function can be evaluated without error at all accepted iterates;
#      otherwise the algorithm might fail at a point where the restoration phase
#      accepts an iterate that is good for the restoration phase problem, but
#      not the original problem.  On the other hand, if the evaluation of the
#      original objective is expensive, this might be costly.
#    Possible values:
#     - no                      [skip evaluation]
#     - yes                     [evaluate at every trial point]
#
# resto_penalty_parameter                0 <  (       1000) <  +inf
#    Penalty parameter in the restoration phase objective function.
#      This is the parameter rho in equation (31a) in the Ipopt implementation
#      paper.
#
# bound_mult_reset_threshold             0 <= (       1000) <  +inf
#    Threshold for resetting bound multipliers after the restoration phase.
#      After returning from the restoration phase, the bound multipliers are
#      updated with a Newton step for complementarity.  Here, the change in the
#      primal variables during the entire restoration phase is taken to be the
#      corresponding primal Newton step. However, if after the update the
#      largest bound multiplier exceeds the threshold specified by this option,
#      the multipliers are all reset to 1.
#
# constr_mult_reset_threshold            0 <= (          0) <  +inf
#    Threshold for resetting equality and inequality multipliers after
#    restoration phase.
#      After returning from the restoration phase, the constraint multipliers
#      are recomputed by a least square estimate.  This option triggers when
#      those least-square estimates should be ignored.
#
#
#
# ### Derivative Checker ###
#
# derivative_test               ("none")
#    Enable derivative checker
#      If this option is enabled, a (slow!) derivative test will be performed
#      before the optimization.  The test is performed at the user provided
#      starting point and marks derivative values that seem suspicious
#    Possible values:
#     - none                    [do not perform derivative test]
#     - first-order             [perform test of first derivatives at starting
#                                point]
#     - second-order            [perform test of first and second derivatives at
#                                starting point]
#     - only-second-order       [perform test of second derivatives at starting
#                                point]
#
# derivative_test_first_index           -2 <= (         -2) <  +inf
#    Index of first quantity to be checked by derivative checker
#      If this is set to -2, then all derivatives are checked.  Otherwise, for
#      the first derivative test it specifies the first variable for which the
#      test is done (counting starts at 0).  For second derivatives, it
#      specifies the first constraint for which the test is done; counting of
#      constraint indices starts at 0, and -1 refers to the objective function
#      Hessian.
#
# derivative_test_perturbation           0 <  (      1e-08) <  +inf
#    Size of the finite difference perturbation in derivative test.
#      This determines the relative perturbation of the variable entries.
#
# derivative_test_tol                    0 <  (     0.0001) <  +inf
#    Threshold for indicating wrong derivative.
#      If the relative deviation of the estimated derivative from the given one
#      is larger than this value, the corresponding derivative is marked as
#      wrong.
#
# derivative_test_print_all     ("no")
#    Indicates whether information for all estimated derivatives should be
#    printed.
#      Determines verbosity of derivative checker.
#    Possible values:
#     - no                      [Print only suspect derivatives]
#     - yes                     [Print all derivatives]
#
# jacobian_approximation        ("exact")
#    Specifies technique to compute constraint Jacobian
#    Possible values:
#     - exact                   [user-provided derivatives]
#     - finite-difference-values [user-provided structure, values by finite
#                                differences]
#
# findiff_perturbation                   0 <  (      1e-07) <  +inf
#    Size of the finite difference perturbation for derivative approximation.
#      This determines the relative perturbation of the variable entries.
#
# point_perturbation_radius              0 <= (         10) <  +inf
#    Maximal perturbation of an evaluation point.
#      If a random perturbation of a points is required, this number indicates
#      the maximal perturbation.  This is for example used when determining the
#      center point at which the finite difference derivative test is executed.
#
#
#
# ### Hessian Approximation ###
#
# limited_memory_max_history             0 <= (          6) <  +inf
#    Maximum size of the history for the limited quasi-Newton Hessian
#    approximation.
#      This option determines the number of most recent iterations that are
#      taken into account for the limited-memory quasi-Newton approximation.
#
# limited_memory_update_type    ("bfgs")
#    Quasi-Newton update formula for the limited memory approximation.
#      Determines which update formula is to be used for the limited-memory
#      quasi-Newton approximation.
#    Possible values:
#     - bfgs                    [BFGS update (with skipping)]
#     - sr1                     [SR1 (not working well)]
#
# limited_memory_initialization ("scalar1")
#    Initialization strategy for the limited memory quasi-Newton approximation.
#      Determines how the diagonal Matrix B_0 as the first term in the limited
#      memory approximation should be computed.
#    Possible values:
#     - scalar1                 [sigma = s^Ty/s^Ts]
#     - scalar2                 [sigma = y^Ty/s^Ty]
#     - constant                [sigma = limited_memory_init_val]
#
# limited_memory_init_val                0 <  (          1) <  +inf
#    Value for B0 in low-rank update.
#      The starting matrix in the low rank update, B0, is chosen to be this
#      multiple of the identity in the first iteration (when no updates have
#      been performed yet), and is constantly chosen as this value, if
#      "limited_memory_initialization" is "constant".
#
# limited_memory_init_val_max            0 <  (      1e+08) <  +inf
#    Upper bound on value for B0 in low-rank update.
#      The starting matrix in the low rank update, B0, is chosen to be this
#      multiple of the identity in the first iteration (when no updates have
#      been performed yet), and is constantly chosen as this value, if
#      "limited_memory_initialization" is "constant".
#
# limited_memory_init_val_min            0 <  (      1e-08) <  +inf
#    Lower bound on value for B0 in low-rank update.
#      The starting matrix in the low rank update, B0, is chosen to be this
#      multiple of the identity in the first iteration (when no updates have
#      been performed yet), and is constantly chosen as this value, if
#      "limited_memory_initialization" is "constant".
#
# limited_memory_max_skipping            1 <= (          2) <  +inf
#    Threshold for successive iterations where update is skipped.
#      If the update is skipped more than this number of successive iterations,
#      we quasi-Newton approximation is reset.
#
# hessian_approximation         ("exact")
#    Indicates what Hessian information is to be used.
#      This determines which kind of information for the Hessian of the
#      Lagrangian function is used by the algorithm.
#    Possible values:
#     - exact                   [Use second derivatives provided by the NLP.]
#     - limited-memory          [Perform a limited-memory quasi-Newton
#                                approximation]
#
# hessian_approximation_space   ("nonlinear-variables")
#    Indicates in which subspace the Hessian information is to be approximated.
#    Possible values:
#     - nonlinear-variables     [only in space of nonlinear variables.]
#     - all-variables           [in space of all variables (without slacks)]
#
#
#
# ### MA27 Linear Solver ###
#
# ma27_pivtol                            0 <  (      1e-08) <  1
#    Pivot tolerance for the linear solver MA27.
#      A smaller number pivots for sparsity, a larger number pivots for
#      stability.  This option is only available if Ipopt has been compiled with
#      MA27.
#
# ma27_pivtolmax                         0 <  (     0.0001) <  1
#    Maximum pivot tolerance for the linear solver MA27.
#      Ipopt may increase pivtol as high as pivtolmax to get a more accurate
#      solution to the linear system.  This option is only available if Ipopt
#      has been compiled with MA27.
#
# ma27_liw_init_factor                   1 <= (          5) <  +inf
#    Integer workspace memory for MA27.
#      The initial integer workspace memory = liw_init_factor * memory required
#      by unfactored system. Ipopt will increase the workspace size by
#      meminc_factor if required.  This option is only available if Ipopt has
#      been compiled with MA27.
#
# ma27_la_init_factor                    1 <= (          5) <  +inf
#    Real workspace memory for MA27.
#      The initial real workspace memory = la_init_factor * memory required by
#      unfactored system. Ipopt will increase the workspace size by
#      meminc_factor if required.  This option is only available if  Ipopt has
#      been compiled with MA27.
#
# ma27_meminc_factor                     1 <= (         10) <  +inf
#    Increment factor for workspace size for MA27.
#      If the integer or real workspace is not large enough, Ipopt will increase
#      its size by this factor.  This option is only available if Ipopt has been
#      compiled with MA27.
#
# ma27_skip_inertia_check       ("no")
#    Always pretend inertia is correct.
#      Setting this option to "yes" essentially disables inertia check. This
#      option makes the algorithm non-robust and easily fail, but it might give
#      some insight into the necessity of inertia control.
#    Possible values:
#     - no                      [check inertia]
#     - yes                     [skip inertia check]
#
# ma27_ignore_singularity       ("no")
#    Enables MA27's ability to solve a linear system even if the matrix is
#    singular.
#      Setting this option to "yes" means that Ipopt will call MA27 to compute
#      solutions for right hand sides, even if MA27 has detected that the matrix
#      is singular (but is still able to solve the linear system). In some cases
#      this might be better than using Ipopt's heuristic of small perturbation
#      of the lower diagonal of the KKT matrix.
#    Possible values:
#     - no                      [Don't have MA27 solve singular systems]
#     - yes                     [Have MA27 solve singular systems]
#
#
#
# ### MA57 Linear Solver ###
#
# ma57_pivtol                            0 <  (      1e-08) <  1
#    Pivot tolerance for the linear solver MA57.
#      A smaller number pivots for sparsity, a larger number pivots for
#      stability. This option is only available if Ipopt has been compiled with
#      MA57.
#
# ma57_pivtolmax                         0 <  (     0.0001) <  1
#    Maximum pivot tolerance for the linear solver MA57.
#      Ipopt may increase pivtol as high as ma57_pivtolmax to get a more
#      accurate solution to the linear system.  This option is only available if
#      Ipopt has been compiled with MA57.
#
# ma57_pre_alloc                         1 <= (          3) <  +inf
#    Safety factor for work space memory allocation for the linear solver MA57.
#      If 1 is chosen, the suggested amount of work space is used.  However,
#      choosing a larger number might avoid reallocation if the suggest values
#      do not suffice.  This option is only available if Ipopt has been compiled
#      with MA57.
#
# ma57_pivot_order                       0 <= (          5) <= 5
#    Controls pivot order in MA57
#      This is INCTL(6) in MA57.
#
#
#
# ### Pardiso Linear Solver ###
#
# pardiso_matching_strategy     ("complete+2x2")
#    Matching strategy to be used by Pardiso
#      This is IPAR(13) in Pardiso manual.  This option is only available if
#      Ipopt has been compiled with Pardiso.
#    Possible values:
#     - complete                [Match complete (IPAR(13)=1)]
#     - complete+2x2            [Match complete+2x2 (IPAR(13)=2)]
#     - constraints             [Match constraints (IPAR(13)=3)]
#
# pardiso_redo_symbolic_fact_only_if_inertia_wrong("no")
#    Toggle for handling case when elements were perturbed by Pardiso.
#      This option is only available if Ipopt has been compiled with Pardiso.
#    Possible values:
#     - no                      [Always redo symbolic factorization when
#                                elements were perturbed]
#     - yes                     [Only redo symbolic factorization when elements
#                                were perturbed if also the inertia was wrong]
#
# pardiso_repeated_perturbation_means_singular("no")
#    Interpretation of perturbed elements.
#      This option is only available if Ipopt has been compiled with Pardiso.
#    Possible values:
#     - no                      [Don't assume that matrix is singular if
#                                elements were perturbed after recent symbolic
#                                factorization]
#     - yes                     [Assume that matrix is singular if elements were
#                                perturbed after recent symbolic factorization]
#
# pardiso_out_of_core_power              0 <= (          0) <  +inf
#    Enables out-of-core variant of Pardiso
#      Setting this option to a positive integer k makes Pardiso work in the
#      out-of-core variant where the factor is split in 2^k subdomains.  This is
#      IPARM(50) in the Pardiso manual.  This option is only available if Ipopt
#      has been compiled with Pardiso.
#
# pardiso_msglvl                         0 <= (          0) <  +inf
#    Pardiso message level
#      This determines the amount of analysis output from the Pardiso solver.
#      This is MSGLVL in the Pardiso manual.
#
# pardiso_skip_inertia_check    ("no")
#    Always pretent inertia is correct.
#      Setting this option to "yes" essentially disables inertia check. This
#      option makes the algorithm non-robust and easily fail, but it might give
#      some insight into the necessity of inertia control.
#    Possible values:
#     - no                      [check inertia]
#     - yes                     [skip inertia check]
#
# pardiso_max_iter                       1 <= (        500) <  +inf
#    Maximum number of Krylov-Subspace Iteration
#      DPARM(1)
#
# pardiso_iter_relative_tol              0 <  (      1e-06) <  1
#    Relative Residual Convergence
#      DPARM(2)
#
# pardiso_iter_coarse_size               1 <= (       5000) <  +inf
#    Maximum Size of Coarse Grid Matrix
#      DPARM(3)
#
# pardiso_iter_max_levels                1 <= (      10000) <  +inf
#    Maximum Size of Grid Levels
#      DPARM(4)
#
# pardiso_iter_dropping_factor           0 <  (        0.5) <  1
#    dropping value for incomplete factor
#      DPARM(5)
#
# pardiso_iter_dropping_schur            0 <  (        0.1) <  1
#    dropping value for sparsify schur complement factor
#      DPARM(6)
#
# pardiso_iter_max_row_fill              1 <= (   10000000) <  +inf
#    max fill for each row
#      DPARM(7)
#
# pardiso_iter_inverse_norm_factor         1 <  (      5e+06) <  +inf
#
#      DPARM(8)
#
# pardiso_iterative             ("no")
#    Switch on iterative solver in Pardiso library
#    Possible values:
#     - no                      []
#     - yes                     []
#
# pardiso_max_droptol_corrections         1 <= (          4) <  +inf
#    Maximal number of decreases of drop tolerance during one solve.
#      This is relevant only for iterative Pardiso options.
#
#
#
# ### Mumps Linear Solver ###
#
# mumps_pivtol                           0 <= (      1e-06) <= 1
#    Pivot tolerance for the linear solver MUMPS.
#      A smaller number pivots for sparsity, a larger number pivots for
#      stability.  This option is only available if Ipopt has been compiled with
#      MUMPS.
#
# mumps_pivtolmax                        0 <= (        0.1) <= 1
#    Maximum pivot tolerance for the linear solver MUMPS.
#      Ipopt may increase pivtol as high as pivtolmax to get a more accurate
#      solution to the linear system.  This option is only available if Ipopt
#      has been compiled with MUMPS.
#
# mumps_mem_percent                      0 <= (       1000) <  +inf
#    Percentage increase in the estimated working space for MUMPS.
#      In MUMPS when significant extra fill-in is caused by numerical pivoting,
#      larger values of mumps_mem_percent may help use the workspace more
#      efficiently.  On the other hand, if memory requirement are too large at
#      the very beginning of the optimization, choosing a much smaller value for
#      this option, such as 5, might reduce memory requirements.
#
# mumps_permuting_scaling                0 <= (          7) <= 7
#    Controls permuting and scaling in MUMPS
#      This is ICNTL(6) in MUMPS.
#
# mumps_pivot_order                      0 <= (          7) <= 7
#    Controls pivot order in MUMPS
#      This is ICNTL(7) in MUMPS.
#
# mumps_scaling                         -2 <= (         77) <= 77
#    Controls scaling in MUMPS
#      This is ICNTL(8) in MUMPS.
#
# mumps_dep_tol                       -inf <  (         -1) <  +inf
#    Pivot threshold for detection of linearly dependent constraints in MUMPS.
#      When MUMPS is used to determine linearly dependent constraints, this is
#      determines the threshold for a pivot to be considered zero.  This is
#      CNTL(3) in MUMPS.
#
#
#
# ### MA28 Linear Solver ###
#
# ma28_pivtol                            0 <  (       0.01) <= 1
#    Pivot tolerance for linear solver MA28.
#      This is used when MA28 tries to find the dependent constraints.
#
#
#
# ### Uncategorized ###
#
# warm_start_target_mu                -inf <  (          0) <  +inf
#    Unsupported!
