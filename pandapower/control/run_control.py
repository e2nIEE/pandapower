# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pandapower as pp
import numpy as np

try:
    import pplog
except:
    import logging as pplog

from pandapower import ppException, LoadflowNotConverged, OPFNotConverged
from pandapower.control.util.auxiliary import asarray

logger = pplog.getLogger(__name__)


class ControllerNotConverged(ppException):
    """
    Exception being raised in case a controller does not converge.
    """
    pass


class NetCalculationNotConverged(ppException):
    """
    Exception being raised in case a controller does not converge.
    """
    pass


def get_controller_order(nets, controller):
    """
    Defining the controller order per level
    Takes the order and level columns from net.controller
    If levels are specified, the levels and orders are executed in ascending order.
    :param net:
    :return: level_list - list of levels to be run
    :return: controller_order - list of controller order lists per level
    """
    # let level be float so that a new level can be added in between of existing ones
    if not hasattr(nets, '__len__') or (isinstance(nets, dict)):
        nets = [nets] * len(controller)
        nets = np.array(nets)
    if nets is not np.ndarray:
        nets = np.array(nets)
    level = controller.level.fillna(0).apply(asarray).values
    # list of sorted unique levels
    level_list = sorted(set(np.concatenate(level)))

    # identify the levels to be run
    controller_order = []
    for l in level_list:
        to_add = controller.in_service.values & [*map(lambda x: l in x, level)]
        rel_controller, order = controller['object'].values[to_add], controller['order'].values[to_add]
        controller_order.append([*zip(rel_controller[order.argsort()], nets[to_add][order.argsort()])])
        # controller_order.append(net.controller[to_add].sort_values(["order"]).object.values)

    logger.debug("levellist: " + str(level_list))
    logger.debug("order: " + str(controller_order))

    return level_list, controller_order


def check_for_initial_run(controller_order):
    """
    Function checking if any of the controllers need an initial power flow
    If net has no controllers, an initial power flow is done by default
    """

    if not len(controller_order[0]):
        return True

    for levelorder in controller_order:
        for ctrl, net in levelorder:
            if hasattr(ctrl, 'initial_powerflow'):
                net.controller.at[ctrl.index, 'initial_run']= ctrl.initial_powerflow
                logger.warning("initial_powerflow is deprecated. Instead of defining initial_powerflow "
                               "please define initial_run in the future.")
                del ctrl.initial_powerflow
            elif hasattr(ctrl, 'initial_run'):
                net.controller.at[ctrl.index, 'initial_run']= ctrl.initial_run
                logger.warning("initial_run as attribute is deprecated. initial_run is now part of the "
                               "net.controller DataFrame")
                del ctrl.initial_run
            if net.controller.at[ctrl.index, 'initial_run']:
                return True
    return False


def ctrl_variables_default(net):
    ctrl_variables = dict()
    if not hasattr(net, "controller") or len(net.controller[net.controller.in_service]) == 0:
        ctrl_variables["level"], ctrl_variables["controller_order"] = [0], [[]]
    else:
        ctrl_variables["level"], ctrl_variables["controller_order"] = get_controller_order(net, net.controller)
    ctrl_variables["run"] = pp.runpp
    ctrl_variables["initial_run"] = check_for_initial_run(
        ctrl_variables["controller_order"])
    ctrl_variables['continue_on_divergence'] = False
    return ctrl_variables


def prepare_run_ctrl(net, ctrl_variables):
    """
    Prepares run control functions. Internal variables needed:

    **controller_order** (list) - Order in which controllers in net.controller will be called
    **runpp** (function) - the runpp function (for time series a faster version is possible)
    **initial_run** (bool) - some controllers need an initial run of the powerflow prior to the control step

    """
    # sort controller_order by order if not already done
    if ctrl_variables is None:
        ctrl_variables = ctrl_variables_default(net)

    ctrl_variables["errors"] = (LoadflowNotConverged, OPFNotConverged, NetCalculationNotConverged)

    return ctrl_variables


def check_final_convergence(run_count, max_iter, net_converged):
    if run_count > max_iter:
        raise ControllerNotConverged("Maximum number of iterations per controller is reached. "
                                     "Some controller did not converge after %i calculations!"
                                     % run_count)
    if not net_converged:
        raise NetCalculationNotConverged("Controller did not converge because the calculation did not converge!")
    else:
        logger.debug("Converged after %i calculations" % run_count)


def get_recycle(ctrl_variables):
    # check if recycle is in ctrl_variables
    recycle, only_v_results = None, False
    if ctrl_variables is not None and "recycle_options" in ctrl_variables:
        recycle = ctrl_variables.get("recycle_options", None)
        if isinstance(recycle, dict):
            only_v_results = recycle.get("only_v_results", False)
    return recycle, only_v_results


def net_initialization(net, ctrl_variables, **kwargs):
    # initial power flow (takes time, but is not needed for every kind of controller)
    initial_run = ctrl_variables['initial_run']
    run_funct = ctrl_variables['run']
    if initial_run:
        run_funct(net, **kwargs)  # run can be runpp, runopf or whatever
    else:
        net["converged"] = True  # assume that the initial state is valid
    ctrl_variables['converged'] = net['converged'] or net['OPF_converged']
    return ctrl_variables


def control_initialization(controller_order):
    # initialize each controller prior to the first power flow
    for levelorder in controller_order:
        for ctrl, net in levelorder:
            ctrl.initialize_control(net)


def _evaluate_net(net, levelorder, ctrl_variables, **kwargs):
    run_funct = ctrl_variables['run']
    errors = ctrl_variables['errors']
    try:
        run_funct(net, **kwargs)  # run can be runpp, runopf or whatever
    except errors:

        if ctrl_variables['continue_on_divergence']:
            # give a chance to controllers to "repair" the control step if load flow
            # didn't converge
            # either implement this in a controller that is likely to cause the error,
            # or define a special "load flow police" controller for your use case
            _control_repair(levelorder)
            # this will raise the error if repair_control did't work
            # it means that repair control has only 1 try
            try:
                run_funct(net, **kwargs)
            except errors:
                pass
    ctrl_variables['converged'] = net['converged'] or net['OPF_converged']
    return ctrl_variables


def control_implementation(net, controller_order, ctrl_variables, max_iter,
                           evaluate_net_fct=_evaluate_net, **kwargs):
    # run each controller step in given controller order
    for levelorder in controller_order:
        # converged gives status about convergence of a controller. Is initialized as False
        ctrl_converged = False
        # run_count is 0 before entering the loop. Is incremented in each controller loop
        converged = ctrl_variables['converged']
        run_count = 0
        while not ctrl_converged and run_count <= max_iter and converged:
            ctrl_converged = _control_step(levelorder, run_count)
            # call to run function (usually runpp) after each controller was called
            # this function is called at least once per level
            if not ctrl_converged:
                run_count += 1
                ctrl_variables = evaluate_net_fct(net, levelorder, ctrl_variables, **kwargs)
        # raises controller not converged
        check_final_convergence(run_count, max_iter, ctrl_variables['converged'])


def _control_step(levelorder, run_count):
    # keep track of stopping criteria
    converged = True
    logger.debug("Controller Iteration #%i" % run_count)
    # run each controller until all are converged
    for ctrl, net in levelorder:
        # call control step while controller ist not converged yet
        if not ctrl.is_converged(net):
            ctrl.control_step(net)
            converged = False
    return converged


def _control_repair(levelorder):
    for ctrl, net in levelorder:
        ctrl.repair_control(net)


def control_finalization(controller_order):
    # call finalize function of each controller
    for levelorder in controller_order:
        for ctrl, net in levelorder:
            ctrl.finalize_control(net)


def run_control(net, ctrl_variables=None, max_iter=30, **kwargs):
    """
    Main function to call a net with controllers
    Function is running control loops for the controllers specified in net.controller

    INPUT:
   **net** - pandapower network with controllers included in net.controller

    OPTIONAL:
       **ctrl_variables** (dict, None) - variables needed internally to calculate the power flow. See prepare_run_ctrl()
       **max_iter** (int, 30) - The maximum number of iterations for controller to converge

    Runs controller until each one converged or max_iter is hit.

    1. Call initialize_control() on each controller
    2. Calculate an inital power flow (if it is enabled, i.e. setting the initial_run veriable to True)
    3. Repeats the following steps in ascending order of controller_order until total convergence of all
       controllers for each level:
        a) Evaluate individual convergence for all controllers in the level
        b) Call control_step() for all controllers in the level on diverged controllers
        c) Calculate power flow (or optionally another function like runopf or whatever you defined)
    4. Call finalize_control() on each controller

    """
    ctrl_variables = prepare_run_ctrl(net, ctrl_variables)
    kwargs["recycle"], kwargs["only_v_results"] = get_recycle(ctrl_variables)

    controller_order = ctrl_variables["controller_order"]

    # initialize each controller prior to the first power flow
    control_initialization(controller_order)

    # initial power flow (takes time, but is not needed for every kind of controller)
    ctrl_variables = net_initialization(net, ctrl_variables, **kwargs)

    # run each controller step in given controller order
    control_implementation(net, controller_order, ctrl_variables, max_iter, **kwargs)

    # call finalize function of each controller
    control_finalization(controller_order)
