# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import numpy as np

import pandapower as pp
from pandapower.control.run_control import run_control, ControllerNotConverged, get_controller_order, \
    check_for_initial_powerflow
from pandapower.control.util.diagnostic import control_diagnostic
from pandapower.control.util.controller_io import dump_controller
from pandapower import LoadflowNotConverged, OPFNotConverged
from pandapower.timeseries.output_writer import OutputWriter

try:
    import pplog
except ImportError:
    import logging as pplog

logger = pplog.getLogger(__name__)
logger.setLevel(level=pplog.WARNING)

# try:
from timeseries.ts_runpp import TimeSeriesRunpp
# except:
#     logger.debug("Only open source timeseries available")



def get_default_output_writer(net, timesteps):
    """
    creates a default output writer for the time series calculation

    INPUT:
        **net** - The pandapower format network
        **time_steps** (list) - time_steps to calculate as list

    RETURN:
        **outpuw_writer** - The default output_writer
    """
    ow = OutputWriter(net, timesteps, output_path=r"./run_timeseries_output")
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'i_ka')
    ow.log_variable('res_bus', 'p_mw')
    ow.log_variable('res_bus', 'q_mvar')
    return ow


def init_outputwriter(net, time_steps, output_writer=None):
    """
    Initilizes output writer. If output_writer is None, default output_writer is created

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **output_writer** - An output_writer


    RETURN:
        **output_writer** - The initialized output_writer
    """
    if output_writer is None:
        output_writer = get_default_output_writer(net, time_steps)
    else:
        # inits output writer before time series calculation
        output_writer.time_steps = time_steps
        output_writer.init_all()
    return output_writer


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar.
    stolen from: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    # logger.info('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end="")
    # Print New Line on Complete
    if iteration == total:
        print("\n")


def controller_not_converged(net, time_step, ts_variables):
    logger.error('ControllerNotConverged at time step %s' % time_step)
    dump_controller(net, time_step)
    if not ts_variables["continue_on_divergence"]:
        raise ControllerNotConverged


def pf_not_converged(time_step, ts_variables):
    logger.error('LoadflowNotConverged at time step %s' % time_step)
    if not ts_variables["continue_on_divergence"]:
        raise LoadflowNotConverged


def run_time_step(net, time_step, ts_variables, **kwargs):
    """
    Time Series step function
    Should be called to run the PANDAPOWER AC power flows based on time series in controllers (or other functions)

    INPUT:
        **net** - The pandapower format network
        **time_step** (int) - time_step to be calculated
        **ts_variables** (dict) - contains settings for controller and time series simulation. See init_time_series()
    """
    ctrl_converged = True
    pf_converged = True
    output_writer = ts_variables["output_writer"]
    # update time step for output writer
    output_writer.time_step = time_step
    # run time step function for each controller
    for levelorder in ts_variables["controller_order"]:
        for ctrl in levelorder:
            ctrl.time_step(time_step)

    try:
        # calls controller init, control steps and run function (runpp usually is called in here)
        run_control(net, ctrl_variables=ts_variables, **kwargs)
    except ControllerNotConverged:
        ctrl_converged = False
        # If controller did not converge do some stuff
        controller_not_converged(net, time_step, ts_variables)
    except (LoadflowNotConverged, OPFNotConverged):
        # If power flow did not converge simulation aborts or continues if continue_on_divergence is True
        pf_converged = False
        pf_not_converged(time_step, ts_variables)

    # save
    output_writer.save_results(time_step, pf_converged=pf_converged, ctrl_converged=ctrl_converged)


def all_controllers_recycleable(net):
    recycleable = np.alltrue(net["controller"]["recycle"].values)
    if not recycleable:
        logger.warning("recycle feature not supported by some controllers in net. I have to deactive recycle")
    return recycleable


def get_run_function(net, output_writer, recycle, **kwargs):
    """
    checks if faster runpp function for time series is possible and activated or if another function is specified
    in **kwargs
    @param kwargs:
    @return:
    """

    recycle = True if recycle and all_controllers_recycleable(net) else False
    recycle_class = None
    if recycle:
        recycle_class = TimeSeriesRunpp(net, output_writer)
        run = recycle_class.ts_runpp
    elif "run" in kwargs:
        run = kwargs.pop("run")
    else:
        run = pp.runpp
    return run, recycle_class


def init_time_steps(net, time_steps, **kwargs):
    if not (isinstance(time_steps, list) or isinstance(time_steps, range)):
        if time_steps is None and ("start_step" in kwargs and "stop_step" in kwargs):
            logger.warning("start_step and stop_step are depricated. "
                           "Please use a tuple like time_steps = (start_step, stop_step) instead or a list")
            time_steps = range(kwargs["start_step"], kwargs["stop_step"] + 1)
        elif isinstance(time_steps, tuple):
            time_steps = range(time_steps[0], time_steps[1])
        else:
            logger.warning("No time steps to calculate are specified. "
                           "I'll check the datasource of the first controller for avaiable time steps")
            max_timestep = net.controller.loc[0].controller.data_source.get_time_steps_len()
            time_steps = range(max_timestep)
    return time_steps


def init_time_series(net, time_steps, output_writer=None, recycle=False, continue_on_divergence=False, verbose=True,
                     **kwargs):
    """
    inits the time series calculation
    creates the dict ts_variables, which includes necessary variables for the time series / control function

    INPUT:
        **net** - The pandapower format network
        **time_steps** (list or tuple, None) - time_steps to calculate as list or tuple (start, stop)
                                                if None, all time steps from provided data source are simulated
    OPTIONAL:
        **output_writer** - A predefined output writer. If None the a default one is created with
                            get_default_output_writer()
        **recycle** (bool, False) - If True faster runpp is used (rather experimental, use with caution)
        **continue_on_divergence** (bool, False) - If True time series calculation continues in case of errors.
        **verbose** (bool, True) - prints progess bar or logger debug messages
    """

    time_steps = init_time_steps(net, time_steps, **kwargs)

    ts_variables = dict()

    output_writer = init_outputwriter(net, time_steps, output_writer)
    level, order = get_controller_order(net)
    # use faster runpp if timeseries possible
    run, recycle_class = get_run_function(net, output_writer, recycle, **kwargs)

    # True at default. Initial power flow is calculated before each control step (some controllers need inits)
    ts_variables["initial_powerflow"] = check_for_initial_powerflow(order)
    # order of controller (controllers are called in a for loop.)
    ts_variables["controller_order"] = order
    # run function to be called in run_control - default is pp.runpp, but can be runopf or whatever you like
    ts_variables["run"] = run
    # recycle class function, which stores some NR variables. Only used if recycle == True
    ts_variables["recycle_class"] = recycle_class
    # output writer, which logs information during the time series simulation
    ts_variables["output_writer"] = output_writer
    # time steps to be calculated (list or range)
    ts_variables["time_steps"] = time_steps
    # If True, a diverged power flow is ignored and the next step is calculated
    ts_variables["continue_on_divergence"] = continue_on_divergence

    if logger.level is not 10 and verbose:
        # simple progress bar
        print_progress_bar(0, len(time_steps), prefix='Progress:', suffix='Complete', length=50)
    return ts_variables


def cleanup(ts_variables):
    if ts_variables["recycle_class"] is not None:
        ts_variables["recycle_class"].cleanup()


def print_progress(i, time_step, time_steps, verbose, **kwargs):
    # simple status print in each time step.
    if logger.level is not 10 and verbose:
        len_timesteps = len(time_steps)
        print_progress_bar(i + 1, len_timesteps, prefix='Progress:', suffix='Complete', length=50)

    # print debug info
    if logger.level == pplog.DEBUG and verbose:
        logger.debug("run time step %i" % time_step)

    # print luigi progress
    if "luigi_progress" in kwargs:
        if i % 365 == 0:
            # print only every 365 time steps
            message = kwargs["luigi_progress"]["message"]
            progress = kwargs["luigi_progress"]["progress"]
            len_timesteps = len(time_steps)
            message("Progress: %d / %d" % (i, len_timesteps))
            progress_percentage = int(((i+1) / len_timesteps) * 100)
            progress(progress_percentage)


def run_timeseries(net, time_steps=None, output_writer=None, recycle=False,
                   continue_on_divergence=False, verbose=True, **kwargs):
    """
    Time Series main function
    Runs multiple PANDAPOWER AC power flows based on time series in controllers
    Optionally other functions than the pp power flow can be called by setting the run function

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **time_steps** (list or tuple, None) - time_steps to calculate as list or tuple (start, stop)
                                                if None, all time steps from provided data source are simulated
        **output_writer** - A predefined output writer. If None the a default one is created with
                            get_default_output_writer()
        **recycle** (bool, False) - If True faster runpp is used (rather experimental, use with caution)
        **continue_on_divergence** (bool, False) - If True time series calculation continues in case of errors.
        **verbose** (bool, True) - prints progress bar or if logger.level == Debug it prints debug messages
        **kwargs** - Keyword arguments for run_control and runpp
    """

    ts_variables = init_time_series(net, time_steps, output_writer, recycle, continue_on_divergence, verbose,
                                    **kwargs)
    control_diagnostic(net)

    for i, time_step in enumerate(ts_variables["time_steps"]):
        print_progress(i, time_step, ts_variables["time_steps"], verbose, **kwargs)
        run_time_step(net, time_step, ts_variables, **kwargs)

    # cleanup functions after the last time step was calculated
    cleanup(ts_variables)
