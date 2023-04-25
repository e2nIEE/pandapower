# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import tempfile
from collections.abc import Iterable
import tqdm

import pandapower as pp
from pandapower import LoadflowNotConverged, OPFNotConverged
from pandapower.control.run_control import ControllerNotConverged, prepare_run_ctrl, \
    run_control, NetCalculationNotConverged
from pandapower.control.util.diagnostic import control_diagnostic
from pandapower.timeseries.output_writer import OutputWriter

try:
    import pandaplan.core.pplog as pplog
except ImportError:
    import logging as pplog

logger = pplog.getLogger(__name__)
logger.setLevel(level=pplog.WARNING)


def init_default_outputwriter(net, time_steps, **kwargs):
    """
    Initializes the output writer. If output_writer is None, default output_writer is created

    INPUT:
        **net** - The pandapower format network

        **time_steps** (list) - time steps to be calculated

    """
    output_writer = kwargs.get("output_writer", None)
    if output_writer is not None:
        # write the output_writer to net
        logger.warning("deprecated: output_writer should not be given to run_timeseries(). "
                       "This overwrites the stored one in net.output_writer.")
        net.output_writer.iat[0, 0] = output_writer
    if "output_writer" not in net or net.output_writer.iat[0, 0] is None:
        # create a default output writer for this net
        ow = OutputWriter(net, time_steps, output_path=tempfile.gettempdir())
        logger.info("No output writer specified. Using default:")
        logger.info(ow)


def init_output_writer(net, time_steps):
    # init output writer before time series calculation
    output_writer = net.output_writer.iat[0, 0]
    output_writer.time_steps = time_steps
    output_writer.init_all(net)


#
# def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
#     """
#     Call in a loop to create terminal progress bar.
#     the code is mentioned in : https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
#     """
#     percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
#     filled_length = int(length * iteration // total)
#     bar = fill * filled_length + '-' * (length - filled_length)
#     # logger.info('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
#     print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end="")
#     # Print New Line on Complete
#     if iteration == total:
#         print("\n")


def controller_not_converged(time_step, ts_variables):
    logger.error('ControllerNotConverged at time step %s' % time_step)
    if not ts_variables["continue_on_divergence"]:
        raise ControllerNotConverged


def pf_not_converged(time_step, ts_variables):
    logger.error('CalculationNotConverged at time step %s' % time_step)
    if not ts_variables["continue_on_divergence"]:
        raise ts_variables['errors'][0]


def control_time_step(controller_order, time_step):
    for levelorder in controller_order:
        for ctrl, net in levelorder:
            ctrl.time_step(net, time_step)


def finalize_step(controller_order, time_step):
    for levelorder in controller_order:
        for ctrl, net in levelorder:
            ctrl.finalize_step(net, time_step)


def output_writer_routine(net, time_step, pf_converged, ctrl_converged, recycle_options):
    output_writer = net["output_writer"].iat[0, 0]
    # update time step for output writer
    output_writer.time_step = time_step
    # save
    output_writer.save_results(net, time_step, pf_converged=pf_converged, ctrl_converged=ctrl_converged,
                               recycle_options=recycle_options)


def _call_output_writer(net, time_step, pf_converged, ctrl_converged, ts_variables):
    output_writer_routine(net, time_step, pf_converged, ctrl_converged, ts_variables['recycle_options'])


def run_time_step(net, time_step, ts_variables, run_control_fct=run_control, output_writer_fct=_call_output_writer,
                  **kwargs):
    """
    Time Series step function
    Is called to run the PANDAPOWER AC power flows with the timeseries module

    INPUT:
        **net** - The pandapower format network

        **time_step** (int) - time_step to be calculated

        **ts_variables** (dict) - contains settings for controller and time series simulation. See init_time_series()
    """
    ctrl_converged = True
    pf_converged = True
    # run time step function for each controller

    control_time_step(ts_variables['controller_order'], time_step)

    try:
        # calls controller init, control steps and run function (runpp usually is called in here)
        run_control_fct(net, ctrl_variables=ts_variables, **kwargs)
    except ControllerNotConverged:
        ctrl_converged = False
        # If controller did not converge do some stuff
        controller_not_converged(time_step, ts_variables)
    except ts_variables['errors']:
        # If power flow did not converge simulation aborts or continues if continue_on_divergence is True
        pf_converged = False
        pf_not_converged(time_step, ts_variables)

    output_writer_fct(net, time_step, pf_converged, ctrl_converged, ts_variables)

    finalize_step(ts_variables['controller_order'], time_step)


def _check_controller_recyclability(net):
    # if a parameter is set to True here, it will be recalculated during the time series simulation
    recycle = dict(trafo=False, gen=False, bus_pq=False)
    if "controller" not in net:
        # everything can be recycled since no controller is in net. But the time series simulation makes no sense
        # then anyway...
        return recycle

    for idx in net.controller.index:
        # todo: write to controller data frame recycle column instead of using self.recycle of controller instance
        ctrl_recycle = net.controller.at[idx, "recycle"]
        if not isinstance(ctrl_recycle, dict):
            # if one controller has a wrong recycle configuration it is deactived
            recycle = False
            break
        # else check which recycle parameter are set to True
        for rp in ["trafo", "bus_pq", "gen"]:
            recycle[rp] = recycle[rp] or ctrl_recycle[rp]

    return recycle


def _check_output_writer_recyclability(net, recycle, run):
    if "output_writer" not in net:
        raise ValueError("OutputWriter not defined")
    ow = net.output_writer.at[0, "object"]
    # results which are read with a faster batch function after the time series simulation
    recycle["batch_read"] = list()
    recycle["only_v_results"] = False
    new_log_variables = list()

    if hasattr(run, "__name__") and run.__name__ == "rundcpp":
        recycle["only_v_results"] = False
        recycle["batch_read"] = False
        return recycle

    for output in ow.log_variables:
        table, variable = output[0], output[1]
        if table not in ["res_bus", "res_line", "res_trafo", "res_trafo3w"] or recycle["trafo"] or len(output) > 2:
            # no fast read of outputs possible if other elements are required as these or tap changer is active
            recycle["only_v_results"] = False
            recycle["batch_read"] = False
            return recycle
        else:
            # fast read is possible
            if variable in ["vm_pu", "va_degree"]:
                new_log_variables.append(('ppc_bus', 'vm'))
                new_log_variables.append(('ppc_bus', 'va'))

            recycle["only_v_results"] = True
            recycle["batch_read"].append((table, variable))

    ow.log_variables = new_log_variables
    ow.log_variable('ppc_bus', 'vm')
    ow.log_variable('ppc_bus', 'va')
    return recycle


def get_recycle_settings(net, **kwargs):
    """
    checks if "run" is specified in kwargs and calls this function in time series loop.
    if "recycle" is in kwargs we use the TimeSeriesRunpp class (not implemented yet)

    INPUT:
        **net** - The pandapower format network

    RETURN:
        **recycle** - a dict with recycle options to be used by runpp
    """

    recycle = kwargs.get("recycle", None)
    if recycle is not False:
        # check if every controller can be recycled and what can be recycled
        recycle = _check_controller_recyclability(net)
        # if still recycle is not None, also check for fast output_writer features
        if recycle is not False:
            recycle = _check_output_writer_recyclability(net, recycle, kwargs.get("run", kwargs.get("run_control_fct")))

    return recycle


def init_time_steps(net, time_steps, **kwargs):
    # initializes time steps if as a range
    if not isinstance(time_steps, Iterable):
        if isinstance(time_steps, tuple):
            time_steps = range(time_steps[0], time_steps[1])
        elif time_steps is None and ("start_step" in kwargs and "stop_step" in kwargs):
            logger.warning("start_step and stop_step are depricated. "
                           "Please use a tuple like time_steps = (start_step, stop_step) instead or a list")
            time_steps = range(kwargs["start_step"], kwargs["stop_step"] + 1)
        else:
            logger.warning("No time steps to calculate are specified. "
                           "I'll check the datasource of the first controller for avaiable time steps")
            ds = net.controller.object.at[0].data_source
            if ds is None:
                raise UserWarning("No time steps are specified and the first controller doesn't have a data source"
                                  "the time steps could be retrieved from")
            else:
                max_timestep = ds.get_time_steps_len()
            time_steps = range(max_timestep)
    return time_steps


def init_time_series(net, time_steps, continue_on_divergence=False, verbose=True,
                     **kwargs):
    """
    inits the time series calculation
    creates the dict ts_variables, which includes necessary variables for the time series / control function

    INPUT:
        **net** - The pandapower format network

        **time_steps** (list or tuple, None) - time_steps to calculate as list or tuple (start, stop)
        if None, all time steps from provided data source are simulated

    OPTIONAL:

        **continue_on_divergence** (bool, False) - If True time series calculation continues in case of errors.

        **verbose** (bool, True) - prints progress bar or logger debug messages
    """

    time_steps = init_time_steps(net, time_steps, **kwargs)

    init_default_outputwriter(net, time_steps, **kwargs)
    # get run function
    run = kwargs.pop("run", pp.runpp)
    recycle_options = None
    if hasattr(run, "__name__") and (run.__name__ == "runpp" or run.__name__ == "rundcpp"):
        # use faster runpp options if possible
        recycle_options = get_recycle_settings(net, run=run, **kwargs)

    init_output_writer(net, time_steps)
    # as base take everything considered when preparing run_control
    ts_variables = prepare_run_ctrl(net, None, run=run, **kwargs)
    # recycle options, which define what can be recycled
    ts_variables["recycle_options"] = recycle_options
    # time steps to be calculated (list or range)
    ts_variables["time_steps"] = time_steps
    # If True, a diverged run is ignored and the next step is calculated
    ts_variables["continue_on_divergence"] = continue_on_divergence
    # print settings
    ts_variables["verbose"] = verbose

    if logger.level != 10 and verbose:
        # simple progress bar
        ts_variables['progress_bar'] = tqdm.tqdm(total=len(time_steps))

    return ts_variables


def cleanup(net, ts_variables):
    if isinstance(ts_variables["recycle_options"], dict):
        # Todo: delete internal variables and dumped results which are not needed
        net._ppc = None  # remove _ppc because if recycle == True and a new timeseries calculation is started with a different setup (in_service of lines or trafos, open switches etc.) it can lead to a disaster


def print_progress(i, time_step, time_steps, verbose, **kwargs):
    # simple status print in each time step.
    if logger.level != 10 and verbose:
        kwargs['ts_variables']["progress_bar"].update(1)

    # print debug info
    if logger.level == pplog.DEBUG and verbose:
        logger.debug("run time step %i" % time_step)

    # call a custom progress function
    if "progress_function" in kwargs:
        func = kwargs["progress_function"]
        func(i, time_step, time_steps, **kwargs)


def run_loop(net, ts_variables, run_control_fct=run_control, output_writer_fct=_call_output_writer, **kwargs):
    """
    runs the time series loop which calls pp.runpp (or another run function) in each iteration

    Parameters
    ----------
    net - pandapower net
    ts_variables - settings for time series

    """
    for i, time_step in enumerate(ts_variables["time_steps"]):
        print_progress(i, time_step, ts_variables["time_steps"], ts_variables["verbose"], ts_variables=ts_variables,
                       **kwargs)
        run_time_step(net, time_step, ts_variables, run_control_fct, output_writer_fct, **kwargs)


def run_timeseries(net, time_steps=None, continue_on_divergence=False, verbose=True, check_controllers=True, **kwargs):
    """
    Time Series main function

    Runs multiple PANDAPOWER AC power flows based on time series which are stored in a **DataSource** inside
    **Controllers**. Optionally other functions than the pp power flow can be called by setting the run function in kwargs

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **time_steps** (list or tuple, None) - time_steps to calculate as list or tuple (start, stop)
        if None, all time steps from provided data source are simulated

        **continue_on_divergence** (bool, False) - If True time series calculation continues in case of errors.

        **verbose** (bool, True) - prints progress bar or if logger.level == Debug it prints debug messages

        **kwargs** - Keyword arguments for run_control and runpp. If "run" is in kwargs the default call to runpp()
        is replaced by the function kwargs["run"]
    """

    ts_variables = init_time_series(net, time_steps, continue_on_divergence, verbose, **kwargs)

    # cleanup ppc before first time step
    cleanup(net, ts_variables)

    if check_controllers:
        control_diagnostic(net) # produces significant overhead if you run many timeseries of short duration
    run_loop(net, ts_variables, **kwargs)

    # cleanup functions after the last time step was calculated
    cleanup(net, ts_variables)
    # both cleanups, at the start AND at the end, are important!
