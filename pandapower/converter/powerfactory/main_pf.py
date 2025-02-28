import sys
import os
import numpy as np
import tkinter as tk
from tkinter.filedialog import askdirectory
import pandas

import pandapower as pp

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

from pandapower.converter.powerfactory.echo_off import echo_off, echo_on
from pandapower.converter.powerfactory.pp_import_functions import from_pf
from pandapower.converter.powerfactory import pf_export_functions as pef
from pandapower.converter.powerfactory import logger_setup as pflog
from pandapower.converter.powerfactory import gui
from pandapower.converter.powerfactory.validate import validate_pf_conversion

pandas.set_option('display.width', 1000)
pandas.set_option('display.max_columns', 100)


def browse_dst(input_panel, entry_path_dst):
    logger.debug('browsing directory')
    path = askdirectory(parent=input_panel, title='Provide path to the target directory '
                                                  'to save the grid in pandapower format')
    entry_path_dst.delete(0, tk.END)
    entry_path_dst.insert(tk.END, path)
    logger.debug('directory received: <%s>' % path)


def get_dst_dir(input_panel, entry_path_dst):
    logger.debug('getting destination path')
    dst_dir = entry_path_dst.get()

    if dst_dir == '':
        logger.warning('no destination directory given')
        browse_dst(input_panel, entry_path_dst)
        dst_dir = entry_path_dst.get()

    return dst_dir


def get_filename(entry_fname, save_as="JSON"):
    logger.debug('getting file name to save network')
    filename = entry_fname.get()
    if save_as == "JSON":
        goal_extension = ".json"
    elif save_as == "Excel":
        goal_extension = ".xlsx"
    else:
        raise NotImplementedError(f"extension {save_as} not implemented")
    _, extension = os.path.splitext(filename)
    logger.debug('filename extension: %s' % extension)
    if filename == '':
        raise RuntimeError('No filename given!')
    elif not extension == goal_extension:
        logger.debug('setting correct file extension')
        entry_fname.insert(len(filename), goal_extension)
        filename = entry_fname.get()
    return filename


def save_net(net, filepath, save_as):
    if save_as == "JSON":
        pp.to_json(net, filepath)
    elif save_as == "Excel":
        pp.to_excel(net, filepath)
    else:
        raise ValueError('tried to save grid as %s to %s and failed :(' % (save_as, filepath))


def exit_gracefully(msg, is_err):
    if is_err:
        logger.error('Execution terminated: %s' % msg, exc_info=True)
    else:
        logger.info('Execution finished: %s' % msg)
    echo_on(app)


def run_export(app, pv_as_slack, pf_variable_p_loads, pf_variable_p_gen, scale_feeder_loads=False,
               flag_graphics='GPS', handle_us="Deactivate", save_as="JSON", tap_opt="nntap",
               export_controller=True, max_iter=None):
    # gather objects from the project
    logger.info('gathering network elements')
    dict_net = pef.create_network_dict(app, flag_graphics)

    logger.info('running load flow calculation')
    echo_off(app, 1, 1, 1)
    pf_load_flow_failed = pef.run_load_flow(app, scale_feeder_loads=scale_feeder_loads)
    echo_off(app, 1, 1, 1)

    logger.info('starting import to pandapower')
    app.SetAttributeModeInternal(1)

    net = from_pf(dict_net, pv_as_slack=pv_as_slack,
                  pf_variable_p_loads=pf_variable_p_loads,
                  pf_variable_p_gen=pf_variable_p_gen, flag_graphics=flag_graphics,
                  tap_opt=tap_opt, export_controller=export_controller,
                  handle_us=handle_us, max_iter=max_iter)
    # save a flag, whether the PowerFactory load flow failed
    app.SetAttributeModeInternal(0)
    net["pf_converged"] = not pf_load_flow_failed

    echo_on(app)
    return net


def run_verify(net, load_flow_params=None):
    logger.info('Validating import...')
    if load_flow_params is None:
        load_flow_params = {
            'tolerance_mva': 1e-9,  # tolerance of load flow calculation
            'calculate_voltage_angles': True,  # set True for meshed networks
            'init': 'dc',  # initialization of load flow: 'flat', 'dc', 'results'
            'PF_MAX_IT': 500  # Pypower option, maximal iterations, passed with kwargs
        }
    logger.debug('load flow params: %s' % load_flow_params)
    all_diffs = validate_pf_conversion(net, **load_flow_params)
    return all_diffs


def calc(app, input_panel, entry_path_dst, entry_fname, is_to_verify, is_debug, pv_as_slack,
         pf_variable_p_loads, pf_variable_p_gen, flag_graphics, handle_us, save_as, tap_opt,
         export_controller, max_iter_entry):
    # check if logger is to be in debug mode
    if is_debug():
        pflog.set_PF_level(logger, app_handler, 'DEBUG')
    logger.debug('starting script')
    echo_off(app, err=1, warn=1, info=1)
    # start_button.config(state="disabled")
    # ask for destination path
    try:
        dst_dir = get_dst_dir(input_panel, entry_path_dst)
    except Exception as err:
        exit_gracefully(err, True)
        return
    # ask for file name
    try:
        filename = get_filename(entry_fname, save_as())
    except RuntimeError as err:
        exit_gracefully(err, True)
        return

    logger.info('the destination directory is: <%s>' % dst_dir)
    filepath = os.path.join(dst_dir, filename)

    # the actual export
    try:
        max_iter = int(max_iter_entry.get())
        logger.info("max_iter: %s" % max_iter)
        net = run_export(app, pv_as_slack(), pf_variable_p_loads(), pf_variable_p_gen(), scale_feeder_loads=False,
                         flag_graphics=flag_graphics(), handle_us=handle_us(), save_as=save_as(), tap_opt=tap_opt(),
                         export_controller=export_controller(), max_iter=max_iter)

        logger.info('saving file to: <%s>' % filepath)
        save_net(net, filepath, save_as())
        logger.info('exported net:\n %s' % net)
    except Exception as err:
        logger.error('Error while exporting net: %s', err, exc_info=True)
    else:
        if is_to_verify():
            try:
                run_verify(net)
            except Exception as err:
                logger.error('Error while verifying net: %s', err, exc_info=True)
            # logger.info('saving file to: <%s>' % filepath)
            # save_net(net, filepath, save_as())
            # logger.info('exported validated net')
    # wrapping up
    exit_gracefully('exiting script', False)
    input_panel.destroy()
    logger.removeHandler(app_handler)
    exit()


# if called from powerfactory, __name__ is also '__main__'
if __name__ == '__main__':
    try:
        import powerfactory as pf
    except:
        pass
    else:
        # power factory app
        app = pf.GetApplication()
        ### for debugging: uncomment app.Show() and add a breakpoint
        # app.Show()
        # echo_off(app)
        # logger, app_handler = pflog.setup_logger(app, __name__, 'INFO')
        # logger = logging.getLogger(__name__)
        app_handler = pflog.AppHandler(app, freeze_app_between_messages=True)
        logger.addHandler(app_handler)

        logger.info('starting application')
        # user to store the project and folders
        user = app.GetCurrentUser()
        logger.debug('got the current user <%s>' % user.loc_name)

        prj = app.GetActiveProject()
        logger.debug('asked for active project')

        if prj is not None:
            project_name = prj.GetAttribute('loc_name')
            logger.info('got the name of active project: <%s>' % project_name)
        elif False:  # user flag for debug
            app.Show()
            echo_on(app)
            app.ActivateProject('test')
            prj = app.GetActiveProject()
            project_name = prj.GetAttribute('loc_name')
            # net = calc('main', dst_dir=r'C:\pp_projects\test')
        else:
            logger.warning(
                'No project is activated. Please activate a project to perform PandaPower '
                'Export!')
            project_name = 'None'
            logger.removeHandler(app_handler)

        gui.make_gui(app, project_name, browse_dst, calc)
