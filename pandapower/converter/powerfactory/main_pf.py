import sys
import os
import numpy as np
import tkinter as tk
from tkinter.filedialog import askdirectory
import pandas

import pandapower as pp
from pandapower import diagnostic

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
root_logger = logging.getLogger()

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


def exit_gracefully(app, input_panel, msg, is_err):
    if is_err:
        logger.error('Execution terminated: %s' % msg, exc_info=True)
    else:
        logger.info('Execution finished: %s' % msg)
    echo_on(app)
    input_panel.destroy()
    # del(app)
    # quit()
    sys.exit(1 if is_err else 0)


def run_export(app, pv_as_slack, pf_variable_p_loads, pf_variable_p_gen, scale_feeder_loads=False,
               flag_graphics='GPS', handle_us="Deactivate", save_as="JSON", tap_opt="nntap",
               export_controller=True, max_iter=None, create_sections=True):
    # gather objects from the project
    logger.info('gathering network elements')
    dict_net = pef.create_network_dict(app, flag_graphics)

    logger.info('running load flow calculation')
    echo_off(app, 1, 1, 1)
    pf_load_flow_failed = pef.run_load_flow(app, scale_feeder_loads=scale_feeder_loads)
    echo_off(app, 1, 1, 1)

    logger.info('starting import to pandapower')
    app.SetAttributeModeInternal(1)

    net = from_pf(dict_net, pv_as_slack=pv_as_slack, pf_variable_p_loads=pf_variable_p_loads,
                  pf_variable_p_gen=pf_variable_p_gen, flag_graphics=flag_graphics, tap_opt=tap_opt,
                  export_controller=export_controller, handle_us=handle_us, max_iter=max_iter,
                  create_sections=create_sections)
    # save a flag, whether the PowerFactory load flow failed
    app.SetAttributeModeInternal(0)
    net["pf_converged"] = not pf_load_flow_failed

    echo_on(app)
    return net


def run_verify(net, load_flow_params=None):
    logger.info('Validating import...')
    if load_flow_params is None:
        load_flow_params = {
            # 'tolerance_mva': 1e-9,  # tolerance of load flow calculation
            # 'calculate_voltage_angles': True,  # set True for meshed networks
            # 'init': 'dc',  # initialization of load flow: 'flat', 'dc', 'results'
            'PF_MAX_IT': 500  # Pypower option, maximal iterations, passed with kwargs
        }
    logger.debug('load flow params: %s' % load_flow_params)
    all_diffs = validate_pf_conversion(net, **load_flow_params)
    return all_diffs


def calc(app, input_panel, entry_path_dst, entry_fname, pv_as_slack, export_controller,
         replace_zero_branches, min_ohm_entry, replace_inf_branches, max_ohm_entry,
         is_to_verify, is_to_diagnostic, is_debug,
         pf_variable_p_loads, pf_variable_p_gen, flag_graphics, handle_us,
         save_as, tap_opt, max_iter_entry, create_sections_entry):
    # check if logger is to be in debug mode
    if is_debug():
        pflog.set_PF_level(root_logger, app_handler, 'DEBUG')
    else:
        pflog.set_PF_level(root_logger, app_handler, 'INFO')
    logger.debug('starting script')
    echo_off(app, err=1, warn=1, info=1)
    # start_button.config(state="disabled")
    # ask for destination path
    try:
        dst_dir = get_dst_dir(input_panel, entry_path_dst)
    except Exception as err:
        exit_gracefully(app, input_panel, err, True)
        return
    # ask for file name
    try:
        filename = get_filename(entry_fname, save_as())
    except RuntimeError as err:
        exit_gracefully(app, input_panel, err, True)
        return

    logger.info('the destination directory is: <%s>' % dst_dir)
    filepath = os.path.join(dst_dir, filename)

    # the actual export
    try:
        max_iter = int(max_iter_entry.get())
        logger.info("max_iter: %s" % max_iter)
        net = run_export(app, pv_as_slack(), pf_variable_p_loads(), pf_variable_p_gen(), scale_feeder_loads=False,
                         flag_graphics=flag_graphics(), handle_us=handle_us(), save_as=save_as(), tap_opt=tap_opt(),
                         export_controller=export_controller(), max_iter=max_iter, create_sections=create_sections_entry())
        if replace_zero_branches():
            min_ohm = float(min_ohm_entry.get())
            # to_replace = (np.abs(net.line.r_ohm_per_km * net.line.length_km +
            #                      1j * net.line.x_ohm_per_km * net.line.length_km) <= min_ohm) & net.line.in_service
            to_replace = (np.abs(net.line.x_ohm_per_km * net.line.length_km) <= min_ohm) & net.line.in_service

            for i in net.line.loc[to_replace].index.values:
                pp.create_replacement_switch_for_branch(net, "line", i)
                net.line.at[i, "in_service"] = False
            if np.any(to_replace):
                logger.info(f"replaced {sum(to_replace)} lines with switches")

            # xward = net.xward[(np.abs(net.xward.r_ohm + 1j * net.xward.x_ohm) <= min_ohm) &
            #                   net.xward.in_service].index.values
            xward = net.xward[(np.abs(net.xward.x_ohm) <= min_ohm) & net.xward.in_service].index.values
            if len(xward) > 0:
                pp.replace_xward_by_ward(net, index=xward, drop=False)
                logger.info(f"replaced {len(xward)} xwards with wards")

            zb_f_ohm = np.square(net.bus.loc[net.impedance.from_bus.values, "vn_kv"].values) / net.impedance.sn_mva
            zb_t_ohm = np.square(net.bus.loc[net.impedance.to_bus.values, "vn_kv"].values) / net.impedance.sn_mva
            # impedance = ((np.abs(net.impedance.rft_pu + 1j * net.impedance.xft_pu) <= min_ohm / zb_f_ohm) |
            #              (np.abs(net.impedance.rtf_pu + 1j * net.impedance.xtf_pu) <= min_ohm / zb_t_ohm) &
            #              net.impedance.in_service)
            impedance = ((np.abs(net.impedance.xft_pu) <= min_ohm / zb_f_ohm) |
                         (np.abs(net.impedance.xtf_pu) <= min_ohm / zb_t_ohm)) & net.impedance.in_service
            for i in net.impedance.loc[impedance].index.values:
                pp.create_replacement_switch_for_branch(net, "impedance", i)
                net.impedance.at[i, "in_service"] = False
            if any(impedance):
                logger.info(f"replaced {sum(impedance)} impedance elements with switches")

            trafos = ((net.trafo.vk_percent/100 * np.square(net.trafo.vn_lv_kv) / net.trafo.sn_mva <= min_ohm) &
                      net.trafo.in_service)
            min_vk_percent = 4.
            net.trafo.loc[trafos, "vk_percent"] = min_vk_percent
            if any(trafos):
                logger.info(f"adjusted {sum(trafos)} 2W transformers by setting vk_percent to {min_vk_percent}")

            trafo3w = ((net.trafo3w.vk_hv_percent / 100 * np.square(net.trafo3w.vn_mv_kv) / net.trafo3w.sn_hv_mva <= min_ohm) |
                       (net.trafo3w.vk_mv_percent / 100 * np.square(net.trafo3w.vn_lv_kv) / net.trafo3w.sn_mv_mva <= min_ohm) |
                       (net.trafo3w.vk_lv_percent / 100 * np.square(net.trafo3w.vn_lv_kv) / net.trafo3w.sn_lv_mva <= min_ohm)) & net.trafo3w.in_service
            net.trafo3w.loc[trafo3w, ["vk_hv_percent", "vk_mv_percent", "vk_lv_percent"]] = min_vk_percent
            if any(trafo3w):
                logger.info(f"adjusted {sum(trafo3w)} 3W transformers by setting vk_percent to {min_vk_percent}")

        if replace_inf_branches():
            max_ohm = float(max_ohm_entry.get())
            to_replace = (np.abs(net.line.r_ohm_per_km * net.line.length_km +
                                 1j * net.line.x_ohm_per_km * net.line.length_km) >= max_ohm) & net.line.in_service

            if np.any(to_replace):
                net.line.loc[to_replace, "in_service"] = False
                logger.info(f"deactivated {sum(to_replace)} lines")

            xward = (np.abs(net.xward.r_ohm + 1j * net.xward.x_ohm) >= max_ohm) & net.xward.in_service
            if np.any(xward):
                net.xward.loc[xward, "in_service"] = False
                logger.info(f"deactivated {sum(xward)} xwards")

            zb_f_ohm = np.square(net.bus.loc[net.impedance.from_bus.values, "vn_kv"].values) / net.impedance.sn_mva
            zb_t_ohm = np.square(net.bus.loc[net.impedance.to_bus.values, "vn_kv"].values) / net.impedance.sn_mva
            impedance = ((np.abs(net.impedance.rft_pu + 1j * net.impedance.xft_pu) >= max_ohm / zb_f_ohm) |
                         (np.abs(net.impedance.rtf_pu + 1j * net.impedance.xtf_pu) >= max_ohm / zb_t_ohm)) & net.impedance.in_service
            if np.any(impedance):
                net.impedance.loc[impedance, "in_service"] = False
                logger.info(f"deactivated {sum(impedance)} impedance elements")

            max_vk_percent = 15
            trafos = (net.trafo.vk_percent/100 * np.square(net.trafo.vn_hv_kv) / net.trafo.sn_mva >= max_ohm) & net.trafo.in_service
            if np.any(trafos):
                # net.trafo.loc[trafos, "in_service"] = False
                net.trafo.loc[trafos, "vk_percent"] = max_vk_percent
                logger.info(f"adjusted {sum(trafos)} 2W transformers with vk_percent of {max_vk_percent}")

            trafo3w = ((net.trafo3w.vk_hv_percent / 100 * np.square(net.trafo3w.vn_hv_kv) / net.trafo3w.sn_hv_mva >= max_ohm) |
                       (net.trafo3w.vk_mv_percent / 100 * np.square(net.trafo3w.vn_mv_kv) / net.trafo3w.sn_mv_mva >= max_ohm) |
                       (net.trafo3w.vk_lv_percent / 100 * np.square(net.trafo3w.vn_hv_kv) / net.trafo3w.sn_lv_mva >= max_ohm)) & net.trafo3w.in_service
            if np.any(trafo3w):
                # net.trafo3w.loc[trafo3w, "in_service"] = False
                net.trafo3w.loc[trafo3w, "vk_hv_percent"] = np.fmin(max_vk_percent, net.trafo3w.loc[trafo3w, "vk_hv_percent"])
                net.trafo3w.loc[trafo3w, "vk_mv_percent"] = np.fmin(max_vk_percent, net.trafo3w.loc[trafo3w, "vk_mv_percent"])
                net.trafo3w.loc[trafo3w, "vk_lv_percent"] = np.fmin(max_vk_percent, net.trafo3w.loc[trafo3w, "vk_lv_percent"])
                logger.info(f"adjusted {sum(trafo3w)} 3W transformers with {max_vk_percent}")

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
        if is_to_diagnostic():
            try:
                diagnostic(net, report_style="compact", warnings_only=True, min_x_ohm=0.01)
            except Exception as err:
                logger.error('Error in diagnostic for net: %s', err, exc_info=True)
    root_logger.removeHandler(app_handler)
    exit_gracefully(app, input_panel, 'exiting script', False)


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
        root_logger.addHandler(app_handler)

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
            root_logger.removeHandler(app_handler)

        gui.make_gui(app, project_name, browse_dst, calc)
