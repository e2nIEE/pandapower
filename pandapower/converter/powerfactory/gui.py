import os
import tkinter as tk
from pandapower.auxiliary import ADict

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

def cancel(app, input_panel):
    logger.debug('received a cancel request from the user')
    input_panel.destroy()
    logger.debug('destroyed input panel, will attempt exit()')
    for h in logger.handlers:
        logger.removeHandler(h)
    exit_gracefully(app, 'exiting script', False)


def calc_test(app, **kwargs):
    logger.info('TESTING')
    exit_gracefully(app, 'test complete', False)


def browse_dst_test(input_panel, entry_path_dst):
    logger.info('BROWSE DESTINATION')
    entry_path_dst.delete(0, tk.END)
    entry_path_dst.insert(tk.END, 'HELLO')


def conf_var_graphics(app, opt_grf, var_graphics):
    logger.debug('configuring options for graphic options menu')
    dia_folder = app.GetProjectFolder('dia')
    dia_grid = [d.loc_name for d in dia_folder.GetContents('*.IntGrfnet')]
    dia_grid.sort()
    logger.info('found network diagrams in project: %s' % dia_grid)
    menu_options = ['no geodata', 'GPS']
    menu_options.extend(dia_grid)

    opt_grf['menu'].delete(0, 'end')
    for option in menu_options:
        opt_grf['menu'].add_command(label=option, command=tk._setit(var_graphics, option))


def make_gui(app, project_name, browse_dst, calc):
    params = ADict()
    working_directory = os.getcwd()
    input_panel = tk.Tk()
    params.input_panel = input_panel
    input_panel.iconbitmap(working_directory + r'\power_factory_files\favicon.ico')

    input_panel.title('Pandapower export')

    # row 0
    label_path_dst = tk.Label(input_panel, anchor='w', text='Path to the destination directory '
                                                            'for pandapower export: ')
    label_path_dst.grid(row=0, column=0, columnspan=3, sticky='w', padx=2, pady=4)

    entry_path_dst = tk.Entry(input_panel, width=50)
    entry_path_dst.delete(0, tk.END)
    entry_path_dst.grid(row=0, column=4, columnspan=3, padx=2, pady=2)
    # entry_path_dst.insert(0, r'C:\pp_projects\test') ##for testing

    params.entry_path_dst = entry_path_dst

    path_dst_button = tk.Button(input_panel, text='Browse', width=8,
                                command=lambda: browse_dst(input_panel, entry_path_dst))
    path_dst_button.grid(row=0, column=7, padx=4, pady=4)

    # row 1
    label_fname = tk.Label(input_panel, anchor='w', text='Provide file name for the network: ')
    label_fname.grid(row=1, column=0, columnspan=3, sticky='w', padx=2, pady=2)

    entry_fname = tk.Entry(input_panel, width=50)
    entry_fname.delete(0, tk.END)
    entry_fname.grid(row=1, column=4, columnspan=3, padx=2, pady=0)
    entry_fname.insert(0, project_name)
    # entry_fname.insert(0, 'test') ##for testing
    params.entry_fname = entry_fname

    # row 2 col 0-1
    L_OPTIONS = ('plini', 'plini_a', 'm:P:bus1')
    G_OPTIONS = ('pgini', 'pgini_a', 'm:P:bus1')
    GRF_OPTIONS = ('no geodata', 'GPS', 'graphic objects')
    US_OPTIONS = ('Deactivate', 'Drop', 'Nothing')
    SAVE_OPTIONS = ("JSON", "Excel")
    TAP_OPTIONS = ("nntap", "c:nntap")

    # PowerFactory variable to be exported as power value for loads
    var_p_loads = tk.StringVar(input_panel)
    var_p_loads.set(L_OPTIONS[0])
    tk.Label(input_panel, anchor='w', text='Loads P variable:').grid(row=2, column=0,
                                                                     sticky=tk.W, pady=0)
    opt_l = tk.OptionMenu(input_panel, var_p_loads, *L_OPTIONS)
    opt_l.grid(row=2, column=1, sticky="ew", pady=0)
    opt_l.config(width=10)

    params.pf_variable_p_loads = var_p_loads.get

    # row 3 col 0-1
    var_p_gen = tk.StringVar(input_panel)
    var_p_gen.set(G_OPTIONS[0])
    tk.Label(input_panel, anchor='w', text='Generators P variable:').grid(row=3, column=0,
                                                                          sticky=tk.W,
                                                                          pady=0)

    opt_g = tk.OptionMenu(input_panel, var_p_gen, *G_OPTIONS)
    opt_g.grid(row=3, column=1, sticky="ew", pady=0)
    opt_g.config(width=10)

    params.pf_variable_p_gen = var_p_gen.get

    # row 4 col 0-1
    var_graphics = tk.StringVar(input_panel)
    var_graphics.set(GRF_OPTIONS[0])
    tk.Label(input_panel, anchor='w', text='Collect coordinates from:').grid(row=4, column=0,
                                                                             sticky=tk.W,
                                                                             pady=0)

    opt_grf = tk.OptionMenu(input_panel, var_graphics, *GRF_OPTIONS)
    opt_grf.grid(row=4, column=1, sticky="ew", pady=0)
    opt_grf.config(width=10)

   
    # refresh graphics option menu
    try:
        conf_var_graphics(app, opt_grf, var_graphics)
    except Exception as err:
        logger.error('could not find network diagrams: %s' % err)
        pass

    params.flag_graphics = var_graphics.get

    # row 5 col 0-1
    var_us = tk.StringVar(input_panel)
    var_us.set(US_OPTIONS[0])
    tk.Label(input_panel, anchor='w', text='Unsupplied Elements:').grid(row=5, column=0,
                                                                             sticky=tk.W,
                                                                             pady=0)

    opt_us = tk.OptionMenu(input_panel, var_us, *US_OPTIONS)
    opt_us.grid(row=5, column=1, sticky="ew", pady=0)
    opt_us.config(width=10)
    
    params.handle_us = var_us.get

    # row 6 col 0-1
    save_as = tk.StringVar(input_panel)
    save_as.set(SAVE_OPTIONS[0])
    tk.Label(input_panel, anchor='w', text='Save As:').grid(row=6, column=0,
                                                                             sticky=tk.W,
                                                                             pady=0)

    opt_save = tk.OptionMenu(input_panel, save_as, *SAVE_OPTIONS)
    opt_save.grid(row=6, column=1, sticky="ew", pady=0)
    opt_save.config(width=10)
    
    params.save_as = save_as.get

    # row 7 col 0-1
    tap_grf = tk.StringVar(input_panel)
    tap_grf.set(TAP_OPTIONS[0])
    tk.Label(input_panel, anchor='w', text='Tap:').grid(row=7, column=0, sticky=tk.W, pady=0)

    opt_tap = tk.OptionMenu(input_panel, tap_grf, *TAP_OPTIONS)
    opt_tap.grid(row=7, column=1, sticky="ew", pady=0)
    opt_tap.config(width=10)
    
    params.tap_opt = tap_grf.get

    # row 8
    max_iter = tk.IntVar(input_panel)
    max_iter.set(30)
    tk.Label(input_panel, anchor='w', text='Max. iterations:').grid(row=8, column=0, sticky=tk.W, pady=0)

    iter_entry = tk.Entry(input_panel, width=8)
    iter_entry.delete(0, tk.END)
    iter_entry.grid(row=8, column=1, padx=2, pady=2, sticky=tk.W)
    iter_entry.insert(0, max_iter.get())
    # entry_fname.insert(0, 'test') ##for testing
    params.max_iter_entry = iter_entry

    # row 2 col 2-3
    PV_SL = tk.IntVar()
    tk.Checkbutton(input_panel, text="Export 'PV' buses as slack buses", variable=PV_SL).grid(row=2,
                                                                                              column=4,
                                                                                              sticky=tk.W)
    params.pv_as_slack = PV_SL.get


    EXPORT_CONTROLLER = tk.IntVar()
    EXPORT_CONTROLLER.set(1)
    tk.Checkbutton(input_panel, text="Export controllers", variable=EXPORT_CONTROLLER).grid(row=3,
                                                                                            column=4,
                                                                                            sticky=tk.W)
    params.export_controller = EXPORT_CONTROLLER.get

    CV_VERIFY = tk.IntVar()
    CV_VERIFY.set(1)
    tk.Checkbutton(input_panel, text="Verify conversion", variable=CV_VERIFY).grid(row=4, column=4,
                                                                                   sticky=tk.W)
    params.is_to_verify = CV_VERIFY.get

    RUN_DIAGNOSTIC = tk.IntVar()
    RUN_DIAGNOSTIC.set(1)
    tk.Checkbutton(input_panel, text="Diagnostic report", variable=RUN_DIAGNOSTIC).grid(row=5,
                                                                                        column=4,
                                                                                        sticky=tk.W)
    params.is_to_diagnostic = RUN_DIAGNOSTIC.get

    LOGGER_DEBUG = tk.IntVar()
    tk.Checkbutton(input_panel, text="Logger in debug mode", variable=LOGGER_DEBUG).grid(row=6,
                                                                                         column=4,
                                                                                         sticky=tk.W)
    params.is_debug = LOGGER_DEBUG.get

    REPLACE_ZERO_BRANCHES = tk.IntVar()
    REPLACE_ZERO_BRANCHES.set(1)
    tk.Checkbutton(input_panel, text="Replace low-impedance branches with switches",
                   variable=REPLACE_ZERO_BRANCHES).grid(row=7, column=4, sticky=tk.W)
    params.replace_zero_branches = REPLACE_ZERO_BRANCHES.get

    tk.Label(input_panel, anchor='w', text='Min. line R and X (Ohm):').grid(row=8, column=4, sticky=tk.W, pady=0)

    min_ohm = tk.Entry(input_panel, width=8)
    min_ohm.delete(0, tk.END)
    min_ohm.grid(row=8, column=5, padx=2, pady=2, sticky=tk.W)
    min_ohm.insert(0, "0.01")
    # entry_fname.insert(0, 'test') ##for testing
    params.min_ohm_entry = min_ohm
    
    # row 2 col 4
    stop_button = tk.Button(input_panel, text='Cancel', width=8,
                            command=lambda: cancel(app, input_panel))
    stop_button.grid(row=2, column=7, sticky='e', padx=4, pady=4)

    # app, get_dst_dir, input_panel, entry_fname, is_to_verify, is_debug, pv_as_slack,
    # pf_variable_p_loads, pf_variable_p_gen, flag_graphics
    start_button = tk.Button(input_panel, text='Export', width=8,
                             command=lambda: calc(app, **params))
    start_button.grid(row=1, column=7, sticky='e', padx=4, pady=0)

    input_panel.mainloop()

# if __name__ == '__main__':
#     import powerfactory
#
#     app = powerfactory.GetApplication()
#     app.Show()
#
#     logger, app_handler = logger_setup.setup_logger(app, 'INFO')
#     make_gui(app, 'test', browse_dst_test, calc_test)
# else:
#     logger = logging.getLogger(__name__)
#     # logger.setLevel('DEBUG')
