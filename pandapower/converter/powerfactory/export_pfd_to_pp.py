import pandapower as pp
from .echo_off import echo_off, echo_on
from .logger_setup import AppHandler, set_PF_level
from .pf_export_functions import run_load_flow, create_network_dict
from .pp_import_functions import from_pf
from .run_import import choose_imp_dir, clear_dir, prj_dgs_import, prj_import

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def from_pfd(app, prj_name: str, path_dst=None, pv_as_slack=False, pf_variable_p_loads='plini',
             pf_variable_p_gen='pgini', flag_graphics='GPS', tap_opt='nntap',
             export_controller=True, handle_us="Deactivate", is_unbalanced=False):
    """

    Args:
        prj_name: Name (”Project”), full qualified name (”Project.IntPrj”) or full qualified path
            (”nUsernProject.IntPrj”) of a project.
        path_dst: Destination for the export of .p file (full file path)
        pv_as_slack: whether "PV" nodes are imported as "Slack" nodes
        pf_variable_p_loads: PowerFactory variable for generators: "plini", "plini_a", "m:P:bus1"
        pf_variable_p_gen: PowerFactory variable for generators: "pgini", "pgini_a", "m:P:bus1"
        flag_graphics: whether geodata comes from graphic objects (*.IntGrf) or GPS
        tap_opt: PowerFactory variable for tap position: "nntap" or "c:nntap"
        export_controller: whether to create and export controllers
        handle_us (str, "Deactivate"): What to do with unsupplied buses -> Can be "Deactivate", "Drop" or "Nothing"

    Returns: pandapower network "net" and controller, saves pp-network as .p file at path_dst

    """

    logger.debug('started')
    echo_off(app)
    user = app.GetCurrentUser()
    logger.debug('user: %s' % user)

    res = app.ActivateProject(prj_name)
    if res == 1:
        raise RuntimeError('Project %s could not be found or activated' % prj_name)

    prj = app.GetActiveProject()

    logger.info('gathering network elements')
    dict_net = create_network_dict(app, flag_graphics)
    pf_load_flow_failed = run_load_flow(app)
    logger.info('exporting network to pandapower')
    app.SetAttributeModeInternal(1)
    net = from_pf(dict_net=dict_net, pv_as_slack=pv_as_slack,
                  pf_variable_p_loads=pf_variable_p_loads,
                  pf_variable_p_gen=pf_variable_p_gen, flag_graphics=flag_graphics,
                  tap_opt=tap_opt, export_controller=export_controller, handle_us=handle_us, is_unbalanced=is_unbalanced)
    # save a flag, whether the PowerFactory load flow failed
    app.SetAttributeModeInternal(0)
    net["pf_converged"] = not pf_load_flow_failed

    logger.info(net)

    prj.Deactivate()
    echo_on(app)
    if path_dst is not None:
        pp.to_json(net, path_dst)
        logger.info('saved net as %s', path_dst)
    return net


# experimental feature
def execute(app, path_src, path_dst, pv_as_slack, scale_feeder_loads=False, var_load='plini',
            var_gen='pgini', flag_graphics='GPS'):
    """
    Executes import of a .dgs file, runs load flow, and exports net as .p
    Args:
        path_src: full path to the input .dgs file
        path_dst: full path to the result .p file
        pv_as_slack: whether "PV" nodes are to be imported as "Slack
        scale_feeder_loads: whether loads are to be scaled according to feeder scaling factor

    Returns: net

    """
    logger.debug('started')
    echo_off(app)
    prj = import_project(path_src, app)

    logger.info('activating project')

    prj.Activate()
    trafo_name, trafo_desc = _check_network(app)

    logger.info('gathering network elements')
    dict_net = create_network_dict(app, flag_graphics=flag_graphics)
    run_load_flow(app, scale_feeder_loads, gen_scaling=0)
    logger.info('exporting network to pandapower')
    app.SetAttributeModeInternal(1)
    net = from_pf(dict_net, pv_as_slack=pv_as_slack, pf_variable_p_loads=var_load,
                  pf_variable_p_gen=var_gen, flag_graphics=flag_graphics)
    app.SetAttributeModeInternal(0)

    logger.info(net)

    prj.Deactivate()
    echo_on(app)

    pp.to_pickle(net, path_dst)

    return net, trafo_name, trafo_desc


def import_project(path_src, app, name="Import" , import_folder="", template=None, clear_import_folder=False):
    user = app.GetCurrentUser()
    logger.debug('user: %s' % user)

    imp_dir = choose_imp_dir(user, import_folder)
    logger.info('Auxiliary import folder: %s', imp_dir)

    if clear_import_folder:
        clear_dir(imp_dir)

    # PF import object
    # com_import = app.GetFromStudyCase('ComPfdimport')
    if '.dgs' in path_src:
        com_import = app.GetFromStudyCase('ComImport')
        logger.info('Importing .dgs project %s' % path_src)
        if template is not None:
            app.ActivateProject(template)
            template = app.GetActiveProject()
        prj_dgs_import(com_import, imp_dir, path_src, name, template)
    elif '.pfd' in path_src:
        com_import = app.GetFromStudyCase('ComPfdimport')
        logger.info('Importing .pfd project %s' % path_src)
        #prj_import(com_import, imp_dir, path_src)
        com_import.g_file = path_src
        com_import.g_target = imp_dir
        # somehow this is not always the case
        assert com_import.g_target == imp_dir
        assert com_import.g_file == path_src
        com_import.Execute()

    try:
        prj = imp_dir.GetContents()[0]
    except:
        raise RuntimeError('could not get the project - failed at import?')

    return prj


def _check_network(app):
    """
    Used in VNS Hessen to make configs and run additional checks on the networks that are
    imported from .dgs
    raises error if the network does not fit to the criteria
    :return: None
    """

    # setting triggers out of service
    triggers = app.GetCalcRelevantObjects('*.SetTime, *.SetTrigger')
    if len(triggers) > 0:
        for t in triggers:
            t.outserv = 1

    # checking if there are feeders in network
    feeder_folder = app.GetDataFolder('ElmFeeder')
    feeders = feeder_folder.GetContents()
    if len(feeders) == 0:
        raise RuntimeError('no feeders found in network!')

    # check if there is External grid and Transformer
    obj_types = ['ElmXnet', 'ElmTr2']
    for ot in obj_types:
        elms = app.GetCalcRelevantObjects('*.%s' % ot)
        if len(elms) == 0:
            raise RuntimeError('there are no elements of type %s in net' % ot)

    # check if there are more than 3 buses
    buses = app.GetCalcRelevantObjects('*.ElmTerm')
    if len(buses) <= 3:
        raise RuntimeError('less equal than 3 buses in net')

    # check if there is more than 1 load
    loads = app.GetCalcRelevantObjects('*.ElmLod*')
    if len(loads) <= 1:
        raise RuntimeError('less equal than 1 load in net')

    trafos = app.GetCalcRelevantObjects('*.ElmTr2')
    if len(trafos) > 1:
        raise RuntimeError('more tan 1 trafo in net')

    for load in loads:
        load_name = load.loc_name
        if 'RLM' in load_name and load.i_scale != 0:
            logger.warning('load %s.%s i_scale' % (load.loc_name, load.GetClassName()))
            load.i_scale = 0
            # raise Exception('Adjusted by load scaling set to True')

    return trafos[0].loc_name, trafos[0].desc


if __name__ == '__main__':
    try:
        import powerfactory as pf

        app = pf.GetApplication()
        app_handler = AppHandler(app, freeze_app_between_messages=True)
        logger.addHandler(app_handler)
        set_PF_level(logger, app_handler, 'INFO')
    except:
        pass

