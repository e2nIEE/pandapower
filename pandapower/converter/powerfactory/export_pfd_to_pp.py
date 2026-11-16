# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging
from typing import Any, Literal

from pandapower.file_io import to_json, to_pickle
from pandapower.network import pandapowerNet

from .echo_off import echo_off, echo_on
from .pf_export_functions import create_network_dict, run_load_flow
from .pp_import_functions import from_pf
from .run_import import choose_imp_dir, clear_dir, prj_dgs_import

logger = logging.getLogger(__name__)


def from_pfd(
    app,
    prj_name: str,
    script_name: str | None = None,
    script_settings: dict | None = None,
    path_dst: str | None = None,
    pv_as_slack: bool = False,
    pf_variable_p_loads: str = "plini",
    pf_variable_p_gen: str = "pgini",
    flag_graphics: Literal["GPS", "IntGrf", "no geodata"] = "GPS",
    tap_opt: str = "nntap",
    export_controller: bool = True,
    handle_us: Literal["Deactivate", "Drop", "Nothing"] = "Deactivate",
    is_unbalanced: bool = False,
    create_sections: bool = True,
    export_pf_ZoneArea: bool = False,
    sc_name: str | None = None,
) -> pandapowerNet:
    """
    Import a DIgSILENT PowerFactory project into a pandapower network.

    Optionally executes a DPL script and a load flow in PowerFactory before converting the active project to a
    pandapower network. The resulting pandapower network can optionally be stored as a JSON (.p) file. A flag
    indicating whether the PowerFactory load flow converged is stored in ``net["pf_converged"]``.

    Parameters:
        app: PowerFactory application object returned by ``GetApplication()``.
        prj_name: Name (``"Project"``), fully qualified name (``"Project.IntPrj"``) or fully qualified path
            (``"nUsernProject.IntPrj"``) of the PowerFactory project to be activated.
        script_name: Name of the DPL script that shall be executed prior to the import to pandapower. If
            None, no script is executed.
        script_settings: Dictionary of arguments for the DPL script, e.g. ``{"Script variable name in
            PF": value}``. Keys must match the script's parameter names in PowerFactory. Ignored if ``script_name`` is
            None.
        path_dst: Destination file path for exporting the pandapower network as JSON (.p file) using
            :func:`pandapower.to_json`. If None, the network is not written to disk.
        pv_as_slack: If True, PowerFactory PV nodes are imported as slack nodes in pandapower.
        pf_variable_p_loads: PowerFactory variable name used for active power of loads, e.g. ``"plini"``,
            ``"plini_a"`` or ``"m:P:bus1"``. Default is ``"plini"``.
        pf_variable_p_gen: PowerFactory variable name used for active power of generators, e.g. ``"pgini"``,
            ``"pgini_a"`` or ``"m:P:bus1"``. Default is ``"pgini"``.
        flag_graphics: Source for geodata. If ``"GPS"``, geodata comes from GPS information. If
            ``"IntGrf"``, geodata is taken from graphic objects (``*.IntGrf``). Default is ``"GPS"``.
        tap_opt: PowerFactory variable for tap position, e.g. ``"nntap"`` or ``"c:nntap"``. Default is
            ``"nntap"``.
        export_controller: If True, creates and exports controllers to the pandapower network.
        handle_us: Action to be taken for unsupplied buses:
            - ``"Deactivate"``: deactivate unsupplied buses and connected elements
            - ``"Drop"``: remove unsupplied buses from the network
            - ``"Nothing"``: leave unsupplied buses unchanged
            Default is ``"Deactivate"``.
        is_unbalanced: If True, import the network as an unbalanced system.
        create_sections: If True, create network sections during the import.
        export_pf_ZoneArea: If True, export Zone and Area information from PowerFactory to the pandapower
            buses.
        sc_name: Name of the PowerFactory Study Case (scenario) to activate before running the load flow
            and exporting. If None, the currently active Study Case is used.

    Returns:
        net: The imported pandapower network. Contains the key ``"pf_converged"`` indicating whether the
            PowerFactory load flow converged (True/False).

    Raises:
        RuntimeError: If the specified PowerFactory project cannot be found or activated.
        UserWarning: If the provided script settings are inconsistent with the script definition or the script execution
            fails.

    Notes:
        If ``path_dst`` is not None, the resulting pandapower network is additionally written
        to disk as a JSON file using :func:`pandapower.to_json`.
    """
    if script_settings is None:
        script_settings = {}

    logger.debug("started")
    echo_off(app)
    user = app.GetCurrentUser()
    logger.debug(f"user: {user}")

    res = app.ActivateProject(prj_name)
    if res == 1:
        raise RuntimeError(f"Project {prj_name} could not be found or activated")

    prj = app.GetActiveProject()

    # scenario | Study Cases | Betriebsfall
    scenario_name_list = [sc.loc_name for sc in app.GetProjectFolder("scen").GetContents()]
    logger.info(f"Available 'Study Cases': {scenario_name_list}")

    if sc_name is not None and sc_name in scenario_name_list:
        sc = app.GetProjectFolder("scen").GetContents(sc_name)[0]
        sc.Activate()
        logger.info(f"Study Case {app.GetActiveScenario().loc_name} is currently active!")
    else:
        logger.info("No Study Case is currently active!")

    logger.info("gathering network elements")
    dict_net = create_network_dict(app, flag_graphics)

    if script_name is not None:
        script = get_script(user, script_name)
        script_values = script.IntExpr
        for parameter_name, new_value in script_settings.items():
            pos = script.IntName.index(parameter_name)
            if script_values[pos] != new_value:
                script_values[pos] = new_value
            else:
                continue
        script.SetAttribute("IntExpr", script_values)
        pf_script_execution_failed = script.Execute()
        if pf_script_execution_failed != 0:
            logger.error("Script execution failed.")
        pf_load_flow_failed = run_load_flow(app)
        if pf_load_flow_failed != 0:
            logger.error("Load flow failed after executing DPL script.")
    else:
        pf_load_flow_failed = run_load_flow(app)

    logger.info("exporting network to pandapower")
    app.SetAttributeModeInternal(1)
    net = from_pf(
        dict_net=dict_net,
        pv_as_slack=pv_as_slack,
        pf_variable_p_loads=pf_variable_p_loads,
        pf_variable_p_gen=pf_variable_p_gen,
        flag_graphics=flag_graphics,
        tap_opt=tap_opt,
        export_controller=export_controller,
        handle_us=handle_us,
        is_unbalanced=is_unbalanced,
        create_sections=create_sections,
        export_pf_ZoneArea=export_pf_ZoneArea,
    )
    # save a flag, whether the PowerFactory load flow failed
    app.SetAttributeModeInternal(0)
    net["pf_converged"] = not pf_load_flow_failed

    logger.info(net)

    prj.Deactivate()
    echo_on(app)
    if path_dst is not None:
        to_json(net, path_dst)
        logger.info("saved net as %s", path_dst)
    return net


def get_script(user, script_name):
    script = None

    for obj in user.GetContents():
        if obj.loc_name == script_name:
            script = obj
            break

    if script is None:
        raise UserWarning(f"Could not find script with name {script_name}.")

    return script


# experimental feature
def execute(
    app,
    path_src,
    path_dst,
    pv_as_slack,
    scale_feeder_loads=False,
    var_load="plini",
    var_gen="pgini",
    flag_graphics: Literal["GPS", "IntGrf", "no geodata"] = "GPS",
    create_sections=True,
) -> tuple[pandapowerNet, Any, Any]:
    """
    Executes import of a .dgs file, runs load flow, and exports net as .p

    Parameters:
        app:
        path_src: full path to the input .dgs file
        path_dst: full path to the result .p file
        pv_as_slack: whether "PV" nodes are to be imported as "Slack
        scale_feeder_loads: whether loads are to be scaled according to feeder scaling factor
        var_load:
        var_gen:
        flag_graphics:
        create_sections:

    Returns:
        a tuple of the imported network, trafo name and trafo description
    """
    logger.debug("started")
    echo_off(app)
    prj = import_project(path_src, app)

    logger.info("activating project")

    prj.Activate()
    trafo_name, trafo_desc = _check_network(app)

    logger.info("gathering network elements")
    dict_net = create_network_dict(app, flag_graphics=flag_graphics)
    run_load_flow(app, scale_feeder_loads, gen_scaling=0)
    logger.info("exporting network to pandapower")
    app.SetAttributeModeInternal(1)
    net = from_pf(
        dict_net,
        pv_as_slack=pv_as_slack,
        pf_variable_p_loads=var_load,
        pf_variable_p_gen=var_gen,
        flag_graphics=flag_graphics,
        create_sections=create_sections,
    )
    app.SetAttributeModeInternal(0)

    logger.info(net)

    prj.Deactivate()
    echo_on(app)

    to_pickle(net, path_dst)

    return net, trafo_name, trafo_desc


def import_project(path_src, app, name="Import", import_folder="", template=None, clear_import_folder=False):
    user = app.GetCurrentUser()
    logger.debug(f"user: {user}")

    imp_dir = choose_imp_dir(user, import_folder)
    logger.info("Auxiliary import folder: %s", imp_dir)

    if clear_import_folder:
        clear_dir(imp_dir)

    # PF import object
    # com_import = app.GetFromStudyCase('ComPfdimport')
    if ".dgs" in path_src:
        com_import = app.GetFromStudyCase("ComImport")
        logger.info(f"Importing .dgs project {path_src}")
        if template is not None:
            app.ActivateProject(template)
            template = app.GetActiveProject()
        prj_dgs_import(com_import, imp_dir, path_src, name, template)
    elif ".pfd" in path_src:
        com_import = app.GetFromStudyCase("ComPfdimport")
        logger.info(f"Importing .pfd project {path_src}")
        # prj_import(com_import, imp_dir, path_src)
        com_import.g_file = path_src
        com_import.g_target = imp_dir
        # somehow this is not always the case
        assert com_import.g_target == imp_dir
        assert com_import.g_file == path_src
        com_import.Execute()

    try:
        prj = imp_dir.GetContents()[0]
    except Exception:  # noqa: BLE001
        raise RuntimeError("could not get the project - failed at import?")

    return prj


def _check_network(app):
    """
    Used in VNS Hessen to make configs and run additional checks on the networks that are imported from .dgs raises
    error if the network does not fit to the criteria

    Returns:
        None
    """

    # setting triggers out of service
    triggers = app.GetCalcRelevantObjects("*.SetTime, *.SetTrigger")
    if len(triggers) > 0:
        for t in triggers:
            t.outserv = 1

    # checking if there are feeders in network
    feeder_folder = app.GetDataFolder("ElmFeeder")
    feeders = feeder_folder.GetContents()
    if len(feeders) == 0:
        raise RuntimeError("no feeders found in network!")

    # check if there is External grid and Transformer
    obj_types = ["ElmXnet", "ElmTr2"]
    for ot in obj_types:
        elms = app.GetCalcRelevantObjects(f"*.{ot}")
        if len(elms) == 0:
            raise RuntimeError(f"there are no elements of type {ot} in net")

    # check if there are more than 3 buses
    buses = app.GetCalcRelevantObjects("*.ElmTerm")
    if len(buses) <= 3:
        raise RuntimeError("less equal than 3 buses in net")

    # check if there is more than 1 load
    loads = app.GetCalcRelevantObjects("*.ElmLod*")
    if len(loads) <= 1:
        raise RuntimeError("less equal than 1 load in net")

    trafos = app.GetCalcRelevantObjects("*.ElmTr2")
    if len(trafos) > 1:
        raise RuntimeError("more tan 1 trafo in net")

    for load in loads:
        load_name = load.loc_name
        if "RLM" in load_name and load.i_scale != 0:
            logger.warning(f"load {load.loc_name}.{load.GetClassName()} i_scale")
            load.i_scale = 0
            # raise Exception('Adjusted by load scaling set to True')

    return trafos[0].loc_name, trafos[0].desc
