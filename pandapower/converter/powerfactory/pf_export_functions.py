try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

def create_network_dict(app, flag_graphics='GPS'):
    # elements to be exported from PowerFactory
    set_object_extentions = {
        # node elements:
        'ElmTerm',
        'ElmXnet',
        'ElmLod',
        'ElmLodlv',
        'ElmLodlvp',
        'ElmLodmv',
        'ElmGenstat',
        'ElmPvsys',
        'ElmSym',
        'ElmAsm',
        'ElmShnt',
        'ElmVac',

        # branch elements:
        'ElmLne',
        'ElmCoup',
        'RelFuse',
        'ElmZpu',
        'ElmSind',
        'StaSwitch',
        'ElmTr2',
        'ElmTr3'

        # we don't gather types anymore, because they are not collected for elements that are out
        #  of service
        #  types:
        # 'TypLne',
        # 'TypTr2',
        # 'TypTr3'
    }

    # here define all element types that have to be configured to have MW as power values
    elm_units = {
        'ElmLod': ['W', 'var', 'VA'],
        'ElmLodlv': ['W', 'var', 'VA'],
        'ElmLodlvp': ['W', 'var', 'VA'],
        'ElmLodmv': ['W', 'var', 'VA'],
        'ElmGenstat': ['W', 'var', 'VA'],
        'ElmPvsys': ['W', 'var', 'VA'],
        'ElmXnet': ['W', 'var', 'VA'],
        'ElmSym': ['W', 'var', 'VA'],
        'ElmAsm': ['W', 'var', 'VA'],
        'ElmShnt': ['W', 'var', 'VA'],
        'ElmZpu': ['W', 'var', 'VA'],
        'ElmSind': ['W', 'var', 'VA', 'V'],
        'ElmVac': ['W', 'var', 'VA'],
        'ElmTr2': ['W', 'var'],
        'ElmTr3': ['W', 'var'],
        'TypTr2': ['W', 'var', 'VA'],
        'TypTr3': ['W', 'var', 'VA'],
        'TypLne': ['A', 'm']
    }

    # make all values in MW
    logger.info('applying unit settings')
    # echo_on(app)
    # apply_unit_settings(app, elm_units, 'M')
    # echo_off(app)

    dict_net = {}

    grid = app.GetSummaryGrid()
    dict_net['ElmNet'] = grid
    dict_net['global_parameters'] = get_global_parameters(app)

    logger.info('collecting network elements')
    for obj in set_object_extentions:
        if obj == 'ElmTerm':
            dict_net[obj] = app.GetCalcRelevantObjects(obj, 1, 0, 1)
        else:
            dict_net[obj] = app.GetCalcRelevantObjects(obj)

    if flag_graphics not in ['GPS', 'no geodata']:
        logger.info('gathering graphic objects')
        dia_name = flag_graphics
        gather_graphic_objects(app, dict_net, dia_name)

    dict_net['lvp_params'] = get_lvp_params(app)

    return dict_net


def get_lvp_params(app):
    """
    Gather parameters for partial LV loads (loads that model an LV customer, the load elements are
    stored in a line element without having own buses in PowerFactory, so the lines in pandapower
    must be split at those points and new buses must be created).
    """
    com_ldf = app.GetFromStudyCase('ComLdf')

    lvp_params = {
        'iopt_sim': com_ldf.iopt_sim,
        'scPnight': com_ldf.scPnight,
        'Sfix': com_ldf.Sfix,
        'cosfix': com_ldf.cosfix,
        'Svar': com_ldf.Svar,
        'cosvar': com_ldf.cosvar,
        'ginf': com_ldf.ginf,
        'i_volt': com_ldf.i_volt
    }

    return lvp_params


def get_global_parameters(app):
    prj = app.GetActiveProject()
    settings = prj.pPrjSettings
    base_sn_mva = settings.Sbase

    # global load and generation scaling
    com_ldf = app.GetFromStudyCase('ComLdf')
    global_load_scaling = com_ldf.scLoadFac * 1e-2
    global_generation_scaling = com_ldf.scGenFac * 1e-2
    global_motor_scaling = com_ldf.scMotFac * 1e-2

    global_parameters = {
        'base_sn_mva': base_sn_mva,
        'global_load_scaling': global_load_scaling,
        'global_generation_scaling': global_generation_scaling,
        'global_motor_scaling': global_motor_scaling,
        'iopt_tem': com_ldf.iopt_tem  # calculate load flow at 20 °C or at max. temperature
    }
    return global_parameters


def gather_graphic_objects(app, dict_net, dia_name):
    dia_folder = app.GetProjectFolder('dia')
    dia_grid = dia_folder.GetContents(dia_name + '.IntGrfnet')
    grf_objs = []
    for d in dia_grid:
        grf_objs.extend(d.GetContents('*.IntGrf'))
    logger.debug('collected graphic objects: %s' % grf_objs)
    dict_net['graphics'] = {grf.pDataObj: grf for grf in grf_objs}


def apply_unit_settings(app, elm_units, exponent):
    prj = app.GetActiveProject()
    # don't forget to define the elements to apply the unit settings to!
    setup_project_power_exponent(prj, exponent)

    for key, val in elm_units.items():
        for unit in val:
            setup_unit_exponents(prj, key, unit, exponent)

    # special case for c_nf_per_km in nf and I in kA:
    for elm in ['TypLne', 'ElmLne', 'ElmLnesec']:
        setup_unit_exponents(prj, elm, 'F/km', 'n')
        setup_unit_exponents(prj, elm, 'F', 'n')
        setup_unit_exponents(prj, elm, 'A', 'k')

    # to make PowerFactory apply the new settings
    prj.Deactivate()
    prj.Activate()


def run_load_flow(app, scale_feeder_loads=False, load_scaling=None, gen_scaling=None,
                  motor_scaling=None):
    """
    :param app: PowerFactory Application object
    :param scale_feeder_loads: if loads have to be scaled according to the feeder scaling factor
    :param motor_scaling: Load flow parameter in PowerFactory ("Load/Generation scaling")
    :param gen_scaling: Load flow parameter in PowerFactory ("Load/Generation scaling")
    :param load_scaling: Load flow parameter in PowerFactory ("Load/Generation scaling")
    :return: None
    """

    com_ldf = app.GetFromStudyCase('ComLdf')

    # com_ldf.iopt_net = 0
    # com_ldf.iopt_at = 1
    # com_ldf.iopt_pq = 0
    if scale_feeder_loads:
        logger.debug('scale_feeder_loads is True')
        com_ldf.iopt_fls = 1

    # com_ldf.errlf = 0.001
    # com_ldf.erreq = 0.01

    if com_ldf.iopt_sim == 1:
        logger.warning(f'Calculation method probabilistic loadflow of lv-loads is activated!'
                       f' The validation will not succeed.')
    if load_scaling is not None:
        logger.debug('scaling loads at %.2f' % load_scaling)
        com_ldf.scLoadFac = load_scaling
    if gen_scaling is not None:
        logger.debug('scaling generators at %.2f' % gen_scaling)
        com_ldf.scGenFac = gen_scaling
    if motor_scaling is not None:
        logger.debug('scaling motors at %.2f' % motor_scaling)
        com_ldf.scMotFac = motor_scaling

    logger.info('---------------------------------------------------------------------------------')
    logger.info('PowerFactory load flow settings:')
    # Active power regulation
    logger.info('Calculation method (AC balanced, AC unbalanced): %s' % com_ldf.iopt_net)
    logger.info(f'Calculation method probabilistic loadflow of lv-loads: {com_ldf.iopt_sim}')
    logger.info('Automatic tap adjustment of phase shifters: %s' % com_ldf.iPST_at)
    logger.info('Consider active power limits: %s' % com_ldf.iopt_plim)
    # Voltage and reactive power regulation
    logger.info('Automatic tap adjustment of transformers: %s' % com_ldf.iopt_at)
    logger.info('Automatic tap adjustment of shunts: %s' % com_ldf.iopt_asht)
    logger.info('Consider reactive power limits: %s' % com_ldf.iopt_lim)
    # Temperature dependency
    logger.info('Calculate at 20 °C/max. temperature: %s' % com_ldf.iopt_tem)
    # Load options
    logger.info('Consider voltage dependency of loads: %s' % com_ldf.iopt_pq)
    logger.info('Feeder load scaling: %s' % com_ldf.iopt_fls)
    logger.info('---------------------------------------------------------------------------------')

    res = com_ldf.Execute()
    # print("pf result", res)
    if res != 0:
        logger.error('Load flow failed due to divergence of %s loops' % ['inner', 'outer'][res - 1])
#        raise RuntimeError(
#            'Load flow failed due to divergence of %s loops' % ['inner', 'outer'][res - 1])
    return res


def setup_unit_exponents(prj, elm_class, unit, exponent):
    cdigexp = 'M' if unit in ['var', 'W', 'VA'] else 'u' if unit in ['F/km', 'F'] else 'k' if unit=='A' else ''
    logger.debug('setting unit exponents: %s, %s, %s, %s' % (elm_class, unit, cdigexp, exponent))
    settings_folder = prj.GetContents('*.SetFold')[0]
    units_folder = settings_folder.GetContents('*.IntUnit')

    if len(units_folder) == 0:
        units_folder = settings_folder.CreateObject('IntUnit', 'Units')
    else:
        units_folder = units_folder[0]

    object_name = elm_class + '-' + unit
    # delete old objects if there are any
    trash = units_folder.GetContents(object_name)
    for item in trash:
        item.Delete()

    unit_setting = units_folder.CreateObject('SetVariable', object_name)
    try:  # version-specific expectation at this point
        unit_setting.filtclass = elm_class if isinstance(elm_class, list) else [elm_class]
    except TypeError:
        unit_setting.filtclass = elm_class[0] if isinstance(elm_class, list) else elm_class
    unit_setting.digunit = unit
    # if unit in ['W', 'var', 'VA']:
    unit_setting.SetAttribute('cdigexp', cdigexp)
    unit_setting.cuserexp = exponent


def setup_project_power_exponent(prj, exponent):
    prj_settings = prj.pPrjSettings
    study_cases = prj.GetContents('*.IntCase', 1)

    prj_settings.cspqexp = exponent  # power MW
    prj_settings.clenexp = 'k'  # length km

    # for base Mva:
    setup_unit_exponents(prj, 'SetPrj', 'VA', 'M')

    for object in study_cases:
        object.cpowexp = exponent
        object.cpexpshc = exponent
        object.campexp = 'k'  # current kA
