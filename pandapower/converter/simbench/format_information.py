from numpy import pi
from pandas import DataFrame

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

__author__ = 'smeinecke'


def sb2pp_base(variable="power"):
    """ converting factor from simbench data structure to pandapower:
        power: simbench in MVA - pandapower in kVA
        current: simbench in A, pandapower in kA
    """
    if variable == "power":
        return 1
    elif variable == "current":
        return 1e-3
    else:
        raise ValueError("The variable %s is unknown to sb2pp_base().")


def csv_tablenames(which):
    """
    Returns specific simbench csv format table names. which can be 'elements', res_elements,
    'profiles', 'types' or a list of these.
    """
    if isinstance(which, str):
        which = [which]
    csv_tablenames = []
    if 'elements' in which:
        csv_tablenames += ["ExternalNet", "Line", "Load", "Shunt", "Node", "Measurement",
                           "PowerPlant", "RES", "Storage", "Substation", "Switch", "Transformer",
                           "Transformer3W", "Coordinates"]
    if 'profiles' in which:
        csv_tablenames += ["LoadProfile", "PowerPlantProfile", "RESProfile", "StorageProfile"]
    if 'types' in which:
        csv_tablenames += ["LineType", "DCLineType", "TransformerType",
                           "Transformer3WType"]
    if 'cases' in which:
        csv_tablenames += ["StudyCases"]
    if 'res_elements' in which:
        csv_tablenames += ["NodePFResult"]
    return csv_tablenames


def _csv_table_pp_dataframe_correspondings(type_):
    csv_tablenames_ = csv_tablenames(['elements', 'types', 'res_elements'])
    # corresponding pandapower dataframe names
    pp_dfnames = ['ext_grid', 'line', 'load', 'shunt', 'bus', 'measurement', 'gen', 'sgen',
                  'storage', 'substation', 'switch', 'trafo', 'trafo3w', 'bus_geodata',
                  'std_types|line', 'dcline', 'std_types|trafo', 'std_types|trafo3w',
                  "res_bus"]
    # append table name lists by combinations of generating elements
    csv_tablenames_ += ['ExternalNet', 'ExternalNet', 'PowerPlant', 'PowerPlant', 'RES', 'RES',
                        'ExternalNet', 'ExternalNet', 'Line']
    pp_dfnames += ['gen', 'sgen', 'ext_grid', 'sgen', 'ext_grid', 'gen', 'ward', 'xward', 'dcline']
    assert len(csv_tablenames_) == len(pp_dfnames)
    if type_ is list:
        return csv_tablenames_, pp_dfnames
    elif type_ is str:
        return ["%s*%s" % (csv_tablename, pp_dfname) for csv_tablename, pp_dfname in
                zip(csv_tablenames_, pp_dfnames)]
    elif type_ is tuple:
        return [(csv_tablename, pp_dfname) for csv_tablename, pp_dfname in
                zip(csv_tablenames_, pp_dfnames)]
    elif type_ is DataFrame:
        # is like pd.DataFrame(_csv_table_pp_dataframe_correspondings(tuple))
        return DataFrame([(csv_tablename, pp_dfname) for csv_tablename, pp_dfname in
                          zip(csv_tablenames_, pp_dfnames)])
    elif isinstance(type_, str):
        if type_ in csv_tablenames_:
            corr = [pp for csv, pp in zip(csv_tablenames_, pp_dfnames) if csv == type_]
        else:
            corr = [csv for csv, pp in zip(csv_tablenames_, pp_dfnames) if pp == type_]
        if len(corr) == 1:
            return corr[0]
        else:
            return corr


def all_dtypes():
    """ This function returns a dict of all simbench csv file column dtypes. """
    dtypes = {
        "Coordinates": [object, float, float, object, int],
        "ExternalNet": [object]*3 + [float]*8 + [object, int],
        "Line": [object]*4 + [float, float, object, int],
        "LineType": [object] + [float]*4 + [object],
        "Load": [object]*3 + [float]*3 + [object, int],
        "LoadProfile": [object] + [float]*len(load_profiles_list(pq_both=True)),
        "Shunt": [object, object, float, float, float, int, object, int],
        "Node": [object]*2 + [float]*5 + [object]*3 + [int],
        "Measurement": [object]*5 + [int],
        "PowerPlant": [object]*5 + [float]*8 + [object, int],
        "PowerPlantProfile": [object],
        "RES": [object]*5 + [float]*3 + [object, int],
        "RESProfile": [object] + [float]*21,
        "Storage": [object]*4 + [float]*11 + [object, int],
        "StorageProfile": [object] + [float]*2,
        "Substation": [object, object, int],
        "Switch": [object]*4 + [int] + [object]*2 + [int],
        "Transformer": [object]*4 + [int, int, object, float, object, object, int],
        "TransformerType": [object] + [float]*8 + [int, object, float, float, int, int, int],
        "Transformer3W": [object]*5 + [float]*3 + [int, object, float, object, object, int],
        "Transformer3WType": [object] + [float]*16 + [int, object] + [float]*6 + [int]*9,
        "DCLineType": [object] + [float]*8,
        "StudyCases": [object] + [float]*6,
        "NodePFResult": [object, float, float, object, object, int]}
    return dtypes


def all_columns():
    """ This function returns a dict of all simbench csv file column names. """
    tablenames = {
        "Coordinates": ['id', 'x', 'y', 'subnet', 'voltLvl'],
        "ExternalNet": ['id', 'node', 'calc_type', 'dspf', 'pExtNet', 'qExtNet',
                        'pWardShunt', 'qWardShunt', 'rXWard',
                        'xXWard', 'vmXWard', 'subnet', 'voltLvl'],
        "Line": ['id', 'nodeA', 'nodeB', 'type', 'length', 'loadingMax', 'subnet', 'voltLvl'],
        "LineType": ['id', 'r', 'x', 'b', 'iMax', 'type'],
        "Load": ['id', 'node', 'profile', 'pLoad', 'qLoad', 'sR', 'subnet', 'voltLvl'],
        "LoadProfile": ['time'] + load_profiles_list(pq_both=True),
        "Shunt": ['id', 'node', 'p0', 'q0', 'vmR', 'step', 'subnet', 'voltLvl'],
        "Node": ['id', 'type', 'vmSetp', 'vaSetp', 'vmR', 'vmMin', 'vmMax', 'substation',
                 'coordID', 'subnet', 'voltLvl'],
        "Measurement": ['id', 'element1', 'element2', 'variable', 'subnet', 'voltLvl'],
        "PowerPlant": ['id', 'node', 'type', 'profile', 'calc_type', 'dspf', 'pPP', 'qPP', 'sR',
                       'pMin', 'pMax', 'qMin', 'qMax', 'subnet', 'voltLvl'],
        "PowerPlantProfile": ['id'],
        "RES": ['id', 'node', 'type', 'profile', 'calc_type', 'pRES', 'qRES', 'sR',
                'subnet', 'voltLvl'],
        "RESProfile": ['time'] + ["%s%i" % (b, x) for b in ["PV", "WP", "BM"] for x in
                                  range(1, 6)] + ['Hydro1', 'Hydro2', 'Waste1', 'Waste2',
                                                  'Gas1', 'Gas2'],
        "Storage": ['id', 'node', 'type', 'profile', 'pStor', 'qStor', 'chargeLevel', 'sR',
                    'max_e_mwh', 'efficiency_percent', 'self-discharge_percent_per_day',
                    'pMin', 'pMax', 'qMin', 'qMax', 'subnet', 'voltLvl'],
        "StorageProfile": ['time', 'PV_Battery', 'E_Mobility'],
        "Substation": ['id', 'subnet', 'voltLvl'],
        "Switch": ['id', 'nodeA', 'nodeB', 'type', 'cond', 'substation', 'subnet',
                   'voltLvl'],
        "Transformer": ['id', 'nodeHV', 'nodeLV', 'type', 'tappos',
                        'autoTap', 'autoTapSide', 'loadingMax', 'substation', 'subnet', 'voltLvl'],
        "TransformerType": ['id', 'sR', 'vmHV', 'vmLV', 'va0', 'vmImp', 'pCu', 'pFe',
                            'iNoLoad', 'tapable', 'tapside', 'dVm', 'dVa', 'tapNeutr',
                            'tapMin', 'tapMax'],
        "Transformer3W": ['id', 'nodeHV', 'nodeMV', 'nodeLV', 'type',
                          'tapposHV', 'tapposMV', 'tapposLV', 'autoTap', 'autoTapSide',
                          'loadingMax', 'substation', 'subnet', 'voltLvl'],
        "Transformer3WType": ['id', 'sRHV', 'sRMV', 'sRLV', 'vmHV', 'vmMV', 'vmLV',
                              'vaHVMV', 'vaHVLV', 'vmImpHVMV', 'vmImpHVLV', 'vmImpMVLV',
                              'pCuHV', 'pCuMV', 'pCuLV', 'pFe', 'iNoLoad', 'tapable',
                              'tapside', 'dVmHV', 'dVmMV', 'dVmLV', 'dVaHV', 'dVaMV',
                              'dVaLV', 'tapNeutrHV', 'tapNeutrMV', 'tapNeutrLV',
                              'tapMinHV', 'tapMinMV', 'tapMinLV', 'tapMaxHV', 'tapMaxMV',
                              'tapMaxLV'],
        "DCLineType": ['id', 'pDCLine', 'relPLosses', 'fixPLosses', 'pMax',
                       'qMinA', 'qMinB', 'qMaxA', 'qMaxB'],
        "StudyCases": ['Study Cases', 'pload', 'qload', 'Wind_p', 'PV_p', 'RES_p', 'Slack_vm'],
        "NodePFResult": ['node', 'vm', 'va', 'substation', 'subnet', 'voltLvl']
        }
    return tablenames


def get_dtypes(tablename):
    """ This function returns simbench csv file column dtypes for a given table name. """
    alldtypes = all_dtypes()
    if tablename in alldtypes.keys():
        if "Profile" in tablename:
            logger.debug("The returned dtypes list of %s is for 1000 profiles columns." % tablename)
        return alldtypes[tablename]
    else:
        raise ValueError('The tablename %s is unknown.' % tablename)


def get_columns(tablename):
    """ This function returns simbench csv file column names for a given table name. """
    allcolumns = all_columns()
    if tablename in allcolumns.keys():
        if "Profile" in tablename:
            logger.debug("The returned column list of %s is given for simbench " % tablename +
                         "dataset and may be incomplete")
        return allcolumns[tablename]
    else:
        raise ValueError('The tablename %s is unknown.' % tablename)


def _csv_pp_column_correspondings(tablename):
    """ Returns a list of tuples giving corresponding parameter names. The tuples are structured as:
        (column name in csv_table,
         column name in pp_df,
         factor to multiply csv column value to receive pp column value) """
    tuples = [
        # Node and node names
        ("node", "bus", None), ("nodeA", "from_bus", None), ("nodeB", "to_bus", None),
        ("nodeHV", "hv_bus", None), ("nodeMV", "mv_bus", None), ("nodeLV", "lv_bus", None),
        ("sR", "sn_mva", sb2pp_base()), ("vmR", "vn_kv", None), ("nodeA", "bus", None),
        ("nodeB", "element", None), ("vmMin", "min_vm_pu", None), ("vmMax", "max_vm_pu", None),
        # Line, LineType and DCLineType
        ("length", "length_km", None), ("pDCLine", "p_mw", sb2pp_base()),
        ("relPLosses", "loss_percent", None), ("fixPLosses", "loss_mw", sb2pp_base()),
        ("qMaxA", "max_q_from_mvar", sb2pp_base()), ("qMinA", "min_q_from_mvar", sb2pp_base()),
        ("qMaxB", "max_q_to_mvar", sb2pp_base()), ("qMinB", "min_q_to_mvar", sb2pp_base()),
        ("r", "r_ohm_per_km", None), ("x", "x_ohm_per_km", None),
        ("b", "c_nf_per_km", 1e3 / (2*pi*50)), ("iMax", "max_i_ka", sb2pp_base("current")),
        ("loadingMax", "max_loading_percent", None),
        # Ward and xWard
        ("pExtNet", "ps_mw", sb2pp_base()),
        ("qExtNet", "qs_mvar", sb2pp_base()), ("cond", "closed", None),
        ("pWardShunt", "pz_mw", sb2pp_base()),
        ("qWardShunt", "qz_mvar", sb2pp_base()), ("rXWard", "r_ohm", None),
        ("xXWard", "x_ohm", None), ("vmXWard", "vm_pu", None),
        # Measurement
        ("variable", "type", None),
        # Storage
        ("eStore", "max_e_mwh", sb2pp_base()), ("etaStore", "efficiency_percent", None),
        ("sdStore", "self-discharge_percent_per_day", sb2pp_base()),
        ("rStore", "resistance_ohm", None), ("chargeLevel", "soc_percent", 100),
        # NodePFResult
        ("vm", "vm_pu", None), ("va", "va_degree", None),
        # TransformerType
        ("vmHV", "vn_hv_kv", None), ("vmMV", "vn_mv_kv", None), ("vmLV", "vn_lv_kv", None),
        ("pFe", "pfe_kw", None), ("iNoLoad", "i0_percent", None),
        ("tappos", "tap_pos", None), ("tapside", "tap_side", None),
        ("vmImp", "vk_percent", None), ("dVm", "tap_step_percent", None),
        ("va0", "shift_degree", None), ("tapNeutr", "tap_neutral", None),
        ("tapMin", "tap_min", None), ("tapMax", "tap_max", None),
        # Transformer3WType
        ("sRHV", "sn_hv_mva", sb2pp_base()), ("sRMV", "sn_mv_mva", sb2pp_base()),
        ("sRLV", "sn_lv_mva", sb2pp_base()), ("vaHVMV", "shift_mv_degree", None),
        ("vaHVLV", "shift_lv_degree", None), ("vmImpHVMV", "vk_hv_percent", None),
        ("vmImpHVLV", "vk_mv_percent", None), ("vmImpMVLV", "vk_lv_percent", None),
        ("dVmHV", "tap_step_percent", None), ("tapNeutrHV", "tap_neutral", None),
#        ("dVmMV", "xxxxxxxx", None), ("dVmLV", "xxxxxxxx", None),
#        ("dVaHV", "xxxxxxxx", None), ("dVaMV", "xxxxxxxx", None),
#        ("dVaLV", "xxxxxxxx", None),
#        ("tapNeutrMV", "xxxxxxxx", None), ("tapNeutrLV", "xxxxxxxx", None),
        ("tapMinHV", "tap_min", None), ("tapMaxHV", "tap_max", None)
#        ("tapMinMV", "xxxxxxxx", None), ("tapMinLV", "xxxxxxxx", None),
#        ("tapMaxMV", "xxxxxxxx", None), ("tapMaxLV", "xxxxxxxx", None)
        # cosidered by _add_vm_va_setpoints_to_buses() and _add_phys_type_and_vm_va_setpoints_to_generation_element_tables():
        # ("vmSetp", "vm_pu", None), ("vaSetp", "va:degree", None),
        ]

    # --- add "pLoad", "qLoad" respectively "pPP", "qPP" or others, according to tablename
    shortcuts = {"PowerPlant": "PP", "ExternalNet": "ExtNet", "Storage": "Stor", "Shunt": "0"}
    if tablename in shortcuts.keys():
        shortcut = shortcuts[tablename]
    else:
        shortcut = tablename
    tuples += [("p"+shortcut, "p_mw", sb2pp_base()), ("q"+shortcut, "q_mvar", sb2pp_base()),
               ("pMax", "max_p_mw", sb2pp_base()), ("pMin", "min_p_mw", sb2pp_base()),
               ("qMax", "max_q_mvar", sb2pp_base()), ("qMin", "min_q_mvar", sb2pp_base())]
    if tablename == "LineType":
        tuples += [("type", "type", None)]
    else:
        tuples += [("type", "std_type", None)]
    return tuples


def load_profiles_list(pq_both=False):
    """ Returns a list of load profiles. If pq_both is True, every entry of the list is appended by
    both, "_pload" and "_qload". """
    profiles = []
    profiles = ['G0-A', 'G0-M', 'G1-A', 'G1-B', 'G1-C', 'G2-A', 'G3-A',
                'G3-M', 'G3-H', 'G4-A', 'G4-B', 'G4-M', 'G4-H', 'G5-A', 'G6-A',
                'H0-A', 'H0-B', 'H0-C', 'H0-H', 'H0-G', 'H0-L',
                'L0-A', 'L1-A', 'L2-A', 'L2-M',
                'BL-H', 'WB-H']
    if not pq_both:
        return profiles
    # else:
    profiles = [prof + pq for pq in ["_pload", "_qload"] for prof in profiles]
    return profiles


def _correct_calc_type(csv_data):
    """ Returns a list of possible calc_types. """
    correct = ["vavm", "pvm", "pq", "Ward", "xWard"]
    misspelled = [['vmva', 'VaVm', 'VmVa', 'slack', 'sl', 'Slack'],
                  ['vmp', 'PVm', 'VmP', 'pv', 'PV', 'Pv'],
                  ["PQ", "Pq"],
                  ["ward", "WARD"],
                  ["xward", "XWard", "XWARD"]]
    for tablename in ["ExternalNet", "PowerPlant", "RES"]:
        if csv_data[tablename].shape[0]:
            for miss, corr in zip(misspelled, correct):
                is_miss = csv_data[tablename].calc_type.isin(miss)
                csv_data[tablename].calc_type.loc[is_miss] = corr
