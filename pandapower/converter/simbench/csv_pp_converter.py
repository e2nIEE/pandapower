"""
This is the csv_pp_converter for the simbench project.
pandapower 2.0.1 <-> simbench format (reasled status from 25.04.2019)
"""

import os
import pandas as pd
import numpy as np
from copy import deepcopy
from packaging import version
import pandapower as pp
from pandapower.plotting import create_generic_coordinates

try:
    import pplog as logging
except ImportError:
    import logging

from pandapower.converter.simbench.auxiliary import *
from pandapower.converter.simbench.format_information import *
from pandapower.converter.simbench.format_information import _correct_calc_type, \
    _csv_table_pp_dataframe_correspondings, _csv_pp_column_correspondings
from pandapower.converter.simbench.read_and_write import *
from pandapower.converter.simbench.read_and_write import _init_csv_tables
from pandapower.converter.simbench.pp_net_manipulation import _extend_pandapower_net_columns, \
    _add_dspf_calc_type_and_phys_type_columns, _add_vm_va_setpoints_to_buses, \
    _prepare_res_bus_table, replace_branch_switches, create_branch_switches, _add_coordID, \
    _set_vm_setpoint_to_trafos
from pandapower.converter.simbench.csv_data_manipulation import *
from pandapower.converter.simbench.csv_data_manipulation import _extend_coordinates_to_node_shape, \
    _sort_switch_nodes_and_prepare_element_and_et, \
    _ensure_single_switch_at_aux_node_and_copy_vm_setp, \
    _add_phys_type_and_vm_va_setpoints_to_element_tables, \
    _ensure_safe_csv_ids, _correct_autoTapSide_of_nonTapTrafos
from pandapower.converter.simbench.pp_net_manipulation import *

logger = logging.getLogger(__name__)

__author__ = 'smeinecke'


def csv2pp(path, sep=';', add_folder_name=None, nrows=None, no_generic_coord=False):
    """
    Conversion function from simbench csv format to pandapower.

    INPUT:
        **path** (str) - path of a folder which includes all csv files

        **sep** (str) - gives the seperator of the csv files

    OPTIONAL:
        **add_folder_name** (str, None) - name of the subfolder in path, in which the csv data is
            located.

        **nrows** (int, None) - number of rows to be read for profiles. If None, all rows will be
            read.

        **no_generic_coord** (bool, False) - if True, no generic coordinates are created in case of
            missing geodata.

    OUTPUT:
        **net** (pandapowerNet) - the created pandapower net from csv files data

    EXAMPLE:
        import simbench_converter as sc
        import os
        sc_path = os.path.split(sc.__file__)[0]
        path = os.path.join(sc_path, "test", 'test_network_1')
        test_network_1 = sc.csv2pp(path=path, sep=';')
    """
    # --- create repository for csv files
    if not isinstance(path, str):
        logger.error('The path must be given as a string')
    path = os.path.join(path, add_folder_name) if add_folder_name else path

    # --- read csv_data
    csv_data = read_csv_data(path, sep, nrows=nrows)

    # run net creation
    net = csv_data2pp(csv_data)

    # ensure geodata
    if not no_generic_coord and any(pd.isnull(net.bus_geodata.x) | pd.isnull(net.bus_geodata.y)):
        del net.bus_geodata
        create_generic_coordinates(net)

    return net


def csv_data2pp(csv_data):
    """ Internal functionality of csv2pp, but with a given dict of csv_data as input. """
    # --- initializations
    csv_data = deepcopy(csv_data)
    net = pp.create_empty_network()

    # --- extend pandapower net columns to store csv information that are unused in pandapower but
    # possible informative or helpful for reconversion
    _extend_pandapower_net_columns(net)

    # --- correction of csv_data and preperation for converting
    _ensure_safe_csv_ids(csv_data)
    reindex_dict_dataframes(csv_data)
    _ensure_single_switch_at_aux_node_and_copy_vm_setp(csv_data, new_type_name="multi_auxiliary")
    _convert_measurement(csv_data)
    _sort_switch_nodes_and_prepare_element_and_et(csv_data)
    convert_node_type(csv_data)
    _correct_calc_type(csv_data)
    _correct_autoTapSide_of_nonTapTrafos(csv_data)
    _add_phys_type_and_vm_va_setpoints_to_element_tables(csv_data)
    _extend_coordinates_to_node_shape(csv_data)
    convert_line_type_acronym(csv_data)

    # --- convert csv_data
    _csv_profiles_to_pp(net, csv_data)
    if "StudyCases" in csv_data.keys() and isinstance(csv_data["StudyCases"], pd.DataFrame) and \
       csv_data["StudyCases"].shape[0]:
        net["loadcases"] = csv_data["StudyCases"]
    _csv_types_to_pp1(net, csv_data)
    _multi_parameter_determination(csv_data)
    _convert_elements_and_types(csv_data, net)
    create_branch_switches(net)
    net.bus.type.loc[net.bus.type == "multi_auxiliary"] = "auxiliary"
    _set_vm_setpoint_to_trafos(net, csv_data)
    _csv_types_to_pp2(net)
    ensure_bus_index_columns_as_int(net)

    return net


def pp2csv(net1, path, export_pp_std_types=False, sep=';', exclude_table=set(), nrows=None,
           mode='w', keep='last', drop_inactive_elements=True,
           round_qLoad_by_voltLvl=False, reserved_aux_node_names=None):
    """
    Conversion function from pandapower to simbench csv format.

    INPUT:
        **net1** (pandapowerNet) - pandapower net that should be converted to simbench csv format

        **path** (str) - folder path, the csv files should be stored into

    OPTIONAL:
        **export_pp_std_types** (False, boolean) - defines whether unused pandapower standard types
            should be stored to simbench csv format

        **sep** (';', str) - gives the seperator of the csv files

        **exclude_table** ({}, set) - set of table names that should not be converted to csv files.
            Overwrites the possibly given list export_results.

        **nrows** (int, None) - number of rows to be write to csv for profiles. If None, all rows
            will be written.

        **mode** ('append', str) - If csv files already exists in the given path, they can
            be appended if mode is 'a' or 'append_unique' they can be replaced if
            mode is 'w'. In case of 'append_unique', only data with unique name, voltLvl and
            subnet are kept (which is controlled by parameter keep).

        **keep** ('last', str) - decides which duplicated data is kept in case of
            mode == "append_unique"

        **drop_inactive_elements** (True, boolean) - Per default, only suppliable, active elements
            get converted to csv files. The user may change this behaviour by setting
            drop_inactive_elements to False.

        **round_qLoad_by_voltLvl** (False, boolean) - If True, qLoad is rounded to variating
            accurancy: EHV: 1kVAr, HV: 100VAr, MV: 100VAr, LV: 1VAr"

        **reserved_aux_node_names** (None, set) - set of strings which are not allowed to be used as
            auxiliary node names

    OUTPUT:
        **reserved_aux_node_names** (set) - reserved_aux_node_names appended by created auxiliary
            node names. Is only returned if given as input

    EXAMPLE:
        import simbench_converter as sc
        import pandapower.networks as pn
        net1 = pn.simple_four_bus_system()
        test_network_1 = sc.csv2pp(net1, "folder", sep=';')
    """
    # --- determine the highest existing coordinate number for case of mode == "append_unique"
    highest_existing_coordinate_number = -1
    aux_nodes_are_reserved = reserved_aux_node_names is not None
    reserved_aux_node_names = reserved_aux_node_names if aux_nodes_are_reserved else set()
    if mode == "append_unique":
        coords = read_csv_data(path, sep, 'Coordinates')
        if coords.shape[0]:
            idx = coords.id.str.startswith("coord_")
            coords_split = coords.id.loc[idx].str.split("coord_")
            coords_values = coords_split.str[-1].astype(int)
            highest_existing_coordinate_number = coords_values.max()

    # --- create csv data and res data as dicts of DataFrames
    csv_data, reserved_aux_node_names = pp2csv_data(
        net1, export_pp_std_types=export_pp_std_types,
        drop_inactive_elements=drop_inactive_elements,
        highest_existing_coordinate_number=highest_existing_coordinate_number,
        round_qLoad_by_voltLvl=round_qLoad_by_voltLvl,
        reserved_aux_node_names=reserved_aux_node_names)

    # --- export the grid data dict DataFrames to csv files
    write2csv(path, csv_data, mode=mode, sep=sep, float_format='%g',
                 keys=set(csv_data.keys())-exclude_table, keep=keep, nrows=nrows)

    if aux_nodes_are_reserved:
        return reserved_aux_node_names


def pp2csv_data(net1, export_pp_std_types=False, drop_inactive_elements=True,
                highest_existing_coordinate_number=-1, round_qLoad_by_voltLvl=False,
                reserved_aux_node_names=None):
    """ Internal functionality of pp2csv, but without writing the determined dict to csv files.
    For parameter explanations, look at pp2csv() docstring. """
    # --- initializations
    net = deepcopy(net1)  # necessary because in net will be changed in converter function
    csv_data = _init_csv_tables(['elements', 'profiles', 'types', 'res_elements'])
    aux_nodes_are_reserved = reserved_aux_node_names is not None

    # --- net data preparation for converting
    _extend_pandapower_net_columns(net)
    if drop_inactive_elements:
        # attention: trafo3ws are not considered in current version of drop_inactive_elements()
        pp.drop_inactive_elements(net, respect_switches=False)
    check_results = pp.deviation_from_std_type(net)
    if check_results:
        logger.warning("There are deviations from standard types in elements: " +
                       str(["%s" % elm for elm in check_results.keys()]) + ". Only the standard " +
                       "type values are converted to csv.")
    convert_parallel_branches(net)
    if net.bus.shape[0] and not net.bus_geodata.shape[0] or (
            net.bus_geodata.shape[0] != net.bus.shape[0]):
        logger.info("Since there are no or incomplete bus_geodata, generic geodata are assumed.")
        net.bus_geodata.drop(net.bus_geodata.index, inplace=True)
        create_generic_coordinates(net)
    merge_busbar_coordinates(net)
    move_slack_gens_to_ext_grid(net)

    scaling_is_not_1 = []
    for i in pp.pp_elements():
        # prevent elements without name
        net[i] = ensure_full_column_data_existence(net, i, 'name')
        avoid_duplicates_in_column(net, i, 'name')
        # log scaling factor different from 1
        if "scaling" in net[i].columns:
            if not np.allclose(net[i]["scaling"].values, 1):
                scaling_is_not_1 += [i]
    if len(scaling_is_not_1):
        logger.warning("In elements " + str(scaling_is_not_1) + "'scaling' differs from 1, which" +
                       " is not converted.")
    # log min_e_mwh
    if not np.allclose(net.storage["min_e_mwh"].dropna().values, 0.):
        logger.warning("Storage parameter 'min_e_mwh' is not converted but differs from 0.")

    # further preparation
    provide_subnet_col(net)
    provide_voltLvl_col(net)
    provide_substation_cols(net)
    convert_node_type(net)
    _add_dspf_calc_type_and_phys_type_columns(net)
    _add_vm_va_setpoints_to_buses(net)
    _prepare_res_bus_table(net)
    reserved_aux_node_names = replace_branch_switches(net, reserved_aux_node_names)
    _convert_measurement(net)
    _add_coordID(net, highest_existing_coordinate_number)

    # --- convert net
    _pp_profiles_to_csv(net, csv_data)
    if "loadcases" in net:
        csv_data["StudyCases"] = net["loadcases"]
    else:
        csv_data["StudyCases"] = pd.DataFrame()
    _pp_types_to_csv1(net, export_pp_std_types)
    _multi_parameter_determination(net)
    _convert_elements_and_types(net, csv_data)
    _pp_types_to_csv2(csv_data)

    if round_qLoad_by_voltLvl:
        _round_qLoad_by_voltLvl(csv_data)

    # --- post_conversion_checks
    _check_id_voltLvl_subnet(csv_data)

    if aux_nodes_are_reserved:
        return csv_data, reserved_aux_node_names
    else:
        return csv_data


def _round_qLoad_by_voltLvl(csv_data):
    for voltLvl, decimals in zip([2, 4, 6, 7], [2, 3, 4, 6]):
        this_voltLvl = csv_data["Load"].voltLvl <= voltLvl
        csv_data["Load"]["qLoad"].loc[this_voltLvl] = np.around(csv_data["Load"]["qLoad"].loc[
            this_voltLvl].values, decimals)


def _check_id_voltLvl_subnet(csv_data):
    # --- ensure every data has an id and test for missing voltLvl and subnet
    for tablename in csv_tablenames(['types']):
        _log_nan_col(csv_data, tablename, "id")
    for tablename in csv_tablenames('elements'):
        _log_nan_col(csv_data, tablename, "id")
        if tablename != "Coordinates":
            _log_nan_col(csv_data, tablename, "voltLvl")
            _log_nan_col(csv_data, tablename, "subnet")


def _log_nan_col(csv_data, tablename, col):
    nans = sum(csv_data[tablename][col].isnull())
    if nans:
        logger.info("There are %i %s without %s data." % (nans, tablename, col))


def _is_pp_type(data):
    return "name" in data.keys()  # used instead of isinstance(data, pp.auxiliary.pandapowerNet)


def convert_node_type(data):
    """ Convertes csv_data["Node"] to short type names and net["bus"] to long type names. """
    if _is_pp_type(data):
        full_names = {"b": "busbar", "m": "muffe", "n": "node", "db": "double busbar"}
        for short, long in full_names.items():
            fit = data["bus"].type == short
            data["bus"].type.loc[fit] = long
    else:
        not_auxiliary = ~data["Node"].type.str.contains("auxiliary")
        if sum(not_auxiliary):
            space_split = data["Node"].type[not_auxiliary].str.split(" ", expand=True)
            if 1 in space_split.columns:
                data["Node"].type.loc[not_auxiliary] = space_split[0].str[0] + \
                    space_split[1].str[0].fillna("")
            else:
                data["Node"].type.loc[not_auxiliary] = space_split[0].str[0]


def convert_line_type_acronym(data):
    """ Convertes type of csv_data["LineType"] or net.std_types["line"]: cs <-> cable, ol <-> ohl
    """
    if _is_pp_type(data):
        if "type" not in data.std_types["line"].columns:
            data.std_types["line"]["type"] = ""
        data.std_types["line"]["type"] = data.std_types["line"].type.replace(
            {"cs": "cable", "ol": "ohl"})
    else:
        if "type" not in data["LineType"].columns:
            data["LineType"]["type"] = ""
        data["LineType"]["type"] = data["LineType"].type.replace({"cable": "cs", "ohl": "ol"})


def _convert_measurement(data):
    """ Converts the measurement columns "side", "element", "measurement_type", respectively
        "element1", "element2", "variable". """
    rename_dict = {"side": "element1", "element": "element2", "measurement_type": "variable"}
    if _is_pp_type(data):
        data["measurement"].rename(columns=rename_dict, inplace=True)
        for element_type in ["trafo", "line", "bus"]:
            idx = data["measurement"].index[data["measurement"]["element_type"] == element_type]
            if element_type in ["trafo", "line"]:
                this_branch_measurements = data["measurement"].loc[idx]
                for side in this_branch_measurements["element1"].unique():
                    idx_side = this_branch_measurements.index[
                        this_branch_measurements["element1"] == side]
                    data["measurement"]["element1"].loc[idx_side] = data.bus.name.loc[data[
                        element_type][side+"_bus"].loc[data["measurement"]["element2"].loc[
                            idx_side]]].values
                data["measurement"]["element2"].loc[idx] = data[element_type].name.loc[data[
                    "measurement"]["element2"].loc[idx]].values
            else:
                data["measurement"]["element1"].loc[idx] = data.bus.name.loc[data["measurement"][
                    "element2"][idx]].values
                data["measurement"]["element2"].loc[idx] = np.nan
    else:
        _sort_measurement_elements(data)
        rename_dict = {y: x for x, y in rename_dict.items()}
        data["Measurement"].rename(columns=rename_dict, inplace=True)
        idx_trafo = data["Measurement"].index[data["Measurement"]["element"].isin(data[
            "Transformer"]["id"])].astype(int)
        idx_line = data["Measurement"].index[data["Measurement"]["element"].isin(data[
            "Line"]["id"])].astype(int)
        idx_bus = data["Measurement"].index.difference(idx_line | idx_trafo).astype(int)
        idx_bus = data["Measurement"].index.difference(idx_line | idx_trafo)
        data["Measurement"]["element_type"] = "bus"
        data["Measurement"]["element_type"].loc[idx_trafo] = "trafo"
        data["Measurement"]["element_type"].loc[idx_line] = "line"
        data["Measurement"]["element"].loc[idx_trafo] = data["Transformer"].index[
            idx_in_2nd_array(data["Measurement"]["element"].loc[idx_trafo].values,
                                data["Transformer"]["id"].values)]
        data["Measurement"]["element"].loc[idx_line] = data["Line"].index[idx_in_2nd_array(
            data["Measurement"]["element"].loc[idx_line].values, data["Line"]["id"].values)]
        data["Measurement"]["element"].loc[idx_bus] = data["Node"].index[idx_in_2nd_array(
            data["Measurement"]["side"].loc[idx_bus].values, data["Node"]["id"].values)]
        bus_indices = data["Node"].index[idx_in_2nd_array(data["Measurement"][
            "side"].values, data["Node"]["id"].values)]
        bus_names = data["Node"]["id"].loc[bus_indices]
        data["Measurement"]["side"] = np.nan
        are_hv = ensure_iterability(bus_names.loc[bus_names.index[idx_trafo]].values == data[
            "Transformer"]["nodeHV"].loc[data["Measurement"]["element"].loc[idx_trafo]].values)
        data["Measurement"].loc[idx_trafo, "side"] = ["hv" if is_hv else "lv" for is_hv in are_hv]
        are_from = ensure_iterability(bus_names.loc[bus_names.index[idx_line]].values == data[
                "Line"]["nodeA"].loc[data["Measurement"]["element"].loc[idx_line]].values)
        data["Measurement"].loc[idx_line, "side"] = ["from" if is_from else "to" for is_from in
                                                     are_from]


def _sort_measurement_elements(csv_data):
    """ Switches element1 and element2 data in measurement table, if element2 is in node names.
    As a result, no node names are in element2 column. """
    idx_el1_is_node = csv_data["Measurement"].element1.isin(csv_data["Node"].id)
    idx_el2_is_node = csv_data["Measurement"].element2.isin(csv_data["Node"].id)
    idx_both_is_node = idx_el1_is_node & idx_el2_is_node
    if sum(idx_both_is_node):
        raise ValueError("In measurement table, element1 and element2 are node names in the" +
                         "indices: " + str(list(idx_both_is_node)))

    el1_data = deepcopy(csv_data["Measurement"].element1[idx_el2_is_node])
    csv_data["Measurement"].element1.loc[idx_el2_is_node] = csv_data["Measurement"].element2[
        idx_el2_is_node]
    csv_data["Measurement"].element2.loc[idx_el2_is_node] = el1_data


def _csv_profiles_to_pp(net, csv_data):
    """ Creates a dict 'profiles' in pp net. """
    for csv_name, pp_name in zip(csv_tablenames("profiles"), pp_profile_names()):
        if csv_data[csv_name].shape[0] > 0:
            net['profiles'][pp_name] = csv_data[csv_name]


def _pp_profiles_to_csv(net, csv_data):
    """ Copies each dataframe from net.profiles into the 'csv_data'. """
    for csv_name, pp_name in zip(csv_tablenames("profiles"), pp_profile_names()):
        try:
            csv_data[csv_name] = net['profiles'][pp_name]
        except (KeyError, AttributeError):
            # csv_data[csv_name] = pd.DataFrame()  # redundant since _init_csv_tables() is performed
            logger.info('The profiles of %s could not be stored.' % pp_name)


def _csv_types_to_pp1(net, csv_data):
    """ In general, only line, trafo und trafo3w are saved in pp net.std_types
        (see _csv_types_to_pp2()). Consequently, this function prepares converting type data of
        dcline and storage.
        Additionally, lower case tapside values are ensured.
    """
    # --- splitting Line table into a table with dcline data and line data
    idx_lines = csv_data["Line"].index[csv_data["Line"].type.isin(csv_data["LineType"].id)]
    idx_dclines = csv_data["Line"].index[csv_data["Line"].type.isin(csv_data["DCLineType"].id)]
    missing = csv_data["Line"].index.difference(idx_lines | idx_dclines)
    if len(missing):
        raise ValueError("In Line table, the types of these line indices misses in LineType and " +
                         "DCLineType table: " + str(list(missing)))
    if len(idx_lines & idx_dclines):
        raise ValueError("In Line table, the types of these line indices occur in LineType and " +
                         "DCLineType table: " + str(list(idx_lines & idx_dclines)))
    csv_data["Line*line"] = csv_data["Line"].loc[idx_lines]
    csv_data["Line*dcline"] = csv_data["Line"].loc[idx_dclines]

    # copy dicts from net.std_types to dataframes in net with name "std_types|element"
    for key in net.std_types.keys():
        net["std_types|%s" % key] = pd.DataFrame([], columns=pd.DataFrame(net.std_types[
            key]).T.columns)

    # ensure lower case tapside values
    csv_data["TransformerType"]["tapside"] = csv_data["TransformerType"]["tapside"].str.lower()
    csv_data["Transformer3WType"]["tapside"] = csv_data["Transformer3WType"]["tapside"].str.lower()


def _csv_types_to_pp2(net):
    """ Moves all DataFrames in net named with "std_types|" to net.std_types as dict and
        loads type data into element tables. """
    element_types = ["line", "trafo", "trafo3w"]
    for element_type in element_types:
        # --- copy type data from dataframes like net["LineType*std_types|line"] into net.std_types
        # --- as dict
        tablename = "std_types|" + element_type
        type_table = net[tablename]
        merged_type_table = merge_dataframes([
            type_table, pd.DataFrame(net.std_types[element_type]).T], sort_index=False,
            sort_column=False)
        if element_type == "line":
            _assume_cs_ohl_line_type(merged_type_table)
        data = {idx: dict(merged_type_table.loc[idx]) for idx in merged_type_table.index}
        pp.create_std_types(net, data, element_type)

        # --- load type data into element tables
        type_columns = net[element_type].columns.intersection(net[tablename].columns)
        uniq_types = net[element_type].std_type.unique()
        for uniq_type in uniq_types:
            this_type_elm = net[element_type].std_type == uniq_type
            net[element_type].loc[this_type_elm, type_columns] = merged_type_table.loc[
                uniq_type, type_columns].values.reshape(1, -1).repeat(sum(this_type_elm), axis=0)

        # --- delete dataframes like net["LineType*std_types|line"]
        del net[tablename]


def _assume_cs_ohl_line_type(line_types_df):
    """ Assumes some known types as cable or overhead lines. """
    cable_name_parts = ["NA", "NY", "N2XS"]
    is_cs = np.array([False]*line_types_df.shape[0])
    for name_part in cable_name_parts:
        is_cs |= pd.Series(line_types_df.index).str.contains(name_part)
#    if "type" not line_types_df.columns:
#        line_types_df["type"] = np.nan
    is_null = line_types_df.type.isnull().values
    line_types_df.type.loc[line_types_df.index[is_null & is_cs]] = "cs"

    # assume all other types as ohl
    line_types_df.type.fillna("ol", inplace=True)


def _pp_types_to_csv1(net, export_pp_std_types):
    """ Ensures that all line, trafo, trafo3w, storage and dcline have a type.
        Provides DataFrames in net for line, trafo and trafo3w elements considering
        export_pp_std_types. """

    # --- add 'std_type' column to net.dcline and net.storage if missing
    for elm in ["dcline"]:
        # 'std_type' will be considered for converter.
        if "std_type" not in net[elm].columns:
            if "type" not in net[elm].columns:
                net[elm]["std_type"] = np.nan
            else:
                # if 'std_type' does not exist but 'type', here 'type' is renamed into 'std_type'
                new_index_dict = {(i): (i if i != "type" else "std_type") for i in net[elm].columns}
                net[elm].rename(columns=new_index_dict, inplace=True)

    # --- add new std_types to line, trafo, trafo3w, storage and dcline tables if missing
    type_params = {
        "line": ["r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km", "g_us_per_km", "max_i_ka"],
        "trafo": ["sn_mva", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent", "pfe_kw",
                  "i0_percent", "shift_degree"],
        "trafo3w": ["sn_hv_mva", "sn_mv_mva", "sn_lv_mva", "vn_hv_kv", "vn_mv_kv", "vn_lv_kv",
                    "vk_hv_percent", "vk_mv_percent", "vk_lv_percent",
                    "vkr_hv_percent", "vkr_mv_percent", "vkr_lv_percent",
                    "pfe_kw", "i0_percent", "shift_mv_degree", "shift_lv_degree"],
        "dcline": ["p_mw", "loss_percent", "loss_mw", "vm_from_pu", "vm_to_pu"]}
    for elm in type_params.keys():
        elms_without_type = net[elm][pd.isnull(net[elm].std_type)]
        uni_dupl_dict = get_unique_duplicated_dict(elms_without_type, subset=type_params[elm])
        for uni, dupl in uni_dupl_dict.items():
            new_typename = net[elm].name.loc[uni] + '_type'
            net[elm].std_type.loc[[uni]+dupl] = new_typename
            if elm in ["line", "trafo", "trafo3w"]:
                pp.create_std_type(net, dict(net[elm].loc[uni, type_params[elm]].T), new_typename,
                                   element=elm, overwrite=False)

    # --- determine line, trafo and trafo3w typenames to be converted,
    # --- considering export_pp_std_types. changes net.std_types dicts into dataframes
    dummy_net = pp.create_empty_network()
    pp_elms_with_type = ["line", "trafo", "trafo3w"]
    for elm in pp_elms_with_type:
        if export_pp_std_types:
            typenames2convert = set(net.std_types[elm].keys())
        else:
            pp_typenames = set(dummy_net.std_types[elm].keys())
            unused_pp_typenames = pp_typenames - set(net[elm].std_type.unique())
            typenames2convert = set(net.std_types[elm].keys()) - unused_pp_typenames
        net.std_types[elm] = pd.DataFrame(net.std_types[elm]).T.loc[typenames2convert].reset_index()
        net.std_types[elm].rename(columns={"index": "std_type"}, inplace=True)

    convert_line_type_acronym(net)


def _pp_types_to_csv2(csv_data):
    """ Sorts Type tables alphabetical and reduces StorageType and DCLineType to unique values.
        Additionally, upper case tapside values are ensured. """
    for tablename in csv_tablenames('types'):
        csv_data[tablename] = csv_data[tablename].sort_values(by=["id"]).reset_index(drop=True)
        if tablename in ["DCLineType"]:
            csv_data[tablename] = csv_data[tablename].loc[~csv_data[tablename]["id"].duplicated()]
    csv_data["TransformerType"]["tapside"] = csv_data["TransformerType"]["tapside"].str.upper()
    csv_data["Transformer3WType"]["tapside"] = csv_data["Transformer3WType"]["tapside"].str.upper()


def _multi_parameter_determination(data):
    """ Adds columns for parameters which must be calculated with respect to multiple parameters """
    trafo3w_tablename = "trafo3w" if _is_pp_type(data) else "Transformer3WType"
    if data[trafo3w_tablename].shape[0]:
        logger.warning("Since there are different Transformer3w models, be aware that there is " +
                       "an information loss to pandapower and missing information for simbench " +
                       "csv format.")
    vkr_3w = ["vkr_%sv_percent" % s for s in ["h", "m", "l"]]
    sn_3w = ["sn_%sv_mva" % s for s in ["h", "m", "l"]]
    pCu_3w = ["pCu%sV" % s for s in ["H", "M", "L"]]
    sR_3w = ["sR%sV" % s for s in ["H", "M", "L"]]
    if _is_pp_type(data):
        # switch
        data["switch"]["closed"] = data["switch"]["closed"].astype(int)
        # trafo type
        data["std_types"]["trafo"]["pCu"] = data["std_types"]["trafo"].vkr_percent * data[
            "std_types"]["trafo"].sn_mva *1e3 / 100
        tapable = (data["std_types"]["trafo"].tap_min != data["std_types"]["trafo"].tap_neutral) | \
            (data["std_types"]["trafo"].tap_max != data["std_types"]["trafo"].tap_neutral)
        data["std_types"]["trafo"]["tapable"] = tapable.astype(int)
        data["std_types"]["trafo"]["tapable"] = tapable.astype(int)
        # trafo3w type
        for vkr, sn, pCu in zip(vkr_3w, sn_3w, pCu_3w):
            data["std_types"]["trafo3w"][pCu] = data["std_types"]["trafo3w"][vkr] * data[
                "std_types"]["trafo3w"][sn] *1e3 / 100
    else:
        # switch
        data["Switch"]["cond"] = data["Switch"]["cond"].astype(bool)
        # trafo type
        table = "TransformerType"
        data[table]["vkr_percent"] = 100 * data[table]["pCu"] / (data[table]["sR"]*1e3)
        if (data[table]["dVa"] != 0).any():
            logger.warning("TransformerType parameter 'dVa' is not converted to pandapower.")
        if not pd.isnull(data["Transformer"].autoTap).all():
            logger.debug("Transformer parameter 'autoTap' is not converted to pandapower.")
        idx_not_tapable = data[table].index[~data[table]["tapable"].astype(bool)]
        if len(idx_not_tapable):
            logger.info("tapMin and tapMax are set equal to tapNeutr for all transformers which " +
                        "are not tabable.")
            data[table].tapMin.loc[idx_not_tapable] = data[table].tapNeutr.loc[idx_not_tapable]
            data[table].tapMax.loc[idx_not_tapable] = data[table].tapNeutr.loc[idx_not_tapable]
        # trafo3w type
        table = "Transformer3WType"
        for vkr, sR, pCu in zip(vkr_3w, sR_3w, pCu_3w):
            data[table][vkr] = 100 * data[table][pCu] / (data[table][sR] * sb2pp_base())


def _convert_elements_and_types(input_data, output_data):
    """ Converts elements and type data.
        First, the tables are renamed by x*y where x is the csv_tablename and y the pp element. If
        it is about a pp type from net.std_types y is std_types|element.
        Furthermore the tables are split if there are multiple tables of the output format the
        data must be integrated in.
        Second, the columns of the tables are renamed by the output column names and the values are
        multiplied with fixed factors.
        Third, the bus reference, which is done by names in simbench csv format and by index in
        pandapower, is translated-
        Finally, the converted data is copied to the output net dict.

        Info: At this point of conversion of pp2csv(), alle type tables are already split. Within
        csv2pp(), storage and storagetype (same for dcline) are merged into only one table.
    """
    _rename_and_split_input_tables(input_data)
    # from here, all relevant information are stored in input_data[corr_str]
    _rename_and_multiply_columns(input_data)
    _replace_name_index(input_data)
    if _is_pp_type(output_data):
        _merge_dcline_and_storage_type_and_element_data(input_data)
    _copy_data(input_data, output_data)


def _rename_and_split_input_tables(data):
    """ Splits the tables of ExternalNet, PowerPlant, RES, ext_grid, gen, sgen and name the df
        according to _csv_table_pp_dataframe_correspondings(). """
    # --- initilizations
    split_gen = ["ext_grid", "gen", "sgen"] if _is_pp_type(data) else [
        "ExternalNet", "PowerPlant", "RES"]
    split_gen_col = "phys_type" if _is_pp_type(data) else "calc_type"
    split_Line = [] if _is_pp_type(data) else ["Line"]
    split_ppelm_into_type_and_elm = ["dcline"] if _is_pp_type(data) else []
    input_elm_col = 1 if _is_pp_type(data) else 0
    output_elm_col = 0 if _is_pp_type(data) else 1
    corr_df = _csv_table_pp_dataframe_correspondings(pd.DataFrame)
    corr_df[2] = corr_df[0] + "*" + corr_df[1]

    # all elements, which need to be converted to multiple output element tables, (dupl) need to be
    # treated specially, thus must be in split lists
    dupl = corr_df[input_elm_col][corr_df[input_elm_col].duplicated()]
    assert dupl.isin(split_gen+split_Line+split_ppelm_into_type_and_elm).all()

    # -- start renaming and in case of dupl also splitting
    for idx in corr_df.index:
        # get actual element tablenames
        input_elm = corr_df.loc[idx, input_elm_col]
        output_elm = corr_df.loc[idx, output_elm_col]
        corr_str = corr_df.loc[idx, 2]

        if input_elm in split_gen:
            data[corr_str] = data[input_elm][data[input_elm][split_gen_col] == _get_split_gen_val(
                output_elm)]
        elif input_elm in split_Line:
            continue  # already done in _csv_types_to_pp1()
        elif input_elm in split_ppelm_into_type_and_elm:
            if "Type" in output_elm:
                is_uniq = ~data[input_elm].std_type.duplicated()
                data[corr_str] = data[input_elm].loc[is_uniq]
            else:
                data[corr_str] = data[input_elm]
        else:  # rename data.keys for all elements without special treatment
            if corr_str not in data.keys():
                if input_elm in data.keys():
                    data[corr_str] = data.pop(input_elm)
                elif _is_pp_type(data) and "std_types" in input_elm:
                    data[corr_str] = pd.DataFrame(data["std_types"][input_elm[10:]])
                else:
                    data[corr_str] = pd.DataFrame()


def _get_split_gen_val(element):
    to_csv = {x: x for x in ["ExternalNet", "PowerPlant", "RES"]}
    to_pp = {"ext_grid": "vavm", "gen": "pvm", "sgen": "pq", "ward": "Ward", "xward": "xWard"}
    if element in to_csv.keys():
        return to_csv[element]
    else:
        return to_pp[element]


def _rename_and_multiply_columns(data):
    """ Renames the columns of all dataframes as needed in output data. """
    to_rename_and_multiply_tuples = _get_parameters_to_rename_and_multiply()
    for corr_str, tuples in to_rename_and_multiply_tuples.items():
        # --- remove "type" from data if "std_type" exists too
        if "std_type" in data[corr_str].columns and "type" in data[corr_str].columns and \
           corr_str != "LineType*std_types|line":
            del data[corr_str]["type"]
        # --- rename
        ordered_tuples = [(x, y, z) if not _is_pp_type(data) else (y, x, z) for x, y, z in tuples]
        columns = data[corr_str].columns
        nan_columns = data[corr_str].columns[data[corr_str].isnull().all()]
        to_rename_dict = dict()
        for x, y, _ in ordered_tuples:
            if y not in columns:  # do not rename if the column name already exists
                to_rename_dict[x] = y
            elif y in nan_columns:  # rename if the column exists with only nan values
                to_rename_dict[x] = y
                del data[corr_str][y]
        data[corr_str].rename(columns=to_rename_dict, inplace=True)

        # --- multiply
        to_multiply = pd.DataFrame([(y, m) for _, y, m in ordered_tuples if not pd.isnull(m) and
                                    y in data[corr_str].columns])
        if to_multiply.shape[0]:
            col = list(to_multiply.iloc[:, 0])
            factors = to_multiply.iloc[:, 1].values
            if _is_pp_type(data):
                factors = 1/factors
            data[corr_str].loc[:, col] *= factors


def _get_parameters_to_rename_and_multiply():
    """ Returns a dict of tuples and a dict of dataframes where csv column names are assigned to
    pandapower columns names which differ. """
    # --- create dummy_net to get pp columns
    dummy_net = pp.create_empty_network()
    _extend_pandapower_net_columns(dummy_net)
    _prepare_res_bus_table(dummy_net)
    for elm in ["dcline"]:
        dummy_net[elm].rename(columns={"type": "std_type"}, inplace=True)
    for elm in ["gen", "sgen"]:
        dummy_net[elm]["vm_pu"] = np.nan
        dummy_net[elm]["va_degree"] = np.nan

    # --- get corresponding tables and dataframes
    corr_strings = _csv_table_pp_dataframe_correspondings(str)
    csv_tablenames_, pp_dfnames = _csv_table_pp_dataframe_correspondings(list)

    # --- initialize tuples_dict
    tuples_dict = dict.fromkeys(corr_strings, [("id", "name", None)])
    tuples_dict['NodePFResult*res_bus'] = []

    # --- determine tuples_dict
    for corr_str, csv_tablename, pp_dfname in zip(corr_strings, csv_tablenames_, pp_dfnames):
        # adapt tuples_dict initialization of Type tables
        if "Type" in csv_tablename:
            tuples_dict[corr_str] = [("id", "std_type", None)]
        # get all column correspodings
        corr_col_tuples = _csv_pp_column_correspondings(csv_tablename)
        # get csv and pp columns
        csv_columns = get_columns(csv_tablename)
        pp_columns = dummy_net[pp_dfname].columns if "std_types" not in pp_dfname else \
            pd.DataFrame(dummy_net["std_types"][pp_dfname[10:]]).T.columns
        # determine tuples_dict: all tuples which are in columns of both, csv and pp
        tuples_dict[corr_str] = tuples_dict[corr_str] + [
            corr_col_tuple for corr_col_tuple in corr_col_tuples if corr_col_tuple[0] in
            csv_columns and corr_col_tuple[1] in pp_columns]
        # unused:
#    dfs_dict = {key: pd.DataFrame(data, columns=["csv_col", "pp_col", "factor"]) for key, data in
#                tuples_dict.items()}
    return tuples_dict


def _replace_name_index(data):
    """ While the simbench csv format assigns connected nodes via names, pandapower assigns via
        indices. This function replaces the assignment of the input data. """
    node_names = {"node", "nodeA", "nodeB", "nodeHV", "nodeMV", "nodeLV"}
    bus_names = {"bus", "from_bus", "to_bus", "hv_bus", "mv_bus", "lv_bus"}
    corr_strings = _csv_table_pp_dataframe_correspondings(str)
    corr_strings.remove("Measurement*measurement")  # already done in convert_measurement()

    if _is_pp_type(data):
        for corr_str in corr_strings:
            for col in node_names & set(data[corr_str].columns):
                data[corr_str][col] = data["Node*bus"]["id"].loc[data[corr_str][col]].values
    else:
        for corr_str in corr_strings:
            for col in bus_names & set(data[corr_str].columns):
                # each bus name must be unique
                data[corr_str][col] = data["Node*bus"].index[idx_in_2nd_array(data[
                    corr_str][col].values, data["Node*bus"]["name"].values)]


def _merge_dcline_and_storage_type_and_element_data(input_data):
    """ Merges dcline and storage data from Type tables into element tables. """
    for tablename in ["DCLine"]:
        elm = tablename.lower()
        corr_str_type = tablename+"Type*"+elm
        tablename = tablename if "DC" not in tablename else tablename.replace("DC", "")
        corr_str = tablename+"*"+elm
        idx_type = input_data[corr_str_type]["std_type"].index[idx_in_2nd_array(
            input_data[corr_str].std_type.values, input_data[corr_str_type]["std_type"].values)]
        Type_col_except_std_type = input_data[corr_str_type].columns.difference(["std_type"])
        if version.parse(pd.__version__) >= version.parse("0.21.0"):
            input_data[corr_str] = input_data[corr_str].reindex(
                columns=input_data[corr_str].columns | Type_col_except_std_type)
        else:
            input_data[corr_str] = input_data[corr_str].reindex_axis(
                input_data[corr_str].columns | Type_col_except_std_type, axis=1)
        input_data[corr_str].loc[:, Type_col_except_std_type] = input_data[corr_str_type].loc[
            idx_type, Type_col_except_std_type].values
        input_data[corr_str_type].drop(input_data[corr_str_type].index, inplace=True)


def _copy_data(input_data, output_data):
    """ Copies the data from output_data[corr_strings] into input_data[element_table]. This function
        handles that some corr_strings are not in output_data.keys() and copies all columns which
        exists in both, output_data[corr_strings] and input_data[element_table]. """
    corr_strings = _csv_table_pp_dataframe_correspondings(str)
    output_names = _csv_table_pp_dataframe_correspondings(list)[int(_is_pp_type(output_data))]

    for corr_str, output_name in zip(corr_strings, output_names):
        if corr_str in input_data.keys() and input_data[corr_str].shape[0]:
            cols_to_copy = list(set(output_data[output_name].columns) &
                                set(input_data[corr_str].columns))
            if version.parse(pd.__version__) >= version.parse("0.23.0"):
                output_data[output_name] = pd.concat([output_data[output_name], input_data[
                    corr_str][cols_to_copy]], ignore_index=True, sort=False).reindex(columns=output_data[
                        output_name].columns)
            elif version.parse(pd.__version__) >= version.parse("0.21.0"):
                output_data[output_name] = pd.concat([output_data[output_name], input_data[
                    corr_str][cols_to_copy]], ignore_index=True).reindex(columns=output_data[
                        output_name].columns)
            else:
                output_data[output_name] = pd.concat([output_data[output_name], input_data[
                    corr_str][cols_to_copy]], ignore_index=True).reindex_axis(columns=output_data[
                        output_name].columns)
            if "std_types" in corr_str and _is_pp_type(output_data):
                output_data[output_name].index = input_data[corr_str]["std_type"]
            _inscribe_fix_values(output_data, output_name)


def _inscribe_fix_values(output_data, output_name):
    """ Writes fix values (which are not given in the input_data format) given in tuples into
        output table columns. """
    if _is_pp_type(output_data):
        fix_values_tuples = [
            ("in_service", True), ("scaling", 1.), ("parallel", 1), ("g_us_per_km", 0.), ("df", 1.),
            ("tap_step_degree", 0.), ("tap_phase_shifter", False), ("const_i_percent", 0.),
            ("const_z_percent", 0.), ("min_e_mwh", 0.), ("slack", False), ("z_ohm", 0.)]
    else:
        fix_values_tuples = [("dVa", 0.), ("dVaHV", 0.), ("dVaMV", 0.), ("dVaLV", 0.)]
    for fxt in fix_values_tuples:
        if fxt[0] in output_data[output_name].columns:
            output_data[output_name][fxt[0]] = fxt[1]
