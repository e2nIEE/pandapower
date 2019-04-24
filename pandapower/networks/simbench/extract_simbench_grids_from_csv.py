import numpy as np
import pandas as pd
import os
from copy import deepcopy
import pandapower as pp

from pandapower.networks.simbench.__init__ import simbench_networks_path
from pandapower.networks.simbench.simbench_code import *
from pandapower.networks.simbench.profiles import filter_unapplied_profiles
from pandapower.networks.simbench.loadcases import filter_loadcases
from pandapower.converter import csv_data2pp, read_csv_data, csv_tablenames, idx_in_2nd_array, \
    ensure_iterability

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

__author__ = "smeinecke"


def all_grids_csv_path(scenario, all_grids_path=None):
    """ Returns the path to all simbench grid csv files.
        scenario in [0, 1, 2]. """
    all_grids_path = all_grids_path if all_grids_path is not None else os.path.join(
        simbench_networks_path, "all_grids_csv")
    scenario = int(scenario)
    if scenario in [1, 2]:
        all_grids_path += "_scenario%i" % scenario
    elif scenario != 0:
        raise ValueError("'scenario' must be in [0, 1, 2], but is %s." % str(scenario))
    return all_grids_path


def _grid_number_dict():
    return {"EHV": {"mixed": 1},
            "HV": {"mixed": 1, "urban": 2},
            "MV": {"rural": 1, "semiurb": 2, "urban": 3, "comm": 4},
            "LV": {"rural1": 1, "rural2": 2, "rural3": 3, "semiurb4": 4, "semiurb5": 5,
                   "urban6": 6}}


def get_bus_bus_switch_indices_from_csv(switch_table, node_table, error_type=ValueError):
    """ Returns a list of indices of bus-bus-switches from csv tables. """
    for nodeX in ["nodeA", "nodeB"]:
        missing_nodeX = switch_table[~switch_table[nodeX].isin(node_table.id)]
        if len(missing_nodeX):
            message = "At Switches " + str(["%s" % name for name in missing_nodeX.id]) + \
                " %s " % nodeX + str(["%s" % name for name in missing_nodeX[nodeX]]) + \
                " do not occur in Node table."
            raise ValueError(message)
    idx_nodeA = idx_in_2nd_array(switch_table.nodeA.values, node_table.id.values)
    idx_nodeB = idx_in_2nd_array(switch_table.nodeB.values, node_table.id.values)

    aux_nodeA = np.array(node_table.type[idx_nodeA] == "auxiliary")
    aux_nodeB = np.array(node_table.type[idx_nodeB] == "auxiliary")

    double_aux = switch_table.id[aux_nodeA & aux_nodeB]
    if len(double_aux):
        message = "Both side auxiliary nodes at Switches " + str([
            "%s" % name for name in double_aux])
        if error_type is None or error_type is False:
            logger.debug(message)
        else:
            raise error_type(message)

    return list(switch_table.index[~(aux_nodeA | aux_nodeB)])


def _simple_hv_subnet_determination(sb_code_parameters):
    """ Determines the hv_subnet, neglecting special cases of complete grid or complete dataset
        download. """
    hv_grid_number = _grid_number_dict()[sb_code_parameters[1]][sb_code_parameters[3]]
    hv_subnet = sb_code_parameters[1] + str(hv_grid_number)
    hv_subnet += ".101" if sb_code_parameters[1] in ["MV", "LV"] else ""
    hv_subnet = hv_subnet if hv_subnet not in ["LV5.101", "LV6.101"] else hv_subnet[:-3] + "201"
    return hv_subnet, hv_grid_number


def _simple_lv_subnets_determination(sb_code_parameters, hv_subnet, hv_grid_number, input_path):
    """ Determines the list of all lv_subnets which are connectable to given hv_subnet.
        This function neglects special cases of complete grid or complete dataset download. """
    if sb_code_parameters[2] == "":
        lv_subnets = []
    elif sb_code_parameters[2] == "HV":
        lv_subnet_list = ["HV1", "HV2"]
        lv_subnets = {"all": lv_subnet_list, 1: ["HV1"], 2: ["HV2"]}[sb_code_parameters[4]]
    else:
        load_data = pd.read_csv(os.path.join(input_path, "Load.csv"), sep=";")
        lv_types = load_data.loc[load_data.subnet.str.startswith(
            hv_subnet + "_" + sb_code_parameters[2])].profile.value_counts()
        filtered_lv_types = lv_types[
            (pd.Series(lv_types.index.str[:2]).str.upper() == sb_code_parameters[2]).values]

        lv_subnet_list = []
        for type_, number in filtered_lv_types.iteritems():
            if type_[:2].upper() == sb_code_parameters[2]:
                if type_[3:] in _grid_number_dict()[sb_code_parameters[2]].keys():
                    lv_subnet_list += [sb_code_parameters[2] + str(_grid_number_dict()[
                        sb_code_parameters[2]][type_[3:]]) + ".%i" % (i+(hv_grid_number)*100) for
                        i in range(1, 1+number)]

        # --- determine lv_subnets for single or all lv grids
        if sb_code_parameters[4] == "all":
            lv_subnets = lv_subnet_list
        elif isinstance(sb_code_parameters[4], str):
            if sb_code_parameters[2]+sb_code_parameters[4] not in lv_subnet_list:
                raise ValueError("'sb_code_parameters[4]' %s is " % sb_code_parameters[4] +
                                 "not in 'lv_subnet_list'.")
            lv_subnets = [sb_code_parameters[2]+sb_code_parameters[4]]
        else:
            raise ValueError("'sb_code_parameters[4]' must be a string, e.g. 'all' or 'MV1.1'" +
                             " (depending on the voltage level).")
    return lv_subnets


def get_relevant_subnets(sb_code_info, input_path):
    """ Determines a list of relevant subnet names of a parameter set, describing a SimBench grid
    selection. This list of subnets can be used to extract the requested SimBench grid from all
    grids data."""
    _, sb_code_parameters = get_simbench_code_and_parameters(sb_code_info)

    # --- in case of complete data download:
    if sb_code_parameters[1] == "complete_data":
        assert sb_code_parameters[2] == ""
        return sb_code_parameters[1], sb_code_parameters[2]

    # --- in case of complete grid download:
    if sb_code_parameters[2] == "HVMVLV":
        sb_code_parameters = deepcopy(sb_code_parameters)
        hv_subnets = []
        lv_subnets = []
        for hv_level, lv_level in zip(["EHV", "HV", "MV"], ["HV", "MV", "LV"]):
            sb_code_parameters[1] = hv_level
            sb_code_parameters[2] = lv_level
            for hv_type in _grid_number_dict()[hv_level].keys():
                sb_code_parameters[3] = hv_type
                hv_subnet, hv_grid_number = _simple_hv_subnet_determination(sb_code_parameters)
                hv_subnets += [hv_subnet]
                lv_subnets += _simple_lv_subnets_determination(
                    sb_code_parameters, hv_subnet, hv_grid_number, input_path)
        return hv_subnets, lv_subnets

    # --- in other cases
    # determine hv_subnet
    hv_subnet, hv_grid_number = _simple_hv_subnet_determination(sb_code_parameters)
    # determine lv_subnets
    lv_subnets = _simple_lv_subnets_determination(
        sb_code_parameters, hv_subnet, hv_grid_number, input_path)
    return hv_subnet, lv_subnets


def _extract_csv_table_by_subnet(csv_table, tablename, relevant_subnets, bus_bus_switches={}):
    """ Extracts csv table by subnet names.

        INPUT:
            **csv_table** (DataFrame)

            **tablename** (str)

            **relevant_subnets** (tuple) - first item is hv_subnet (str), second lv_subnets (list of
                strings)

        OPTIONAL:
            **bus_bus_switches** (set, {}) - indices of bus-bus-switches in csv DataFrame.
                Only used if tablename == "Switch".
    """
    hv_subnets = ensure_iterability(relevant_subnets[0])
    lv_subnets = relevant_subnets[1]
    if "complete_data" in hv_subnets or \
       isinstance(csv_table, pd.DataFrame) and not csv_table.shape[0]:
        return csv_table  # no extraction needed
    csv_table = deepcopy(csv_table)

    if isinstance(csv_table, pd.DataFrame) and "subnet" in csv_table.columns:
        logger.debug("Start extracting %s" % tablename)
        subnet_split = csv_table.subnet.str.split("_", expand=True)

        # --- hv_elms: all elements starting with hv_subnet
        hv_elms = set(subnet_split.index[subnet_split[0].isin(hv_subnets)])
        if tablename in ["Node", "Coordinates", "Measurement", "Switch", "Substation"]:
            # including elements that subnet data is hv_subnet between 1st "_" and 2nd "_" or end
            hv_elms_to_add = set(subnet_split.index[subnet_split[1].isin(hv_subnets)])
            if tablename == "Switch":
                hv_elms_to_add = hv_elms_to_add & bus_bus_switches
            elif tablename == "Measurement":
                bus_measurements = set(csv_table.index[
                    pd.isnull(csv_table[["element1", "element2"]]).any(axis=1)])
                hv_elms_to_add = hv_elms_to_add & bus_measurements
            hv_elms |= hv_elms_to_add

        # --- lv_elms: all elements starting with lv_subnet
        lv_elms = set(subnet_split.index[subnet_split[0].isin(lv_subnets)])

        lv_hv_elms = set()
        hv_lv_elms = set()
        if 1 in subnet_split.columns:

            # --- lv_hv_elms
            if tablename in ["Node", "Coordinates", "Measurement", "Switch", "Substation"]:
                # all elements with a higher voltage level before 1st "_" than after the first "_"
                subnet_split_level = pd.DataFrame(None, index=subnet_split.index,
                                                  columns=subnet_split.columns)
                subnet_split.loc[[0, 1]] = subnet_split.loc[[0, 1]].replace(
                    {"EHV": 1, "HV": 3, "MV": 5, "LV": 5}
                )
                # for col in [0, 1]:
                #     for level_str, level_int in zip(["EHV", "HV", "MV", "LV"], [1, 3, 5, 7]):
                #         subnet_split_level.loc[
                #             subnet_split[col].fillna("").str.contains(level_str), col] = level_int
                lv_hv_elms = set(subnet_split_level.index[
                    subnet_split_level[0] > subnet_split_level[1]])
            else:
                # all elements with lv_subnet before 1st "_"
                # and is hv_subnet between 1st "_" and 2nd or end
                lv_hv_elms = set(subnet_split.index[
                    (subnet_split[0].isin(lv_subnets)) & (subnet_split[1].isin(hv_subnets))])

            # --- hv_lv_elms: all elements with hv_subnet before 1st "_"
            # and is lv_subnet between 1st "_" and 2nd or end
            if tablename not in ["Node", "Coordinates", "Switch", "Substation"]:
                hv_lv_elms = set(subnet_split.index[
                    (subnet_split[0].isin(hv_subnets)) & (subnet_split[1].isin(lv_subnets))])
            if tablename == "Measurement":
                hv_lv_elms -= bus_measurements

        # --- determine indices to drop and examine dropping
        drop_idx = (set(csv_table.index) - hv_elms - lv_elms) | hv_lv_elms | lv_hv_elms
        csv_table.drop(drop_idx, inplace=True)
        no_extraction = False
    else:
        no_extraction = "Profile" not in tablename and "Type" not in tablename and \
            tablename != "StudyCases"
        if no_extraction:
            logger.warning("From %s no extraction can be made by 'subnet'." % tablename)
    return csv_table


def _get_extracted_csv_table(relevant_subnets, tablename, input_path, sep=";"):
    """ Returns extracted csv data of the requested SimBench grid. """
    csv_table = read_csv_data(input_path, sep=sep, tablename=tablename)
    if tablename == "Switch":
        node_table = read_csv_data(input_path, sep=sep, tablename="Node")
        bus_bus_switches = set(get_bus_bus_switch_indices_from_csv(csv_table, node_table))
    else:
        bus_bus_switches = {}
    extracted_csv_table = _extract_csv_table_by_subnet(csv_table, tablename, relevant_subnets,
                                                       bus_bus_switches=bus_bus_switches)
    return extracted_csv_table


def _get_extracted_csv_data_from_dict(csv_data, relevant_subnets):
    """ Returns extracted csv data of the requested SimBench grid from given csv data dict. """
    csv_data = deepcopy(csv_data)
    if "Node" in csv_data.keys() and "Switch" in csv_data.keys():
        bus_bus_switches = set(get_bus_bus_switch_indices_from_csv(
            csv_data["Switch"], csv_data["Node"]))
    else:
        bus_bus_switches = {}
    for key in csv_data.keys():
        csv_data[key] = _extract_csv_table_by_subnet(csv_data[key], key, relevant_subnets,
                                                     bus_bus_switches=bus_bus_switches)
    return csv_data


def get_extracted_csv_data(relevant_subnets, input_path, sep=";", **kwargs):
    """ Returns extracted csv data of the requested SimBench grid
    (per default from all SimBench grids csv data).
    **kwargs are ignored.
    """
    # --- import input data
    if 'complete_data' in relevant_subnets[0]:  # return complete data
        return read_csv_data(input_path, sep=sep)
    else:
        csv_data = dict()
        for tablename in csv_tablenames(['elements', 'profiles', 'types', 'cases']):
            csv_data[tablename] = _get_extracted_csv_table(relevant_subnets, tablename,
                                                           input_path=input_path)
    return csv_data


def _get_connected_buses_via_bus_bus_switch(net, buses):
    """ Returns a set of buses which are connected to 'buses' via bus-bus switches. """
    buses = set(ensure_iterability(buses))
    add_buses = [1]  # set anything to add_buses to start the while loop
    while(len(add_buses)):
        add_buses = pp.get_connected_buses(net, buses, consider=("s"))
        buses |= add_buses
    return buses


def generate_no_sw_variant(net):
    """ Drops all bus-bus switches and fuses buses which were connected by bus-bus switches.
        Furthermore drop all closed line and trafo switches. """
    # get bus-bus switch indices and close the switches to let get_connected_buses() find them
    bus_bus_sw = net.switch.index[net.switch.et == "b"]
    net.switch.loc[bus_bus_sw, "closed"] = True

    # determine to_fuse dict
    # to_fuse is used to fuse buses which are connected via bus-bus switches
    to_fuse = dict()
    already_considered = set()
    for bbs in bus_bus_sw:
        bus1 = net.switch.bus.at[bbs]
        if bus1 not in already_considered:
            bus2 = net.switch.element.at[bbs]
            to_fuse[bus1] = _get_connected_buses_via_bus_bus_switch(net, {bus1, bus2}) - {bus1}
            already_considered |= {bus1}
            already_considered |= to_fuse[bus1]

    # drop all closed switches (which now also includes all bus-bus switches (see above))
    net.switch.drop(net.switch.index[net.switch.closed], inplace=True)

    # fuse buses which are connected via bus-bus switches
    for b1, b2 in to_fuse.items():
        pp.fuse_buses(net, b1, b2)


def get_simbench_net(sb_code_info, input_path=None):
    """
    Returns the simbench net, requested by a given SimBench code information. Please have a look
    into jupyter notebook tutorials to learn more about simbench grids and the meaning of SimBench
    codes.

    INPUT:
        sb_code_info (str or list) - simbench code which defines which simbench grid is requested,
            e.g. '1-MVLV-urban-all-0-sw' requests a grid with the urban MV grid and all connected
            LV grids, both of SimBench version 1, scenario zero and with full switch representation.

    OPTIONAL:
        input_path (path) - option to change the path to all simbench grid csv files. However, a
            change should not be necessary.
    OUTPUT:
        net (pandapowerNet)

    EXAMPLE:

        import pandapower.networks as nw

        net = nw.get_simbench_net('1-MVLV-urban-all-0-sw')
    """
    # --- get relevant subnets
    sb_code, sb_code_parameters = get_simbench_code_and_parameters(sb_code_info)
    input_path = input_path if input_path is not None else all_grids_csv_path(sb_code_parameters[5])
    relevant_subnets = get_relevant_subnets(sb_code_parameters, input_path)

    # --- get_extracted_csv_data and convert this data to pandapower net
    csv_data = get_extracted_csv_data(relevant_subnets, input_path)
    filter_unapplied_profiles(csv_data)
    filter_loadcases(csv_data)
    net = csv_data2pp(csv_data)

    # --- remove switches if wanted by sb_code_info
    if not sb_code_parameters[6]:  # remove Switches
        generate_no_sw_variant(net)

    return net


if __name__ == '__main__':
    pass
