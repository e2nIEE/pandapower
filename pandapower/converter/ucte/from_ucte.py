# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging
import time
from pandapower.converter.ucte.ucte_converter import UCTE2pandapower
from pandapower.converter.ucte.ucte_parser import UCTEParser
from pandapower.auxiliary import pandapowerNet
from pandapower.toolbox import get_connected_buses

logger = logging.getLogger('ucte.from_ucte')


def from_ucte_dict(ucte_parser: UCTEParser, slack_as_gen: bool = True, clip_small_x_values: bool = True) -> pandapowerNet:
    """
    Creates a pandapower net from an UCTE data structure.

    :param UCTEParser ucte_parser: The UCTEParser with parsed UCTE data.
    :param bool slack_as_gen: decides whether slack elements are converted as gen or ext_grid elements, default True

    :return: A pandapower net.
    :rtype: pandapowerNet

    """
    ucte_converter = UCTE2pandapower(slack_as_gen=slack_as_gen, clip_small_x_values=clip_small_x_values)
    net = ucte_converter.convert(ucte_parser.get_data())
    return net


def from_ucte(ucte_file: str, slack_as_gen: bool = True, clip_small_x_values: bool = False, harmonize_voltage_setpoints: bool = False) -> pandapowerNet:
    """
    Converts net data stored as an UCTE file to a pandapower net.

    :param str ucte_file: path to the ucte file which includes all the data of the grid (EHV or HV or both)
    :param bool slack_as_gen: decides whether slack elements are converted as gen or ext_grid elements.
    :param bool clip_small_x_values: decides whether small X values shall be clipped to 0.05 Ohm (recommendation from UCTE-DEF)
    :param bool harmonize_voltage_setpoints: decides whether voltage setpoints of electrically connected gens shall be harmonized

    :return: A pandapower net
    :rtype: pandapowerNet

    :example:
        >>> import os
        >>> from pandapower import pp_dir
        >>> from pandapower.converter.ucte.from_ucte import from_ucte
        >>>
        >>> ucte_file = os.path.join(pp_dir, "test", "converter", "testfiles", "test_ucte_DK.uct")
        >>> net = from_ucte(ucte_file)
    """
    # Note:
    # the converter functionality from_ucte() and internal functions are structured similar to
    # the cim converter

    time_start_parsing = time.time()

    ucte_parser = UCTEParser(ucte_file)
    ucte_parser.parse_file()

    time_start_converting = time.time()

    pp_net = from_ucte_dict(ucte_parser, slack_as_gen=slack_as_gen, clip_small_x_values=clip_small_x_values)

    if harmonize_voltage_setpoints:
        average_voltage_setpoints(pp_net)

    time_end_converting = time.time()

    logger.info("Needed time for parsing from ucte: %s" % (time_start_converting - time_start_parsing))
    logger.info("Needed time for converting from ucte: %s" % (time_end_converting - time_start_converting))
    logger.info("Total Time (from_ucte()): %s" % (time_end_converting - time_start_parsing))

    return pp_net

def average_voltage_setpoints(net: pandapowerNet) -> None:
    """
    Adjust generator voltage setpoints by averaging vm_pu for generators that
    appear to represent the same physical voltage setpoint across connected buses.

    This function mutates net.gen in-place.

    Algorithm / detailed behavior:
    - Create a temporary 'prefix' column from the first 7 characters of each
        generator name. Generators that share this prefix are treated as a group
        of candidates for having a shared voltage setpoint.
    - For each group with more than one distinct name:
        - Collect the buses where those generators are connected.
        - Starting from the first bus in the list, expand the set of buses by
            repeatedly calling get_connected_buses(..., consider=('s'), respect_switches=False)
            until no new buses are found. This performs a local neighborhood search
            across switch connections to find buses electrically tied to the start bus.
        - Find which of the group's generator buses (beyond the first) lie within
            the expanded connected-bus set. If any are found, consider the first bus
            and those matching buses as "critical" for averaging.
        - Compute the arithmetic mean of vm_pu for generators connected to the
            critical buses and assign that mean to all of those generators' vm_pu.
    - Remove the temporary 'prefix' column.

    Parameters
    - net: pandapowerNet
            A pandapower network object expected to contain a net.gen table with at
            least the following columns:
            - name (string)
            - bus (bus indices)
            - vm_pu (voltage setpoint in per-unit)

    Returns
    - None

    Side effects and notes
    - The function modifies net.gen['vm_pu'] in-place for generators deemed to
        share a voltage setpoint.
    - A temporary column 'prefix' is added to net.gen and removed before returning.
    - Grouping is done by the first 7 characters of the generator name; adjust
        this slice if your naming convention differs.
    - The function depends on get_connected_buses to discover electrically
        connected buses; its behavior (which element types are considered and how
        switches are handled) affects which generators are averaged together.

    Example (conceptual)
    - If generators "GEN_A_01" and "GEN_A_02" share the same 7-char prefix and
        their buses are connected through switches, their vm_pu values will be
        replaced by their common average.
    """
    net.gen["prefix"] = net.gen["name"].str[:7]
    name_sets = (
        net.gen
        .groupby("prefix")["name"]
        .apply(lambda x: set(x) if len(x) > 1 else None)
        .dropna()
        .tolist()
    )
    for name_set in name_sets:
        list_names = list(name_set)
        connected_buses = list(net.gen.loc[net.gen.name.isin(list_names), 'bus'].values)
        aux_buses = connected_buses[0:1]
        len_aux_buses = len(aux_buses)
        len_changed = True
        while len_changed:
            aux_buses += get_connected_buses(net, aux_buses, consider=('s'), respect_switches=False)
            if len(aux_buses) > len_aux_buses:
                len_aux_buses = len(aux_buses)
            else:
                len_changed = False
        matches = list(set(aux_buses) & set(connected_buses[1:]))
        if len(matches):
            critical_buses = matches+connected_buses[0:1]
            net.gen.loc[net.gen.bus.isin(critical_buses), 'vm_pu'] = net.gen.loc[net.gen.bus.isin(critical_buses), 'vm_pu'].mean()
    net.gen = net.gen.drop(columns="prefix")
