# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import gc
import warnings

import numpy as np
import pandas as pd
from packaging.version import Version

from pandapower.auxiliary import ets_to_element_types

from pandapower import __version__

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def get_element_index(net, element_type, name, exact_match=True):
    """
    Returns the element(s) identified by a name or regex and its element-table.

    INPUT:
      **net** - pandapower network

      **element_type** - Table to get indices from ("line", "bus", "trafo" etc.)

      **name** - Name of the element to match.

    OPTIONAL:
      **exact_match** (boolean, True) -
          True: Expects exactly one match, raises UserWarning otherwise.
          False: returns all indices containing the name

    OUTPUT:
      **index** - The index (or indices in case of exact_match=False) of matching element(s).
    """
    if exact_match:
        idx = net[element_type][net[element_type]["name"] == name].index
        if len(idx) == 0:
            raise UserWarning(f"There is no {element_type} with name {name}")
        if len(idx) > 1:
            raise UserWarning(f"Duplicate {element_type} names for {name}")
        return idx[0]
    else:
        return net[element_type][net[element_type]["name"].str.contains(name)].index


def get_element_indices(net, element_type, name, exact_match=True):
    """
    Returns a list of element(s) identified by a name or regex and its element-table -> Wrapper
    function of get_element_index()

    INPUT:
      **net** - pandapower network

      **element_type** (str, string iterable) - Element table to get indices from
      ("line", "bus", "trafo" etc.).

      **name** (str) - Name of the element to match.

    OPTIONAL:
      **exact_match** (boolean, True)

          - True: Expects exactly one match, raises UserWarning otherwise.
          - False: returns all indices containing the name

    OUTPUT:
      **index** (list) - List of the indices of matching element(s).

    EXAMPLE:
        >>> import pandapower.networks as pn
        >>> import pandapower as pp
        >>> net = pn.example_multivoltage()
        >>> # get indices of only one element type (buses in this example):
        >>> pp.get_element_indices(net, "bus", ["Bus HV%i" % i for i in range(1, 4)])
        [32, 33, 34]
        >>> # get indices of only two element type (first buses, second lines):
        >>> pp.get_element_indices(net, ["bus", "line"], "HV", exact_match=False)
        [Int64Index([32, 33, 34, 35], dtype='int64'), Int64Index([0, 1, 2, 3, 4, 5], dtype='int64')]
        >>> pp.get_element_indices(net, ["bus", "line"], ["Bus HV3", "MV Line6"])
        [34, 11]
    """
    if isinstance(element_type, str) and isinstance(name, str):
        element_type = [element_type]
        name = [name]
    else:
        element_type = element_type if not isinstance(element_type, str) else \
            [element_type] * len(name)
        name = name if not isinstance(name, str) else [name] * len(element_type)
    if len(element_type) != len(name):
        raise ValueError("'element_type' and 'name' must have the same length.")
    idx = []
    for et, nam in zip(element_type, name):
        idx += [get_element_index(net, et, nam, exact_match=exact_match)]
    return idx


def next_bus(net, bus, element_id, et='line', **kwargs):
    """
    Returns the index of the second bus an element is connected to, given a
    first one. E.g. the from_bus given the to_bus of a line.

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    bus : int
        index of bus
    element_id : int
        index of element
    et : str, optional
        which branch element type to consider, by default 'line'

    Returns
    -------
    int
        index of next connected bus
    """
    # todo: what to do with trafo3w?
    if et == 'line' or et == 'l':
        bc = ["from_bus", "to_bus"]
    elif et == 'trafo' or et == 't':
        bc = ["hv_bus", "lv_bus"]
    elif et == "switch" and list(net[et].loc[element_id, ["et"]].values) == ['b']:
        # Raises error if switch is not a bus-bus switch
        bc = ["bus", "element"]
    else:
        raise Exception("unknown element type")
    nb = list(net[et].loc[element_id, bc].values)
    nb.remove(bus)
    return nb[0]


def get_connected_elements(net, element_type, buses, respect_switches=True, respect_in_service=False):
    """
     Returns elements connected to a given buses.

     INPUT:
        **net** (pandapowerNet)

        **element_type** (string, name of the element table)

        **buses** (single integer or iterable of ints)

     OPTIONAL:
        **respect_switches** (boolean, True)

            - True: open switches will be respected
            - False: open switches will be ignored

        **respect_in_service** (boolean, False)

            - True: in_service status of connected lines will be respected
            - False: in_service status will be ignored

     OUTPUT:
        **connected_elements** (set) - Returns connected elements.

    """

    if not hasattr(buses, "__iter__"):
        buses = [buses]

    if element_type in ["line", "l"]:
        element_type = "l"
        element_table = net.line
        connected_elements = set(net.line.index[net.line.from_bus.isin(buses) |
                                                net.line.to_bus.isin(buses)])

    elif element_type in ["dcline"]:
        element_table = net.dcline
        connected_elements = set(net.dcline.index[net.dcline.from_bus.isin(buses) |
                                                  net.dcline.to_bus.isin(buses)])

    elif element_type in ["trafo"]:
        element_type = "t"
        element_table = net.trafo
        connected_elements = set(net["trafo"].index[(net.trafo.hv_bus.isin(buses)) |
                                                    (net.trafo.lv_bus.isin(buses))])
    elif element_type in ["trafo3w", "t3w"]:
        element_type = "t3w"
        element_table = net.trafo3w
        connected_elements = set(net["trafo3w"].index[(net.trafo3w.hv_bus.isin(buses)) |
                                                      (net.trafo3w.mv_bus.isin(buses)) |
                                                      (net.trafo3w.lv_bus.isin(buses))])
    elif element_type == "impedance":
        element_table = net.impedance
        connected_elements = set(net["impedance"].index[(net.impedance.from_bus.isin(buses)) |
                                                        (net.impedance.to_bus.isin(buses))])
    elif element_type == "measurement":
        element_table = net[element_type]
        connected_elements = set(net.measurement.index[(net.measurement.element.isin(buses)) |
                                                       (net.measurement.element_type == "bus")])
    elif element_type in pp_elements(bus=False, branch_elements=False):
        element_table = net[element_type]
        connected_elements = set(element_table.index[(element_table.bus.isin(buses))])
    elif element_type in ['_equiv_trafo3w']:
        # ignore '_equiv_trafo3w'
        return {}
    else:
        raise UserWarning(f"Unknown element type {element_type}!")

    if respect_switches and element_type in ["l", "t", "t3w"]:
        open_switches = get_connected_switches(net, buses, consider=element_type, status="open")
        if open_switches:
            open_and_connected = net.switch.loc[net.switch.index.isin(open_switches) &
                                                net.switch.element.isin(connected_elements)].index
            connected_elements -= set(net.switch.element[open_and_connected])

    if respect_in_service and "in_service" in element_table and not element_table.empty:
        connected_elements -= set(element_table[~element_table.in_service].index)

    return connected_elements


def get_connected_buses(net, buses, consider=("l", "s", "t", "t3", "i"), respect_switches=True,
                        respect_in_service=False):
    """
     Returns buses connected to given buses. The source buses will NOT be returned.

     INPUT:
        **net** (pandapowerNet)

        **buses** (single integer or iterable of ints)

     OPTIONAL:
        **respect_switches** (boolean, True)

            - True: open switches will be respected
            - False: open switches will be ignored

        **respect_in_service** (boolean, False)

            - True: in_service status of connected buses will be respected
            - False: in_service status will be ignored

        **consider** (iterable, ("l", "s", "t", "t3", "i")) - Determines, which types of
        connections will be considered.

            l: lines

            s: switches

            t: trafos

            t3: trafo3ws

            i: impedances

     OUTPUT:
        **cl** (set) - Returns connected buses.

    """
    if not hasattr(buses, "__iter__"):
        buses = [buses]

    cb = set()
    if "l" in consider or 'line' in consider:
        in_service_constr = net.line.in_service if respect_in_service else True
        opened_lines = set(net.switch.loc[(~net.switch.closed) & (
            net.switch.et == "l")].element.unique()) if respect_switches else set()
        connected_fb_lines = set(net.line.index[(
            net.line.from_bus.isin(buses)) & ~net.line.index.isin(opened_lines) &
            in_service_constr])
        connected_tb_lines = set(net.line.index[(
            net.line.to_bus.isin(buses)) & ~net.line.index.isin(opened_lines) & in_service_constr])
        cb |= set(net.line[net.line.index.isin(connected_tb_lines)].from_bus)
        cb |= set(net.line[net.line.index.isin(connected_fb_lines)].to_bus)

    if "s" in consider or 'switch' in consider:
        cs = get_connected_switches(net, buses, consider='b',
                                    status="closed" if respect_switches else "all")
        cb |= set(net.switch[net.switch.index.isin(cs)].element)
        cb |= set(net.switch[net.switch.index.isin(cs)].bus)

    if "t" in consider or 'trafo' in consider:
        in_service_constr = net.trafo.in_service if respect_in_service else True
        opened_trafos = set(net.switch.loc[(~net.switch.closed) & (
            net.switch.et == "t")].element.unique()) if respect_switches else set()
        connected_hvb_trafos = set(net.trafo.index[(
            net.trafo.hv_bus.isin(buses)) & ~net.trafo.index.isin(opened_trafos) &
            in_service_constr])
        connected_lvb_trafos = set(net.trafo.index[(
            net.trafo.lv_bus.isin(buses)) & ~net.trafo.index.isin(opened_trafos) &
            in_service_constr])
        cb |= set(net.trafo.loc[net.trafo.index.isin(connected_lvb_trafos)].hv_bus.values)
        cb |= set(net.trafo.loc[net.trafo.index.isin(connected_hvb_trafos)].lv_bus.values)

    # Gives the lv mv and hv buses of a 3 winding transformer
    if "t3" in consider or 'trafo3w' in consider:
        in_service_constr3w = net.trafo3w.in_service if respect_in_service else True
        if respect_switches:
            opened_buses_hv = set(net.switch.loc[
                ~net.switch.closed & (net.switch.et == "t3") &
                net.switch.bus.isin(net.trafo3w.hv_bus)].bus.unique())
            opened_buses_mv = set(net.switch.loc[
                ~net.switch.closed & (net.switch.et == "t3") &
                net.switch.bus.isin(net.trafo3w.mv_bus)].bus.unique())
            opened_buses_lv = set(net.switch.loc[
                ~net.switch.closed & (net.switch.et == "t3") &
                net.switch.bus.isin(net.trafo3w.lv_bus)].bus.unique())
        else:
            opened_buses_hv = opened_buses_mv = opened_buses_lv = set()

        hvb_trafos3w = net.trafo3w.index[
            net.trafo3w.hv_bus.isin(buses) & ~net.trafo3w.hv_bus.isin(opened_buses_hv) &
            in_service_constr3w]
        mvb_trafos3w = net.trafo3w.index[
            net.trafo3w.mv_bus.isin(buses) & ~net.trafo3w.mv_bus.isin(opened_buses_mv) &
            in_service_constr3w]
        lvb_trafos3w = net.trafo3w.index[
            net.trafo3w.lv_bus.isin(buses) & ~net.trafo3w.lv_bus.isin(opened_buses_lv) &
            in_service_constr3w]

        cb |= (set(net.trafo3w.loc[hvb_trafos3w].mv_bus) | set(
            net.trafo3w.loc[hvb_trafos3w].lv_bus) - opened_buses_mv - opened_buses_lv)
        cb |= (set(net.trafo3w.loc[mvb_trafos3w].hv_bus) | set(
            net.trafo3w.loc[mvb_trafos3w].lv_bus) - opened_buses_hv - opened_buses_lv)
        cb |= (set(net.trafo3w.loc[lvb_trafos3w].hv_bus) | set(
            net.trafo3w.loc[lvb_trafos3w].mv_bus) - opened_buses_hv - opened_buses_mv)

    if "i" in consider or 'impedance' in consider:
        in_service_constr = net.impedance.in_service if respect_in_service else True
        connected_fb_impedances = set(net.impedance.index[
                                          (net.impedance.from_bus.isin(buses)) & in_service_constr])
        connected_tb_impedances = set(net.impedance.index[
                                          (net.impedance.to_bus.isin(buses)) & in_service_constr])
        cb |= set(net.impedance[net.impedance.index.isin(connected_tb_impedances)].from_bus)
        cb |= set(net.impedance[net.impedance.index.isin(connected_fb_impedances)].to_bus)

    if respect_in_service:
        cb -= set(net.bus[~net.bus.in_service].index)

    return cb - set(buses)


def get_connected_buses_at_element(net, element_index, element_type, respect_in_service=False):
    """
     Returns buses connected to a given branch element. In case of a bus switch, two buses
     will be returned, else one.

     INPUT:
        **net** (pandapowerNet)

        **element_index** (integer)

        **element_type** (string) - Type of the source element:

            l, line: line

            s, switch: switch

            t, trafo: trafo

            t3, trafo3w: trafo3w

            i, impedance: impedance

     OPTIONAL:
        **respect_in_service** (boolean, False)

        True: in_service status of connected buses will be respected

        False: in_service status will be ignored

     OUTPUT:
        **cl** (set) - Returns connected switches.

    """

    cb = set()
    if element_type == 'l' or element_type == 'line':
        cb.add(net.line.from_bus.at[element_index])
        cb.add(net.line.to_bus.at[element_index])
    elif element_type == 's' or element_type == 'switch':
        cb.add(net.switch.bus.at[element_index])
        if net.switch.et.at[element_index] == 'b':
            cb.add(net.switch.element.at[element_index])
    elif element_type == 't' or element_type == 'trafo':
        cb.add(net.trafo.hv_bus.at[element_index])
        cb.add(net.trafo.lv_bus.at[element_index])
    elif element_type == 't3' or element_type == 'trafo3w':
        cb.add(net.trafo3w.hv_bus.at[element_index])
        cb.add(net.trafo3w.mv_bus.at[element_index])
        cb.add(net.trafo3w.lv_bus.at[element_index])
    elif element_type == 'i' or element_type == 'impedance':
        cb.add(net.impedance.from_bus.at[element_index])
        cb.add(net.impedance.to_bus.at[element_index])

    if respect_in_service:
        cb -= set(net.bus[~net.bus.in_service].index)

    return cb


def get_connected_buses_at_switches(net, switches):
    """
    Returns a set of buses connected to given switches.

    INPUT:
        **net** (pandapowerNet)

        **switches** (single integer or iterable of ints)

    OUTPUT:
       **buses** (set) - Returns connected buses
    """
    if not hasattr(switches, "__iter__"):
        switches = [switches]
    switches = np.array(switches)
    buses = set(net.switch.bus.loc[switches])
    bebd = branch_element_bus_dict()
    for element_type, et in zip(["bus", "line", "trafo", "trafo3w"], ["b", "l", "t", "t3"]):
        is_et = (net.switch.et.loc[switches] == et).values
        if not any(is_et):
            continue
        elif et == "b":
            buses |= set(net.switch.element.loc[switches[is_et]])
            continue
        for bus_col in bebd[element_type]:
            buses |= set(net[element_type].loc[net.switch.element.loc[switches[is_et]], bus_col])
    return buses


def get_connected_switches(net, buses, consider=('b', 'l', 't', 't3', 'i'), status="all",
                           include_element_connections=False):
    """
    Returns switches connected to given buses.

    INPUT:
        **net** (pandapowerNet)

        **buses** (single integer or iterable of ints)

    OPTIONAL:
        **consider** (iterable, ("l", "s", "t", "t3)) -  Determines, which types of connections
                                                      will be considered.
                                                      l: lines
                                                      b: bus-bus-switches
                                                      t: transformers
                                                      t3: 3W transformers
                                                      i: impedance

        **status** (string, ("all", "closed", "open")) -  Determines, which switches will
                                                            be considered

        **include_element_connections** (bool, False) - If True, also the other bus of the connected
        element, e.g. the other line ending, is included

    OUTPUT:
       **cl** (set) - Returns connected switches.
    """

    if not hasattr(buses, "__iter__"):
        buses = [buses]

    if status == "closed":
        switch_selection = net.switch.closed
    elif status == "open":
        switch_selection = ~net.switch.closed
    else:
        switch_selection = np.full(len(net.switch), True, dtype=bool)
        if status != "all":
            logger.warning("Unknown switch status \"%s\" selected! "
                           "Selecting all switches by default." % status)

    if include_element_connections:
        bebd = branch_element_bus_dict()

    cs = set()
    for et in consider:
        if et == 'b':
            cs |= set(net['switch'].index[
                          (net['switch']['bus'].isin(buses) |
                           net['switch']['element'].isin(buses)) &
                          (net['switch']['et'] == 'b') & switch_selection])
        else:
            cs |= set(net['switch'].index[(net['switch']['bus'].isin(buses)) &
                                          (net['switch']['et'] == et) & switch_selection])
            if include_element_connections:
                element_type = ets_to_element_types(et)
                sw_idx = net.switch.index[(net.switch.et == et) & switch_selection]
                element_buses = net[element_type].loc[list(net.switch.element.loc[sw_idx]),
                                                      bebd[element_type]]
                isin_df = pd.concat([element_buses[col].isin(buses) for col in element_buses],
                                    axis=1)
                cs |= set(sw_idx[isin_df.any(axis=1)])
    return cs


def get_connected_elements_dict(
        net, buses, respect_switches=True, respect_in_service=False, include_empty_lists=False,
        element_types=None, **kwargs):
    """Returns a dict of lists of connected elements.

    Parameters
    ----------
    net : _type_
        _description_
    buses : iterable of buses
        buses as origin to search for connected elements
    respect_switches : bool, optional
        _description_, by default True
    respect_in_service : bool, optional
        _description_, by default False
    include_empty_lists : bool, optional
        if True, the output doesn't have values of empty lists but may lack of element types as
        keys, by default False
    element_types : iterable of strings, optional
        types elements which are analyzed for connection. If not given, all pandapower element types
        are analyzed. That list of all element types can also be restricted by key word arguments
        "connected_buses", "connected_bus_elements", "connected_branch_elements" and
        "connected_other_elements", by default None

    Returns
    -------
    dict[str,list]
        elements connected to given buses
    """
    if element_types is None:
        element_types = pp_elements(
            bus=kwargs.get("connected_buses", True),
            bus_elements=kwargs.get("connected_bus_elements", True),
            branch_elements=kwargs.get("connected_branch_elements", True),
            other_elements=kwargs.get("connected_other_elements", True),
            cost_tables=False,
            res_elements=False)

    connected = dict()
    for et in element_types:
        if et == "bus":
            conn = get_connected_buses(net, buses, respect_switches=respect_switches,
                                       respect_in_service=respect_in_service)
        elif et == "switch":
            conn = get_connected_switches(net, buses)
        else:
            conn = get_connected_elements(
                net, et, buses, respect_switches=respect_switches,
                respect_in_service=respect_in_service)
        if include_empty_lists or len(conn):
            connected[et] = list(conn)
    return connected


def get_connecting_branches(net, buses1, buses2, branch_elements=None):
    """
    Gets/Drops branches that connects any bus of buses1 with any bus of buses2.
    """
    branch_dict = branch_element_bus_dict(include_switch=True)
    if branch_elements is not None:
        branch_dict = {key: branch_dict[key] for key in branch_elements}
    if "switch" in branch_dict:
        branch_dict["switch"].append("element")

    found = {et: set() for et in branch_dict.keys()}
    for et, bus_types in branch_dict.items():
        for bus1 in bus_types:
            for bus2 in bus_types:
                if bus2 != bus1:
                    idx = net[et].index[net[et][bus1].isin(buses1) & net[et][bus2].isin(buses2)]
                    if et == "switch":
                        idx = idx.intersection(net[et].index[net[et].et == "b"])
                    found[et] |= set(idx)
    return {key: val for key, val in found.items() if len(val)}


def get_gc_objects_dict():
    """
    This function is based on the code in mem_top module
    Summarize object types that are tracket by the garbage collector in the moment.
    Useful to test if there are memoly leaks.
    :return: dictionary with keys corresponding to types and values to the number of objects of the
    type
    """
    objs = gc.get_objects()
    nums_by_types = dict()

    for obj in objs:
        _type = type(obj)
        nums_by_types[_type] = nums_by_types.get(_type, 0) + 1
    return nums_by_types


def false_elm_links(net, element_type, col, target_element_type):
    """
    Returns which indices have links to elements of other element tables which does not exist in the
    net.

    Examples
    --------
    >>> false_elm_links(net, "line", "to_bus", "bus")  # exemplary input 1
    >>> false_elm_links(net, "poly_cost", "element", net["poly_cost"]["et"])  # exemplary input 2
    """
    if isinstance(target_element_type, str):
        return net[element_type][col].index[~net[element_type][col].isin(net[
            target_element_type].index)]
    else:  # target_element_type is an iterable, e.g. a Series such as net["poly_cost"]["et"]
        df = pd.DataFrame({"element": net[element_type][col].values, "et": target_element_type,
                           "indices": net[element_type][col].index.values})
        df = df.set_index("et")
        false_links = pd.Index([])
        for et in df.index:
            false_links = false_links.union(pd.Index(df.loc[et].indices.loc[
                ~df.loc[et].element.isin(net[et].index)]))
        return false_links


def false_elm_links_loop(net, element_types=None):
    """
    Returns a dict of elements which indices have links to elements of other element tables which
    does not exist in the net.
    This function is an outer loop for get_false_links() applications.
    """
    false_links = dict()
    element_types = element_types if element_types is not None else pp_elements(
        bus=False, cost_tables=True)
    bebd = branch_element_bus_dict(include_switch=True)
    for element_type in element_types:
        if net[element_type].shape[0]:
            fl = pd.Index([])
            # --- define col and target_element_type
            if element_type in bebd.keys():
                for col in bebd[element_type]:
                    fl = fl.union(false_elm_links(net, element_type, col, "bus"))
            elif element_type in {"poly_cost", "pwl_cost"}:
                fl = fl.union(false_elm_links(net, element_type, "element", net[element_type][
                    "et"]))
            elif element_type == "measurement":
                fl = fl.union(false_elm_links(net, element_type, "element", net[element_type][
                    "element_type"]))
            else:
                fl = fl.union(false_elm_links(net, element_type, "bus", "bus"))
            if len(fl):
                false_links[element_type] = fl
    return false_links


def pp_elements(bus=True, bus_elements=True, branch_elements=True, other_elements=True,
                cost_tables=False, res_elements=False):
    """
    Returns a set of pandapower elements.
    """
    pp_elms = set()
    if bus:
        pp_elms |= {"bus"}
        if res_elements:
            pp_elms |= {"res_bus"}
    pp_elms |= set([el[0] for el in element_bus_tuples(
        bus_elements=bus_elements, branch_elements=branch_elements, res_elements=res_elements)])
    if other_elements:
        pp_elms |= {"measurement"}
    if cost_tables:
        pp_elms |= {"poly_cost", "pwl_cost"}
    return pp_elms


def branch_element_bus_dict(include_switch=False, sort=None):
    """
    Returns a dict with keys of branch elements and values of bus column names as list.
    """
    msg = ("The parameter 'sort' is deprecated to function branch_element_bus_dict() with "
           "pp.version >= 2.12. The default was False but the behaviour was changed to True.")
    if sort is not None:
        if Version(__version__) < Version('2.13'):
            warnings.warn(msg, category=DeprecationWarning)
        else:
            raise DeprecationWarning(msg)
    elif Version(__version__) < Version('2.13'):
        logger.debug(msg)

    ebts = element_bus_tuples(bus_elements=False, branch_elements=True, res_elements=False)
    bebd = dict()
    for et, bus in ebts:
        if et in bebd.keys():
            bebd[et].append(bus)
        else:
            bebd[et] = [bus]
    if not include_switch:
        del bebd["switch"]
    return bebd


def element_bus_tuples(bus_elements=True, branch_elements=True, res_elements=False):
    """
    Utility function
    Provides the tuples of elements and corresponding columns for buses they are connected to
    :param bus_elements: whether tuples for bus elements e.g. load, sgen, ... are included
    :param branch_elements: whether branch elements e.g. line, trafo, ... are included
    :param res_elements: whether result table names e.g. res_sgen, res_line, ... are included
    :param return_type: which type the output has
    :return: list of tuples with element names and column names
    """
    if Version(__version__) < Version('2.13'):
        logger.debug("element_bus_tuples() returns a list of tuples instead of a set of tuples "
                     "since pp.version >= 2.12.")
    ebts = list()
    if bus_elements:
        ebts += [("sgen", "bus"), ("load", "bus"), ("ext_grid", "bus"), ("gen", "bus"),
                 ("ward", "bus"), ("xward", "bus"), ("shunt", "bus"),
                 ("storage", "bus"), ("asymmetric_load", "bus"), ("asymmetric_sgen", "bus"),
                 ("motor", "bus")]
    if branch_elements:
        ebts += [("line", "from_bus"), ("line", "to_bus"), ("impedance", "from_bus"),
                ("impedance", "to_bus"), ("switch", "bus"), ("trafo", "hv_bus"),
                ("trafo", "lv_bus"), ("trafo3w", "hv_bus"), ("trafo3w", "mv_bus"),
                ("trafo3w", "lv_bus"), ("dcline", "from_bus"), ("dcline", "to_bus")]
    if res_elements:
        elements_without_res = ["switch", "measurement", "asymmetric_load", "asymmetric_sgen"]
        ebts += [("res_" + ebt[0], ebt[1]) for ebt in ebts if ebt[0] not in elements_without_res]
    return ebts


def count_elements(net, return_empties=False, **kwargs):
    """Counts how much elements of which element type exist in the pandapower net

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    return_empties : bool, optional
        whether element types should be listed if no element exist, by default False

    Other Parameters
    ----------------
    kwargs : dict[str,bool], optional
        arguments (passed to pp_elements()) to narrow considered element types.
        If nothing is passed, an empty dict is passed to pp_elements(), by default None

    Returns
    -------
    pd.Series
        number of elements per element type existing in the net

    See also
    --------
    count_group_elements

    Examples
    --------
    >>> import pandapower as pp
    >>> import pandapower.networks as nw
    >>> pp.count_elements(nw.case9(), bus_elements=False)
    bus     9
    line    9
    dtype: int32
    """
    return pd.Series({et: net[et].shape[0] for et in pp_elements(**kwargs) if return_empties or \
                      bool(net[et].shape[0])}, dtype=np.int64)