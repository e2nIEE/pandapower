# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import gc

import numpy as np
import pandas as pd
import numbers

from pandapower.toolbox_general_issues import pp_elements, branch_element_bus_dict

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def get_element_index(net, element, name, exact_match=True):
    """
    Returns the element(s) identified by a name or regex and its element-table.

    INPUT:
      **net** - pandapower network

      **element** - Table to get indices from ("line", "bus", "trafo" etc.)

      **name** - Name of the element to match.

    OPTIONAL:
      **exact_match** (boolean, True) -
          True: Expects exactly one match, raises UserWarning otherwise.
          False: returns all indices containing the name

    OUTPUT:
      **index** - The indices of matching element(s).
    """
    if exact_match:
        idx = net[element][net[element]["name"] == name].index
        if len(idx) == 0:
            raise UserWarning(f"There is no {element} with name {name}")
        if len(idx) > 1:
            raise UserWarning(f"Duplicate {element} names for {name}")
        return idx[0]
    else:
        return net[element][net[element]["name"].str.contains(name)].index


def get_element_indices(net, element, name, exact_match=True):
    """
    Returns a list of element(s) identified by a name or regex and its element-table -> Wrapper
    function of get_element_index()

    INPUT:
      **net** - pandapower network

      **element** (str, string iterable) - Element table to get indices from
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
        >>> idx1 = pp.get_element_indices(net, "bus", ["Bus HV%i" % i for i in range(1, 4)])
        >>> idx2 = pp.get_element_indices(net, ["bus", "line"], "HV", exact_match=False)
        >>> idx3 = pp.get_element_indices(net, ["bus", "line"], ["Bus HV3", "MV Line6"])
    """
    if isinstance(element, str) and isinstance(name, str):
        element = [element]
        name = [name]
    else:
        element = element if not isinstance(element, str) else [element] * len(name)
        name = name if not isinstance(name, str) else [name] * len(element)
    if len(element) != len(name):
        raise ValueError("'element' and 'name' must have the same length.")
    idx = []
    for elm, nam in zip(element, name):
        idx += [get_element_index(net, elm, nam, exact_match=exact_match)]
    return idx


def next_bus(net, bus, element_id, et='line', **kwargs):
    """
    Returns the index of the second bus an element is connected to, given a
    first one. E.g. the from_bus given the to_bus of a line.
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


def get_connected_elements(net, element, buses, respect_switches=True, respect_in_service=False):
    """
     Returns elements connected to a given bus.

     INPUT:
        **net** (pandapowerNet)

        **element** (string, name of the element table)

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

    if element in ["line", "l"]:
        element = "l"
        element_table = net.line
        connected_elements = set(net.line.index[net.line.from_bus.isin(buses) |
                                                net.line.to_bus.isin(buses)])

    elif element in ["dcline"]:
        element_table = net.dcline
        connected_elements = set(net.dcline.index[net.dcline.from_bus.isin(buses) |
                                                  net.dcline.to_bus.isin(buses)])

    elif element in ["trafo"]:
        element = "t"
        element_table = net.trafo
        connected_elements = set(net["trafo"].index[(net.trafo.hv_bus.isin(buses)) |
                                                    (net.trafo.lv_bus.isin(buses))])
    elif element in ["trafo3w", "t3w"]:
        element = "t3w"
        element_table = net.trafo3w
        connected_elements = set(net["trafo3w"].index[(net.trafo3w.hv_bus.isin(buses)) |
                                                      (net.trafo3w.mv_bus.isin(buses)) |
                                                      (net.trafo3w.lv_bus.isin(buses))])
    elif element == "impedance":
        element_table = net.impedance
        connected_elements = set(net["impedance"].index[(net.impedance.from_bus.isin(buses)) |
                                                        (net.impedance.to_bus.isin(buses))])
    elif element == "measurement":
        element_table = net[element]
        connected_elements = set(net.measurement.index[(net.measurement.element.isin(buses)) |
                                                       (net.measurement.element_type == "bus")])
    elif element in pp_elements(bus=False, branch_elements=False):
        element_table = net[element]
        connected_elements = set(element_table.index[(element_table.bus.isin(buses))])
    elif element in ['_equiv_trafo3w']:
        # ignore '_equiv_trafo3w'
        return {}
    else:
        raise UserWarning("Unknown element! ", element)

    if respect_switches and element in ["l", "t", "t3w"]:
        open_switches = get_connected_switches(net, buses, consider=element, status="open")
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


def get_connected_buses_at_element(net, element, et, respect_in_service=False):
    """
     Returns buses connected to a given line, switch or trafo. In case of a bus switch, two buses
     will be returned, else one.

     INPUT:
        **net** (pandapowerNet)

        **element** (integer)

        **et** (string) - Type of the source element:

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
    if et == 'l' or et == 'line':
        cb.add(net.line.from_bus.at[element])
        cb.add(net.line.to_bus.at[element])
    elif et == 's' or et == 'switch':
        cb.add(net.switch.bus.at[element])
        if net.switch.et.at[element] == 'b':
            cb.add(net.switch.element.at[element])
    elif et == 't' or et == 'trafo':
        cb.add(net.trafo.hv_bus.at[element])
        cb.add(net.trafo.lv_bus.at[element])
    elif et == 't3' or et == 'trafo3w':
        cb.add(net.trafo3w.hv_bus.at[element])
        cb.add(net.trafo3w.mv_bus.at[element])
        cb.add(net.trafo3w.lv_bus.at[element])
    elif et == 'i' or et == 'impedance':
        cb.add(net.impedance.from_bus.at[element])
        cb.add(net.impedance.to_bus.at[element])

    if respect_in_service:
        cb -= set(net.bus[~net.bus.in_service].index)

    return cb


def get_connected_switches(net, buses, consider=('b', 'l', 't', 't3'), status="all"):
    """
    Returns switches connected to given buses.

    INPUT:
        **net** (pandapowerNet)

        **buses** (single integer or iterable of ints)

    OPTIONAL:
        **consider** (iterable, ("l", "s", "t", "t3))    - Determines, which types of connections
                                                      will be considered.
                                                      l: lines
                                                      b: bus-bus-switches
                                                      t: transformers
                                                      t3: 3W transformers

        **status** (string, ("all", "closed", "open"))    - Determines, which switches will
                                                            be considered
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


def false_elm_links(net, elm, col, target_elm):
    """
    Returns which indices have links to elements of other element tables which does not exist in the
    net.

    EXAMPLE 1:
        elm = "line"
        col = "to_bus"
        target_elm = "bus"

    EXAMPLE 2:
        elm = "poly_cost"
        col = "element"
        target_elm = net["poly_cost"]["et"]
    """
    if isinstance(target_elm, str):
        return net[elm][col].index[~net[elm][col].isin(net[target_elm].index)]
    else:  # target_elm is an iterable, e.g. a Series such as net["poly_cost"]["et"]
        df = pd.DataFrame({"element": net[elm][col].values, "et": target_elm,
                           "indices": net[elm][col].index.values})
        df = df.set_index("et")
        false_links = pd.Index([])
        for et in df.index:
            false_links = false_links.union(pd.Index(df.loc[et].indices.loc[
                ~df.loc[et].element.isin(net[et].index)]))
        return false_links


def false_elm_links_loop(net, elms=None):
    """
    Returns a dict of elements which indices have links to elements of other element tables which
    does not exist in the net.
    This function is an outer loop for get_false_links() applications.
    """
    false_links = dict()
    elms = elms if elms is not None else pp_elements(bus=False, cost_tables=True)
    bebd = branch_element_bus_dict(include_switch=True)
    for elm in elms:
        if net[elm].shape[0]:
            fl = pd.Index([])
            # --- define col and target_elm
            if elm in bebd.keys():
                for col in bebd[elm]:
                    fl = fl.union(false_elm_links(net, elm, col, "bus"))
            elif elm in {"poly_cost", "pwl_cost"}:
                fl = fl.union(false_elm_links(net, elm, "element", net[elm]["et"]))
            elif elm == "measurement":
                fl = fl.union(false_elm_links(net, elm, "element", net[elm]["element_type"]))
            else:
                fl = fl.union(false_elm_links(net, elm, "bus", "bus"))
            if len(fl):
                false_links[elm] = fl
    return false_links


def read_from_net(net, element, index, variable, flag='auto'):
    """
    Reads values from the specified element table at the specified index in the column according to the specified variable
    Chooses the method to read based on flag

    Parameters
    ----------
    net
    element : str
        element table in pandapower net; can also be a results table
    index : int or array_like
        index of the element table where values are read from
    variable : str
        column of the element table
    flag : str
        defines which underlying function to use, can be one of ['auto', 'single_index', 'all_index', 'loc', 'object']

    Returns
    -------
    values
        the values of the variable for the element table according to the index
    """
    if flag == "single_index":
        return _read_from_single_index(net, element, variable, index)
    elif flag == "all_index":
        return _read_from_all_index(net, element, variable)
    elif flag == "loc":
        return _read_with_loc(net, element, variable, index)
    elif flag == "object":
        return _read_from_object_attribute(net, element, variable, index)
    elif flag == "auto":
        auto_flag, auto_variable = _detect_read_write_flag(net, element, index, variable)
        return read_from_net(net, element, index, auto_variable, auto_flag)
    else:
        raise NotImplementedError("read: flag must be one of ['auto', 'single_index', 'all_index', 'loc', 'object']")


def write_to_net(net, element, index, variable, values, flag='auto'):
    """
    Writes values to the specified element table at the specified index in the column according to the specified variable
    Chooses the method to write based on flag

    Parameters
    ----------
    net
    element : str
        element table in pandapower net
    index : int or array_like
        index of the element table where values are written to
    variable : str
        column of the element table
    flag : str
        defines which underlying function to use, can be one of ['auto', 'single_index', 'all_index', 'loc', 'object']

    Returns
    -------
    None
    """
    # write functions faster, depending on type of element_index
    if flag == "single_index":
        _write_to_single_index(net, element, index, variable, values)
    elif flag == "all_index":
        _write_to_all_index(net, element, variable, values)
    elif flag == "loc":
        _write_with_loc(net, element, index, variable, values)
    elif flag == "object":
        _write_to_object_attribute(net, element, index, variable, values)
    elif flag == "auto":
        auto_flag, auto_variable = _detect_read_write_flag(net, element, index, variable)
        write_to_net(net, element, index, auto_variable, values, auto_flag)
    else:
        raise NotImplementedError("write: flag must be one of ['auto', 'single_index', 'all_index', 'loc', 'object']")


def _detect_read_write_flag(net, element, index, variable):
    if variable.startswith('object'):
        # write to object attribute
        return "object", variable.split(".")[1]
    elif isinstance(index, numbers.Number):
        # use .at if element_index is integer for speedup
        return "single_index", variable
    # commenting this out for now, see issue 609
    # elif net[element].index.equals(Index(index)):
    #     # use : indexer if all elements are in index
    #     return "all_index", variable
    else:
        # use common .loc
        return "loc", variable


# read functions:
def _read_from_single_index(net, element, variable, index):
    return net[element].at[index, variable]


def _read_from_all_index(net, element, variable):
    return net[element].loc[:, variable].values


def _read_with_loc(net, element, variable, index):
    return net[element].loc[index, variable].values


def _read_from_object_attribute(net, element, variable, index):
    if hasattr(index, '__iter__') and len(index) > 1:
        values = np.array(shape=index.shape)
        for i, idx in enumerate(index):
            values[i] = getattr(net[element]["object"].at[idx], variable)
    else:
        values = getattr(net[element]["object"].at[index], variable)
    return values


# write functions:
def _write_to_single_index(net, element, index, variable, values):
    net[element].at[index, variable] = values


def _write_to_all_index(net, element, variable, values):
    net[element].loc[:, variable] = values


def _write_with_loc(net, element, index, variable, values):
    net[element].loc[index, variable] = values


def _write_to_object_attribute(net, element, index, variable, values):
    if hasattr(index, '__iter__') and len(index) > 1:
        for idx, val in zip(index, values):
            setattr(net[element]["object"].at[idx], variable, val)
    else:
        setattr(net[element]["object"].at[index], variable, values)
