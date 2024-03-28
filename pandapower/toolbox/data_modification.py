# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from collections import defaultdict
import uuid

import numpy as np
import pandas as pd
from pandapower.auxiliary import get_indices
from pandapower.create import create_empty_network
from pandapower.toolbox.comparison import compare_arrays
from pandapower.toolbox.element_selection import element_bus_tuples, pp_elements

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def add_column_from_node_to_elements(net, column, replace, elements=None, branch_bus=None,
                                     verbose=True):
    """
    Adds column data to elements, inferring them from the column data of buses they are
    connected to.

    INPUT:
        **net** (pandapowerNet) - the pandapower net that will be changed

        **column** (string) - name of column that should be copied from the bus table to the element
        table

        **replace** (boolean) - if True, an existing column in the element table will be overwritten

        **elements** (list) - list of elements that should get the column values from the bus table

        **branch_bus** (list) - defines which bus should be considered for branch elements.
        'branch_bus' must have the length of 2. One entry must be 'from_bus' or 'to_bus', the
        other 'hv_bus' or 'lv_bus'

    EXAMPLE:
        compare to add_zones_to_elements()
    """
    branch_bus = ["from_bus", "hv_bus"] if branch_bus is None else branch_bus
    if column not in net.bus.columns:
        raise ValueError("%s is not in net.bus.columns" % column)
    elements = elements if elements is not None else pp_elements(bus=False, other_elements=False)
    elements_to_replace = elements if replace else [
        el for el in elements if column not in net[el].columns or net[el][column].isnull().all()]
    # bus elements
    for element, bus_type in element_bus_tuples(bus_elements=True, branch_elements=False):
        if element in elements_to_replace:
            net[element][column] = net["bus"][column].loc[net[element][bus_type]].values
    # branch elements
    to_validate = {}
    for element, bus_type in element_bus_tuples(bus_elements=False, branch_elements=True):
        if element in elements_to_replace:
            if bus_type in (branch_bus + ["bus"]):  # copy data, append branch_bus for switch.bus
                net[element][column] = net["bus"][column].loc[net[element][bus_type]].values
            else:  # save data for validation
                to_validate[element] = net["bus"][column].loc[net[element][bus_type]].values
    # validate branch elements, but do not validate double and switches at all
    already_validated = ["switch"]
    for element, bus_type in element_bus_tuples(bus_elements=False, branch_elements=True):
        if (element in elements_to_replace) & (element not in already_validated):
            already_validated += [element]
            crossing = sum(~compare_arrays(net[element][column].values, to_validate[element]))
            if crossing > 0:
                if verbose:
                    logger.warning("There have been %i %ss with different " % (crossing, element) +
                                   "%s data at from-/hv- and to-/lv-bus" % column)
                else:
                    logger.debug("There have been %i %ss with different " % (crossing, element) +
                                 "%s data at from-/hv- and to-/lv-bus" % column)


def add_column_from_element_to_elements(net, column, replace, elements=None,
                                        continue_on_missing_column=True):
    """
    Adds column data to elements, inferring them from the column data of the elements linked by the
    columns "element" and "element_type" or "et".

    INPUT:
        **net** (pandapowerNet) - the pandapower net that will be changed

        **column** (string) - name of column that should be copied from the tables of the elements.

        **replace** (boolean) - if True, an existing column will be overwritten

        **elements** (list) - list of elements that should get the column values from the linked
        element tables. If None, all elements with the columns "element" and "element_type" or
        "et" are considered (these are currently "measurement" and "switch").

        **continue_on_missing_column** (Boolean, True) - If False, a error will be raised in case of
        an element table has no column 'column' although this element is refered in 'elements'.
        E.g. 'measurement' is in 'elements' and in net.measurement is a trafo measurement but
        in net.trafo there is no column 'name' although column=='name' - ni this case
        'continue_on_missing_column' acts.

    EXAMPLE:
        import pandapower as pp
        import pandapower.networks as pn
        net = pn.create_cigre_network_mv()
        pp.create_measurement(net, "i", "trafo", 5, 3, 0, side="hv")
        pp.create_measurement(net, "i", "line", 5, 3, 0, side="to")
        pp.create_measurement(net, "p", "bus", 5, 3, 2)
        print(net.measurement.name.values, net.switch.name.values)
        pp.add_column_from_element_to_elements(net, "name", True)
        print(net.measurement.name.values, net.switch.name.values)
    """
    elements = elements if elements is not None else pp_elements()
    elements_with_el_and_et_column = [el for el in elements if "element" in net[el].columns and (
            "element_type" in net[el].columns or "et" in net[el].columns)]
    elements_to_replace = elements_with_el_and_et_column if replace else [
        el for el in elements_with_el_and_et_column if column not in net[el].columns or net[el][
            column].isnull().all()]
    for el in elements_to_replace:
        et_col = "element_type" if "element_type" in net[el].columns else "et"
        element_type = net[el][et_col]
        for short, complete in [("t", "trafo"), ("t3", "trafo3w"), ("l", "line"), ("s", "switch"),
                                ("b", "bus")]:
            element_type.loc[element_type == short] = complete
        element_types_without_column = [et for et in set(element_type) if column not in
                                        net[et].columns]
        if len(element_types_without_column):
            message = "%s is not in net[et].columns with et in " % column + str(
                element_types_without_column)
            if not continue_on_missing_column:
                raise KeyError(message)
            else:
                logger.debug(message)
        for et in list(set(element_type) - set(element_types_without_column)):
            idx_et = element_type.index[element_type == et]
            net[el].loc[idx_et, column] = net[et][column].loc[net[el].element[idx_et]].values


def add_zones_to_elements(net, replace=True, elements=None, **kwargs):
    """
    Adds zones to elements, inferring them from the zones of buses they are connected to.
    """
    elements = ["line", "trafo", "ext_grid", "switch"] if elements is None else elements
    add_column_from_node_to_elements(net, "zone", replace=replace, elements=elements, **kwargs)


def reindex_buses(net, bus_lookup):
    """
    Changes the index of net.bus and considers the new bus indices in all other pandapower element
    tables.

    INPUT:
      **net** - pandapower network

      **bus_lookup** (dict) - the keys are the old bus indices, the values the new bus indices
    """
    not_fitting_bus_lookup_keys = set(bus_lookup.keys()) - set(net.bus.index)
    if len(not_fitting_bus_lookup_keys):
        logger.error("These bus indices are unknown to net. Thus, they cannot be reindexed: " +
                     str(not_fitting_bus_lookup_keys))

    missing_bus_indices = sorted(set(net.bus.index) - set(bus_lookup.keys()))
    if len(missing_bus_indices):
        bus_lookup.update({b: b for b in missing_bus_indices})

    # --- reindex buses
    net.bus.index = get_indices(net.bus.index, bus_lookup)
    net.res_bus.index = get_indices(net.res_bus.index, bus_lookup)
    net.res_bus_3ph.index = get_indices(net.res_bus_3ph.index, bus_lookup)

    # --- adapt link in bus elements
    for element, value in element_bus_tuples():
        net[element][value] = get_indices(net[element][value], bus_lookup)
    net["bus_geodata"].set_index(get_indices(net["bus_geodata"].index, bus_lookup), inplace=True)

    # --- adapt group link
    if net.group.shape[0]:
        for row in np.arange(net.group.shape[0], dtype=np.int64)[
                (net.group.element_type == "bus").values & net.group.reference_column.isnull().values]:
            net.group.element.iat[row] = list(get_indices(net.group.element.iat[row], bus_lookup))

    # --- adapt measurement link
    bus_meas = net.measurement.element_type == "bus"
    net.measurement.loc[bus_meas, "element"] = get_indices(net.measurement.loc[bus_meas, "element"],
                                                           bus_lookup)
    side_meas = pd.to_numeric(net.measurement.side, errors="coerce").notnull()
    net.measurement.loc[side_meas, "side"] = get_indices(net.measurement.loc[side_meas, "side"],
                                                         bus_lookup)

    # --- adapt switch link
    bb_switches = net.switch[net.switch.et == "b"]
    net.switch.loc[bb_switches.index, "element"] = get_indices(bb_switches.element, bus_lookup)

    return bus_lookup


def create_continuous_bus_index(net, start=0, store_old_index=False):
    """
    Creates a continuous bus index starting at 'start' and replaces all
    references of old indices by the new ones.

    INPUT:
      **net** - pandapower network

    OPTIONAL:
      **start** - index begins with "start"

      **store_old_index** - if True, stores the old index in net.bus["old_index"]

    OUTPUT:
      **bus_lookup** - mapping of old to new index
    """
    net.bus.sort_index(inplace=True)
    if store_old_index:
        net.bus["old_index"] = net.bus.index.values
    new_bus_idxs = list(np.arange(start, len(net.bus) + start))
    bus_lookup = dict(zip(net["bus"].index.values, new_bus_idxs))
    reindex_buses(net, bus_lookup)
    return bus_lookup


def reindex_elements(net, element_type, new_indices=None, old_indices=None, lookup=None):
    """
    Changes the index of the DataFrame net[element_type].

    Parameters
    ----------
    net : pp.pandapowerNet
        net with elements to reindex
    element_type : str
        name of element type to rename, e.g. "gen" or "load"
    new_indices : typing.Union[list[int], pandas.Index[int]], optional
        new indices to set, by default None
    old_indices : typing.Union[list[int], pandas.Index[int]], optional
        old indices to be replaced. If not given, all indices are
        assumed in case of given new_indices, and all lookup keys are assumed in case of given
        lookup, by default None
    lookup : dict[int,int], optional
        lookup to assign new indices to old indices, by default None

    Notes
    -----
    Either new_indices or lookup must be given.
    old_indices can be given to limit the indices to be replaced. In case of given new_indices,
    both must have the same length.
    If element_type is "group", be careful to give new_indices without passing old_indices because
    group indices do not need to be unique.

    Examples
    --------
    >>> net = pp.create_empty_network()
    >>> idx0 = pp.create_bus(net, 110)
    >>> idx1 = 4
    >>> idx2 = 7
    >>> # Reindex using 'new_indices':
    >>> pp.reindex_elements(net, "bus", [idx1])  # passing old_indices=[idx0] is optional
    >>> net.bus.index
    Int64Index([4], dtype='int64')
    >>> # Reindex using 'lookup':
    >>> pp.reindex_elements(net, "bus", lookup={idx1: idx2})
    Int64Index([7], dtype='int64')
    """
    if not net[element_type].shape[0]:
        return
    if new_indices is None and lookup is None:
        raise ValueError("Either new_indices or lookup must be given.")
    elif new_indices is not None and lookup is not None:
        raise ValueError("Only one can be considered, new_indices or lookup.")
    if new_indices is not None and not len(new_indices) or lookup is not None and not len(
            lookup.keys()):
        return

    if new_indices is not None:
        old_indices = old_indices if old_indices is not None else net[element_type].index
        assert len(new_indices) == len(old_indices)
        lookup = dict(zip(old_indices, new_indices))
    elif old_indices is None:
        old_indices = net[element_type].index.intersection(lookup.keys())

    if element_type == "bus":
        reindex_buses(net, lookup)
        return

    # --- reindex
    new_index = pd.Series(net[element_type].index, index=net[element_type].index)
    if element_type != "group":
        new_index.loc[old_indices] = get_indices(old_indices, lookup)
    else:
        new_index.loc[old_indices] = get_indices(new_index.loc[old_indices].values, lookup)
    net[element_type].set_index(pd.Index(new_index.values), inplace=True)

    # --- adapt group link
    if net.group.shape[0]:
        for row in np.arange(net.group.shape[0], dtype=np.int64)[
                (net.group.element_type == element_type).values & \
                net.group.reference_column.isnull().values]:
            net.group.element.iat[row] = list(get_indices(net.group.element.iat[row], lookup))

    # --- adapt measurement link
    if element_type in ["line", "trafo", "trafo3w"]:
        affected = net.measurement[(net.measurement.element_type == element_type) &
                                   (net.measurement.element.isin(old_indices))]
        if len(affected):
            net.measurement.loc[affected.index, "element"] = get_indices(affected.element, lookup)

    # --- adapt switch link
    if element_type in ["line", "trafo"]:
        affected = net.switch[(net.switch.et == element_type[0]) &
                              (net.switch.element.isin(old_indices))]
        if len(affected):
            net.switch.loc[affected.index, "element"] = get_indices(affected.element, lookup)

    # --- adapt line_geodata index
    if element_type == "line" and "line_geodata" in net and net["line_geodata"].shape[0]:
        idx_name = net.line_geodata.index.name
        place_holder = uuid.uuid4()
        net["line_geodata"][place_holder] = net["line_geodata"].index
        net["line_geodata"].loc[old_indices.intersection(net.line_geodata.index), place_holder] = (
            get_indices(old_indices.intersection(net.line_geodata.index), lookup))
        net["line_geodata"] = net["line_geodata"].set_index(place_holder)
        net["line_geodata"].index.name = idx_name

    # --- adapt index in cost dataframes
    for cost_df in ["pwl_cost", "poly_cost"]:
        element_in_cost_df = (net[cost_df].et == element_type) & net[cost_df].element.isin(old_indices)
        if sum(element_in_cost_df):
            net[cost_df].loc[element_in_cost_df, "element"] = get_indices(net[cost_df].element[
                element_in_cost_df], lookup)


def create_continuous_elements_index(net, start=0, add_df_to_reindex=set()):
    """
    Creating a continuous index for all the elements, starting at zero and replaces all references
    of old indices by the new ones.

    INPUT:
      **net** - pandapower network with unodered indices

    OPTIONAL:
      **start** - index begins with "start"

      **add_df_to_reindex** - by default all useful pandapower elements for power flow will be
      selected. Customized DataFrames can also be considered here.

    OUTPUT:
      **net** - pandapower network with odered and continuous indices

    """
    element_types = pp_elements(res_elements=True)

    # create continuous bus index
    create_continuous_bus_index(net, start=start)
    element_types -= {"bus", "bus_geodata", "res_bus"}

    element_types |= add_df_to_reindex

    # run reindex_elements() for all element_types
    for et in list(element_types):
        net[et].sort_index(inplace=True)
        new_index = list(np.arange(start, len(net[et]) + start))

        if et in net and isinstance(net[et], pd.DataFrame):
            if et in ["bus_geodata", "line_geodata"]:
                logger.info(et + " don't need to bo included to 'add_df_to_reindex'. It is " +
                            "already included by et=='" + et.split("_")[0] + "'.")
            else:
                reindex_elements(net, et, new_index)
        else:
            logger.debug("No indices could be changed for element '%s'." % et)


def set_scaling_by_type(net, scalings, scale_load=True, scale_sgen=True):
    """
    Sets scaling of loads and/or sgens according to a dictionary
    mapping type to a scaling factor. Note that the type-string is case
    sensitive.
    E.g. scaling = {"pv": 0.8, "bhkw": 0.6}

    :param net:
    :param scalings: A dictionary containing a mapping from element type to
    :param scale_load:
    :param scale_sgen:
    """
    if not isinstance(scalings, dict):
        raise UserWarning("The parameter scaling has to be a dictionary, "
                          "see docstring")

    def scaleit(what):
        et = net[what]
        et["scaling"] = [scale[t] if scale[t] is not None else s for t, s in
                         zip(et.type.values, et.scaling.values)]

    scale = defaultdict(lambda: None, scalings)
    if scale_load:
        scaleit("load")
    if scale_sgen:
        scaleit("sgen")


def set_data_type_of_columns_to_default(net):
    """
    Overwrites dtype of DataFrame columns of PandapowerNet elements to default dtypes defined in
    pandapower. The function "convert_format" does that authomatically for nets saved with
    pandapower versions below 1.6. If this is required for versions starting with 1.6, it should be
    done manually with this function.

    INPUT:
      **net** - pandapower network with unodered indices

    OUTPUT:
      No output; the net passed as input has pandapower-default dtypes of columns in element tables.

    """
    new_net = create_empty_network()
    for key, item in net.items():
        if isinstance(item, pd.DataFrame):
            for col in item.columns:
                if key in new_net and col in new_net[key].columns:
                    if new_net[key][col].dtype == net[key][col].dtype:
                        continue
                    if set(item.columns) == set(new_net[key]):
                        net[key] = net[key].reindex(new_net[key].columns, axis=1)
                    net[key][col] = net[key][col].astype(new_net[key][col].dtype,
                                                         errors="ignore")
