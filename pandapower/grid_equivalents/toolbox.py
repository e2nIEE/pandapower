from copy import deepcopy
from functools import reduce
import operator
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.toolbox
import pandapower.topology as top
from pandapower.grid_equivalents.auxiliary import drop_internal_branch_elements

try:
    from pandaplan.core import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def getFromDict(dict_, keys):
    """ Get value from nested dict """
    return reduce(operator.getitem, keys, dict_)


def setInDict(dict_, keys, value):
    """ Set value to nested dict """
    getFromDict(dict_, keys[:-1])[keys[-1]] = value


def appendSetInDict(dict_, keys, set_):
    """ Use case specific: append existing value of type set in nested dict """
    getFromDict(dict_, keys[:-1])[keys[-1]] |= set_


def setSetInDict(dict_, keys, set_):
    """ Use case specific: set new or append existing value of type set in nested dict """
    if isinstance(getFromDict(dict_, keys[:-1]), dict):
        if keys[-1] in getFromDict(dict_, keys[:-1]).keys():
            if isinstance(getFromDict(dict_, keys), set):
                appendSetInDict(dict_, keys, set_)
            else:
                raise ValueError("The set in the nested dict cannot be appended since it actually "
                                 "is not a set but a " + str(type(getFromDict(dict_, keys))))
        else:
            setInDict(dict_, keys, set_)
    else:
        raise ValueError("This function expects a dict for 'getFromDict(dict_, " + str(keys[:-1]) +
                         ")', not a" + str(type(getFromDict(dict_, keys[:-1]))))


def append_set_to_dict(dict_, set_, keys):
    """ Appends a nested dict by the values of a set, independant if the keys already exist or not.
    """
    keys = pp.ensure_iterability(keys)

    # ensure that the dict way to the last key exist
    for pos, _ in enumerate(keys[:-1]):
        if isinstance(getFromDict(dict_, keys[:pos]), dict):
            if keys[pos] not in getFromDict(dict_, keys[:pos]).keys():
                setInDict(dict_, keys[:pos+1], dict())
        else:
            raise ValueError("This function expects a dict for 'getFromDict(dict_, " +
                             str(keys[:pos]) + ")', not a" + str(type(getFromDict(
                                 dict_, keys[:pos]))))

    # set the value
    setSetInDict(dict_, keys, set_)


def set_bus_zone_by_boundary_branches(net, all_boundary_branches):
    """
    Set integer values (0, 1, 2, ...) to net.bus.zone with regard to the given boundary branches in
    'all_boundary_branches'.

    INPUT:
        **net** - pandapowerNet

        **all_boundary_branches** (dict) - defines which element indices are boundary branches.
            The dict keys must be pandapower elements, e.g. "line" or "trafo"
    """
    include = dict.fromkeys(["line", "trafo", "trafo3w", "impedance"])
    for elm in include.keys():
        if elm in all_boundary_branches.keys():
            include[elm] = net[elm].index.difference(all_boundary_branches[elm])
        else:
            include[elm] = True

    mg = top.create_nxgraph(net, include_lines=include["line"], include_impedances=include[
        "impedance"], include_trafos=include["trafo"], include_trafo3ws=include["trafo3w"])
    cc = top.connected_components(mg)
    ccl = [set_ for set_ in cc]
    areas = []

    while len(ccl):
        # check intersections of the first area with all other unchecked areas (remains in ccl) and
        # then add first area unionized with all intersectioned other areas to "areas"
        areas += [ccl.pop(0)]
        n_last_area = -1
        while n_last_area != len(areas[-1]):
            # check as long as len(areas[-1]) not changes anymore - needed because there can be
            # intersections of remaining areas with the buses added to areas[-1]
            # within the last while loop iteration via union
            n_last_area = len(areas[-1])
            for i, set_ in enumerate(ccl):
                if set_.intersection(areas[-1]):
                    areas[-1] |= ccl.pop(i)

    for i, area in enumerate(areas):
        net.bus.loc[list(area), "zone"] = i


def get_boundaries_by_bus_zone_with_boundary_branches(net):
    """
    Only in_service branches and closed switches are considered.

    INPUT:
        **net** - pandapower net - In net.bus.zone different zones must be given.

    OUTPUT:
        **boundary_buses** - dict of boundary buses - for each zone the internal and external
        boundary buses are given as well as the external boundary buses for each other zone.
        Furthermore the value of the boundary_buses key "all" concludes all boundary buses of all
        zones.
        Example:
            {"all": {0, 1, 3},
             1: {"all": {0, 1, 3},
                 "external": {3},
                 "internal": {0, 1},
                 2: {3}
                 },
             2: {"all": {1, 2, 3, 4},
                 "external": {0, 1, 4},
                 "internal": {2, 3},
                 1: {0, 1},
                 3: {4}
                 },
             3: {"all": {2, 4},
                 "external": {2},
                 "internal": {4},
                 2: {2}
                 }
             }

        **boundary_branches** - dict of branch elements - for each zone a set of the corresponding
        boundary boundary branches as well as "all" boundary branches
        Example:
            {"all": {"line": {0, 1},
                     "trafo": {0}
                     },
             1: {"line": {0},
                 "trafo": {0}
                 },
             2: {"line": {0, 1}},
             3: {"line": {1}}
             }
    """
    def append_boundary_buses_externals_per_zone(boundary_buses, boundaries, zone, other_zone_cols):
        """ iterate throw all boundaries which matches this_zone and add the other_zone_bus to
        boundary_buses """
        for idx, ozc in other_zone_cols.items():
            other_zone = boundaries[zone_cols].values[idx, ozc]
            if isinstance(other_zone, np.generic):
                other_zone = other_zone.item()
            if zone == other_zone:
                continue  # this happens if 2 trafo3w connections are in zone
            other_zone_bus = boundaries[buses].values[idx, ozc]
            append_set_to_dict(boundary_buses, {other_zone_bus}, [zone, other_zone])

    if "all" in set(net.bus.zone.values):
        raise ValueError("'all' is not a proper zone name.")  # all is used later for other purpose
    branch_elms = pandapower.toolbox.pp_elements(bus=False, bus_elements=False, branch_elements=True,
                                                                   other_elements=False, res_elements=False)
    branch_tuples = pandapower.toolbox.element_bus_tuples(bus_elements=False, branch_elements=True,
                                                                            res_elements=False) + [("switch", "element")]
    branch_dict = {branch_elm: [] for branch_elm in branch_elms}
    for elm, bus in branch_tuples:
        branch_dict[elm] += [bus]

    zones = net.bus.zone.unique()
    boundary_branches = {zone if net.bus.zone.dtype == object else zone.item():
                         dict() for zone in zones}
    boundary_branches["all"] = dict()
    boundary_buses = {zone if net.bus.zone.dtype == object else zone.item():
                      {"all": set(), "internal": set(), "external": set()} for zone in zones}
    boundary_buses["all"] = set()

    for elm, buses in branch_dict.items():
        idx = net[elm].index[net[elm].in_service] if elm != "switch" else net.switch.index[
            (net.switch.et == "b") & net.switch.closed]
        boundaries = deepcopy(net[elm][buses].loc[idx])  # copy relevant info from net[elm]

        zone_cols = list()
        for i, bus_col in enumerate(buses):
            # add to which zones the connected buses belong to
            boundaries["zone%i" % i] = net.bus.zone.loc[boundaries[bus_col].values].values
            zone_cols.append("zone%i" % i)
            # compare the zones and conclude if the branches can be boundaries
            if i > 0:
                boundaries["is_boundary"] |= boundaries["zone%i" % i] != boundaries["zone0"]
            else:
                boundaries["is_boundary"] = False
        # reduce the DataFrame 'boundaries' to those branches which actually are boundaries
        boundaries = boundaries.loc[boundaries["is_boundary"],
                                    boundaries.columns.difference(["is_boundary"])]

        # determine boundary_branches and boundary_buses
        if len(boundaries):
            boundary_branches["all"][elm] = set(boundaries.index)
            boundary_buses["all"] |= set(boundaries[buses].values.flatten())

            for zone in set(boundaries[zone_cols].values.flatten()):

                # determine which columns belong to this zone and which not
                this_zone_col = np.zeros(boundaries.shape[0])*np.nan
                for i, _ in enumerate(buses):
                    this_zone_col[boundaries[zone_cols[i]] == zone] = i
                this_zone_col = pd.Series(this_zone_col).dropna().astype(np.int64)
                other_zone_col1 = pd.Series(np.ones(this_zone_col.shape, dtype=np.int64),
                                            index=this_zone_col.index) - this_zone_col
                if len(buses) == 3:
                    other_zone_col1.loc[other_zone_col1 < 0] = 0
                    other_zone_col2 = pd.Series(3 * np.ones(this_zone_col.shape, dtype=np.int64),
                                                index=this_zone_col.index) - \
                        this_zone_col - other_zone_col1

                # fill zone dependant values to boundary_branches and boundary_buses
                boundary_branches[zone][elm] = set(boundaries.index[this_zone_col.index])

                nint = set(boundaries[buses].values[this_zone_col.index, this_zone_col.values])
                ext = set(boundaries[buses].values[other_zone_col1.index, other_zone_col1.values])
                boundary_buses[zone]["internal"] |= nint
                boundary_buses[zone]["external"] |= ext
                boundary_buses[zone]["all"] |= ext | nint
                if len(buses) == 3:
                    ext = set(boundaries[buses].values[
                        other_zone_col2.index, other_zone_col2.values])
                    boundary_buses[zone]["external"] |= ext
                    boundary_buses[zone]["all"] |= ext

                append_boundary_buses_externals_per_zone(
                    boundary_buses, boundaries, zone, other_zone_col1)
                if len(buses) == 3:
                    append_boundary_buses_externals_per_zone(
                        boundary_buses, boundaries, zone, other_zone_col2)

    # check for missing zone connections
    zones_without_connection = list()
    for zone, bra in boundary_branches.items():
        if zone != "all" and not bra:
            zones_without_connection.append(zone)
    if len(zones_without_connection):
        logger.warning("These zones have no connections to other zones: " + str(
            zones_without_connection))

    return boundary_buses, boundary_branches


def get_connected_switch_buses_groups(net, buses):
    all_buses = set()
    bus_dict = []
    mg_sw = top.create_nxgraph(net, include_trafos=False,
                               include_trafo3ws=False,
                               respect_switches=True,
                               include_lines=False,
                               include_impedances=False)
    for bbus in buses:
        if bbus in all_buses:
            continue
        new_bus_set = set(top.connected_component(mg_sw, bbus))
        all_buses |= new_bus_set
        bus_dict.append(list(new_bus_set))
    return all_buses, bus_dict


if __name__ == "__main__":
    pass
