from copy import deepcopy
from functools import reduce
import operator
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
import pandapower as pp
import pandapower.topology as top
from pandapower.grid_equivalents.auxiliary import drop_internal_branch_elements, ensure_origin_id
from pandapower.grid_equivalents.get_equivalent import \
    merge_internal_net_and_equivalent_external_net

from pandaplan.core.network_equivalents.get_equivalent import get_equivalent

try:
    from pandaplan.core import pplog as logging
except ImportError:
    import logging
try:
    from pandaplan.core.misc.groups import Group
    group_imported = True
except ImportError:
    group_imported = False
try:
    from simbench import voltlvl_idx
    simbench_imported = True
except ImportError:
    simbench_imported = False
try:
    from Distributed_OPF.network_equivalents.REI_toolbox import get_rei_eq
    dOPF_imported = True
except ImportError:
    dOPF_imported = False

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
        net.bus.zone.loc[area] = i


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
        for idx, ozc in other_zone_cols.iteritems():
            other_zone = boundaries[zone_cols].values[idx, ozc]
            if isinstance(other_zone, np.generic):
                other_zone = other_zone.item()
            if zone == other_zone:
                continue  # this happens if 2 trafo3w connections are in zone
            other_zone_bus = boundaries[buses].values[idx, ozc]
            append_set_to_dict(boundary_buses, {other_zone_bus}, [zone, other_zone])

    if "all" in set(net.bus.zone.values):
        raise ValueError("'all' is not a proper zone name.")  # all is used later for other purpose
    branch_elms = pp.pp_elements(bus=False, bus_elements=False, branch_elements=True,
                                 other_elements=False, res_elements=False)
    branch_tuples = pp.element_bus_tuples(bus_elements=False, branch_elements=True,
                                          res_elements=False) | {("switch", "element")}
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
                this_zone_col = pd.Series(this_zone_col).dropna().astype(int)
                other_zone_col1 = pd.Series(np.ones(this_zone_col.shape, dtype=int),
                                            index=this_zone_col.index) - this_zone_col
                if len(buses) == 3:
                    other_zone_col1.loc[other_zone_col1 < 0] = 0
                    other_zone_col2 = pd.Series(3 * np.ones(this_zone_col.shape, dtype=int),
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


def split_grid_by_bus_zone_with_boundary_branches(net, **kwargs):
    """
    INPUT:
        **net** - pandapower net - In net.bus.zone different zones must be given.

    OPTIONAL:
        ****kwargs** - with_oos_eq_loads (bool)

    OUTPUT:
        **boundary_buses** - dict of boundary buses - for each zone the internal and external
        boundary buses are given. Furthermore the value of the boundary_buses key "all" concludes
        all boundary buses of all zones

        **boundary_branches** - dict of branch elements - for each zone a set of the corresponding
        boundary boundary branches as well as "all" boundary branches

        **nets_i** - dict of pandapower nets - the internal grid of each zone as far as the
        internal boundary buses

        **nets_ib** - dict of pandapower nets - the internal grid and boundary grid of each zone
         as far as the external boundary buses

        **nets_ib0** - dict of pandapower nets - same as nets_ib but with elements at the external
        boundary buses

        **nets_ib_eq_load** - dict of pandapower nets - similar to nets_ib but with equivalent
        loads at the external boundary buses instead of original elements at the external boundary
        buses

        **nets_b** - dict of pandapower nets - the boundary grid connected to each zone
    """
    if "all" in set(net.bus.zone.values):
        raise ValueError("'all' is not a proper zone name.")  # all is used later for other purpose
    boundary_buses, boundary_branches = get_boundaries_by_bus_zone_with_boundary_branches(net)

    nets_i = dict()
    nets_ib = dict()
    nets_ib0 = dict()
    nets_ib_eq_load = dict()
    nets_b = dict()
    n_add_load1 = 1
    n_add_load2 = 1
    for zone in boundary_buses.keys():
        if zone == "all":
            continue
        this_zone_buses = set(net.bus.index[net.bus.zone == zone])

        # --- get splitted grids
        # nets_i (only the internal net, no boundary buses and branches)
        nets_i[zone] = pp.select_subnet(net, this_zone_buses, include_results=True)
        nets_i[zone]["bus"].sort_index(inplace=True)
        if kwargs.get("with_oos_eq_loads", False):
            create_eq_loads(nets_i[zone], boundary_buses[zone]["internal"],
                            idx_start=net.load.index.values.max()+n_add_load2,
                            zone=zone, in_service=False)
            n_add_load2 += (nets_i[zone].load.name == "equivalent load").sum()

        # nets_ib (the internal and the boundary branch including the external boundary buses and
        # their bus elements)
        nets_ib[zone] = pp.select_subnet(
            net, this_zone_buses.union(boundary_buses[zone]["external"]), include_results=True)
        nets_ib[zone]["bus"].sort_index(inplace=True)
        pp.drop_inner_branches(nets_ib[zone], boundary_buses[zone]["external"])

        # nets_ib0 (as nets_ib but without the bus elements at the external boundary buses)
        nets_ib0[zone] = deepcopy(nets_ib[zone])
        pp.drop_elements_at_buses(nets_ib0[zone], boundary_buses[zone]["external"],
                                  branch_elements=False)

        # nets_ib_eq_load (as nets_ib0 but with equivalent loads at the external boundary buses)
        # -> used in decomp approach
        nets_ib_eq_load[zone] = deepcopy(nets_ib0[zone])
        create_eq_loads(nets_ib_eq_load[zone], boundary_buses[zone]["external"],
                        boundary_branches[zone], zone=zone,
                        idx_start=net.load.index.values.max()+n_add_load1)
        n_add_load1 += (nets_ib_eq_load[zone].load.name == "equivalent load").sum()

        # nets_b (only the boundary branches)
        nets_b[zone] = deepcopy(nets_ib[zone])
        full_drop_buses = nets_i[zone].bus.index.difference(boundary_buses["all"])
        simple_drop_buses = nets_i[zone].bus.index.intersection(boundary_buses["all"])
        pp.drop_buses(nets_b[zone], full_drop_buses, drop_elements=True)
        pp.drop_buses(nets_b[zone], simple_drop_buses, drop_elements=False)
        drop_internal_branch_elements(nets_b[zone], simple_drop_buses)
        pp.drop_elements_at_buses(nets_b[zone], simple_drop_buses, branch_elements=False)

    return boundary_buses, boundary_branches, nets_i, nets_ib, nets_ib0, nets_ib_eq_load, nets_b


def create_eq_gens(net, buses, branches=None, idx_start=None, sign=-1,
                   name="equivalent gen", zone=None, other_zone=None, **kwargs):
    """ Same as create_eq_loads """
    return _create_eq_elms(net, buses, "gen", branches=branches, idx_start=idx_start, sign=sign,
                           name=name, zone=zone, other_zone=other_zone, **kwargs)


def create_eq_loads(net, buses, branches=None, idx_start=None, sign=1,
                    name="equivalent load", zone=None, other_zone=None, **kwargs):
    """
    Create loads at 'buses' with p and q values equal to sums of p, q power flows over the given
    branches.

    INPUT:
        **net** - pandapower net to be manipulated

        **buses** (iterable) - buses at which additional loads should be created

        **branches** (dict of (element: set of element indices) or dict of those (with buses as
        keys)) - selection of branches to be considered to sum p and q power flows to be set to the
        loads. If None, within all branches, all connecting branches must be found and are then
        considered for summation.
        Example 1: {'trafo': {0, 1, 2}}
        Example 2: {1: {'trafo': {0, 1}}, 2: {'trafo': {2}} (if buses is [1, 2])

    OPTIONAL:
        **idx_start** (int, None) - Starting index for creating the loads. I.e. if 'idx_start' == 3
        and len(buses) == 2, then the indices of the created loads will be 3 and 4.

        **sign** (1 or -1, 1) - If 1, load get the power which flows out of the branches.

        **name** (str or iterable of strings (with length of buses), 'equivalent load') - Value
        to be set to the new net.load.name

        **zone** (value, None) - This value will be included in net.load.origin_id, if
        this column exist

        **other_zone** (value, None) - This value will be included in net.load.origin_id, if
        this column exist

        ****kwargs** - key word arguments for pp.create_load(), e.g. in_service=False.

    OUTPUT:
        new_idx - list of indices of the new loads
    """
    return _create_eq_elms(net, buses, "load", branches=branches, idx_start=idx_start, sign=sign,
                           name=name, zone=zone, other_zone=other_zone, **kwargs)


def _create_eq_elms(net, buses, elm, branches=None, idx_start=None, sign=1,
                    name=None, zone=None, other_zone=None, **kwargs):
    """
    Internal function of create_eq_loads() or create_eq_gens()
    """
    name = name if name is not None else f"equivalent {elm}"

    # --- check existing results and return if not available
    cols = {"load": ["p_mw", "q_mvar"], "gen": ["p_mw", "vm_pu"]}[elm]
    if len(buses - set(net.res_bus.index)) or net.res_bus.loc[
            buses, cols].isnull().any().any():
        logger.warning(f"No {elm}s could be added to 'net_ib_eq_load' since bus results " +
                       "are missing.")
        return pd.Index([])

    # --- run functionality
    if branches is not None:
        branches_has_buses_keys = not len(set(branches.keys()).symmetric_difference(set(buses)))
    names = pp.ensure_iterability(name, len(buses))

    new_idx = []
    for no, (bus, name) in enumerate(zip(buses, names)):
        bra = branches if branches is None or not branches_has_buses_keys else branches[bus]
        idx = idx_start + no if idx_start is not None else None

        p = sign * get_branch_power(net, bus, "p_mw", bra)
        if elm == "load":
            q = sign * get_branch_power(net, bus, "q_mvar", bra)
            new = pp.create_load(net, bus, p, q, name=name, index=idx, **kwargs)
        elif elm == "gen":
            vm = net.res_bus.vm_pu.at[bus]
            new = pp.create_gen(net, bus, p, vm, name=name, index=idx, **kwargs)
        else:
            raise NotImplementedError(f"elm={elm} is not implemented.")
        if "origin_id" in net[elm].columns:
            net[elm].origin_id.loc[new] = eq_name(elm, other_zone, zone, number=no)
        new_idx.append(new)
    return pd.Index(new_idx)


def get_branch_power(net, bus, power_type, branches_dict=None):
    """
    Sums power of branches connected to 'bus'. The power is summed negative (= how much power flows
    into the bus).

    INPUT:
        **net** - pandapower net

        **bus** (int) - index of the bus whose connected branches power flows are summed

        **power_type** (str) - should be "p_mw" or "q_mvar"

    OPTIONAL:
        **branches_dict** (dict, None) - if given, only branches within 'branches_dict' are
        considered for summing the power. An exemplary input is {"line": {0, 1, 2}, "trafo": {1}}.
    """
    connected_branches = pp.get_connected_elements_dict(
        net, [bus], connected_buses=False, connected_bus_elements=False,
        connected_branch_elements=True, connected_other_elements=False)

    power = 0
    bus_types = ["from_bus", "to_bus", "hv_bus", "lv_bus", "mv_bus"]
    for elm, idxs in connected_branches.items():
        if branches_dict is not None:
            if elm in branches_dict.keys():
                idxs = set(branches_dict[elm]).intersection(set(idxs))
            else:
                continue
        for idx in idxs:
            for bus_type in bus_types:
                if bus_type in net[elm].columns and net[elm][bus_type].at[idx] == bus:
                    col = power_type[0] + "_" + bus_type.split("_")[0] + power_type[1:]
                    power -= net["res_"+elm][col].at[idx]
                    break
    return power


if __name__ == "__main__":
    pass
