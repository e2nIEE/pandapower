from copy import deepcopy
from functools import reduce
import operator
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
import pandapower as pp
import pandapower.topology as top

from pandapower.grid_equivalents.auxiliary import drop_internal_branch_elements, ensure_origin_id
from pandapower.grid_equivalents.get_equivalent import get_equivalent, \
    merge_internal_net_and_equivalent_external_net

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

try:
    from misc.groups import Group

    group_imported = True
except ImportError:
    group_imported = False

try:
    from simbench import voltlvl_idx

    simbench_imported = True
except ImportError:
    simbench_imported = False

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
                setInDict(dict_, keys[:pos + 1], dict())
        else:
            raise ValueError("This function expects a dict for 'getFromDict(dict_, " +
                             str(keys[:pos]) + ")', not a" + str(type(getFromDict(
                dict_, keys[:pos]))))

    # set the value
    setSetInDict(dict_, keys, set_)


def eq_name(eq_type, other_zone=None, zone=None, number=None):
    number_str = "" if number is None else " %i" % number
    st = "%s%s equivalent" % (eq_type, number_str)
    if other_zone is not None:
        st += " of zone "
        if isinstance(other_zone, str):
            st += "'%s'" % other_zone
        else:
            st += str(other_zone)
    if zone is not None:
        st += " at zone "
        if isinstance(zone, str):
            st += "'%s'" % zone
        else:
            st += str(zone)
    return st


def set_bus_zone_by_boundary_branches(net, all_boundary_branches):
    """
    Set integer values (0, 1, 2, ...) to net.bus.zone with regard to the given boundary branches in
    'all_boundary_branches'.

    INPUT:
        **net** - pandapowerNet

        **all_boundary_branches** (dict) - defines which element indices are boundary branches.
            The dict keys must be pandapower elements, e.g. "line" or "trafo"
    """
    include = dict.fromkeys(["line", "dcline", "trafo", "trafo3w", "impedance"])
    for elm in include.keys():
        if elm in all_boundary_branches.keys():
            include[elm] = net[elm].index.difference(all_boundary_branches[elm])
        else:
            include[elm] = True

    mg = top.create_nxgraph(net, include_lines=include["line"], include_impedances=include["impedance"],
                            include_dclines=include["dcline"], include_trafos=include["trafo"],
                            include_trafo3ws=include["trafo3w"])
    cc = top.connected_components(mg)
    ccl = [list(c) for c in cc]
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
            for i, c in enumerate(ccl):
                if np.intersect1d(c, areas[-1]):
                    areas[-1] = np.union1d(areas[-1], ccl.pop(i))

    for i, area in enumerate(areas):
        net.bus.zone.loc[area] = i


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
                    power -= net["res_" + elm][col].at[idx]
                    break
    return power


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
        return pd.Index()

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


def create_eq_gens(net, buses, branches=None, idx_start=None, sign=-1,
                   name="equivalent gen", zone=None, other_zone=None, **kwargs):
    """ Same as create_eq_loads """
    return _create_eq_elms(net, buses, "gen", branches=branches, idx_start=idx_start, sign=sign,
                           name=name, zone=zone, other_zone=other_zone, **kwargs)


def split_grid_by_bus_zone(net, boundary_bus_zones=None, eq_type=None, separate_eqs=True,
                           **kwargs):
    """
    INPUT:
        **net** - pandapower net - In net.bus.zone different zones must be given.

    OPTIONAL:
        **boundary_bus_zones** - strings in net.bus.zone which are to be considered as boundary
        buses

        **eq_type** (str, None) - If given, equivalent elements are added to the boundaries

        **separate_eqs** (bool, True) - Flag whether the equivalents (if eq_type is given)
        should be calculated by each external zone indivudually

        ****kwargs** key word arguments such as "only_connected_groups"

    OUTPUT:
        **nets_ib** - dict of subnets

        **boundary_buses** - dict of boundary buses (details at the docstrings of the subfunctions)
        A difference between with boundary_bus_zones and without is that the letter does
        additionally contain the key "internal".
    """
    if net.bus.zone.isnull().any():
        raise ValueError("There are NaNs in net.bus.zone")
    if boundary_bus_zones is None:
        return nets_ib_by_bus_zone_with_boundary_branches(
            net, eq_type=eq_type, separate_eqs=separate_eqs, **kwargs)
    else:
        return split_grid_by_bus_zone_with_boundary_buses(
            net, boundary_bus_zones, eq_type=eq_type, separate_eqs=separate_eqs, **kwargs)


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
                this_zone_col = np.zeros(boundaries.shape[0]) * np.nan
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


def nets_ib_by_bus_zone_with_boundary_branches(
        net, eq_type=None, separate_eqs=True, duplicated_boundary_bus_elements=True, **kwargs):
    """
    INPUT:
        **net** - pandapower net - In net.bus.zone different zones must be given.

    OPTIONAL:
        **eq_type** (str, None) - If given, equivalent elements are added to the boundaries

        **separate_eqs** (bool, True) - If True and if eq_type is not None, subnets with
        different zones are represented by indivuduals equivalents

        **duplicated_boundary_bus_elements** (bool, True) - if True, bus elements at boundary buses
        are included in all connected nets_ib. If False, only the first nets_ib includes these bus
        elements.

    OUTPUT:
        **nets_ib** - dict of pandapower nets - the internal grid and boundary grid of each zone
         as far as the external boundary buses

        **boundary_buses** - dict of boundary buses - for each zone the internal and external
        boundary buses are given. Furthermore the value of the boundary_buses key "all" concludes
        all boundary buses of all zones

        **boundary_branches** - dict of branch elements - for each zone a set of the corresponding
        boundary boundary branches as well as "all" boundary branches
    """
    eq_type = eq_type if not isinstance(eq_type, str) else eq_type.lower()
    if eq_type is not None and eq_type not in ["load", "gen", "rei", "ward", "xward"]:
        raise ValueError("eq_type %s is unknown." % str(eq_type))

    nets_ib = dict()  # internal + boundary: the internal and up to the external boundary buses
    # are included
    boundary_buses, boundary_branches = get_boundaries_by_bus_zone_with_boundary_branches(net)

    # --- define nets_ib with consideration of eq_type (very similar to part of
    # split_grid_by_bus_zone_with_boundary_buses())
    n_add_elm1 = 1
    fully_included_boundary_buses = set()
    for zone in boundary_buses.keys():
        if zone == "all":
            continue

        this_zone_buses = set(net.bus.index[net.bus.zone == zone])

        # direct calculation of nets_ib and continue
        if eq_type in ["rei", "ward", "xward"] and not separate_eqs:
            nets_ib[zone] = get_equivalent(
                net, eq_type, boundary_buses=boundary_buses[zone]["external"],
                internal_buses=this_zone_buses, elm_col=kwargs.get("elm_col", None))
            continue
        # all other cases comes here: ...

        nets_ib[zone] = pp.select_subnet(
            net, this_zone_buses.union(boundary_buses[zone]["external"]), include_results=True)
        nets_ib[zone]["bus"].sort_index(inplace=True)
        if not duplicated_boundary_bus_elements:
            bb2dr = fully_included_boundary_buses & boundary_buses[zone]["internal"]
            pp.drop_elements_at_buses(nets_ib[zone], bb2dr, branch_elements=False)
            fully_included_boundary_buses |= boundary_buses[zone]["internal"]

        # drop all elements at external boundary buses
        pp.drop_inner_branches(nets_ib[zone], boundary_buses[zone]["external"])
        for elm in pp.pp_elements(bus=False, branch_elements=False, other_elements=False):
            pp.drop_elements_at_buses(nets_ib[zone], boundary_buses[zone]["external"],
                                      branch_elements=False, drop_measurements=False)

        if eq_type is None:
            continue

        elif eq_type in ["load", "gen"] and not separate_eqs:
            create_eq_fct = {"load": create_eq_loads, "gen": create_eq_gens}[eq_type]
            eq_elms = create_eq_fct(nets_ib[zone], boundary_buses[zone]["external"],
                                    idx_start=net[eq_type].index.values.max() + n_add_elm1,
                                    zone=zone)
            n_add_elm1 += len(eq_elms)
            if group_imported:
                elm_col = kwargs.get("elm_col", None)
                if elm_col is not None:
                    eq_elms = nets_ib[zone][eq_type][elm_col].loc[eq_elms]
                Group(nets_ib[zone], {eq_type: eq_elms}, name=eq_name(eq_type, zone=zone),
                      elm_col=elm_col)
            continue

        if not separate_eqs:
            logger.error("With separate_eqs is False, this point should not be"
                         "reached. Hereafter other_zones are iterated")
            continue

        for other_zone, bb in boundary_buses[zone].items():
            if other_zone in ["all", "internal", "external"]:
                continue

            if eq_type in ["load", "gen"] and separate_eqs:
                create_eq_fct = {"load": create_eq_loads, "gen": create_eq_gens}[eq_type]
                eq_elms = create_eq_fct(nets_ib[zone], bb,
                                        idx_start=net[eq_type].index.values.max() + n_add_elm1,
                                        zone=zone, other_zone=other_zone)
                n_add_elm1 += len(eq_elms)
                if group_imported:
                    elm_col = kwargs.get("elm_col", None)
                    if elm_col is not None:
                        eq_elms = nets_ib[zone][eq_type][elm_col].loc[eq_elms]
                    Group(nets_ib[zone], {eq_type: eq_elms},
                          name=eq_name(eq_type, other_zone, zone), elm_col=elm_col)

            elif eq_type in ["rei", "ward", "xward"] and separate_eqs:
                raise NotImplementedError("eq_type '%s' and separate_eqs is %s is not implemented" %
                                          (eq_type, str(separate_eqs)))
            else:
                raise NotImplementedError("This else should not be reached!")

    return nets_ib, boundary_buses, boundary_branches


def split_grid_by_bus_zone_with_boundary_buses(
        net, boundary_bus_zones, eq_type=None, separate_eqs=True, only_connected_groups=False,
        duplicated_boundary_bus_elements=True, **kwargs):
    """
    INPUT:
        **net** (pandapower net) - In net.bus.zone different zones must be given.

        **boundary_bus_zones** - values in net.bus.zone which are to be considered as boundary
        buses

    OPTIONAL:
        **eq_type** (str, None) - If given, equivalent elements are added to the boundaries

        **separate_eqs** (bool, True) - If True and if eq_type is not None, subnets with
        different zones are represented by indivuduals equivalents

        **only_connected_groups** (bool, False) - if True, an error is raised if buses of the same
        zone are not directly connected

        **duplicated_boundary_bus_elements** (bool, True) - if True, bus elements at boundary buses
        are included in all connected nets_ib. If False, only the first nets_ib includes these bus
        elements.

    OUTPUT:
        **nets_ib** - dict of pandapower nets - the internal grid and boundary grid of each zone
         as far as the external boundary buses
        **boundary_buses** - dict of boundary buses - for each zone the external (always equal to
        all - since no internal boundaries are considered here)
        boundary buses are given as well as the external boundary buses for each other zone.
        Furthermore the value of the boundary_buses key "all" concludes all boundary buses of all
        zones.
        Example:
            {"all": {0, 1, 3},
             1: {"all": {3},
                 "external": {3},
                 2: {3}
                 },
             2: {"all": {0, 1, 4},
                 "external": {0, 1, 4},
                 1: {0, 1},
                 3: {4}
                 },
             3: {"all": {2},
                 "external": {2},
                 2: {2}
                 }
             }

        **boundary_branches** - empty dict
    """
    if is_string_dtype(net.bus["zone"]) and "all" in net.bus.zone.values:
        raise ValueError("'all' is not a proper zone name.")  # all is used later for other purpose

    eq_type = eq_type if not isinstance(eq_type, str) else eq_type.lower()
    if eq_type is not None and eq_type not in ["load", "gen", "rei", "ward", "xward"]:
        raise ValueError("eq_type %s is unknown." % str(eq_type))

    boundary_bus_zones = pp.ensure_iterability(boundary_bus_zones)

    # initialize boundary_buses and nets_ib (and boundary_branches)
    boundary_buses = {"all": set(net.bus.index[net.bus.zone.isin(boundary_bus_zones)])}
    boundary_branches = dict()
    nets_ib_buses = dict()
    nets_ib = dict()

    # create topology graphs and bus groups
    mg = top.create_nxgraph(net, nogobuses=boundary_buses["all"])
    cc = top.connected_components(mg)

    # --- check validity + fill nets_ib_buses and boundary_buses
    for bus_group in cc:
        zones = net.bus.zone.loc[bus_group].unique()
        if len(zones) > 1:
            raise ValueError("These zones exist in a group of " + str(len(bus_group)) +
                             " connected bus which should have only one zone: " + str(zones))
        else:
            zone = zones[0] if net.bus.zone.dtype == object else zones[0].item()

        conn_buses = pp.get_connected_buses(net, bus_group)
        conn_other_zone_buses = conn_buses - boundary_buses["all"]

        # raise if other buses than boundary_buses["all"] are boundaries
        if len(conn_other_zone_buses) > 10:
            raise ValueError(str(len(conn_other_zone_buses)) + " buses are connected to zone " +
                             str(zone) + " buses although they are no boundary buses." % zone)
        elif len(conn_other_zone_buses) > 0:
            raise ValueError("Theses buses are connected to zone " + str(
                zone) + " buses although they are no boundary buses: " + str(conn_other_zone_buses))

        if zone in boundary_buses.keys():  # buses of this zone has already been considered in
            # boundary_buses by another bus_group -> same zone without connection

            message = "Zone " + str(zone) + " exist in multiple bus groups. These are the" + \
                      " zones of the bus groups: " + str(zones)
            if only_connected_groups:
                raise ValueError(message)
            else:
                logger.warning(message)

        # fill nets_ib_buses
        append_set_to_dict(nets_ib_buses, bus_group | conn_buses, [zone])

        # fill boundary_buses[zone]["all"] and boundary_buses[zone]["external"] which is the same
        append_set_to_dict(boundary_buses, conn_buses, [zone, "all"])
        boundary_buses[zone]["external"] = boundary_buses[zone]["all"]

    # fill boundary_buses[zone1][zone2]
    for zone1 in nets_ib_buses.keys():
        for zone2 in nets_ib_buses.keys():
            if zone1 == zone2:
                continue
            overlap = boundary_buses[zone1]["all"] & boundary_buses[zone2]["all"]
            if len(overlap):
                append_set_to_dict(boundary_buses, overlap, [zone1, zone2])
                append_set_to_dict(boundary_buses, overlap, [zone2, zone1])

    # --- define nets_ib with consideration of eq_type (very similar to part of
    # split_grid_by_bus_zone_with_boundary_branches())
    n_add_elm1 = 1
    fully_included_boundary_buses = set()
    for zone, buses in nets_ib_buses.items():

        if eq_type in ["rei", "ward", "xward"] and not separate_eqs:
            nets_ib[zone] = get_equivalent(
                net, eq_type, boundary_buses=boundary_buses[zone]["all"],
                internal_buses=net.bus.index[net.bus.zone == zone],
                elm_col=kwargs.get("elm_col", None))
            continue

        nets_ib[zone] = pp.select_subnet(net, buses, include_results=True)
        nets_ib[zone]["bus"].sort_index(inplace=True)
        if not duplicated_boundary_bus_elements:
            bb2dr = fully_included_boundary_buses & boundary_buses[zone]["all"]
            pp.drop_elements_at_buses(nets_ib[zone], bb2dr, branch_elements=False)
            fully_included_boundary_buses |= boundary_buses[zone]["all"]

        if eq_type is None:
            continue

        elif eq_type in ["load", "gen"] and not separate_eqs:
            create_eq_fct = {"load": create_eq_loads, "gen": create_eq_gens}[eq_type]
            eq_elms = create_eq_fct(nets_ib[zone], boundary_buses[zone]["all"],
                                    idx_start=net[eq_type].index.values.max() + n_add_elm1,
                                    zone=zone)
            n_add_elm1 += len(eq_elms)
            if group_imported:
                elm_col = kwargs.get("elm_col", None)
                if elm_col is not None:
                    eq_elms = nets_ib[zone][eq_type][elm_col].loc[eq_elms]
                Group(nets_ib[zone], {eq_type: eq_elms}, name=eq_name(eq_type, zone=zone),
                      elm_col=elm_col)
            continue

        if not separate_eqs:
            logger.error("With separate_eqs is False, this point should not be"
                         "reached. Hereafter other_zones are iterated")
            continue
        for other_zone, bb in boundary_buses[zone].items():
            if other_zone in ["all", "external", "internal"]:
                continue

            if eq_type in ["load", "gen"] and separate_eqs:
                create_eq_fct = {"load": create_eq_loads, "gen": create_eq_gens}[eq_type]
                eq_elms = create_eq_fct(nets_ib[zone], bb,
                                        idx_start=net[eq_type].index.values.max() + n_add_elm1,
                                        zone=zone, other_zone=other_zone)
                n_add_elm1 += len(eq_elms)
                if group_imported:
                    elm_col = kwargs.get("elm_col", None)
                    if elm_col is not None:
                        eq_elms = nets_ib[zone][eq_type][elm_col].loc[eq_elms]
                    Group(nets_ib[zone], {eq_type: eq_elms},
                          name=eq_name(eq_type, other_zone, zone),
                          elm_col=elm_col)

            elif eq_type in ["rei", "ward", "xward"] and separate_eqs:
                raise NotImplementedError("eq_type '%s' and separate_eqs is %s is not implemented" %
                                          (eq_type, str(separate_eqs)))
            else:
                raise NotImplementedError("This else should not be reached!")

    return nets_ib, boundary_buses, boundary_branches


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
                            idx_start=net.load.index.values.max() + n_add_load2,
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
                        idx_start=net.load.index.values.max() + n_add_load1)
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


def get_bus_lookup_by_name(net, bus_lookup_by_name):
    return {net.bus.index[net.bus.name == name][0]: idx for name, idx in
            bus_lookup_by_name.items() if name in net.bus.name.values}


def dict_sum_value(dict1, dict2):
    """
    Return a dict with the sum of values of both input dicts.
    """
    output = deepcopy(dict1)
    dict2 = deepcopy(dict2)
    for key in set(dict2.keys()) & set(output.keys()):
        output[key] += dict2[key]
        del dict2[key]  # to not overwrite the new output values by dict2 in the line
        # "output.update(dict2)"
    output.update(dict2)  # include all values of dict2 which keys are not in dict1
    return output


def is_res_table_with_same_idx(net, elm):
    return "res_" + elm in net.keys() and \
           isinstance(net["res_" + elm], pd.DataFrame) and \
           net["res_" + elm].shape[0] == net[elm].shape[0] and \
           all(net["res_" + elm].index == net[elm].index)


def _sort_from_to_buses(net, elm, idx):
    # determine from and to columns to switch
    from_cols = [col for col in net[elm].columns if "from_" in col]
    to_cols = [col.replace("from", "to") for col in from_cols]
    if elm == "impedance":
        ft_cols = [col for col in net[elm].columns if "ft_" in col]
        tf_cols = [col.replace("ft_", "tf_") for col in ft_cols]
        from_cols += ft_cols
        to_cols += tf_cols
    # for every column which includes "from_", there must be a counterpart:
    assert not set(to_cols) - set(net[elm].columns)
    # sort:
    net[elm].loc[idx, from_cols + to_cols] = net[elm].loc[idx, to_cols + from_cols].values


def sort_from_to_buses(net, elm):
    """ sorts the given element table by from_bus and to_bus columns. """
    if net[elm].shape[0]:
        cols = ["from_bus", "to_bus"]
        assert not set(cols) - set(net[elm].columns)

        idx_to_sort = net[elm].index.values[np.argsort(net[elm][cols].values, axis=1)[
                                            :, 0].astype(bool)]

        if len(idx_to_sort):

            # sort element table
            _sort_from_to_buses(net, elm, idx_to_sort)

            # sort result table
            if is_res_table_with_same_idx(net, elm):
                _sort_from_to_buses(net, "res_" + elm, idx_to_sort)

            # correct side entries in measurement table
            if net["measurement"].shape[0]:
                meas_idx_to_sort = net.measurement.index[
                    (net.measurement.element_type == elm) &
                    (net.measurement.element.isin(idx_to_sort))]
                from_meas_idx_to_sort = meas_idx_to_sort[net.measurement.side.loc[
                                                             meas_idx_to_sort] == "from"]
                net.measurement.loc[from_meas_idx_to_sort] = "to"
                net.measurement.loc[meas_idx_to_sort.difference(from_meas_idx_to_sort)] = "from"


def sort_net_dfs(net, dfs=None):
    """ sorts element dataframes by name and bus columns. """
    dfs = dfs if dfs is not None else pp.pp_elements()
    for elm in dfs:
        if "from_bus" in net[elm].columns and "to_bus" in net[elm].columns:
            sort_from_to_buses(net, elm)

        # determine columns to sort
        cols_to_sort = sorted([bus_col for element, bus_col in pp.element_bus_tuples() if
                               element == elm])
        if "name" in net[elm].columns and not net[elm].name.isnull().all():
            cols_to_sort = ["name"] + cols_to_sort

        if len(cols_to_sort):

            # sort
            net[elm].sort_values(cols_to_sort, inplace=True)

            # sort and reindex result table
            if is_res_table_with_same_idx(net, elm):
                net["res_" + elm] = net["res_" + elm].loc[net[elm].index]
                net["res_" + elm].index = range(net[elm].shape[0])

            # reindex
            net[elm].index = range(net[elm].shape[0])


def split_grid_at_hvmv_connections(net, calc_volt_angles=True):
    """
    A dict of dicts of nets is returned while the input net is changed. The return consists of
    the mv and lv level net part which is removed in the input net.

    OUTPUT EXAMPLE:
        {"a": {0: pp net, 1: pp net}, "b": {2: pp net, 3: pp net, 4: pp net}}
    """
    if not simbench_imported:
        raise ImportError("'simbench' cannot be imported.")

    hvmv_trafos = set(voltlvl_idx(net, "trafo", [1, 3], branch_bus="hv_bus")) & set(
        voltlvl_idx(net, "trafo", [5, 7], branch_bus="lv_bus"))
    hv_mv_trafo3ws = (set(voltlvl_idx(net, "trafo3w", [1, 3], branch_bus="hv_bus")) & set(
        voltlvl_idx(net, "trafo3w", [5, 7], branch_bus="lv_bus")))
    hvmvmv_trafo3ws = hv_mv_trafo3ws & set(voltlvl_idx(net, "trafo3w", [5, 7], branch_bus="mv_bus"))
    hvhvmv_trafo3ws = hv_mv_trafo3ws - hvmvmv_trafo3ws

    # all mv boundary buses
    mv_bbs = set(net.trafo.lv_bus.loc[hvmv_trafos]) | set(net.trafo3w.lv_bus.loc[
                                                              hv_mv_trafo3ws]) | set(
        net.trafo3w.mv_bus.loc[hvmvmv_trafo3ws])

    mg = top.create_nxgraph(net, include_trafos=net.trafo.index.difference(hvmv_trafos),
                            include_trafo3ws=net.trafo3w.index.difference(hv_mv_trafo3ws))
    cc = top.connected_components(mg)
    hv_net_extist = False
    iterators = dict()
    eq_sgens = pd.DataFrame(columns=["name", "bus", "p", "q"])

    # --- create mv_nets
    mv_nets = dict()
    for i, busgroup in enumerate(cc):

        # check correctness of vn_kv values
        vn_kvs = net.bus.vn_kv.loc[busgroup]
        if not len(vn_kvs):
            raise ValueError("busgroup is empty")
        if (vn_kvs >= 70).all():  # it is a pure EHV-HV grid
            hv_net_extist = True
            continue
        elif not (vn_kvs < 70).all():  # it is neither a pure EHV-HV grid nor a pure MV-LV grid
            raise ValueError("The subnets should include only (EHV and HV) or (MV and LV). " +
                             "However, this subnet include these voltages: " + str(vn_kvs))

        if len(busgroup) == 1:
            continue

        # check zone correctness
        zones = net.bus.zone.loc[busgroup].unique()
        if len(zones) > 1:
            raise ValueError("All MV subnets should have only one zone. This include " + str(zones))

        # store information on controllability
        conn_gen = pp.get_connected_elements(net, "gen", busgroup, respect_switches=False)
        if "controllable" not in net.gen.columns and len(conn_gen) or \
                "controllable" in net.gen.columns and net.gen.controllable.loc[conn_gen].any() or \
                "controllable" in net.sgen.columns and net.sgen.controllable.any() and \
                net.sgen.controllable.loc[pp.get_connected_elements(
                    net, "sgen", busgroup, respect_switches=False)].any() or \
                "controllable" in net.load.columns and net.load.controllable.any() and \
                net.load.controllable.loc[pp.get_connected_elements(
                    net, "load", busgroup, respect_switches=False)].any() or \
                "controllable" in net.storage.columns and net.storage.controllable.any() and \
                net.storage.controllable.loc[pp.get_connected_elements(
                    net, "storage", busgroup, respect_switches=False)].any():
            pass
        else:
            continue

        if zones[0] not in iterators.keys():
            iterators[zones[0]] = 1
        else:
            iterators[zones[0]] += 1

        # create (mv) subnet
        subnet = pp.select_subnet(net, busgroup, include_results=True, keep_everything_else=False)
        bbs = subnet.bus.index.intersection(mv_bbs)
        subnet_name = "mv_%s_%i" % (zones[0], iterators[zones[0]])
        n_slacks = subnet.ext_grid.shape[0] + subnet.gen.slack.sum()
        if n_slacks:
            logger.info("Subnet %s already has %i slacks." % (subnet_name, n_slacks))
        if len(bbs) > 1:
            logger.info("In subnet %s, %i boundary " % (subnet_name, len(bbs)) +
                        "buses exist. Therefore %s ext_grids " % len(bbs) +
                        "will be added.")

        # replace gens by sgens and create ext_grid at boundary buses (bbs)
        pp.replace_gen_by_sgen(subnet, gens=pp.get_connected_elements(subnet, "gen", bbs))
        for bb in bbs:
            pp.create_ext_grid(subnet, bb, subnet.res_bus.vm_pu.at[bb],
                               subnet.res_bus.va_degree.at[bb], name="hv_eq")

        # save mv_net
        if zones[0] not in mv_nets:
            mv_nets[zones[0]] = dict()
        mv_nets[zones[0]][iterators[zones[0]]] = subnet
        mv_nets[zones[0]][iterators[zones[0]]].name = subnet_name

        # save eq_sgens information
        for bb in bbs:
            powers = net.res_trafo.loc[pp.get_connected_elements(
                net, "trafo", bb, respect_switches=False) & hvmv_trafos, [
                                           "p_lv_mw", "q_lv_mvar"]].sum(axis=0).values
            powers += net.res_trafo3w.loc[pp.get_connected_elements(
                net, "trafo3w", bb, respect_switches=False) & hv_mv_trafo3ws, [
                                              "p_lv_mw", "q_lv_mvar"]].sum(axis=0).values
            powers += net.res_trafo3w.loc[pp.get_connected_elements(
                net, "trafo3w", bb, respect_switches=False) & hvmvmv_trafo3ws, [
                                              "p_mv_mw", "q_mv_mvar"]].sum(axis=0).values
            eq_sgens = eq_sgens.append({'name': "mv_%s_%i" % (zones[0], iterators[zones[0]]),
                                        'bus': bb,
                                        "p": powers[0],
                                        "q": powers[1]}, ignore_index=True)

    if not hv_net_extist:
        raise ValueError("There is no net group with only EHV and HV buses.")
    hv_net = deepcopy(net)

    # --- drop all elements connected to mv_bbs except hvmv_trafos and hv_mv_trafo3ws (mostly
    # --- copied from pandapower.toolbox)
    pp.drop_elements_at_buses(hv_net, mv_bbs, branch_elements=False)
    for element, columns in pp.branch_element_bus_dict().items():
        for column in columns:
            eid = hv_net[element][hv_net[element][column].isin(mv_bbs)].index
            if element == 'line':
                pp.drop_lines(hv_net, eid)
            elif "trafo" in element:
                not2drop = hvmv_trafos if element == "trafo" else hv_mv_trafo3ws
                pp.drop_trafos(hv_net, eid.difference(not2drop), table=element)
            else:
                n_el = hv_net[element].shape[0]
                hv_net[element].drop(eid, inplace=True)
                # res_element
                res_element = "res_" + element
                if res_element in hv_net.keys() and isinstance(hv_net[res_element], pd.DataFrame):
                    res_eid = hv_net[res_element].index.intersection(eid)
                    hv_net[res_element].drop(res_eid, inplace=True)
                if hv_net[element].shape[0] < n_el:
                    logger.info("dropped %d %s elements" % (n_el - hv_net[element].shape[0],
                                                            element))
    hv_net.switch.drop(pp.get_connected_switches(net, mv_bbs, consider=('b')), inplace=True)

    # --- create sgens equivalent for mv nets

    # controllable mv net eqs
    new_sgen1 = pp.create_sgens(hv_net, eq_sgens.bus.values, eq_sgens.p.values, eq_sgens.q.values,
                                controllable=True, name=eq_sgens.name.values)

    # uncontrollable mv net eqs at hvmv trafos
    sgens_powers = pd.DataFrame(np.zeros((len(mv_bbs), 2)), index=mv_bbs, columns=["p", "q"])
    sgens_powers.loc[hv_net.trafo.lv_bus.loc[hvmv_trafos], ["p", "q"]] += \
        hv_net.res_trafo.loc[hvmv_trafos, ["p_lv_mw", "q_lv_mvar"]].values
    sgens_powers.loc[hv_net.trafo3w.lv_bus.loc[hv_mv_trafo3ws], ["p", "q"]] += \
        hv_net.res_trafo3w.loc[hv_mv_trafo3ws, ["p_lv_mw", "q_lv_mvar"]].values
    sgens_powers.loc[hv_net.trafo3w.mv_bus.loc[hvmvmv_trafo3ws], ["p", "q"]] += \
        hv_net.res_trafo3w.loc[hvmvmv_trafo3ws, ["p_lv_mw", "q_lv_mvar"]].values
    sgens_powers.loc[eq_sgens.bus, ["p", "q"]] -= eq_sgens[["p", "q"]].values
    sgens_powers.drop(sgens_powers.loc[eq_sgens.bus].index[
                          np.isclose(sgens_powers.loc[eq_sgens.bus].values, 0).all(axis=1)], inplace=True)
    new_sgen2 = pp.create_sgens(hv_net, sgens_powers.index, sgens_powers.p.values,
                                sgens_powers.q.values, controllable=False, name="mv_eq")

    # add origin_id to new sgens (and all other elements if missing)
    ensure_origin_id(hv_net)

    # --- drop all inactive elements and run powerflow
    pp.drop_inactive_elements(hv_net)
    pp.runpp(hv_net, calculate_voltage_angles=calc_volt_angles, run_control=True)
    return mv_nets, hv_net


def merge_splitted_grid_by_bus_zone(nets_i, nets_b):
    """ The indices of nets_i should fit to nets_b. """
    net_zones = sorted(nets_i.keys())
    boundary_net = deepcopy(nets_b[net_zones[0]])

    net = deepcopy(nets_i[net_zones[0]])
    # hard_merging_nets(net, nets_i[net_zones[1]], allow_reindexing=False, add_elms={"bus_geodata"})
    net = pp.merge_nets(nets_i[net_zones[0]], nets_i[net_zones[1]], validate=False,
                        create_continuous_bus_indices=bool(nets_i[net_zones[1]].bus.shape[0]))
    pp.reindex_buses(net, {i1: i2 for i1, i2 in zip(net.bus.index.difference(
        nets_i[net_zones[0]].bus.index), nets_i[net_zones[1]].bus.index)})

    dupl_buses = net.bus.index.intersection(boundary_net.bus.index)
    boundary_net.bus.drop(dupl_buses, inplace=True)
    boundary_net.res_bus.drop(boundary_net.res_bus.index.intersection(dupl_buses), inplace=True)
    boundary_net.bus_geodata.drop(boundary_net.bus_geodata.index.intersection(dupl_buses),
                                  inplace=True)
    # hard_merging_nets(net, boundary_net, allow_reindexing=False, add_elms={"bus_geodata"})
    net = pp.merge_nets(net, boundary_net, validate=False,
                        create_continuous_bus_indices=bool(boundary_net.bus.shape[0]))
    return net


def hard_merging_nets(net1, net2, allow_reindexing=True, reindex_first=True, add_elms=None):
    """ appends all element tables of net1 by data from net2. No check of connectivity. """
    add_elms = add_elms if add_elms is not None else set()
    elms = pp.pp_elements(res_elements=True) | {"res_bus"} | add_elms
    for elm in elms:
        dupl_idx = net1[elm].index.intersection(net2[elm].index)
        if len(dupl_idx):
            if elm == "bus" or not allow_reindexing:
                raise ValueError("These indices are in both, net1[%s] and net2[%s]: " % (
                    elm, elm) + str(list(dupl_idx)))
            else:
                max_idx = max(net1[elm].index.values.max(), net2[elm].index.values.max())
                new_index = np.arange(max_idx + 1, max_idx + 1 + len(dupl_idx))
                if reindex_first:
                    pp.reindex_elements(net2, elm, new_index, dupl_idx)
                else:
                    pp.reindex_elements(net1, elm, new_index, dupl_idx)
        net1[elm] = pd.concat([net1[elm], net2[elm]], axis=0, sort=False)
        # net1[elm].sort_index(inplace=True)


def merge_grid_at_hvmv_connections(hv_net, mv_nets):
    hv_net = deepcopy(hv_net)
    mv_nets = deepcopy(mv_nets)

    if "origin_id" in hv_net.bus.columns:
        col = "origin_id"
    else:
        logger.warning("Since 'origin_id' is missing in hv_net.bus.columns, 'name' is used.")
        col = "name"

    for zone, mv_net1 in mv_nets.items():
        for no, mv_net in mv_net1.items():
            assert col in mv_net.bus.columns
            idx_bus = hv_net.bus.index[hv_net.bus[col].isin(mv_net.bus[col])]

            # drop eq sgen in hv_net
            idx_sgen = hv_net.sgen.index[hv_net.sgen.bus.isin(idx_bus) &
                                         (hv_net.sgen.name == mv_net.name)]
            hv_net.sgen.drop(idx_sgen, inplace=True)
            hv_net.res_sgen.drop(hv_net.res_sgen.index.intersection(idx_sgen), inplace=True)

            # drop eq ext_grid in mv_net
            idx_ext_grid = mv_net.ext_grid.index[mv_net.ext_grid.bus.isin(idx_bus) &
                                                 (mv_net.ext_grid.name == "hv_eq")]
            mv_net.ext_grid.drop(idx_ext_grid, inplace=True)

            # drop duplicated bus in hv_net
            hv_net.bus.drop(idx_bus, inplace=True)
            hv_net.res_bus.drop(hv_net.res_bus.index.intersection(idx_bus), inplace=True)
            hv_net.bus_geodata.drop(hv_net.bus_geodata.index.intersection(idx_bus), inplace=True)

            # merge mv_net into hv_net
            hard_merging_nets(hv_net, mv_net, add_elms={"bus_geodata"}, reindex_first=False)
            # hv_net = pp.merge_nets(hv_net, mv_net, validate=False)

    for elm in hv_net.keys():
        if isinstance(hv_net[elm], pd.DataFrame) and hv_net[elm].shape[0]:
            hv_net[elm].sort_index(inplace=True)

    return hv_net


def get_connected_switch_buses(net, boundary_buses):
    boundary_buses_inclusive_bswitch = set()
    mg_sw = top.create_nxgraph(net, include_trafos=False,
                               include_trafo3ws=False,
                               respect_switches=True,
                               include_lines=False,
                               include_dclines=False,
                               include_impedances=False)
    for bbus in boundary_buses:
        boundary_buses_inclusive_bswitch |= set(top.connected_component(mg_sw, bbus))
    return boundary_buses_inclusive_bswitch
