# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Detection, connection and removal of islanded grid groups in a JAO net."""

import logging
from functools import reduce

import pandas as pd

from pandapower.converter.jao._common import _get_bus_idx
from pandapower.network import pandapowerNet
from pandapower.toolbox.grid_modification import drop_buses, fuse_buses
from pandapower.topology import connected_components, create_nxgraph

logger = logging.getLogger(__name__)


def get_grid_groups(net: pandapowerNet, **kwargs) -> pd.DataFrame:
    """Returns a DataFrame of the connected components ("grid groups") of the net."""
    notravbuses_dict = {} if "notravbuses" not in kwargs else {"notravbuses": kwargs.pop("notravbuses")}
    grid_group_buses = [set_ for set_ in connected_components(create_nxgraph(net, **kwargs), **notravbuses_dict)]
    grid_groups = pd.DataFrame({"buses": grid_group_buses})
    grid_groups["n_buses"] = grid_groups["buses"].apply(len)
    return grid_groups


def drop_islanded_grid_groups(net: pandapowerNet, min_bus_number: int | str, **kwargs) -> None:
    """
    Drops grid groups that are islanded and include a number of buses below min_bus_number.

    :param panadpowerNet net: net in which islanded grid groups will be dropped
    :param Optional[int|str] min_bus_number: Threshold value to decide which small grid groups should be dropped and which large grid
        groups should be kept. If all islanded grid groups should be dropped except of the one
        largest, set "max". If all grid groups that do not contain a slack element should be
        dropped, set "unsupplied".
    """

    def _grid_groups_to_drop_by_min_bus_number():
        return grid_groups.loc[grid_groups["n_buses"] < min_bus_number]

    grid_groups = get_grid_groups(net, **kwargs)

    if min_bus_number == "unsupplied":
        slack_buses = set(net.ext_grid.loc[net.ext_grid.in_service, "bus"]) | set(
            net.gen.loc[net.gen.in_service & net.gen.slack, "bus"]
        )
        grid_groups_to_drop = grid_groups.loc[~grid_groups.buses.apply(lambda x: not x.isdisjoint(slack_buses))]

    elif min_bus_number == "max":
        min_bus_number = grid_groups["n_buses"].max()
        grid_groups_to_drop = _grid_groups_to_drop_by_min_bus_number()

    elif isinstance(min_bus_number, int):
        grid_groups_to_drop = _grid_groups_to_drop_by_min_bus_number()

    else:
        raise NotImplementedError(f"{min_bus_number=} is not implemented. Use an int, 'max', or 'unsupplied' instead.")

    buses_to_drop = reduce(set.union, grid_groups_to_drop.buses)
    drop_buses(net, buses_to_drop)
    logger.info(
        f"drop_islanded_grid_groups() drops {len(grid_groups_to_drop)} grid groups with a "
        f"total of {grid_groups_to_drop.n_buses.sum()} buses."
    )


def _invent_connections_between_grid_groups(
    net: pandapowerNet, minimal_trafo_invention: bool = False, **kwargs
) -> None:
    """
    Adds connections between islanded grid groups via:

    - adding transformers between equally named buses that have different voltage level and lay in different groups
    - merge buses of same voltage level, different grid groups and equal name base
    - fuse buses that are close to each other

    :param pandapowerNet net: net to be manipulated
    :param Optional[bool] minimal_trafo_invention: if True, adding transformers stops when no grid groups is islanded anymore (does not apply
        for release version 5 or 6, i.e. it does not care what value is passed to
        minimal_trafo_invention). If False, all equally named buses that have different voltage
        level and lay in different groups will be connected via additional transformers,
        (default: False)
    """
    grid_groups = get_grid_groups(net)
    bus_idx = _get_bus_idx(net)
    bus_grid_groups = pd.concat(
        [pd.Series(group, index=buses) for group, buses in zip(grid_groups.index, grid_groups.buses)]
    ).sort_index()

    # treat for example "Wuergau" equally as "Wuergau (2)":
    location_names = pd.Series(bus_idx.index.get_level_values(0))
    location_names = location_names.str.replace(r"(.) \([0-9]+\)", r"\1", regex=True)
    bus_idx.index = pd.MultiIndex.from_arrays(
        [location_names.values, bus_idx.index.get_level_values(1).to_numpy()], names=bus_idx.index.names
    )

    # --- add Transformers between equally named buses that have different voltage level and lay in
    # --- different groups
    connected_vn_kvs_by_trafos = (
        pd.DataFrame(
            {
                "hv": net.bus.vn_kv.loc[net.trafo.hv_bus].to_list(),
                "lv": net.bus.vn_kv.loc[net.trafo.lv_bus].to_list(),
                "index": net.trafo.index,
            }
        )
        .set_index(["hv", "lv"])
        .sort_index()
    )
    dupl_location_names = location_names[location_names.duplicated()]

    for location_name in dupl_location_names:
        if minimal_trafo_invention and len(bus_grid_groups.unique()) <= 1:
            break  # break with regard to minimal_trafo_invention
        grid_groups_at_location = bus_grid_groups.loc[bus_idx.loc[location_name].values]
        grid_groups_at_location = grid_groups_at_location.drop_duplicates()
        if len(grid_groups_at_location) < 2:
            continue
        elif len(grid_groups_at_location) > 2:
            raise NotImplementedError(
                "Code is not provided to invent Transformer connections "
                "between locations with more than two grid groups, i.e. "
                "voltage levels."
            )
        TSO = net.bus.zone.at[grid_groups_at_location.index[0]]
        vn_kvs = net.bus.vn_kv.loc[grid_groups_at_location.index].sort_values(ascending=False)
        try:
            trafos_connecting_same_voltage_levels = connected_vn_kvs_by_trafos.loc[tuple(vn_kvs)]
        except KeyError:
            logger.info(
                f"For location {location_name}, no transformer data can be reused since "
                f"no transformer connects {vn_kvs.sort_values(ascending=False).iat[0]} kV "
                f"and {vn_kvs.sort_values(ascending=False).iat[1]} kV."
            )
            continue
        trafos_of_same_TSO = trafos_connecting_same_voltage_levels.loc[
            (
                net.bus.zone.loc[net.trafo.hv_bus.loc[trafos_connecting_same_voltage_levels.values.flatten()].values]
                == TSO
            ).values
        ].values.flatten()

        # from which trafo parameters are copied:
        tr_to_be_copied = (
            trafos_of_same_TSO[0]
            if len(trafos_of_same_TSO)
            else trafos_connecting_same_voltage_levels.values.flatten()[0]
        )

        # copy transformer data
        duplicated_row = net.trafo.loc[[tr_to_be_copied]].copy()
        duplicated_row.index = [net.trafo.index.max() + 1]  # adjust index
        duplicated_row.hv_bus = vn_kvs.index[0]  # type: ignore[attr-defined] # adjust hv_bus, lv_bus
        duplicated_row.lv_bus = vn_kvs.index[1]  # type: ignore[attr-defined] # adjust hv_bus, lv_bus
        duplicated_row.name = "additional transformer to connect the grid"  # type: ignore[attr-defined]
        net.trafo = pd.concat([net.trafo, duplicated_row])

        bus_grid_groups.loc[bus_grid_groups == grid_groups_at_location.iat[1]] = grid_groups_at_location.iat[0]

    # --- merge buses of same voltage level, different grid groups and equal name base
    bus_name_splits = net.bus.name.str.split(r"[ -/]+", expand=True)
    buses_with_single_base = net.bus.name.loc[(~bus_name_splits.isnull()).sum(axis=1) == 1]
    for idx, name_base in buses_with_single_base.items():
        same_name_base = net.bus.drop(idx).name.str.contains(name_base)
        if not any(same_name_base):
            continue
        other_group = bus_grid_groups.drop(idx) != bus_grid_groups.at[idx]
        same_vn = net.bus.drop(idx).vn_kv == net.bus.vn_kv.at[idx]
        is_fuse_candidate = same_name_base & other_group & same_vn
        if not any(is_fuse_candidate):
            continue
        to_fuse = bus_grid_groups.drop(idx).loc[is_fuse_candidate].drop_duplicates()
        fuse_buses(net, idx, set(to_fuse.index))

        bus_grid_groups.loc[bus_grid_groups.isin(bus_grid_groups.drop(idx).loc[is_fuse_candidate].unique())] = (
            grid_groups_at_location.iat[0]
        )
        bus_grid_groups = bus_grid_groups.drop(to_fuse.index)

    # --- fuse buses that are close to each other
    for name1, name2 in [
        ("CROISIERE", "BOLLENE (POSTE RESEAU)"),
        ("CAEN", "DRONNIERE (LA)"),
        ("TRINITE-VICTOR", "MENTON/TRINITE VICTOR"),
    ]:
        b1 = net.bus.index[net.bus.name == name1]
        b2 = net.bus.index[net.bus.name == name2]
        if len(b1) == 1 and len(b2) >= 1:
            fuse_buses(net, b1[0], set(b2))
            bus_grid_groups = bus_grid_groups.drop(b2)
        else:
            logger.info(
                f"Buses of the following names were intended to be fused but were not found.\n'{name1}' and '{name2}'"
            )
