# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging
from collections.abc import Iterable

import numpy.typing as npt
import pandas as pd
from numpy import nan
from typing_extensions import deprecated

from pandapower import pandapowerNet
from pandapower.create._utils import (
    _add_to_entries_if_not_nan,
    _get_multiple_index_with_check,
    _set_multiple_entries,
)
from pandapower.network_schema.tools.validation.network_validation import validate_network
from pandapower.network_structure import get_default_value
from pandapower.plotting.geo import _is_valid_number
from pandapower.pp_types import BusType, Int

logger = logging.getLogger(__name__)

DEFAULT_BUS_TYPE: BusType = get_default_value("bus", "type")
DEFAULT_BUS_DC_TYPE: BusType = get_default_value("bus_dc", "type")


def _geodata_to_geo_series(
    data: Iterable[tuple[float, float]] | None, coords: Iterable[list[list[float]]] | None, nr_buses: int
) -> list[str] | str | None:
    if data is None and coords is None:
        return None
    if data is not None and coords is not None:
        raise ValueError("Cannot specify both geodata and coords")
    geo = []
    if data is not None:
        for g in data:
            if isinstance(g, tuple):
                if len(g) != 2:
                    raise ValueError("geodata tuples must be of length 2")
                elif not _is_valid_number(g[0]):
                    raise UserWarning("geodata x must be a valid number")
                elif not _is_valid_number(g[1]):
                    raise UserWarning("geodata y must be a valid number")
                else:
                    x, y = g
                    geo.append(f'{{"coordinates": [{x}, {y}], "type": "Point"}}')
            else:
                raise TypeError("geodata must be iterable of tuples of (x, y) coordinates")
        if len(geo) == 1:
            geo = [geo[0]] * nr_buses
        if len(geo) != nr_buses:
            raise ValueError("geodata must be a single point or have the same length as nr_buses")
    else:
        if coords is None:
            return None  # unreachable but required for type narrowing
        logger.warning(
            "There is no support for LineString geodata on a bus. Some functionality might not work as intended."
            " Use at your own risk."
        )
        logger.warning("coords will not be verified.")
        geo = [f'{{"coordinates":{c}, "type":"LineString"}}' for c in coords]
    return geo if nr_buses > 1 else geo[0]


@deprecated("Use create_buses with nr_buses=1 instead.")
def create_bus(
    net: pandapowerNet,
    vn_kv: float,
    name: str | None = None,
    index: Int | None = None,
    geodata: tuple[float, float] | None = None,
    type: BusType = DEFAULT_BUS_TYPE,
    zone: str | None = None,
    in_service: bool = get_default_value("bus", "in_service"),
    max_vm_pu: float = nan,
    min_vm_pu: float = nan,
    coords: list[list[float]] | None = None,
    **kwargs,
) -> Int:
    """
    Adds one bus in table net["bus"].

    Buses are the nodes of the network that all other elements connect to.

    Parameters:
        net: The pandapower network in which the element is created
        vn_kv: The grid voltage level.
        name: the name for this bus
        index: Force a specified ID if it is available. If None, the index one higher than the highest already existing
            index is selected.
        geodata: (x, y) tuple coordinates used for plotting
        type:Type of the bus. "n" - node, "b" - busbar, "m" - muff
        zone: grid region
        in_service: True for in_service or False for out of service
        max_vm_pu: Maximum bus voltage in p.u. - necessary for OPF
        min_vm_pu: Minimum bus voltage in p.u. - necessary for OPF
        coords: (no support) list (len=2) of list (len=2) busbar coordinates to plot the bus with multiple points.
            coords is typically a list of tuples (start and endpoint of the busbar) - Example: [(x1, y1), (x2, y2)]

    Returns:
        The unique ID of the created element

    Example:
        >>> create_bus(net, 20., name="bus1")
    """
    return create_buses(
        net, 1, vn_kv, index, name, type, geodata, zone, in_service, max_vm_pu, min_vm_pu, coords, **kwargs
    )[0]


@deprecated("Use create_buses_dc with nr_buses=1 instead.")
def create_bus_dc(
    net: pandapowerNet,
    vn_kv: float,
    name: str | None = None,
    index: Int | None = None,
    geodata: tuple[float, float] | None = None,
    type: BusType = DEFAULT_BUS_DC_TYPE,
    zone: str | None = None,
    in_service: bool = get_default_value("bus_dc", "in_service"),
    max_vm_pu: float = nan,
    min_vm_pu: float = nan,
    coords: list[list[float]] | None = None,
    **kwargs,
) -> Int:
    """
    Adds one dc bus in table net["bus_dc"].

    Buses are the nodes of the network that all other elements connect to.

    Parameters:
        net: The pandapower network in which the element is created
        vn_kv: The grid voltage level.
        name: the name for this dc bus
        index: Force a specified ID if it is available. If None, the \
            index one higher than the highest already existing index is selected.
        geodata: coordinates used for plotting
        type: Type of the bus. "n" - node, "b" - busbar, "m" - muff
        zone: grid region
        in_service: True for in_service or False for out of service
        max_vm_pu: necessary for OPF
        min_vm_pu: necessary for OPF
        coords: busbar coordinates to plot
            the dc bus with multiple points. coords is typically a list of tuples (start and endpoint of
            the busbar) - Example: [(x1, y1), (x2, y2)]

    Returns:
        The unique ID of the created element

    Example:
        >>> create_bus_dc(net, 20., name="bus1")
    """
    return create_buses_dc(
        net, 1, vn_kv, index, name, type, geodata, zone, in_service, max_vm_pu, min_vm_pu, coords, **kwargs
    )[0]


def create_buses(
    net: pandapowerNet,
    nr_buses: int,
    vn_kv: float | Iterable[float],
    index: Int | Iterable[Int] | None = None,
    name: Iterable[str] | None = None,
    type: BusType | Iterable[BusType] = DEFAULT_BUS_TYPE,
    geodata: tuple[float, float] | Iterable[tuple[float, float]] | None = None,
    zone: str | Iterable[str] | None = None,
    in_service: bool | Iterable[bool] = get_default_value("bus", "in_service"),
    max_vm_pu: float | Iterable[float] = nan,
    min_vm_pu: float | Iterable[float] = nan,
    coords: list[list[list[float]]] | None = None,
    skip_validation: bool = False,
    **kwargs,
) -> npt.NDArray[Int]:
    """
    Adds several buses in table net["bus"] at once.

    Buses are the nodal points of the network that all other elements connect to.

    Parameters:
        net: The pandapower network in which the element is created
        nr_buses: The number of buses that is created
        vn_kv: The grid voltage level.
        name: the name for this bus
        index: Force specified IDs if available. If None, the indices higher than the highest already existing index are
            selected.

        geodata: (x,y)-tuple or Iterable of (x, y)-tuples with length == nr_buses, coordinates used for plotting
        type: Type of the buses. "n" - auxiliary node, "b" - busbar, "m" - muff
        zone: grid region
        in_service: True for in_service or False for out of service
        max_vm_pu: necessary for OPF
        min_vm_pu: necessary for OPF
        coords: busbar coordinates to plot the bus with multiple points. coords is typically a list of tuples
            (start and endpoint of the busbar) - Example for 3 buses:
            [[(x11, y11), (x12, y12)], [(x21, y21), (x22, y22)], [(x31, y31), (x32, y32)]]
        skip_validation: if set to true validate_network will not be run after creating.
            (Only use this if performance is critical)


    Returns:
        The IDs of the created elements
    """
    index = _get_multiple_index_with_check(net, "bus", index, nr_buses)

    if geodata:
        if isinstance(geodata, tuple) and isinstance(geodata[0], (int, float)):
            geo = _geodata_to_geo_series([geodata], coords, nr_buses)
        else:
            assert hasattr(geodata, "__iter__"), "geodata must be an iterable"
            geo = _geodata_to_geo_series(geodata, coords, nr_buses)  # type: ignore
    else:
        geo = _geodata_to_geo_series(geodata, coords, nr_buses)

    entries = {"vn_kv": vn_kv, "type": type, "zone": zone, "in_service": in_service, "name": name, "geo": geo, **kwargs}

    min_vm_pu_exists: bool = pd.notna(min_vm_pu) if pd.api.types.is_scalar(min_vm_pu) else pd.notna(min_vm_pu).any()  # type: ignore[attr-defined,arg-type]
    max_vm_pu_exists: bool = pd.notna(max_vm_pu) if pd.api.types.is_scalar(max_vm_pu) else pd.notna(max_vm_pu).any()  # type: ignore[attr-defined,arg-type]
    if min_vm_pu_exists or max_vm_pu_exists or "min_vm_pu" in net.bus.columns:
        _add_to_entries_if_not_nan(
            net,
            "bus",
            entries,
            index,
            "min_vm_pu",
            min_vm_pu,
            default_val=get_default_value("bus", "min_vm_pu"),
        )
        _add_to_entries_if_not_nan(
            net,
            "bus",
            entries,
            index,
            "max_vm_pu",
            max_vm_pu,
            default_val=get_default_value("bus", "max_vm_pu"),
        )
    _set_multiple_entries(net, "bus", index, entries=entries)

    if not skip_validation:
        validate_network(net)
    return index


def create_buses_dc(
    net: pandapowerNet,
    nr_buses_dc: int,
    vn_kv: float | Iterable[float],
    index: Int | Iterable[Int] | None = None,
    name: Iterable[str] | None = None,
    type: BusType | Iterable[BusType] = DEFAULT_BUS_DC_TYPE,
    geodata: Iterable[tuple[float, float]] | None = None,
    zone: str | None = None,
    in_service: bool | Iterable[bool] = get_default_value("bus_dc", "in_service"),
    max_vm_pu: float | Iterable[float] = nan,
    min_vm_pu: float | Iterable[float] = nan,
    coords: list[list[list[float]]] | None = None,
    skip_validation: bool = False,
    **kwargs,
) -> npt.NDArray[Int]:
    """
    Adds several dc buses in table net["bus_dc"] at once.

    Buses are the nodal points of the network that all other elements connect to.

    Parameters:
        net: The pandapower network in which the element is created
        nr_buses_dc: The number of dc buses that is created
        vn_kv: The grid voltage level.
        index: Force specified IDs if available. If None, the indices \
            higher than the highest already existing index are selected.
        name: the name for this dc bus
        type: Type of the bus. "n" - auxiliary node, "b" - busbar, "m" - muff
        geodata: (x,y)-tuple or list of tuples with length == nr_buses_dc, coordinates used for plotting
        zone: grid region
        in_service: True for in_service or False for out of service
        max_vm_pu: necessary for OPF
        min_vm_pu: necessary for OPF
        coords: busbar coordinates to plot the dc bus with multiple points. coords is typically a list of tuples
            (start and endpoint of the busbar) - Example for 3 dc buses:
            [[(x11, y11), (x12, y12)], [(x21, y21), (x22, y22)], [(x31, y31), (x32, y32)]]
        skip_validation: if set to true validate_network will not be run after creating.
            (Only use this if performance is critical)


    Returns:
        The unique indices ID of the created elements

    Example:
        >>> create_buses_dc(net, 2, [20., 20.], name=["bus1","bus2"])
    """
    index = _get_multiple_index_with_check(net, "bus_dc", index, nr_buses_dc)

    if geodata:
        if isinstance(geodata, tuple) and isinstance(geodata[0], (int, float)):
            geo = _geodata_to_geo_series([geodata], coords, nr_buses_dc)
        else:
            assert hasattr(geodata, "__iter__"), "geodata must be an iterable"
            geo = _geodata_to_geo_series(geodata, coords, nr_buses_dc)  # type: ignore
    else:
        geo = _geodata_to_geo_series(geodata, coords, nr_buses_dc)

    entries = {"vn_kv": vn_kv, "type": type, "zone": zone, "in_service": in_service, "name": name, "geo": geo, **kwargs}

    min_vm_pu_exists: bool = pd.notna(min_vm_pu) if pd.api.types.is_scalar(min_vm_pu) else pd.notna(min_vm_pu).any()  # type: ignore[attr-defined,arg-type]
    max_vm_pu_exists: bool = pd.notna(max_vm_pu) if pd.api.types.is_scalar(max_vm_pu) else pd.notna(max_vm_pu).any()  # type: ignore[attr-defined,arg-type]
    if min_vm_pu_exists or max_vm_pu_exists or "min_vm_pu" in net.bus.columns:
        _add_to_entries_if_not_nan(
            net,
            "bus_dc",
            entries,
            index,
            "min_vm_pu",
            min_vm_pu,
            default_val=get_default_value("bus_dc", "min_vm_pu"),
        )
        _add_to_entries_if_not_nan(
            net,
            "bus_dc",
            entries,
            index,
            "max_vm_pu",
            max_vm_pu,
            default_val=get_default_value("bus_dc", "max_vm_pu"),
        )
    _set_multiple_entries(net, "bus_dc", index, entries=entries)

    if not skip_validation:
        validate_network(net)

    return index
