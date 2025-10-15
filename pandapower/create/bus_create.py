# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging
from typing import Iterable

from numpy import nan
import numpy.typing as npt

from pandapower.auxiliary import pandapowerNet
from pandapower.plotting.geo import _is_valid_number
from pandapower.pp_types import BusType, Int
from pandapower.create._utils import (
    _add_to_entries_if_not_nan,
    _geodata_to_geo_series,
    _get_index_with_check,
    _get_multiple_index_with_check,
    _set_entries,
    _set_multiple_entries,
    _set_value_if_not_nan
)

logger = logging.getLogger(__name__)


def create_bus(
    net: pandapowerNet,
    vn_kv: float,
    name: str | None = None,
    index: Int | None = None,
    geodata: tuple[float, float] | None = None,
    type: BusType = "b",
    zone: str | None = None,
    in_service: bool = True,
    max_vm_pu: float = nan,
    min_vm_pu: float = nan,
    coords: list[tuple[float, float]] | None = None,
    **kwargs
) -> Int:
    """
    Adds one bus in table net["bus"].

    Buses are the nodes of the network that all other elements connect to.

    INPUT:
        **net** (pandapowerNet) - The pandapower network in which the element is created

        **vn_kv** (float) - The grid voltage level.

    OPTIONAL:
        **name** (string, default None) - the name for this bus

        **index** (int, default None) - Force a specified ID if it is available. If None, the \
            index one higher than the highest already existing index is selected.

        **geodata** ((x,y)-tuple, default None) - coordinates used for plotting

        **type** (string, default "b") - Type of the bus. "n" - node,
        "b" - busbar, "m" - muff

        **zone** (string, None) - grid region

        **in_service** (boolean) - True for in_service or False for out of service

        **max_vm_pu** (float, NAN) - Maximum bus voltage in p.u. - necessary for OPF

        **min_vm_pu** (float, NAN) - Minimum bus voltage in p.u. - necessary for OPF

        **coords** (list (len=2) of tuples (len=2), default None) - busbar coordinates to plot
        the bus with multiple points. coords is typically a list of tuples (start and endpoint of
        the busbar) - Example: [(x1, y1), (x2, y2)]

    OUTPUT:
        **index** (int) - The unique ID of the created element

    EXAMPLE:
        create_bus(net, 20., name="bus1")
    """
    index = _get_index_with_check(net, "bus", index)

    if geodata is not None:
        if isinstance(geodata, tuple):
            if len(geodata) != 2:
                raise UserWarning("geodata must be given as (x, y) tuple")
            elif not _is_valid_number(geodata[0]):
                raise UserWarning("geodata x must be a valid number")
            elif not _is_valid_number(geodata[1]):
                raise UserWarning("geodata y must be a valid number")
            geo = f'{{"coordinates":[{geodata[0]},{geodata[1]}], "type":"Point"}}'
        else:
            raise UserWarning("geodata must be a valid coordinate tuple")
    else:
        geo = None

    if coords is not None:
        raise UserWarning("busbar plotting is not implemented fully and will likely be removed in the future")

    entries = {"name": name, "vn_kv": vn_kv, "type": type, "zone": zone, "in_service": in_service, "geo": geo, **kwargs}
    _set_entries(net, "bus", index, True, entries=entries)

    # column needed by OPF. 0. and 2. are the default maximum / minimum voltages
    _set_value_if_not_nan(net, index, min_vm_pu, "min_vm_pu", "bus", default_val=0.)
    _set_value_if_not_nan(net, index, max_vm_pu, "max_vm_pu", "bus", default_val=2.)

    return index


def create_bus_dc(
    net: pandapowerNet,
    vn_kv: float,
    name: str | None = None,
    index: Int | None = None,
    geodata: tuple[float, float] | None = None,
    type: BusType = "b",
    zone: str | None = None,
    in_service: bool = True,
    max_vm_pu: float = nan,
    min_vm_pu: float = nan,
    coords: list[tuple[float, float]] | None = None,
    **kwargs
) -> Int:
    """
    Adds one dc bus in table net["bus_dc"].

    Buses are the nodes of the network that all other elements connect to.

    INPUT:
        **net** (pandapowerNet) - The pandapower network in which the element is created

        **vn_kv** (float) - The grid voltage level.

    OPTIONAL:
        **name** (string, default None) - the name for this dc bus

        **index** (int, default None) - Force a specified ID if it is available. If None, the \
            index one higher than the highest already existing index is selected.

        **geodata** ((x,y)-tuple, default None) - coordinates used for plotting

        **type** (string, default "b") - Type of the bus. "n" - node,
        "b" - busbar, "m" - muff

        **zone** (string, None) - grid region

        **in_service** (boolean) - True for in_service or False for out of service

        **max_vm_pu** (float, NAN) - Maximum dc bus voltage in p.u. - necessary for OPF

        **min_vm_pu** (float, NAN) - Minimum dc bus voltage in p.u. - necessary for OPF

        **coords** (list (len=2) of tuples (len=2), default None) - busbar coordinates to plot
        the dc bus with multiple points. coords is typically a list of tuples (start and endpoint of
        the busbar) - Example: [(x1, y1), (x2, y2)]

    OUTPUT:
        **index** (int) - The unique ID of the created element

    EXAMPLE:
        create_bus_dc(net, 20., name="bus1")
    """
    index = _get_index_with_check(net, "bus_dc", index)

    if geodata is not None:
        if isinstance(geodata, tuple):
            if len(geodata) != 2:
                raise UserWarning("geodata must be given as (x, y) tuple")
            elif not _is_valid_number(geodata[0]):
                raise UserWarning("geodata x must be a valid number")
            elif not _is_valid_number(geodata[1]):
                raise UserWarning("geodata y must be a valid number")
            else:
                geo = f'{{"coordinates":[{geodata[0]},{geodata[1]}], "type":"Point"}}'
        else:
            raise UserWarning("geodata must be a valid coordinate tuple")
    else:
        geo = None

    if coords is not None:
        raise UserWarning("busbar plotting is not implemented fully and will likely be removed in the future")

    entries = {"name": name, "vn_kv": vn_kv, "type": type, "zone": zone, "in_service": in_service, "geo": geo, **kwargs}
    _set_entries(net, "bus_dc", index, True, entries=entries)

    # column needed by OPF. 0. and 2. are the default maximum / minimum voltages
    _set_value_if_not_nan(net, index, min_vm_pu, "min_vm_pu", "bus_dc", default_val=0.)
    _set_value_if_not_nan(net, index, max_vm_pu, "max_vm_pu", "bus_dc", default_val=2.)

    return index


def create_buses(
    net: pandapowerNet,
    nr_buses: int,
    vn_kv: float | Iterable[float],
    index: Int | Iterable[Int] | None = None,
    name: Iterable[str] | None = None,
    type: BusType | Iterable[BusType] = "b",
    geodata: Iterable[tuple[float, float]] | None = None,
    zone: str | Iterable[str] | None = None,
    in_service: bool | Iterable[bool] = True,
    max_vm_pu: float | Iterable[float] = nan,
    min_vm_pu: float | Iterable[float] = nan,
    coords: list[list[tuple[float, float]]] | None = None,
    **kwargs
) -> npt.NDArray[Int]:
    """
    Adds several buses in table net["bus"] at once.

    Buses are the nodal points of the network that all other elements connect to.

    Input:
        **net** (pandapowerNet) - The pandapower network in which the element is created

        **nr_buses** (int) - The number of buses that is created

        **vn_kv** (float) - The grid voltage level.

    OPTIONAL:
        **name** (list of string, default None) - the name for this bus

        **index** (list of int, default None) - Force specified IDs if available. If None, the indices \
            higher than the highest already existing index are selected.

        **geodata** ((x,y)-tuple or Iterable of (x, y)-tuples with length == nr_buses,
            default None) - coordinates used for plotting

        **type** (string, default "b") - Type of the bus. "n" - auxiliary node,
        "b" - busbar, "m" - muff

        **zone** (string, None) - grid region

        **in_service** (list of boolean) - True for in_service or False for out of service

        **max_vm_pu** (list of float, NAN) - Maximum bus voltage in p.u. - necessary for OPF

        **min_vm_pu** (list of float, NAN) - Minimum bus voltage in p.u. - necessary for OPF

        **coords** (list (len=nr_buses) of list (len=2) of tuples (len=2), default None) - busbar
            coordinates to plot the bus with multiple points. coords is typically a list of tuples
            (start and endpoint of the busbar) - Example for 3 buses:
            [[(x11, y11), (x12, y12)], [(x21, y21), (x22, y22)], [(x31, y31), (x32, y32)]]


    OUTPUT:
        **index** (numpy.ndarray (int)) - The unique indices IDs of the created elements
    """
    index = _get_multiple_index_with_check(net, "bus", index, nr_buses)

    if geodata:
        if isinstance(geodata, tuple) and (isinstance(geodata[0], int) or isinstance(geodata[0], float)):
            geo = _geodata_to_geo_series([geodata], nr_buses)
        else:
            assert hasattr(geodata, "__iter__"), "geodata must be an iterable"
            geo = _geodata_to_geo_series(geodata, nr_buses)
    else:
        geo = [None] * nr_buses  # type: ignore[list-item,assignment]

    if coords:
        raise UserWarning("busbar plotting is not implemented fully and will likely be removed in the future")

    entries = {"vn_kv": vn_kv, "type": type, "zone": zone, "in_service": in_service, "name": name, "geo": geo, **kwargs}
    _add_to_entries_if_not_nan(net, "bus", entries, index, "min_vm_pu", min_vm_pu)
    _add_to_entries_if_not_nan(net, "bus", entries, index, "max_vm_pu", max_vm_pu)
    _set_multiple_entries(net, "bus", index, entries=entries)
    net.bus.loc[net.bus.geo == "", "geo"] = None  # overwrite

    return index


def create_buses_dc(
    net: pandapowerNet,
    nr_buses_dc: int,
    vn_kv: float | Iterable[float],
    index: Int | Iterable[Int] | None = None,
    name: Iterable[str] | None = None,
    type: BusType | Iterable[BusType] = "b",
    geodata: Iterable[tuple[float, float]] | None = None,
    zone: str | None= None,
    in_service: bool | Iterable[bool] = True,
    max_vm_pu: float | Iterable[float] = nan,
    min_vm_pu: float | Iterable[float] = nan,
    coords: list[list[tuple[float, float]]] | None = None,
    **kwargs
) -> npt.NDArray[Int]:
    """
    Adds several dc buses in table net["bus_dc"] at once.

    Buses are the nodal points of the network that all other elements connect to.

    Input:
        **net** (pandapowerNet) - The pandapower network in which the element is created

        **nr_buses_dc** (int) - The number of dc buses that is created

        **vn_kv** (float) - The grid voltage level.

    OPTIONAL:
        **index** (list of int, default None) - Force specified IDs if available. If None, the indices \
            higher than the highest already existing index are selected.

        **name** (list of string, default None) - the name for this dc bus

        **type** (string, default "b") - Type of the dc bus. "n" - auxilary node,
        "b" - busbar, "m" - muff

        **geodata** ((x,y)-tuple or list of tuples with length == nr_buses_dc, default None) -
        coordinates used for plotting

        **zone** (string, None) - grid region

        **in_service** (list of boolean) - True for in_service or False for out of service

        **max_vm_pu** (list of float, NAN) - Maximum bus voltage in p.u. - necessary for OPF

        **min_vm_pu** (list of float, NAN) - Minimum bus voltage in p.u. - necessary for OPF

        **coords** (list (len=nr_buses_dc) of list (len=2) of tuples (len=2), default None) - busbar
        coordinates to plot the dc bus with multiple points. coords is typically a list of tuples
        (start and endpoint of the busbar) - Example for 3 dc buses:
        [[(x11, y11), (x12, y12)], [(x21, y21), (x22, y22)], [(x31, y31), (x32, y32)]]


    OUTPUT:
        **index** (numpy.ndarray (int)) - The unique indices ID of the created elements

    EXAMPLE:
        create_buses_dc(net, 2, [20., 20.], name=["bus1","bus2"])
    """
    index = _get_multiple_index_with_check(net, "bus_dc", index, nr_buses_dc)

    if geodata:
        if isinstance(geodata, tuple) and (isinstance(geodata[0], int) or isinstance(geodata[0], float)):
            geo = _geodata_to_geo_series([geodata], nr_buses_dc)
        else:
            assert hasattr(geodata, "__iter__"), "geodata must be an iterable"
            geo = _geodata_to_geo_series(geodata, nr_buses_dc)
    else:
        geo = [None] * nr_buses_dc  # type: ignore[list-item,assignment]

    if coords:
        raise UserWarning("busbar plotting is not implemented fully and will likely be removed in the future")

    entries = {"vn_kv": vn_kv, "type": type, "zone": zone, "in_service": in_service, "name": name, "geo": geo, **kwargs}
    _add_to_entries_if_not_nan(net, "bus_dc", entries, index, "min_vm_pu", min_vm_pu)
    _add_to_entries_if_not_nan(net, "bus_dc", entries, index, "max_vm_pu", max_vm_pu)
    _set_multiple_entries(net, "bus_dc", index, entries=entries)

    return index

