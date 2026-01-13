# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import numpy.typing as npt

from pandapower import pandapowerNet
from pandapower.pp_types import Int, CostElementType, PWLPowerType
from pandapower.create._utils import (
    _cost_existance_check,
    _costs_existance_check,
    _get_index_with_check,
    _get_multiple_index_with_check,
    _set_entries,
    _set_multiple_entries,
)

logger = logging.getLogger(__name__)


def create_pwl_cost(
    net: pandapowerNet,
    element: Int | Iterable[Int],
    et: CostElementType,
    points: list[list[float]],
    power_type: PWLPowerType = "p",
    index: int | None = None,
    check: bool = True,
    **kwargs,
) -> Int:
    """
    Creates an entry for piecewise linear costs for an element. The currently supported elements are
     - Generator
     - External Grid
     - Static Generator
     - Load
     - Dcline
     - Storage

    INPUT:
        **element** (int) - ID of the element in the respective element table

        **et** (string) - element type, one of "gen", "sgen", "ext_grid", "load",
                                "dcline", "storage"

        **points** - (list) list of lists with [[p1, p2, c1], [p2, p3, c2], ...] where c(n) \
                            defines the costs between p(n) and p(n+1)

    OPTIONAL:
        **power_type** - (string) - Type of cost ["p", "q"] are allowed for active or reactive power

        **index** (int, index) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **check** (bool, True) - raises UserWarning if costs already exist to this element.

    OUTPUT:
        **index** (int) - The unique ID of created cost entry

    EXAMPLE:
        The cost function is given by the x-values p1 and p2 with the slope m between those points.\
        The constant part b of a linear function y = m*x + b can be neglected for OPF purposes. \
        The intervals have to be continuous (the starting point of an interval has to be equal to \
        the end point of the previous interval).

        To create a gen with costs of 1€/MW between 0 and 20 MW and 2€/MW between 20 and 30:

        create_pwl_cost(net, 0, "gen", [[0, 20, 1], [20, 30, 2]])
    """
    if isinstance(element, (list, tuple)):
        element = element[0]
    if check and _cost_existance_check(net, element, et, power_type=power_type):
        raise UserWarning(f"There already exist costs for {et} {element}")

    index = _get_index_with_check(net, "pwl_cost", index, "piecewise_linear_cost")

    entries = {"power_type": power_type, "element": element, "et": et, "points": points, **kwargs}
    _set_entries(net, "pwl_cost", index, entries=entries)
    return index


def create_pwl_costs(
    net: pandapowerNet,
    elements: Sequence,
    et: CostElementType | Iterable[str],
    points: list[list[list[float]]],
    power_type: PWLPowerType | Iterable[str] = "p",
    index: int | None = None,
    check: bool = True,
    **kwargs,
) -> npt.NDArray[np.integer]:
    """
    Creates entries for piecewise linear costs for multiple elements. The currently supported elements are
     - Generator
     - External Grid
     - Static Generator
     - Load
     - Dcline
     - Storage

    INPUT:
        **elements** (iterable of integers) - IDs of the elements in the respective element table

        **et** (string or iterable) - element type, one of "gen", "sgen", "ext_grid", "load",
                                "dcline", "storage"

        **points** - (list of lists of lists) with [[p1, p2, c1], [p2, p3, c2], ...] for each element
        where c(n) defines the costs between p(n) and p(n+1)

    OPTIONAL:
        **power_type** - (string or iterable) - Type of cost ["p", "q"] are allowed for active or
        reactive power

        **index** (int, index) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **check** (bool, True) - raises UserWarning if costs already exist to this element.

    OUTPUT:
        **index** (int) - The unique ID of created cost entry

    EXAMPLE:
        The cost function is given by the x-values p1 and p2 with the slope m between those points.\
        The constant part b of a linear function y = m*x + b can be neglected for OPF purposes. \
        The intervals have to be continuous (the starting point of an interval has to be equal to \
        the end point of the previous interval).

        To create a gen with costs of 1€/MW between 0 and 20 MW and 2€/MW between 20 and 30:

        create_pwl_costs(net, [0, 1], ["gen", "sgen"], [[[0, 20, 1], [20, 30, 2]], \
            [[0, 20, 1], [20, 30, 2]]])
    """
    if not hasattr(elements, "__iter__") and not isinstance(elements, str):
        raise ValueError(f"An iterable is expected for elements, not {elements}.")
    if not hasattr(points, "__iter__"):
        if len(points) != len(elements):
            raise ValueError(
                f"It should be the same, but len(elements) is {len(elements)} whereas len(points) is{len(points)}."
            )
        if not hasattr(points[0], "__iter__") or len(points[0]) == 0 or not hasattr(points[0][0], "__iter__"):
            raise ValueError("A list of lists of lists is expected for points.")
    if check:
        bool_ = _costs_existance_check(net, elements, et, power_type=power_type)
        if np.sum(bool_) >= 1:
            raise UserWarning("There already exist costs for {np.sum(bool_)} elements.")

    index = _get_multiple_index_with_check(net, "pwl_cost", index, len(elements), "piecewise_linear_cost")
    entries = {"power_type": power_type, "element": elements, "et": et, "points": points, **kwargs}
    _set_multiple_entries(net, "pwl_cost", index, entries=entries)
    return index


def create_poly_cost(
    net: pandapowerNet,
    element: Int | Iterable[Int],
    et: CostElementType,
    cp1_eur_per_mw: float,
    cp0_eur: float = 0,
    cq1_eur_per_mvar: float = 0,
    cq0_eur: float = 0,
    cp2_eur_per_mw2: float = 0,
    cq2_eur_per_mvar2: float = 0,
    index: int | None = None,
    check: bool = True,
    **kwargs,
) -> Int:
    """
    Creates an entry for polynomial costs for an element. The currently supported elements are:
     - Generator ("gen")
     - External Grid ("ext_grid")
     - Static Generator ("sgen")
     - Load ("load")
     - Dcline ("dcline")
     - Storage ("storage")

    INPUT:
        **element** (int) - ID of the element in the respective element table

        **et** (string) - Type of element ["gen", "sgen", "ext_grid", "load", "dcline", "storage"]
        are possible

        **cp1_eur_per_mw** (float) - Linear costs per MW

        **cp0_eur=0** (float) - Offset active power costs in euro

        **cq1_eur_per_mvar=0** (float) - Linear costs per Mvar

        **cq0_eur=0** (float) - Offset reactive power costs in euro

        **cp2_eur_per_mw2=0** (float) - Quadratic costs per MW

        **cq2_eur_per_mvar2=0** (float) - Quadratic costs per Mvar

    OPTIONAL:

        **index** (int, index) - Force a specified ID if it is available. If None, the index one
        higher than the highest already existing index is selected.

        **check** (bool, True) - raises UserWarning if costs already exist to this element.

    OUTPUT:
        **index** (int) - The unique ID of created cost entry

    EXAMPLE:
        The polynomial cost function is given by the linear and quadratic cost coefficients.

        create_poly_cost(net, 0, "load", cp1_eur_per_mw=0.1)
    """
    if isinstance(element, (list, tuple)):
        element = element[0]
    if check and _cost_existance_check(net, element, et):
        raise UserWarning(f"There already exist costs for {et} {element}")

    index = _get_index_with_check(net, "poly_cost", index)

    entries = {
        "element": element,
        "et": et,
        "cp0_eur": cp0_eur,
        "cp1_eur_per_mw": cp1_eur_per_mw,
        "cq0_eur": cq0_eur,
        "cq1_eur_per_mvar": cq1_eur_per_mvar,
        "cp2_eur_per_mw2": cp2_eur_per_mw2,
        "cq2_eur_per_mvar2": cq2_eur_per_mvar2,
        **kwargs,
    }
    _set_entries(net, "poly_cost", index, entries=entries)
    return index


def create_poly_costs(
    net: pandapowerNet,
    elements: Sequence,
    et: CostElementType | Iterable[str],
    cp1_eur_per_mw: float | Iterable[float],
    cp0_eur: float | Iterable[float] = 0,
    cq1_eur_per_mvar: float | Iterable[float] = 0,
    cq0_eur: float | Iterable[float] = 0,
    cp2_eur_per_mw2: float | Iterable[float] = 0,
    cq2_eur_per_mvar2: float | Iterable[float] = 0,
    index: int | None = None,
    check: bool = True,
    **kwargs,
) -> npt.NDArray[np.array]:
    """
    Creates entries for polynomial costs for multiple elements. The currently supported elements are:
     - Generator ("gen")
     - External Grid ("ext_grid")
     - Static Generator ("sgen")
     - Load ("load")
     - Dcline ("dcline")
     - Storage ("storage")

    INPUT:
        **elements** (iterable of integers) - IDs of the elements in the respective element table

        **et** (string or iterable) - Type of element ["gen", "sgen", "ext_grid", "load", "dcline",
            "storage"] are possible

        **cp1_eur_per_mw** (float or iterable) - Linear costs per MW

        **cp0_eur=0** (float or iterable) - Offset active power costs in euro

        **cq1_eur_per_mvar=0** (float or iterable) - Linear costs per Mvar

        **cq0_eur=0** (float or iterable) - Offset reactive power costs in euro

        **cp2_eur_per_mw2=0** (float or iterable) - Quadratic costs per MW

        **cq2_eur_per_mvar2=0** (float or iterable) - Quadratic costs per Mvar

    OPTIONAL:

        **index** (int, index) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **check** (bool, True) - raises UserWarning if costs already exist to this element.

    OUTPUT:
        **index** (int) - The unique ID of created cost entry

    EXAMPLE:
        The polynomial cost function is given by the linear and quadratic cost coefficients.
        If the first two loads have active power cost functions of the kind
        c(p) = 0.5 + 1 * p + 0.1 * p^2, the costs are created as follows:

        create_poly_costs(net, [0, 1], "load", cp0_eur=0.5, cp1_eur_per_mw=1, cp2_eur_per_mw2=0.1)
    """
    if not hasattr(elements, "__iter__") and not isinstance(elements, str):
        raise ValueError(f"An iterable is expected for elements, not {elements}.")
    if check:
        bool_ = _costs_existance_check(net, elements, et)
        if np.sum(bool_) >= 1:
            raise UserWarning(f"There already exist costs for {np.sum(bool_)} elements.")

    index = _get_multiple_index_with_check(net, "poly_cost", index, len(elements), "poly_cost")

    entries = {
        "element": elements,
        "et": et,
        "cp0_eur": cp0_eur,
        "cp1_eur_per_mw": cp1_eur_per_mw,
        "cq0_eur": cq0_eur,
        "cq1_eur_per_mvar": cq1_eur_per_mvar,
        "cp2_eur_per_mw2": cp2_eur_per_mw2,
        "cq2_eur_per_mvar2": cq2_eur_per_mvar2,
        **kwargs,
    }
    _set_multiple_entries(net, "poly_cost", index, entries=entries)
    return index
