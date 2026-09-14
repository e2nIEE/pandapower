# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging

import pandas as pd

from pandapower import pandapowerNet
from pandapower.create._utils import _get_index_with_check, _set_entries
from pandapower.pp_types import Int, MeasurementElementType, MeasurementSideType, MeasurementType

logger = logging.getLogger(__name__)

_element_type_side_map = {"line": ["from", "to"], "trafo": ["hv", "lv"], "trafo3w": ["hv", "mv", "lv"]}

def create_measurement(
    net: pandapowerNet,
    meas_type: MeasurementType,
    element_type: MeasurementElementType,
    value: float,
    std_dev: float,
    element: int,
    side: int | MeasurementSideType | None = None,
    check_existing: bool = False,
    index: Int | None = None,
    name: str | None = None,
    **kwargs,
) -> Int:
    """
    Creates a measurement, which is used by the estimation module. Possible types of measurements \
    are: v, p, q, i, va, ia

    Parameters:
        net: the network to which the measurement will be added
        meas_type: Type of measurement. "v", "p", "q", "i", "va" and "ia" are possible
        element_type: Clarifies which element is measured. "bus", "line", "trafo", "trafo3w", "load", "gen", "sgen",
            "shunt", "ward", "xward" and "ext_grid" are possible
        value: Measurement value.
        std_dev: Standard deviation in the same unit as the measurement
        element: Index of the measured element
        side: Only used for line or trafo(3w). Side defines at which point of the element the measurement is
            gathered. For lines this may be "from", "to" to denote the side with the from_bus or to_bus. It can also be
            the index of the from_bus or to_bus. For trafo(3w), it can be "hv", ("mv"), "lv" or the corresponding bus
            index, respectively.
        check_existing: Check for and replace existing measurements for this bus, type and element_type. Set it to False
            for performance improvements which can cause unsafe behavior.
        index: Index of the measurement in the measurement table. Should not exist already.
        name: Name of measurement

    Returns:
        Index of the created measurement

    Example:
        2 MW load measurement with 0.05 MW standard deviation on bus 0:
        
        >>> create_measurement(net, "p", "bus", 2., 0.05., 0)

        4.5 MVar line measurement with 0.1 MVAr standard deviation on the "to_bus" side of line 2:
        
        >>> create_measurement(net, "q", "line", 4.5, 0.1, 2, "to")
    """
    if side is None and element_type in ("line", "trafo", "trafo3w"):
        raise UserWarning(f"The element type '{element_type}' requires parameter 'side' to be set")

    # convert index side to string side
    if pd.notna(side) and not isinstance(side, str):
        possible_cols = [f"{name}_bus" for name in _element_type_side_map[element_type]]
        element_col = net[element_type].loc[element, possible_cols].isin(side)
        col_name = element_col.index[element_col].tolist()
        if len(col_name) > 1:
            raise UserWarning(f"found multiple columns with the required side index: {col_name}")
        if len(col_name) == 0:
            raise UserWarning(f"side index not found in element row:\n{net[element_type].loc[element]}")
        side = col_name[0][:-4]  # slice away the _bus at the end of the column name

    # check if side is valid for element_type
    if pd.notna(side) and side not in _element_type_side_map[element_type]:
        raise AttributeError(
            f"The element_type '{element_type}' requires parameter 'side' to be "
            f"from {_element_type_side_map[element_type]}"
        )

    if element is not None and element not in net[element_type].index.to_numpy():
        raise UserWarning(f"{element_type} with index={element} does not exist")

    if meas_type in ("i", "ia") and element_type == "bus":
        raise UserWarning("Line current measurements cannot be placed at buses")

    if meas_type in ("v", "va") and element_type != "bus":
        raise UserWarning(f"Voltage measurements can only be placed at a bus, not at {element_type}")

    index = _get_index_with_check(net, "measurement", index)

    if check_existing:
        if side is None:
            existing = net.measurement[
                (net.measurement.measurement_type == meas_type)
                & (net.measurement.element_type == element_type)
                & (net.measurement.element == element)
                & (pd.isnull(net.measurement.side))
            ].index
        else:
            existing = net.measurement[
                (net.measurement.measurement_type == meas_type)
                & (net.measurement.element_type == element_type)
                & (net.measurement.element == element)
                & (net.measurement.side == side)
            ].index
        if len(existing) == 1:
            index = existing[0]
        elif len(existing) > 1:
            raise UserWarning("More than one measurement of this type exists")

    entries = {
        "name": name,
        "measurement_type": meas_type.lower(),
        "element_type": element_type,
        "element": element,
        "value": value,
        "std_dev": std_dev,
        "side": side,
        **kwargs,
    }
    _set_entries(net, "measurement", index, entries=entries)
    return index
