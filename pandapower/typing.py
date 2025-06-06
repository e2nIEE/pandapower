from typing import Literal, Union

import numpy as np

# union type for integers from the Python standard library and numpy
Int = Union[int, np.integer]
# union type for integers and floats from the Python standard library and numpy
Float = Union[float, np.number]

BusType = Literal["n", "b", "m"]
CostElementType = Literal["gen", "sgen", "ext_grid", "load", "dcline", "storage"]
GeneratorType = Literal["current_source", "async", "async_doubly_fed"]
LineType = Literal["ol", "cs"]
HVLVType = Literal["hv", "lv"]
HVMVLVType = Literal["hv", "mv", "lv"]
MeasurementType = Literal["v", "p", "q", "i", "va", "ia"]
MeasurementElementType = Literal["bus", "line", "trafo", "trafow3", "load", "gen", "sgen", "shunt", "ward", "xward",
                                 "ext_grid"]
PWLPowerType = Literal["p", "q"]
SwitchElementType = Literal[
    "b",  # bus
    "l",  # line
    "t",  # transformer
]
SwitchType = Literal[
    "LS",  # load switch
    "CB",  # circuit breaker
    "LBS",  # load break switch
    "DS",  # disconnecting switch
]
TapChangerType = Literal["Ratio", "Symmetrical", "Ideal"]
TapChangerWithTabularType = Literal["Ratio", "Symmetrical", "Ideal", "Tabular"]
UnderOverExcitedType = Literal["underexcited", "overexcited"]
WyeDeltaType = Literal["wye", "delta"]