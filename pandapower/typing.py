from typing import Literal, Union

import numpy as np

# union type for integers from the Python standard library and numpy
Int = Union[int, np.integer]
# union type for integers and floats from the Python standard library and numpy
Float = Union[float, np.number]

BusType = Literal["n", "b", "m"]
GeneratorType = Literal["current_source", "async", "async_doubly_fed"]
UnderOverExcitedType = Literal["underexcited", "overexcited"]
WyeDeltaType = Literal["wye", "delta"]