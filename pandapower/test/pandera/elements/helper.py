import numpy as np
import pandas as pd

# Boolean types
bools = [True, False, np.bool_(True), np.bool_(False)]
# Numeric types
floats = [0, 0.0, 1.0, float("inf"), float("-inf"), np.float32(1.0), np.float64(1.0)]
ints = [0, 1, -1, 42, np.int8(1), np.int16(1), np.int32(1), np.int64(1), np.uint8(1)]
# String types
strings = [
    "True",
    "False",
    "true",
    "false",
    "1",
    "0",
    "yes",
    "no",
    "not a number",
    "",
    " ",
]
others = [
    # None and NaN variants
    None,
    pd.NaT,
    np.nan,
    # Collections
    {},
    {"value": True},
    # Objects
    object(),
    type,
    lambda x: x,
    # Complex numbers
    complex(1, 0),
    1 + 0j,
]

not_boolean_list = [*others, *strings, *ints, *floats]
not_floats_list = [*others, *strings, *ints, *bools]
not_strings_list = [*others, *strings, *ints, *bools]
