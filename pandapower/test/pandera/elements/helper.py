import numpy as np
import pandas as pd

# Boolean types
bools = [True, False, np.bool_(True), np.bool_(False)]
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
# Numeric types
# ints
zero = [np.int64(0), 0]
positiv_ints = [1, 42, np.int64(1)]
negativ_ints = [-1, -42, np.int64(-1)]
all_allowed_ints = [*zero, *positiv_ints, *negativ_ints]
not_allowed_ints = [np.int8(1), np.int16(1), np.int32(1), np.uint8(1), np.uint16(1), np.uint32(1)]
all_ints = [*all_allowed_ints, *not_allowed_ints]
# floats
float_zero = [0.0, np.float64(0.0)]
positiv_floats = [1.0, float("inf"), np.float64(1.0)]
negativ_floats = [-1.0, float("-inf"), np.float64(-1.0)]
all_allowed_floats = [*float_zero, *positiv_floats, *negativ_floats]
not_allowed_floats = [
    np.float32(1.0),
    np.float16(1.0),
]
all_floats = [*all_allowed_floats, *not_allowed_floats]

not_boolean_list = [*others, *strings, *all_ints, *all_floats]
not_floats_list = [*others, *strings, *all_ints, *bools]
not_strings_list = [*others, *all_ints, *bools, *all_floats]
