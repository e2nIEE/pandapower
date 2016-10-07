__author__ = 'jdollichon'

import pandapower
import numpy as np

def test_get_indices():
    a = [i+100 for i in range(10)]
    lookup = {idx: pos for pos, idx in enumerate(a)}
    lookup["before_fuse"] = a

    # First without fused busses no magic here
    # after fuse
    result = pandapower.auxiliary.get_indices([102, 107], lookup, fused_indices=True)
    assert np.array_equal(result, [2, 7])

    # before fuse
    result = pandapower.auxiliary.get_indices([2, 7], lookup, fused_indices=False)
    assert np.array_equal(result, [102, 107])

    # Same setup EXCEPT we have fused busses now (bus 102 and 107 are fused)
    lookup[107] = lookup[102]

    # after fuse
    result = pandapower.auxiliary.get_indices([102, 107], lookup, fused_indices=True)
    assert np.array_equal(result, [2, 2])

    # before fuse
    result = pandapower.auxiliary.get_indices([2, 7], lookup, fused_indices=False)
    assert np.array_equal(result, [102, 107])