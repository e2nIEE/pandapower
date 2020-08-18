# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

import pandapower as pp
import pandapower.networks as nw
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pypower.makeLODF import makeLODF

from pandapower.test.loadflow.result_test_network_generator import result_test_network_generator_dcpp
from pandapower.test.toolbox import add_grid_connection, create_test_line, assert_net_equal


def test_PTDF():
    net = nw.case9()
    pp.rundcpp(net)
    _, ppci = _pd2ppc(net)

    ptdf = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"],
                    using_sparse_solver=False)
    _ = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"],
                 result_side=1, using_sparse_solver=False)
    ptdf_sparse = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"],
                           using_sparse_solver=True)

    if not np.allclose(ptdf, ptdf_sparse):
        raise AssertionError("Sparse PTDF has differenct result against dense PTDF")
    if not ptdf.shape == (ppci["bus"].shape[0], ppci["branch"].shape[0]):
        raise AssertionError("PTDF has wrong dimension")
    if not np.all(~np.isnan(ptdf)):
        raise AssertionError("PTDF has NaN value")


def test_LODF():
    net = nw.case9()
    pp.rundcpp(net)
    _, ppci = _pd2ppc(net)

    ptdf = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"])
    lodf = makeLODF(ppci["branch"], ptdf)
    if not lodf.shape == (ppci["branch"].shape[0], ppci["branch"].shape[0]):
        raise AssertionError("LODF has wrong dimension")


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
