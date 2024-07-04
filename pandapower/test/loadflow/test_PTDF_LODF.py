# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

import pandapower as pp
import pandapower.networks as nw
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pypower.makeLODF import makeLODF, makeOTDF, outage_results_OTDF

from pandapower.test.loadflow.result_test_network_generator import result_test_network_generator_dcpp
from pandapower.test.helper_functions import add_grid_connection, create_test_line, assert_net_equal


def test_PTDF():
    net = nw.case30()
    pp.rundcpp(net)
    _, ppci = _pd2ppc(net)

    ptdf = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"],
                    using_sparse_solver=False)
    _ = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"],
                 result_side=1, using_sparse_solver=False)
    ptdf_sparse = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"],
                           using_sparse_solver=True)
    ptdf_reduced = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"], 
                            using_sparse_solver=False, 
                            branch_id=list(range(15)), reduced=True)
    ptdf_reduced_sparse = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"], 
                                   using_sparse_solver=True, 
                                   branch_id=list(range(15)), reduced=True)

    if not np.allclose(ptdf, ptdf_sparse):
        raise AssertionError("Sparse PTDF has differenct result against dense PTDF")
    if not ptdf.shape == (ppci["branch"].shape[0], ppci["bus"].shape[0]):
        raise AssertionError("PTDF has wrong dimension")
    if not np.all(~np.isnan(ptdf)):
        raise AssertionError("PTDF has NaN value")
    if not ptdf_reduced.shape == (15, ppci["bus"].shape[0]):
        raise AssertionError("Reduced PTDF has wrong dimension")
    if not ptdf_reduced_sparse.shape == (15,ppci["bus"].shape[0]):
        raise AssertionError("Sparse reduced PTDF has wrong dimension")

def test_PTDF_large():
    net = nw.case9241pegase()
    pp.rundcpp(net)
    _, ppci = _pd2ppc(net)

    ptdf_sparse = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"],
                            using_sparse_solver=True)
    if not ptdf_sparse.shape == (ppci["branch"].shape[0], ppci["bus"].shape[0]):
        raise AssertionError("PTDF has wrong dimension")

def test_LODF():
    net = nw.case9()
    pp.rundcpp(net)
    _, ppci = _pd2ppc(net)

    ptdf = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"])
    lodf = makeLODF(ppci["branch"], ptdf)
    if not lodf.shape == (ppci["branch"].shape[0], ppci["branch"].shape[0]):
        raise AssertionError("LODF has wrong dimension")


def test_OTDF():
    net = nw.case9()
    mg = pp.topology.create_nxgraph(net, respect_switches=True)
    # roots = np.r_[net.ext_grid.bus.values, net.gen.bus.values]
    # stubs = pp.topology.determine_stubs(net, roots=roots, mg=mg, respect_switches=True)  # no lines are stubs here?
    # stubs = pp.toolbox.get_connected_elements(net, "line", roots)  # because not n-1 lines here are those
    c = pp.topology.find_graph_characteristics(g=mg, roots=net.ext_grid.bus.values, characteristics=["bridges"])
    bridges = np.array([pp.topology.lines_on_path(mg, p) for p in c["bridges"]]).flatten()
    # outage_lines = [i for i in net.line.index.values if i not in stubs and i not in bridges]
    outage_lines = np.array([i for i in net.line.index.values if i not in bridges])
    pp.rundcpp(net)
    _, ppci = _pd2ppc(net)
    ptdf = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"])
    lodf = makeLODF(ppci["branch"], ptdf)
    OTDF = makeOTDF(ptdf, lodf, outage_lines)
    Pbus = -net.res_bus.p_mw.values  # must be in generation reference frame
    nminus1_otdf = (OTDF @ Pbus.reshape(-1, 1)).reshape(outage_lines.shape[0], -1)

    # Test selected outages
    n_lines = len(net.line)
    for outage, line in enumerate(outage_lines):
        otdf_outage_result = (OTDF[outage * n_lines:outage * n_lines + n_lines, :] @ Pbus)

        # Run power flow for the outage scenario
        net.line.at[line, "in_service"] = False
        pp.rundcpp(net)
        pf_outage_result = net.res_line.p_from_mw.values
        net.line.at[line, "in_service"] = True

        # Compare the results
        assert np.allclose(otdf_outage_result, pf_outage_result, rtol=0, atol=1e-12)


def test_OTDF_outage_results():
    net = nw.case9()
    mg = pp.topology.create_nxgraph(net, respect_switches=True)
    # roots = np.r_[net.ext_grid.bus.values, net.gen.bus.values]
    # stubs = pp.topology.determine_stubs(net, roots=roots, mg=mg, respect_switches=True)  # no lines are stubs here?
    # stubs = pp.toolbox.get_connected_elements(net, "line", roots)  # because not n-1 lines here are those
    c = pp.topology.find_graph_characteristics(g=mg, roots=net.ext_grid.bus.values, characteristics=["bridges"])
    bridges = np.array([pp.topology.lines_on_path(mg, p) for p in c["bridges"]]).flatten()
    # outage_lines = [i for i in net.line.index.values if i not in stubs and i not in bridges]
    outage_lines = np.array([i for i in net.line.index.values if i not in bridges])
    pp.rundcpp(net)
    _, ppci = _pd2ppc(net)
    ptdf = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"])
    lodf = makeLODF(ppci["branch"], ptdf)
    OTDF = makeOTDF(ptdf, lodf, outage_lines)
    Pbus = -net.res_bus.p_mw.values  # must be in generation reference frame
    nminus1_otdf = outage_results_OTDF(OTDF, Pbus, outage_lines)

    # now obtain the outage results by performing power flow calculations:
    nminus1_pf = []
    for i in outage_lines:
        net.line.at[i, "in_service"] = False
        pp.rundcpp(net)
        nminus1_pf.append(net.res_line.p_from_mw.values.copy())
        net.line.at[i, "in_service"] = True

    nminus1_pf = np.vstack(nminus1_pf)

    assert np.allclose(nminus1_otdf, nminus1_pf, rtol=0, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
