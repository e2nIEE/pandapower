# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

"""Tests for the ppc mapping of branch voltage angle difference limits (issue #2385)."""

import numpy as np
import pytest

from pandapower.create import (
    create_buses,
    create_ext_grid,
    create_impedances,
    create_lines,
    create_loads,
    create_transformers3w_from_parameters,
    create_transformers_from_parameters,
)
from pandapower.network import pandapowerNet
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.idx_brch import ANGMAX, ANGMIN
from pandapower.run import runopp, runpp

LINE_PARAMS = {"r_ohm_per_km": 0.1, "x_ohm_per_km": 0.4, "c_nf_per_km": 0.0, "max_i_ka": 1.0}


def _ppc_angle_limits(net):
    runpp(net)  # initialise net._options
    net["_options"]["mode"] = "opf"
    _, ppci = _pd2ppc(net)
    return ppci["branch"][:, [ANGMIN, ANGMAX]], net._pd2ppc_lookups["branch"]


def _net_with_all_branch_elements():
    net = pandapowerNet(name="angle_limits")
    hv = create_buses(net, 3, vn_kv=110.0)
    lv = create_buses(net, 2, vn_kv=10.0)
    create_ext_grid(net, hv[0])
    create_lines(
        net,
        [hv[0]],
        [hv[1]],
        length_km=10.0,
        line_params=LINE_PARAMS,
        max_loading_percent=100.0,
    )
    create_transformers_from_parameters(
        net,
        [hv[1]],
        [lv[0]],
        sn_mva=100.0,
        vn_hv_kv=110.0,
        vn_lv_kv=10.0,
        vk_percent=10.0,
        vkr_percent=0.3,
        pfe_kw=0.0,
        i0_percent=0.0,
        max_loading_percent=100.0,
    )
    create_impedances(
        net,
        [hv[1]],
        [hv[2]],
        rft_pu=0.001,
        xft_pu=0.01,
        sn_mva=100.0,
    )
    return net


def _net_with_trafo3w():
    net = pandapowerNet(name="angle_limits_3w")
    hv = create_buses(net, 1, vn_kv=110.0)
    mv = create_buses(net, 1, vn_kv=20.0)
    lv = create_buses(net, 1, vn_kv=10.0)
    create_ext_grid(net, hv[0])
    create_transformers3w_from_parameters(
        net,
        hv,
        mv,
        lv,
        vn_hv_kv=110.0,
        vn_mv_kv=20.0,
        vn_lv_kv=10.0,
        sn_hv_mva=100.0,
        sn_mv_mva=50.0,
        sn_lv_mva=50.0,
        vk_hv_percent=10.0,
        vk_mv_percent=10.0,
        vk_lv_percent=10.0,
        vkr_hv_percent=0.3,
        vkr_mv_percent=0.3,
        vkr_lv_percent=0.3,
        pfe_kw=0.0,
        i0_percent=0.0,
        tap_at_star_point=False,
    )
    return net


def test_default_is_unconstrained():
    """Networks without the angle columns must keep the pypower 'unconstrained' sentinels."""
    net = _net_with_all_branch_elements()
    limits, _ = _ppc_angle_limits(net)
    # ANGMIN == -360 and ANGMAX == 360 for every branch
    assert np.all(limits[:, 0] == -360.0)
    assert np.all(limits[:, 1] == 360.0)


def test_per_element_limits():
    net = _net_with_all_branch_elements()
    net.line["min_angle_degree"] = -30.0
    net.line["max_angle_degree"] = 20.0
    net.trafo["min_angle_degree"] = -5.0
    net.trafo["max_angle_degree"] = 5.0
    net.impedance["min_angle_degree"] = -40.0

    limits, lookup = _ppc_angle_limits(net)

    for element, lo, hi in (("line", -30.0, 20.0), ("trafo", -5.0, 5.0), ("impedance", -40.0, 360.0)):
        f, t = lookup[element]
        np.testing.assert_allclose(limits[f:t, 0], lo)
        np.testing.assert_allclose(limits[f:t, 1], hi)


def test_nan_is_unconstrained_but_other_rows_keep_limits():
    net = _net_with_all_branch_elements()
    # two lines, only one constrained
    create_lines(
        net,
        [0],
        [2],
        length_km=10.0,
        line_params=LINE_PARAMS,
        min_angle_degree=-10.0,
        max_angle_degree=10.0,
    )
    net.line["min_angle_degree"] = [-30.0, np.nan]
    net.line["max_angle_degree"] = [20.0, np.nan]

    limits, lookup = _ppc_angle_limits(net)
    f, t = lookup["line"]
    np.testing.assert_allclose(limits[f:t, 0], [-30.0, -360.0])
    np.testing.assert_allclose(limits[f:t, 1], [20.0, 360.0])


def test_trafo3w_limits_on_all_three_branches():
    net = _net_with_trafo3w()
    net.trafo3w["min_angle_degree"] = -33.0
    net.trafo3w["max_angle_degree"] = 44.0

    limits, lookup = _ppc_angle_limits(net)
    f, t = lookup["trafo3w"]
    assert t - f == 3
    np.testing.assert_allclose(limits[f:t, 0], -33.0)
    np.testing.assert_allclose(limits[f:t, 1], 44.0)


def test_trafo3w_limits_are_per_transformer():
    """With two 3w transformers the limits must not be mixed up between the six ppc branches.

    The ppc branches are ordered side-wise (hv side of all trafos, then mv, then lv), so a naive
    implementation that repeats the first limit would still pass a single-transformer test.
    """
    net = _net_with_trafo3w()
    create_transformers3w_from_parameters(
        net,
        [0],
        [1],
        [2],
        vn_hv_kv=110.0,
        vn_mv_kv=20.0,
        vn_lv_kv=10.0,
        sn_hv_mva=100.0,
        sn_mv_mva=50.0,
        sn_lv_mva=50.0,
        vk_hv_percent=10.0,
        vk_mv_percent=10.0,
        vk_lv_percent=10.0,
        vkr_hv_percent=0.3,
        vkr_mv_percent=0.3,
        vkr_lv_percent=0.3,
        pfe_kw=0.0,
        i0_percent=0.0,
        tap_at_star_point=False,
    )
    net.trafo3w["min_angle_degree"] = [-11.0, -22.0]
    net.trafo3w["max_angle_degree"] = [13.0, 26.0]

    limits, lookup = _ppc_angle_limits(net)
    f, t = lookup["trafo3w"]
    assert t - f == 6
    # each side block holds both transformers in trafo3w index order
    for side_start in (f, f + 2, f + 4):
        np.testing.assert_allclose(limits[side_start : side_start + 2, 0], [-11.0, -22.0])
        np.testing.assert_allclose(limits[side_start : side_start + 2, 1], [13.0, 26.0])


def test_trafo3w_nan_is_unconstrained():
    net = _net_with_trafo3w()
    limits, lookup = _ppc_angle_limits(net)
    f, t = lookup["trafo3w"]
    np.testing.assert_allclose(limits[f:t, 0], -360.0)
    np.testing.assert_allclose(limits[f:t, 1], 360.0)


def test_trafo3w_limit_applies_per_star_branch_not_to_the_terminal_difference():
    """The limit bounds hv->star and star->mv/lv separately, so the terminal difference can reach 2x the limit.

    This is the documented (and intended) behaviour of the star-point representation. The test pins it so a
    change of the branch topology or of the limit mapping cannot silently alter the constraint semantics.
    """
    net = pandapowerNet(name="angle_limits_3w_semantics")
    hv = create_buses(net, 1, vn_kv=110.0)
    mv = create_buses(net, 1, vn_kv=20.0)
    lv = create_buses(net, 1, vn_kv=10.0)
    create_ext_grid(net, hv[0], min_p_mw=-1000.0, max_p_mw=1000.0)
    create_transformers3w_from_parameters(
        net,
        hv,
        mv,
        lv,
        vn_hv_kv=110.0,
        vn_mv_kv=20.0,
        vn_lv_kv=10.0,
        sn_hv_mva=100.0,
        sn_mv_mva=100.0,
        sn_lv_mva=100.0,
        vk_hv_percent=10.0,
        vk_mv_percent=10.0,
        vk_lv_percent=10.0,
        vkr_hv_percent=0.3,
        vkr_mv_percent=0.3,
        vkr_lv_percent=0.3,
        pfe_kw=0.0,
        i0_percent=0.0,
        tap_at_star_point=False,
        max_loading_percent=100.0,
        min_angle_degree=-2.0,
        max_angle_degree=2.0,
    )
    # load only the mv terminal, so hv->star and star->mv carry the same power and split the angle evenly
    create_loads(net, [mv[0]], p_mw=60.0, q_mvar=0.0, min_p_mw=0.0, max_p_mw=60.0)

    runopp(net, verbose=False)

    va = net.res_bus.va_degree
    # the loaded mv terminal: hv->star and star->mv each drop about half of the total, so the terminal
    # difference exceeds the per-star-branch limit of 2 degree and can approach 2 * 2 = 4 degree
    terminal_difference = va[hv[0]] - va[mv[0]]
    assert terminal_difference > 2.0 + 1e-6, (
        f"expected the terminal difference to exceed the per-star-branch limit of 2 degree, got {terminal_difference}"
    )
    assert terminal_difference <= 4.0 + 1e-6
    # the unloaded lv star branch drops almost no angle, so its terminal difference stays within the limit
    assert abs(va[hv[0]] - va[lv[0]]) <= 2.0 + 1e-6


@pytest.mark.parametrize("column", ["min_angle_degree", "max_angle_degree"])
def test_only_one_column_present(column):
    net = _net_with_all_branch_elements()
    net.line[column] = -20.0 if column == "min_angle_degree" else 20.0

    limits, lookup = _ppc_angle_limits(net)
    f, t = lookup["line"]
    expected_min = -20.0 if column == "min_angle_degree" else -360.0
    expected_max = 20.0 if column == "max_angle_degree" else 360.0
    np.testing.assert_allclose(limits[f:t, 0], expected_min)
    np.testing.assert_allclose(limits[f:t, 1], expected_max)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
