# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandas as pd
import pytest

from pandapower.networks import create_cigre_network_mv
from pandapower.toolbox.element_selection import pp_elements
from pandapower.toolbox.power_factor import signing_system_value, cosphi_pos_neg_from_pq, pq_from_cosphi, \
    cosphi_to_pos, cosphi_from_pq, cosphi_from_pos, sync_q_from_cos_phi, create_cos_phi_from_network, \
    create_cos_phi_constant


def test_signing_system_value():
    assert signing_system_value("sgen") == -1
    assert signing_system_value("load") == 1
    for bus_elm in pp_elements(bus=False, branch_elements=False, other_elements=False):
        assert signing_system_value(bus_elm) in [1, -1]
    with pytest.raises(ValueError):
        signing_system_value("sdfjio")


def test_pq_from_cosphi():
    p, q = pq_from_cosphi(1 / 0.95, 0.95, "underexcited", "load")
    assert np.isclose(p, 1)
    assert np.isclose(q, 0.3286841051788632)

    s = np.array([1, 1, 1])
    cosphi = np.array([1, 0.5, 0])
    pmode = np.array(["load", "load", "load"])
    qmode = np.array(["underexcited", "underexcited", "underexcited"])
    p, q = pq_from_cosphi(s, cosphi, qmode, pmode)
    excpected_values = (np.array([1, 0.5, 0]), np.array([0, 0.8660254037844386, 1]))
    assert np.allclose(p, excpected_values[0])
    assert np.allclose(q, excpected_values[1])

    pmode = "gen"
    p, q = pq_from_cosphi(s, cosphi, qmode, pmode)
    assert np.allclose(p, excpected_values[0])
    assert np.allclose(q, -excpected_values[1])

    qmode = "overexcited"
    p, q = pq_from_cosphi(s, cosphi, qmode, pmode)
    assert np.allclose(p, excpected_values[0])
    assert np.allclose(q, excpected_values[1])

    with pytest.raises(ValueError):
        pq_from_cosphi(1, 0.95, "ohm", "gen")

    p, q = pq_from_cosphi(0, 0.8, "overexcited", "gen")
    assert np.isclose(p, 0)
    assert np.isclose(q, 0)


def test_cosphi_from_pq():
    cosphi, s, qmode, pmode = cosphi_from_pq(1, 0.4)
    assert np.isclose(cosphi, 0.9284766908852593)
    assert np.isclose(s, 1.077032961426901)
    assert qmode == 'underexcited'
    assert pmode == 'load'

    p = np.array([1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1])
    q = np.array([1, -1, 0, 0.5, -0.5, 1, -1, 0, 1, -1, 0])
    cosphi, s, qmode, pmode = cosphi_from_pq(p, q)
    assert np.allclose(cosphi[[0, 1, 8, 9]], 2 ** 0.5 / 2)
    assert np.allclose(cosphi[[3, 4]], 0.89442719)
    assert np.allclose(cosphi[[2, 10]], 1)
    assert pd.Series(cosphi[[5, 6, 7]]).isnull().all()
    assert np.allclose(s, (p ** 2 + q ** 2) ** 0.5)
    assert all(pmode == np.array(["load"] * 5 + ["undef"] * 3 + ["gen"] * 3))
    ind_cap_ind = ["underexcited", "overexcited", "underexcited"]
    assert all(qmode == np.array(ind_cap_ind + ["underexcited", "overexcited"] + ind_cap_ind * 2))


def test_cosphi_to_pos():
    assert np.isclose(cosphi_to_pos(0.96), 0.96)
    assert np.isclose(cosphi_to_pos(-0.94), 1.06)
    assert np.isclose(cosphi_to_pos(-0.96), 1.04)
    assert np.allclose(cosphi_to_pos([0.96, -0.94, -0.96]), np.array([0.96, 1.06, 1.04]))


def test_cosphi_from_pos():
    assert np.isclose(cosphi_from_pos(0.96), 0.96)
    assert np.isclose(cosphi_from_pos(1.06), -0.94)
    assert np.isclose(cosphi_from_pos(1.04), -0.96)
    assert np.allclose(cosphi_from_pos([0.96, 1.06, 1.04]), np.array([0.96, -0.94, -0.96]))


def test_cosphi_pos_neg():
    assert np.isclose(np.round(cosphi_pos_neg_from_pq(2, 0.), 5), 1)
    assert np.isclose(np.round(cosphi_pos_neg_from_pq(0.76, 0.25), 5), 0.94993)
    assert np.isclose(np.round(cosphi_pos_neg_from_pq(-0.76, 0.25), 5), 0.94993)
    assert np.isclose(np.round(cosphi_pos_neg_from_pq(0.76, -0.25), 5), -0.94993)
    assert np.allclose(
        np.round(cosphi_pos_neg_from_pq([0.76, 0.76, 0.76, 0.76], [0.25, -0.25, 0, 0.1]), 5),
        np.array([0.94993, -0.94993, 1., 0.99145]))
    assert np.allclose(
        np.round(cosphi_pos_neg_from_pq(
            [0.76, 0.76, -0.76, 0.76, 0, 0.1],
            [0.25, -0.25, 0.25, 0.1, 0.1, 0]), 5),
        np.array([0.94993, -0.94993, 0.94993, 0.99145, np.nan, 1]), equal_nan=True)

def test_create_cos_phi_from_network():
    net = create_cigre_network_mv("pv_wind")

    # Loads have both p>0 and q>0 in this network
    create_cos_phi_from_network(net, "load")
    assert "cos_phi" in net.load.columns
    assert len(net.load["cos_phi"]) == len(net.load)

    # All loads have p>0, so all cos_phi should be valid (not 1.0 default)
    cos_phi = net.load["cos_phi"].to_numpy()
    assert np.all(np.abs(cos_phi) <= 1.0)
    assert np.all(np.abs(cos_phi) > 0.0)
    # Loads have positive q → positive cos_phi (pos_neg convention)
    assert np.all(cos_phi > 0)

    # Roundtrip: sync should reproduce original q
    original_q = net.load["q_mvar"].to_numpy().copy()
    sync_q_from_cos_phi(net, "load", net.load.index)
    assert np.allclose(net.load["q_mvar"].to_numpy(), original_q, atol=1e-6)

    # Sgens in CIGRE MV have q=0 → cos_phi should be 1.0
    create_cos_phi_from_network(net, "sgen")
    assert np.allclose(net.sgen["cos_phi"].to_numpy(), 1.0)

    # Set some nonzero q on sgens and re-extract
    net.sgen.loc[0, "q_mvar"] = -0.5
    net.sgen.loc[1, "q_mvar"] = 0.3
    net.sgen.loc[0, "p_mw"] = 2.0
    net.sgen.loc[1, "p_mw"] = 1.5
    create_cos_phi_from_network(net, "sgen")
    assert net.sgen.at[0, "cos_phi"] < 0  # negative q → negative cos_phi
    assert net.sgen.at[1, "cos_phi"] > 0  # positive q → positive cos_phi

    # Roundtrip for those two
    original_q_sgen = net.sgen.loc[[0, 1], "q_mvar"].to_numpy().copy()
    sync_q_from_cos_phi(net, "sgen", [0, 1])
    assert np.allclose(net.sgen.loc[[0, 1], "q_mvar"].to_numpy(), original_q_sgen, atol=1e-6)

    # p=0 element gets cos_phi=1.0
    net.sgen.loc[2, "p_mw"] = 0.0
    net.sgen.loc[2, "q_mvar"] = 0.0
    create_cos_phi_from_network(net, "sgen")
    assert np.isclose(net.sgen.at[2, "cos_phi"], 1.0)


def test_create_cos_phi_constant():
    net = create_cigre_network_mv("pv_wind")

    # sgen + underexcited → signing_system_value("sgen")=-1 → negative cos_phi
    create_cos_phi_constant(net, "sgen", cos_phi=0.95, mode="underexcited")
    assert np.allclose(net.sgen["cos_phi"].to_numpy(), -0.95)

    # sgen + overexcited → sign flipped → positive
    create_cos_phi_constant(net, "sgen", cos_phi=0.95, mode="overexcited")
    assert np.allclose(net.sgen["cos_phi"].to_numpy(), 0.95)

    # load + underexcited → signing_system_value("load")=+1 → positive
    create_cos_phi_constant(net, "load", cos_phi=0.9, mode="underexcited")
    assert np.allclose(net.load["cos_phi"].to_numpy(), 0.9)

    # load + overexcited → negative
    create_cos_phi_constant(net, "load", cos_phi=0.9, mode="overexcited")
    assert np.allclose(net.load["cos_phi"].to_numpy(), -0.9)

    # Does NOT modify q_mvar
    original_q = net.sgen["q_mvar"].to_numpy().copy()
    create_cos_phi_constant(net, "sgen", cos_phi=0.8)
    assert np.allclose(net.sgen["q_mvar"].to_numpy(), original_q)


def test_sync_q_from_cos_phi():
    net = create_cigre_network_mv("pv_wind")

    # Raises without cos_phi column
    with pytest.raises(KeyError, match="cos_phi"):
        sync_q_from_cos_phi(net, "sgen", net.sgen.index)

    # Setup constant cos_phi and verify formula
    create_cos_phi_constant(net, "sgen", cos_phi=0.95, mode="underexcited")
    net.sgen["p_mw"] = 1.0
    sync_q_from_cos_phi(net, "sgen", net.sgen.index)

    # cos_phi = -0.95 → q = abs(1.0) * tan(arccos(0.95)) * sign(-0.95) = -0.3287...
    expected_q = -np.tan(np.arccos(0.95))
    assert np.allclose(net.sgen["q_mvar"].to_numpy(), expected_q, atol=1e-6)

    # p=0 → q=0 regardless of cos_phi
    net.sgen["p_mw"] = 0.0
    sync_q_from_cos_phi(net, "sgen", net.sgen.index)
    assert np.allclose(net.sgen["q_mvar"].to_numpy(), 0.0)

    # cos_phi=1.0 → q=0 regardless of p
    net.sgen["cos_phi"] = 1.0
    net.sgen["p_mw"] = 5.0
    sync_q_from_cos_phi(net, "sgen", net.sgen.index)
    assert np.allclose(net.sgen["q_mvar"].to_numpy(), 0.0, atol=1e-10)

    # Partial index: only selected elements are synced
    create_cos_phi_constant(net, "sgen", cos_phi=0.9, mode="underexcited")
    net.sgen["p_mw"] = 2.0
    net.sgen["q_mvar"] = 999.0  # sentinel
    sync_q_from_cos_phi(net, "sgen", [net.sgen.index[0]])
    assert not np.isclose(net.sgen.at[net.sgen.index[0], "q_mvar"], 999.0)
    assert np.allclose(net.sgen["q_mvar"].to_numpy()[1:], 999.0)

    # q sign follows cos_phi sign, q magnitude uses abs(p)
    net.sgen.loc[0, "cos_phi"] = 0.9
    net.sgen.loc[1, "cos_phi"] = -0.9
    net.sgen.loc[0, "p_mw"] = 2.0
    net.sgen.loc[1, "p_mw"] = -2.0  # negative p (unusual)
    sync_q_from_cos_phi(net, "sgen", [0, 1])
    # Both have abs(p)=2, abs(cos_phi)=0.9 → same |q|
    assert np.isclose(abs(net.sgen.at[0, "q_mvar"]), abs(net.sgen.at[1, "q_mvar"]))
    # Signs follow cos_phi sign
    assert net.sgen.at[0, "q_mvar"] > 0  # cos_phi=+0.9
    assert net.sgen.at[1, "q_mvar"] < 0  # cos_phi=-0.9

if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
