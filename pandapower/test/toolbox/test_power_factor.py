import numpy as np
import pandas as pd
import pytest

import pandapower as pp
import pandapower.toolbox


def test_signing_system_value():
    assert pp.signing_system_value("sgen") == -1
    assert pp.signing_system_value("load") == 1
    for bus_elm in pandapower.toolbox.pp_elements(bus=False, branch_elements=False, other_elements=False):
        assert pp.signing_system_value(bus_elm) in [1, -1]
    try:
        pp.signing_system_value("sdfjio")
        assert False
    except ValueError:
        pass


def test_pq_from_cosphi():
    p, q = pp.pq_from_cosphi(1 / 0.95, 0.95, "underexcited", "load")
    assert np.isclose(p, 1)
    assert np.isclose(q, 0.3286841051788632)

    s = np.array([1, 1, 1])
    cosphi = np.array([1, 0.5, 0])
    pmode = np.array(["load", "load", "load"])
    qmode = np.array(["underexcited", "underexcited", "underexcited"])
    p, q = pp.pq_from_cosphi(s, cosphi, qmode, pmode)
    excpected_values = (np.array([1, 0.5, 0]), np.array([0, 0.8660254037844386, 1]))
    assert np.allclose(p, excpected_values[0])
    assert np.allclose(q, excpected_values[1])

    pmode = "gen"
    p, q = pp.pq_from_cosphi(s, cosphi, qmode, pmode)
    assert np.allclose(p, excpected_values[0])
    assert np.allclose(q, -excpected_values[1])

    qmode = "overexcited"
    p, q = pp.pq_from_cosphi(s, cosphi, qmode, pmode)
    assert np.allclose(p, excpected_values[0])
    assert np.allclose(q, excpected_values[1])

    with pytest.raises(ValueError):
        pp.pq_from_cosphi(1, 0.95, "ohm", "gen")

    p, q = pp.pq_from_cosphi(0, 0.8, "overexcited", "gen")
    assert np.isclose(p, 0)
    assert np.isclose(q, 0)


def test_cosphi_from_pq():
    cosphi, s, qmode, pmode = pp.cosphi_from_pq(1, 0.4)
    assert np.isclose(cosphi, 0.9284766908852593)
    assert np.isclose(s, 1.077032961426901)
    assert qmode == 'underexcited'
    assert pmode == 'load'

    p = np.array([1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1])
    q = np.array([1, -1, 0, 0.5, -0.5, 1, -1, 0, 1, -1, 0])
    cosphi, s, qmode, pmode = pp.cosphi_from_pq(p, q)
    assert np.allclose(cosphi[[0, 1, 8, 9]], 2 ** 0.5 / 2)
    assert np.allclose(cosphi[[3, 4]], 0.89442719)
    assert np.allclose(cosphi[[2, 10]], 1)
    assert pd.Series(cosphi[[5, 6, 7]]).isnull().all()
    assert np.allclose(s, (p ** 2 + q ** 2) ** 0.5)
    assert all(pmode == np.array(["load"] * 5 + ["undef"] * 3 + ["gen"] * 3))
    ind_cap_ind = ["underexcited", "overexcited", "underexcited"]
    assert all(qmode == np.array(ind_cap_ind + ["underexcited", "overexcited"] + ind_cap_ind * 2))


if __name__ == '__main__':
    pytest.main([__file__, "-x"])