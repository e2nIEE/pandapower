import numpy as np

from pandapower.converter.pypower import from_ppc

def test_from_ppc_opf_controllable():
    ppc = {
        "version": "2",
        "baseMVA": 100.0,
        "bus": np.array(
            [
                [0, 3, 0, 0, 0, 0, 1, 1.0, 0.0, 110.0, 1, 1.1, 0.9],
                [1, 2, 0, 0, 0, 0, 1, 1.0, 0.0, 110.0, 1, 1.1, 0.9],
            ],
            dtype=float,
        ),
        "branch": np.array(
            [
                [0, 1, 0.01, 0.05, 0, 100, 100, 100, 0, 0, 1, -30, 30],
            ],
            dtype=float,
        ),
        "gen": np.array(
            [
                [0, 50, 0, 100, -100, 1.0, 100, 1, 100, 0],
                [1, 20, 0, 100, -100, 1.0, 100, 1, 100, 0],
            ],
            dtype=float,
        ),
    }

    net = from_ppc(ppc, validate_conversion=False, set_opf_controllable=True)

    assert len(net.ext_grid) == 1
    assert len(net.gen) == 1
    assert net.ext_grid["controllable"].all()
    assert net.gen["controllable"].all()