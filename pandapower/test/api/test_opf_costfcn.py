from numpy import array
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from pandapower.pypower.idx_cost import POLYNOMIAL
from pandapower.pypower.opf_costfcn import opf_costfcn


class SimpleOPFCostModel:
    def __init__(self):
        self.ppc = {
            "baseMVA": 100.0,
            "gen": array([[0.0]]),
            "gencost": array([
                [POLYNOMIAL, 0.0, 0.0, 3.0, 0.1, 1.0, 0.0],
            ]),
        }
        self.cost_params = {
            "N": csr_matrix([[1.0, 2.0]]),
            "Cw": array([3.0]),
            "H": csr_matrix([[4.0]]),
            "dd": array([1.0]),
            "rh": array([0.0]),
            "kk": array([0.0]),
            "mm": array([1.0]),
        }
        self.vv = {
            "i1": {"Pg": 0, "Qg": 1, "y": 2},
            "iN": {"Pg": 1, "Qg": 2, "y": 2},
        }

    def get_ppc(self):
        return self.ppc

    def get_cost_params(self):
        return self.cost_params

    def get_idx(self):
        return self.vv, None, None, None

    def getN(self, element_type, name):
        if element_type == "var" and name == "y":
            return 0
        raise KeyError((element_type, name))


def test_opf_costfcn_generalized_cost_hessian():
    om = SimpleOPFCostModel()
    x = array([1.0, 0.5])

    f, df, d2f = opf_costfcn(x, om, return_hessian=True)

    assert_allclose(f, 1114.0)
    assert_allclose(df, array([2111.0, 22.0]))
    assert_allclose(
        d2f.toarray(),
        array([
            [2004.0, 8.0],
            [8.0, 16.0],
        ])
    )


def test_opf_costfcn_generalized_cost_hessian_with_none_H():
    om = SimpleOPFCostModel()
    om.cost_params["H"] = None

    x = array([1.0, 0.5])

    f, df, d2f = opf_costfcn(x, om, return_hessian=True)

    assert_allclose(f, 1106.0)
    assert_allclose(df, array([2103.0, 6.0]))
    assert_allclose(
        d2f.toarray(),
        array([
            [2000.0, 0.0],
            [0.0, 0.0],
        ])
    )