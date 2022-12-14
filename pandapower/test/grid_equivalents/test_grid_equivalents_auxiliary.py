import pytest
import pandapower as pp
import pandapower.grid_equivalents
import pandapower.networks


def test_drop_internal_branch_elements():
    net = pp.networks.example_simple()
    pp.grid_equivalents.drop_internal_branch_elements(net, net.bus.index)
    assert not net.line.shape[0]
    assert not net.trafo.shape[0]

    net = pp.networks.example_simple()
    n_trafo = net.trafo.shape[0]
    pp.grid_equivalents.drop_internal_branch_elements(net, net.bus.index, branch_elements=["line"])
    assert not net.line.shape[0]
    assert net.trafo.shape[0] == n_trafo

    net = pp.networks.example_simple()
    n_trafo = net.trafo.shape[0]
    pp.grid_equivalents.drop_internal_branch_elements(net, [2, 3, 4, 5])
    assert set(net.line.index) == {0, 2, 3}
    assert set(net.trafo.index) == set()

    net = pp.networks.example_simple()
    n_trafo = net.trafo.shape[0]
    pp.grid_equivalents.drop_internal_branch_elements(net, [4, 5, 6])
    assert set(net.line.index) == {0}
    assert set(net.trafo.index) == {0}


if __name__ == "__main__":
    if 1:
        pytest.main(['-x', __file__])
    else:
        test_drop_internal_branch_elements()
    pass