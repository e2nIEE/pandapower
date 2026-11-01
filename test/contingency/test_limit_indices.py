import numpy as np
import pytest

from pandapower.contingency import check_elements_within_limits, get_element_limits, run_contingency
from pandapower.create import (
    create_bus,
    create_ext_grid,
    create_line_from_parameters,
    create_load,
)
from pandapower.network import pandapowerNet


@pytest.mark.parametrize("nminus1", [False, True])
@pytest.mark.parametrize(
    "count, partial, low_capacity, expected",
    [
        (2, True, True, True),
        (3, True, False, True),
        (2, False, False, True),
        (2, False, True, False),
    ],
)
def test_partial_loading_limits_ac(count, partial, low_capacity, expected, nminus1):
    net = pandapowerNet(name="partial_contingency_limits")
    a = create_bus(net, vn_kv=110, min_vm_pu=0.9, max_vm_pu=1.1, index=10)
    b = create_bus(net, vn_kv=110, min_vm_pu=0.9, max_vm_pu=1.1, index=30)
    create_ext_grid(net, a)
    for k in range(count):
        create_line_from_parameters(
            net,
            a,
            b,
            length_km=1,
            r_ohm_per_km=0.1,
            x_ohm_per_km=0.2,
            c_nf_per_km=0,
            max_i_ka=0.1 if k == 0 and low_capacity else 0.6,
            max_loading_percent=np.nan if k == 0 and partial else 100.0,
            index=5 + 10 * k,
        )
    create_load(net, b, p_mw=100, q_mvar=20)
    results = run_contingency(net, {"line": {"index": net.line.index.values}}, numba=False, raise_errors=True)
    limits = get_element_limits(net)
    assert limits["line"]["index"].tolist() == net.line.index[1 if partial else 0 :].tolist()
    assert check_elements_within_limits(limits, results, nminus1=nminus1) == expected


@pytest.mark.parametrize("element", ["line", "trafo", "trafo3w"])
@pytest.mark.parametrize("nminus1", [False, True])
@pytest.mark.parametrize("overloaded", [False, True])
def test_reordered_loading_limits(element, nminus1, overloaded):
    # Result order is deliberately opposite to constraint order. A positional
    # comparison misses the overload (or rejects the valid control).
    limits = {
        element: {
            "index": np.array([5, 15]),
            "max_limit": np.array([40.0, 100.0]),
            "max_limit_nminus1": np.array([40.0, 100.0]),
        }
    }
    load = np.array([80.0, 60.0 if overloaded else 30.0])
    results = {element: {"index": np.array([15, 5]), "loading_percent": load, "max_loading_percent": load}}
    assert check_elements_within_limits(limits, results, nminus1=nminus1) == (not overloaded)


@pytest.mark.parametrize(
    "field, value, nminus1",
    [
        ("vm_pu", 0.95, False),
        ("vm_pu", 1.2, False),
        ("vm_pu", 0.8, False),
        ("max_vm_pu", 1.2, True),
        ("min_vm_pu", 0.8, True),
    ],
)
def test_partial_bus_limits(field, value, nminus1):
    limits = {
        "bus": {
            "index": np.array([30]),
            "max_limit": np.array([1.1]),
            "min_limit": np.array([0.9]),
            "max_limit_nminus1": np.array([1.1]),
            "min_limit_nminus1": np.array([0.9]),
        }
    }
    values = {
        "index": np.array([10, 30]),
        "vm_pu": np.array([1.5, 1.0]),
        "max_vm_pu": np.array([1.5, 1.0]),
        "min_vm_pu": np.array([0.5, 1.0]),
    }
    values[field][1] = value
    assert check_elements_within_limits(limits, {"bus": values}, nminus1=nminus1) == (value == 0.95)


@pytest.mark.parametrize("nminus1", [False, True])
@pytest.mark.parametrize("overheated", [False, True])
def test_partial_temperature_limits(nminus1, overheated):
    limits = {"line": {"index": np.array([15]), "max_temperature_degree_celsius": np.array([80.0])}}
    temperature = 90.0 if overheated else 70.0
    values = {
        "index": np.array([5, 15]),
        "temperature_degree_celsius": np.array([100.0, 70.0]),
        "max_temperature_degree_celsius": np.array([100.0, temperature]),
    }
    if not nminus1:
        values["temperature_degree_celsius"][1] = temperature
    assert check_elements_within_limits(limits, {"line": values}, nminus1=nminus1) == (not overheated)


@pytest.mark.parametrize("results", [{}, {"line": {"index": np.array([5])}}])
def test_missing_constrained_results(results):
    limits = {"line": {"index": np.array([15]), "max_limit": np.array([100.0])}}
    with pytest.raises(ValueError, match="constrained line"):
        check_elements_within_limits(limits, results)


def test_results_without_extracted_constraints():
    assert check_elements_within_limits({}, {"line": {"index": np.array([5]), "loading_percent": np.array([200.0])}})
