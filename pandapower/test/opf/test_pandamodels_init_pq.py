import numpy as np
import pandas as pd
import pytest

from pandapower.create import create_empty_network
from pandapower.converter.pandamodels.to_pm import add_pm_gen_start_values_from_results
from pandapower.runpm import runpm, runpm_dc_opf, runpm_ac_opf

def test_add_pm_gen_start_values_from_results():
    net = create_empty_network()
    net.res_ext_grid = pd.DataFrame({"p_mw": [100.0], "q_mvar": [50.0]}, index=[0])
    net.res_gen = pd.DataFrame({"p_mw": [30.0], "q_mvar": [10.0]}, index=[2])
    net.res_sgen = pd.DataFrame({"p_mw": [5.0], "q_mvar": [1.0]}, index=[4])
    net._pd2pm_lookups = {
        "ext_grid": np.array([1]),
        "gen": np.array([-1, -1, 2]),
        "sgen_controllable": np.array([-1, -1, -1, -1, 3]),
    }
    pm = {"gen": {"1": {}, "2": {}, "3": {}}}
    add_pm_gen_start_values_from_results(net, pm)

    assert np.isclose(pm["gen"]["1"]["pg_start"], 100.0)
    assert np.isclose(pm["gen"]["1"]["qg_start"], 50.0)
    assert np.isclose(pm["gen"]["2"]["pg_start"], 30.0)
    assert np.isclose(pm["gen"]["2"]["qg_start"], 10.0)
    assert np.isclose(pm["gen"]["3"]["pg_start"], 5.0)
    assert np.isclose(pm["gen"]["3"]["qg_start"], 1.0)

@pytest.mark.parametrize(
    "run_function",
    [
        runpm,
        runpm_dc_opf,
        runpm_ac_opf,
    ],
)
def test_runpm_wrappers_pass_init_pq_to_options(monkeypatch, run_function):
    captured_options = []
    def fake_runpm(net, *args, **kwargs):
        captured_options.append(net._options.copy())
    monkeypatch.setitem(run_function.__globals__, "_runpm", fake_runpm)
    net = create_empty_network()
    run_function(net, init_pq="results")

    assert len(captured_options) == 1
    assert captured_options[0]["init_pq"] == "results"