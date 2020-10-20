import pytest

from pandapower.pf import run_newton_raphson_pf
from pandapower.pf.run_newton_raphson_pf import _run_newton_raphson_pf
from pandapower.pf.run_newton_raphson_pf import _run_fast_decoupled_pf
from pandapower.pf.run_newton_raphson_pf import \
    _run_ac_pf_without_qlims_enforced
from pandapower.pf.run_newton_raphson_pf import \
    _run_ac_pf_with_qlims_enforced


@pytest.fixture
def mock_run_dc_pf(monkeypatch):

    def _mock_run_dc_pf(ppci):

        ppci["init_va_degree"] = True
        return ppci

    monkeypatch.setattr(run_newton_raphson_pf, "_run_dc_pf", _mock_run_dc_pf)


@pytest.fixture
def mock_run_ac_pf_without_qlims(monkeypatch):

    def _mock_run_ac_pf_without_qlims(ppci, options):

        algorithm = options["algorithm"]

        ppci.update({"enforced_qlims": False, "algorithm": algorithm})

        success, iterations = None, None

        return ppci, success, iterations

    monkeypatch.setattr(
        run_newton_raphson_pf, "_run_ac_pf_without_qlims_enforced",
        _mock_run_ac_pf_without_qlims
    )


@pytest.fixture
def mock_run_ac_pf_with_qlims(monkeypatch):

    def _mock_run_ac_pf_with_qlims(ppci, options):

        algorithm = options["algorithm"]

        ppci.update({"enforced_qlims": True, "algorithm": algorithm})

        success, iterations, bus, gen, branch = None, None, None, None, None

        return ppci, success, iterations, bus, gen, branch

    monkeypatch.setattr(
        run_newton_raphson_pf, "_run_ac_pf_with_qlims_enforced",
        _mock_run_ac_pf_with_qlims
    )


@pytest.fixture
def mock_ppci_to_pfsoln(monkeypatch):

    def _mock_ppci_to_pfsoln(ppci, options):

        return None, None, None

    monkeypatch.setattr(run_newton_raphson_pf, "ppci_to_pfsoln",
                        _mock_ppci_to_pfsoln)


@pytest.fixture
def mock_store_results_from_pf_in_ppci(monkeypatch):

    def _mock_store_results_from_pf_in_ppci(ppci, bus, gen, branch, success,
                                            iterations, et):

        return ppci

    monkeypatch.setattr(
        run_newton_raphson_pf, "_store_results_from_pf_in_ppci",
        _mock_store_results_from_pf_in_ppci
    )


@pytest.fixture
def mock_get_numba_functions(monkeypatch):

    def _mock_get_numba_functions(ppci, options):
        return "makeYBus", "pfsoln"

    monkeypatch.setattr(run_newton_raphson_pf, "_get_numba_functions",
                        _mock_get_numba_functions)


@pytest.fixture
def mock_get_pf_variables_from_ppci(monkeypatch):

    def _mock_get_pf_variables_from_ppci(ppci):

        return ("baseMVA", "bus", "gen", "branch", "ref", "pv", "pq", "on",
                "gbus", "V0", "ref_gens")

    monkeypatch.setattr(run_newton_raphson_pf, "_get_pf_variables_from_ppci",
                        _mock_get_pf_variables_from_ppci)


@pytest.fixture
def mock_get_Y_bus(monkeypatch):

    def _mock_get_Y_bus(ppci, options, makeYbus, baseMVA, bus, branch):
        return ppci, "Ybus", "Yf", "Yt"

    monkeypatch.setattr(run_newton_raphson_pf, "_get_Y_bus", _mock_get_Y_bus)


@pytest.fixture
def mock_get_Sbus(monkeypatch):

    def _mock_get_Sbus(ppci, options_recycle):
        return "Sbus"

    monkeypatch.setattr(run_newton_raphson_pf, "_get_Sbus", _mock_get_Sbus)


@pytest.fixture
def mock_newtonpf(monkeypatch):

    def _mock_newtonpf(Ybus, Sbus, V0, pv, pq, ppci, options):
        return "V", "success", "iterations", "J", "Vm_it", "Va_it"

    monkeypatch.setattr(run_newton_raphson_pf, "newtonpf", 
                        _mock_newtonpf)


@pytest.fixture
def mock_decoupledpf(monkeypatch):

    def _mock_decoupledpf(Ybus, Sbus, V0, pv, pq, ppci, options):
        return "V", "success", "iterations", "Bp", "Bpp", "Vm_it", "Va_it"

    monkeypatch.setattr(run_newton_raphson_pf, "decoupledpf",
                        _mock_decoupledpf)


@pytest.fixture
def mock_run_ac_without_qlims(monkeypatch):
    pass


@pytest.mark.parametrize(
    "options, expected",
    [
        pytest.param({"init_va_degree": "dc", "enforce_q_lims": False,
                      "mode": "pf", "ac": True, "algorithm": "nr"},
                     {"init_va_degree": True, "enforced_qlims": False,
                      "algorithm": "nr"},
                     id="dc_init_and_enforce_qlims_false"),
        pytest.param({"init_va_degree": None, "enforce_q_lims": True,
                      "mode": "pf", "ac": True, "algorithm": "nr"},
                     {"enforced_qlims": True, "algorithm": "nr"},
                     id="no_dc_init_and_enforce_qlims_true")
    ]
)
def test_run_newton_raphson_pf(mock_run_dc_pf, mock_run_ac_pf_with_qlims,
                               mock_run_ac_pf_without_qlims,
                               mock_ppci_to_pfsoln,
                               mock_store_results_from_pf_in_ppci,
                               options, expected):
    
    ppci = dict()

    res = _run_newton_raphson_pf(ppci, options)

    assert res == expected


@pytest.mark.parametrize(
    "options, expected",
    [
        pytest.param({"init_va_degree": "dc", "enforce_q_lims": False,
                      "mode": "pf", "ac": True, "algorithm": "fdbx"},
                     {"init_va_degree": True, "enforced_qlims": False,
                      "algorithm": "fdbx"},
                     id="dc_init_and_enforce_qlims_false"),
        pytest.param({"init_va_degree": None, "enforce_q_lims": True,
                      "mode": "pf", "ac": True, "algorithm": "fdxb"},
                     {"enforced_qlims": True, "algorithm": "fdxb"},
                     id="no_dc_init_and_enforce_qlims_true")
    ]
)
def test_run_fast_decoupled_pf(mock_run_dc_pf, mock_run_ac_pf_without_qlims,
                                mock_run_ac_pf_with_qlims, mock_ppci_to_pfsoln,
                                mock_store_results_from_pf_in_ppci, options,
                                expected):

    ppci = dict()

    res = _run_fast_decoupled_pf(ppci, options)

    assert res == expected


@pytest.mark.parametrize("ppci, options, expected",
    [
        pytest.param({"internal": {}},
                     {"algorithm": "nr", "recycle": False},
                     ({"internal": {"J": "J", "Bp": None, "Bpp": None,
                                    "Vm_it": "Vm_it", "Va_it": "Va_it",
                                    "bus": "bus", "gen": "gen",
                                    "branch": "branch", "baseMVA": "baseMVA",
                                    "V": "V", "pv": "pv", "pq": "pq",
                                    "ref": "ref", "Sbus": "Sbus",
                                    "ref_gens": "ref_gens", "Ybus": "Ybus",
                                    "Yf": "Yf","Yt": "Yt"}
                       }, 'success', 'iterations')),
        pytest.param({"internal": {}},
                     {"algorithm": "fdbx", "recycle": False},
                     ({"internal": {"J": None, "Bp": "Bp", "Bpp": "Bpp",
                                    "Vm_it": "Vm_it", "Va_it": "Va_it",
                                    "bus": "bus", "gen": "gen",
                                    "branch": "branch", "baseMVA": "baseMVA",
                                    "V": "V", "pv": "pv", "pq": "pq",
                                    "ref": "ref", "Sbus": "Sbus",
                                    "ref_gens": "ref_gens", "Ybus": "Ybus",
                                    "Yf": "Yf","Yt": "Yt"}
                       }, 'success', 'iterations'))
    ]
)
def test_run_ac_pf_without_qlims_enforced(
    mock_get_numba_functions, mock_get_pf_variables_from_ppci, mock_get_Y_bus,
    mock_get_Sbus, mock_newtonpf, mock_decoupledpf, ppci, options, expected
):

    
    ppci, success, iterations = _run_ac_pf_without_qlims_enforced(ppci,
                                                                  options)

    assert expected == (ppci, success, iterations)


@pytest.mark.xfail
def test_run_ac_pf_with_qlims_enforced():
    # TODO: Fazer esta funcao de testes
    #   Notar que -> net["options"]["enforce_qlims"] deveria ser True ou False,
    #   mas assume valor 2 para um if interno. Avaliar. Colocar entradas que
    #   percorram todos os IFs internos

    # Esta funcao de fato implementa algumas logicas, que devem ser testadas.
    # Devemos de fato testar a logica implementada e se ela faz sentido
    pytest.xfail("teste ainda deve ser implementado")
