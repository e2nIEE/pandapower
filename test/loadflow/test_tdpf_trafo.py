# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Tests of the transformer thermal model integration into the TDPF.

The transformer temperatures are post-processing results of the optional `transformer-thermal-model`
library. These tests cover the nameplate adapter, the physical correctness of the results and the
lifecycle (columns only appear when the calculation actually ran).
"""

import numpy as np
import pytest

from pandapower import pandapowerNet
from pandapower.create import (
    create_buses,
    create_ext_grid,
    create_load,
    create_sgen,
    create_transformer3w_from_parameters,
    create_transformer_from_parameters,
)
from pandapower.pf import transformer_thermal
from pandapower.pf.transformer_thermal import (
    HOT_SPOT_3W_COLUMNS,
    HOT_SPOT_COLUMN,
    TEMPERATURE_COLUMN,
    TOP_OIL_COLUMN,
    calculate_transformer_temperatures,
    specs_from_trafo,
    specs_from_trafo3w,
)
from pandapower.run import runpp

pytest.importorskip("transformer_thermal_model")

# 2-winding transformer parameters used throughout the tests
TRAFO_PARAMS = {
    "sn_mva": 40.0,
    "vn_hv_kv": 110.0,
    "vn_lv_kv": 10.0,
    "vk_percent": 10.0,
    "vkr_percent": 0.5,
    "pfe_kw": 20.0,
    "i0_percent": 0.1,
}
# 3-winding transformer parameters used throughout the tests
TRAFO3W_PARAMS = {
    "sn_hv_mva": 100.0,
    "sn_mv_mva": 50.0,
    "sn_lv_mva": 25.0,
    "vn_hv_kv": 110.0,
    "vn_mv_kv": 20.0,
    "vn_lv_kv": 10.0,
    "vk_hv_percent": 10.0,
    "vk_mv_percent": 8.0,
    "vk_lv_percent": 12.0,
    "vkr_hv_percent": 0.5,
    "vkr_mv_percent": 0.4,
    "vkr_lv_percent": 0.6,
    "pfe_kw": 20.0,
    "i0_percent": 0.1,
}


def _two_bus_net(p_mw=20.0, q_mvar=5.0, **kwargs):
    """A minimal net with one two winding transformer and a load at the LV bus."""
    net = pandapowerNet(name="trafo_thermal_2w")
    hv_bus, lv_bus = create_buses(net, nr_buses=2, vn_kv=[110.0, 10.0])
    create_ext_grid(net, hv_bus)
    params = {**TRAFO_PARAMS, **kwargs}
    create_transformer_from_parameters(net, hv_bus=hv_bus, lv_bus=lv_bus, **params)
    if p_mw >= 0:
        create_load(net, lv_bus, p_mw=p_mw, q_mvar=q_mvar)
    else:
        create_sgen(net, lv_bus, p_mw=-p_mw, q_mvar=-q_mvar)
    return net


def _three_bus_net(p_mw_mv=10.0, p_mw_lv=10.0):
    """A minimal net with one three winding transformer and loads at the MV and LV bus."""
    net = pandapowerNet(name="trafo_thermal_3w")
    hv_bus, mv_bus, lv_bus = create_buses(net, nr_buses=3, vn_kv=[110.0, 20.0, 10.0])
    create_ext_grid(net, hv_bus)
    create_transformer3w_from_parameters(net, hv_bus=hv_bus, mv_bus=mv_bus, lv_bus=lv_bus, **TRAFO3W_PARAMS)
    create_load(net, mv_bus, p_mw=p_mw_mv, q_mvar=0.2 * p_mw_mv)
    create_load(net, lv_bus, p_mw=p_mw_lv, q_mvar=0.2 * p_mw_lv)
    return net


# --------------------------------------------------------------- nameplate adapter
def test_specs_from_trafo():
    net = _two_bus_net()
    specs = specs_from_trafo(net, 0)
    sn_mva = TRAFO_PARAMS["sn_mva"]
    assert specs.no_load_loss == pytest.approx(TRAFO_PARAMS["pfe_kw"] * 1e3)
    assert specs.load_loss == pytest.approx(TRAFO_PARAMS["vkr_percent"] / 100.0 * sn_mva * 1e6)
    assert specs.nom_load_sec_side == pytest.approx(sn_mva * 1e3 / (np.sqrt(3.0) * TRAFO_PARAMS["vn_lv_kv"]))


def test_specs_from_trafo3w_pairwise_losses():
    """The vkr suffixes denote the *pair*, not the winding, and are based on min(sn_a, sn_b)."""
    net = _three_bus_net()
    specs = specs_from_trafo3w(net, 0)

    sn_hv, sn_mv, sn_lv = TRAFO3W_PARAMS["sn_hv_mva"], TRAFO3W_PARAMS["sn_mv_mva"], TRAFO3W_PARAMS["sn_lv_mva"]
    # vkr_hv_percent is the HV <-> MV pair
    assert specs.load_loss_hv_mv == pytest.approx(TRAFO3W_PARAMS["vkr_hv_percent"] / 100.0 * min(sn_hv, sn_mv) * 1e6)
    # vkr_mv_percent is the MV <-> LV pair
    assert specs.load_loss_mv_lv == pytest.approx(TRAFO3W_PARAMS["vkr_mv_percent"] / 100.0 * min(sn_mv, sn_lv) * 1e6)
    # vkr_lv_percent is the HV <-> LV pair -- note the HV/LV swap compared to the suffix
    assert specs.load_loss_hv_lv == pytest.approx(TRAFO3W_PARAMS["vkr_lv_percent"] / 100.0 * min(sn_hv, sn_lv) * 1e6)
    assert specs.no_load_loss == pytest.approx(TRAFO3W_PARAMS["pfe_kw"] * 1e3)
    # per winding ratings map to nom_load / nom_power
    assert specs.hv_winding.nom_power == pytest.approx(sn_hv)
    assert specs.mv_winding.nom_power == pytest.approx(sn_mv)
    assert specs.lv_winding.nom_power == pytest.approx(sn_lv)
    assert specs.hv_winding.nom_load == pytest.approx(sn_hv * 1e3 / (np.sqrt(3.0) * TRAFO3W_PARAMS["vn_hv_kv"]))


def test_specs_from_trafo3w_matches_pandapower_star_conversion():
    """The pairwise losses must reproduce pandapower's own star equivalent winding losses."""
    from pandapower.build_branch import wye_delta_vector, z_br_to_bus_vector

    net = _three_bus_net()
    trafo = net.trafo3w.loc[0]
    vkr_3w = np.array([[trafo.vkr_hv_percent], [trafo.vkr_mv_percent], [trafo.vkr_lv_percent]])
    sn = np.array([[trafo.sn_hv_mva], [trafo.sn_mv_mva], [trafo.sn_lv_mva]])
    vkr_star = wye_delta_vector(z_br_to_bus_vector(vkr_3w, sn), sn)[:, 0]
    pandapower_winding_losses = vkr_star / 100.0 * sn[:, 0] * 1e6

    # the library derives the per winding (star) losses from the pairwise losses we pass in
    specs = specs_from_trafo3w(net, 0)
    transformer = transformer_thermal.ThreeWindingTransformer(
        user_specs=specs, cooling_type=transformer_thermal.CoolerType.ONAN
    )
    library_winding_losses = np.array(
        [
            transformer.specs._get_loss_hc(),
            transformer.specs._get_loss_mc(),
            transformer.specs._get_loss_lc(),
        ]
    )

    # order of the pandapower star conversion is hv, mv, lv -- same as library hc/mc/lc
    assert np.allclose(library_winding_losses, pandapower_winding_losses)


# --------------------------------------------------------------- physical correctness
def test_trafo_temperatures_are_steady_state():
    """The default (tdpf_delay_s=None) result must be the analytic IEC 60076-7 steady state."""
    net = _two_bus_net()
    runpp(net, tdpf=True)

    load_a = abs(net.res_trafo.at[0, "i_lv_ka"]) * 1e3
    specs = specs_from_trafo(net, 0)
    load_ratio = (load_a / specs.nom_load_sec_side) ** 2
    total_loss_ratio = (specs.no_load_loss + specs.load_loss * load_ratio) / (specs.no_load_loss + specs.load_loss)
    expected_top_oil = 35.0 + 60.0 * total_loss_ratio**0.8

    assert net.res_trafo.at[0, TOP_OIL_COLUMN] == pytest.approx(expected_top_oil, rel=1e-6)
    # the hot-spot is above the top-oil temperature and the alias equals it
    assert net.res_trafo.at[0, HOT_SPOT_COLUMN] == pytest.approx(net.res_trafo.at[0, TEMPERATURE_COLUMN])
    assert net.res_trafo.at[0, TEMPERATURE_COLUMN] > net.res_trafo.at[0, TOP_OIL_COLUMN]


def test_trafo_temperatures_cooling_type():
    """ONAF has the shorter oil/winding time constants, so a transient step heats up faster."""

    def transient(cooling_type):
        net = _two_bus_net(p_mw=30.0, q_mvar=8.0)
        net.trafo["cooling_type"] = cooling_type
        runpp(net, tdpf=True, tdpf_delay_s=1800.0)
        return net.res_trafo.at[0, TOP_OIL_COLUMN]

    # the steady state is identical for both cooling types; the difference is the thermal inertia
    assert transient("ONAF") > transient("ONAN")


def test_trafo_temperatures_cooling_type_steady_state():
    """Without a time delay (steady state) the cooling type does not change the result."""
    results = []
    for cooling_type in ("ONAN", "ONAF"):
        net = _two_bus_net(p_mw=30.0, q_mvar=8.0)
        net.trafo["cooling_type"] = cooling_type
        runpp(net, tdpf=True)
        results.append(net.res_trafo.at[0, TOP_OIL_COLUMN])
    assert results[0] == pytest.approx(results[1])


def test_trafo_temperature_uses_ambient_temperature():
    net = _two_bus_net()
    net.trafo["air_temperature_degree_celsius"] = 20.0
    runpp(net, tdpf=True)
    cold = net.res_trafo.at[0, TOP_OIL_COLUMN]

    net = _two_bus_net()
    net.trafo["air_temperature_degree_celsius"] = 40.0
    runpp(net, tdpf=True)
    warm = net.res_trafo.at[0, TOP_OIL_COLUMN]

    # the top-oil steady state shifts by the ambient difference
    assert warm - cold == pytest.approx(20.0, abs=0.5)


def test_trafo_reverse_power_flow():
    """Heating scales with I^2, so the current magnitude must be used, not the signed power."""
    forward = _two_bus_net(p_mw=20.0)
    runpp(forward, tdpf=True)

    # force the exact same current magnitude with the opposite sign and recalculate
    reverse = _two_bus_net(p_mw=20.0)
    runpp(reverse, tdpf=True)
    reverse.res_trafo.at[0, "i_lv_ka"] = -abs(reverse.res_trafo.at[0, "i_lv_ka"])
    calculate_transformer_temperatures(reverse)
    assert reverse.res_trafo.at[0, TEMPERATURE_COLUMN] == pytest.approx(
        forward.res_trafo.at[0, TEMPERATURE_COLUMN], rel=1e-12
    )


def test_trafo_parallel_factor():
    """Two parallel transformers at twice the current must be as hot as one at the per unit current."""
    single = _two_bus_net(p_mw=20.0, q_mvar=5.0, parallel=1)
    runpp(single, tdpf=True)

    double = _two_bus_net(p_mw=40.0, q_mvar=10.0, parallel=2)
    runpp(double, tdpf=True)

    assert double.res_trafo.at[0, TOP_OIL_COLUMN] == pytest.approx(single.res_trafo.at[0, TOP_OIL_COLUMN], rel=1e-4)


def test_trafo3w_temperatures():
    net = _three_bus_net()
    runpp(net, tdpf=True)
    row = net.res_trafo3w.loc[0]

    for column in HOT_SPOT_3W_COLUMNS:
        assert np.isfinite(row[column])
    # the reported temperature is the hottest winding
    hotspots = [row[column] for column in HOT_SPOT_3W_COLUMNS]
    assert row[TEMPERATURE_COLUMN] == pytest.approx(max(hotspots))
    assert np.isfinite(row[TOP_OIL_COLUMN])
    assert row[TEMPERATURE_COLUMN] > row[TOP_OIL_COLUMN]


def test_equivalence_with_direct_library_call():
    """Guards the adapter against drift from the library API."""
    from transformer_thermal_model.cooler import CoolerType
    from transformer_thermal_model.model import Model
    from transformer_thermal_model.schemas import InputProfile
    from transformer_thermal_model.transformer import PowerTransformer

    net = _two_bus_net()
    runpp(net, tdpf=True)

    current_a = abs(net.res_trafo.at[0, "i_lv_ka"]) * 1e3
    profile = InputProfile.create(
        datetime_index=transformer_thermal._profile_index(transformer_thermal.STEADY_STATE_DT_MINUTES),
        load_profile=[current_a, current_a],
        ambient_temperature_profile=[35.0, 35.0],
    )
    transformer = PowerTransformer(user_specs=specs_from_trafo(net, 0), cooling_type=CoolerType.ONAN)
    output = Model(temperature_profile=profile, transformer=transformer).run()

    assert net.res_trafo.at[0, TOP_OIL_COLUMN] == pytest.approx(float(output.top_oil_temp_profile.iloc[-1]), rel=1e-12)
    assert net.res_trafo.at[0, HOT_SPOT_COLUMN] == pytest.approx(
        float(output.hot_spot_temp_profile.iloc[-1]), rel=1e-12
    )


# --------------------------------------------------------------- lifecycle / regression
def test_no_columns_when_tdpf_disabled():
    net = _two_bus_net()
    runpp(net, tdpf=False)
    assert TEMPERATURE_COLUMN not in net.res_trafo.columns
    assert TOP_OIL_COLUMN not in net.res_trafo.columns
    assert HOT_SPOT_COLUMN not in net.res_trafo.columns


def test_no_columns_when_library_missing(monkeypatch, caplog):
    """Without the optional package the line TDPF behaviour must be unchanged."""
    monkeypatch.setattr(transformer_thermal, "TRANSFORMER_THERMAL_AVAILABLE", False)
    net = _two_bus_net()
    with caplog.at_level("WARNING"):
        runpp(net, tdpf=True)

    assert any("transformer-thermal-model" in record.message for record in caplog.records)
    assert TEMPERATURE_COLUMN not in net.res_trafo.columns
    assert TOP_OIL_COLUMN not in net.res_trafo.columns
    # the regular power flow results are still there
    assert np.isfinite(net.res_trafo.at[0, "i_lv_ka"])


def test_trafo_thermal_without_tdpf_lines():
    """A net with transformers but no TDPF lines still gets transformer temperatures."""
    net = _two_bus_net()
    assert len(net.line) == 0
    runpp(net, tdpf=True)
    assert np.isfinite(net.res_trafo.at[0, TEMPERATURE_COLUMN])
    assert np.isfinite(net.res_trafo.at[0, TOP_OIL_COLUMN])


def test_out_of_service_trafo_gets_nan():
    """An out-of-service transformer takes no part in the thermal calculation and stays NaN."""
    net = _two_bus_net()
    # a second transformer so that the calculation still runs while one unit is out of service
    hv_bus = int(net.trafo.at[0, "hv_bus"])
    lv_bus = int(net.trafo.at[0, "lv_bus"])
    create_transformer_from_parameters(net, hv_bus=hv_bus, lv_bus=lv_bus, **TRAFO_PARAMS)
    net.trafo.loc[1, "in_service"] = False
    runpp(net, tdpf=True)

    assert np.isfinite(net.res_trafo.at[0, TEMPERATURE_COLUMN])
    assert np.isnan(net.res_trafo.at[1, TEMPERATURE_COLUMN])


def test_warm_start_chains_previous_top_oil_temperature():
    """The time series tracker must feed the previous top-oil temperature into the next step."""
    tracker = transformer_thermal.TransformerThermalTracker()
    net = _two_bus_net()
    runpp(net, tdpf=True)
    steady_state = net.res_trafo.at[0, TOP_OIL_COLUMN]

    # one hour at the same loading: starting from the steady state must stay there
    tracker.top_oil_temperature = {("trafo", 0): steady_state}
    tracker.run(net, dt_minutes=60.0)
    assert net.res_trafo.at[0, TOP_OIL_COLUMN] == pytest.approx(steady_state, rel=1e-6)

    # starting from ambient (cold) after one hour must be *below* the steady state
    tracker.reset()
    tracker.run(net, dt_minutes=60.0)
    assert net.res_trafo.at[0, TOP_OIL_COLUMN] < steady_state


def test_tdpf_delay_gives_transient_below_steady_state():
    net = _two_bus_net()
    runpp(net, tdpf=True)
    steady_state = net.res_trafo.at[0, TOP_OIL_COLUMN]

    net = _two_bus_net()
    runpp(net, tdpf=True, tdpf_delay_s=3600.0)
    transient = net.res_trafo.at[0, TOP_OIL_COLUMN]

    assert transient < steady_state
    assert transient > 35.0


def test_invalid_cooling_type_raises():
    net = _two_bus_net()
    net.trafo["cooling_type"] = "FOO"
    with pytest.raises(UserWarning, match="invalid cooling_type"):
        runpp(net, tdpf=True)


def test_cooling_type_is_normalised():
    net = _two_bus_net()
    net.trafo["cooling_type"] = "onan"
    runpp(net, tdpf=True)
    assert net.trafo.at[0, "cooling_type"] == "ONAN"


def test_default_columns_are_filled():
    net = _two_bus_net()
    runpp(net, tdpf=True)
    assert net.trafo.at[0, "cooling_type"] == "ONAN"
    assert net.trafo.at[0, "air_temperature_degree_celsius"] == pytest.approx(35.0)
