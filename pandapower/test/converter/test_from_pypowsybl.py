# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandapower.converter.pypowsybl.pypowsybl_converter import PyPowSyBlConverter

pypowsybl_network = pytest.importorskip("pypowsybl.network")
convert_from_pandapower = pypowsybl_network.convert_from_pandapower

logger = logging.getLogger(__name__)


def _create_powsybl_ieee_network(pyp: Any, factory_name: str) -> Any:
    factory: Any = getattr(pyp.network, factory_name, None)

    if factory is None:
        pytest.skip(f"pypowsybl.network.{factory_name} is not available")

    return factory()


def case_ieee9(pyp: Any) -> Any:
    return _create_powsybl_ieee_network(pyp, "create_ieee9")


def case_ieee14(pyp: Any) -> Any:
    return _create_powsybl_ieee_network(pyp, "create_ieee14")


def case_ieee118(pyp: Any) -> Any:
    return _create_powsybl_ieee_network(pyp, "create_ieee118")


def case_custom_1(pyp: Any) -> Any:
    net = pyp.network.create_empty()

    net.create_substations(id=["S1"], name=["Substation 1"])

    net.create_voltage_levels(
        id=["10 kV"],
        substation_id=["S1"],
        topology_kind=["BUS_BREAKER"],
        nominal_v=[10.0],
    )
    net.create_voltage_levels(
        id=["0.4 kV"],
        substation_id=["S1"],
        topology_kind=["BUS_BREAKER"],
        nominal_v=[0.4],
    )

    net.create_buses(id=["mv_bus"], voltage_level_id=["10 kV"])
    net.create_buses(id=["lv_bus"], voltage_level_id=["0.4 kV"])

    net.create_buses(
        id=["low2", "low3", "low4", "low5", "low6"],
        voltage_level_id=["0.4 kV"] * 5,
    )

    net.create_generators(
        id="grid",
        voltage_level_id="10 kV",
        bus_id="mv_bus",
        min_p=-1,
        max_p=1.0,
        target_p=0.08,
        voltage_regulator_on=True,
        target_v=10,
    )

    net.create_2_windings_transformers(
        id="transformer",
        voltage_level1_id="10 kV",
        bus1_id="mv_bus",
        voltage_level2_id="0.4 kV",
        bus2_id="lv_bus",
        b=0,
        g=0,
        r=0.01,
        x=0.04,
        rated_u2=0.4,
        rated_u1=10.0,
        rated_s=0.25,
    )

    r = 0.01
    x = 0.005
    net.create_lines(
        id="line0",
        voltage_level1_id="0.4 kV",
        bus1_id="lv_bus",
        voltage_level2_id="0.4 kV",
        bus2_id="low2",
        b1=1e-6,
        b2=1e-6,
        g1=0,
        g2=0,
        r=r,
        x=x,
    )
    net.create_lines(
        id="line1",
        voltage_level1_id="0.4 kV",
        bus1_id="low2",
        voltage_level2_id="0.4 kV",
        bus2_id="low3",
        b1=1e-6,
        b2=1e-6,
        g1=0,
        g2=0,
        r=r,
        x=x,
    )
    net.create_lines(
        id="line2",
        voltage_level1_id="0.4 kV",
        bus1_id="low3",
        voltage_level2_id="0.4 kV",
        bus2_id="low4",
        b1=1e-6,
        b2=1e-6,
        g1=0,
        g2=0,
        r=r,
        x=x,
    )
    net.create_lines(
        id="line3",
        voltage_level1_id="0.4 kV",
        bus1_id="low4",
        voltage_level2_id="0.4 kV",
        bus2_id="low5",
        b1=1e-6,
        b2=1e-6,
        g1=0,
        g2=0,
        r=r,
        x=x,
    )
    net.create_lines(
        id="line4",
        voltage_level1_id="0.4 kV",
        bus1_id="low6",
        voltage_level2_id="0.4 kV",
        bus2_id="low5",
        b1=1e-6,
        b2=1e-6,
        g1=0,
        g2=0,
        r=r,
        x=x,
    )
    net.create_lines(
        id="line5",
        voltage_level1_id="0.4 kV",
        bus1_id="lv_bus",
        voltage_level2_id="0.4 kV",
        bus2_id="low6",
        b1=1e-6,
        b2=1e-6,
        g1=0,
        g2=0,
        r=r,
        x=x,
    )

    net.create_loads(
        id="load low3", voltage_level_id="0.4 kV", bus_id="low3", p0=0.03, q0=0.01
    )

    net.create_loads(
        id="load low5", voltage_level_id="0.4 kV", bus_id="low5", p0=0.04, q0=0.015
    )

    return net


def case_custom_2(pyp: Any) -> Any:
    net = pyp.network.create_empty()

    net.create_substations(id=["S1"], name=["Substation 1"])

    net.create_voltage_levels(
        id=["10 kV"],
        substation_id=["S1"],
        topology_kind=["BUS_BREAKER"],
        nominal_v=[10],
    )

    net.create_buses(id=["mv_bus1"], voltage_level_id=["10 kV"])
    net.create_buses(id=["mv_bus2"], voltage_level_id=["10 kV"])

    net.create_loads(
        id=["test_load"], voltage_level_id="10 kV", bus_id="mv_bus2", p0=10, q0=3
    )

    net.create_generators(
        id="grid",
        voltage_level_id="10 kV",
        bus_id="mv_bus1",
        min_p=0,
        max_p=200,
        target_p=100,
        voltage_regulator_on=True,
        target_v=10,
    )

    net.create_switches(
        id="switch",
        voltage_level_id="10 kV",
        bus1_id="mv_bus1",
        bus2_id="mv_bus2",
        kind="LOAD_BREAK_SWITCH",
        open=False,
    )

    return net


def case_custom_3(pyp: Any) -> Any:
    net = pyp.network.create_empty()

    net.create_substations(id=["S1"], name=["Substation 1"])

    net.create_voltage_levels(
        id=["110 kV"],
        substation_id=["S1"],
        topology_kind=["BUS_BREAKER"],
        nominal_v=[110],
    )
    net.create_voltage_levels(
        id=["20 kV"],
        substation_id=["S1"],
        topology_kind=["BUS_BREAKER"],
        nominal_v=[20],
    )
    net.create_voltage_levels(
        id=["10 kV"],
        substation_id=["S1"],
        topology_kind=["BUS_BREAKER"],
        nominal_v=[10],
    )

    net.create_buses(id=["hv_bus"], voltage_level_id=["110 kV"])
    net.create_buses(id=["mv_bus"], voltage_level_id=["20 kV"])
    net.create_buses(id=["lv_bus"], voltage_level_id=["10 kV"])

    net.create_loads(
        id=["test_load"], voltage_level_id="20 kV", bus_id="mv_bus", p0=10, q0=3
    )

    net.create_generators(
        id="grid",
        voltage_level_id="110 kV",
        bus_id="hv_bus",
        min_p=0,
        max_p=200,
        target_p=100,
        voltage_regulator_on=True,
        target_v=110,
    )

    net.create_3_windings_transformers(
        id="T-1",
        rated_u0=110,
        voltage_level1_id="110 kV",
        bus1_id="hv_bus",
        voltage_level2_id="20 kV",
        bus2_id="mv_bus",
        voltage_level3_id="10 kV",
        bus3_id="lv_bus",
        b1=1e-6,
        g1=1e-6,
        r1=0.5,
        x1=10,
        rated_u1=110,
        rated_s1=100,
        b2=1e-6,
        g2=1e-6,
        r2=0.5,
        x2=10,
        rated_u2=20,
        rated_s2=100,
        b3=1e-6,
        g3=1e-6,
        r3=0.5,
        x3=10,
        rated_u3=10,
        rated_s3=100,
    )
    return net


def case_pp_custom_1(pandapower: Any) -> Any:
    pp = pandapower

    net = pp.create_empty_network()

    mv_busbar = pp.create_bus(net, vn_kv=10, name="10 kV", type="b")
    lv_busbar = pp.create_bus(net, vn_kv=0.4, name="0.4 kV", type="b")

    bus_low2 = pp.create_bus(net, vn_kv=0.4, name="low1", type="n")
    bus_low3 = pp.create_bus(net, vn_kv=0.4, name="low2", type="n")
    bus_low4 = pp.create_bus(net, vn_kv=0.4, name="low3", type="n")
    bus_low5 = pp.create_bus(net, vn_kv=0.4, name="low4", type="n")
    bus_low6 = pp.create_bus(net, vn_kv=0.4, name="low5", type="n")

    pp.create_gen(
        net,
        bus=mv_busbar,
        p_mw=0.08,
        vm_pu=1.02,
        name="grid",
        min_p_mw=-1.0,
        max_p_mw=1.0,
        min_q_mvar=-1.0,
        max_q_mvar=1.0,
        slack=True,
        slack_weight=1.0,
    )

    pp.create_transformer(
        net, hv_bus=mv_busbar, lv_bus=lv_busbar, std_type="0.25 MVA 10/0.4 kV"
    )
    net.trafo.shift_degree = 0

    pp.create_line(
        net,
        from_bus=lv_busbar,
        to_bus=bus_low2,
        length_km=0.2,
        std_type="NAYY 4x120 SE",
    )
    pp.create_line(
        net,
        from_bus=bus_low2,
        to_bus=bus_low3,
        length_km=0.2,
        std_type="NAYY 4x120 SE",
    )
    pp.create_line(
        net,
        from_bus=bus_low3,
        to_bus=bus_low4,
        length_km=0.2,
        std_type="NAYY 4x120 SE",
    )
    pp.create_line(
        net,
        from_bus=bus_low4,
        to_bus=bus_low5,
        length_km=0.2,
        std_type="NAYY 4x120 SE",
    )
    pp.create_line(
        net,
        from_bus=bus_low6,
        to_bus=bus_low5,
        length_km=0.2,
        std_type="NAYY 4x120 SE",
    )
    pp.create_line(
        net,
        from_bus=lv_busbar,
        to_bus=bus_low6,
        length_km=0.2,
        std_type="NAYY 4x120 SE",
    )

    pp.create_load(net, bus=bus_low3, p_mw=0.03, q_mvar=0.01, name="load_low3")

    pp.create_load(net, bus=bus_low5, p_mw=0.03, q_mvar=0.01, name="load_low5")

    return net


PYP_NETWORK_CASES = [
    ("ieee9", case_ieee9),
    ("ieee14", case_ieee14),
    ("ieee118", case_ieee118),
    ("custom_1", case_custom_1),
    ("custom_2", case_custom_2),
    ("custom_3", case_custom_3),
]

PANDAPOWER_NETWORK_CASES = [("pp_custom_1", case_pp_custom_1)]


@pytest.fixture(scope="module")
def pyp() -> Any:
    return pytest.importorskip("pypowsybl")


@pytest.fixture(scope="module")
def pyp_lf() -> Any:
    return pytest.importorskip("pypowsybl.loadflow")


@pytest.fixture(scope="module")
def pandapower() -> Any:
    return pytest.importorskip("pandapower")


def _save_network_as_xiidm(network: Any, path: Path) -> Path:
    network.save(str(path), format="XIIDM")
    assert path.exists()
    return path


def _powsybl_loadflow_must_converge(network: Any, pyp_lf: Any, label: str) -> list[Any]:
    """Run a powsybl AC load flow and fail the test if it does not converge."""
    results = pyp_lf.run_ac(network)

    assert results, f"{label}: pypowsybl returned no load-flow component results"

    statuses = [str(getattr(result, "status", "")).upper() for result in results]
    assert all(status.rsplit(".", 1)[-1] == "CONVERGED" for status in statuses), (
        f"{label}: pypowsybl load-flow did not converge: {statuses}"
    )

    return results


def _pandapower_to_powsybl(pandapower_net: Any, tmp_path: Path) -> tuple[Any, Path]:
    """Convert a pandapower net back to pypowsybl and store it as an XIIDM file."""
    xiidm_path = tmp_path / "roundtrip.xiidm"

    pyp_network = convert_from_pandapower(pandapower_net)
    pyp_network.save(str(xiidm_path), format="XIIDM")

    assert xiidm_path.exists()

    return pyp_network, xiidm_path


def _get_original_switch_ids(original_network: Any) -> set[str]:
    original_switches = _get_table_or_empty(original_network, "get_switches")

    if original_switches.empty:
        return set()

    return set(original_switches.index.astype(str))


def _align_roundtrip_table_by_original_ids(
    original_table: pd.DataFrame, roundtrip_table: pd.DataFrame, component_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align original and roundtrip tables by stable ids or preserved name values."""
    original_table = _normalize_table_index(original_table)
    roundtrip_table = _normalize_table_index(roundtrip_table)

    original_ids = set(original_table.index.astype(str))
    roundtrip_ids = set(roundtrip_table.index.astype(str))

    if original_ids == roundtrip_ids:
        original_table = original_table.loc[sorted(original_table.index)]
        roundtrip_table = roundtrip_table.loc[sorted(roundtrip_table.index)]
        return original_table, roundtrip_table

    if "name" in roundtrip_table.columns:
        roundtrip_names = roundtrip_table["name"].astype(str)

        if roundtrip_names.is_unique and set(roundtrip_names) == original_ids:
            roundtrip_table = roundtrip_table.copy()
            roundtrip_table.index = roundtrip_names
            roundtrip_table.index.name = original_table.index.name

            original_table = original_table.loc[sorted(original_table.index)]
            roundtrip_table = roundtrip_table.loc[sorted(roundtrip_table.index)]

            return original_table, roundtrip_table

    raise AssertionError(
        f"{component_name} ids changed during powsybl -> pandapower -> powsybl roundtrip "
        f"and could not be matched by the roundtrip 'name' column.\n"
        f"Only in original index: {sorted(original_ids - roundtrip_ids)}\n"
        f"Only in roundtrip index: {sorted(roundtrip_ids - original_ids)}\n"
        f"Roundtrip names: "
        f"{sorted(roundtrip_table['name'].astype(str)) if 'name' in roundtrip_table.columns else 'no name column'}"
    )


def _assert_roundtrip_line_shunt_sums_match(
    original_lines: pd.DataFrame, roundtrip_lines: pd.DataFrame
) -> None:
    """Check that total line shunt conductance and susceptance survive the roundtrip."""
    original_lines, roundtrip_lines = _align_roundtrip_table_by_original_ids(
        original_lines, roundtrip_lines, "lines"
    )

    shunt_pairs = [("g1", "g2", "g1 + g2"), ("b1", "b2", "b1 + b2")]

    for first_column, second_column, label in shunt_pairs:
        if first_column not in original_lines.columns:
            continue

        if second_column not in original_lines.columns:
            continue

        if first_column not in roundtrip_lines.columns:
            continue

        if second_column not in roundtrip_lines.columns:
            continue

        original_sum = (
            pd.to_numeric(original_lines[first_column], errors="coerce")
            + pd.to_numeric(original_lines[second_column], errors="coerce")
        ).to_numpy(dtype=float)

        roundtrip_sum = (
            pd.to_numeric(roundtrip_lines[first_column], errors="coerce")
            + pd.to_numeric(roundtrip_lines[second_column], errors="coerce")
        ).to_numpy(dtype=float)

        comparable_mask = np.isfinite(original_sum)

        if not comparable_mask.any():
            continue

        np.testing.assert_allclose(
            roundtrip_sum[comparable_mask],
            original_sum[comparable_mask],
            rtol=1e-6,
            atol=1e-9,
            err_msg=(
                f"lines total shunt value {label!r} differs after "
                f"powsybl -> pandapower -> powsybl roundtrip"
            ),
        )


def _original_bus_result_frame(network: Any) -> pd.DataFrame:
    """Build a bus result table from the original powsybl bus-breaker view."""
    buses = network.get_bus_breaker_view_buses(all_attributes=True).copy()

    required_columns = ["v_mag", "v_angle"]
    missing_columns = [
        column for column in required_columns if column not in buses.columns
    ]

    if missing_columns:
        pytest.skip(f"pypowsybl bus result columns missing: {missing_columns}")

    result = buses[required_columns].astype(float).copy()
    result = result.replace([np.inf, -np.inf], np.nan).dropna()
    result.index = result.index.astype(str)

    return result


def _roundtrip_bus_result_frame(
    roundtrip_network: Any, pandapower_network: Any
) -> pd.DataFrame:
    """Build a roundtrip powsybl bus result table mapped back to pandapower bus names."""
    buses = roundtrip_network.get_bus_breaker_view_buses(all_attributes=True).copy()

    required_columns = ["v_mag", "v_angle"]
    missing_columns = [
        column for column in required_columns if column not in buses.columns
    ]

    if missing_columns:
        pytest.skip(
            f"pypowsybl roundtrip bus-breaker result columns missing: {missing_columns}"
        )

    rows = []

    for roundtrip_bus_name, bus_row in buses.iterrows():
        pp_bus_index = _extract_bus_number(roundtrip_bus_name)

        if pp_bus_index is None:
            continue

        if pp_bus_index not in pandapower_network.bus.index:
            continue

        original_bus_name = str(pandapower_network.bus.loc[pp_bus_index, "name"])

        rows.append(
            {
                "bus_name": original_bus_name,
                "v_mag": float(bus_row["v_mag"]),
                "v_angle": float(bus_row["v_angle"]),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        pytest.skip(
            "No roundtrip buses could be mapped back to original pandapower bus names"
        )

    result = result.replace([np.inf, -np.inf], np.nan).dropna()
    result = result.set_index("bus_name").sort_index()

    return result


def _pandapower_bus_result_frame(pandapower_network: Any) -> pd.DataFrame:
    """Build a pandapower bus result table using physical voltage magnitudes in kV."""
    if not hasattr(pandapower_network, "res_bus"):
        pytest.skip("pandapower result table res_bus is missing")

    if pandapower_network.res_bus.empty:
        pytest.skip("pandapower res_bus is empty. Run pandapower.runpp() first.")

    rows = []

    for pp_bus_index, pp_bus in pandapower_network.bus.iterrows():
        if pp_bus_index not in pandapower_network.res_bus.index:
            continue

        pp_res_bus = pandapower_network.res_bus.loc[pp_bus_index]

        rows.append(
            {
                "bus_name": str(pp_bus["name"]),
                "v_mag": float(pp_bus["vn_kv"]) * float(pp_res_bus["vm_pu"]),
                "v_angle": float(pp_res_bus["va_degree"]),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        pytest.skip("No pandapower bus results could be mapped to bus names")

    result = result.replace([np.inf, -np.inf], np.nan).dropna()
    result = result.set_index("bus_name").sort_index()

    return result


def _split_trailing_number(value: Any) -> tuple[str, int | None]:
    text = str(value)

    start = len(text)

    while start > 0 and text[start - 1].isdigit():
        start -= 1

    if start == len(text):
        return text, None

    return text[:start], int(text[start:])


def _extract_bus_number(value: Any) -> int | None:
    _, number = _split_trailing_number(value)
    return number


def _bus_sort_key(bus_name: str) -> tuple[str, int]:
    prefix, number = _split_trailing_number(bus_name)

    if number is None:
        return prefix, -1

    return prefix, number


def _assert_same_bus_loadflow_results(
    original_network: Any, roundtrip_network: Any, pandapower_network: Any
) -> None:
    """Compare original and roundtrip powsybl bus load-flow results."""
    original_buses = _original_bus_result_frame(original_network)
    roundtrip_buses = _roundtrip_bus_result_frame(roundtrip_network, pandapower_network)

    common_buses = original_buses.index.intersection(roundtrip_buses.index)

    assert len(common_buses) > 0, (
        "No matching normalized bus numbers found between original and roundtrip network.\n"
        f"Original bus: {list(original_buses.index)}\n "
        f"Roundtrip bus: {list(roundtrip_buses.index)}"
    )

    assert len(common_buses) == len(original_buses) == len(roundtrip_buses), (
        "Roundtrip changed the comparable bus set.\n"
        f"Original bus: {list(original_buses.index)}\n"
        f"Roundtrip bus: {list(roundtrip_buses.index)}\n"
        f"Common bus: {list(common_buses)}"
    )

    bus_order = sorted(common_buses, key=_bus_sort_key)

    original_buses = original_buses.loc[bus_order]
    roundtrip_buses = roundtrip_buses.loc[bus_order]

    np.testing.assert_allclose(
        roundtrip_buses["v_mag"].to_numpy(),
        original_buses["v_mag"].to_numpy(),
        rtol=5e-3,
        atol=1e-2,
        err_msg="Bus voltage magnitudes differ after roundtrip conversion",
    )

    original_angles = (
        original_buses["v_angle"].to_numpy() - original_buses["v_angle"].to_numpy()[0]
    )
    roundtrip_angles = (
        roundtrip_buses["v_angle"].to_numpy() - roundtrip_buses["v_angle"].to_numpy()[0]
    )

    np.testing.assert_allclose(
        roundtrip_angles,
        original_angles,
        rtol=5e-3,
        atol=1e-1,
        err_msg="Bus voltage angles differ after roundtrip conversion",
    )


def _assert_same_pandapower_bus_loadflow_results(
    original_network: Any, pandapower_network: Any
) -> None:
    """Compare original powsybl bus results with converted pandapower bus results."""
    original_buses = _original_bus_result_frame(original_network)
    pandapower_buses = _pandapower_bus_result_frame(pandapower_network)

    common_buses = original_buses.index.intersection(pandapower_buses.index)

    assert len(common_buses) > 0, (
        "No matching buses found between original powsybl network and pandapower network.\n"
        f"Original powsybl buses: {list(original_buses.index)}\n"
        f"Pandapower buses: {list(pandapower_buses.index)}"
    )

    assert len(common_buses) == len(original_buses) == len(pandapower_buses), (
        "Converted pandapower network changed the comparable bus set.\n"
        f"Original powsybl buses: {list(original_buses.index)}\n"
        f"Pandapower buses: {list(pandapower_buses.index)}\n"
        f"Common buses: {list(common_buses)}"
    )

    bus_order = sorted(common_buses, key=_bus_sort_key)

    original_buses = original_buses.loc[bus_order]
    pandapower_buses = pandapower_buses.loc[bus_order]

    regulated_bus_names = set()

    original_generators = original_network.get_generators(all_attributes=True)
    pandapower_generators = pandapower_network.gen.copy()

    for generator_id, generator in original_generators.iterrows():
        if "voltage_regulator_on" not in generator.index:
            continue

        if not bool(generator["voltage_regulator_on"]):
            continue

        if "target_v" not in generator.index:
            continue

        target_v = float(generator["target_v"])

        if not np.isfinite(target_v):
            continue

        matching_pp_generator = pandapower_generators[
            pandapower_generators["name"].astype(str) == str(generator_id)
        ]

        if matching_pp_generator.empty:
            continue

        pp_bus_index = int(matching_pp_generator.iloc[0]["bus"])
        bus_name = _pandapower_bus_name(pandapower_network, pp_bus_index)

        if bus_name not in original_buses.index:
            continue

        original_v_mag = float(original_buses["v_mag"].loc[bus_name])

        if np.isclose(original_v_mag, target_v, rtol=5e-3, atol=1e-2):
            regulated_bus_names.add(bus_name)

    regulated_common_buses = [
        bus_name for bus_name in bus_order if bus_name in regulated_bus_names
    ]

    if regulated_common_buses:
        np.testing.assert_allclose(
            pandapower_buses.loc[regulated_common_buses, "v_mag"].to_numpy(),
            original_buses.loc[regulated_common_buses, "v_mag"].to_numpy(),
            rtol=5e-3,
            atol=1e-2,
            err_msg=(
                "Regulated bus voltage magnitudes differ after "
                "powsybl -> pandapower conversion"
            ),
        )

    np.testing.assert_allclose(
        pandapower_buses["v_mag"].to_numpy(),
        original_buses["v_mag"].to_numpy(),
        rtol=5e-2,
        atol=1e-1,
        err_msg=(
            "Bus voltage magnitudes differ too much after "
            "powsybl -> pandapower conversion"
        ),
    )

    original_angles = (
        original_buses["v_angle"].to_numpy() - original_buses["v_angle"].to_numpy()[0]
    )

    pandapower_angles = (
        pandapower_buses["v_angle"].to_numpy()
        - pandapower_buses["v_angle"].to_numpy()[0]
    )

    cross_voltage_line_ids = _get_cross_voltage_line_ids_from_pandapower(
        pandapower_network
    )

    angle_atol = 15.0 if cross_voltage_line_ids else 1.0

    np.testing.assert_allclose(
        pandapower_angles,
        original_angles,
        rtol=5e-2,
        atol=angle_atol,
        err_msg="Bus voltage angles differ after powsybl -> pandapower conversion",
    )


def _get_cross_voltage_line_ids_from_pandapower(pandapower_network: Any) -> set[str]:
    """Return powsybl line ids that were converted to pandapower impedance elements."""
    if not hasattr(pandapower_network, "impedance"):
        return set()

    pp_impedances = pandapower_network.impedance.copy()

    if pp_impedances.empty:
        return set()

    if "powsybl_element_type" not in pp_impedances.columns:
        return set()

    cross_voltage_impedances = pp_impedances[
        pp_impedances["powsybl_element_type"].astype(str) == "cross_voltage_line"
    ]

    if cross_voltage_impedances.empty:
        return set()

    if "powsybl_id" in cross_voltage_impedances.columns:
        return set(cross_voltage_impedances["powsybl_id"].astype(str))

    if "name" in cross_voltage_impedances.columns:
        return set(cross_voltage_impedances["name"].astype(str))

    return set()


def _assert_basic_network_size_is_preserved(
    original_network: Any, roundtrip_network: Any, pandapower_network: Any
) -> None:
    """Check that roundtrip component counts match the expected convertible elements."""
    cross_voltage_line_ids = _get_cross_voltage_line_ids_from_pandapower(
        pandapower_network
    )

    checks = [
        ("bus-breaker buses", lambda network: network.get_bus_breaker_view_buses(), 0),
        ("loads", lambda network: network.get_loads(), 0),
        ("generators", lambda network: network.get_generators(), 0),
        (
            "lines",
            lambda network: _get_table_or_empty(network, "get_lines"),
            len(cross_voltage_line_ids),
        ),
        (
            "2W transformers",
            lambda network: _get_table_or_empty(network, "get_2_windings_transformers"),
            0,
        ),
        (
            "3W transformers",
            lambda network: _get_table_or_empty(network, "get_3_windings_transformers"),
            len(_get_table_or_empty(original_network, "get_3_windings_transformers")),
        ),
        (
            "switches",
            lambda network: _get_table_or_empty(network, "get_switches"),
            len(_get_table_or_empty(original_network, "get_switches")),
        ),
    ]

    for component_name, getter, expected_missing_count in checks:
        original_count = len(getter(original_network))
        roundtrip_count = len(getter(roundtrip_network))

        expected_roundtrip_count = original_count - expected_missing_count

        assert roundtrip_count == expected_roundtrip_count, (
            f"Number of {component_name} changed during roundtrip: "
            f"Original = {original_count}\n"
            f"Expected missing because of pandapower impedance mapping = {expected_missing_count}\n"
            f"Expected roundtrip = {expected_roundtrip_count}\n"
            f"Actual roundtrip = {roundtrip_count}"
        )


def _get_first_existing_column(
    frame: pd.DataFrame, candidates: list[str]
) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column

    return None


def _require_first_existing_column(
    frame: pd.DataFrame, candidates: list[str], component_name: str
) -> str:
    column = _get_first_existing_column(frame, candidates)

    assert column is not None, (
        f"No matching column found for {component_name}. "
        f"Candidates = {candidates}, available columns = {list(frame.columns)}"
    )

    return column


def _assert_close_value(
    actual: Any, expected: Any, label: str, rtol: float = 1e-6, atol: float = 1e-9
) -> None:
    assert np.isclose(float(actual), float(expected), rtol=rtol, atol=atol), (
        f"{label} mismatch: expected powsybl = {expected}, pandapower = {actual}"
    )


def _pandapower_frequency_hz(pandapower_network: Any) -> float:
    if "f_hz" in pandapower_network:
        return float(pandapower_network["f_hz"])

    return 50.0


def _pandapower_bus_name(pandapower_network: Any, bus_index: Any) -> str:
    return str(pandapower_network.bus.loc[int(bus_index), "name"])


def _assert_converted_loads_match_powsybl(
    original_network: Any, pandapower_network: Any
) -> None:
    original_loads = original_network.get_loads(all_attributes=True).copy()
    pp_loads = pandapower_network.load.copy()

    assert len(original_loads) == len(pp_loads), (
        "Number of loads changed during conversion: "
        f"powsybl = {len(original_loads)}, pandapower = {len(pp_loads)}"
    )

    if original_loads.empty:
        return

    p_column = _get_first_existing_column(
        original_loads, ["p0", "p", "p_mw", "target_p"]
    )

    q_column = _get_first_existing_column(
        original_loads, ["q0", "q", "q_mvar", "target_q"]
    )

    if p_column is None:
        pytest.skip(
            f"No active power column found in powsybl loads. "
            f"Available columns: {list(original_loads.columns)}"
        )

    if q_column is None:
        pytest.skip(
            f"No reactive power column found in powsybl loads. "
            f"Available columns: {list(original_loads.columns)}"
        )

    for load_id, pyp_load in original_loads.iterrows():
        matching_pp_load = pp_loads[pp_loads["name"].astype(str) == str(load_id)]

        assert not matching_pp_load.empty, (
            f"Load {load_id!r} exists in powsybl but not in pandapower. "
            f"Pandapower loads: {list(pp_loads['name'].astype(str))}"
        )

        pp_load = matching_pp_load.iloc[0]

        expected_p_mw = float(pyp_load[p_column])
        expected_q_mvar = float(pyp_load[q_column])

        assert np.isclose(
            float(pp_load["p_mw"]), expected_p_mw, rtol=1e-6, atol=1e-9
        ), (
            f"Load {load_id!r} active power mismatch: "
            f"powsybl = {expected_p_mw}, pandapower = {pp_load['p_mw']}"
        )

        assert np.isclose(
            float(pp_load["q_mvar"]), expected_q_mvar, rtol=1e-6, atol=1e-9
        ), (
            f"Load {load_id!r} reactive power mismatch: "
            f"powsybl = {expected_q_mvar}, pandapower = {pp_load['q_mvar']}"
        )

        assert int(pp_load["bus"]) in pandapower_network.bus.index, (
            f"Load {load_id!r} is connected to an invalid pandapower bus: "
            f"{pp_load['bus']}"
        )


def _assert_converted_generators_match_powsybl(
    original_network: Any, pandapower_network: Any
) -> None:
    original_generators = original_network.get_generators(all_attributes=True).copy()
    pp_gens = pandapower_network.gen.copy()

    assert len(pp_gens) == len(original_generators), (
        "Wrong number of converted generators. "
        f"powsybl = {len(original_generators)}, pandapower = {len(pp_gens)}"
    )

    assert "name" in pp_gens.columns

    pp_gen_names = set(pp_gens["name"].astype(str))
    original_gen_ids = set(original_generators.index.astype(str))

    assert pp_gen_names == original_gen_ids, (
        "Generator ids were not preserved as pandapower gen names.\n"
        f"Only in powsybl: {sorted(original_gen_ids - pp_gen_names)}\n"
        f"Only in pandapower: {sorted(pp_gen_names - original_gen_ids)}\n"
    )

    assert "slack" in pp_gens.columns
    assert int(pp_gens["slack"].astype(bool).sum()) == 1, (
        "Exactly one pandapower generator must have slack = True"
    )

    for gen_id, pyp_gen in original_generators.iterrows():
        pp_gen = pp_gens[pp_gens["name"].astype(str) == str(gen_id)].iloc[0]

        expected_p = float(pyp_gen["target_p"])
        actual_p = float(pp_gen["p_mw"])

        assert np.isclose(actual_p, expected_p, rtol=1e-6, atol=1e-9), (
            f"Generator {gen_id!r} target_p mismatch: "
            f"powsybl = {expected_p}, pandapower = {actual_p}"
        )

        if "min_p" in original_generators.columns and "min_p_mw" in pp_gens.columns:
            expected_min_p = float(pyp_gen["min_p"])

            if np.isfinite(expected_min_p):
                actual_min_p = float(pp_gen["min_p_mw"])

                assert np.isclose(actual_min_p, expected_min_p, rtol=1e-6, atol=1e-9), (
                    f"Generator {gen_id!r} min_p mismatch: "
                    f"powsybl = {expected_min_p}, pandapower = {actual_min_p}"
                )

        if "max_p" in original_generators.columns and "max_p_mw" in pp_gens.columns:
            expected_max_p = float(pyp_gen["max_p"])

            if np.isfinite(expected_max_p):
                actual_max_p = float(pp_gen["max_p_mw"])

                assert np.isclose(actual_max_p, expected_max_p, rtol=1e-6, atol=1e-9), (
                    f"Generator {gen_id!r} max_p mismatch: "
                    f"powsybl = {expected_max_p}, pandapower = {actual_max_p}"
                )


def _assert_converted_lines_match_powsybl(
    original_network: Any, pandapower_network: Any
) -> None:
    """Check converted pandapower line and impedance elements against powsybl lines."""
    original_lines = _get_table_or_empty(original_network, "get_lines")

    pp_lines = pandapower_network.line.copy()

    if hasattr(pandapower_network, "impedance"):
        pp_impedances = pandapower_network.impedance.copy()
    else:
        pp_impedances = pd.DataFrame()

    if not pp_impedances.empty and "powsybl_element_type" in pp_impedances.columns:
        pp_line_impedances = pp_impedances[
            pp_impedances["powsybl_element_type"].astype(str) == "cross_voltage_line"
        ].copy()
    else:
        pp_line_impedances = pd.DataFrame()

    converted_line_count = len(pp_lines) + len(pp_line_impedances)

    assert len(original_lines) == converted_line_count, (
        "Number of converted line-like elements changed during conversion: "
        f"powsybl lines = {len(original_lines)}, "
        f"pandapower = {len(pp_lines)}, "
        f"pandapower cross-voltage impedances = {len(pp_line_impedances)}"
    )

    if original_lines.empty:
        return

    r_column = _require_first_existing_column(
        original_lines, ["r", "r_ohm"], "line resistance"
    )

    x_column = _require_first_existing_column(
        original_lines, ["x", "x_ohm"], "line reactance"
    )

    g1_column = _get_first_existing_column(original_lines, ["g1", "g1_s"])
    g2_column = _get_first_existing_column(original_lines, ["g2", "g2_s"])
    b1_column = _get_first_existing_column(original_lines, ["b1", "b1_s"])
    b2_column = _get_first_existing_column(original_lines, ["b2", "b2_s"])

    bus1_column = _get_first_existing_column(
        original_lines, ["bus_breaker_bus1_id", "bus1_bus_id", "bus1_name"]
    )

    bus2_column = _get_first_existing_column(
        original_lines, ["bus_breaker_bus2_id", "bus2_bus_id", "bus2_name"]
    )

    frequency_hz = _pandapower_frequency_hz(pandapower_network)

    for line_id, pyp_line in original_lines.iterrows():
        matching_pp_line = pp_lines[pp_lines["name"].astype(str) == str(line_id)]

        matching_pp_impedance = pd.DataFrame()

        if not pp_line_impedances.empty:
            if "powsybl_id" in pp_line_impedances.columns:
                matching_pp_impedance = pp_line_impedances[
                    pp_line_impedances["powsybl_id"].astype(str) == str(line_id)
                ]

            if matching_pp_impedance.empty and "name" in pp_line_impedances.columns:
                matching_pp_impedance = pp_line_impedances[
                    pp_line_impedances["name"].astype(str) == str(line_id)
                ]

        assert not matching_pp_line.empty or not matching_pp_impedance.empty, (
            f"Line {line_id!r} exists in powsybl but not in pandapower line or impedance.\n"
            f"Pandapower lines: {list(pp_lines['name'].astype(str)) if 'name' in pp_lines.columns else []}\n"
            f"Pandapower impedances: "
            f"{list(pp_line_impedances['powsybl_id'].astype(str)) if 'powsybl_id' in pp_line_impedances.columns else []}"
        )

        if not matching_pp_line.empty:
            pp_line = matching_pp_line.iloc[0]

            assert int(pp_line["from_bus"]) in pandapower_network.bus.index, (
                f"Line {line_id!r} has invalid from_bus: {pp_line['from_bus']}"
            )

            assert int(pp_line["to_bus"]) in pandapower_network.bus.index, (
                f"Line {line_id!r} has invalid to_bus: {pp_line['to_bus']}"
            )

            length_km = float(pp_line["length_km"])

            _assert_finite_positive(
                length_km, f"Line {line_id!r} has invalid length_km: {length_km}"
            )

            parallel = 1.0

            if "parallel" in pp_lines.columns:
                parallel = float(pp_line["parallel"])

            if not np.isfinite(parallel) or parallel <= 0.0:
                parallel = 1.0

            actual_r_ohm = float(pp_line["r_ohm_per_km"]) * length_km / parallel
            actual_x_ohm = float(pp_line["x_ohm_per_km"]) * length_km / parallel

            expected_r_ohm = float(pyp_line[r_column])
            expected_x_ohm = float(pyp_line[x_column])

            _assert_close_value(
                actual_r_ohm,
                expected_r_ohm,
                f"Line {line_id!r} resistance r",
                rtol=1e-6,
                atol=1e-9,
            )

            _assert_close_value(
                actual_x_ohm,
                expected_x_ohm,
                f"Line {line_id!r} reactance x",
                rtol=1e-6,
                atol=1e-9,
            )

            if (
                g1_column is not None
                and g2_column is not None
                and "g_us_per_km" in pp_lines.columns
            ):
                expected_g_s = float(pyp_line[g1_column]) + float(pyp_line[g2_column])
                actual_g_s = float(pp_line["g_us_per_km"]) * 1e-6 * length_km * parallel

                _assert_close_value(
                    actual_g_s,
                    expected_g_s,
                    f"Line {line_id!r} conductance g1 + g2",
                    rtol=1e-5,
                    atol=1e-12,
                )

            if (
                b1_column is not None
                and b2_column is not None
                and "c_nf_per_km" in pp_lines.columns
            ):
                expected_b_s = float(pyp_line[b1_column]) + float(pyp_line[b2_column])
                actual_b_s = (
                    2.0
                    * np.pi
                    * frequency_hz
                    * float(pp_line["c_nf_per_km"])
                    * 1e-9
                    * length_km
                    * parallel
                )

                _assert_close_value(
                    actual_b_s,
                    expected_b_s,
                    f"Line {line_id!r} susceptance b1 + b2",
                    rtol=1e-5,
                    atol=1e-12,
                )

            if bus1_column is not None and bus2_column is not None:
                expected_buses = {
                    str(pyp_line[bus1_column]),
                    str(pyp_line[bus2_column]),
                }

                actual_buses = {
                    _pandapower_bus_name(pandapower_network, pp_line["from_bus"]),
                    _pandapower_bus_name(pandapower_network, pp_line["to_bus"]),
                }

                assert actual_buses == expected_buses, (
                    f"Line {line_id!r} terminal buses changed.\n"
                    f"powsybl = {expected_buses}\n"
                    f"pandapower = {actual_buses}"
                )
        else:
            pp_impedance = matching_pp_impedance.iloc[0]

            expected_r_ohm = float(pyp_line[r_column])
            expected_x_ohm = float(pyp_line[x_column])

            from_bus = int(pp_impedance["from_bus"])
            to_bus = int(pp_impedance["to_bus"])

            from_vn_kv = float(pandapower_network.bus.at[from_bus, "vn_kv"])
            to_vn_kv = float(pandapower_network.bus.at[to_bus, "vn_kv"])

            sn_mva = float(pp_impedance["sn_mva"])

            from_z_base_ohm = (from_vn_kv**2) / sn_mva
            to_z_base_ohm = (to_vn_kv**2) / sn_mva

            actual_r_from_ohm = float(pp_impedance["rft_pu"]) * from_z_base_ohm
            actual_x_from_ohm = float(pp_impedance["xft_pu"]) * from_z_base_ohm

            actual_r_to_ohm = float(pp_impedance["rtf_pu"]) * to_z_base_ohm
            actual_x_to_ohm = float(pp_impedance["xtf_pu"]) * to_z_base_ohm

            _assert_close_value(
                actual_r_from_ohm,
                expected_r_ohm,
                f"Impedance-line {line_id!r} from-side resistance r",
                rtol=1e-6,
                atol=1e-9,
            )

            _assert_close_value(
                actual_x_from_ohm,
                expected_x_ohm,
                f"Impedance-line {line_id!r} from-side reactance x",
                rtol=1e-6,
                atol=1e-9,
            )

            _assert_close_value(
                actual_r_to_ohm,
                expected_r_ohm,
                f"Impedance-line {line_id!r} to-side resistance r",
                rtol=1e-6,
                atol=1e-9,
            )

            _assert_close_value(
                actual_x_to_ohm,
                expected_x_ohm,
                f"Impedance-line {line_id!r} to-side reactance x",
                rtol=1e-6,
                atol=1e-9,
            )

            if (
                g1_column is not None
                and g2_column is not None
                and "powsybl_g_total_s" in pp_impedance.index
            ):
                expected_g_s = float(pyp_line[g1_column]) + float(pyp_line[g2_column])
                actual_g_s = float(pp_impedance["powsybl_g_total_s"])

                _assert_close_value(
                    actual_g_s,
                    expected_g_s,
                    f"Impedance-line {line_id!r} conductance g1 + g2",
                    rtol=1e-5,
                    atol=1e-12,
                )

            if (
                b1_column is not None
                and b2_column is not None
                and "powsybl_b_total_s" in pp_impedance.index
            ):
                expected_b_s = float(pyp_line[b1_column]) + float(pyp_line[b2_column])
                actual_b_s = float(pp_impedance["powsybl_b_total_s"])

                _assert_close_value(
                    actual_b_s,
                    expected_b_s,
                    f"Impedance-line {line_id!r} susceptance b1 + b2",
                    rtol=1e-5,
                    atol=1e-12,
                )

            if bus1_column is not None and bus2_column is not None:
                expected_buses = {
                    str(pyp_line[bus1_column]),
                    str(pyp_line[bus2_column]),
                }

                actual_buses = {
                    _pandapower_bus_name(pandapower_network, from_bus),
                    _pandapower_bus_name(pandapower_network, to_bus),
                }

                assert actual_buses == expected_buses, (
                    f"Impedance-line {line_id!r} terminal buses changed.\n"
                    f"powsybl = {expected_buses}\n"
                    f"pandapower = {actual_buses}"
                )


def _assert_finite_positive(value: Any, message: str) -> None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(message) from exc

    assert np.isfinite(numeric_value) and numeric_value > 0.0, message


def _to_test_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return default
        return bool(value)

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y", "closed"}:
        return True

    if text in {"false", "0", "no", "n", "open"}:
        return False

    return default


def _get_table_or_empty(network: Any, getter_name: str) -> pd.DataFrame:
    getter = getattr(network, getter_name, None)

    if getter is None:
        return pd.DataFrame()

    table = getter(all_attributes=True)

    if table is None:
        return pd.DataFrame()

    return table.copy()


def _is_valid_positive_number(value: Any) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False

    return np.isfinite(value) and value > 0.0


def _is_valid_finite_number(value: Any) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False

    return np.isfinite(value)


def _assert_converted_2w_transformers_match_powsybl(
    original_network: Any, pandapower_network: Any
) -> None:
    """Check converted pandapower two-windings transformers against powsybl data."""
    original_trafos = _get_table_or_empty(
        original_network, "get_2_windings_transformers"
    )

    pp_trafos = pandapower_network.trafo.copy()

    assert len(original_trafos) == len(pp_trafos), (
        "Number of 2-winding transformers changed during conversion: "
        f"powsybl = {len(original_trafos)}, pandapower = {len(pp_trafos)}"
    )

    if original_trafos.empty:
        return

    for trafo_id, pyp_trafo in original_trafos.iterrows():
        matching_pp_trafo = pp_trafos[pp_trafos["name"].astype(str) == str(trafo_id)]

        assert not matching_pp_trafo.empty, (
            f"2W transformer {trafo_id!r} exists in powsybl but not in pandapower. "
            f"Pandapower transformers: {list(pp_trafos['name'].astype(str))}"
        )

        pp_trafo = matching_pp_trafo.iloc[0]

        assert int(pp_trafo["hv_bus"]) in pandapower_network.bus.index, (
            f"2W transformer {trafo_id!r} has invalid hv_bus: {pp_trafo['hv_bus']}"
        )

        assert int(pp_trafo["lv_bus"]) in pandapower_network.bus.index, (
            f"2W transformer {trafo_id!r} has invalid lv_bus: {pp_trafo['lv_bus']}"
        )

        _assert_finite_positive(
            pp_trafo["sn_mva"],
            f"2W transformer {trafo_id!r} has invalid sn_mva: {pp_trafo['sn_mva']}",
        )

        _assert_finite_positive(
            pp_trafo["vn_hv_kv"],
            f"2W transformer {trafo_id!r} has invalid vn_kv_hv: {pp_trafo['vn_hv_kv']}",
        )

        _assert_finite_positive(
            pp_trafo["vn_lv_kv"],
            f"2W transformer {trafo_id!r} has invalid vn_lv_kv: {pp_trafo['vn_lv_kv']}",
        )

        for column in [
            "vk_percent",
            "vkr_percent",
            "pfe_kw",
            "i0_percent",
            "shift_degree",
        ]:
            assert np.isfinite(float(pp_trafo[column])), (
                f"2W transformer {trafo_id!r} has invalid {column}:{pp_trafo[column]}"
            )

        rated_u1_column = _get_first_existing_column(
            original_trafos, ["rated_u1", "ratedU1"]
        )

        rated_u2_column = _get_first_existing_column(
            original_trafos, ["rated_u2", "ratedU2"]
        )

        if rated_u1_column is not None and rated_u2_column is not None:
            expected_voltages = sorted(
                [float(pyp_trafo[rated_u1_column]), float(pyp_trafo[rated_u2_column])],
                reverse=True,
            )

            actual_voltages = [float(pp_trafo["vn_hv_kv"]), float(pp_trafo["vn_lv_kv"])]

            np.testing.assert_allclose(
                actual_voltages,
                expected_voltages,
                rtol=1e-6,
                atol=1e-9,
                err_msg=f"2W transformer {trafo_id!r} voltage levels differ",
            )

        rated_s_column = _get_first_existing_column(
            original_trafos, ["rated_s", "ratedS"]
        )

        if rated_s_column is not None:
            expected_sn_mva = float(pyp_trafo[rated_s_column])

            if np.isfinite(expected_sn_mva) and expected_sn_mva > 0.0:
                _assert_close_value(
                    float(pp_trafo["sn_mva"]),
                    expected_sn_mva,
                    f"2W transformer {trafo_id!r} rated power sn_mva",
                    rtol=1e-6,
                    atol=1e-9,
                )

        r_column = _get_first_existing_column(original_trafos, ["r", "r_ohm"])
        x_column = _get_first_existing_column(original_trafos, ["x", "x_ohm"])
        g_column = _get_first_existing_column(original_trafos, ["g", "g_s"])
        b_column = _get_first_existing_column(original_trafos, ["b", "b_s"])

        if (
            r_column is not None
            and x_column is not None
            and rated_s_column is not None
            and rated_u2_column is not None
            and _is_valid_positive_number(pyp_trafo[rated_s_column])
            and _is_valid_positive_number(pyp_trafo[rated_u2_column])
            and _is_valid_finite_number(pyp_trafo[r_column])
            and _is_valid_finite_number(pyp_trafo[x_column])
        ):
            rated_s_mva = float(pyp_trafo[rated_s_column])
            rated_u2_kv = float(pyp_trafo[rated_u2_column])

            z_base_ohm = rated_u2_kv**2 / rated_s_mva

            expected_vkr_percent = 100.0 * float(pyp_trafo[r_column]) / z_base_ohm
            expected_vk_percent = (
                100.0 * np.hypot(float(pyp_trafo[r_column]), float(pyp_trafo[x_column]))
            ) / z_base_ohm

            _assert_close_value(
                float(pp_trafo["vkr_percent"]),
                expected_vkr_percent,
                f"2W transformer {trafo_id!r} vkr_percent",
                rtol=1e-5,
                atol=1e-7,
            )

            _assert_close_value(
                float(pp_trafo["vk_percent"]),
                expected_vk_percent,
                f"2W transformer {trafo_id!r} vk_percent",
                rtol=1e-5,
                atol=1e-7,
            )

        if (
            g_column is not None
            and b_column is not None
            and rated_s_column is not None
            and rated_u2_column is not None
            and _is_valid_positive_number(pyp_trafo[rated_s_column])
            and _is_valid_positive_number(pyp_trafo[rated_u2_column])
            and _is_valid_finite_number(pyp_trafo[g_column])
            and _is_valid_finite_number(pyp_trafo[b_column])
        ):
            rated_s_mva = float(pyp_trafo[rated_s_column])
            rated_u2_kv = float(pyp_trafo[rated_u2_column])

            expected_pfe_kw = rated_u2_kv**2 * float(pyp_trafo[g_column]) * 1000.0

            y_base_s = rated_s_mva / rated_u2_kv**2
            y_no_load_pu = (
                complex(float(pyp_trafo[g_column]), float(pyp_trafo[b_column]))
                / y_base_s
            )

            expected_i0_percent = 100.0 * abs(y_no_load_pu)

            _assert_close_value(
                float(pp_trafo["pfe_kw"]),
                expected_pfe_kw,
                f"2W transformer {trafo_id!r} pfe_kw",
                rtol=1e-5,
                atol=1e-6,
            )

            _assert_close_value(
                float(pp_trafo["i0_percent"]),
                expected_i0_percent,
                f"2W transformer {trafo_id!r} i0_percent",
                rtol=1e-5,
                atol=1e-7,
            )


def _assert_converted_3w_transformers_match_powsybl(
    original_network: Any, pandapower_network: Any
) -> None:
    """Check converted pandapower three-winding transformers against powsybl data."""
    original_trafos = _get_table_or_empty(
        original_network, "get_3_windings_transformers"
    )

    pp_trafos = pandapower_network.trafo3w.copy()

    assert len(original_trafos) == len(pp_trafos), (
        "Number of 3-windings transformers changed during conversion: "
        f"powsybl = {len(original_trafos)}, pandapower = {len(pp_trafos)}"
    )

    if original_trafos.empty:
        return

    for trafo_id, pyp_trafo in original_trafos.iterrows():
        matching_pp_trafo = pp_trafos[pp_trafos["name"].astype(str) == str(trafo_id)]

        assert not matching_pp_trafo.empty, (
            f"3W transformer {trafo_id!r} exists in powsybl but not in pandapower. "
            f"Pandapower 3W transformers: {list(pp_trafos['name'].astype(str))}"
        )

        pp_trafo = matching_pp_trafo.iloc[0]

        for bus_column in ["hv_bus", "mv_bus", "lv_bus"]:
            assert int(pp_trafo[bus_column]) in pandapower_network.bus.index, (
                f"3W transformer {trafo_id!r} has invalid {bus_column}: "
                f"{pp_trafo[bus_column]}"
            )

        for column in ["sn_hv_mva", "sn_mv_mva", "sn_lv_mva"]:
            _assert_finite_positive(
                pp_trafo[column],
                f"3W transformer {trafo_id!r} has invalid {column}: {pp_trafo[column]}",
            )

        for column in ["vn_hv_kv", "vn_mv_kv", "vn_lv_kv"]:
            _assert_finite_positive(
                pp_trafo[column],
                f"3W transformer {trafo_id!r} has invalid {column}: {pp_trafo[column]}",
            )

        for column in [
            "vk_hv_percent",
            "vk_mv_percent",
            "vk_lv_percent",
            "vkr_hv_percent",
            "vkr_mv_percent",
            "vkr_lv_percent",
            "pfe_kw",
            "i0_percent",
            "shift_mv_degree",
            "shift_lv_degree",
        ]:
            assert np.isfinite(float(pp_trafo[column])), (
                f"3W transformer {trafo_id!r} has invalid {column}: {pp_trafo[column]}"
            )

        rated_u_columns: list[Any] = [
            _get_first_existing_column(original_trafos, ["rated_u1", "ratedU1"]),
            _get_first_existing_column(original_trafos, ["rated_u2", "ratedU2"]),
            _get_first_existing_column(original_trafos, ["rated_u3", "ratedU3"]),
        ]

        if all(column is not None for column in rated_u_columns):
            expected_voltages = sorted(
                [float(pyp_trafo[column]) for column in rated_u_columns], reverse=True
            )

            actual_voltages = [
                float(pp_trafo["vn_hv_kv"]),
                float(pp_trafo["vn_mv_kv"]),
                float(pp_trafo["vn_lv_kv"]),
            ]

            np.testing.assert_allclose(
                actual_voltages,
                expected_voltages,
                rtol=1e-6,
                atol=1e-9,
                err_msg=f"3W transformer {trafo_id!r} voltage levels differ",
            )

        rated_s_columns: list[Any] = [
            _get_first_existing_column(original_trafos, ["rated_s1", "ratedS1"]),
            _get_first_existing_column(original_trafos, ["rated_s2", "ratedS2"]),
            _get_first_existing_column(original_trafos, ["rated_s3", "ratedS3"]),
        ]

        if all(column is not None for column in rated_u_columns + rated_s_columns):
            powsybl_sides = [
                (
                    float(pyp_trafo[rated_u_columns[0]]),
                    float(pyp_trafo[rated_s_columns[0]]),
                ),
                (
                    float(pyp_trafo[rated_u_columns[1]]),
                    float(pyp_trafo[rated_s_columns[1]]),
                ),
                (
                    float(pyp_trafo[rated_u_columns[2]]),
                    float(pyp_trafo[rated_s_columns[2]]),
                ),
            ]

            powsybl_sides = sorted(
                powsybl_sides, key=lambda item: item[0], reverse=True
            )

            expected_sn_values = [
                powsybl_sides[0][1],
                powsybl_sides[1][1],
                powsybl_sides[2][1],
            ]

            actual_sn_values = [
                float(pp_trafo["sn_hv_mva"]),
                float(pp_trafo["sn_mv_mva"]),
                float(pp_trafo["sn_lv_mva"]),
            ]

            np.testing.assert_allclose(
                actual_sn_values,
                expected_sn_values,
                rtol=1e-6,
                atol=1e-9,
                err_msg=f"3W transformers {trafo_id!r} rated powers differ",
            )


def _assert_converted_switches_match_powsybl(
    original_network: Any, pandapower_network: Any
) -> None:
    """Check converted pandapower bus-bus switches against powsybl switches."""
    original_switches = _get_table_or_empty(original_network, "get_switches")

    pp_switches = pandapower_network.switch.copy()

    assert len(original_switches) == len(pp_switches), (
        "Number of switches changed during conversion: "
        f"powsybl = {len(original_switches)}, pandapower = {len(pp_switches)}"
    )

    if original_switches.empty:
        return

    for switch_index, pp_switch in pp_switches.iterrows():
        assert int(pp_switch["bus"]) in pandapower_network.bus.index, (
            f"Switch {switch_index} is connected to an invalid bus: {pp_switch['bus']}"
        )

        assert pp_switch["et"] == "b", (
            f"Switch {switch_index} has invalid element type et = {pp_switch['et']!r}"
        )

        assert int(pp_switch["element"]) in pandapower_network.bus.index, (
            f"Bus-bus switch {switch_index} points to invalid bus element: "
            f"{pp_switch['element']}"
        )

        assert isinstance(pp_switch["closed"], (bool, np.bool_)), (
            f"Switch {switch_index} has invalid closed value: {pp_switch['closed']}"
        )

    if "name" in pp_switches.columns:
        pp_switch_names = set(pp_switches["name"].astype(str))
        original_switch_names = set(original_switches.index.astype(str))

        assert pp_switch_names == original_switch_names, (
            "Switch names changed during conversion.\n"
            f"powsybl switches = {sorted(original_switch_names)}\n"
            f"pandapower switches = {sorted(pp_switch_names)}"
        )

    if "open" in original_switches.columns and "name" in pp_switches.columns:
        for switch_id, pyp_switch in original_switches.iterrows():
            matching_pp_switch = pp_switches[
                pp_switches["name"].astype(str) == str(switch_id)
            ]

            assert not matching_pp_switch.empty, (
                f"Switch {switch_id!r} exists in powsybl but not in pandapower"
            )

            expected_closed = not _to_test_bool(pyp_switch["open"], default=False)
            actual_closed = bool(matching_pp_switch.iloc[0]["closed"])

            assert actual_closed == expected_closed, (
                f"Switch {switch_id!r} open/closed state mismatch: "
                f"powsybl open = {pyp_switch['open']}, "
                f"pandapower closed = {actual_closed}"
            )


def _normalize_table_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = result.index.astype(str)
    return result.sort_index()


def _assert_roundtrip_numeric_columns_close(
    original_table: pd.DataFrame,
    roundtrip_table: pd.DataFrame,
    columns: list[str],
    component_name: str,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> None:
    """Compare selected numeric columns between original and roundtrip powsybl tables."""
    if original_table.empty and roundtrip_table.empty:
        return

    original_table, roundtrip_table = _align_roundtrip_table_by_original_ids(
        original_table, roundtrip_table, component_name
    )

    missing_columns = [
        column
        for column in columns
        if column in original_table.columns and column not in roundtrip_table.columns
    ]

    assert not missing_columns, (
        f"{component_name} columns disappeared during roundtrip: {missing_columns}\n"
        f"Original columns: {list(original_table.columns)}\n"
        f"Roundtrip columns: {list(roundtrip_table.columns)}"
    )

    compared_columns = [
        column
        for column in columns
        if column in original_table.columns and column in roundtrip_table.columns
    ]

    if not compared_columns:
        return

    for column in compared_columns:
        original_values = pd.to_numeric(
            original_table[column], errors="coerce"
        ).to_numpy(dtype=float)

        roundtrip_values = pd.to_numeric(
            roundtrip_table[column], errors="coerce"
        ).to_numpy(dtype=float)

        comparable_mask = np.isfinite(original_values)

        if not comparable_mask.any():
            continue

        np.testing.assert_allclose(
            roundtrip_values[comparable_mask],
            original_values[comparable_mask],
            rtol=rtol,
            atol=atol,
            err_msg=(
                f"{component_name} column {column!r} differs after "
                f"powsybl -> pandapower -> powsybl roundtrip"
            ),
        )


def _assert_roundtrip_bool_columns_equal(
    original_table: pd.DataFrame,
    roundtrip_table: pd.DataFrame,
    columns: list[str],
    component_name: str,
) -> None:
    """Compare selected boolean state columns between original and roundtrip tables."""
    if original_table.empty and roundtrip_table.empty:
        return

    original_table, roundtrip_table = _align_roundtrip_table_by_original_ids(
        original_table, roundtrip_table, component_name
    )

    for column in columns:
        if column not in original_table.columns:
            continue

        assert column in roundtrip_table.columns, (
            f"{component_name} boolean column {column!r} disappeared during roundtrip.\n"
            f"Original columns: {list(original_table.columns)}\n"
            f"Roundtrip columns: {list(roundtrip_table.columns)}"
        )

        original_values = original_table[column].map(_to_test_bool)
        roundtrip_values = roundtrip_table[column].map(_to_test_bool)

        assert original_values.equals(roundtrip_values), (
            f"{component_name} boolean column {column!r} differs after roundtrip.\n"
            f"Original:\n{original_values}\n"
            f"Roundtrip:\n{roundtrip_values}"
        )


def _assert_roundtrip_loads_match_powsybl(
    original_network: Any, roundtrip_network: Any
) -> None:
    original_loads = _get_table_or_empty(original_network, "get_loads")

    roundtrip_loads = _get_table_or_empty(roundtrip_network, "get_loads")

    _assert_roundtrip_numeric_columns_close(
        original_loads,
        roundtrip_loads,
        columns=["p0", "q0", "target_p", "target_q"],
        component_name="loads",
        rtol=1e-6,
        atol=1e-9,
    )


def _assert_roundtrip_generators_match_powsybl(
    original_network: Any, roundtrip_network: Any
) -> None:
    original_generators = _get_table_or_empty(original_network, "get_generators")

    roundtrip_generators = _get_table_or_empty(roundtrip_network, "get_generators")

    _assert_roundtrip_numeric_columns_close(
        original_generators,
        roundtrip_generators,
        columns=["target_v", "rated_s"],
        component_name="generators",
        rtol=1e-5,
        atol=1e-8,
    )

    _assert_roundtrip_bool_columns_equal(
        original_generators,
        roundtrip_generators,
        columns=["voltage_regulator_on", "connected"],
        component_name="generators",
    )


def _assert_roundtrip_lines_match_powsybl(
    original_network: Any, roundtrip_network: Any, pandapower_network: Any
) -> None:
    original_lines = original_network.get_lines(all_attributes=True)
    roundtrip_lines = roundtrip_network.get_lines(all_attributes=True)

    cross_voltage_line_ids = _get_cross_voltage_line_ids_from_pandapower(
        pandapower_network
    )

    if cross_voltage_line_ids:
        original_lines = original_lines[
            ~original_lines.index.astype(str).isin(cross_voltage_line_ids)
        ].copy()

    assert len(original_lines) == len(roundtrip_lines), (
        f"Number of comparable lines changed during roundtrip:\n"
        f"Comparable original lines = {len(original_lines)}\n"
        f"Roundtrip lines = {len(roundtrip_lines)}\n"
        f"Excluded cross-voltage lines = {sorted(cross_voltage_line_ids)}"
    )

    if original_lines.empty:
        return

    _assert_roundtrip_numeric_columns_close(
        original_lines,
        roundtrip_lines,
        columns=["r", "x"],
        component_name="lines",
        rtol=1e-6,
        atol=1e-9,
    )

    _assert_roundtrip_line_shunt_sums_match(original_lines, roundtrip_lines)

    _assert_roundtrip_bool_columns_equal(
        original_lines,
        roundtrip_lines,
        columns=["connected1", "connected2"],
        component_name="lines",
    )


def _assert_roundtrip_2w_transformers_match_powsybl(
    original_network: Any, roundtrip_network: Any
) -> None:
    original_trafos = _get_table_or_empty(
        original_network, "get_2_windings_transformers"
    )

    roundtrip_trafos = _get_table_or_empty(
        roundtrip_network, "get_2_windings_transformers"
    )

    _assert_roundtrip_numeric_columns_close(
        original_trafos,
        roundtrip_trafos,
        columns=["r", "x", "g", "b", "rated_u1", "rated_u2", "rho", "alpha"],
        component_name="2W transformers",
        rtol=1e-5,
        atol=1e-8,
    )

    _assert_roundtrip_bool_columns_equal(
        original_trafos,
        roundtrip_trafos,
        columns=["connected1", "connected2"],
        component_name="2W transformers",
    )


def _get_original_3w_transformer_ids(original_network: Any) -> set[str]:
    original_trafos = _get_table_or_empty(
        original_network, "get_3_windings_transformers"
    )

    if original_trafos.empty:
        return set()

    return set(original_trafos.index.astype(str))


def _assert_roundtrip_3w_transformers_match_powsybl(
    original_network: Any, roundtrip_network: Any
) -> None:
    original_trafos = _get_table_or_empty(
        original_network, "get_3_windings_transformers"
    )

    roundtrip_trafos = _get_table_or_empty(
        roundtrip_network, "get_3_windings_transformers"
    )

    if original_trafos.empty and roundtrip_trafos.empty:
        return

    if not original_trafos.empty and roundtrip_trafos.empty:
        logger.info(
            "3W transformers roundtrip comparison is not executed because "
            "pypowsybl.convert_from_pandapower does not recreate pandapower "
            "three-windings transformers as powsybl 3W transformers."
        )
        return

    _assert_roundtrip_numeric_columns_close(
        original_trafos,
        roundtrip_trafos,
        columns=[
            "rated_u1",
            "rated_u2",
            "rated_u3",
            "rated_s1",
            "rated_s2",
            "rated_s3",
            "r1",
            "r2",
            "r3",
            "x1",
            "x2",
            "x3",
            "g1",
            "g2",
            "g3",
            "b1",
            "b2",
            "b3",
        ],
        component_name="3W transformers",
        rtol=1e-5,
        atol=1e-8,
    )

    _assert_roundtrip_bool_columns_equal(
        original_trafos,
        roundtrip_trafos,
        columns=["connected1", "connected2", "connected3"],
        component_name="3W transformers",
    )


def _assert_roundtrip_switches_match_powsybl(
    original_network: Any, roundtrip_network: Any
) -> None:
    original_switches = _get_table_or_empty(original_network, "get_switches")

    roundtrip_switches = _get_table_or_empty(roundtrip_network, "get_switches")

    if original_switches.empty and roundtrip_switches.empty:
        return

    if not original_switches.empty and roundtrip_switches.empty:
        logger.info(
            "Switch roundtrip comparison is not executed because "
            "pypowsybl.convert_from_pandapower does not recreate pandapower "
            "bus-bus switches as powsybl switches."
        )
        return

    _assert_roundtrip_bool_columns_equal(
        original_switches,
        roundtrip_switches,
        columns=["open"],
        component_name="switches",
    )


def _assert_roundtrip_static_values_match_powsybl(
    original_network: Any, roundtrip_network: Any, pandapower_network: Any
) -> None:
    _assert_roundtrip_loads_match_powsybl(original_network, roundtrip_network)

    _assert_roundtrip_generators_match_powsybl(original_network, roundtrip_network)

    _assert_roundtrip_lines_match_powsybl(
        original_network, roundtrip_network, pandapower_network
    )

    _assert_roundtrip_2w_transformers_match_powsybl(original_network, roundtrip_network)

    _assert_roundtrip_3w_transformers_match_powsybl(original_network, roundtrip_network)

    _assert_roundtrip_switches_match_powsybl(original_network, roundtrip_network)


def _run_conversion_roundtrip_cycle(
    case_name: str, original_network: Any, tmp_path: Path, pyp_lf: Any, pandapower: Any
) -> None:
    """Run the full powsybl -> pandapower -> powsybl conversion test cycle."""
    _powsybl_loadflow_must_converge(
        original_network, pyp_lf, label=f"original {case_name}"
    )

    xiidm_path = _save_network_as_xiidm(
        original_network, tmp_path / f"{case_name}.xiidm"
    )

    converter = PyPowSyBlConverter()
    pandapower_network, _, json_path, *_ = converter._powsybl_to_pandapower(
        filename=str(xiidm_path),
        log_static_comparison=False,
        log_loadflow_comparison=False,
    )

    assert Path(json_path).exists()
    assert len(pandapower_network.bus) > 0

    pandapower.runpp(
        pandapower_network,
        algorithm="nr",
        calculate_voltage_angles=True,
        init="dc",
        max_iteration=50,
        tolerance_mva=1e-7,
    )

    assert bool(pandapower_network.converged)

    _assert_same_pandapower_bus_loadflow_results(original_network, pandapower_network)

    _assert_converted_loads_match_powsybl(original_network, pandapower_network)

    _assert_converted_generators_match_powsybl(original_network, pandapower_network)

    _assert_converted_lines_match_powsybl(original_network, pandapower_network)

    _assert_converted_2w_transformers_match_powsybl(
        original_network, pandapower_network
    )

    _assert_converted_3w_transformers_match_powsybl(
        original_network, pandapower_network
    )

    _assert_converted_switches_match_powsybl(original_network, pandapower_network)

    original_switch_ids = _get_original_switch_ids(original_network)

    if original_switch_ids:
        logger.info(
            "Exact powsybl roundtrip load-flow is not comparable because "
            "the original network contains powsybl bus-breaker switches. "
            "pypowsybl.convert_from_pandapower may collapse closed bus-bus switches "
            "and does not necessarily recreate them as powsybl switches. "
            "Skipped final roundtrip bus comparison for switches: %s",
            sorted(original_switch_ids),
        )
        return

    original_3w_trafo_ids = _get_original_3w_transformer_ids(original_network)

    if original_3w_trafo_ids:
        logger.info(
            "Exact powsybl roundtrip load-flow is not comparable because "
            "the original network contains powsybl three-windings transformers. "
            "pypowsybl.convert_from_pandapower does not necessarily recreate "
            "pandapower trafo3w elements as powsybl 3W transformers. "
            "Skipped final roundtrip bus comparison for 3W transformers: %s",
            sorted(original_3w_trafo_ids),
        )
        return

    roundtrip_network, roundtrip_xiidm_path = _pandapower_to_powsybl(
        pandapower_network, tmp_path
    )

    assert roundtrip_xiidm_path.exists()

    _assert_basic_network_size_is_preserved(
        original_network, roundtrip_network, pandapower_network
    )

    _assert_roundtrip_static_values_match_powsybl(
        original_network, roundtrip_network, pandapower_network
    )

    cross_voltage_line_ids = _get_cross_voltage_line_ids_from_pandapower(
        pandapower_network
    )

    if cross_voltage_line_ids:
        logger.info(
            "Exact powsybl roundtrip load-flow is not comparable because "
            "cross-voltage powsybl lines were converted to pandapower.impedance "
            "and powsybl.convert_from_pandapower does not recreate them as powsybl lines. "
            "Excluded lines: %s",
            sorted(cross_voltage_line_ids),
        )
        return

    _powsybl_loadflow_must_converge(
        roundtrip_network, pyp_lf, label=f"roundtrip {case_name}"
    )

    _assert_same_bus_loadflow_results(
        original_network, roundtrip_network, pandapower_network
    )


@pytest.mark.parametrize(
    "case_name, network_factory",
    PYP_NETWORK_CASES,
    ids=[case_name for case_name, _ in PYP_NETWORK_CASES],
)
def test_network_conversion_and_roundtrip_static_checks(
    case_name, network_factory, tmp_path, pyp, pyp_lf, pandapower
) -> None:
    original_network = network_factory(pyp)

    _run_conversion_roundtrip_cycle(
        case_name=case_name,
        original_network=original_network,
        tmp_path=tmp_path,
        pyp_lf=pyp_lf,
        pandapower=pandapower,
    )


def _assert_same_pandapower_network_results(
    original_pp_net: Any, roundtrip_pp_net: Any
) -> None:
    """Compare pandapower results tables before and after a powsybl roundtrip."""
    assert len(original_pp_net.bus) == len(roundtrip_pp_net.bus), (
        f"Number of buses changed: "
        f"original = {len(original_pp_net.bus)}, "
        f"roundtrip = {len(roundtrip_pp_net.bus)}"
    )

    assert len(original_pp_net.load) == len(roundtrip_pp_net.load), (
        f"Number of loads changed: "
        f"original = {len(original_pp_net.load)}, "
        f"roundtrip = {len(roundtrip_pp_net.load)}"
    )

    assert len(original_pp_net.gen) == len(roundtrip_pp_net.gen), (
        f"Number of generators changed: "
        f"original = {len(original_pp_net.gen)}, "
        f"roundtrip = {len(roundtrip_pp_net.gen)}"
    )

    assert len(original_pp_net.line) == len(roundtrip_pp_net.line), (
        f"Number of lines changed: "
        f"original = {len(original_pp_net.line)}, "
        f"roundtrip = {len(roundtrip_pp_net.line)}"
    )

    assert len(original_pp_net.trafo) == len(roundtrip_pp_net.trafo), (
        f"Number of 2W transformers changed: "
        f"original = {len(original_pp_net.trafo)}, "
        f"roundtrip = {len(roundtrip_pp_net.trafo)}"
    )

    original_v_mag = original_pp_net.res_bus["vm_pu"].to_numpy(
        dtype=float
    ) * original_pp_net.bus["vn_kv"].to_numpy(dtype=float)

    roundtrip_v_mag = roundtrip_pp_net.res_bus["vm_pu"].to_numpy(
        dtype=float
    ) * roundtrip_pp_net.bus["vn_kv"].to_numpy(dtype=float)

    np.testing.assert_allclose(
        roundtrip_v_mag,
        original_v_mag,
        rtol=5e-2,
        atol=1e-1,
        err_msg=(
            "Pandapower bus voltage magnitudes differ after "
            "pandapower -> powsybl -> pandapower roundtrip"
        ),
    )

    original_angles = (
        original_pp_net.res_bus["va_degree"].to_numpy(dtype=float)
        - original_pp_net.res_bus["va_degree"].to_numpy(dtype=float)[0]
    )

    roundtrip_angles = (
        roundtrip_pp_net.res_bus["va_degree"].to_numpy(dtype=float)
        - roundtrip_pp_net.res_bus["va_degree"].to_numpy(dtype=float)[0]
    )

    np.testing.assert_allclose(
        roundtrip_angles,
        original_angles,
        rtol=5e-2,
        atol=1.0,
        err_msg=(
            "Pandapower bus voltage angles differ after "
            "pandapower -> powsybl -> pandapower roundtrip"
        ),
    )


def _run_conversion_pandapower_cycle(
    case_name: str, original_pp_net: Any, tmp_path: Path, pyp_lf: Any, pandapower: Any
) -> None:
    """Run the full pandapower -> powsybl -> pandapower conversion test cycle."""
    pandapower.runpp(
        original_pp_net,
        algorithm="nr",
        calculate_voltage_angles=True,
        init="dc",
        max_iteration=50,
        tolerance_mva=1e-7,
    )

    assert bool(original_pp_net.converged)

    pyp_network = convert_from_pandapower(original_pp_net)

    _powsybl_loadflow_must_converge(
        pyp_network, pyp_lf, label=f"powsybl converted from pandapower {case_name}"
    )

    xiidm_path = _save_network_as_xiidm(
        pyp_network, tmp_path / f"{case_name}_from_pandapower.xiidm"
    )

    converter = PyPowSyBlConverter()

    roundtrip_pp_net, _, json_path, *_ = converter._powsybl_to_pandapower(
        filename=str(xiidm_path),
        log_static_comparison=False,
        log_loadflow_comparison=False,
    )

    assert Path(json_path).exists()
    assert len(roundtrip_pp_net.bus) > 0

    pandapower.runpp(
        roundtrip_pp_net,
        algorithm="nr",
        calculate_voltage_angles=True,
        init="dc",
        max_iteration=50,
        tolerance_mva=1e-7,
    )

    assert bool(roundtrip_pp_net.converged)

    _assert_same_pandapower_network_results(original_pp_net, roundtrip_pp_net)


@pytest.mark.parametrize(
    "case_name, network_factory",
    PANDAPOWER_NETWORK_CASES,
    ids=[case_name for case_name, _ in PANDAPOWER_NETWORK_CASES],
)
def test_pandapower_powsybl_pandapower_roundtrip(
    case_name, network_factory, tmp_path, pyp_lf, pandapower
) -> None:
    original_pp_net = network_factory(pandapower)

    _run_conversion_pandapower_cycle(
        case_name=case_name,
        original_pp_net=original_pp_net,
        tmp_path=tmp_path,
        pyp_lf=pyp_lf,
        pandapower=pandapower,
    )
