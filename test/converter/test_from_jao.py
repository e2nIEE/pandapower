# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import itertools
import json
import os

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from pandapower.converter.jao import from_jao
from pandapower.converter.jao._common import _get_bus_idx
from pandapower.converter.jao._correction import _repair_incomplete_line_data
from pandapower.converter.jao._geodata import (
    _add_bus_geo,
    _fill_geo_at_one_sided_branches_without_geo_extent,
    _parse_html_str,
)
from pandapower.converter.jao._grid_groups import (
    _invent_connections_between_grid_groups,
    _same_name_base_mask,
    drop_islanded_grid_groups,
)
from pandapower.converter.jao._locnames import _normalize_locname
from pandapower.converter.jao._transformers import (
    _allocate_trafos_to_buses_and_create_buses,
    _find_trafo_locations,
    _get_transformer_voltages,
    _select_rated_current_a,
)
from pandapower.create import create_buses, create_lines_from_parameters, create_transformers_from_parameters
from pandapower.network import pandapowerNet
from pandapower.toolbox.comparison import nets_equal
from test import test_path


def test_from_jao_with_testfile():
    testfile = os.path.join(test_path, "converter", "jao_testfiles", "testfile.xlsx")
    assert os.path.isfile(testfile)

    # --- net1
    net1 = from_jao(testfile, None, False)

    assert len(net1.bus) == 10
    assert len(net1.line) == 7
    assert net1.line.Tieline.sum() == 2
    assert len(net1.trafo) == 1

    # line data conversion
    assert np.all(
        (0.01 < net1.line[["r_ohm_per_km", "x_ohm_per_km"]]) & (net1.line[["r_ohm_per_km", "x_ohm_per_km"]] < 0.4)
    )
    assert np.all((0.5 < net1.line["c_nf_per_km"]) & (net1.line["c_nf_per_km"] < 25))
    assert np.all(net1.line["g_us_per_km"] < 1)
    assert np.all((0.2 < net1.line["max_i_ka"]) & (net1.line["max_i_ka"] < 5))

    # trafo data conversion
    assert 100 < net1.trafo.sn_mva.iat[0] < 1000
    assert 6 < net1.trafo.vk_percent.iat[0] < 65
    assert 0.25 < net1.trafo.vkr_percent.iat[0] < 1.2
    assert 10 < net1.trafo.pfe_kw.iat[0] < 1000
    assert net1.trafo.i0_percent.iat[0] < 0.1
    assert np.isclose(net1.trafo.shift_degree.iat[0], 90)
    assert np.isclose(net1.trafo.tap_step_degree.iat[0], 1.794)
    assert net1.trafo.tap_min.iat[0] == -17
    assert net1.trafo.tap_max.iat[0] == 17

    # --- net2
    net2 = from_jao(testfile, None, True)
    nets_equal(net1, net2)  # extend_data_for_grid_group_connections makes no difference here

    # --- net3
    net3 = from_jao(testfile, None, True, drop_grid_groups_islands=True)
    assert len(net3.bus) == 6
    assert len(net3.line) == 5
    assert net3.line.Tieline.sum() == 1
    assert len(net3.trafo) == 1


@pytest.mark.parametrize("du_value", [0, None])
def test_from_jao_sets_ideal_tap_changer_type(tmp_path, du_value):
    testfile = os.path.join(test_path, "converter", "jao_testfiles", "testfile.xlsx")
    # du == 0 (or missing) with dphi != 0 identifies an ideal phase shifter. The deprecated
    # tap_phase_shifter flag is ignored by pandapower, so the supported tap_changer_type must be set.
    modified_testfile = tmp_path / f"testfile_ideal_phase_shifter_{du_value}.xlsx"
    workbook = load_workbook(testfile)
    sheet = workbook["Transformers"]
    du_column = next(cell.column for cell in sheet[2] if "Phase Regulation" in str(cell.value))
    sheet.cell(row=3, column=du_column).value = du_value
    workbook.save(modified_testfile)

    net = from_jao(str(modified_testfile), None, False)

    assert net.trafo.tap_changer_type.iat[0] == "Ideal"
    assert net.trafo.tap_step_percent.iat[0] == 0
    assert net.trafo.tap_step_degree.iat[0] == pytest.approx(1.794)


def test_from_jao_trafo_params_use_primary_winding_base(tmp_path):
    testfile = os.path.join(test_path, "converter", "jao_testfiles", "testfile.xlsx")
    # the primary winding can be the lower-voltage side. R/X/B/G and the rated current refer to it,
    # so sn_mva, the per-unit impedance base and pfe_kw must follow the primary voltage, not the hv side.
    original = from_jao(testfile, None, False)

    modified_testfile = tmp_path / "testfile_swapped_voltage_sides.xlsx"
    workbook = load_workbook(testfile)
    sheet = workbook["Transformers"]
    primary_column = next(cell.column for cell in sheet[2] if str(cell.value).strip() == "Primary")
    secondary_column = next(cell.column for cell in sheet[2] if str(cell.value).strip() == "Secondary")
    primary_kv = sheet.cell(row=3, column=primary_column).value
    secondary_kv = sheet.cell(row=3, column=secondary_column).value
    sheet.cell(row=3, column=primary_column).value = secondary_kv
    sheet.cell(row=3, column=secondary_column).value = primary_kv
    workbook.save(modified_testfile)

    swapped = from_jao(str(modified_testfile), None, False)

    # hv/lv allocation is unaffected, only the primary winding voltage differs
    assert swapped.trafo.vn_hv_kv.iat[0] == original.trafo.vn_hv_kv.iat[0]
    assert swapped.trafo.vn_lv_kv.iat[0] == original.trafo.vn_lv_kv.iat[0]
    # the swapped primary winding is now the original secondary winding
    swapped_primary_ratio = secondary_kv / primary_kv
    assert swapped.trafo.sn_mva.iat[0] == pytest.approx(original.trafo.sn_mva.iat[0] * swapped_primary_ratio)
    assert swapped.trafo.vk_percent.iat[0] == pytest.approx(original.trafo.vk_percent.iat[0] / swapped_primary_ratio)
    assert swapped.trafo.vkr_percent.iat[0] == pytest.approx(original.trafo.vkr_percent.iat[0] / swapped_primary_ratio)
    assert swapped.trafo.pfe_kw.iat[0] == pytest.approx(original.trafo.pfe_kw.iat[0] * swapped_primary_ratio**2)
    assert swapped.trafo.i0_percent.iat[0] == pytest.approx(original.trafo.i0_percent.iat[0])


def test_from_jao_with_missing_trafo_location_names(tmp_path):
    testfile = os.path.join(test_path, "converter", "jao_testfiles", "testfile.xlsx")
    # an all-NaN Location / Full Name column is read as float64 and must not break the .str accessors
    modified_testfile = tmp_path / "testfile_missing_trafo_location.xlsx"
    workbook = load_workbook(testfile)
    workbook["Transformers"].cell(row=3, column=1).value = None
    workbook.save(modified_testfile)

    net = from_jao(str(modified_testfile), None, False)

    assert len(net.trafo) == 1
    assert "Transformer 0" in set(net.bus.name)


def test_repair_incomplete_line_data_matches_name_variants():
    cols = pd.MultiIndex.from_tuples(
        [
            (None, "NE_name"),
            (None, "Voltage_level(kV)"),
            ("Substation_1", "Full_name"),
            ("Substation_2", "Full_name"),
            ("Electrical Parameters", "Length_(km)"),
        ]
    )
    lines = pd.DataFrame(
        [
            # known voltage under the accent spelling of the substation name
            ["X", 380.0, "Röhrsdorf", "Ambiguous", 10.0],
            # an accent/case variant of the same name must receive the same voltage
            ["X", np.nan, "Rohrsdorf", "Ambiguous", 10.0],
            # makes "Ambiguous" unsuited as a source of a unique voltage
            ["Y", 220.0, "Ambiguous", "Else", 10.0],
            # no usable name -> voltage must remain unknown
            ["Z", np.nan, None, "Ambiguous", 10.0],
        ],
        columns=cols,
    )
    data = {"Lines": lines, "Tielines": lines.iloc[0:0].copy()}

    _repair_incomplete_line_data(data)

    vn_col = (None, "Voltage_level(kV)")
    repaired = data["Lines"]
    assert repaired.at[0, vn_col] == 380.0
    assert repaired.at[1, vn_col] == 380.0
    assert repaired.at[2, vn_col] == 220.0
    assert pd.isna(repaired.at[3, vn_col])


def test_parse_html_str_widget_id():
    point = {"lng": [1.0, 2.0], "lat": [3.0, 4.0]}
    args = [None] * 7
    args[0] = [[[point]]]
    args[6] = ["EIC Code:<b> EIC1</b><b>NE name: Line A: </b>"]
    geo = {"x": {"calls": [{"method": "addPolylines", "args": args}]}}
    # widget id differs from the one initially hardcoded in the converter
    html = f'<script type="application/json" data-for="htmlwidget-0123456789abcdef0123">{json.dumps(geo)}</script>'

    line_geo_data = _parse_html_str(html)
    assert len(line_geo_data) == 4
    assert set(line_geo_data["geo_dim"]) == {"lng", "lat"}

    with pytest.raises(AssertionError):
        _parse_html_str("<html><body>no geodata</body></html>")


def test_add_bus_geo_uses_only_connected_endpoints():
    net = pandapowerNet(name="jao_geo")
    bus = create_buses(net, 1, 380, name=["SUB"])[0]
    bus_a = create_buses(net, 1, 380, name=["A"])[0]
    bus_b = create_buses(net, 1, 380, name=["B"])[0]
    create_lines_from_parameters(net, bus, bus_a, 1.0, 0.1, 0.1, 1.0, 1.0, name=["L1"])  # bus is "from"
    create_lines_from_parameters(net, bus_b, bus, 1.0, 0.1, 0.1, 1.0, 1.0, name=["L2"])  # bus is "to"
    net.line["EIC_Code"] = ["EIC1", "EIC2"]
    net.line["Tieline"] = [False, False]

    rows = []
    for eic, name, endpoint, lng, lat in [
        ("EIC1", "L1", "from", 1.0, 1.0),
        ("EIC1", "L1", "to", 2.0, 2.0),
        ("EIC2", "L2", "from", 8.0, 8.0),
        ("EIC2", "L2", "to", 9.0, 9.0),
    ]:
        rows += [[eic, name, endpoint, "lng", lng], [eic, name, endpoint, "lat", lat]]
    line_geo_data = pd.DataFrame(rows, columns=["EIC_Code", "name", "bus", "geo_dim", "value"])

    _add_bus_geo(net, line_geo_data)

    # bus is connected to EIC1 "from" and EIC2 "to"; the far endpoints must not be used
    coords = json.loads(net.bus.geo.at[bus])["coordinates"]
    assert coords in ([1.0, 1.0], [9.0, 9.0])
    assert coords not in ([2.0, 2.0], [8.0, 8.0])
    assert json.loads(net.bus.geo.at[bus_a])["coordinates"] == [2.0, 2.0]
    assert json.loads(net.bus.geo.at[bus_b])["coordinates"] == [8.0, 8.0]


def _make_disconnected_net(group_sizes):
    net = pandapowerNet(name="jao_grid_groups")
    groups = []
    for size in group_sizes:
        first_bus = len(net.bus)
        buses = create_buses(net, size, 380, name=[f"B{first_bus + i}" for i in range(size)])
        for from_bus, to_bus in itertools.pairwise(buses):
            create_lines_from_parameters(net, from_bus, to_bus, 1.0, 0.1, 0.1, 1.0, 1.0)
        groups.append(set(buses))
    return net, groups


def test_drop_islanded_grid_groups_max_keeps_single_largest():
    # tie at the largest size: exactly one of the tied groups must be kept, the other one dropped
    net, (group_a, group_b, _) = _make_disconnected_net([3, 3, 2])
    drop_islanded_grid_groups(net, "max")
    assert len(net.bus) == 3
    assert set(net.bus.index) in (group_a, group_b)

    # distinct sizes: the largest group must be kept
    net, (large, _) = _make_disconnected_net([4, 2])
    drop_islanded_grid_groups(net, "max")
    assert set(net.bus.index) == large

    # only one group: nothing can be dropped and nothing must be raised
    net, _ = _make_disconnected_net([2])
    drop_islanded_grid_groups(net, "max")
    assert len(net.bus) == 2


def test_same_name_base_mask_matches_complete_tokens():
    # "A" must not match names in which it is only a substring
    names = pd.Series(["A", "AUX", "A NORD", "PARIS-A", "A1"])
    assert _same_name_base_mask(names, "A").tolist() == [True, False, True, True, False]

    # exact and token-separated name variants must still match
    names = pd.Series(["CAEN", "CAEN (2)", "CAEN NORD", "CAEN-X", "CAENX"])
    assert _same_name_base_mask(names, "CAEN").tolist() == [True, True, True, True, False]


def test_invent_connections_without_duplicate_location_name():
    # no duplicated normalized location name exists, so the trafo step defines no grid group id
    net = pandapowerNet(name="jao_grid_groups")
    create_buses(net, 1, 380, name=["A"])
    create_buses(net, 1, 380, name=["A NORD"])

    _invent_connections_between_grid_groups(net)

    assert len(net.bus) == 1
    assert list(net.bus.name) == ["A"]


def test_invent_connections_keeps_own_grid_group():
    net = pandapowerNet(name="jao_grid_groups")
    x = create_buses(net, 1, 380, name=["X"])[0]
    x_2 = create_buses(net, 1, 380, name=["X 2"])[0]
    dup = create_buses(net, 1, 220, name=["DUP"])[0]
    dup_x = create_buses(net, 1, 220, name=["DUP X"])[0]
    e_220 = create_buses(net, 1, 220, name=["E"])[0]
    e_380 = create_buses(net, 1, 380, name=["E"])[0]
    create_lines_from_parameters(net, x, dup_x, 1.0, 0.1, 0.1, 1.0, 1.0)
    create_lines_from_parameters(net, x_2, dup, 1.0, 0.1, 0.1, 1.0, 1.0)
    create_lines_from_parameters(net, dup, e_220, 1.0, 0.1, 0.1, 1.0, 1.0)
    create_lines_from_parameters(net, dup_x, e_380, 1.0, 0.1, 0.1, 1.0, 1.0)
    net.bus["zone"] = "TSO"  # the transformer step requires net.bus.zone

    _invent_connections_between_grid_groups(net)

    # fusing X with X 2 merges both islands; DUP and DUP X are then already in the same
    # island and must not be fused again based on a grid group id of an unrelated location
    assert len(net.bus) == 5
    assert sorted(net.bus.name) == ["DUP", "DUP X", "E", "E", "X"]


def test_normalize_locname_transliterates_polish_l():
    # "Ł"/"ł" do not decompose under NFKD and must be transliterated explicitly
    assert _normalize_locname("Łódź") == "lodz"
    assert _normalize_locname("ŁÓDŹ") == "lodz"
    assert _normalize_locname("Miłosna") == _normalize_locname("Milosna") == "milosna"
    assert _normalize_locname("Zabłocie") == "zablocie"


def test_trafo_locations_match_known_jao_name_variants():
    # bus spellings as they occur in the JAO Lines sheets of the 2024-2026 releases
    bus_location_names = ["CHELM", "Mikulowa", "OLSZTYN-MATKI", "Stanislawow", "Vierraden"]
    # formerly handled by the hardcoded rename_locnames table of the old converter
    expectations = {
        "PSTMIKULOWA": "Mikulowa",
        "PSTMikulowa-PF4": "Mikulowa",
        "TR Chelm-A2": "CHELM",
        "TR OLSZTYN-MATK-A3": "OLSZTYN-MATKI",
        "TR OLSZTYN-MATKII-A3": "OLSZTYN-MATKI",
        "TR STANISLAWOW-A1": "Stanislawow",
        "TR VIERRADEN-A1": "Vierraden",
        "PST Vierraden 441": "Vierraden",
    }

    resolved = _find_trafo_locations(pd.Series(list(expectations)), bus_location_names)

    assert resolved.tolist() == list(expectations.values())


def test_find_trafo_locations_without_bus_locations():
    # no bus locations at all: names must be left unresolved (warning + fallback bus), not raise
    resolved = _find_trafo_locations(pd.Series(["TR A-A1", "B"]), [])

    assert len(resolved) == 2
    assert resolved.isna().all()


def test_find_trafo_locations_accepts_all_missing_location_column():
    # an all-NaN Location / Full Name column is read as float64, which .str cannot be used on
    resolved = _find_trafo_locations(pd.Series([np.nan, np.nan]), ["A"])

    assert resolved.isna().all()


def test_select_rated_current_a_prefers_fixed_then_max():
    cols = pd.MultiIndex.from_tuples(
        [
            ("Maximum Current Imax (A) primary", "Fixed"),
            ("Maximum Current Imax (A) primary", "Max"),
            ("Maximum Current Imax (A) primary", "Min"),
            ("Location", "Full Name"),
        ]
    )
    trafos = pd.DataFrame(
        [
            [1000.0, np.nan, 500.0, "Fixed only"],
            [np.nan, 900.0, 400.0, "Max only"],
            [1200.0, 800.0, 700.0, "Fixed and Max"],
        ],
        columns=cols,
    )
    assert _select_rated_current_a(trafos).tolist() == [1000.0, 900.0, 1200.0]

    # a Min-only row must not silently fall back to the minimum rating
    min_only = pd.DataFrame([[np.nan, np.nan, 600.0, "Min only"]], columns=cols)
    with pytest.raises(ValueError, match="Min only"):
        _select_rated_current_a(min_only)


def test_get_transformer_voltages_rejects_missing_voltage_level():
    cols = pd.MultiIndex.from_tuples([("Voltage_level(kV)", "Primary"), ("Voltage_level(kV)", "Secondary")])
    bus_idx = pd.Series([0, 1], index=pd.MultiIndex.from_arrays([["A", "B"], [380.0, 380.0]], names=["name", "vn_kv"]))

    trafos = pd.DataFrame([[380.0, 110.0], [220.0, 400.0]], columns=cols)
    vn_hv_kv, vn_lv_kv, vn_primary_kv = _get_transformer_voltages({"Transformers": trafos}, bus_idx)
    assert vn_hv_kv.tolist() == [380.0, 400.0]
    assert vn_lv_kv.tolist() == [110.0, 220.0]
    # the primary winding is not necessarily the hv side
    assert vn_primary_kv.tolist() == [380.0, 220.0]

    # a missing voltage level on either side must not silently propagate NaN
    trafos = pd.DataFrame([[380.0, 110.0], [220.0, np.nan], [np.nan, np.nan]], columns=cols)
    with pytest.raises(ValueError) as excinfo:
        _get_transformer_voltages({"Transformers": trafos}, bus_idx)
    assert "[1, 2]" in str(excinfo.value)


def test_allocate_trafos_creates_separate_buses_for_missing_locations():
    cols = pd.MultiIndex.from_tuples([("Location", "Full Name"), ("Location", "TSO")])
    data = {"Transformers": pd.DataFrame([[np.nan, "A"], ["Unknown Sub", "C"], [np.nan, "B"]], columns=cols)}
    net = pandapowerNet(name="jao_trafos")
    create_buses(net, 1, 380, name=["Existing"])

    conn = _allocate_trafos_to_buses_and_create_buses(
        net, data, _get_bus_idx(net), np.array([380.0, 380.0, 380.0]), np.array([110.0, 110.0, 110.0])
    )

    # every transformer without a location gets its own bus; the raw fallback name is kept otherwise
    assert len(net.bus) == 7
    assert conn.hv_bus.nunique() == 3
    assert {"Transformer 0", "Transformer 2", "Unknown Sub"} <= set(net.bus.name)


def test_allocate_trafos_reuses_buses_across_passes():
    cols = pd.MultiIndex.from_tuples([("Location", "Full Name"), ("Location", "TSO")])
    data = {"Transformers": pd.DataFrame([["A", "TSO1"], ["A", "TSO2"]], columns=cols)}
    net = pandapowerNet(name="jao_trafos")
    create_buses(net, 1, 380, name=["A"], zone="TSO")

    conn = _allocate_trafos_to_buses_and_create_buses(
        net, data, _get_bus_idx(net), np.array([220.0, 380.0]), np.array([110.0, 220.0])
    )

    # the second transformer's lv side must reuse the A/220 bus created for the first one's hv side
    a_220 = net.bus.index[(net.bus.name == "A") & (net.bus.vn_kv == 220)]
    assert len(net.bus) == 3
    assert len(a_220) == 1
    assert conn.hv_bus.iat[0] == conn.lv_bus.iat[1] == a_220[0]


def _add_trafo(net, hv_bus, lv_bus):
    create_transformers_from_parameters(net, [hv_bus], [lv_bus], 1.0, 380, 110, 0, 1, 0, 0, 0)


def test_fill_geo_propagates_through_transformers():
    geo = '{"coordinates": [0.0, 0.0], "type": "Point"}'

    # --- transformer only: the geo of one side must reach the other side
    net = pandapowerNet(name="jao_geo_trafo")
    hv_bus = create_buses(net, 1, 380, name=["HV"])[0]
    lv_bus = create_buses(net, 1, 110, name=["LV"])[0]
    _add_trafo(net, hv_bus, lv_bus)
    net.bus["geo"] = [geo, None]
    net.line["geo"] = pd.Series(index=net.line.index, dtype=object)

    _fill_geo_at_one_sided_branches_without_geo_extent(net)

    assert net.bus.geo.at[lv_bus] == geo

    # --- line followed by a transformer: propagation must continue through the trafo
    net = pandapowerNet(name="jao_geo_chain")
    bus0 = create_buses(net, 1, 380, name=["B0"])[0]
    bus1 = create_buses(net, 1, 380, name=["B1"])[0]
    bus2 = create_buses(net, 1, 110, name=["B2"])[0]
    create_lines_from_parameters(net, bus0, bus1, 1.0, 0.1, 0.1, 1.0, 1.0, name=["L"])
    _add_trafo(net, bus1, bus2)
    net.bus["geo"] = [geo, None, None]

    _fill_geo_at_one_sided_branches_without_geo_extent(net)

    assert net.bus.geo.at[bus1] == geo
    assert net.bus.geo.at[bus2] == geo


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
