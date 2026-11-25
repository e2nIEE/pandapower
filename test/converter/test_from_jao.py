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
from pandapower.converter.jao._common import _drop_duplicates_and_join_TSO, _get_bus_idx, _get_float_column
from pandapower.converter.jao._correction import _data_correction, _repair_incomplete_line_data
from pandapower.converter.jao._geodata import (
    _add_bus_geo,
    _fill_geo_at_one_sided_branches_without_geo_extent,
    _parse_html_str,
)
from pandapower.converter.jao._grid_groups import (
    _invent_connections_between_grid_groups,
    _same_name_base_mask,
    drop_islanded_grid_groups,
    get_grid_groups,
)
from pandapower.converter.jao._locnames import _normalize_locname
from pandapower.converter.jao._transformers import (
    _allocate_trafos_to_buses_and_create_buses,
    _create_transformers_and_buses,
    _find_trafo_locations,
    _get_transformer_voltages,
    _select_rated_current_a,
)
from pandapower.create import LineParams, create_buses, create_lines, create_transformers_from_parameters
from pandapower.network import pandapowerNet
from pandapower.toolbox.comparison import nets_equal
from test import test_path


def _line_params() -> LineParams:
    return {"r_ohm_per_km": 0.1, "x_ohm_per_km": 0.1, "c_nf_per_km": 1.0, "max_i_ka": 1.0}


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
    assert net1.trafo.i0_percent.iat[0] == pytest.approx(11.07440227, rel=1e-6)
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


def test_from_jao_corrects_string_dtype_phase_regulation(tmp_path):
    testfile = os.path.join(test_path, "converter", "jao_testfiles", "testfile.xlsx")
    # a column that only holds string cells is read as pandas string dtype, which is not an object
    # dtype. Values like "1,5/0" must still be reduced to their second number.
    modified_testfile = tmp_path / "testfile_string_phase_regulation.xlsx"
    workbook = load_workbook(testfile)
    sheet = workbook["Transformers"]
    du_column = next(cell.column for cell in sheet[2] if "Phase Regulation" in str(cell.value))
    dphi_column = next(cell.column for cell in sheet[2] if "Angle Regulation" in str(cell.value))
    sheet.cell(row=3, column=du_column).value = "1,5/0"
    sheet.cell(row=3, column=dphi_column).value = "1/2,5"
    workbook.save(modified_testfile)

    net = from_jao(str(modified_testfile), None, False)

    assert net.trafo.tap_changer_type.iat[0] == "Ideal"
    assert net.trafo.tap_step_percent.iat[0] == 0
    assert net.trafo.tap_step_degree.iat[0] == pytest.approx(2.5)


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
    # with a fixed rated current and B/G, the no-load current in % of rated current scales with the primary voltage
    assert swapped.trafo.i0_percent.iat[0] == pytest.approx(original.trafo.i0_percent.iat[0] * swapped_primary_ratio)


def test_from_jao_trafo_i0_percent_independent_of_net_base():
    testfile = os.path.join(test_path, "converter", "jao_testfiles", "testfile.xlsx")
    # i0_percent is a transformer rating quantity; the net base must not rescale it
    base_1 = from_jao(testfile, None, False, sn_mva=1.0)
    base_100 = from_jao(testfile, None, False, sn_mva=100.0)

    assert base_100.trafo.i0_percent.iat[0] == pytest.approx(base_1.trafo.i0_percent.iat[0])


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
            ("Electrical Parameters", "Resistance_R(Ω)"),
            ("Electrical Parameters", "Reactance_X(Ω)"),
            ("Electrical Parameters", "Susceptance_B(μS)"),
        ]
    )
    lines = pd.DataFrame(
        [
            # known voltage under the accent spelling of the substation name
            ["X", 380.0, "Röhrsdorf", "Ambiguous", 10.0, 0.1, 0.2, 5.0],
            # an accent/case variant of the same name must receive the same voltage
            ["X", np.nan, "Rohrsdorf", "Ambiguous", 10.0, 0.1, 0.2, 5.0],
            # makes "Ambiguous" unsuited as a source of a unique voltage
            ["Y", 220.0, "Ambiguous", "Else", 10.0, 0.1, 0.2, 5.0],
            # no usable name -> voltage must remain unknown
            ["Z", np.nan, None, "Ambiguous", 10.0, 0.1, 0.2, 5.0],
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


def test_repair_incomplete_line_data_counts_normalized_name_candidates():
    cols = pd.MultiIndex.from_tuples(
        [
            (None, "NE_name"),
            (None, "Voltage_level(kV)"),
            ("Substation_1", "Full_name"),
            ("Substation_2", "Full_name"),
            ("Electrical Parameters", "Length_(km)"),
            ("Electrical Parameters", "Resistance_R(Ω)"),
            ("Electrical Parameters", "Reactance_X(Ω)"),
            ("Electrical Parameters", "Susceptance_B(μS)"),
        ]
    )
    lines = pd.DataFrame(
        [
            # two spellings of the same substation, both matching the same NE_name token
            ["X", 380.0, "KROSNO", "Else", 10.0, 0.1, 0.2, 5.0],
            ["X", 380.0, "Krosno", "Else", 10.0, 0.1, 0.2, 5.0],
            ["Y", 380.0, "Vierraden", "Else", 10.0, 0.1, 0.2, 5.0],
            # the two raw spellings are one location, so the name must be derived unambiguously
            ["KROSNO - Missing", 380.0, None, "Other", 10.0, 0.1, 0.2, 5.0],
            # the other endpoint is the same location, only spelled differently -> no candidate
            ["VIERRADEN - Same", 380.0, None, "vierraden", 10.0, 0.1, 0.2, 5.0],
        ],
        columns=cols,
    )
    data = {"Lines": lines, "Tielines": lines.iloc[0:0].copy()}

    _repair_incomplete_line_data(data)

    s1 = ("Substation_1", "Full_name")
    repaired = data["Lines"]
    assert _normalize_locname(repaired.at[3, s1]) == "krosno"
    assert pd.isna(repaired.at[4, s1])


def test_repair_incomplete_line_data_matches_hyphenated_names():
    cols = pd.MultiIndex.from_tuples(
        [
            (None, "NE_name"),
            (None, "Voltage_level(kV)"),
            ("Substation_1", "Full_name"),
            ("Substation_2", "Full_name"),
            ("Electrical Parameters", "Length_(km)"),
            ("Electrical Parameters", "Resistance_R(Ω)"),
            ("Electrical Parameters", "Reactance_X(Ω)"),
            ("Electrical Parameters", "Susceptance_B(μS)"),
        ]
    )
    lines = pd.DataFrame(
        [
            # complete rows make these spellings known
            ["Else - Belle-Epine", 380.0, "Else", "Belle-Epine", 10.0, 0.1, 0.2, 5.0],
            ["Else - Wuhlheide (Abzweig)", 380.0, "Else", "Wuhlheide (Abzweig)", 10.0, 0.1, 0.2, 5.0],
            # hyphens and parentheses inside a name are not endpoint separators
            ["Belle-Epine - Else", 380.0, None, "Else", 10.0, 0.1, 0.2, 5.0],
            # the populated other endpoint is still excluded
            ["Belle-Epine - Else", 380.0, "Belle-Epine", None, 10.0, 0.1, 0.2, 5.0],
            ["Wuhlheide (Abzweig) - Else", 380.0, None, "Else", 10.0, 0.1, 0.2, 5.0],
        ],
        columns=cols,
    )
    data = {"Lines": lines, "Tielines": lines.iloc[0:0].copy()}

    _repair_incomplete_line_data(data)

    s1, s2 = ("Substation_1", "Full_name"), ("Substation_2", "Full_name")
    repaired = data["Lines"]
    assert repaired.at[2, s1] == "Belle-Epine"
    assert repaired.at[3, s2] == "Else"
    assert repaired.at[4, s1] == "Wuhlheide (Abzweig)"


def test_repair_incomplete_line_data_assigns_into_all_missing_endpoint_column():
    cols = pd.MultiIndex.from_tuples(
        [
            (None, "NE_name"),
            (None, "Voltage_level(kV)"),
            ("Substation_1", "Full_name"),
            ("Substation_2", "Full_name"),
            ("Electrical Parameters", "Length_(km)"),
            ("Electrical Parameters", "Resistance_R(Ω)"),
            ("Electrical Parameters", "Reactance_X(Ω)"),
            ("Electrical Parameters", "Susceptance_B(μS)"),
        ]
    )
    lines = pd.DataFrame([["A - B", 380.0, "A", "B", 10.0, 0.1, 0.2, 5.0]], columns=cols)
    tielines = pd.DataFrame(
        [
            ["A - B", 380.0, None, "B", 10.0, 0.1, 0.2, 5.0],
            ["X - C", 380.0, None, "C", 10.0, 0.1, 0.2, 5.0],
        ],
        columns=cols,
    )
    s1 = ("Substation_1", "Full_name")
    # a sheet without any Substation_1 values reads that column as float
    tielines[s1] = tielines[s1].astype(float)
    data = {"Lines": lines, "Tielines": tielines}

    _repair_incomplete_line_data(data)

    repaired = data["Tielines"]
    assert repaired.at[0, s1] == "A"
    assert pd.isna(repaired.at[1, s1])


def test_repair_incomplete_line_data_keeps_electrically_populated_rows():
    cols = pd.MultiIndex.from_tuples(
        [
            (None, "NE_name"),
            (None, "Voltage_level(kV)"),
            ("Substation_1", "Full_name"),
            ("Substation_2", "Full_name"),
            ("Electrical Parameters", "Length_(km)"),
            ("Electrical Parameters", "Resistance_R(Ω)"),
            ("Electrical Parameters", "Reactance_X(Ω)"),
            ("Electrical Parameters", "Susceptance_B(μS)"),
            (None, "TSO"),
        ]
    )
    lines = pd.DataFrame(
        [
            # metadata only, no substation, voltage or electrical data -> dropped
            ["A-B", None, None, None, None, None, None, None, "TSO"],
            # electrically populated rows without endpoints -> kept for downstream validation
            ["A-B", None, None, None, None, 0.1, None, None, None],
            ["A-B", None, None, None, None, None, None, 5.0, None],
        ],
        columns=cols,
    )
    data = {"Lines": lines, "Tielines": lines.iloc[0:0].copy()}

    _repair_incomplete_line_data(data)

    assert list(data["Lines"].index) == [1, 2]


_JAO_LINE_COLUMNS = pd.MultiIndex.from_tuples(
    [
        ("Substation_1", "Full_name"),
        ("Substation_2", "Full_name"),
        ("Unnamed: 0", "NE_name"),
        ("Unnamed: 1", "Voltage_level(kV)"),
        ("Electrical Parameters", "Length_(km)"),
        ("Electrical Parameters", "Resistance_R(Ω)"),
        ("Electrical Parameters", "Reactance_X(Ω)"),
        ("Electrical Parameters", "Susceptance_B(μS)"),
        ("Maximum Current Imax (A)", "Fixed"),
    ]
)
_JAO_TRAFO_COLUMNS = pd.MultiIndex.from_tuples(
    [("Maximum Current Imax (A) primary", col) for col in ("Min", "Max", "Fixed")]
    + [("Voltage_level(kV)", col) for col in ("Primary", "Secondary")]
    + [
        ("Location", "Full Name"),
        ("Phase Shifting Properties", "Taps used for RAO"),
        ("Phase Shifting Properties", "Phase Regulation δu (%)"),
        ("Phase Shifting Properties", "Angle Regulation δu (%)"),
    ]
)


@pytest.mark.parametrize(
    "fixed_values, expected",
    [(["1000", "\xa0"], [1000.0, 999000.0]), (["10,5", "-"], [10.5, 999000.0])],
)
def test_data_correction_imax_fixed_with_string_dtype(fixed_values, expected):
    # pandas 3 reads an all-string Imax column as the str dtype, which rejects assigning
    # parsed numbers via .loc; the values must be assigned by column key instead
    def line_sheet():
        df = pd.DataFrame(
            [
                ["A", "B", "A-B", 380, 1.0, 0.1, 0.2, 5.0, fixed_values[0]],
                ["A", "C", "A-C", 380, 1.0, 0.1, 0.2, 5.0, fixed_values[1]],
            ],
            columns=_JAO_LINE_COLUMNS,
        )
        df[("Maximum Current Imax (A)", "Fixed")] = df[("Maximum Current Imax (A)", "Fixed")].astype("str")
        return df

    data = {
        "Lines": line_sheet(),
        "Tielines": line_sheet(),
        "Transformers": pd.DataFrame(columns=_JAO_TRAFO_COLUMNS),
    }

    _data_correction(data, "", 999)

    for key in ("Lines", "Tielines"):
        result = data[key][("Maximum Current Imax (A)", "Fixed")]
        assert result.tolist() == expected
        assert result.dtype == np.float64


@pytest.mark.parametrize(
    "taps_values, taps_dtype, expected",
    [
        ([0, 0], "int64", ["0;0", "0;0"]),
        ([0.0, np.nan], "float64", ["0;0", "0;0"]),
        (["<-6;6>", "-5/5", None], "object", ["-6;6", "-5;5", "0;0"]),
    ],
)
def test_data_correction_rao_taps_normalization(taps_values, taps_dtype, expected):
    # the taps column may be read as an integer/float dtype, into which text cannot be assigned
    # via .loc; the whole column must be replaced by the normalized text Series
    count = len(taps_values)
    data = {
        "Lines": pd.DataFrame(columns=_JAO_LINE_COLUMNS),
        "Tielines": pd.DataFrame(columns=_JAO_LINE_COLUMNS),
        "Transformers": pd.DataFrame(
            {
                ("Maximum Current Imax (A) primary", "Min"): [np.nan] * count,
                ("Maximum Current Imax (A) primary", "Max"): [np.nan] * count,
                ("Maximum Current Imax (A) primary", "Fixed"): [1000.0] * count,
                ("Voltage_level(kV)", "Primary"): [380.0] * count,
                ("Voltage_level(kV)", "Secondary"): [220.0] * count,
                ("Location", "Full Name"): [f"TR {idx}" for idx in range(count)],
                ("Phase Shifting Properties", "Taps used for RAO"): pd.Series(taps_values, dtype=taps_dtype),
                ("Phase Shifting Properties", "Phase Regulation δu (%)"): [0.0] * count,
                ("Phase Shifting Properties", "Angle Regulation δu (%)"): [1.0] * count,
            }
        ),
    }

    _data_correction(data, "", 999)

    result = data["Transformers"][("Phase Shifting Properties", "Taps used for RAO")]
    assert result.tolist() == expected


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
    assert line_geo_data["EIC_Code"].unique().tolist() == ["EIC1"]

    with pytest.raises(AssertionError):
        _parse_html_str("<html><body>no geodata</body></html>")


def test_add_bus_geo_uses_only_connected_endpoints():
    net = pandapowerNet(name="jao_geo")
    bus = create_buses(net, 1, 380, name=["SUB"])[0]
    bus_a = create_buses(net, 1, 380, name=["A"])[0]
    bus_b = create_buses(net, 1, 380, name=["B"])[0]
    create_lines(net, bus, bus_a, 1.0, _line_params(), name=["L1"])  # bus is "from"
    create_lines(net, bus_b, bus, 1.0, _line_params(), name=["L2"])  # bus is "to"
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
            create_lines(net, from_bus, to_bus, 1.0, _line_params())
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
    create_lines(net, x, dup_x, 1.0, _line_params())
    create_lines(net, x_2, dup, 1.0, _line_params())
    create_lines(net, dup, e_220, 1.0, _line_params())
    create_lines(net, dup_x, e_380, 1.0, _line_params())
    net.bus["zone"] = "TSO"  # the transformer step requires net.bus.zone

    _invent_connections_between_grid_groups(net)

    # fusing X with X 2 merges both islands; DUP and DUP X are then already in the same
    # island and must not be fused again based on a grid group id of an unrelated location
    assert len(net.bus) == 5
    assert sorted(net.bus.name) == ["DUP", "DUP X", "E", "E", "X"]


def test_invent_connections_skips_buses_fused_earlier():
    net = pandapowerNet(name="jao_grid_groups")
    create_buses(net, 1, 380, name=["DOUBLE"])
    create_buses(net, 1, 380, name=["Left"])
    create_buses(net, 1, 380, name=["DOUBLE"])
    create_buses(net, 1, 380, name=["Right"])
    create_lines(net, [0, 2], [1, 3], 1.0, _line_params())
    net.bus["zone"] = "TSO"  # the transformer step requires net.bus.zone

    _invent_connections_between_grid_groups(net)

    # the second DOUBLE bus is fused into the first one; the stale loop entry of the
    # removed bus must be skipped instead of being processed again
    assert len(net.bus) == 3
    assert sorted(net.bus.name) == ["DOUBLE", "Left", "Right"]


def test_invent_connections_uses_other_voltage_levels_of_location():
    def make_net(template_hv, template_lv):
        net = pandapowerNet(name="jao_grid_groups")
        a_380 = create_buses(net, 1, 380, name=["DUP"])[0]
        a_220 = create_buses(net, 1, 220, name=["DUP"])[0]
        b_110 = create_buses(net, 1, 110, name=["DUP"])[0]
        t_h = create_buses(net, 1, template_hv, name=["T_H"])[0]
        t_l = create_buses(net, 1, template_lv, name=["T_L"])[0]
        create_lines(net, a_380, a_220, 1.0, _line_params())
        create_transformers_from_parameters(net, [t_h], [t_l], 1.0, template_hv, template_lv, 0, 1, 0, 0, 0)
        net.bus["zone"] = "TSO"  # the transformer step requires net.bus.zone
        return net, a_220, b_110, len(get_grid_groups(net))

    # the representative pair 380/110 has no transformer template, but 220/110 does
    net, a_220, b_110, n_groups = make_net(220, 110)
    _invent_connections_between_grid_groups(net)
    assert len(net.trafo) == 2
    added = net.trafo.iloc[-1]
    assert added.hv_bus == a_220
    assert added.lv_bus == b_110
    assert len(get_grid_groups(net)) == n_groups - 1

    # no voltage pair of the location is supported by a transformer -> nothing is invented
    net, _, _, n_groups = make_net(380, 220)
    _invent_connections_between_grid_groups(net)
    assert len(net.trafo) == 1
    assert len(get_grid_groups(net)) == n_groups


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


def test_find_trafo_locations_keeps_suffix_token_frames_aligned():
    # one case-insensitive suffix split keeps both token frames aligned, so a noise token after a
    # suffix is still dropped; uppercase suffixes like -A2 keep working
    resolved = _find_trafo_locations(
        pd.Series(["TR BYCZYNA-A2", "New York KV-A1", "TR BYCZYNA-td3"]), ["Byczyna", "New York"]
    )

    assert resolved.tolist() == ["Byczyna", "New York", "Byczyna"]


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


@pytest.mark.parametrize("dtype", [None, "string"])
def test_drop_duplicates_and_join_ts_o_with_missing_values(dtype):
    bus_df = pd.DataFrame(
        [
            ["A", 1.0, "AB"],
            ["A", 1.0, None],
            ["A", 1.0, "CD"],
            ["B", 1.0, None],
            ["B", 1.0, ""],
        ],
        columns=["name", "vn_kv", "TSO"],
    )
    if dtype is not None:
        bus_df = bus_df.astype({"TSO": dtype})

    joined = _drop_duplicates_and_join_TSO(bus_df)

    # missing TSO values are ignored, non-empty ones keep the existing slash-join behavior and a
    # location without any TSO gets an empty zone
    assert joined.to_dict("records") == [
        {"name": "A", "vn_kv": 1.0, "TSO": "AB/CD"},
        {"name": "B", "vn_kv": 1.0, "TSO": ""},
    ]


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


def test_allocate_trafos_reuses_fallback_buses_across_passes():
    cols = pd.MultiIndex.from_tuples([("Location", "Full Name"), ("Location", "TSO")])
    data = {"Transformers": pd.DataFrame([["Unknown Sub", "T1"], ["Unknown Sub", "T2"]], columns=cols)}
    net = pandapowerNet(name="jao_trafos")
    create_buses(net, 1, 380, name=["Existing"])

    conn = _allocate_trafos_to_buses_and_create_buses(
        net, data, _get_bus_idx(net), np.array([220.0, 380.0]), np.array([110.0, 220.0])
    )

    # the nonblank fallback name is the bus identity, so the second transformer's lv side reuses the
    # Unknown Sub/220 bus created for the first one's hv side instead of duplicating it
    unknown_220 = net.bus.index[(net.bus.name == "Unknown Sub") & (net.bus.vn_kv == 220)]
    assert len(net.bus) == 4
    assert len(unknown_220) == 1
    assert conn.hv_bus.iat[0] == conn.lv_bus.iat[1] == unknown_220[0]


def test_allocate_trafos_keeps_blank_locations_separate():
    cols = pd.MultiIndex.from_tuples([("Location", "Full Name"), ("Location", "TSO")])
    data = {"Transformers": pd.DataFrame([[np.nan, "A"], [np.nan, "B"]], columns=cols)}
    net = pandapowerNet(name="jao_trafos")
    create_buses(net, 1, 380, name=["Existing"])

    conn = _allocate_trafos_to_buses_and_create_buses(
        net, data, _get_bus_idx(net), np.array([380.0, 380.0]), np.array([380.0, 380.0])
    )

    # blank locations keep their own identity, so equal-voltage sides are not merged
    assert len(net.bus) == 5
    assert set(conn.hv_bus).isdisjoint(set(conn.lv_bus))
    assert {net.bus.name.at[conn.hv_bus.iat[0]], net.bus.name.at[conn.hv_bus.iat[1]]} == {
        "Transformer 0",
        "Transformer 1",
    }


def test_create_transformers_and_buses_skips_empty_transformers_sheet():
    cols = pd.MultiIndex.from_tuples(
        [
            ("Voltage_level(kV)", "Primary"),
            ("Voltage_level(kV)", "Secondary"),
            ("Maximum Current Imax (A) primary", "Fixed"),
            ("Maximum Current Imax (A) primary", "Max"),
            ("Location", "Full Name"),
            ("Location", "TSO"),
        ]
    )
    net = pandapowerNet(name="jao_line_only")
    create_buses(net, 2, 380, name=["A", "B"])
    create_lines(net, 0, 1, 1.0, _line_params())

    # a line-only workbook has the Transformers headers but no data rows; a completely blank
    # sheet is read without any columns
    for trafos in [pd.DataFrame(columns=cols), pd.DataFrame()]:
        _create_transformers_and_buses(net, {"Transformers": trafos})

    assert len(net.trafo) == 0
    assert len(net.bus) == 2


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
    create_lines(net, bus0, bus1, 1.0, _line_params(), name=["L"])
    _add_trafo(net, bus1, bus2)
    net.bus["geo"] = [geo, None, None]

    _fill_geo_at_one_sided_branches_without_geo_extent(net)

    assert net.bus.geo.at[bus1] == geo
    assert net.bus.geo.at[bus2] == geo


def test_get_float_column_handles_string_dtype():
    # a text dtype column rejects assigning the numeric fill via .loc; the non-breaking space
    # must be replaced by a missing value before casting
    df = pd.DataFrame({"c": pd.Series(["\xa0", "0", "1.5", np.nan], dtype="string")})
    assert _get_float_column(df, "c").tolist() == [0.0, 0.0, 1.5, 0.0]


def test_data_correction_keeps_rows_with_only_current_rating():
    lines = pd.DataFrame(
        [
            ["A", "B", "A-B", 380, 10.0, 0.1, 0.2, 5.0, 1000.0],
            # only the current rating is real data -> must be kept for endpoint validation
            [None, None, "A-B current only", None, None, None, None, None, 1234.0],
            # no real electrical data at all -> dropped even though a fallback would be filled later
            [None, None, "A-B no data", None, None, None, None, None, np.nan],
        ],
        columns=_JAO_LINE_COLUMNS,
    )
    data = {
        "Lines": lines,
        "Tielines": lines.iloc[0:0].copy(),
        "Transformers": pd.DataFrame(columns=_JAO_TRAFO_COLUMNS),
    }

    _data_correction(data, "", 999)

    assert list(data["Lines"].index) == [0, 1]
    assert data["Lines"].at[1, ("Maximum Current Imax (A)", "Fixed")] == 1234.0


@pytest.mark.parametrize("trafos", [pd.DataFrame(), pd.DataFrame(columns=_JAO_TRAFO_COLUMNS)])
def test_data_correction_returns_early_for_empty_transformers(trafos):
    lines = pd.DataFrame([["A", "B", "A-B", 380, 10.0, 0.1, 0.2, 5.0, 1000.0]], columns=_JAO_LINE_COLUMNS)
    data = {"Lines": lines, "Tielines": lines.iloc[0:0].copy(), "Transformers": trafos}

    assert _data_correction(data, "html", 999)[0] == "html"


def test_parse_html_str_applies_locname_renames_to_decoded_names():
    point = {"lng": [1.0, 2.0], "lat": [3.0, 4.0]}
    args = [None] * 7
    args[0] = [[[point]]]
    args[6] = ["EIC Code:<b> KROSNOX</b><b>NE name: KROSNO - A: </b>"]
    geo = {"x": {"calls": [{"method": "addPolylines", "args": args}]}}
    html = f'<script type="application/json" data-for="htmlwidget-0123">{json.dumps(geo)}</script>'

    line_geo_data = _parse_html_str(html, [("KROSNO", "Krosno")])

    assert line_geo_data["name"].unique().tolist() == ["Krosno - A:"]
    # the rename must not touch identifiers or other JSON content
    assert line_geo_data["EIC_Code"].unique().tolist() == ["KROSNOX"]


def test_data_correction_does_not_rewrite_html_document():
    lines = pd.DataFrame(
        [
            ["KROSNO", "Else", "KROSNO - Else", 380, 10.0, 0.1, 0.2, 5.0, 1000.0],
            ["Krosno", "Else", "Krosno - Else", 380, 10.0, 0.1, 0.2, 5.0, 1000.0],
        ],
        columns=_JAO_LINE_COLUMNS,
    )
    data = {
        "Lines": lines,
        "Tielines": lines.iloc[0:0].copy(),
        "Transformers": pd.DataFrame(columns=_JAO_TRAFO_COLUMNS),
    }
    html = "<b>NE name: KROSNO - Else: </b>"

    assert _data_correction(data, html, 999)[0] == html


def test_data_correction_returns_post_repair_locname_mapping():
    lines = pd.DataFrame(
        [
            ["Röhrsdorf", "Else", "Röhrsdorf - Else", 380.0, 10.0, 0.1, 0.2, 5.0, 1000.0],
            ["Rohrsdorf", "Else", "Rohrsdorf - Else", 380.0, 10.0, 0.1, 0.2, 5.0, 1000.0],
            # the repair adds another "Röhrsdorf" occurrence, which changes the most frequent spelling
            [None, "Else", "Röhrsdorf - Else", 380.0, 10.0, 0.1, 0.2, 5.0, 1000.0],
        ],
        columns=_JAO_LINE_COLUMNS,
    )
    data = {
        "Lines": lines,
        "Tielines": lines.iloc[0:0].copy(),
        "Transformers": pd.DataFrame(columns=_JAO_TRAFO_COLUMNS),
    }

    _, rename_locnames = _data_correction(data, "", 999)

    # the mapping must be derived after the endpoint repair and match the consolidated workbook names
    assert rename_locnames == [("Rohrsdorf", "Röhrsdorf")]
    assert set(data["Lines"][("Substation_1", "Full_name")]) == {"Röhrsdorf"}

    # the same mapping is reused for the decoded HTML labels
    point = {"lng": [1.0, 2.0], "lat": [3.0, 4.0]}
    args = [None] * 7
    args[0] = [[[point]]]
    args[6] = ["EIC Code:<b> EIC1</b><b>NE name: Rohrsdorf - Else: </b>"]
    geo = {"x": {"calls": [{"method": "addPolylines", "args": args}]}}
    html = f'<script type="application/json" data-for="htmlwidget-0123">{json.dumps(geo)}</script>'

    line_geo_data = _parse_html_str(html, rename_locnames)

    assert line_geo_data["name"].unique().tolist() == ["Röhrsdorf - Else:"]


def test_data_correction_does_not_replace_inside_longer_trafo_token():
    lines = pd.DataFrame(
        [
            ["X", "X NORD", "X - X NORD", 380, 1.0, 0.1, 0.2, 5.0, 1000.0],
            ["X.", "X NORD", "X. - X NORD", 380, 1.0, 0.1, 0.2, 5.0, 1000.0],
        ],
        columns=_JAO_LINE_COLUMNS,
    )
    trafos = pd.DataFrame(
        [
            [np.nan, np.nan, 1000.0, 380.0, 110.0, "TR X.NORD", "0;0", 0.0, 0.0],
            [np.nan, np.nan, 1000.0, 380.0, 110.0, "TR X.", "0;0", 0.0, 0.0],
        ],
        columns=_JAO_TRAFO_COLUMNS,
    )
    data = {"Lines": lines, "Tielines": lines.iloc[0:0].copy(), "Transformers": trafos}

    _, rename_locnames = _data_correction(data, "", 999)

    assert ("X.", "X") in rename_locnames
    locations = data["Transformers"][("Location", "Full Name")]
    # the mapping may not rewrite the station token inside a longer transformer name
    assert locations.tolist() == ["TR X.NORD", "TR X"]


def test_data_correction_normalizes_documented_header_variants():
    cols = pd.MultiIndex.from_tuples(
        [
            ("Substation_1", "Full_name"),
            ("Substation_2", "Full_name"),
            ("Unnamed: 0", "NE_name"),
            ("Unnamed: 1", "Voltage_level(kV)"),
            ("Electrical Parameters", "Length_(km)"),
            ("Electrical Parameters", "Resistance_R(ohm)"),
            ("Electrical Parameters", "Reactance_X(ohm)"),
            ("Electrical Parameters", "Susceptance_B(μS)"),
            ("Maximum Current Imax (A)", "Fixed"),
        ]
    )
    lines = pd.DataFrame([["A", "B", "A-B", 380, 10.0, "0,1", "0,2", 5.0, 1000.0]], columns=cols)
    data = {
        "Lines": lines,
        "Tielines": lines.copy(),
        "Transformers": pd.DataFrame(columns=_JAO_TRAFO_COLUMNS),
    }

    _data_correction(data, "", 999)

    for key in ("Lines", "Tielines"):
        assert ("Electrical Parameters", "Resistance_R(Ω)") in data[key].columns
        assert ("Electrical Parameters", "Reactance_X(Ω)") in data[key].columns
        assert data[key][("Electrical Parameters", "Resistance_R(Ω)")].iat[0] == pytest.approx(0.1)
        assert data[key][("Electrical Parameters", "Reactance_X(Ω)")].iat[0] == pytest.approx(0.2)


def test_data_correction_keeps_transformers_with_only_impedance_data():
    cols = pd.MultiIndex.from_tuples(
        list(_JAO_TRAFO_COLUMNS)
        + [
            ("Electrical Parameters (primary) at neutral tap", "Resistance_R(Ω)"),
            ("Electrical Parameters (primary) at neutral tap", "Reactance_X(Ω)"),
        ]
    )
    trafos = pd.DataFrame(
        [[np.nan, np.nan, np.nan, np.nan, np.nan, "TR R only", np.nan, 0.0, 0.0, 0.1, 0.2]],
        columns=cols,
    )
    data = {
        "Lines": pd.DataFrame(columns=_JAO_LINE_COLUMNS),
        "Tielines": pd.DataFrame(columns=_JAO_LINE_COLUMNS),
        "Transformers": trafos,
    }

    _data_correction(data, "", 999)

    # impedance data is electrical data -> keep the row so the winding voltage validator reports it
    assert list(data["Transformers"].index) == [0]
    bus_idx = pd.Series([0], index=pd.MultiIndex.from_arrays([["A"], [380.0]], names=["name", "vn_kv"]))
    with pytest.raises(ValueError, match=r"\[0\]"):
        _get_transformer_voltages(data, bus_idx)


def test_repair_incomplete_line_data_excludes_other_endpoint_span():
    cols = pd.MultiIndex.from_tuples(
        [
            (None, "NE_name"),
            (None, "Voltage_level(kV)"),
            ("Substation_1", "Full_name"),
            ("Substation_2", "Full_name"),
            ("Electrical Parameters", "Length_(km)"),
            ("Electrical Parameters", "Resistance_R(Ω)"),
            ("Electrical Parameters", "Reactance_X(Ω)"),
            ("Electrical Parameters", "Susceptance_B(μS)"),
        ]
    )
    lines = pd.DataFrame(
        [
            ["Belle - Else", 380.0, "Belle", "Else", 10.0, 0.1, 0.2, 5.0],
            ["Belle-Epine - Else", 380.0, "Belle-Epine", "Else", 10.0, 0.1, 0.2, 5.0],
            # "Belle" also matches inside the other endpoint's span and must not become a candidate
            ["Belle-Epine - Else", 380.0, None, "Belle-Epine", 10.0, 0.1, 0.2, 5.0],
        ],
        columns=cols,
    )
    data = {"Lines": lines, "Tielines": lines.iloc[0:0].copy()}

    _repair_incomplete_line_data(data)

    assert data["Lines"].at[2, ("Substation_1", "Full_name")] == "Else"


def test_repair_incomplete_line_data_preserves_longer_station_span():
    cols = pd.MultiIndex.from_tuples(
        [
            (None, "NE_name"),
            (None, "Voltage_level(kV)"),
            ("Substation_1", "Full_name"),
            ("Substation_2", "Full_name"),
            ("Electrical Parameters", "Length_(km)"),
            ("Electrical Parameters", "Resistance_R(Ω)"),
            ("Electrical Parameters", "Reactance_X(Ω)"),
            ("Electrical Parameters", "Susceptance_B(μS)"),
        ]
    )
    lines = pd.DataFrame(
        [
            ["Belle - Else", 380.0, "Belle", "Else", 10.0, 0.1, 0.2, 5.0],
            ["Belle-Epine - Else", 380.0, "Belle-Epine", "Else", 10.0, 0.1, 0.2, 5.0],
            ["Epine - X", 380.0, "Epine", "X", 10.0, 0.1, 0.2, 5.0],
            # "Belle" is a station on its own here, so the "Belle" inside "Belle-Epine" must be kept
            ["Belle - Belle-Epine", 380.0, None, "Belle", 10.0, 0.1, 0.2, 5.0],
        ],
        columns=cols,
    )
    data = {"Lines": lines, "Tielines": lines.iloc[0:0].copy()}

    _repair_incomplete_line_data(data)

    # the longer station must win instead of the shorter fragment "Epine"
    assert data["Lines"].at[3, ("Substation_1", "Full_name")] == "Belle-Epine"


def test_repair_incomplete_line_data_leaves_ambiguous_fragment_unresolved():
    cols = pd.MultiIndex.from_tuples(
        [
            (None, "NE_name"),
            (None, "Voltage_level(kV)"),
            ("Substation_1", "Full_name"),
            ("Substation_2", "Full_name"),
            ("Electrical Parameters", "Length_(km)"),
            ("Electrical Parameters", "Resistance_R(Ω)"),
            ("Electrical Parameters", "Reactance_X(Ω)"),
            ("Electrical Parameters", "Susceptance_B(μS)"),
        ]
    )
    lines = pd.DataFrame(
        [
            ["Belle - Else", 380.0, "Belle", "Else", 10.0, 0.1, 0.2, 5.0],
            ["Epine - Else", 380.0, "Epine", "Else", 10.0, 0.1, 0.2, 5.0],
            # "Belle-Epine" is not a known location, so the populated "Belle" cannot be located
            # unambiguously and must not be resolved to the shorter "Epine"
            ["Belle - Belle-Epine", 380.0, None, "Belle", 10.0, 0.1, 0.2, 5.0],
        ],
        columns=cols,
    )
    data = {"Lines": lines, "Tielines": lines.iloc[0:0].copy()}

    _repair_incomplete_line_data(data)

    assert pd.isna(data["Lines"].at[2, ("Substation_1", "Full_name")])


@pytest.mark.parametrize("min_bus_number", ["max", 6, "unsupplied"])
def test_drop_islanded_grid_groups_empty_net(min_bus_number):
    net = pandapowerNet(name="empty")

    drop_islanded_grid_groups(net, min_bus_number)

    assert len(net.bus) == 0


def test_invent_connections_ignores_out_of_service_copies_for_group_cache():
    net = pandapowerNet(name="jao_grid_groups")
    net.bus["zone"] = "TSO"
    d1h = create_buses(net, 1, 380, name=["DUP1"])[0]
    d2h = create_buses(net, 1, 380, name=["DUP2"])[0]
    d1l = create_buses(net, 1, 220, name=["DUP1"])[0]
    d2l = create_buses(net, 1, 220, name=["DUP2"])[0]
    create_lines(net, d1h, d2h, 1.0, _line_params())
    create_lines(net, d1l, d2l, 1.0, _line_params())
    t_h = create_buses(net, 1, 380, name=["T_H"])[0]
    t_l = create_buses(net, 1, 220, name=["T_L"])[0]
    create_transformers_from_parameters(net, [t_h], [t_l], 1.0, 380, 220, 0, 1, 0, 0, 0, in_service=False)

    _invent_connections_between_grid_groups(net, minimal_trafo_invention=True)

    # the out-of-service copies do not connect the groups, so both locations must be processed
    assert len(net.trafo) == 3
    assert not net.trafo.in_service.any()


@pytest.mark.parametrize(
    "du, dphi, expected_type, expected_percent",
    [
        (2.0, 0.0, "Ratio", 2.0),
        (0.0, 0.0, None, 0.0),
        (1e-12, 1.794, "Ideal", 0.0),
    ],
)
def test_from_jao_sets_tap_changer_type_by_regulation(tmp_path, du, dphi, expected_type, expected_percent):
    testfile = os.path.join(test_path, "converter", "jao_testfiles", "testfile.xlsx")
    modified_testfile = tmp_path / f"testfile_tap_{du}_{dphi}.xlsx"
    workbook = load_workbook(testfile)
    sheet = workbook["Transformers"]
    du_column = next(cell.column for cell in sheet[2] if "Phase Regulation" in str(cell.value))
    dphi_column = next(cell.column for cell in sheet[2] if "Angle Regulation" in str(cell.value))
    sheet.cell(row=3, column=du_column).value = du
    sheet.cell(row=3, column=dphi_column).value = dphi
    workbook.save(modified_testfile)

    net = from_jao(str(modified_testfile), None, False)

    if expected_type is None:
        assert "tap_changer_type" not in net.trafo.columns or pd.isna(net.trafo.tap_changer_type.iat[0])
    else:
        assert net.trafo.tap_changer_type.iat[0] == expected_type
    if expected_type == "Ideal":
        assert net.trafo.tap_step_percent.iat[0] == 0
    else:
        assert net.trafo.tap_step_percent.iat[0] == pytest.approx(expected_percent)


def test_find_trafo_locations_voltage_aware_uses_full_joined_base():
    resolved = _find_trafo_locations(pd.Series(["TR New York 380 kV TR 404"]), ["New York 380"])

    assert resolved.tolist() == ["New York 380"]


def test_allocate_trafos_avoids_generated_name_collision():
    cols = pd.MultiIndex.from_tuples([("Location", "Full Name"), ("Location", "TSO")])
    data = {"Transformers": pd.DataFrame([[np.nan, "A"]], columns=cols)}
    net = pandapowerNet(name="jao_trafos")
    create_buses(net, 1, 220, name=["Transformer 0"])

    _allocate_trafos_to_buses_and_create_buses(net, data, _get_bus_idx(net), np.array([380.0]), np.array([220.0]))

    # the generated label must not collide with the explicit location name
    assert list(net.bus.name).count("Transformer 0") == 1
    assert "Transformer 0 (2)" in set(net.bus.name)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
