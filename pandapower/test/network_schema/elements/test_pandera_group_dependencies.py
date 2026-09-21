# test_pandera_group_dependencies.py
"""Tests for group dependency validation (opf, sc, qcc, q_lim_enforced)

Group dependencies work as follows:
- Columns with a specific group tag (e.g., "opf", "sc") are optional by default
- When validate_network(net, "group_name") is called, those columns become required
- All columns in the group must be present together for validation to pass
- Use add_tag_group() helper to add all group columns to a network element
"""

import pandas as pd
import pandera as pa
import pytest

from pandapower.create import (
    create_bus,
    create_bus_dc,
    create_gen,
    create_ext_grid,
    create_sgen,
    create_storage,
    create_line,
    create_line_dc_from_parameters,
    create_shunt,
    create_transformer,
    create_transformer3w,
    create_dcline,
    create_motor,
)
from pandapower.create._utils import add_tag_group
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network


class TestOpfGroupDependency:
    """Tests for OPF group dependencies.

    OPF group columns are optional by default but become required when
    validate_network(net, "opf") is called.
    """

    @pytest.mark.parametrize(
        "element,create_func,create_kwargs",
        [
            ("gen", create_gen, {"bus": 0, "p_mw": 1.0, "vm_pu": 1.0, "in_service": True}),
            ("ext_grid", create_ext_grid, {"bus": 0, "vm_pu": 1.0}),
            ("sgen", create_sgen, {"bus": 0, "p_mw": 1.0, "in_service": True}),
            ("storage", create_storage, {"bus": 0, "p_mw": 1.0, "max_e_mwh": 100.0, "in_service": True}),
            ("shunt", create_shunt, {"bus": 0, "q_mvar": 1.0, "in_service": True}),
            ("bus", create_bus, {"vn_kv": 0.4}),
            ("bus_dc", create_bus_dc, {"vn_kv": 0.4}),
            ("line", create_line, {"from_bus": 0, "to_bus": 1, "length_km": 1.0, "std_type": "NAYY 4x50 SE"}),
            ("line_dc", create_line_dc_from_parameters, {"from_bus_dc": 0, "to_bus_dc": 1, "length_km": 1.0, "r_ohm_per_km": 0.1, "max_i_ka": 0.5}),
            ("trafo", create_transformer, {"hv_bus": 0, "lv_bus": 1, "std_type": "25 MVA 110/20 kV"}),
            ("trafo3w", create_transformer3w, {"hv_bus": 0, "mv_bus": 1, "lv_bus": 2, "std_type": "63/25/38 MVA 110/20/10 kV"}),
            ("dcline", create_dcline, {"from_bus": 0, "to_bus": 1, "p_mw": 100.0, "loss_percent": 1.0, "loss_mw": 1.0, "vm_from_pu": 1.0, "vm_to_pu": 1.0}),
        ],
    )
    def test_opf_group_all_absent_valid(self, element, create_func, create_kwargs):
        """Test: OPF group columns absent is valid (base validation)"""
        net = pandapowerNet(name="test_opf_group_all_absent_valid")
        create_bus(net, 0.4)

        if element == "line":
            create_bus(net, 0.4)
            create_bus(net, 0.4)

        if element == "line_dc":
            create_bus_dc(net, 0.4)
            create_bus_dc(net, 0.4)

        if element in ["trafo", "trafo3w"]:
            create_bus(net, 20.0)
            create_bus(net, 0.4)
            if element == "trafo3w":
                create_bus(net, 10.0)

        if element == "dcline":
            create_bus(net, 110.0)
            create_bus(net, 110.0)

        create_func(net, **create_kwargs)
        validate_network(net)

    @pytest.mark.parametrize(
        "element,create_func,create_kwargs",
        [
            ("gen", create_gen, {"bus": 0, "p_mw": 1.0, "vm_pu": 1.0, "in_service": True, "max_p_mw": 100.0, "min_p_mw": 0.0, "max_q_mvar": 50.0, "min_q_mvar": -50.0, "controllable": False}),
            ("ext_grid", create_ext_grid, {"bus": 0, "vm_pu": 1.0, "max_p_mw": 100.0, "min_p_mw": 0.0, "max_q_mvar": 50.0, "min_q_mvar": -50.0}),
            ("sgen", create_sgen, {"bus": 0, "p_mw": 1.0, "in_service": True, "max_p_mw": 100.0, "min_p_mw": 0.0, "max_vm_pu": 1.1, "min_vm_pu": 0.9, "max_q_mvar": 50.0, "min_q_mvar": -50.0}),
            ("storage", create_storage, {"bus": 0, "p_mw": 1.0, "max_e_mwh": 100.0, "in_service": True, "max_p_mw": 100.0, "min_p_mw": 0.0, "max_q_mvar": 50.0, "min_q_mvar": -50.0, "controllable": False}),
            ("shunt", create_shunt, {"bus": 0, "q_mvar": 1.0, "in_service": True, "step": 1}),
            ("bus", create_bus, {"vn_kv": 0.4, "max_vm_pu": 1.1, "min_vm_pu": 0.9}),
            ("bus_dc", create_bus_dc, {"vn_kv": 0.4, "max_vm_pu": 1.1, "min_vm_pu": 0.9}),
            ("line", create_line, {"from_bus": 0, "to_bus": 1, "length_km": 1.0, "std_type": "NAYY 4x50 SE", "max_loading_percent": 100.0}),
            ("line_dc", create_line_dc_from_parameters, {"from_bus_dc": 0, "to_bus_dc": 1, "length_km": 1.0, "r_ohm_per_km": 0.1, "max_i_ka": 0.5, "max_loading_percent": 100.0}),
            ("trafo", create_transformer, {"hv_bus": 0, "lv_bus": 1, "std_type": "25 MVA 110/20 kV", "max_loading_percent": 100.0}),
            ("trafo3w", create_transformer3w, {"hv_bus": 0, "mv_bus": 1, "lv_bus": 2, "std_type": "63/25/38 MVA 110/20/10 kV", "max_loading_percent": 100.0}),
            ("dcline", create_dcline, {"from_bus": 0, "to_bus": 1, "p_mw": 100.0, "loss_percent": 1.0, "loss_mw": 1.0, "vm_from_pu": 1.0, "vm_to_pu": 1.0, "max_p_mw": 200.0, "min_q_from_mvar": -50.0, "max_q_from_mvar": 50.0, "min_q_to_mvar": -50.0, "max_q_to_mvar": 50.0}),
        ],
    )
    def test_opf_group_all_present_valid(self, element, create_func, create_kwargs):
        """Test: OPF group columns all present is valid"""
        net = pandapowerNet(name="test_opf_group_all_present_valid")
        create_bus(net, 0.4, max_vm_pu=1.1, min_vm_pu=0.9)

        if element in ["line", "trafo", "trafo3w", "dcline"]:
            create_bus(net, 0.4)
        if element == "trafo3w":
            create_bus(net, 0.4)
        if element == "line_dc":
            create_bus_dc(net, 0.4)
            create_bus_dc(net, 0.4)
        if element == "dcline":
            create_bus(net, 110.0)
            create_bus(net, 110.0)

        create_func(net, **create_kwargs)
        add_tag_group(net, "opf")
        validate_network(net, groups_to_validate="opf")

    @pytest.mark.parametrize(
        "element,create_func,create_kwargs,partial_column",
        [
            ("gen", create_gen, {"bus": 0, "p_mw": 1.0, "vm_pu": 1.0, "in_service": True}, "max_p_mw"),
            ("ext_grid", create_ext_grid, {"bus": 0, "vm_pu": 1.0}, "max_p_mw"),
            ("sgen", create_sgen, {"bus": 0, "p_mw": 1.0, "in_service": True}, "max_p_mw"),
            ("storage", create_storage, {"bus": 0, "p_mw": 1.0, "max_e_mwh": 100.0, "in_service": True}, "max_p_mw"),
            ("shunt", create_shunt, {"bus": 0, "q_mvar": 1.0, "in_service": True}, "step"),
            ("bus", create_bus, {"vn_kv": 0.4}, "min_vm_pu"),
            ("bus_dc", create_bus_dc, {"vn_kv": 0.4}, "min_vm_pu"),
            ("line", create_line, {"from_bus": 0, "to_bus": 1, "length_km": 1.0, "std_type": "NAYY 4x50 SE"}, "max_loading_percent"),
            ("line_dc", create_line_dc_from_parameters, {"from_bus_dc": 0, "to_bus_dc": 1, "length_km": 1.0, "r_ohm_per_km": 0.1, "max_i_ka": 0.5}, "max_loading_percent"),
            ("trafo", create_transformer, {"hv_bus": 0, "lv_bus": 1, "std_type": "25 MVA 110/20 kV"}, "max_loading_percent"),
            ("trafo3w", create_transformer3w, {"hv_bus": 0, "mv_bus": 1, "lv_bus": 2, "std_type": "63/25/38 MVA 110/20/10 kV"}, "max_loading_percent"),
            ("dcline", create_dcline, {"from_bus": 0, "to_bus": 1, "p_mw": 100.0, "loss_percent": 1.0, "loss_mw": 1.0, "vm_from_pu": 1.0, "vm_to_pu": 1.0}, "max_p_mw"),
        ],
    )
    def test_opf_group_partial_invalid(self, element, create_func, create_kwargs, partial_column):
        """Test: OPF group columns partial (not all) is invalid"""
        net = pandapowerNet(name="test_opf_group_partial_invalid")
        create_bus(net, 0.4)

        if element == "line":
            create_bus(net, 0.4)

        if element == "line_dc":
            create_bus_dc(net, 0.4)
            create_bus_dc(net, 0.4)

        if element in ["trafo", "trafo3w"]:
            create_bus(net, 20.0)
            create_bus(net, 0.4)
            if element == "trafo3w":
                create_bus(net, 10.0)

        if element == "dcline":
            create_bus(net, 110.0)
            create_bus(net, 110.0)

        create_func(net, **create_kwargs)
        net[element][partial_column] = 100.0
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net, groups_to_validate="opf")


class TestScGroupDependency:
    """Tests for SC (short-circuit) group dependencies.

    SC group columns are optional by default but become required when
    validate_network(net, "sc") is called.
    """

    @pytest.mark.parametrize(
        "element,create_func,create_kwargs",
        [
            ("gen", create_gen, {"bus": 0, "p_mw": 1.0, "vm_pu": 1.0, "in_service": True}),
            ("ext_grid", create_ext_grid, {"bus": 0, "vm_pu": 1.0}),
            ("sgen", create_sgen, {"bus": 0, "p_mw": 1.0, "in_service": True}),
            ("line", create_line, {"from_bus": 0, "to_bus": 1, "length_km": 1.0, "std_type": "NAYY 4x50 SE"}),
            ("trafo", create_transformer, {"hv_bus": 0, "lv_bus": 1, "std_type": "25 MVA 110/20 kV"}),
            ("trafo3w", create_transformer3w, {"hv_bus": 0, "mv_bus": 1, "lv_bus": 2, "std_type": "63/25/38 MVA 110/20/10 kV"}),
            ("motor", create_motor, {"bus": 0, "pn_mech_mw": 1.0, "cos_phi": 0.9}),
        ],
    )
    def test_sc_group_all_absent_valid(self, element, create_func, create_kwargs):
        """Test: SC group columns absent is valid (base validation)"""
        net = pandapowerNet(name="test_sc_group_all_absent_valid")
        create_bus(net, 0.4)

        if element == "line":
            create_bus(net, 0.4)
            create_bus(net, 0.4)

        if element in ["trafo", "trafo3w"]:
            create_bus(net, 20.0)
            create_bus(net, 0.4)
            if element == "trafo3w":
                create_bus(net, 10.0)

        create_func(net, **create_kwargs)
        validate_network(net)

    @pytest.mark.parametrize(
        "element,create_func,create_kwargs",
        [
            ("gen", create_gen, {"bus": 0, "p_mw": 1.0, "vm_pu": 1.0, "in_service": True}),
            ("ext_grid", create_ext_grid, {"bus": 0, "vm_pu": 1.0}),
            ("sgen", create_sgen, {"bus": 0, "p_mw": 1.0, "in_service": True}),
            ("line", create_line, {"from_bus": 0, "to_bus": 1, "length_km": 1.0, "std_type": "NAYY 4x50 SE"}),
            ("motor", create_motor, {"bus": 0, "pn_mech_mw": 1.0, "cos_phi": 0.9, "cos_phi_n": 0.85, "efficiency_n_percent": 95.0, "lrc_pu": 6.0, "rx": 0.4, "vn_kv": 0.4}),
        ],
    )
    def test_sc_group_all_present_valid(self, element, create_func, create_kwargs):
        """Test: SC group columns all present is valid"""
        net = pandapowerNet(name="test_sc_group_all_present_valid")
        create_bus(net, 0.4)

        if element == "line":
            create_bus(net, 0.4)
            create_bus(net, 0.4)

        create_func(net, **create_kwargs)
        add_tag_group(net, "sc")
        validate_network(net, groups_to_validate="sc")

    @pytest.mark.parametrize(
        "element,create_func,create_kwargs,partial_column",
        [
            ("gen", create_gen, {"bus": 0, "p_mw": 1.0, "vm_pu": 1.0, "in_service": True}, "vn_kv"),
            ("ext_grid", create_ext_grid, {"bus": 0, "vm_pu": 1.0}, "s_sc_max_mva"),
            ("sgen", create_sgen, {"bus": 0, "p_mw": 1.0, "in_service": True}, "xn_ohm"),
            ("line", create_line, {"from_bus": 0, "to_bus": 1, "length_km": 1.0, "std_type": "NAYY 4x50 SE"}, "c_nf_per_km"),
            ("trafo", create_transformer, {"hv_bus": 0, "lv_bus": 1, "std_type": "25 MVA 110/20 kV"}, "vk_percent"),
            ("trafo3w", create_transformer3w, {"hv_bus": 0, "mv_bus": 1, "lv_bus": 2, "std_type": "63/25/38 MVA 110/20/10 kV"}, "vk_hv_percent"),
            ("motor", create_motor, {"bus": 0, "pn_mech_mw": 1.0, "cos_phi": 0.9}, "vn_kv"),
        ],
    )
    def test_sc_group_partial_invalid(self, element, create_func, create_kwargs, partial_column):
        """Test: SC group columns partial (not all) is invalid"""
        net = pandapowerNet(name="test_sc_group_partial_invalid")
        create_bus(net, 0.4)

        if element == "line":
            create_bus(net, 0.4)
            create_bus(net, 0.4)

        if element in ["trafo", "trafo3w"]:
            create_bus(net, 20.0)
            create_bus(net, 0.4)
            if element == "trafo3w":
                create_bus(net, 10.0)

        create_func(net, **create_kwargs)
        net[element][partial_column] = 1.0
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net, groups_to_validate="sc")


class TestQccGroupDependency:
    """Tests for QCC (reactive capability curve) group dependencies.

    QCC group columns are optional by default but become required when
    validate_network(net, "qcc") is called.
    """

    @pytest.mark.parametrize(
        "element,create_func,create_kwargs",
        [
            ("gen", create_gen, {"bus": 0, "p_mw": 1.0, "vm_pu": 1.0, "in_service": True}),
            ("sgen", create_sgen, {"bus": 0, "p_mw": 1.0, "in_service": True}),
        ],
    )
    def test_qcc_group_all_absent_valid(self, element, create_func, create_kwargs):
        """Test: QCC group columns absent is valid (base validation)"""
        net = pandapowerNet(name="test_qcc_group_all_absent_valid")
        create_bus(net, 0.4)

        create_func(net, **create_kwargs)
        validate_network(net)

    @pytest.mark.parametrize(
        "element,create_func,create_kwargs",
        [
            ("gen", create_gen, {"bus": 0, "p_mw": 1.0, "vm_pu": 1.0, "in_service": True}),
            ("sgen", create_sgen, {"bus": 0, "p_mw": 1.0, "in_service": True}),
        ],
    )
    def test_qcc_group_all_present_valid(self, element, create_func, create_kwargs):
        """Test: QCC group columns all present is valid"""
        net = pandapowerNet(name="test_qcc_group_all_present_valid")
        create_bus(net, 0.4)

        create_func(net, **create_kwargs)
        add_tag_group(net, "qcc")
        validate_network(net, groups_to_validate="qcc")

    @pytest.mark.parametrize(
        "element,create_func,create_kwargs,partial_column",
        [
            ("gen", create_gen, {"bus": 0, "p_mw": 1.0, "vm_pu": 1.0, "in_service": True}, "id_q_capability_characteristic"),
            ("sgen", create_sgen, {"bus": 0, "p_mw": 1.0, "in_service": True}, "id_q_capability_characteristic"),
            ("sgen", create_sgen, {"bus": 0, "p_mw": 1.0, "in_service": True}, "curve_style"),
            ("sgen", create_sgen, {"bus": 0, "p_mw": 1.0, "in_service": True}, "reactive_capability_curve"),
        ],
    )
    def test_qcc_group_partial_invalid(self, element, create_func, create_kwargs, partial_column):
        """Test: QCC group columns partial (not all) is invalid"""
        net = pandapowerNet(name="test_qcc_group_partial_invalid")
        create_bus(net, 0.4)

        create_func(net, **create_kwargs)

        if partial_column == "curve_style":
            net[element][partial_column] = pd.Series(["straightLineYValues"], dtype="string")
        elif partial_column == "id_q_capability_characteristic":
            net[element][partial_column] = pd.Series([0], dtype="Int64")
        elif partial_column == "reactive_capability_curve":
            net[element][partial_column] = pd.Series([True], dtype="boolean")
        else:
            net[element][partial_column] = 1.0

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net, groups_to_validate="qcc")


class TestQlimEnforcedGroupDependency:
    """Tests for q_lim_enforced group dependencies.

    This group applies to gen element only - max_q_mvar and min_q_mvar must be
    present together if either is present.
    """

    def test_q_lim_enforced_group_all_absent_valid(self):
        """Test: q_lim_enforced group columns absent is valid"""
        net = pandapowerNet(name="test_q_lim_enforced_all_absent")
        create_bus(net, 0.4)
        create_gen(net, bus=0, p_mw=1.0, vm_pu=1.0, in_service=True)
        validate_network(net)

    def test_q_lim_enforced_group_all_present_valid(self):
        """Test: q_lim_enforced group columns all present is valid"""
        net = pandapowerNet(name="test_q_lim_enforced_all_present")
        create_bus(net, 0.4)
        create_gen(
            net,
            bus=0,
            p_mw=1.0,
            vm_pu=1.0,
            in_service=True,
            max_q_mvar=50.0,
            min_q_mvar=-50.0,
        )
        validate_network(net, groups_to_validate="q_lim_enforced")

    def test_q_lim_enforced_group_partial_invalid(self):
        """Test: q_lim_enforced group columns partial is invalid"""
        net = pandapowerNet(name="test_q_lim_enforced_partial")
        create_bus(net, 0.4)
        create_gen(
            net,
            bus=0,
            p_mw=1.0,
            vm_pu=1.0,
            in_service=True,
            max_q_mvar=50.0,
        )
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net, groups_to_validate="q_lim_enforced")
