# test_ext_grid.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus, create_ext_grid
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network

from pandapower.test.network_schema.elements.helper import (
    strings,
    bools,
    not_strings_list,
    not_floats_list,
    not_boolean_list,
    positiv_ints_plus_zero,
    positiv_floats,
    positiv_floats_plus_zero,
    negativ_floats,
    negativ_floats_plus_zero,
    not_ints_list,
    negativ_ints,
    all_allowed_floats,
)


class TestExtGridRequiredFields:
    """Tests for required ext_grid fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["vm_pu"], positiv_floats),
                itertools.product(["va_degree"], all_allowed_floats),
                itertools.product(["slack_weight"], all_allowed_floats),
                itertools.product(["in_service"], bools),
                itertools.product(["controllable"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = pandapowerNet(name="test_valid_required_values")
        create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1
        create_bus(net, 0.4, index=42)

        create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0.0, in_service=True)

        net.ext_grid[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["vm_pu"], [float(np.nan), pd.NA, *negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["va_degree"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["slack_weight"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, *not_boolean_list]),
                itertools.product(["controllable"], [float(np.nan), pd.NA, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = pandapowerNet(name="test_invalid_required_values")
        create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1

        create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0.0, in_service=True)

        net.ext_grid[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestExtGridOptionalFields:
    """Tests for optional ext_grid fields, including group dependencies (opf, sc, 3ph)"""

    def test_all_optional_fields_valid(self):
        """Test: ext_grid with every optional field is valid and dependencies satisfied"""
        net = pandapowerNet(name="test_all_optional_fields_valid")
        b0 = create_bus(net, 0.4)

        create_ext_grid(
            net,
            bus=b0,
            vm_pu=1.02,
            va_degree=0.0,
            in_service=True,
            # OPF group
            max_p_mw=100.0,
            min_p_mw=-100.0,
            max_q_mvar=50.0,
            min_q_mvar=-50.0,
            # SC group
            s_sc_max_mva=1000.0,
            s_sc_min_mva=500.0,
            # 3PH group (also part of SC group)
            rx_max=0.5,
            rx_min=0.1,
            r0x0_max=0.2,
            x0x_max=3.0,
            name="ext grid 1",
        )
        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: ext_grid with optional fields including nulls, with dependencies respected"""
        net = pandapowerNet(name="test_optional_fields_with_nulls")
        b0 = create_bus(net, 0.4)

        # Row 1: OPF present, SC/3PH absent
        create_ext_grid(
            net,
            bus=b0,
            vm_pu=1.01,
            va_degree=0.0,
            in_service=True,
            max_p_mw=80.0,
            min_p_mw=-80.0,
            max_q_mvar=40.0,
            min_q_mvar=-40.0,
            name="alpha",
        )
        # Row 2: SC + 3PH present, OPF absent
        create_ext_grid(
            net,
            bus=b0,
            vm_pu=1.03,
            va_degree=0.0,
            in_service=True,
            s_sc_max_mva=1100.0,
            s_sc_min_mva=600.0,
            rx_max=0.6,
            rx_min=0.2,
            r0x0_max=0.3,
            x0x_max=2.5,
            name=None,
        )
        # Row 3: no optional groups, only name
        create_ext_grid(
            net,
            bus=b0,
            vm_pu=1.0,
            va_degree=0.0,
            in_service=True,
            name="gamma",
        )

        # Set name column with nulls
        net.ext_grid["name"] = pd.Series(["alpha", pd.NA, "gamma"], dtype=pd.StringDtype())

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["max_p_mw"], all_allowed_floats),
                itertools.product(["min_p_mw"], all_allowed_floats),
                itertools.product(["max_q_mvar"], all_allowed_floats),
                itertools.product(["min_q_mvar"], all_allowed_floats),
                itertools.product(["s_sc_max_mva"], positiv_floats),
                itertools.product(["s_sc_min_mva"], positiv_floats),
                itertools.product(["rx_max"], positiv_floats_plus_zero),
                itertools.product(["rx_min"], positiv_floats_plus_zero),
                itertools.product(["r0x0_max"], positiv_floats_plus_zero),
                itertools.product(["x0x_max"], positiv_floats_plus_zero),

                itertools.product(["origin_id"], [pd.NA, *strings]),
                itertools.product(["origin_class"], [pd.NA, *strings]),
                itertools.product(["substation"], [pd.NA, *strings]),
                itertools.product(["terminal"], [pd.NA, *strings]),
                itertools.product(["description"], [pd.NA, *strings]),
                itertools.product(["RegulatingControl.mode"], [pd.NA, *strings]),
                itertools.product(["RegulatingControl.targetValue"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["RegulatingControl.enabled"], [pd.NA, *bools]),  # Note: bools, not floats
                itertools.product(["referencePriority"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["p_mw"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["q_mvar"], [float(np.nan), *all_allowed_floats]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = pandapowerNet(name="test_valid_optional_values")
        b0 = create_bus(net, 0.4)

        e0 = create_ext_grid(
            net,
            bus=b0,
            vm_pu=1.02,
            va_degree=0.0,
            in_service=True,
            # OPF group
            max_p_mw=100.0,
            min_p_mw=-100.0,
            max_q_mvar=50.0,
            min_q_mvar=-50.0,
            # SC group
            s_sc_max_mva=1000.0,
            s_sc_min_mva=500.0,
            # 3PH group (also part of SC group)
            rx_max=0.5,
            rx_min=0.1,
            r0x0_max=0.2,
            x0x_max=3.0,
            name="ext grid 0",
        )
        create_ext_grid(
            net,
            bus=b0,
            vm_pu=1.02,
            va_degree=0.0,
            in_service=True,
            # OPF group
            max_p_mw=np.nan,
            min_p_mw=np.nan,
            max_q_mvar=np.nan,
            min_q_mvar=np.nan,
            # SC group
            s_sc_max_mva=np.nan,
            s_sc_min_mva=np.nan,
            # 3PH group (also part of SC group)
            rx_max=np.nan,
            rx_min=np.nan,
            r0x0_max=np.nan,
            x0x_max=np.nan,
            name="ext grid 1",
        )

        net.ext_grid.loc[e0, parameter] = valid_value

        # Initialize CIM columns with proper dtype first
        if parameter in ["origin_id", "origin_class", "substation", "terminal", "description",
                         "RegulatingControl.mode"]:
            # Initialize all CIM columns with default values first
            for col in ["origin_id", "origin_class", "substation", "terminal", "description", "RegulatingControl.mode"]:
                if col not in net.ext_grid.columns:
                    net.ext_grid[col] = pd.Series(["test"], dtype=pd.StringDtype())

            # Then set the specific parameter value
            if pd.isna(valid_value):
                net.ext_grid[parameter] = pd.Series([pd.NA, "test"], dtype=pd.StringDtype())
            else:
                net.ext_grid[parameter] = pd.Series([valid_value, "test"], dtype=pd.StringDtype())
        elif parameter == "RegulatingControl.enabled":
            if parameter not in net.ext_grid.columns:
                net.ext_grid[parameter] = pd.Series([True], dtype=pd.BooleanDtype())
            if pd.isna(valid_value):
                net.ext_grid[parameter] = pd.Series([pd.NA, True], dtype=pd.BooleanDtype())
            else:
                net.ext_grid[parameter] = pd.Series([valid_value, True], dtype=pd.BooleanDtype())
        else:
            net.ext_grid.loc[e0, parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["max_p_mw"], not_floats_list),
                itertools.product(["min_p_mw"], not_floats_list),
                itertools.product(["max_q_mvar"], not_floats_list),
                itertools.product(["min_q_mvar"], not_floats_list),
                itertools.product(["s_sc_max_mva"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["s_sc_min_mva"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["rx_max"], [*negativ_floats, *not_floats_list]),
                itertools.product(["rx_min"], [*negativ_floats, *not_floats_list]),
                itertools.product(["r0x0_max"], [*negativ_floats, *not_floats_list]),
                itertools.product(["x0x_max"], [*negativ_floats, *not_floats_list]),

                itertools.product(["origin_id"], not_strings_list),
                itertools.product(["origin_class"], not_strings_list),
                itertools.product(["substation"], not_strings_list),
                itertools.product(["terminal"], not_strings_list),
                itertools.product(["description"], not_strings_list),
                itertools.product(["RegulatingControl.mode"], not_strings_list),
                itertools.product(["RegulatingControl.targetValue"], not_floats_list),
                itertools.product(["RegulatingControl.enabled"], not_boolean_list),  # Note: bools, not floats
                itertools.product(["referencePriority"], not_floats_list),
                itertools.product(["p_mw"], not_floats_list),
                itertools.product(["q_mvar"], not_floats_list),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are not accepted"""
        net = pandapowerNet(name="test_invalid_optional_values")
        b0 = create_bus(net, 0.4)
        create_ext_grid(
            net,
            bus=b0,
            vm_pu=1.02,
            va_degree=0.0,
            in_service=True,
            # OPF group
            max_p_mw=100.0,
            min_p_mw=-100.0,
            max_q_mvar=50.0,
            min_q_mvar=-50.0,
            # SC group
            s_sc_max_mva=1000.0,
            s_sc_min_mva=500.0,
            # 3PH group (also part of SC group)
            rx_max=0.5,
            rx_min=0.1,
            r0x0_max=0.2,
            x0x_max=3.0,
            name="ext grid 1",
        )
        net.ext_grid[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestExtGridForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """Test: bus FK must reference an existing bus index"""
        net = pandapowerNet(name="test_invalid_bus_index")
        b0 = create_bus(net, 0.4)

        create_ext_grid(net, bus=b0, vm_pu=1.0, va_degree=0.0, in_service=True)

        net.ext_grid["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FK works with non-sequential bus indices"""
        net = pandapowerNet(name="test_valid_bus_index_non_sequential")
        create_bus(net, 0.4, index=10)
        create_bus(net, 0.4, index=42)
        create_bus(net, 0.4, index=100)

        create_ext_grid(net, bus=10, vm_pu=1.0, va_degree=0.0, in_service=True)
        create_ext_grid(net, bus=42, vm_pu=1.02, va_degree=0.0, in_service=True)
        create_ext_grid(net, bus=100, vm_pu=0.98, va_degree=0.0, in_service=False)

        validate_network(net)


class TestExtGridResults:
    """Tests for ext_grid results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_ext_grid_result_totals(self):
        """Test: aggregated p_mw / q_mvar results are consistent"""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_ext_grid_3ph_results(self):
        """Test: 3-phase results contain valid values per phase"""
        pass
