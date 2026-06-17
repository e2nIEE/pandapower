# test_pandera_vsc_stacked_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus, create_bus_dc, create_vsc_stacked
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network

from pandapower.test.network_schema.elements.helper import (
    strings,
    bools,
    not_strings_list,
    not_floats_list,
    not_boolean_list,
    positiv_ints_plus_zero,
    not_ints_list,
    negativ_ints,
    all_allowed_floats,
)


class TestVSCSTACKEDRequiredFields:
    """Tests for required vsc_stacked fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["bus_dc_plus"], positiv_ints_plus_zero),
                itertools.product(["bus_dc_minus"], positiv_ints_plus_zero),
                itertools.product(["r_ohm"], all_allowed_floats),
                itertools.product(["x_ohm"], all_allowed_floats),
                itertools.product(["r_dc_ohm"], all_allowed_floats),
                itertools.product(["pl_dc_mw"], all_allowed_floats),
                itertools.product(["control_mode_ac"], ["vm_pu", "q_mvar", "slack", "p_mw"]),
                itertools.product(["control_value_ac"], all_allowed_floats),
                itertools.product(["control_mode_dc"], ["vm_pu", "p_mw"]),
                itertools.product(["control_value_dc"], all_allowed_floats),
                itertools.product(["controllable"], bools),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = pandapowerNet(name="test_valid_required_values")
        create_bus(net, vn_kv=110.0)  # index 0
        create_bus(net, vn_kv=110.0)  # index 1
        create_bus(net, vn_kv=110.0, index=42)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1
        create_bus_dc(net, vn_kv=110.0, index=42)

        create_vsc_stacked(
            net,
            bus=0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
            name="test",
        )

        net.vsc_stacked[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [float(np.nan), pd.NA, None, *negativ_ints, *not_ints_list]),
                itertools.product(["bus_dc_plus"], [float(np.nan), pd.NA, None, *negativ_ints, *not_ints_list]),
                itertools.product(["bus_dc_minus"], [float(np.nan), pd.NA, None, *negativ_ints, *not_ints_list]),
                itertools.product(["r_ohm"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["x_ohm"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["r_dc_ohm"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["pl_dc_mw"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["control_mode_ac"], [float(np.nan), pd.NA, None, *strings, *not_strings_list]),
                itertools.product(["control_value_ac"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["control_mode_dc"], [float(np.nan), pd.NA, None, *strings, *not_strings_list]),
                itertools.product(["control_value_dc"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["controllable"], [float(np.nan), pd.NA, None, *not_boolean_list]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, None, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = pandapowerNet(name="test_invalid_required_values")
        create_bus(net, 110.0)  # index 0
        create_bus(net, 110.0)  # index 1
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1
        create_bus_dc(net, vn_kv=110.0, index=42)
        create_vsc_stacked(
            net,
            bus=0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
        )

        net.vsc_stacked[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestVSCSTACKEDOptionalFields:
    """Tests for optional vsc_stacked fields"""

    def test_all_optional_fields_valid(self):
        """Test: vsc_stacked with all optional fields set is valid"""
        net = pandapowerNet(name="test_all_optional_fields_valid")
        b0 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1

        create_vsc_stacked(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
            name="VSC Stacked A",
        )

        net.vsc_stacked["name"] = net.vsc_stacked["name"].astype("string")
        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: vsc_stacked with optional fields including nulls is valid"""
        net = pandapowerNet(name="test_optional_fields_with_nulls")
        b0 = create_bus(net, 110.0)
        b1 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1

        # Row 1: name set
        create_vsc_stacked(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
            name="hello",
        )
        # Row 2: name None
        create_vsc_stacked(
            net,
            bus=b1,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.2,
            x_ohm=0.3,
            r_dc_ohm=0.1,
            pl_dc_mw=0.5,
            control_mode_ac="q_mvar",
            control_value_ac=0.5,
            control_mode_dc="vm_pu",
            control_value_dc=1.0,
            controllable=False,
            in_service=False,
            name=None,
        )

        net.vsc_stacked["name"] = pd.Series(["V1", pd.NA], dtype=pd.StringDtype())
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = pandapowerNet(name="test_valid_optional_values")
        b0 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1
        create_vsc_stacked(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
            name="initial",
        )

        net.vsc_stacked[parameter] = pd.Series([valid_value], dtype=pd.StringDtype())
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [float(np.nan), *not_strings_list]),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected"""
        net = pandapowerNet(name="test_invalid_optional_values")
        b0 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1
        create_vsc_stacked(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
        )

        net.vsc_stacked[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestVscStackedNullableColumns:
    """Tests for nullable columns - testing NA/NaN acceptance"""

    def test_all_nullable_string_columns_na_valid(self):
        """Test: All nullable string columns can be NA"""
        net = create_empty_network()
        b0 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1

        create_vsc_stacked(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
        )

        # Set all nullable string columns to NA
        net.vsc_stacked["name"] = pd.Series([pd.NA], dtype=pd.StringDtype())

        validate_network(net)

    def test_individual_nullable_string_column_na_valid(self):
        """Test: name column accepts NA individually"""
        net = create_empty_network()
        b0 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1

        create_vsc_stacked(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
        )

        net.vsc_stacked["name"] = pd.Series([pd.NA], dtype=pd.StringDtype())
        validate_network(net)

    def test_mixed_null_and_valid_values_in_rows(self):
        """Test: Multiple rows with mixed NA and valid values"""
        net = create_empty_network()
        b0 = create_bus(net, 110.0)
        b1 = create_bus(net, 110.0)
        b2 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1
        create_bus_dc(net, vn_kv=110.0)  # index 2

        # Row 1: name filled
        create_vsc_stacked(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
            name="VSC A",
        )

        # Row 2: name will be NA
        create_vsc_stacked(
            net,
            bus=b1,
            bus_dc_plus=1,
            bus_dc_minus=2,
            r_ohm=0.2,
            x_ohm=0.3,
            r_dc_ohm=0.1,
            pl_dc_mw=0.5,
            control_mode_ac="q_mvar",
            control_value_ac=0.5,
            control_mode_dc="vm_pu",
            control_value_dc=1.0,
            controllable=False,
            in_service=False,
        )

        # Row 3: name filled again
        create_vsc_stacked(
            net,
            bus=b2,
            bus_dc_plus=0,
            bus_dc_minus=2,
            r_ohm=0.15,
            x_ohm=0.25,
            r_dc_ohm=0.08,
            pl_dc_mw=0.3,
            control_mode_ac="slack",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=5.0,
            controllable=True,
            in_service=True,
            name="VSC C",
        )

        # Set nullable columns with mixed values
        net.vsc_stacked["name"] = pd.Series(["VSC A", pd.NA, "VSC C"], dtype=pd.StringDtype())

        validate_network(net)

    def test_name_column_all_na_multiple_rows_valid(self):
        """Test: name column can be NA for all rows"""
        net = create_empty_network()
        b0 = create_bus(net, 110.0)
        b1 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1

        create_vsc_stacked(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
        )

        create_vsc_stacked(
            net,
            bus=b1,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.2,
            x_ohm=0.3,
            r_dc_ohm=0.1,
            pl_dc_mw=0.5,
            control_mode_ac="q_mvar",
            control_value_ac=0.5,
            control_mode_dc="vm_pu",
            control_value_dc=1.0,
            controllable=False,
            in_service=False,
        )

        # All rows have NA for name
        net.vsc_stacked["name"] = pd.Series([pd.NA, pd.NA], dtype=pd.StringDtype())

        validate_network(net)


class TestVscStackedNullableColumns:
    """Tests for nullable columns - testing NA/NaN acceptance"""

    def test_all_nullable_string_columns_na_valid(self):
        """Test: All nullable string columns can be NA"""
        net = pandapowerNet(name="test_all_nullable_string_columns_na_valid")
        b0 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1

        create_vsc_stacked(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
        )

        # Set all nullable string columns to NA
        net.vsc_stacked["name"] = pd.Series([pd.NA], dtype=pd.StringDtype())

        validate_network(net)

    def test_individual_nullable_string_column_na_valid(self):
        """Test: name column accepts NA individually"""
        net = pandapowerNet(name="test_individual_nullable_string_column_na_valid")
        b0 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1

        create_vsc_stacked(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
        )

        net.vsc_stacked["name"] = pd.Series([pd.NA], dtype=pd.StringDtype())
        validate_network(net)

    def test_mixed_null_and_valid_values_in_rows(self):
        """Test: Multiple rows with mixed NA and valid values"""
        net = pandapowerNet(name="test_mixed_null_and_valid_values_in_rows")
        b0 = create_bus(net, 110.0)
        b1 = create_bus(net, 110.0)
        b2 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1
        create_bus_dc(net, vn_kv=110.0)  # index 2

        # Row 1: name filled
        create_vsc_stacked(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
            name="VSC A",
        )

        # Row 2: name will be NA
        create_vsc_stacked(
            net,
            bus=b1,
            bus_dc_plus=1,
            bus_dc_minus=2,
            r_ohm=0.2,
            x_ohm=0.3,
            r_dc_ohm=0.1,
            pl_dc_mw=0.5,
            control_mode_ac="q_mvar",
            control_value_ac=0.5,
            control_mode_dc="vm_pu",
            control_value_dc=1.0,
            controllable=False,
            in_service=False,
        )

        # Row 3: name filled again
        create_vsc_stacked(
            net,
            bus=b2,
            bus_dc_plus=0,
            bus_dc_minus=2,
            r_ohm=0.15,
            x_ohm=0.25,
            r_dc_ohm=0.08,
            pl_dc_mw=0.3,
            control_mode_ac="slack",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=5.0,
            controllable=True,
            in_service=True,
            name="VSC C",
        )

        # Set nullable columns with mixed values
        net.vsc_stacked["name"] = pd.Series(["VSC A", pd.NA, "VSC C"], dtype=pd.StringDtype())

        validate_network(net)

    def test_name_column_all_na_multiple_rows_valid(self):
        """Test: name column can be NA for all rows"""
        net = pandapowerNet(name="test_invalid_bus_index")
        b0 = create_bus(net, 110.0)
        b1 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1

        create_vsc_stacked(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
        )

        create_vsc_stacked(
            net,
            bus=b1,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.2,
            x_ohm=0.3,
            r_dc_ohm=0.1,
            pl_dc_mw=0.5,
            control_mode_ac="q_mvar",
            control_value_ac=0.5,
            control_mode_dc="vm_pu",
            control_value_dc=1.0,
            controllable=False,
            in_service=False,
        )

        # All rows have NA for name
        net.vsc_stacked["name"] = pd.Series([pd.NA, pd.NA], dtype=pd.StringDtype())

        validate_network(net)


class TestVSCSTACKEDSchemaForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """Test: bus FK must reference an existing bus index"""
        net = pandapowerNet(name="test_invalid_bus_index")
        b0 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1
        create_vsc_stacked(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
        )

        net.vsc_stacked["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_invalid_bus_dc_plus_index(self):
        """Test: bus_dc_plus FK must reference an existing dc bus index"""
        net = pandapowerNet(name="test_invalid_bus_dc_plus_index")
        b0 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1
        create_vsc_stacked(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
        )

        net.vsc_stacked["bus_dc_minus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_invalid_bus_dc_plus_index(self):
        """Test: bus_dc_minus FK must reference an existing dc bus index"""
        net = pandapowerNet(name="test_invalid_bus_dc_plus_index")
        b0 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1
        create_vsc_stacked(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
        )

        net.vsc_stacked["bus_dc_plus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FKs work with non-sequential bus indices"""
        net = pandapowerNet(name="test_invalid_bus_dc_minus_index")
        create_bus(net, 110.0, index=10)
        create_bus(net, 110.0, index=42)
        create_bus(net, 110.0, index=100)
        create_bus_dc(net, vn_kv=110.0, index=5)
        create_bus_dc(net, vn_kv=110.0, index=15)
        create_bus_dc(net, vn_kv=110.0, index=25)

        create_vsc_stacked(
            net,
            bus=10,
            bus_dc_plus=5,
            bus_dc_minus=15,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
        )
        create_vsc_stacked(
            net,
            bus=42,
            bus_dc_plus=15,
            bus_dc_minus=25,
            r_ohm=0.2,
            x_ohm=0.3,
            r_dc_ohm=0.1,
            pl_dc_mw=0.5,
            control_mode_ac="q_mvar",
            control_value_ac=0.5,
            control_mode_dc="vm_pu",
            control_value_dc=1.0,
            controllable=False,
            in_service=False,
        )
        create_vsc_stacked(
            net,
            bus=100,
            bus_dc_plus=5,
            bus_dc_minus=25,
            r_ohm=0.15,
            x_ohm=0.25,
            r_dc_ohm=0.08,
            pl_dc_mw=0.3,
            control_mode_ac="slack",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=5.0,
            controllable=True,
            in_service=True,
        )

        validate_network(net)


class TestVCSSTACKEDResults:
    """Tests for vsc_stacked results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_vsc_stacked_result_totals(self):
        """Test: aggregated p_mw / q_mvar results are consistent"""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_vsc_stacked_internal_results(self):
        """Test: internal vm/va and dc quantities are within expected ranges"""
        pass