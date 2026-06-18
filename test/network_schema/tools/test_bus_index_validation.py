"""
Test for the :module:`pandapower.network_schema.tools.validation.bus_index_validation`
"""

import pandas as pd
import pytest
from pandera.errors import SchemaError

from pandapower.network_schema.tools.validation.bus_index_validation import (
    _bus_index_validation,
    _create_multi_column_reference_schema,
)
from pandapower.create import (
    create_bus, create_load, create_ext_grid, create_line,
    create_transformer, create_gen, create_bus_dc, create_vsc, create_switch
)
from pandapower.create.utils import _get_index_with_check
from pandapower.network import pandapowerNet
from pandapower.network_schema.line import line_schema
from pandapower.network_schema.bus import bus_schema
from pandapower.network_schema.bus_dc import bus_dc_schema
from pandapower.network_schema.switch import switch_schema
from pandapower.network_schema.vsc import vsc_schema
from pandapower.network_schema.load import load_schema
from pandapower.network_schema.trafo import trafo_schema


class TestCreateMultiColumnReferenceSchema:
    """
    Tests for the :func:`_create_multi_column_reference_schema` function.
    """

    def test_schema_validates_correct_columns(self) -> None:
        """
        Test that schema validates the specified columns.
        """
        reference_df = pd.DataFrame(index=[1, 2, 3])
        schema = _create_multi_column_reference_schema(reference_df, ["from_bus", "to_bus"], "line")

        # Valid DataFrame should pass
        valid_df = pd.DataFrame({"from_bus": [1, 2], "to_bus": [2, 3]})
        validated_df = schema.validate(valid_df)
        assert validated_df is not None

    def test_schema_fails_on_invalid_values(self) -> None:
        """
        Test that schema raises error for invalid values.
        """
        reference_df = pd.DataFrame(index=[1, 2, 3])
        schema = _create_multi_column_reference_schema(reference_df, ["from_bus", "to_bus"], "line")

        # Invalid DataFrame with values not in reference
        invalid_df = pd.DataFrame({"from_bus": [1, 4], "to_bus": [2, 3]})
        with pytest.raises(SchemaError):
            schema.validate(invalid_df)

    def test_schema_is_non_strict(self) -> None:
        """
        Test that the created schema is non-strict (allows extra columns).
        """
        reference_df = pd.DataFrame(index=[1, 2, 3])
        schema = _create_multi_column_reference_schema(reference_df, ["from_bus"], "line")

        # DataFrame with extra columns should still pass
        df_with_extra = pd.DataFrame({"from_bus": [1, 2], "extra_col": [10, 20]})
        validated_df = schema.validate(df_with_extra)
        assert validated_df is not None

    def test_schema_columns_are_not_nullable(self) -> None:
        """
        Test that schema columns are not nullable.
        """
        reference_df = pd.DataFrame(index=[1, 2, 3])
        schema = _create_multi_column_reference_schema(reference_df, ["from_bus"], "line")

        assert schema.columns["from_bus"].nullable is False


class TestBusIndexValidation:
    """
    Tests for the :func:`_bus_index_validation` function.
    """

    def test_line_with_valid_buses(self) -> None:
        """
        Test that validation passes for line with valid bus references.
        """
        from pandapower.create import create_bus, create_line

        net = pandapowerNet(name="test_line_with_valid_buses")
        b0 = create_bus(net, vn_kv=0.4)
        b1 = create_bus(net, vn_kv=0.4)
        create_line(net, from_bus=b0, to_bus=b1, length_km=0.1, std_type="NAYY 4x50 SE")

        # Should not raise any error
        _bus_index_validation("line", line_schema, net)

    def test_line_with_invalid_buses(self) -> None:
        """
        Test that validation raises error for invalid bus references.
        """

        net = pandapowerNet(name="test_line_with_invalid_buses")
        b0 = create_bus(net, vn_kv=0.4)
        b1 = create_bus(net, vn_kv=0.4)
        invalid_bus = _get_index_with_check(net, "bus", 99)
        l0 = create_line(net, from_bus=b0, to_bus=b1, length_km=0.1, std_type="NAYY 4x50 SE")
        # Test with incorrect index at from_bus
        net.line.at[l0, "from_bus"] = invalid_bus
        with pytest.raises(SchemaError):
            _bus_index_validation("line", line_schema, net)

        # Reset to valid
        net.line.at[l0, "from_bus"] = b0
        _bus_index_validation("line", line_schema, net)

        # Test with incorrect index at to_bus
        net.line.at[l0, "to_bus"] = invalid_bus
        with pytest.raises(SchemaError):
            _bus_index_validation("line", line_schema, net)

    def test_skips_validation_for_bus_element(self) -> None:
        """
        Test that validation is skipped for 'bus' element.
        """
        net = pandapowerNet(name="test_skips_validation_for_bus_element")
        create_bus(net, vn_kv=0.4)

        # Should not raise any error - bus is the reference table itself
        _bus_index_validation("bus", bus_schema, net)

    def test_skips_validation_for_bus_dc_element(self) -> None:
        """
        Test that validation is skipped for 'bus_dc' element.
        """
        net = pandapowerNet(name="test_skips_validation_for_bus_dc_element")
        create_bus_dc(net, vn_kv=0.4)

        # Should not raise any error - bus_dc is the reference table itself
        _bus_index_validation("bus_dc", bus_dc_schema, net)

    def test_validates_switch_with_element_column(self) -> None:
        """
        Test that switch validation includes the 'element' column.
        """
        net = pandapowerNet(name="test_validates_switch_with_element_column")
        b1 = create_bus(net, vn_kv=0.4)
        b2 = create_bus(net, vn_kv=0.4)
        l1 = create_line(net, from_bus=b1, to_bus=b2, length_km=0.1, std_type="NAYY 4x50 SE")
        # Create switch from bus to line
        create_switch(net, bus=b1, element=l1, et="l")

        # Should not raise any error
        _bus_index_validation("switch", switch_schema, net)

    def test_validates_vsc_with_ac_and_dc_buses(self) -> None:
        """
        Test validation of VSC with both AC and DC bus columns.
        """
        net = pandapowerNet(name="test_validates_vsc_with_ac_and_dc_buses")
        # AC buses
        b0 = create_bus(net, vn_kv=110.0)
        # DC buses
        dc0 = create_bus_dc(net, vn_kv=0.4)

        # Create VSC with both AC and DC bus references
        create_vsc(net, b0, dc0, r_ohm=1, x_ohm=1, r_dc_ohm=1)

        # Should not raise any error
        _bus_index_validation("vsc", vsc_schema, net)

    def test_vsc_with_invalid_dc_bus(self) -> None:
        """
        Test validation fails when DC bus reference is invalid in VSC.
        """
        net = pandapowerNet(name="test_vsc_with_invalid_dc_bus")
        b0 = create_bus(net, vn_kv=110.0)
        dc0 = create_bus_dc(net, vn_kv=0.4)
        invalid_dc_bus = _get_index_with_check(net, "bus_dc", 99)

        vsc0 = create_vsc(net, b0, dc0, r_ohm=1, x_ohm=1, r_dc_ohm=1)

        # Set invalid index
        net.vsc.at[vsc0, "bus_dc"] = invalid_dc_bus

        with pytest.raises(SchemaError):
            _bus_index_validation("vsc", vsc_schema, net)

    def test_load_with_valid_bus(self) -> None:
        """
        Test that only columns present in net[element] are validated.
        """
        net = pandapowerNet(name="test_load_with_valid_bus")
        b1 = create_bus(net, vn_kv=0.4)

        # Create a load (which only has 'bus' column, not 'from_bus' or 'to_bus')
        create_load(net, bus=b1, p_mw=0.1, q_mvar=0.05)

        # Should not raise any error
        _bus_index_validation("load", load_schema, net)


class TestBusIndexValidationIntegration:
    """
    Integration tests for bus index validation with actual pandapower networks.
    """

    def test_full_network_validation(self):
        """
        Test validation on a more complete network.
        """
        # Create a simple network
        net = pandapowerNet(name="test_full_network_validation")

        # Create buses
        b1 = create_bus(net, vn_kv=110.0, name="Bus 1")
        b2 = create_bus(net, vn_kv=110.0, name="Bus 2")
        b3 = create_bus(net, vn_kv=20.0, name="Bus 3")

        # Create external grid
        create_ext_grid(net, bus=b1, vm_pu=1.0)

        # Create lines
        create_line(net, from_bus=b1, to_bus=b2, length_km=10.0, std_type="149-AL1/24-ST1A 10.0")
        create_line(net, from_bus=b2, to_bus=b3, length_km=5.0, std_type="149-AL1/24-ST1A 10.0")

        # Create transformers
        create_transformer(net, hv_bus=b2, lv_bus=b3, std_type="25 MVA 110/20 kV")

        # Create loads
        create_load(net, bus=b3, p_mw=10.0, q_mvar=5.0)

        # Create generators
        create_gen(net, bus=b2, p_mw=20.0, vm_pu=1.0)

        # Test line validation
        _bus_index_validation("line", line_schema, net)

        # Test load validation
        _bus_index_validation("load", load_schema, net)

        # Test trafo validation
        _bus_index_validation("trafo", trafo_schema, net)
