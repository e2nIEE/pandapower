# test_pandera_load_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus, create_load
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

# ZIP percentage ranges
percent_valid = [0.0, 50.0, 100.0]
percent_invalid = [-0.1, 100.1]


class TestLoadRequiredFields:
    """Tests for required load fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["p_mw"], all_allowed_floats),
                itertools.product(["q_mvar"], all_allowed_floats),
                itertools.product(["scaling"], positiv_floats_plus_zero),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = pandapowerNet(name="test_valid_required_values")
        create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1
        create_bus(net, 0.4, index=42)

        create_load(net, bus=0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        net.load[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [float(np.nan), pd.NA, *negativ_ints, *not_ints_list]),
                itertools.product(["p_mw"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["q_mvar"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["scaling"], [float(np.nan), pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = pandapowerNet(name="test_invalid_required_values")
        create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1

        create_load(net, bus=0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        net.load[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_zip_group_complete_valid(self):
        """Test: ZIP group with all columns present is valid"""
        net = pandapowerNet(name="test_zip_group_complete_valid")
        b0 = create_bus(net, 0.4)
        create_load(net, bus=b0, p_mw=1.0, q_mvar=0.1, scaling=1.0, in_service=True)

        # Set complete ZIP group
        net.load["const_z_p_percent"] = 20.0
        net.load["const_i_p_percent"] = 30.0
        net.load["const_z_q_percent"] = 10.0
        net.load["const_i_q_percent"] = 40.0

        validate_network(net)

    def test_zip_group_complete_invalid(self):
        """Test: ZIP group with all columns present is valid"""
        net = pandapowerNet(name="test_zip_group_complete_valid")
        b0 = create_bus(net, 0.4)
        create_load(net, bus=b0, p_mw=1.0, q_mvar=0.1, scaling=1.0, in_service=True)

        # Set complete ZIP group
        net.load["const_z_p_percent"] = '20.0'
        net.load["const_i_p_percent"] = 30.0
        net.load["const_z_q_percent"] = 10.0
        net.load["const_i_q_percent"] = 40.0

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestLoadOptionalFields:
    """Tests for optional load fields and ZIP group dependencies"""

    def test_all_optional_fields_valid(self):
        """Test: load with all optional fields and complete ZIP group is valid"""
        net = pandapowerNet(name="test_all_optional_fields_valid")
        b0 = create_bus(net, 0.4)

        create_load(
            net,
            bus=b0,
            p_mw=1.2,
            q_mvar=0.3,
            scaling=1.0,
            in_service=True,
            name="Load A",
            sn_mva=1.0,
            type="wye",
            zone="area-1",
            max_p_mw=2.0,
            min_p_mw=-1.0,
            max_q_mvar=1.0,
            min_q_mvar=-0.5,
        )
        # ZIP group (complete)
        net.load["const_z_p_percent"] = 20.0
        net.load["const_i_p_percent"] = 30.0
        net.load["const_z_q_percent"] = 10.0
        net.load["const_i_q_percent"] = 40.0

        # controllable (optional but not nullable)
        net.load["controllable"] = pd.Series([True], dtype=bool)

        # CIM columns
        net.load["origin_id"] = pd.Series(["cim_id_1"], dtype=pd.StringDtype())
        net.load["origin_class"] = pd.Series(["EnergyConsumer"], dtype=pd.StringDtype())
        net.load["terminal"] = pd.Series(["term_1"], dtype=pd.StringDtype())
        net.load["description"] = pd.Series(["Test load"], dtype=pd.StringDtype())

        # Ensure string dtypes for string columns
        net.load["name"] = net.load["name"].astype(pd.StringDtype())
        net.load["type"] = net.load["type"].astype(pd.StringDtype())
        net.load["zone"] = net.load["zone"].astype(pd.StringDtype())

        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: optional fields including nulls are valid when ZIP group is not triggered"""
        net = pandapowerNet(name="test_optional_fields_with_nulls")
        b0 = create_bus(net, 0.4)

        # Create 3 loads with different optional fields
        create_load(net, bus=b0, p_mw=1.0, q_mvar=0.1, scaling=1.0, in_service=True)
        create_load(net, bus=b0, p_mw=0.5, q_mvar=0.0, scaling=0.8, in_service=True)
        create_load(net, bus=b0, p_mw=2.0, q_mvar=0.2, scaling=1.1, in_service=False)

        # Assign optional fields with nulls
        net.load["name"] = pd.Series(["L1", pd.NA, "L3"], dtype=pd.StringDtype())
        net.load["zone"] = pd.Series(["Z1", pd.NA, "Z3"], dtype=pd.StringDtype())
        net.load["type"] = pd.Series([pd.NA, "delta", pd.NA], dtype=pd.StringDtype())
        net.load["sn_mva"] = [float("nan"), 2.0, float("nan")]

        # CIM columns with mixed nulls
        net.load["origin_id"] = pd.Series(["cim_1", pd.NA, pd.NA], dtype=pd.StringDtype())
        net.load["origin_class"] = pd.Series([pd.NA, pd.NA, "EnergyConsumer"], dtype=pd.StringDtype())
        net.load["terminal"] = pd.Series([pd.NA, pd.NA, pd.NA], dtype=pd.StringDtype())
        net.load["description"] = pd.Series([pd.NA, "Desc 2", pd.NA], dtype=pd.StringDtype())

        validate_network(net)



    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                # Nullable string columns - include pd.NA directly
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["zone"], [pd.NA, *strings]),
                itertools.product(["origin_id"], [pd.NA, *strings]),
                itertools.product(["origin_class"], [pd.NA, *strings]),
                itertools.product(["terminal"], [pd.NA, *strings]),
                itertools.product(["description"], [pd.NA, *strings]),
                # type has isin constraint - test valid values only (NA tested separately)
                itertools.product(["type"], ["wye", "delta"]),
                # Nullable float columns - include float(np.nan) directly
                itertools.product(["sn_mva"], [float(np.nan), *positiv_floats]),
                itertools.product(["max_p_mw"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["min_p_mw"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["max_q_mvar"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["min_q_mvar"], [float(np.nan), *all_allowed_floats]),
                # controllable is optional but NOT nullable
                itertools.product(["controllable"], bools),
                # ZIP group columns - test non-NA values only here (NA tested separately)
                itertools.product(["const_z_p_percent"], percent_valid),
                itertools.product(["const_i_p_percent"], percent_valid),
                itertools.product(["const_z_q_percent"], percent_valid),
                itertools.product(["const_i_q_percent"], percent_valid),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted (ZIP group satisfied when needed)"""
        net = pandapowerNet(name="test_valid_optional_values")
        b0 = create_bus(net, 0.4)

        create_load(net, bus=b0, p_mw=1.0, q_mvar=0.1, scaling=1.0, in_service=True)

        # ZIP group columns
        zip_columns = ["const_z_p_percent", "const_i_p_percent", "const_z_q_percent", "const_i_q_percent"]

        # Satisfy ZIP group for ZIP-related parameters
        if parameter in zip_columns:
            net.load["const_z_p_percent"] = 20.0
            net.load["const_i_p_percent"] = 30.0
            net.load["const_z_q_percent"] = 10.0
            net.load["const_i_q_percent"] = 40.0

        # Handle dtype preservation for nullable columns
        nullable_string_columns = ["name", "zone", "origin_id", "origin_class", "terminal", "description", "type"]

        if parameter in nullable_string_columns:
            net.load[parameter] = pd.Series([valid_value], dtype=pd.StringDtype())
        elif parameter == "controllable":
            net.load[parameter] = pd.Series([valid_value], dtype=bool)
        else:
            net.load[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                # String columns - invalid types
                itertools.product(["name"], not_strings_list),
                itertools.product(["zone"], not_strings_list),
                itertools.product(["origin_id"], not_strings_list),
                itertools.product(["origin_class"], not_strings_list),
                itertools.product(["terminal"], not_strings_list),
                itertools.product(["description"], not_strings_list),
                # type - anything but 'wye'/'delta' (and non-strings)
                itertools.product(["type"], [*strings, *not_strings_list]),
                # sn_mva must be > 0 if provided
                itertools.product(["sn_mva"], [*negativ_floats_plus_zero, *not_floats_list]),
                # P/Q limits are just floats (any value allowed)
                itertools.product(["max_p_mw"], not_floats_list),
                itertools.product(["min_p_mw"], not_floats_list),
                itertools.product(["max_q_mvar"], not_floats_list),
                itertools.product(["min_q_mvar"], not_floats_list),
                # controllable is NOT nullable
                itertools.product(["controllable"], [float(np.nan), pd.NA, *not_boolean_list]),
                # ZIP columns must be in [0, 100]
                itertools.product(["const_z_p_percent"], [float(np.nan), pd.NA, *percent_invalid, *not_floats_list]),
                itertools.product(["const_i_p_percent"], [float(np.nan), pd.NA, *percent_invalid, *not_floats_list]),
                itertools.product(["const_z_q_percent"], [float(np.nan), pd.NA, *percent_invalid, *not_floats_list]),
                itertools.product(["const_i_q_percent"], [float(np.nan), pd.NA, *percent_invalid, *not_floats_list]),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected (ZIP group satisfied when needed)"""
        net = pandapowerNet(name="test_invalid_optional_values")
        b0 = create_bus(net, 0.4)

        create_load(net, bus=b0, p_mw=1.0, q_mvar=0.1, scaling=1.0, in_service=True)

        # Provide complete ZIP group so only the target parameter triggers failure
        net.load["const_z_p_percent"] = 20.0
        net.load["const_i_p_percent"] = 30.0
        net.load["const_z_q_percent"] = 10.0
        net.load["const_i_q_percent"] = 40.0

        net.load[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestLoadForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """Test: bus FK must reference an existing bus index"""
        net = pandapowerNet(name="test_invalid_bus_index")
        b0 = create_bus(net, 0.4)
        create_load(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)

        net.load["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FK works with non-sequential bus indices"""
        net = pandapowerNet(name="test_valid_bus_index_non_sequential")
        create_bus(net, 0.4, index=10)
        create_bus(net, 0.4, index=42)
        create_bus(net, 0.4, index=100)

        create_load(net, bus=10, p_mw=1.0, q_mvar=0.1, scaling=1.0, in_service=True)
        create_load(net, bus=42, p_mw=2.0, q_mvar=0.2, scaling=0.9, in_service=True)
        create_load(net, bus=100, p_mw=0.5, q_mvar=0.05, scaling=1.1, in_service=False)

        validate_network(net)


class TestLoadResults:
    """Tests for load results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_load_result_totals(self):
        """Test: aggregated p_mw / q_mvar results are consistent"""
        pass
