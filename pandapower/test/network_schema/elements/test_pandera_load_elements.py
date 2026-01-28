import itertools
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_empty_network, create_bus, create_load
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
        net = create_empty_network()
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
                itertools.product(["bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["p_mw"], not_floats_list),
                itertools.product(["q_mvar"], not_floats_list),
                itertools.product(["scaling"], [*negativ_floats, *not_floats_list]),
                itertools.product(["in_service"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = create_empty_network()
        create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1

        create_load(net, bus=0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        net.load[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestLoadOptionalFields:
    """Tests for optional load fields and ZIP group dependencies"""

    def test_all_optional_fields_valid(self):
        """Test: load with all optional fields and complete ZIP group is valid"""
        net = create_empty_network()
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

        # nullable boolean
        net.load["controllable"] = pd.Series([True], dtype="boolean")

        # ensure string dtypes for string columns
        net.load["name"] = net.load["name"].astype("string")
        net.load["type"] = net.load["type"].astype("string")
        net.load["zone"] = net.load["zone"].astype("string")

        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: optional fields including nulls are valid when ZIP group is not triggered"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        # Create 3 loads with different optional fields
        create_load(net, bus=b0, p_mw=1.0, q_mvar=0.1, scaling=1.0, in_service=True)
        create_load(net, bus=b0, p_mw=0.5, q_mvar=0.0, scaling=0.8, in_service=True)
        create_load(net, bus=b0, p_mw=2.0, q_mvar=0.2, scaling=1.1, in_service=False)

        # Assign optional fields with nulls
        net.load["name"] = pd.Series(["L1", pd.NA, "L3"], dtype="string")
        net.load["zone"] = pd.Series(["Z1", pd.NA, "Z3"], dtype="string")
        net.load["type"] = pd.Series([pd.NA, "delta", pd.NA], dtype="string")
        net.load["sn_mva"] = [None, 2.0, None]
        net.load["controllable"] = pd.Series([pd.NA, True, False], dtype="boolean")

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], strings),
                itertools.product(["type"], ["wye", "delta"]),
                itertools.product(["zone"], strings),
                itertools.product(["sn_mva"], positiv_floats),  # gt(0)
                itertools.product(["controllable"], bools),
                itertools.product(["max_p_mw"], all_allowed_floats),
                itertools.product(["min_p_mw"], all_allowed_floats),
                itertools.product(["max_q_mvar"], all_allowed_floats),
                itertools.product(["min_q_mvar"], all_allowed_floats),
                itertools.product(["const_z_p_percent"], percent_valid),
                itertools.product(["const_i_p_percent"], percent_valid),
                itertools.product(["const_z_q_percent"], percent_valid),
                itertools.product(["const_i_q_percent"], percent_valid),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted (ZIP group satisfied when needed)"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_load(net, bus=b0, p_mw=1.0, q_mvar=0.1, scaling=1.0, in_service=True)

        # Satisfy ZIP group to avoid dependency failures when setting any ZIP-related column
        net.load["const_z_p_percent"] = 20.0
        net.load["const_i_p_percent"] = 30.0
        net.load["const_z_q_percent"] = 10.0
        net.load["const_i_q_percent"] = 40.0

        if parameter in {"name", "type", "zone"}:
            net.load[parameter] = pd.Series([valid_value], dtype="string")
        elif parameter == "controllable":
            net.load[parameter] = pd.Series([valid_value], dtype="boolean")
        else:
            net.load[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["type"], [*strings, *not_strings_list]),  # anything but 'wye'/'delta'
                itertools.product(["zone"], not_strings_list),
                itertools.product(["sn_mva"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["controllable"], not_boolean_list),
                itertools.product(["max_p_mw"], not_floats_list),
                itertools.product(["min_p_mw"], not_floats_list),
                itertools.product(["max_q_mvar"], not_floats_list),
                itertools.product(["min_q_mvar"], not_floats_list),
                itertools.product(["const_z_p_percent"], [*percent_invalid, *not_floats_list]),
                itertools.product(["const_i_p_percent"], [*percent_invalid, *not_floats_list]),
                itertools.product(["const_z_q_percent"], [*percent_invalid, *not_floats_list]),
                itertools.product(["const_i_q_percent"], [*percent_invalid, *not_floats_list]),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected (ZIP group satisfied when needed)"""
        net = create_empty_network()
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
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        create_load(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)

        net.load["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestLoadResults:
    """Tests for load results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_load_result_totals(self):
        """Test: aggregated p_mw / q_mvar results are consistent"""
        pass
