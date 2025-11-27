
# test_pandera_bi_vsc_elements.py

import itertools
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_empty_network, create_bus, create_bus_dc
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


def _create_valid_bi_vsc_row(required=True, bus=0, bus_dc_plus=0, bus_dc_minus=1):
    df = {
        "bus": bus,
        "bus_dc_plus": bus_dc_plus,
        "bus_dc_minus": bus_dc_minus,
        "r_ohm": 0.1,
        "x_ohm": 0.05,
        "r_dc_ohm": 0.02,
        "pl_dc_mw": 0.5,
        "control_mode_ac": "vm_pu",
        "control_value_ac": 1.0,
        "control_mode_dc": "p_mw",
        "control_value_dc": 10.0,
        "controllable": True,
        "in_service": True,
    }
    if not required:
        df["name"] = "test"
    return df


class TestBiVSCRequiredFields:
    """Tests for required bi_vsc fields"""
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
                itertools.product(["control_mode_ac"], strings),
                itertools.product(["control_value_ac"], all_allowed_floats),
                itertools.product(["control_mode_dc"], strings),
                itertools.product(["control_value_dc"], all_allowed_floats),
                itertools.product(["controllable"], bools),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = create_empty_network()
        create_bus(net, 0.4)           # index 0
        create_bus(net, 0.4)           # index 1
        create_bus(net, 0.4, index=42)
        create_bus_dc(net, vn_kv=110.0)          # index 0
        create_bus_dc(net, vn_kv=110.0)          # index 1
        create_bus_dc(net, vn_kv=110.0, index=42)

        # Create a valid bi_vsc element
        row = _create_valid_bi_vsc_row(bus=0, bus_dc_plus=0, bus_dc_minus=1)
        net.bi_vsc = pd.DataFrame([row])
        net.bi_vsc[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["bus_dc_plus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["bus_dc_minus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["r_ohm"], not_floats_list),
                itertools.product(["x_ohm"], not_floats_list),
                itertools.product(["r_dc_ohm"], not_floats_list),
                itertools.product(["pl_dc_mw"], not_floats_list),
                itertools.product(["control_mode_ac"], not_strings_list),
                itertools.product(["control_value_ac"], not_floats_list),
                itertools.product(["control_mode_dc"], not_strings_list),
                itertools.product(["control_value_dc"], not_floats_list),
                itertools.product(["controllable"], not_boolean_list),
                itertools.product(["in_service"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = create_empty_network()
        create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1
        create_bus_dc(net, vn_kv=110.0)          # index 0
        create_bus_dc(net, vn_kv=110.0)          # index 1

        row = _create_valid_bi_vsc_row(bus=0, bus_dc_plus=0, bus_dc_minus=1)
        net.bi_vsc = pd.DataFrame([row])
        net.bi_vsc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestBiVSCOptionalFields:
    """Tests for optional bi_vsc fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"],[pd.NA, *strings]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        create_bus_dc(net, vn_kv=110.0)          # index 0
        create_bus_dc(net, vn_kv=110.0)          # index 1

        row = _create_valid_bi_vsc_row(required=False, bus=b0, bus_dc_plus=0, bus_dc_minus=1)
        net.bi_vsc = pd.DataFrame([row])
        net.bi_vsc[parameter] = valid_value
        net.bi_vsc["name"] = net.bi_vsc["name"].astype("string")

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1

        row = _create_valid_bi_vsc_row(required=False, bus=b0, bus_dc_plus=0, bus_dc_minus=1)
        net.bi_vsc = pd.DataFrame([row])
        net.bi_vsc[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestBiVSCForeignKey:
    """Tests for foreign key constraints"""

    @pytest.mark.parametrize("fk_field", ["bus", "bus_dc_plus", "bus_dc_minus"])
    def test_invalid_fk_index(self, fk_field):
        """Test: bus and bus_dc FKs must reference existing indices"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        create_bus(net, 0.4)

        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1

        row = _create_valid_bi_vsc_row(bus=b0, bus_dc_plus=0, bus_dc_minus=1)
        net.bi_vsc = pd.DataFrame([row])

        net.bi_vsc[fk_field] = 9999  # invalid references

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestBiVSCResults:
    """Tests for bi_vsc results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_bi_vsc_result_totals(self):
        """Test: aggregated p_mw / q_mvar / dc results are consistent"""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_bi_vsc_ac_dc_results(self):
        """Test: AC and DC side results contain valid values"""
        pass