# test_pandera_motor_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus, create_motor
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network

from pandapower.test.network_schema.elements.helper import (
    strings,
    bools,
    not_strings_list,
    not_floats_list,
    not_boolean_list,
    positiv_ints_plus_zero,
    negativ_ints,
    not_ints_list,
    positiv_floats_plus_zero,
    negativ_floats,
    zero_float,
)

# Ranges from schema

ratio_valid = [0.0, 0.5, 1.0]  # for cos_phi, cos_phi_n
ratio_invalid = [-0.1, 1.1]
percent_valid = [0.0, 50.0, 100.0]  # for efficiency_percent, efficiency_n_percent, loading_percent
percent_invalid = [-0.1, 100.1]


class TestMotorRequiredFields:
    """All columns except 'name' are required"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["pn_mech_mw"], positiv_floats_plus_zero),
                itertools.product(["cos_phi"], [*ratio_valid, *zero_float]),
                itertools.product(["efficiency_percent"], [*percent_valid, *positiv_floats_plus_zero]),
                itertools.product(["loading_percent"], [*percent_valid, *positiv_floats_plus_zero]),
                itertools.product(["scaling"], positiv_floats_plus_zero),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        net = pandapowerNet(name="test_valid_required_values")
        create_bus(net, 0.4)  # 0
        create_bus(net, 0.4)  # 1
        create_bus(net, 0.4, index=42)

        create_motor(
            net,
            bus=0,
            pn_mech_mw=1.0,
            cos_phi=0.9,
            efficiency_percent=90.0,
            loading_percent=50.0,
            scaling=1.0,
            in_service=True,
            name="M1",
        )

        net.motor[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [float(np.nan), pd.NA, *negativ_ints, *not_ints_list]),
                itertools.product(["pn_mech_mw"], [float(np.nan), pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["cos_phi"], [float(np.nan), pd.NA, *ratio_invalid, *not_floats_list]),
                itertools.product(["cos_phi_n"], [*ratio_invalid, *not_floats_list]),
                itertools.product(["efficiency_percent"], [float(np.nan), pd.NA, *percent_invalid, *not_floats_list]),
                itertools.product(["efficiency_n_percent"], [*percent_invalid, *not_floats_list]),
                itertools.product(["loading_percent"], [float(np.nan), pd.NA, *percent_invalid, *not_floats_list]),
                itertools.product(["scaling"], [float(np.nan), pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["lrc_pu"], [*negativ_floats, *not_floats_list]),
                itertools.product(["rx"], [*negativ_floats, *not_floats_list]),
                itertools.product(["vn_kv"], [*negativ_floats, *not_floats_list]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        net = pandapowerNet(name="test_invalid_required_values")
        create_bus(net, 0.4)
        create_bus(net, 0.4)

        create_motor(
            net,
            bus=0,
            pn_mech_mw=1.0,
            cos_phi=0.9,
            efficiency_percent=90.0,
            loading_percent=50.0,
            scaling=1.0,
            in_service=True,
            name="M1",
        )

        net.motor[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestMotorOptionalFields:

    def test_optional_fields_with_nulls(self):
        """Test: name field can be null"""
        net = pandapowerNet(name="test_optional_fields_with_nulls")
        b0 = create_bus(net, 0.4)

        create_motor(
            net,
            bus=b0,
            pn_mech_mw=1.0,
            cos_phi=0.9,
            efficiency_percent=90.0,
            loading_percent=60.0,
            scaling=1.0,
            in_service=True,
            name=None,
        )
        create_motor(
            net,
            bus=b0,
            pn_mech_mw=2.0,
            cos_phi=0.8,
            efficiency_percent=85.0,
            loading_percent=40.0,
            scaling=1.2,
            in_service=False,
            name="M2",
        )

        net.motor["name"] = pd.Series([pd.NA, "M2"], dtype="string")
        validate_network(net)



    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["origin_id"], [pd.NA, *strings]),
                itertools.product(["origin_class"], [pd.NA, *strings]),
                itertools.product(["terminal"], [pd.NA, *strings]),
                itertools.product(["description"], [pd.NA, *strings]),
                itertools.product(["cos_phi_n"], [0.0, 0.5, 1.0]),
                itertools.product(["efficiency_n_percent"], [0.0, 50.0, 100.0]),
                itertools.product(["lrc_pu"], [0.0, 5.0, 10.0]),
                itertools.product(["rx"], [0.0, 0.1, 0.5]),
                itertools.product(["vn_kv"], [0.0, 0.38, 10.0]),
            )
        ),
    )
    def test_all_optional_fields_valid(self, parameter, valid_value):
        """Test: all optional fields can be set to valid values"""
        net = pandapowerNet(name="test_all_optional_fields_valid")
        b0 = create_bus(net, 0.4)

        create_motor(
            net,
            bus=b0,
            pn_mech_mw=1.0,
            cos_phi=0.9,
            efficiency_percent=90.0,
            loading_percent=60.0,
            scaling=1.0,
            in_service=True,
            name="M1",
        )
        if parameter in ["name", "origin_id", "origin_class", "terminal", "description"]:
            net.motor[parameter] = pd.Series([valid_value], dtype="string")
        else:
            net.motor[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(itertools.chain(
            itertools.product(["name"], [float(np.nan), *not_strings_list]),
            itertools.product(["origin_id"], [float(np.nan), *not_strings_list]),
            itertools.product(["origin_class"], [float(np.nan), *not_strings_list]),
            itertools.product(["terminal"], [float(np.nan), *not_strings_list]),
            itertools.product(["description"], [float(np.nan), *not_strings_list]),
            itertools.product(["cos_phi_n"], [float(np.nan),pd.NA, *ratio_invalid, *not_floats_list]),
            itertools.product(["efficiency_n_percent"], [float(np.nan),pd.NA, *percent_invalid, *not_floats_list]),
            itertools.product(["lrc_pu"], [float(np.nan),pd.NA, *negativ_floats, *not_floats_list]),
            itertools.product(["rx"], [float(np.nan),pd.NA, *negativ_floats, *not_floats_list]),
            itertools.product(["vn_kv"], [float(np.nan),pd.NA, *negativ_floats, *not_floats_list]),
        )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected"""
        net = pandapowerNet(name="test_invalid_optional_values")
        b0 = create_bus(net, 0.4)

        create_motor(
            net,
            bus=b0,
            pn_mech_mw=1.0,
            cos_phi=0.9,
            efficiency_percent=90.0,
            loading_percent=60.0,
            scaling=1.0,
            in_service=True,
            name="ok",
        )
        net.motor[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestMotorForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """Test: bus FK must reference an existing bus index"""
        net = pandapowerNet(name="test_invalid_bus_index")
        b0 = create_bus(net, 0.4)

        create_motor(
            net,
            bus=b0,
            pn_mech_mw=1.0,
            cos_phi=0.9,
            efficiency_percent=90.0,
            loading_percent=50.0,
            scaling=1.0,
            in_service=True,
        )

        net.motor["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestMotorResults:
    """Tests for motor results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_motor_result_totals(self):
        """Aggregated p_mw / q_mvar results are consistent"""
        pass
