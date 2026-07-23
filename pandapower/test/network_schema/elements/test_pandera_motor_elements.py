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
                itertools.product(["cos_phi_n"], [*ratio_valid, *zero_float]),
                itertools.product(["efficiency_percent"], [*percent_valid, *positiv_floats_plus_zero]),
                itertools.product(["efficiency_n_percent"], [*percent_valid, *positiv_floats_plus_zero]),
                itertools.product(["loading_percent"], [*percent_valid, *positiv_floats_plus_zero]),
                itertools.product(["scaling"], positiv_floats_plus_zero),
                itertools.product(["lrc_pu"], positiv_floats_plus_zero),
                itertools.product(["rx"], positiv_floats_plus_zero),
                itertools.product(["vn_kv"], positiv_floats_plus_zero),
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
            cos_phi_n=0.8,
            efficiency_percent=90.0,
            efficiency_n_percent=92.0,
            loading_percent=50.0,
            scaling=1.0,
            lrc_pu=6.0,
            rx=0.1,
            vn_kv=0.4,
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
            cos_phi_n=0.8,
            efficiency_percent=90.0,
            efficiency_n_percent=92.0,
            loading_percent=50.0,
            scaling=1.0,
            lrc_pu=6.0,
            rx=0.1,
            vn_kv=0.4,
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

    def test_all_optional_fields_valid(self):
        """Test: all optional fields can be set to valid values"""
        net = pandapowerNet(name="test_all_optional_fields_valid")
        b0 = create_bus(net, 0.4)

        create_motor(
            net,
            bus=b0,
            pn_mech_mw=1.0,
            cos_phi=0.9,
            cos_phi_n=0.8,
            efficiency_percent=90.0,
            efficiency_n_percent=92.0,
            loading_percent=50.0,
            scaling=1.0,
            lrc_pu=6.0,
            rx=0.1,
            vn_kv=0.4,
            in_service=True,
            name="M1",
            origin_id="id1",
            origin_class="class1",
            terminal="t1",
            description="desc1",
        )
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["origin_id"], [pd.NA, *strings]),
                itertools.product(["origin_class"], [pd.NA, *strings]),
                itertools.product(["terminal"], [pd.NA, *strings]),
                itertools.product(["description"], [pd.NA, *strings]),
            )
        ),
    )
    def test_required_fields_nan_invalid(self, parameter, invalid_value):
        net = pandapowerNet(name="test_required_fields_nan_invalid")
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
        net.motor[parameter] = pd.Series([invalid_value], dtype="string")
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(itertools.chain(
            itertools.product(["name"], [float(np.nan), *not_strings_list]),
            itertools.product(["origin_id"], [float(np.nan), *not_strings_list]),
            itertools.product(["origin_class"], [float(np.nan), *not_strings_list]),
            itertools.product(["terminal"], [float(np.nan), *not_strings_list]),
            itertools.product(["description"], [float(np.nan), *not_strings_list]),
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


class TestMotorScGroupFields:
    """Tests for short-circuit (sc) group fields with dependency validation"""

    def test_sc_group_with_complete_values(self):
        net = pandapowerNet(name="test_sc_group_with_complete_values")
        b0 = create_bus(net, 0.4)

        create_motor(
            net,
            bus=b0,
            pn_mech_mw=1.0,
            cos_phi=0.9,
            cos_phi_n=0.8,
            efficiency_percent=90.0,
            efficiency_n_percent=92.0,
            loading_percent=60.0,
            scaling=1.0,
            lrc_pu=6.0,
            rx=0.1,
            vn_kv=0.4,
            in_service=True,
            name=None,
        )
        validate_network(net)

    def test_sc_group_all_nan_valid(self):
        """Test: sc group with all NaN values is valid (group not triggered)"""
        net = pandapowerNet(name="test_sc_group_all_nan_valid")
        b0 = create_bus(net, 0.4)

        create_motor(
            net,
            bus=b0,
            pn_mech_mw=2.0,
            cos_phi=0.8,
            cos_phi_n=0.7,
            efficiency_percent=85.0,
            efficiency_n_percent=88.0,
            loading_percent=40.0,
            scaling=1.2,
            lrc_pu=4.0,
            rx=0.15,
            vn_kv=0.4,
            in_service=False,
            name="M2",
        )

        # Explicitly set all sc columns to NaN
        net.motor["cos_phi_n"] = float(np.nan)
        net.motor["efficiency_n_percent"] = float(np.nan)
        net.motor["lrc_pu"] = float(np.nan)
        net.motor["rx"] = float(np.nan)
        net.motor["vn_kv"] = float(np.nan)

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["cos_phi_n"], [*ratio_valid, *zero_float]),
                itertools.product(["efficiency_n_percent"], [*percent_valid, *positiv_floats_plus_zero]),
                itertools.product(["lrc_pu"], [*positiv_floats_plus_zero]),
                itertools.product(["rx"], [*positiv_floats_plus_zero]),
                itertools.product(["vn_kv"], [*positiv_floats_plus_zero]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        net = pandapowerNet(name="test_valid_optional_values")
        b0 = create_bus(net, 0.4)

        # Create motor with complete sc group
        create_motor(
            net,
            bus=b0,
            pn_mech_mw=1.0,
            cos_phi=0.9,
            cos_phi_n=0.8,
            efficiency_percent=90.0,
            efficiency_n_percent=92.0,
            loading_percent=50.0,
            scaling=1.0,
            lrc_pu=6.0,
            rx=0.1,
            vn_kv=0.4,
            in_service=True,
            name="M1",
        )

        net.motor[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["cos_phi_n"], [pd.NA, *ratio_invalid, *not_floats_list]),
                itertools.product(["efficiency_n_percent"], [pd.NA, *percent_invalid, *not_floats_list]),
                itertools.product(["lrc_pu"], [pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["rx"], [pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["vn_kv"], [pd.NA, *negativ_floats, *not_floats_list]),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        net = pandapowerNet(name="test_invalid_optional_values")
        b0 = create_bus(net, 0.4)

        create_motor(
            net,
            bus=b0,
            pn_mech_mw=1.0,
            cos_phi=0.9,
            cos_phi_n=0.8,
            efficiency_percent=90.0,
            efficiency_n_percent=92.0,
            loading_percent=50.0,
            scaling=1.0,
            lrc_pu=6.0,
            rx=0.1,
            vn_kv=0.4,
            in_service=True,
            name="ok",
        )
        net.motor[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_sc_group_partial_invalid(self):
        """Test: sc group must be complete - only cos_phi_n set is invalid"""
        net = pandapowerNet(name="test_sc_group_partial_invalid")
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
        net.motor["cos_phi_n"] = 0.8
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


    def test_sc_group_partial_missing_one_invalid(self):
        """Test: sc group must be complete - missing one column is invalid"""
        net = pandapowerNet(name="test_sc_group_partial_missing_one_invalid")
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
        # Set all but one sc column
        net.motor["cos_phi_n"] = 0.8
        net.motor["efficiency_n_percent"] = 92.0
        net.motor["lrc_pu"] = 6.0
        net.motor["rx"] = 0.1
        # vn_kv is missing
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_multiple_motors_mixed_sc_groups(self):
        net = pandapowerNet(name="test_multiple_motors_mixed_sc_groups")
        b0 = create_bus(net, 0.4)

        # Row 1: sc group complete
        create_motor(
            net,
            bus=b0,
            pn_mech_mw=1.0,
            cos_phi=0.9,
            cos_phi_n=0.8,
            efficiency_percent=90.0,
            efficiency_n_percent=92.0,
            loading_percent=50.0,
            scaling=1.0,
            lrc_pu=6.0,
            rx=0.1,
            vn_kv=0.4,
            in_service=True,
        )

        # Row 2: sc group all NaN
        create_motor(
            net,
            bus=b0,
            pn_mech_mw=2.0,
            cos_phi=0.85,
            efficiency_percent=88.0,
            loading_percent=60.0,
            scaling=1.0,
            in_service=False,
        )

        # Row 3: sc group complete again
        create_motor(
            net,
            bus=b0,
            pn_mech_mw=1.5,
            cos_phi=0.88,
            cos_phi_n=0.85,
            efficiency_percent=91.0,
            efficiency_n_percent=93.0,
            loading_percent=55.0,
            scaling=0.9,
            lrc_pu=5.5,
            rx=0.15,
            vn_kv=0.38,
            in_service=True,
        )

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

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FK works with non-sequential bus indices"""
        net = pandapowerNet(name="test_valid_bus_index_non_sequential")
        create_bus(net, 0.4, index=10)
        create_bus(net, 0.4, index=42)
        create_bus(net, 0.4, index=100)

        create_motor(
            net, bus=10, pn_mech_mw=1.0, cos_phi=0.9,
            efficiency_percent=90.0, loading_percent=50.0,
            scaling=1.0, in_service=True,
        )
        create_motor(
            net, bus=42, pn_mech_mw=2.0, cos_phi=0.85,
            efficiency_percent=88.0, loading_percent=60.0,
            scaling=1.0, in_service=True,
        )
        create_motor(
            net, bus=100, pn_mech_mw=0.5, cos_phi=0.92,
            efficiency_percent=92.0, loading_percent=40.0,
            scaling=0.9, in_service=False,
        )

        validate_network(net)


class TestMotorResults:
    """Tests for motor results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_motor_result_totals(self):
        """Aggregated p_mw / q_mvar results are consistent"""
        pass
