# test_pandera_motor_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_empty_network, create_bus, create_motor
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
        net = create_empty_network()
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
                itertools.product(["bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["pn_mech_mw"], [*negativ_floats, *not_floats_list]),
                itertools.product(["cos_phi"], [*ratio_invalid, *not_floats_list]),
                itertools.product(["cos_phi_n"], [*ratio_invalid, *not_floats_list]),
                itertools.product(["efficiency_percent"], [*percent_invalid, *not_floats_list]),
                itertools.product(["efficiency_n_percent"], [*percent_invalid, *not_floats_list]),
                itertools.product(["loading_percent"], [*percent_invalid, *not_floats_list]),
                itertools.product(["scaling"], [*negativ_floats, *not_floats_list]),
                itertools.product(["lrc_pu"], [*negativ_floats, *not_floats_list]),
                itertools.product(["rx"], [*negativ_floats, *not_floats_list]),
                itertools.product(["vn_kv"], [*negativ_floats, *not_floats_list]),
                itertools.product(["in_service"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        net = create_empty_network()
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

    @pytest.mark.parametrize(
        "parameter",
        [
            "pn_mech_mw",
            "cos_phi",
            "cos_phi_n",
            "efficiency_percent",
            "efficiency_n_percent",
            "loading_percent",
            "scaling",
            "lrc_pu",
            "rx",
            "vn_kv",
            "in_service",
        ],
    )
    def test_required_fields_nan_invalid(self, parameter):
        net = create_empty_network()
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
        )

        net.motor[parameter] = float(np.nan)
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestMotorOptionalFields:
    """Only 'name' is optional"""

    def test_optional_fields_with_nulls(self):
        net = create_empty_network()
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
            lrc_pu=5.0,
            rx=0.2,
            vn_kv=0.4,
            in_service=True,
            name=None,
        )
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

        net.motor["name"] = pd.Series([pd.NA, "M2"], dtype="string")
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
        net = create_empty_network()
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
            name="M1",
        )
        net.motor[parameter] = pd.Series([valid_value], dtype="string")
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(itertools.chain(itertools.product(["name"], not_strings_list))),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        net = create_empty_network()
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
            name="ok",
        )
        net.motor[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestMotorForeignKey:
    """Foreign key constraints"""

    def test_invalid_bus_index(self):
        net = create_empty_network()
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
        )

        net.motor["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestMotorResults:
    """Motor results"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_motor_result_totals(self):
        """Aggregated p_mw / q_mvar results are consistent"""
        pass
