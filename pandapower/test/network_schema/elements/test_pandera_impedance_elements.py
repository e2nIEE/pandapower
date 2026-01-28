import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower import create_impedance
from pandapower.create import create_empty_network, create_bus
from pandapower.network_schema.tools.validation.network_validation import validate_network
from pandapower.test.network_schema.elements.helper import (
    strings,
    all_allowed_floats,
    not_floats_list,
    not_strings_list,
    not_boolean_list,
    positiv_ints_plus_zero,
    negativ_ints,
    not_ints_list,
    positiv_floats,
    negativ_floats_plus_zero,
    positiv_floats_plus_zero,
    negativ_floats,
    bools,
)


class TestImpedanceRequiredFields:
    """Tests for required Impedance fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["from_bus"], positiv_ints_plus_zero),
                itertools.product(["to_bus"], positiv_ints_plus_zero),
                itertools.product(["rft_pu"], all_allowed_floats),
                itertools.product(["xft_pu"], all_allowed_floats),
                itertools.product(["rtf_pu"], all_allowed_floats),
                itertools.product(["xtf_pu"], all_allowed_floats),
                itertools.product(["gf_pu"], all_allowed_floats),
                itertools.product(["bf_pu"], all_allowed_floats),
                itertools.product(["gt_pu"], all_allowed_floats),
                itertools.product(["bt_pu"], all_allowed_floats),
                itertools.product(["sn_mva"], positiv_floats),  # strictly > 0
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        net = create_empty_network()
        create_bus(net, 0.4, index=0)
        create_bus(net, 0.4, index=1)
        create_bus(net, 0.4, index=42)
        # Provide all required fields explicitly
        create_impedance(
            net,
            from_bus=0,
            to_bus=1,
            rft_pu=0.0,
            xft_pu=0.0,
            rtf_pu=0.0,
            xtf_pu=0.0,
            gf_pu=0.0,
            bf_pu=0.0,
            gt_pu=0.0,
            bt_pu=0.0,
            sn_mva=100.0,
            in_service=True,
        )
        net.impedance[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["from_bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["to_bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["rft_pu"], not_floats_list),
                itertools.product(["xft_pu"], not_floats_list),
                itertools.product(["rtf_pu"], not_floats_list),
                itertools.product(["xtf_pu"], not_floats_list),
                itertools.product(["gf_pu"], not_floats_list),
                itertools.product(["bf_pu"], not_floats_list),
                itertools.product(["gt_pu"], not_floats_list),
                itertools.product(["bt_pu"], not_floats_list),
                itertools.product(["sn_mva"], [*negativ_floats_plus_zero, *not_floats_list]),  # <= 0 invalid
                itertools.product(["in_service"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        net = create_empty_network()
        create_bus(net, 0.4)
        create_bus(net, 0.4)
        create_impedance(
            net,
            from_bus=0,
            to_bus=1,
            rft_pu=0.0,
            xft_pu=0.0,
            rtf_pu=0.0,
            xtf_pu=0.0,
            gf_pu=0.0,
            bf_pu=0.0,
            gt_pu=0.0,
            bt_pu=0.0,
            sn_mva=100.0,
            in_service=True,
        )
        net.impedance[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestImpedanceOptionalFields:
    """Tests for optional impedance fields"""

    def test_full_optional_fields_validation(self):
        """Impedance with all optional fields is valid"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)
        # Optional zero-sequence and name provided
        create_impedance(
            net,
            from_bus=b0,
            to_bus=b1,
            rft_pu=0.01,
            xft_pu=0.02,
            rtf_pu=0.03,
            xtf_pu=0.04,
            gf_pu=0.0,
            bf_pu=0.0,
            gt_pu=0.0,
            bt_pu=0.0,
            sn_mva=100.0,
            in_service=False,
            name="lorem ipsum",
            rft0_pu=0.1,
            xft0_pu=0.2,
            rtf0_pu=0.3,
            xtf0_pu=0.4,
            gf0_pu=0.0,
            bf0_pu=0.0,
            gt0_pu=0.0,
            bt0_pu=0.0,
        )
        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Optional fields can be missing or null"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        # Only required fields
        create_impedance(
            net,
            from_bus=b0,
            to_bus=b1,
            rft_pu=0.0,
            xft_pu=0.0,
            rtf_pu=0.0,
            xtf_pu=0.0,
            gf_pu=0.0,
            bf_pu=0.0,
            gt_pu=0.0,
            bt_pu=0.0,
            sn_mva=50.0,
            in_service=True,
        )

        # Set some optionals to NaN/None
        for col in ["name", "rft0_pu", "xft0_pu", "rtf0_pu", "xtf0_pu", "gf0_pu", "bf0_pu", "gt0_pu", "bt0_pu"]:
            # Ensure column exists and assign a null
            if col not in net.impedance.columns:
                net.impedance[col] = pd.Series([float(np.nan)], index=net.impedance.index)
            net.impedance[col].iat[0] = pd.NA

        # Set name dtype properly
        net.impedance["name"] = net.impedance["name"].astype(pd.StringDtype())

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["rft0_pu"], positiv_floats),
                itertools.product(["xft0_pu"], positiv_floats),
                itertools.product(["rtf0_pu"], positiv_floats),
                itertools.product(["xtf0_pu"], positiv_floats),
                itertools.product(["gf0_pu"], [float(np.nan), *positiv_floats_plus_zero, *negativ_floats_plus_zero]),
                itertools.product(["bf0_pu"], [float(np.nan), *positiv_floats_plus_zero, *negativ_floats_plus_zero]),
                itertools.product(["gt0_pu"], [float(np.nan), *positiv_floats_plus_zero, *negativ_floats_plus_zero]),
                itertools.product(["bt0_pu"], [float(np.nan), *positiv_floats_plus_zero, *negativ_floats_plus_zero]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)
        create_impedance(
            net,
            from_bus=b0,
            to_bus=b1,
            rft_pu=0.0,
            xft_pu=0.0,
            rtf_pu=0.0,
            xtf_pu=0.0,
            gf_pu=0.0,
            bf_pu=0.0,
            gt_pu=0.0,
            bt_pu=0.0,
            sn_mva=10.0,
            in_service=False,
        )
        if parameter == "name":
            net.impedance[parameter] = pd.Series([valid_value], dtype=pd.StringDtype())
        else:
            net.impedance[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                # zero-sequence impedance parts must be > 0 if provided
                itertools.product(["rft0_pu"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["xft0_pu"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["rtf0_pu"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["xtf0_pu"], [*negativ_floats_plus_zero, *not_floats_list]),
                # zero-sequence shunt parts are floats if provided (no >0 check)
                itertools.product(["gf0_pu"], not_floats_list),
                itertools.product(["bf0_pu"], not_floats_list),
                itertools.product(["gt0_pu"], not_floats_list),
                itertools.product(["bt0_pu"], not_floats_list),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)
        create_impedance(
            net,
            from_bus=b0,
            to_bus=b1,
            rft_pu=0.0,
            xft_pu=0.0,
            rtf_pu=0.0,
            xtf_pu=0.0,
            gf_pu=0.0,
            bf_pu=0.0,
            gt_pu=0.0,
            bt_pu=0.0,
            sn_mva=10.0,
            in_service=True,
        )
        net.impedance[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestImpedanceResults:
    """Tests for impedance results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_impedance_power_results(self):
        """Test: Power results are consistent"""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_impedance_currents_results(self):
        """Test: Currents results are within valid range"""
        pass
