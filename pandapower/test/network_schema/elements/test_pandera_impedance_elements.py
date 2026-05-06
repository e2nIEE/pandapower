# test_pandera_impedance_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus, create_impedance
from pandapower.network import pandapowerNet
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
                itertools.product(["sn_mva"], positiv_floats),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        net = pandapowerNet(name="test_valid_required_values")
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
                itertools.product(["rft_pu"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["xft_pu"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["rtf_pu"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["xtf_pu"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["gf_pu"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["bf_pu"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["gt_pu"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["bt_pu"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["sn_mva"], [float(np.nan), pd.NA, *negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        net = pandapowerNet(name="test_invalid_required_values")
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
        net = pandapowerNet(name="test_full_optional_fields_validation")
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
        net = pandapowerNet(name="test_optional_fields_with_nulls")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        # Row 1: All optional fields filled
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
            name="Impedance A",
            rft0_pu=0.1,
            xft0_pu=0.2,
            rtf0_pu=0.3,
            xtf0_pu=0.4,
        )

        # Row 2: All optional fields NA/NaN
        create_impedance(
            net,
            from_bus=b0,
            to_bus=b1,
            rft_pu=0.05,
            xft_pu=0.06,
            rtf_pu=0.07,
            xtf_pu=0.08,
            gf_pu=0.1,
            bf_pu=0.1,
            gt_pu=0.1,
            bt_pu=0.1,
            sn_mva=50.0,
            in_service=False,
        )

        # Set nullable columns with mixed values
        net.impedance["name"] = pd.Series(["Impedance A", pd.NA], dtype=pd.StringDtype())
        net.impedance["origin_id"] = pd.Series([pd.NA, pd.NA], dtype=pd.StringDtype())
        net.impedance["origin_class"] = pd.Series(["CIM_Class", pd.NA], dtype=pd.StringDtype())
        net.impedance["description"] = pd.Series([pd.NA, "Some description"], dtype=pd.StringDtype())
        net.impedance["terminal_to"] = pd.Series([pd.NA, pd.NA], dtype=pd.StringDtype())
        net.impedance["terminal_from"] = pd.Series([pd.NA, pd.NA], dtype=pd.StringDtype())

        # Float columns with mixed NaN
        net.impedance["rft0_pu"] = [0.1, float("nan")]
        net.impedance["xft0_pu"] = [0.2, float("nan")]
        net.impedance["rtf0_pu"] = [0.3, float("nan")]
        net.impedance["xtf0_pu"] = [0.4, float("nan")]
        net.impedance["gf0_pu"] = [float("nan"), 0.5]
        net.impedance["bf0_pu"] = [0.6, float("nan")]
        net.impedance["gt0_pu"] = [float("nan"), float("nan")]
        net.impedance["bt0_pu"] = [0.7, 0.8]

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["origin_id"], [pd.NA, *strings]),
                itertools.product(["origin_class"], [pd.NA, *strings]),
                itertools.product(["description"], [pd.NA, *strings]),
                itertools.product(["terminal_to"], [pd.NA, *strings]),
                itertools.product(["terminal_from"], [pd.NA, *strings]),
                # Nullable float columns (zero-sequence impedance) - include float(np.nan) directly
                itertools.product(["rft0_pu"], [float(np.nan), *positiv_floats]),
                itertools.product(["xft0_pu"], [float(np.nan), *positiv_floats]),
                itertools.product(["rtf0_pu"], [float(np.nan), *positiv_floats]),
                itertools.product(["xtf0_pu"], [float(np.nan), *positiv_floats]),
                # Nullable float columns (zero-sequence shunt) - any float allowed
                itertools.product(["gf0_pu"], [float(np.nan), *positiv_floats_plus_zero, *negativ_floats_plus_zero]),
                itertools.product(["bf0_pu"], [float(np.nan), *positiv_floats_plus_zero, *negativ_floats_plus_zero]),
                itertools.product(["gt0_pu"], [float(np.nan), *positiv_floats_plus_zero, *negativ_floats_plus_zero]),
                itertools.product(["bt0_pu"], [float(np.nan), *positiv_floats_plus_zero, *negativ_floats_plus_zero]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        net = pandapowerNet(name="test_valid_optional_values")
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
                itertools.product(["origin_id"], not_strings_list),
                itertools.product(["origin_class"], not_strings_list),
                itertools.product(["description"], not_strings_list),
                itertools.product(["terminal_to"], not_strings_list),
                itertools.product(["terminal_from"], not_strings_list),
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
        net = pandapowerNet(name="test_invalid_optional_values")
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


class TestImpedanceForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_from_bus_index(self):
        """Test: from_bus FK must reference an existing bus index"""
        net = pandapowerNet(name="test_invalid_from_bus_index")()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

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
            in_service=True,
        )

        net.impedance["from_bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_invalid_to_bus_index(self):
        """Test: to_bus FK must reference an existing bus index"""
        net = pandapowerNet(name="test_invalid_to_bus_index")()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

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
            in_service=True,
        )

        net.impedance["to_bus"] = 9999
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
