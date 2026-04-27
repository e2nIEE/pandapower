# test_dcline.py

import itertools

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus, create_dcline
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


class TestDclineRequiredFields:
    """Tests for required dcline fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["from_bus"], positiv_ints_plus_zero),
                itertools.product(["to_bus"], positiv_ints_plus_zero),
                itertools.product(["p_mw"], all_allowed_floats),
                itertools.product(["loss_percent"], positiv_floats_plus_zero),
                itertools.product(["loss_mw"], positiv_floats_plus_zero),
                itertools.product(["vm_from_pu"], positiv_floats),
                itertools.product(["vm_to_pu"], positiv_floats),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = pandapowerNet(name="test_valid_required_values")
        create_bus(net, 0.4)
        create_bus(net, 0.4)
        create_bus(net, 0.4, index=42)

        create_dcline(
            net,
            from_bus=0,
            to_bus=1,
            p_mw=0.0,
            loss_percent=0.0,
            loss_mw=0.0,
            vm_from_pu=1.0,
            vm_to_pu=1.0,
            in_service=True,
        )
        net.dcline[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["from_bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["to_bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["p_mw"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["loss_percent"], [float(np.nan), pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["loss_mw"], [float(np.nan), pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["vm_from_pu"], [float(np.nan), pd.NA, *negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["vm_to_pu"], [float(np.nan), pd.NA, *negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = pandapowerNet(name="test_invalid_required_values")
        create_bus(net, 0.4)
        create_bus(net, 0.4)

        create_dcline(
            net,
            from_bus=0,
            to_bus=1,
            p_mw=0.0,
            loss_percent=0.0,
            loss_mw=0.0,
            vm_from_pu=1.0,
            vm_to_pu=1.0,
            in_service=True,
        )
        net.dcline[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestDclineOptionalFields:
    """Tests for optional dcline fields"""

    def test_all_optional_fields_valid(self):
        """Test: dcline with every optional field is valid"""
        net = pandapowerNet(name="test_all_optional_fields_valid")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)
        create_dcline(
            net,
            from_bus=b0,
            to_bus=b1,
            p_mw=10.0,
            loss_percent=1.0,
            loss_mw=0.1,
            vm_from_pu=1.02,
            vm_to_pu=1.01,
            in_service=True,
            name="lorem ipsum",
            max_p_mw=20.0,
            min_p_mw=0.0,
            min_q_from_mvar=-50.0,
            max_q_from_mvar=50.0,
            min_q_to_mvar=-40.0,
            max_q_to_mvar=40.0,
        )
        validate_network(net)

    # TODO failing on dependending columns
    def test_optional_fields_with_nulls(self):
        """Test: dcline with optional fields including nulls is valid"""
        net = pandapowerNet(name="test_optional_fields_with_nulls")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)
        create_dcline(
            net,
            from_bus=b0,
            to_bus=b1,
            p_mw=10.0,
            loss_percent=1.0,
            loss_mw=0.1,
            vm_from_pu=1.02,
            vm_to_pu=1.01,
            in_service=True,
        )
        create_dcline(
            net,
            from_bus=b0,
            to_bus=b1,
            p_mw=10.0,
            loss_percent=1.0,
            loss_mw=0.1,
            vm_from_pu=1.02,
            vm_to_pu=1.01,
            in_service=True,
            name="lorem ipsum",
        )
        create_dcline(
            net,
            from_bus=b0,
            to_bus=b1,
            p_mw=10.0,
            loss_percent=1.0,
            loss_mw=0.1,
            vm_from_pu=1.02,
            vm_to_pu=1.01,
            in_service=True,
            max_p_mw=20.0,
            min_p_mw=0.0,
            min_q_from_mvar=-50.0,
            max_q_from_mvar=50.0,
            min_q_to_mvar=-40.0,
            max_q_to_mvar=40.0,
        )

        # Set name column with nulls
        net.dcline["name"] = pd.Series([pd.NA, "lorem schnipsum", pd.NA], dtype=pd.StringDtype())

        validate_network(net)

    # TODO failing on dependending columns
    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["max_p_mw"], all_allowed_floats),
                itertools.product(["min_p_mw"], all_allowed_floats),
                itertools.product(["min_q_from_mvar"], all_allowed_floats),
                itertools.product(["max_q_from_mvar"], all_allowed_floats),
                itertools.product(["min_q_to_mvar"], all_allowed_floats),
                itertools.product(["max_q_to_mvar"], all_allowed_floats),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        net = pandapowerNet(name="test_valid_optional_values")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)
        dc0 = create_dcline(
            net,
            from_bus=b0,
            to_bus=b1,
            p_mw=10.0,
            loss_percent=1.0,
            loss_mw=0.1,
            vm_from_pu=1.02,
            vm_to_pu=1.01,
            in_service=True,
            name="lorem ipsum",
            max_p_mw=20.0,
            min_p_mw=0.0,
            min_q_from_mvar=-50.0,
            max_q_from_mvar=50.0,
            min_q_to_mvar=-40.0,
            max_q_to_mvar=40.0,
        )
        create_dcline(
            net,
            from_bus=b0,
            to_bus=b1,
            p_mw=10.0,
            loss_percent=1.0,
            loss_mw=0.1,
            vm_from_pu=1.02,
            vm_to_pu=1.01,
            in_service=True,
            name="lorem ipsum"
        )
        if parameter == "name":
            net.dcline[parameter] = pd.Series([valid_value], dtype=pd.StringDtype())
        else:
            net.dcline.loc[dc0, parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["max_p_mw"], not_floats_list),
                itertools.product(["min_p_mw"], not_floats_list),
                itertools.product(["min_q_from_mvar"], not_floats_list),
                itertools.product(["max_q_from_mvar"], not_floats_list),
                itertools.product(["min_q_to_mvar"], not_floats_list),
                itertools.product(["max_q_to_mvar"], not_floats_list),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected"""
        net = pandapowerNet(name="test_invalid_optional_values")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)
        create_dcline(
            net,
            from_bus=b0,
            to_bus=b1,
            p_mw=1.0,
            loss_percent=0.0,
            loss_mw=0.0,
            vm_from_pu=1.0,
            vm_to_pu=1.0,
            # Provide complete OPF group so only target parameter triggers failure
            max_p_mw=20.0,
            min_p_mw=0.0,
            min_q_from_mvar=-50.0,
            max_q_from_mvar=50.0,
            min_q_to_mvar=-40.0,
            max_q_to_mvar=40.0,
        )

        net.dcline[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_opf_group_partial_missing_invalid(self):
        """Test: OPF group must be complete if any OPF value is set"""
        net = pandapowerNet(name="test_opf_group_partial_missing_invalid")()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)
        create_dcline(
            net,
            from_bus=b0,
            to_bus=b1,
            p_mw=10.0,
            loss_percent=1.0,
            loss_mw=0.1,
            vm_from_pu=1.02,
            vm_to_pu=1.01,
            in_service=True,
        )
        net.dcline["max_p_mw"] = 100.0

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestDclineForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_from_bus_index(self):
        """Test: from_bus FK must reference an existing bus index"""
        net = pandapowerNet(name="test_invalid_from_bus_index")()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_dcline(
            net,
            from_bus=b0,
            to_bus=b1,
            p_mw=10.0,
            loss_percent=1.0,
            loss_mw=0.1,
            vm_from_pu=1.02,
            vm_to_pu=1.01,
            in_service=True,
        )

        net.dcline["from_bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_invalid_to_bus_index(self):
        """Test: to_bus FK must reference an existing bus index"""
        net = pandapowerNet(name="test_invalid_to_bus_index")()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_dcline(
            net,
            from_bus=b0,
            to_bus=b1,
            p_mw=10.0,
            loss_percent=1.0,
            loss_mw=0.1,
            vm_from_pu=1.02,
            vm_to_pu=1.01,
            in_service=True,
        )

        net.dcline["to_bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestDclineResults:
    """Tests for dcline results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_dcline_voltage_results(self):
        """Test: Voltage results are within valid range"""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_dcline_power_results(self):
        """Test: Power results are consistent"""
        pass
