# test_pandera_storage_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_empty_network, create_bus, create_storage
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
    positiv_floats,
    positiv_floats_plus_zero,
    negativ_floats_plus_zero,
    all_allowed_floats,
    percent_valid,
    percent_invalid,
)


class TestStorageRequiredFields:
    """Tests for required storage fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["p_mw"], all_allowed_floats),
                itertools.product(["q_mvar"], all_allowed_floats),
                itertools.product(["sn_mva"], positiv_floats),
                itertools.product(["scaling"], [*positiv_floats_plus_zero, *negativ_floats_plus_zero]),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Valid required values are accepted"""
        net = create_empty_network()
        create_bus(net, 0.4)  # 0
        create_bus(net, 0.4)  # 1
        create_bus(net, 0.4, index=42)

        create_storage(net, bus=0, p_mw=0.5, q_mvar=0.1, scaling=1.0, in_service=True, max_e_mwh=10.0)
        net.storage[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["p_mw"], not_floats_list),
                itertools.product(["q_mvar"], not_floats_list),
                itertools.product(["sn_mva"], not_floats_list),
                itertools.product(["scaling"], not_floats_list),
                itertools.product(["in_service"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Invalid required values are rejected"""
        net = create_empty_network()
        create_bus(net, 0.4)  # 0
        create_bus(net, 0.4)  # 1

        create_storage(net, bus=0, p_mw=0.5, q_mvar=0.1, scaling=1.0, in_service=True, max_e_mwh=10.0)
        net.storage[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestStorageOptionalFields:
    """Tests for optional storage fields and OPF group dependencies"""

    def test_all_optional_fields_valid(self):
        """All optional fields set and OPF group complete"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_storage(net, bus=b0, p_mw=0.5, q_mvar=0.1, scaling=1.0, in_service=True, max_e_mwh=10.0)

        # Optional fields
        net.storage["name"] = pd.Series(["Storage A"], dtype="string")
        net.storage["type"] = pd.Series(["li-ion"], dtype="string")
        net.storage["sn_mva"] = 1.0
        net.storage["max_e_mwh"] = 10.0
        net.storage["min_e_mwh"] = 0.0
        net.storage["soc_percent"] = 50.0

        # OPF group (must be complete if any set)
        net.storage["max_p_mw"] = 1.5
        net.storage["min_p_mw"] = -1.5
        net.storage["max_q_mvar"] = 0.8
        net.storage["min_q_mvar"] = -0.8
        net.storage["controllable"] = pd.Series([True], dtype="boolean")

        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Optional fields incl. nulls are valid when OPF group is not triggered"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        # Row 1
        create_storage(net, bus=b0, p_mw=0.2, q_mvar=0.0, scaling=1.0, in_service=True, max_e_mwh=10.0)
        # Row 2
        create_storage(net, bus=b0, p_mw=-0.4, q_mvar=0.1, scaling=1.0, in_service=False, max_e_mwh=10.0)
        # Row 3
        create_storage(net, bus=b0, p_mw=0.0, q_mvar=-0.2, scaling=0.8, in_service=True, max_e_mwh=10.0)

        net.storage["name"] = pd.Series(["A", pd.NA, "C"], dtype="string")
        net.storage["type"] = pd.Series([pd.NA, "flywheel", pd.NA], dtype="string")
        net.storage["sn_mva"] = [float(np.nan), 2.0, float(np.nan)]
        net.storage["soc_percent"] = [float(np.nan), 20.0, 75.0]
        net.storage["max_e_mwh"] = [float(np.nan), float(np.nan), 5.0]
        net.storage["min_e_mwh"] = [float(np.nan), 0.0, float(np.nan)]

        validate_network(net)

    def test_opf_group_partial_missing_invalid(self):
        """OPF group must be complete if any OPF value is set"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        # Case 1: only max_p_mw
        create_storage(net, bus=b0, p_mw=0.1, q_mvar=0.0, scaling=1.0, in_service=True, max_e_mwh=10.0)
        net.storage["max_p_mw"] = 1.0
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

        # Case 2: only controllable
        create_storage(net, bus=b0, p_mw=0.2, q_mvar=0.1, scaling=1.0, in_service=True, max_e_mwh=10.0)
        net.storage["controllable"] = pd.Series([True], dtype="boolean")
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

        # Case 3: only min_q_mvar
        create_storage(net, bus=b0, p_mw=-0.2, q_mvar=0.0, scaling=1.0, in_service=True, max_e_mwh=10.0)
        net.storage["min_q_mvar"] = -0.5
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], strings),
                itertools.product(["type"], strings),
                itertools.product(["max_e_mwh"], all_allowed_floats),
                itertools.product(["min_e_mwh"], all_allowed_floats),
                itertools.product(["soc_percent"], percent_valid),
                itertools.product(["max_p_mw"], all_allowed_floats),
                itertools.product(["min_p_mw"], all_allowed_floats),
                itertools.product(["max_q_mvar"], all_allowed_floats),
                itertools.product(["min_q_mvar"], all_allowed_floats),
                itertools.product(["controllable"], bools),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Valid optional values are accepted (OPF group satisfied when needed)"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_storage(net, bus=b0, p_mw=0.3, q_mvar=0.0, sn_mva=1.0, scaling=1.0, in_service=True, max_e_mwh=10.0)

        # Satisfy OPF group to avoid dependency failures
        net.storage["max_p_mw"] = 1.0
        net.storage["min_p_mw"] = -1.0
        net.storage["max_q_mvar"] = 0.6
        net.storage["min_q_mvar"] = -0.6
        net.storage["controllable"] = pd.Series([True], dtype="boolean")

        if parameter in {"name", "type"}:
            net.storage[parameter] = pd.Series([valid_value], dtype="string")
        elif parameter == "controllable":
            net.storage[parameter] = pd.Series([valid_value], dtype="boolean")
        else:
            net.storage[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["type"], not_strings_list),
                itertools.product(["sn_mva"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["max_e_mwh"], not_floats_list),
                itertools.product(["min_e_mwh"], not_floats_list),
                itertools.product(["soc_percent"], [*percent_invalid, *not_floats_list]),
                itertools.product(["max_p_mw"], not_floats_list),
                itertools.product(["min_p_mw"], not_floats_list),
                itertools.product(["max_q_mvar"], not_floats_list),
                itertools.product(["min_q_mvar"], not_floats_list),
                itertools.product(["controllable"], not_boolean_list),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Invalid optional values are rejected (OPF group satisfied)"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_storage(net, bus=b0, p_mw=0.3, q_mvar=0.0, scaling=1.0, in_service=True, max_e_mwh=10.0)

        # Provide complete OPF group so only the target parameter triggers failure
        net.storage["max_p_mw"] = 1.0
        net.storage["min_p_mw"] = -1.0
        net.storage["max_q_mvar"] = 0.6
        net.storage["min_q_mvar"] = -0.6
        net.storage["controllable"] = pd.Series([True], dtype="boolean")

        net.storage[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestStorageForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """bus must reference an existing bus index"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_storage(net, bus=b0, p_mw=0.5, q_mvar=0.0, scaling=1.0, in_service=True, max_e_mwh=10.0)
        net.storage["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestStorageResults:
    """Tests for storage results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_storage_result_totals(self):
        """Aggregated p_mw / q_mvar results are consistent"""
        pass
