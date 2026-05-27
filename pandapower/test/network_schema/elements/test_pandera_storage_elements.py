# test_pandera_storage_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus, create_storage
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
    positiv_floats,
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
                itertools.product(["sn_mva"], [float(np.nan), *positiv_floats]),
                itertools.product(["scaling"], all_allowed_floats),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Valid required values are accepted"""
        net = pandapowerNet(name="test_valid_required_values")
        create_bus(net, 0.4)  # 0
        create_bus(net, 0.4)  # 1
        create_bus(net, 0.4, index=42)

        create_storage(net, bus=0, p_mw=0.5, q_mvar=0.1, scaling=1.0, in_service=True, max_e_mwh=10.0)
        if parameter == "sn_mva":
            net.storage[parameter] = pd.Series([valid_value], dtype="float64")
        else:
            net.storage[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [float(np.nan), pd.NA, *negativ_ints, *not_ints_list]),
                itertools.product(["p_mw"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["q_mvar"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["sn_mva"], [pd.NA, *not_floats_list]),
                itertools.product(["scaling"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Invalid required values are rejected"""
        net = pandapowerNet(name="test_invalid_required_values")
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
        net = pandapowerNet(name="test_all_optional_fields_valid")
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
        net.storage["controllable"] = pd.Series([True], dtype=bool)

        # CIM columns
        net.storage["origin_id"] = pd.Series(["cim_id_1"], dtype=pd.StringDtype())
        net.storage["origin_class"] = pd.Series(["BatteryUnit"], dtype=pd.StringDtype())
        net.storage["terminal"] = pd.Series(["term_1"], dtype=pd.StringDtype())
        net.storage["description"] = pd.Series(["Test storage"], dtype=pd.StringDtype())

        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Optional fields incl. nulls are valid when OPF group is not triggered"""
        net = pandapowerNet(name="test_optional_fields_with_nulls")
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

        # CIM columns with mixed nulls
        net.storage["origin_id"] = pd.Series(["cim_1", pd.NA, pd.NA], dtype=pd.StringDtype())
        net.storage["origin_class"] = pd.Series([pd.NA, pd.NA, "BatteryUnit"], dtype=pd.StringDtype())
        net.storage["terminal"] = pd.Series([pd.NA, pd.NA, pd.NA], dtype=pd.StringDtype())
        net.storage["description"] = pd.Series([pd.NA, "Desc 2", pd.NA], dtype=pd.StringDtype())

        validate_network(net)

    def test_opf_group_complete_valid(self):
        """OPF group with all columns present is valid"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        create_storage(net, bus=b0, p_mw=0.1, q_mvar=0.0, scaling=1.0, in_service=True, max_e_mwh=10.0)

        # Set complete OPF group
        net.storage["max_p_mw"] = 1.0
        net.storage["min_p_mw"] = -1.0
        net.storage["max_q_mvar"] = 0.6
        net.storage["min_q_mvar"] = -0.6
        net.storage["controllable"] = pd.Series([True], dtype=bool)

        validate_network(net)

    def test_opf_group_partial_missing_invalid(self):
        """OPF group must be complete if any OPF value is set"""

        # Case 1: only max_p_mw
        net = pandapowerNet(name="test_opf_group_partial_missing_invalid0")
        b0 = create_bus(net, 0.4)
        create_storage(net, bus=b0, p_mw=0.1, q_mvar=0.0, scaling=1.0, in_service=True, max_e_mwh=10.0)
        net.storage["max_p_mw"] = 1.0
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

        # Case 2: only controllable
        net = pandapowerNet(name="test_opf_group_partial_missing_invalid1")
        b0 = create_bus(net, 0.4)
        create_storage(net, bus=b0, p_mw=0.2, q_mvar=0.1, scaling=1.0, in_service=True, max_e_mwh=10.0)
        net.storage["controllable"] = pd.Series([True], dtype="boolean")
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

        # Case 3: only min_q_mvar
        net = pandapowerNet(name="test_opf_group_partial_missing_invalid2")
        b0 = create_bus(net, 0.4)
        create_storage(net, bus=b0, p_mw=-0.2, q_mvar=0.0, scaling=1.0, in_service=True, max_e_mwh=10.0)
        net.storage["min_q_mvar"] = -0.5
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

        # Case 4: missing only controllable
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        create_storage(net, bus=b0, p_mw=0.1, q_mvar=0.0, scaling=1.0, in_service=True, max_e_mwh=10.0)
        net.storage["max_p_mw"] = 1.0
        net.storage["min_p_mw"] = -1.0
        net.storage["max_q_mvar"] = 0.6
        net.storage["min_q_mvar"] = -0.6
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["type"], [pd.NA, *strings]),
                itertools.product(["max_e_mwh"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["min_e_mwh"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["soc_percent"], [float(np.nan), *percent_valid]),
                #OPF columns
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
        net = pandapowerNet(name="test_valid_optional_values")
        b0 = create_bus(net, 0.4)

        create_storage(net, bus=b0, p_mw=0.3, q_mvar=0.0, sn_mva=1.0, scaling=1.0, in_service=True, max_e_mwh=10.0)

        # Satisfy OPF group to avoid dependency failures
        net.storage["max_p_mw"] = 1.0
        net.storage["min_p_mw"] = -1.0
        net.storage["max_q_mvar"] = 0.6
        net.storage["min_q_mvar"] = -0.6
        net.storage["controllable"] = pd.Series([True], dtype=bool)

        if parameter in {"name", "type"}:
            net.storage[parameter] = pd.Series([valid_value], dtype="string")
        elif parameter == "controllable":
            net.storage[parameter] = pd.Series([valid_value], dtype=bool)
        else:
            net.storage[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [float(np.nan), *not_strings_list]),
                itertools.product(["type"], [float(np.nan), *not_strings_list]),
                itertools.product(["sn_mva"], [pd.NA, *negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["max_e_mwh"], [pd.NA, *not_floats_list]),
                itertools.product(["min_e_mwh"], [pd.NA, *not_floats_list]),
                itertools.product(["soc_percent"], [pd.NA, *percent_invalid, *not_floats_list]),
                itertools.product(["max_p_mw"], [pd.NA, *not_floats_list]),
                itertools.product(["min_p_mw"], [pd.NA, *not_floats_list]),
                itertools.product(["max_q_mvar"], [pd.NA, *not_floats_list]),
                itertools.product(["min_q_mvar"], [pd.NA, *not_floats_list]),
                itertools.product(["controllable"], [float(np.nan), *not_boolean_list]),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Invalid optional values are rejected (OPF group satisfied)"""
        net = pandapowerNet(name="test_invalid_optional_values")
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

    @pytest.mark.xfail
    def test_opf_group_all_null_valid(self): #TODO controllable is not nullable
        """Test: OPF group columns can all be NA/NaN together (group not triggered)"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_storage(net, bus=b0, p_mw=0.5, q_mvar=0.1, scaling=1.0, in_service=True, max_e_mwh=10.0)

        # Set all OPF columns to NA/NaN with correct dtypes
        net.storage["max_p_mw"] = float(np.nan)
        net.storage["min_p_mw"] = float(np.nan)
        net.storage["max_q_mvar"] = float(np.nan)
        net.storage["min_q_mvar"] = float(np.nan)

        validate_network(net)

    @pytest.mark.xfail #TODO controllable is not nullable
    def test_opf_group_mixed_rows_valid(self):
        """Test: Multiple rows where OPF group is complete in some rows, all NaN in others"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        # Row 1: OPF group complete
        create_storage(net, bus=b0, p_mw=0.5, q_mvar=0.1, scaling=1.0, in_service=True, max_e_mwh=10.0)

        # Row 2: no OPF columns
        create_storage(net, bus=b1, p_mw=-0.3, q_mvar=0.0, scaling=0.8, in_service=False, max_e_mwh=5.0)

        # Set OPF group - Row 1 complete, Row 2 all NaN
        net.storage["max_p_mw"] = [1.0, float(np.nan)]
        net.storage["min_p_mw"] = [-1.0, float(np.nan)]
        net.storage["max_q_mvar"] = [0.6, float(np.nan)]
        net.storage["min_q_mvar"] = [-0.6, float(np.nan)]

        validate_network(net)


class TestStorageForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """bus must reference an existing bus index"""
        net = pandapowerNet(name="test_invalid_bus_index")
        b0 = create_bus(net, 0.4)

        create_storage(net, bus=b0, p_mw=0.5, q_mvar=0.0, scaling=1.0, in_service=True, max_e_mwh=10.0)
        net.storage["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FK works with non-sequential bus indices"""
        net = create_empty_network()
        create_bus(net, 0.4, index=10)
        create_bus(net, 0.4, index=42)
        create_bus(net, 0.4, index=100)

        create_storage(net, bus=10, p_mw=0.5, q_mvar=0.1, scaling=1.0, in_service=True, max_e_mwh=10.0)
        create_storage(net, bus=42, p_mw=-0.3, q_mvar=0.0, scaling=0.8, in_service=True, max_e_mwh=5.0)
        create_storage(net, bus=100, p_mw=0.0, q_mvar=-0.1, scaling=1.2, in_service=False, max_e_mwh=8.0)

        validate_network(net)


class TestStorageResults:
    """Tests for storage results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_storage_result_totals(self):
        """Aggregated p_mw / q_mvar results are consistent"""
        pass
