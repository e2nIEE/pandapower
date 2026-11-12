import pandera.errors as pa_errors
import pytest

from pandapower.network_schema.tools.validation.network_validation import (
    ValidationIssue,
    format_validation_errors,
    validate_network,
)
from pandapower.networks import example_simple


@pytest.fixture
def net():
    return example_simple()


class TestValidateNetworkCollectErrors:
    def test_valid_net_returns_empty_dict(self, net):
        assert validate_network(net, collect_errors=True) == {}

    def test_collects_issues_of_multiple_elements(self, net):
        net.bus.loc[net.bus.index[0], "vn_kv"] = -1
        net.line.loc[net.line.index[0], "length_km"] = -1

        errors = validate_network(net, collect_errors=True)

        assert set(errors) == {"bus", "line"}
        assert all(isinstance(issue, ValidationIssue) for issues in errors.values() for issue in issues)

    def test_collects_all_issues_of_one_element(self, net):
        net.line.loc[net.line.index[0], "length_km"] = -1
        net.line.loc[net.line.index[1], "max_i_ka"] = 0

        errors = validate_network(net, collect_errors=True)

        assert len(errors["line"]) == 2
        assert {issue.column for issue in errors["line"]} == {"length_km", "max_i_ka"}

    def test_issue_message_is_readable(self, net):
        net.line.loc[net.line.index[0], "length_km"] = -1

        issue = validate_network(net, collect_errors=True)["line"][0]

        assert issue.element == "line"
        assert issue.column == "length_km"
        assert issue.reason == "DATAFRAME_CHECK"
        assert "length_km" in str(issue)
        assert "greater_than(0)" in str(issue)
        assert "-1" in str(issue)

    def test_foreign_key_issue_is_readable(self, net):
        net.line.loc[net.line.index[0], "from_bus"] = 999

        issue = validate_network(net, collect_errors=True)["line"][0]

        assert "references indices that do not exist" in str(issue)
        assert "999" in str(issue)

    def test_missing_column_issue_is_readable(self, net):
        net.bus = net.bus.drop(columns=["vn_kv"])

        issue = validate_network(net, collect_errors=True)["bus"][0]

        assert "required column 'vn_kv' is missing" in str(issue)

    def test_null_value_issue_is_readable(self, net):
        net.line.loc[net.line.index[0], "length_km"] = None

        issue = validate_network(net, collect_errors=True)["line"][0]

        assert "contains null values" in str(issue)

    def test_failure_cases_are_truncated(self):
        from pandapower.networks import create_cigre_network_mv

        net = create_cigre_network_mv()
        net.line.loc[:, "length_km"] = -1

        issue = validate_network(net, collect_errors=True)["line"][0]

        assert "more" in str(issue)

    def test_schema_level_issue_is_readable(self):
        from pandapower.networks import example_multivoltage

        errors = validate_network(example_multivoltage(), collect_errors=True)

        message = str(errors["trafo"][0])
        assert "tap configuration columns have dependency violations" in message

    def test_raises_without_collect_errors(self, net):
        net.line.loc[net.line.index[0], "length_km"] = -1

        with pytest.raises(pa_errors.SchemaError):
            validate_network(net)

    def test_collects_exceptions_instead_of_raising(self, net):
        del net["bus"]

        errors = validate_network(net, collect_errors=True)

        assert "line" in errors
        assert "validation could not be performed" in str(errors["line"][0])
        assert "KeyError" in str(errors["line"][0])


class TestGroupsToValidate:
    """
    Tests for the ``groups_to_validate`` argument, which accepts a single group name as a
    string or several groups as a list or set.
    """

    def test_single_group_as_string(self, net):
        # ext_grid has no sc columns in example_simple, so requiring the sc group fails
        with pytest.raises(pa_errors.SchemaError):
            validate_network(net, groups_to_validate="sc")

    def test_groups_as_list_raises_on_missing_sc_columns(self, net):
        with pytest.raises(pa_errors.SchemaError):
            validate_network(net, groups_to_validate=["sc", "3ph"])

    def test_groups_as_set_raises_on_missing_sc_columns(self, net):
        with pytest.raises(pa_errors.SchemaError):
            validate_network(net, groups_to_validate={"sc", "3ph"})

    def test_groups_as_list_is_collected_not_reported_as_preparation_error(self, net):
        """A list must not raise a TypeError, but report the missing group columns."""
        errors = validate_network(net, groups_to_validate=["sc", "3ph"], collect_errors=True)

        messages = format_validation_errors(errors)
        assert "could not be performed" not in messages
        assert "TypeError" not in messages
        # the sc group makes the sc columns required, so they are reported as missing
        assert "required column 's_sc_max_mva' is missing" in messages

    def test_groups_as_list_checks_the_requested_groups(self, net):
        """The requested groups must actually be validated, not silently ignored."""
        base_errors = validate_network(net, collect_errors=True)
        group_errors = validate_network(net, groups_to_validate=["sc", "3ph"], collect_errors=True)

        assert base_errors == {}
        assert group_errors != {}
        assert "ext_grid" in group_errors
        assert "line" in group_errors

    def test_groups_as_list_equals_groups_as_set(self, net):
        as_list = validate_network(net, groups_to_validate=["sc", "3ph"], collect_errors=True)
        as_set = validate_network(net, groups_to_validate={"sc", "3ph"}, collect_errors=True)

        assert format_validation_errors(as_list) == format_validation_errors(as_set)

    def test_no_groups_validates_base_schema_only(self, net):
        assert validate_network(net, groups_to_validate=None, collect_errors=True) == {}


class TestFormatValidationErrors:
    def test_empty_errors_format_to_empty_string(self):
        assert format_validation_errors({}) == ""

    def test_one_line_per_issue(self, net):
        net.bus.loc[net.bus.index[0], "vn_kv"] = -1
        net.line.loc[net.line.index[0], "length_km"] = -1

        report = format_validation_errors(validate_network(net, collect_errors=True))

        lines = report.splitlines()
        assert len(lines) == 2
        assert all(line.startswith(("bus: ", "line: ")) for line in lines)


class TestValidationIssue:
    def test_str_and_repr_use_message(self):
        issue = ValidationIssue(element="line", message="line: something is wrong")

        assert str(issue) == "line: something is wrong"
        assert repr(issue) == "line: something is wrong"
