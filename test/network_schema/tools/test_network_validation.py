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
