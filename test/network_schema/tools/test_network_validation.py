import pytest

import pandapower.networks as pn
from pandapower import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import (
    NETWORKS_EXCLUDED_FROM_VALIDATION,
    validate_network,
)


class TestExcludedNetworks:
    """Tests for the NETWORKS_EXCLUDED_FROM_VALIDATION skip list."""

    def test_case11_iwamoto_is_excluded(self):
        assert "case11_iwamoto" in NETWORKS_EXCLUDED_FROM_VALIDATION

    def test_excluded_network_does_not_raise(self):
        net = pn.case11_iwamoto()
        validate_network(net)

    def test_excluded_network_does_not_raise_with_groups(self):
        net = pn.case11_iwamoto()
        validate_network(net, groups_to_validate="sc")
        validate_network(net, groups_to_validate=["sc", "3ph"])

    def test_exclusion_is_name_based_not_content_based(self):
        """A net with the same invalid data but a different name must still be validated."""
        net = pandapowerNet(name="not_the_excluded_case")
        net["line"] = pn.case11_iwamoto().line.copy()
        with pytest.raises(Exception, match="Validation failed for line"):
            validate_network(net)

    def test_renamed_excluded_network_is_validated(self):
        net = pn.case11_iwamoto()
        net.name = "renamed_case11"
        with pytest.raises(Exception, match="Validation failed for line"):
            validate_network(net)

    @pytest.mark.parametrize("name", ["case9", "case30", "case33bw"])
    def test_non_excluded_networks_are_still_validated(self, name):
        validate_network(getattr(pn, name)())
