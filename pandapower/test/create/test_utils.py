from pandera.errors import SchemaError
import pytest

from pandapower.create._utils import add_tag_group_to_df
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network


class TestAddTagGroupToDF:
    def test_add_sc_group(self):
        net = pandapowerNet(name="test_add_sc_group")
        add_tag_group_to_df(net, "ext_grid", "sc")

        validate_network(net)
        with pytest.raises(SchemaError):
            validate_network(net, "sc")

        # add remaining sc columns
        add_tag_group_to_df(net, "gen", "sc")
        add_tag_group_to_df(net, "line", "sc")
        add_tag_group_to_df(net, "sgen", "sc")
        add_tag_group_to_df(net, "trafo", "sc")
        add_tag_group_to_df(net, "trafo3w", "sc")

        validate_network(net, "sc")
