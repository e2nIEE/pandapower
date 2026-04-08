# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest

from pandapower.create import (
    create_empty_network, create_bus, create_line, create_transformer, create_transformer3w_from_parameters,
    create_switch, create_switches,
)
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_create_switches():
    net = create_empty_network()
    # standard
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 15)
    b4 = create_bus(net, 15)
    l1 = create_line(net, b1, b2, length_km=1, std_type="48-AL1/8-ST1A 10.0")
    t1 = create_transformer(net, b2, b3, std_type="160 MVA 380/110 kV")

    sw = create_switches(
        net,
        buses=[b1, b2, b3],
        elements=[l1, t1, b4],
        et=["l", "t", "b"],
        z_ohm=0.0,
        test_kwargs="aaa",
    )

    assert net.switch.bus.at[0] == b1
    assert net.switch.bus.at[1] == b2
    assert net.switch.bus.at[2] == b3
    assert net.switch.element.at[sw[0]] == l1
    assert net.switch.element.at[sw[1]] == t1
    assert net.switch.element.at[sw[2]] == b4
    assert net.switch.et.at[0] == "l"
    assert net.switch.et.at[1] == "t"
    assert net.switch.et.at[2] == "b"
    assert net.switch.z_ohm.at[0] == 0
    assert net.switch.z_ohm.at[1] == 0
    assert net.switch.z_ohm.at[2] == 0
    assert net.switch.test_kwargs.at[0] == "aaa"
    assert net.switch.test_kwargs.at[1] == "aaa"
    assert net.switch.test_kwargs.at[2] == "aaa"

    validate_network(net)


def test_create_switches_raise_errorexcept():
    net = create_empty_network()
    # standard
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 15)
    b4 = create_bus(net, 15)
    b5 = create_bus(net, 0.9)
    b6 = create_bus(net, 0.4)
    l1 = create_line(net, b1, b2, length_km=1, std_type="48-AL1/8-ST1A 10.0")
    t1 = create_transformer(net, b2, b3, std_type="160 MVA 380/110 kV")
    t3w1 = create_transformer3w_from_parameters(
        net,
        hv_bus=b4,
        mv_bus=b5,
        lv_bus=b6,
        vn_hv_kv=15.0,
        vn_mv_kv=0.9,
        vn_lv_kv=0.45,
        sn_hv_mva=0.6,
        sn_mv_mva=0.5,
        sn_lv_mva=0.4,
        vk_hv_percent=1.0,
        vk_mv_percent=1.0,
        vk_lv_percent=1.0,
        vkr_hv_percent=0.3,
        vkr_mv_percent=0.3,
        vkr_lv_percent=0.3,
        pfe_kw=0.2,
        i0_percent=0.3,
        #tap_neutral=0.0,  # FIXME: either remove this or add tap_pos and tap_side
    )
    sw = create_switch(net, bus=b1, element=l1, et="l", z_ohm=0.0)
    with pytest.raises(
            UserWarning, match=r"Switches with indexes \[0\] already exist."
    ):
        create_switches(
            net,
            buses=[b1, b2, b3],
            elements=[l1, t1, b4],
            et=["l", "t", "b"],
            z_ohm=0.0,
            index=[sw, 1, 2],
        )
    with pytest.raises(
            UserWarning, match=r"Cannot attach to buses \{6\}, they do not exist"
    ):
        create_switches(
            net, buses=[6, b2, b3], elements=[l1, t1, b4], et=["l", "t", "b"], z_ohm=0.0
        )
    with pytest.raises(UserWarning, match=r"Cannot attach to lines {(:?np\.int64\(1\)|1)}, they do not exist"):
        create_switches(
            net, buses=[b1, b2, b3], elements=[1, t1, b4], et=["l", "t", "b"], z_ohm=0.0
        )
    with pytest.raises(UserWarning, match=rf"Line not connected \(line element, bus\): \[\({l1}, {b3}\)\]"):
        create_switches(
            net,
            buses=[b3, b2, b3],
            elements=[l1, t1, b4],
            et=["l", "t", "b"],
            z_ohm=0.0,
        )
    with pytest.raises(UserWarning, match=r"Cannot attach to trafos {(:?np\.int64\(1\)|1)}, they do not exist"):
        create_switches(
            net, buses=[b1, b2, b3], elements=[l1, 1, b4], et=["l", "t", "b"], z_ohm=0.0
        )
    with pytest.raises(
            UserWarning, match=rf"Trafo not connected \(trafo element, bus\): \[\({b1}, {t1}\)\]"
    ):
        create_switches(
            net,
            buses=[b1, b1, b3],
            elements=[l1, t1, b4],
            et=["l", "t", "b"],
            z_ohm=0.0,
        )
    with pytest.raises(
            UserWarning, match=r"Cannot attach to buses {(:?np\.int64\(6\)|6)}, they do not exist"
    ):
        create_switches(
            net, buses=[b1, b2, b3], elements=[l1, t1, 6], et=["l", "t", "b"], z_ohm=0.0
        )
    with pytest.raises(UserWarning, match=r"Cannot attach to trafo3ws {(:?np\.int64\(1\)|1)}, they do not exist"):
        create_switches(
            net,
            buses=[b1, b2, b3],
            elements=[l1, t1, 1],
            et=["l", "t", "t3"],
            z_ohm=0.0,
        )
    with pytest.raises(
            UserWarning, match=rf"Trafo3w not connected \(trafo3w element, bus\): \[\({t3w1}, {b3}\)\]"
    ):
        create_switches(
            net,
            buses=[b1, b2, b3],
            elements=[l1, t1, t3w1],
            et=["l", "t", "t3"],
            z_ohm=0.0,
        )
    with pytest.raises(
        UserWarning, match=r"Cannot attach to buses {(:?np\.int64\(12398\)|12398)}, they do not exist"
    ):
        create_switches(
            net,
            buses=[b1, b2],
            elements=[b3, 12398],
            et=["b", "b"],
            z_ohm=0.0,
        )
    with pytest.raises(
            UserWarning, match=r"Cannot attach to buses \{13098\}, they do not exist"
    ):
        create_switches(
            net,
            buses=[b1, 13098],
            elements=[b2, b3],
            et=["b", "b"],
            z_ohm=0.0,
        )

    validate_network(net)
