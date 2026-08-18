# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pytest

import pandapower as pp
from pandapower.create import (
    create_empty_network, create_bus, create_ext_grid, create_line_from_parameters,
    create_load, create_switch, create_gen, create_transformer_from_parameters,
    create_sgen, create_shunt,
)
from pandapower.run import runpp
from pandapower.toolbox import compute_switch_flows


def _make_two_bus_coupler_net():
    """Two buses connected by a zero-impedance switch, load on one side.

    ext_grid (bus 0) --line-- (bus 1) --switch-- (bus 2) --load
    """
    net = create_empty_network()
    b0 = create_bus(net, vn_kv=20.0, name="slack")
    b1 = create_bus(net, vn_kv=20.0, name="bus1")
    b2 = create_bus(net, vn_kv=20.0, name="bus2")

    create_ext_grid(net, b0, vm_pu=1.0)
    create_line_from_parameters(net, b0, b1, length_km=10, r_ohm_per_km=0.1,
                                x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
    create_switch(net, bus=b1, element=b2, et="b", closed=True)
    create_load(net, b2, p_mw=1.0, q_mvar=0.5)
    return net, b0, b1, b2


class TestComputeSwitchFlowsBasic:
    """Basic scenarios: single coupler, open switches, not-converged."""

    def test_single_coupler_power_balance(self):
        """The switch must carry the entire load since it's the only path."""
        net, b0, b1, b2 = _make_two_bus_coupler_net()
        runpp(net)
        assert np.isnan(net.res_switch.at[0, "p_from_mw"])

        compute_switch_flows(net)

        p_from = net.res_switch.at[0, "p_from_mw"]
        q_from = net.res_switch.at[0, "q_from_mvar"]
        p_to = net.res_switch.at[0, "p_to_mw"]
        q_to = net.res_switch.at[0, "q_to_mvar"]
        i_ka = net.res_switch.at[0, "i_ka"]

        assert not np.isnan(p_from)
        assert not np.isnan(i_ka)

        # Power balance: from + to ≈ 0 (zero-impedance, no losses)
        assert np.isclose(p_from + p_to, 0, atol=1e-10)
        assert np.isclose(q_from + q_to, 0, atol=1e-10)

        # Switch must carry approximately the load power
        load_p = net.res_load.at[0, "p_mw"]
        load_q = net.res_load.at[0, "q_mvar"]
        assert np.isclose(abs(p_from), load_p, atol=1e-6)
        assert np.isclose(abs(q_from), load_q, atol=1e-6)

        # Current must be positive
        assert i_ka > 0

    def test_open_switch_zero_flow(self):
        """An open bus-bus switch must have zero flow (set by runpp already)."""
        net, b0, b1, b2 = _make_two_bus_coupler_net()
        net.switch.at[0, "closed"] = False
        # Add line so bus 2 is still connected (otherwise non-convergence)
        create_line_from_parameters(net, b1, b2, length_km=1, r_ohm_per_km=0.1,
                                    x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        runpp(net)
        compute_switch_flows(net)

        # Open switches should remain zero / not be filled with flow
        assert net.res_switch.at[0, "i_ka"] == 0

    def test_not_converged_raises(self):
        """Must raise when load flow has not converged."""
        net, _, _, _ = _make_two_bus_coupler_net()
        # Don't run load flow
        net.converged = False
        with pytest.raises(UserWarning, match="did not converge"):
            compute_switch_flows(net)

    def test_no_switches(self):
        """Net without switches should return silently."""
        net = create_empty_network()
        b0 = create_bus(net, vn_kv=20.0)
        b1 = create_bus(net, vn_kv=20.0)
        create_ext_grid(net, b0)
        create_line_from_parameters(net, b0, b1, length_km=10, r_ohm_per_km=0.1,
                                    x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        create_load(net, b1, p_mw=1.0)
        runpp(net)
        compute_switch_flows(net)  # should not raise

    def test_impedance_switch_not_overwritten(self):
        """Switches with z_ohm > 0 already have results; they must not change."""
        net, b0, b1, b2 = _make_two_bus_coupler_net()
        net.switch.at[0, "z_ohm"] = 0.001
        runpp(net)

        p_before = net.res_switch.at[0, "p_from_mw"]
        compute_switch_flows(net)
        p_after = net.res_switch.at[0, "p_from_mw"]

        assert np.isclose(p_before, p_after)


class TestComputeSwitchFlowsChain:
    """Multiple couplers in series (chain topology)."""

    def test_three_bus_chain(self):
        """ext_grid -- (b0) --line-- (b1) --sw0-- (b2) --sw1-- (b3) --load

        sw0 must carry the full load; sw1 must also carry the full load.
        """
        net = create_empty_network()
        b0 = create_bus(net, vn_kv=20.0)
        b1 = create_bus(net, vn_kv=20.0)
        b2 = create_bus(net, vn_kv=20.0)
        b3 = create_bus(net, vn_kv=20.0)

        create_ext_grid(net, b0)
        create_line_from_parameters(net, b0, b1, length_km=10, r_ohm_per_km=0.1,
                                    x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        sw0 = create_switch(net, bus=b1, element=b2, et="b")
        sw1 = create_switch(net, bus=b2, element=b3, et="b")
        create_load(net, b3, p_mw=2.0, q_mvar=1.0)

        runpp(net)
        compute_switch_flows(net)

        load_p = net.res_load.at[0, "p_mw"]
        load_q = net.res_load.at[0, "q_mvar"]

        # Both switches carry the full load
        for sw in [sw0, sw1]:
            assert np.isclose(abs(net.res_switch.at[sw, "p_from_mw"]), load_p, atol=1e-6)
            assert np.isclose(abs(net.res_switch.at[sw, "q_from_mvar"]), load_q, atol=1e-6)
            assert net.res_switch.at[sw, "i_ka"] > 0

    def test_chain_with_intermediate_load(self):
        """(b0)--line--(b1)--sw0--(b2)--sw1--(b3)
            ext_grid      load1=1MW     load2=2MW

        sw0 must carry 3 MW, sw1 must carry 2 MW.
        """
        net = create_empty_network()
        b0 = create_bus(net, vn_kv=20.0)
        b1 = create_bus(net, vn_kv=20.0)
        b2 = create_bus(net, vn_kv=20.0)
        b3 = create_bus(net, vn_kv=20.0)

        create_ext_grid(net, b0)
        create_line_from_parameters(net, b0, b1, length_km=10, r_ohm_per_km=0.1,
                                    x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        sw0 = create_switch(net, bus=b1, element=b2, et="b")
        sw1 = create_switch(net, bus=b2, element=b3, et="b")
        create_load(net, b2, p_mw=1.0)
        create_load(net, b3, p_mw=2.0)

        runpp(net)
        compute_switch_flows(net)

        # sw1 carries only load at b3
        assert np.isclose(abs(net.res_switch.at[sw1, "p_from_mw"]),
                           net.res_load.at[1, "p_mw"], atol=1e-6)
        # sw0 carries both loads
        total_load = net.res_load.at[0, "p_mw"] + net.res_load.at[1, "p_mw"]
        assert np.isclose(abs(net.res_switch.at[sw0, "p_from_mw"]), total_load, atol=1e-6)


class TestComputeSwitchFlowsBranching:
    """Tree topologies with branches."""

    def test_t_junction(self):
        """            load1=1MW
                          |
        ext_grid--(b0)--line--(b1)--sw0--(b2)--sw1--(b3)--load2=2MW
                                          |
                                         sw2
                                          |
                                        (b4)--load3=3MW

        sw0 carries 1+2+3=6 MW, sw1 carries 2 MW, sw2 carries 3 MW.
        """
        net = create_empty_network()
        b0 = create_bus(net, vn_kv=20.0)
        b1 = create_bus(net, vn_kv=20.0)
        b2 = create_bus(net, vn_kv=20.0)
        b3 = create_bus(net, vn_kv=20.0)
        b4 = create_bus(net, vn_kv=20.0)

        create_ext_grid(net, b0)
        create_line_from_parameters(net, b0, b1, length_km=10, r_ohm_per_km=0.1,
                                    x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        sw0 = create_switch(net, bus=b1, element=b2, et="b")
        sw1 = create_switch(net, bus=b2, element=b3, et="b")
        sw2 = create_switch(net, bus=b2, element=b4, et="b")

        create_load(net, b2, p_mw=1.0)
        create_load(net, b3, p_mw=2.0)
        create_load(net, b4, p_mw=3.0)

        runpp(net)
        compute_switch_flows(net)

        l0 = net.res_load.at[0, "p_mw"]
        l1 = net.res_load.at[1, "p_mw"]
        l2 = net.res_load.at[2, "p_mw"]

        assert np.isclose(abs(net.res_switch.at[sw0, "p_from_mw"]), l0 + l1 + l2, atol=1e-6)
        assert np.isclose(abs(net.res_switch.at[sw1, "p_from_mw"]), l1, atol=1e-6)
        assert np.isclose(abs(net.res_switch.at[sw2, "p_from_mw"]), l2, atol=1e-6)


class TestComputeSwitchFlowsGenAndBranch:
    """Scenarios with generators and outgoing branches within a fused group."""

    def test_generator_on_fused_bus(self):
        """Generator on one fused bus, load on the other.

        (b0)--line--(b1)--sw--(b2)
        ext_grid      gen=5MW   load=3MW

        The switch should carry |gen - load| = 2 MW toward the line.
        But sign depends on direction: the gen produces 5MW at b1,
        load consumes 3MW at b2.  Net demand at b2 is 3MW, so the
        switch carries 3MW from b1 to b2.
        """
        net = create_empty_network()
        b0 = create_bus(net, vn_kv=20.0)
        b1 = create_bus(net, vn_kv=20.0)
        b2 = create_bus(net, vn_kv=20.0)

        create_ext_grid(net, b0)
        create_line_from_parameters(net, b0, b1, length_km=10, r_ohm_per_km=0.1,
                                    x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        sw = create_switch(net, bus=b1, element=b2, et="b")
        create_gen(net, b1, p_mw=5.0, vm_pu=1.0)
        create_load(net, b2, p_mw=3.0)

        runpp(net)
        compute_switch_flows(net)

        load_p = net.res_load.at[0, "p_mw"]
        assert np.isclose(abs(net.res_switch.at[sw, "p_from_mw"]), load_p, atol=1e-6)

    def test_branch_leaving_fused_group(self):
        """Branch exits from a fused bus.

        ext_grid--(b0)--line0--(b1)--sw--(b2)--line1--(b3)--load
        """
        net = create_empty_network()
        b0 = create_bus(net, vn_kv=20.0)
        b1 = create_bus(net, vn_kv=20.0)
        b2 = create_bus(net, vn_kv=20.0)
        b3 = create_bus(net, vn_kv=20.0)

        create_ext_grid(net, b0)
        create_line_from_parameters(net, b0, b1, length_km=10, r_ohm_per_km=0.1,
                                    x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        sw = create_switch(net, bus=b1, element=b2, et="b")
        create_line_from_parameters(net, b2, b3, length_km=5, r_ohm_per_km=0.1,
                                    x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        create_load(net, b3, p_mw=2.0, q_mvar=1.0)

        runpp(net)
        compute_switch_flows(net)

        # The switch must carry what the outgoing line takes from b2
        line1_p = abs(net.res_line.at[1, "p_from_mw"])
        line1_q = abs(net.res_line.at[1, "q_from_mvar"])
        sw_p = abs(net.res_switch.at[sw, "p_from_mw"])
        sw_q = abs(net.res_switch.at[sw, "q_from_mvar"])

        assert np.isclose(sw_p, line1_p, atol=1e-6)
        assert np.isclose(sw_q, line1_q, atol=1e-6)

    def test_sgen_and_shunt(self):
        """Ensure static generators and shunts are accounted for."""
        net = create_empty_network()
        b0 = create_bus(net, vn_kv=20.0)
        b1 = create_bus(net, vn_kv=20.0)
        b2 = create_bus(net, vn_kv=20.0)

        create_ext_grid(net, b0)
        create_line_from_parameters(net, b0, b1, length_km=10, r_ohm_per_km=0.1,
                                    x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        sw = create_switch(net, bus=b1, element=b2, et="b")
        create_sgen(net, b2, p_mw=2.0, q_mvar=0.0)
        create_shunt(net, b2, q_mvar=-0.5, p_mw=0.0)
        create_load(net, b2, p_mw=5.0, q_mvar=1.0)

        runpp(net)
        compute_switch_flows(net)

        # Net demand at b2: load - sgen + shunt
        net_p_b2 = net.res_load.at[0, "p_mw"] - net.res_sgen.at[0, "p_mw"] + net.res_shunt.at[0, "p_mw"]
        assert np.isclose(abs(net.res_switch.at[sw, "p_from_mw"]), abs(net_p_b2), atol=1e-4)


class TestComputeSwitchFlowsCycleDetection:
    """Cycle detection for parallel zero-impedance paths."""

    def test_cycle_raises(self):
        """Two parallel zero-impedance switches between the same buses must raise."""
        net = create_empty_network()
        b0 = create_bus(net, vn_kv=20.0)
        b1 = create_bus(net, vn_kv=20.0)
        b2 = create_bus(net, vn_kv=20.0)

        create_ext_grid(net, b0)
        create_line_from_parameters(net, b0, b1, length_km=10, r_ohm_per_km=0.1,
                                    x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        # Two parallel switches b1 <-> b2
        create_switch(net, bus=b1, element=b2, et="b")
        create_switch(net, bus=b1, element=b2, et="b")
        create_load(net, b2, p_mw=1.0)

        runpp(net)
        with pytest.raises(ValueError, match="cycle"):
            compute_switch_flows(net)

    def test_loop_of_three_raises(self):
        """Three buses in a loop: b1--sw--b2--sw--b3--sw--b1."""
        net = create_empty_network()
        b0 = create_bus(net, vn_kv=20.0)
        b1 = create_bus(net, vn_kv=20.0)
        b2 = create_bus(net, vn_kv=20.0)
        b3 = create_bus(net, vn_kv=20.0)

        create_ext_grid(net, b0)
        create_line_from_parameters(net, b0, b1, length_km=10, r_ohm_per_km=0.1,
                                    x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        create_switch(net, bus=b1, element=b2, et="b")
        create_switch(net, bus=b2, element=b3, et="b")
        create_switch(net, bus=b3, element=b1, et="b")
        create_load(net, b2, p_mw=1.0)

        runpp(net)
        with pytest.raises(ValueError, match="cycle"):
            compute_switch_flows(net)


class TestComputeSwitchFlowsSignConvention:
    """Verify sign convention matches Pandapower's from/to convention."""

    def test_sign_convention(self):
        """Power flows from bus column toward element column when load is on element side."""
        net, b0, b1, b2 = _make_two_bus_coupler_net()
        runpp(net)
        compute_switch_flows(net)

        # Switch: bus=b1, element=b2; load on b2
        # Power should flow from b1 to b2, so p_from > 0, p_to < 0
        p_from = net.res_switch.at[0, "p_from_mw"]
        p_to = net.res_switch.at[0, "p_to_mw"]
        assert p_from > 0, "Power should flow from bus toward element (load side)"
        assert p_to < 0


class TestComputeSwitchFlowsValidation:
    """Cross-validate against z_ohm > 0 results for small impedance."""

    def test_cross_validate_with_impedance(self):
        """Results from nodal balance should be close to results with small z_ohm.

        Uses z_ohm=0.01 (not smaller, since very low impedance branches can
        cause convergence issues in Newton-Raphson for small test networks).
        """
        net = create_empty_network()
        b0 = create_bus(net, vn_kv=20.0)
        b1 = create_bus(net, vn_kv=20.0)
        b2 = create_bus(net, vn_kv=20.0)

        create_ext_grid(net, b0)
        create_line_from_parameters(net, b0, b1, length_km=10, r_ohm_per_km=0.1,
                                    x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        create_switch(net, bus=b1, element=b2, et="b")
        create_load(net, b2, p_mw=1.5, q_mvar=0.5)

        # Run with z_ohm=0 + nodal balance
        runpp(net)
        compute_switch_flows(net)
        p_nodal = net.res_switch.at[0, "p_from_mw"]
        q_nodal = net.res_switch.at[0, "q_from_mvar"]
        i_nodal = net.res_switch.at[0, "i_ka"]

        # Run with small z_ohm for comparison
        net.switch.at[0, "z_ohm"] = 0.01
        runpp(net)
        p_z = net.res_switch.at[0, "p_from_mw"]
        q_z = net.res_switch.at[0, "q_from_mvar"]
        i_z = net.res_switch.at[0, "i_ka"]

        # Tolerances account for the non-zero impedance introducing small losses
        assert np.isclose(p_nodal, p_z, rtol=1e-2), \
            f"P mismatch: nodal={p_nodal:.6f}, z_ohm={p_z:.6f}"
        assert np.isclose(q_nodal, q_z, rtol=1e-2), \
            f"Q mismatch: nodal={q_nodal:.6f}, z_ohm={q_z:.6f}"
        assert np.isclose(i_nodal, i_z, rtol=5e-2), \
            f"I mismatch: nodal={i_nodal:.6f}, z_ohm={i_z:.6f}"


class TestComputeSwitchFlowsMultipleGroups:
    """Independent fused groups should be computed independently."""

    def test_two_independent_groups(self):
        """Two separate fused groups with different loads."""
        net = create_empty_network()
        b0 = create_bus(net, vn_kv=20.0)
        b1 = create_bus(net, vn_kv=20.0)
        b2 = create_bus(net, vn_kv=20.0)
        b3 = create_bus(net, vn_kv=20.0)
        b4 = create_bus(net, vn_kv=20.0)

        create_ext_grid(net, b0)
        create_line_from_parameters(net, b0, b1, length_km=10, r_ohm_per_km=0.1,
                                    x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        create_line_from_parameters(net, b0, b3, length_km=10, r_ohm_per_km=0.1,
                                    x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)

        sw0 = create_switch(net, bus=b1, element=b2, et="b")
        sw1 = create_switch(net, bus=b3, element=b4, et="b")

        create_load(net, b2, p_mw=1.0)
        create_load(net, b4, p_mw=3.0)

        runpp(net)
        compute_switch_flows(net)

        assert np.isclose(abs(net.res_switch.at[sw0, "p_from_mw"]),
                           net.res_load.at[0, "p_mw"], atol=1e-6)
        assert np.isclose(abs(net.res_switch.at[sw1, "p_from_mw"]),
                           net.res_load.at[1, "p_mw"], atol=1e-6)


class TestComputeSwitchFlowsLoadingPercent:
    """Loading percent computation when in_ka is available."""

    def test_loading_percent(self):
        net, b0, b1, b2 = _make_two_bus_coupler_net()
        in_ka = 0.1
        net.switch.at[0, "in_ka"] = in_ka
        runpp(net)
        compute_switch_flows(net)

        i_ka = net.res_switch.at[0, "i_ka"]
        expected_loading = i_ka / in_ka * 100
        assert np.isclose(net.res_switch.at[0, "loading_percent"], expected_loading, atol=1e-6)


class TestComputeSwitchFlowsDcline:
    """DC line as a branch leaving a fused group."""

    def test_dcline_outflow(self):
        """Power leaving through a dcline must be accounted for.

        ext_grid--(b0)--line--(b1)--sw--(b2)--dcline--(b3)--load
        """
        from pandapower.create import create_dcline

        net = create_empty_network()
        b0 = create_bus(net, vn_kv=20.0)
        b1 = create_bus(net, vn_kv=20.0)
        b2 = create_bus(net, vn_kv=20.0)
        b3 = create_bus(net, vn_kv=20.0)

        create_ext_grid(net, b0)
        create_ext_grid(net, b3, vm_pu=1.0)
        create_line_from_parameters(net, b0, b1, length_km=10, r_ohm_per_km=0.1,
                                    x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        sw = create_switch(net, bus=b1, element=b2, et="b")
        create_dcline(net, from_bus=b2, to_bus=b3, p_mw=5.0, loss_percent=1.0,
                      loss_mw=0.1, vm_from_pu=1.0, vm_to_pu=1.0)
        create_load(net, b3, p_mw=10.0)

        runpp(net)
        compute_switch_flows(net)

        # Switch must carry the dcline from-side flow
        p_dcline = abs(net.res_dcline.at[0, "p_from_mw"])
        p_sw = abs(net.res_switch.at[sw, "p_from_mw"])
        assert np.isclose(p_sw, p_dcline, atol=1e-4), \
            f"Switch P={p_sw:.4f} should match dcline from P={p_dcline:.4f}"
        assert net.res_switch.at[sw, "i_ka"] > 0


class TestComputeSwitchFlowsDeenergized:
    """De-energized fused groups should be skipped gracefully."""

    def test_deenergized_group_skipped(self):
        """Fused group with vm=0 should not produce results (no crash)."""
        net, b0, b1, b2 = _make_two_bus_coupler_net()
        runpp(net)

        # Manually set bus voltage to 0 to simulate de-energized state
        net.res_bus.at[b1, "vm_pu"] = 0.0
        net.res_bus.at[b2, "vm_pu"] = 0.0

        compute_switch_flows(net)
        # Should remain NaN (skipped), not crash
        assert np.isnan(net.res_switch.at[0, "p_from_mw"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
