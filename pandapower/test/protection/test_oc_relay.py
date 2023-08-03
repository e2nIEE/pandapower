import pytest
import copy
import numpy as np
import pandapower as pp
from pandapower import load_std_type, create_std_type
from pandapower.control import plot_characteristic
from pandapower.protection.protection_devices.fuse import Fuse
from pandapower.protection.run_protection import calculate_protection_times
import pandapower.shortcircuit as sc
from pandapower.test.helper_functions import assert_net_equal
from pandapower.protection.protection_devices.ocrelay import OCRelay
from pandapower.protection.utility_functions import create_sc_bus

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False


def test_oc_relay_dtoc():
    net = oc_relay_net()

    for k in range(6):
        OCRelay(net, switch_index=k, oc_relay_type='DTOC', time_settings=[0.07, 0.5, 0.3])

    net_sc = create_sc_bus(net, sc_line_id=4, sc_fraction=0.5)
    sc.calc_sc(net_sc, bus=max(net_sc.bus.index), branch_results=True)
    protection_results = calculate_protection_times(net_sc, scenario='sc')
    assert protection_results.trip_melt_time_s.at[0] == 1.4
    assert protection_results.trip_melt_time_s.at[2] == 1.1
    assert protection_results.trip_melt_time_s.at[4] == 0.07


def test_oc_relay_idmt():
    net = oc_relay_net()
    net.switch.type = 'CB_IDMT'

    for k in range(6):
        OCRelay(net, switch_index=k, oc_relay_type='IDMT', time_settings=[1, 0.5])

    net_sc = create_sc_bus(net, sc_line_id=4, sc_fraction=0.5)
    sc.calc_sc(net_sc, bus=max(net_sc.bus.index), branch_results=True)
    protection_results = calculate_protection_times(net_sc, scenario='sc')
    #todo: double check what protection results should be with Gourab
    assert protection_results.trip_melt_time_s.at[0] == 1.4
    assert protection_results.trip_melt_time_s.at[2] == 1.1
    assert protection_results.trip_melt_time_s.at[4] == 0.07


def test_oc_relay_idtoc():
    net = oc_relay_net()
    net.switch.type = 'CB_IDTOC'

    for k in range(6):
        OCRelay(net, switch_index=k, oc_relay_type='IDTOC', time_settings=[0.07, 0.5, 0.3, 1, 0.5])

    net_sc = create_sc_bus(net, sc_line_id=4, sc_fraction=0.5)
    sc.calc_sc(net_sc, bus=max(net_sc.bus.index), branch_results=True)
    protection_results = calculate_protection_times(net_sc, scenario='sc')
    #todo: double check what protection results should be with Gourab
    assert protection_results.trip_melt_time_s.at[0] == 1.4
    assert protection_results.trip_melt_time_s.at[2] == 1.1
    assert protection_results.trip_melt_time_s.at[4] == 0.07


def oc_relay_net():
    import pandapower as pp

    # create an empty network
    net = pp.create_empty_network()

    # create buses
    pp.create_buses(net, nr_buses=7, vn_kv=20, index=[0, 1, 2, 3, 4, 5, 6], name=None, type="n",
                    geodata=[(0, 0), (0, -1), (-2, -2), (-2, -4), (2, -2), (2, -3), (2, -4)])

    # create external grids
    pp.create_ext_grid(net, 0, vm_pu=1.0, va_degree=0, s_sc_max_mva=100, s_sc_min_mva=50, rx_max=0.1, rx_min=0.1)

    pp.create_lines(net, from_buses=[0, 1, 2, 1, 4, 5], to_buses=[1, 2, 3, 4, 5, 6], length_km=[2, 5, 4, 4, 0.5, 0.5],
                    std_type="NAYY 4x50 SE",
                    name=None, index=[0, 1, 2, 3, 4, 5], df=1., parallel=1)

    net.line["endtemp_degree"] = 250

    # Define switches
    pp.create_switches(net, buses=[0, 1, 1, 2, 4, 5], elements=
                       [0, 1, 3, 2, 4, 5], et='l', type="CB_DTOC")
    # define load
    pp.create_loads(net, buses=[3, 6], p_mw=[5, 2], q_mvar=[1, 1], const_z_percent=0, const_i_percent=0, sn_mva=None,
                    name=None, scaling=1., index=[0, 1])

    return net




