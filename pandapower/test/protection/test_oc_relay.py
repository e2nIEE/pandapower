import numpy as np
import pandapower as pp
from pandapower.protection.run_protection import calculate_protection_times
import pandapower.shortcircuit as sc
from pandapower.protection.protection_devices.ocrelay import OCRelay
from pandapower.protection.utility_functions import create_sc_bus
from pandapower.protection.utility_functions import plot_tripped_grid_protection_device

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
    assert np.isclose(protection_results.trip_melt_time_s.at[0], 4.8211)
    assert np.isclose(protection_results.trip_melt_time_s.at[2], 4.3211)
    assert np.isclose(protection_results.trip_melt_time_s.at[4], 3.8211)


def test_oc_relay_idtoc():
    net = oc_relay_net()
    net.switch.type = 'CB_IDTOC'
    for k in range(6):
        OCRelay(net, switch_index=k, oc_relay_type='IDTOC', time_settings=[0.07, 0.5, 0.3, 1, 0.5])
    net_sc = create_sc_bus(net, sc_line_id=4, sc_fraction=0.5)
    sc.calc_sc(net_sc, bus=max(net_sc.bus.index), branch_results=True)
    protection_results = calculate_protection_times(net_sc, scenario='sc')
    assert protection_results.trip_melt_time_s.at[0] == 1.4
    assert protection_results.trip_melt_time_s.at[2] == 1.1
    assert protection_results.trip_melt_time_s.at[4] == 0.07


def test_oc_relay_plots():
    net = oc_relay_net()
    net.switch.at[2, "type"] = 'CB_IDMT'
    net.switch.at[3, "type"] = 'CB_IDMT'
    net.switch.at[4, "type"] = 'CB_IDTOC'
    net.switch.at[5, "type"] = 'CB_IDTOC'

    oc_relay_type_list = ['DTOC', 'DTOC', 'IDMT', 'IDMT', 'IDTOC', 'IDTOC']
    time_settings_list = [[0.07, 0.5, 0.3],
                          [0.07, 0.5, 0.3],
                          [1, 0.4],
                          [1, 0.4],
                          [0.07, 0.5, 0.3, 1, 0.4],
                          [0.07, 0.5, 0.3, 1, 0.4]]

    for k in range(6):
        OCRelay(net, switch_index=k, oc_relay_type=oc_relay_type_list[k], time_settings=time_settings_list[k])

    net_sc = create_sc_bus(net, sc_line_id=4, sc_fraction=0.5)
    sc.calc_sc(net_sc, bus=max(net_sc.bus.index), branch_results=True)
    protection_results = calculate_protection_times(net_sc, scenario='sc')
    print('\n#################################################\n\n')
    print(protection_results)
    for k in range(6):
        plt.figure(k)
        net.protection.object.at[k].plot_protection_characteristic(net=net)


def test_select_k_alpha():
    net = oc_relay_net()
    net.switch.type = 'CB_IDMT'

    inverse_type_list = ['standard_inverse', 'very_inverse', 'extremely_inverse', 'long_inverse',
                         'standard_inverse', 'standard_inverse']
    for q in range(6):
        OCRelay(net, switch_index=q, oc_relay_type='IDMT', time_settings=[1, 0.5],
                curve_type=inverse_type_list[q])

    k_list = [0.14, 13.5, 80, 120, 0.14, 0.14]
    alpha_list = [0.02, 1, 2, 1, 0.02, 0.02]
    for q in range(6):
        assert net.protection.object.at[q].k == k_list[q]
        assert net.protection.object.at[q].alpha == alpha_list[q]


def test_plot_tripped_grid_protection_device():
    net = oc_relay_net()

    for k in range(6):
        OCRelay(net, switch_index=k, oc_relay_type='DTOC', time_settings=[0.07, 0.5, 0.3])

    net_sc = create_sc_bus(net, sc_line_id=2, sc_fraction=0.5)
    sc.calc_sc(net_sc, bus=max(net_sc.bus.index), branch_results=True)
    protection_results = calculate_protection_times(net_sc, scenario='sc')
    plot_tripped_grid_protection_device(net_sc, protection_results, sc_bus=max(net_sc.bus.index), sc_location=0.5)


def oc_relay_net():
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




