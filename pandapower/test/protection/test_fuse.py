import copy

import numpy as np

import pandapower as pp
from pandapower import load_std_type
from pandapower.control import plot_characteristic
from pandapower.protection.protection_devices.fuse import Fuse
from pandapower.protection.run_protection import calculate_protection_times
import pandapower.shortcircuit as sc
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False


def test_fuse_protection_function_old():
    # NO LONGER VALID; protection_function() WILL NOW RETURN DICTIONARY WITH MORE DETAILED RESULTS
    net = fuse_test_net2()  # create radial network with four switches
    f3 = Fuse(net=net, rated_i_a=315, fuse_type="Siemens NH-2-315",
              switch_index=3)  # add fuse object to switch 3, fuse_type = "Siemens NH-2-315"
    f3.create_characteristic(net, [550, 920, 1900, 5000, 11000],
                             [4800, 120, 7, 0.1, 0.004])  # manually enter x and y values for fuse characteristic
    # plot_characteristic(net.characteristic.at[f.characteristic_index, "object"], 0, 1,
    #                     xlabel="Current I (kA)", ylabel="Time t (s)")

    # create fault at bus 3, check that fuse melts in 0.1451 seconds
    net_sc3 = net
    sc.calc_sc(net_sc3, bus=3, branch_results=True)
    print("net_sc3 has fault at bus 3")
    print(net_sc3)

    melt_time_s3 = f3.protection_function(net_sc3)
    print("melt_time_s : ", melt_time_s3)
    assert np.isclose(melt_time_s3, 0.145108), 'melt_time_s3 should be close to 0.145108 seconds for fault at bus 3'
    assert f3.has_tripped(), 'f3 should be tripped'
    assert not net_sc3.switch.at[f3.switch_index, 'closed'], 'switch 3 should be open (closed == False)'

    # create fault at bus 2, check that fuse does not melt (should equal inf)
    net_sc2 = net
    sc.calc_sc(net_sc2, bus=2, branch_results=True)
    print("net_sc2 has fault at bus 2")
    melt_time_s2 = f3.protection_function(net_sc2)
    print("melt_time_s : ", melt_time_s2)
    assert np.isinf(melt_time_s2), 'melt_time_s2 should be inf'
    assert not f3.has_tripped(), 'f3 should not be tripped'
    assert net_sc2.switch.at[f3.switch_index, 'closed'], 'switch 3 should be closed (closed == True)'

    # calculate_protection_times(net_sc)

def test_protection_function_new():
    net = fuse_test_net2()  # create radial network with four switches
    f3 = Fuse(net=net, rated_i_a=315, fuse_type="Siemens NH-2-315",
              switch_index=3)  # add fuse object to switch 3, fuse_type = "Siemens NH-2-315"
    f3.create_characteristic(net, [550, 920, 1900, 5000, 11000],
                             [4800, 120, 7, 0.1, 0.004])  # manually enter x and y values for fuse characteristic
    # plot_characteristic(net.characteristic.at[f.characteristic_index, "object"], 0, 1,
    #                     xlabel="Current I (kA)", ylabel="Time t (s)")

    # create fault at bus 3, check that fuse melts in 0.1451 seconds
    net_sc3 = net
    sc.calc_sc(net_sc3, bus=3, branch_results=True)
    print("\nnet_sc3 has fault at bus 3\n")
    print(net_sc3)

    protection_result = f3.protection_function(net_sc3)
    print(protection_result)
    assert np.isclose(protection_result['act_time_s'], 0.145108), 'melt_time_s3 should be close to 0.145108 seconds for fault at bus 3'
    assert not net_sc3.switch.at[f3.switch_index, 'closed'], 'switch 3 should be open (closed == False)'

def test_calculate_protection_times():
    # test calculate_protection_times() from run_protection.py

    # create radial network
    net = fuse_test_net2()
    # add five fuses; HV trafo, LV trafo, line0, line1 and load
    HVTrafoFuse = Fuse(net=net, rated_i_a=63, fuse_type="HV 63A", switch_index=0)
    LVTrafoFuse = Fuse(net=net, rated_i_a=630, fuse_type="Siemens NH-2-630", switch_index=1)
    Line0Fuse = Fuse(net=net, rated_i_a=425, fuse_type="Siemens NH-2-425", switch_index=2)
    Line1Fuse = Fuse(net=net, rated_i_a=315, fuse_type="Siemens NH-2-315", switch_index=3)
    LoadFuse = Fuse(net=net, rated_i_a=224, fuse_type="Siemens NH-2-224", switch_index=4)

    # add fuse characteristics (done manually for now because no function to automate this (yet))
    HVTrafoFuse.create_characteristic(net, [189, 220, 300, 350, 393, 450, 530, 700, 961], [10, 2.84, 0.368, 0.164, 0.1, 0.0621, 0.0378, 0.0195, 0.01])
    LVTrafoFuse.create_characteristic(net, [1200, 2000, 4800, 12000, 26000], [4800, 120, 7, 0.1, 0.004])
    Line0Fuse.create_characteristic(net, [850, 1500, 3050, 7500, 16500], [4800, 120, 7, 0.1, 0.004])
    Line1Fuse.create_characteristic(net, [550, 920, 1900, 5000, 11000], [4800, 120, 7, 0.1, 0.004])
    LoadFuse.create_characteristic(net, [400, 750, 1453, 3025, 4315, 7600], [4800, 120, 7, 0.2, 0.04, 0.004])

    # create fault at bus 3
    net_sc3 = copy.deepcopy(net)
    sc.calc_sc(net_sc3, bus=3, branch_results=True)
    print("\nnet_sc3 has fault at bus 3\n")
    print(net_sc3)

    # evaluate results from calculate_protection_times
    df_protection_results3 = calculate_protection_times(net_sc3)

    # create fault at bus 2
    net_sc2 = copy.deepcopy(net)
    sc.calc_sc(net_sc2, bus=2, branch_results=True)
    print("\nnet_sc2 has fault at bus 2\n")
    print(net_sc2)

    df_protection_results2 = calculate_protection_times(net_sc2)


def test_plot_characteristic():
    # create net and one fuse
    net = fuse_test_net2()
    HVTrafoFuse = Fuse(net=net, rated_i_a=63, fuse_type="HV 63A", switch_index=0)
    HVTrafoFuse.create_characteristic(net, [189, 220, 300, 350, 393, 450, 530, 700, 961],
                                      [10, 2.84, 0.368, 0.164, 0.1, 0.0621, 0.0378, 0.0195, 0.01])
    print(net.characteristic)
    plot_characteristic(net.characteristic.at[HVTrafoFuse.characteristic_index, "object"], start=189, stop=961)
    plt.show()
    # result: plot_characteristic produces regular plot. Need log-log plot for fuses. Create plot method for fuse instead

def test_fuse_plot_protection_characteristic():
    # test plot_protection_characteristic method of Fuse class
    net = fuse_test_net2()
    HVTrafoFuse = Fuse(net=net, rated_i_a=63, fuse_type="HV 63A", switch_index=0)
    HVTrafoFuse.create_characteristic(net, [189, 220, 300, 350, 393, 450, 530, 700, 961],
                                      [10, 2.84, 0.368, 0.164, 0.1, 0.0621, 0.0378, 0.0195, 0.01])
    HVTrafoFuse.plot_protection_characteristic(net)
    plt.show()


def test_create_fuse_from_std_type():
    # figure out a clean way to create fuses from std type library
    net = fuse_test_net2()

    # create list of fuse selections, loop through in for loop
    fuse_list = ["HV 63A", "Siemens NH-2-630", "Siemens NH-2-425", "Siemens NH-2-315", "Siemens NH-2-224"]

    for k in range(5):
        Fuse(net=net, switch_index=k, fuse_type=fuse_list[k])

    fuse_info = load_std_type(net=net, name="HV 10A", element="fuse")
    print(fuse_info)

    print(net)
    print(net.protection.at[0, 'object'])

def test_calc_prot_times_with_std_lib():
    net = fuse_test_net2()

    # create list of fuse selections, loop through in for loop
    fuse_list = ["HV 63A", "Siemens NH-2-630", "Siemens NH-2-425", "Siemens NH-2-315", "Siemens NH-2-224"]

    for k in range(5):
        Fuse(net=net, switch_index=k, fuse_type=fuse_list[k])

    # test faults at each bus
    print(net.bus)
    for row_index, row in net.bus.iterrows():
        net_sc = copy.deepcopy(net)
        sc.calc_sc(net_sc, bus=row_index, branch_results=True)
        print("\nnet_sc has fault at bus " + str(row_index) + '\n')
        df_protection_results = calculate_protection_times(net_sc)


def fuse_test_net2():
    # network with transformer to test HV fuse curve_select

    net = pp.create_empty_network()

    # create buses
    pp.create_buses(net, nr_buses=4, vn_kv=[20, 0.4, 0.4, 0.4], index=[0, 1, 2, 3], name=None, type="n",
                    geodata=[(0, 0), (0, -2), (0, -4), (0, -6)])

    # create external grid
    pp.create_ext_grid(net, 0, vm_pu=1.0, va_degree=0, s_sc_max_mva=100, s_sc_min_mva=50, rx_max=0.1, rx_min=0.1)

    pp.create_lines_from_parameters(net, from_buses=[1, 2], to_buses=[2, 3], length_km=[0.1, 0.1], r_ohm_per_km=0.2067,
                                    x_ohm_per_km=0.080424, c_nf_per_km=261, name=None, index=[0, 1], max_i_ka=0.27)

    net.line["endtemp_degree"] = 250

    # create transformer
    pp.create_transformer(net, hv_bus=0, lv_bus=1, std_type="0.63 MVA 20/0.4 kV")

    # Define trafo fuses
    pp.create_switches(net, buses=[0, 1], elements=[0, 0], et='t', type="fuse")

    # Define line fuses
    pp.create_switches(net, buses=[1, 2, 3], elements=[0, 1, 1], et='l', type="fuse")

    # define load
    pp.create_load(net, bus=3, p_mw=0.1, q_mvar=0, const_z_percent=0, const_i_percent=0, sn_mva=.1,
                   name=None, scaling=1., index=0)

    return net


def fuse_test_net3():
    # network with transformer to test HV fuse curve_select
    # load switch (index 4) is configured as bus-bus

    net = pp.create_empty_network()

    # create buses
    pp.create_buses(net, nr_buses=5, vn_kv=[20, 0.4, 0.4, 0.4, 0.4], index=[0, 1, 2, 3], name=None, type="n",
                    geodata=[(0, 0), (0, -2), (0, -4), (0, -6)])

    # create external grid
    pp.create_ext_grid(net, 0, vm_pu=1.0, va_degree=0, s_sc_max_mva=100, s_sc_min_mva=50, rx_max=0.1, rx_min=0.1)

    pp.create_lines_from_parameters(net, from_buses=[1, 2], to_buses=[2, 3], length_km=[0.1, 0.1], r_ohm_per_km=0.2067,
                                    x_ohm_per_km=0.080424, c_nf_per_km=261, name=None, index=[0, 1], max_i_ka=0.27)

    net.line["endtemp_degree"] = 250

    # create transformer
    pp.create_transformer(net, hv_bus=0, lv_bus=1, std_type="0.63 MVA 20/0.4 kV")

    # Define trafo fuses
    pp.create_switches(net, buses=[0, 1], elements=[0, 0], et='t', type="fuse")

    # Define line fuses
    pp.create_switches(net, buses=[1, 2, 3], elements=[0, 1, 1], et='l', type="fuse")

    # define load
    pp.create_load(net, bus=3, p_mw=0.1, q_mvar=0, const_z_percent=0, const_i_percent=0, sn_mva=.1,
                   name=None, scaling=1., index=0)

    return net
