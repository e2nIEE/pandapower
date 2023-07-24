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
    net = fuse_test_net3()

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

def test_net3():
    # investigate why bus-bus switch isn't working
    net = fuse_test_net3()
   # print(net)
   # print(net.bus)
   # print(net.line)
    print(net.switch)
    pp.runpp(net)
   # print(net.res_bus)
   # print("\n\n\n")
   # print(net.res_line)
    print("\n\n\n")
    print(net.res_switch)
   # print("\n\n\n")
   # print(net.res_load)
   # print("\n\n\n")
    net_sc = copy.deepcopy(net)
    net_sc = sc.calc_sc(net_sc, bus=4, branch_results=True)
   # print(net.res_bus_sc)
   # print("\n\n\n")
   # print(net.res_line_sc)
    print("\n\n\n")
    print(net.res_switch_sc)

    assert not np.isnan(net.res_switch.i_ka.at[4]),           'i_ka for switch 4 should not be NaN '
    assert not np.isnan(net_sc.res_switch_sc.i_kss.at[4]), 'ikss_ka for switch 4 should not be NaN '


    '''# taken from plotting_structural.ipnyb
    try:
        import seaborn
        colors = seaborn.color_palette()
    except:
        colors = ["b", "g", "r", "c", "y"]
    net.bus_geodata.drop(net.bus_geodata.index, inplace=True)
    net.line_geodata.drop(net.line_geodata.index, inplace=True)
    plot.create_generic_coordinates(net, respect_switches=True)  # create artificial coordinates with the igraph package
    plot.fuse_geodata(net)
    bc = plot.create_bus_collection(net, net.bus.index, size=.2, color=colors[0], zorder=10)
    tlc, tpc = plot.create_trafo_collection(net, net.trafo.index, color="g")
    lcd = plot.create_line_collection(net, net.line.index, color="grey", linewidths=0.5, use_bus_geodata=True)
    sc1 = plot.create_bus_collection(net, net.ext_grid.bus.values, patch_type="rect", size=.5, color="y", zorder=11)
    plot.draw_collections([lcd, bc, tlc, tpc, sc1], figsize=(8, 6))'''


def test_powerflow_simple():
    # test the res_switch output from the pandapower.networks.example_simple()
    net = pp.networks.example_simple()
    pp.runpp(net)
    print(net)
    print(net.switch)
    print(net.res_switch)


def test_powerflow_multivoltage():
    # test the res_switch output from the pandapower.networks.example_multivoltage
    net = pp.networks.example_multivoltage()
    pp.runpp(net)
    print(net)
    print(net.switch)
    print(net.res_switch)

def test_modified_simple():
    # include z_ohm values for switches, which will hopefully resolve the NaN isue
    net = modified_simple_net()
    pp.runpp(net)
    print(net)
    print(net.switch)
    print(net.res_switch)


def fuse_test_net3():
    # network with transformer to test HV fuse curve_select
    net = pp.create_empty_network()
    # create buses
    pp.create_buses(net, nr_buses=4, vn_kv=[20, 0.4, 0.4, 0.4, 0.4], index=[0, 1, 2, 3, 4], name=None, type="n",
                    geodata=[(0, 0), (0, -2), (0, -4), (0, -6), (0, -8)])
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
    pp.create_switches(net, buses=[1, 2], elements=[0, 1], et='l', type="fuse")
    # Define load fuse (bus-bus switch)
    pp.create_switch(net, bus=3, element=4, et='b', type="fuse", z_ohm=0.0001)
    # define load
    pp.create_load(net, bus=4, p_mw=0.1, q_mvar=0, const_z_percent=0, const_i_percent=0, sn_mva=.1,
                   name=None, scaling=1., index=0)
    return net


def fuse_test_net2():
    # network with transformer to test HV fuse curve_select
    # load switch (index 4) is configured as bus-bus

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

def modified_simple_net():
    # modify simple network from create_examples.py to include resistance values for switches
    # this should hopefully the NaN issue during power flow and short circuit calculations

    net = pp.create_empty_network()

    # create buses
    bus1 = pp.create_bus(net, name="HV Busbar", vn_kv=110., type="b")
    bus2 = pp.create_bus(net, name="HV Busbar 2", vn_kv=110., type="b")
    bus3 = pp.create_bus(net, name="HV Transformer Bus", vn_kv=110., type="n")
    bus4 = pp.create_bus(net, name="MV Transformer Bus", vn_kv=20., type="n")
    bus5 = pp.create_bus(net, name="MV Main Bus", vn_kv=20., type="b")
    bus6 = pp.create_bus(net, name="MV Bus 1", vn_kv=20., type="b")
    bus7 = pp.create_bus(net, name="MV Bus 2", vn_kv=20., type="b")

    # create external grid
    pp.create_ext_grid(net, bus1, vm_pu=1.02, va_degree=50)

    # create transformer
    pp.create_transformer(net, bus3, bus4, name="110kV/20kV transformer",
                          std_type="25 MVA 110/20 kV")
    # create lines
    pp.create_line(net, bus1, bus2, length_km=10,
                   std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV", name="Line 1")
    line2 = pp.create_line(net, bus5, bus6, length_km=2.0,
                           std_type="NA2XS2Y 1x240 RM/25 12/20 kV", name="Line 2")
    line3 = pp.create_line(net, bus6, bus7, length_km=3.5,
                           std_type="48-AL1/8-ST1A 20.0", name="Line 3")
    line4 = pp.create_line(net, bus7, bus5, length_km=2.5,
                           std_type="NA2XS2Y 1x240 RM/25 12/20 kV", name="Line 4")

    # create bus-bus switches
    pp.create_switch(net, bus2, bus3, et="b", type="CB", z_ohm=0.1)
    pp.create_switch(net, bus4, bus5, et="b", type="CB", z_ohm=0.1)

    # create bus-line switches
    pp.create_switch(net, bus5, line2, et="l", type="LBS", closed=True, z_ohm=0.1)
    pp.create_switch(net, bus6, line2, et="l", type="LBS", closed=True, z_ohm=0.1)
    pp.create_switch(net, bus6, line3, et="l", type="LBS", closed=True, z_ohm=0.1)
    pp.create_switch(net, bus7, line3, et="l", type="LBS", closed=False, z_ohm=0.1)
    pp.create_switch(net, bus7, line4, et="l", type="LBS", closed=True, z_ohm=0.1)
    pp.create_switch(net, bus5, line4, et="l", type="LBS", closed=True, z_ohm=0.1)

    # create load
    pp.create_load(net, bus7, p_mw=2, q_mvar=4, scaling=0.6, name="load")

    # create generator
    pp.create_gen(net, bus6, p_mw=6, max_q_mvar=3, min_q_mvar=-3, vm_pu=1.03,
                  name="generator")

    # create static generator
    pp.create_sgen(net, bus7, p_mw=2, q_mvar=-0.5, name="static generator")

    # create shunt
    pp.create_shunt(net, bus3, q_mvar=-0.96, p_mw=0, name='Shunt')

    return net
