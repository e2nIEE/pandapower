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
from pandapower.protection.utility_functions import plot_tripped_grid_protection_device


try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False


def test_protection_function():
    net = fuse_test_net2()  # create radial network with four switches

    # add fuse object to switch 3, fuse_type = "Siemens NH-2-315"
    f3 = Fuse(net=net, rated_i_a=315, fuse_type="Siemens NH-2-315", switch_index=3)

    # create fault at bus 3, check that fuse melts in 0.1451 seconds
    sc.calc_sc(net, bus=3, branch_results=True)

    protection_result = f3.protection_function(net, scenario="sc")
    assert protection_result['trip_melt'] == True, 'trip_melt should be True'
    assert net.protection.at[0, "object"].tripped == True, 'Fuse.tripped should be True'
    assert np.isclose(protection_result['trip_melt_time_s'],
                      0.145108), 'melt_time_s3 should be close to 0.145108 seconds for fault at bus 3'


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
    HVTrafoFuse.create_characteristic(net, [189, 220, 300, 350, 393, 450, 530, 700, 961],
                                      [10, 2.84, 0.368, 0.164, 0.1, 0.0621, 0.0378, 0.0195, 0.01])
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


def test_fuse_plot_protection_characteristic():
    # test plot_protection_characteristic method of Fuse class
    net = fuse_test_net2()
    Fuse1 = Fuse(net=net, rated_i_a=63, fuse_type="Siemens NH-1-160", switch_index=0)
    Fuse1.plot_protection_characteristic(net)


def test_plot_tripped_grid_protection_device1():
    net = fuse_test_net3()  # create radial network with four switches

    # create list of fuse selections, loop through in for loop
    fuse_list = ["HV 63A", "Siemens NH-2-630", "Siemens NH-2-425", "Siemens NH-2-315", "Siemens NH-2-224"]

    for k in range(5):
        Fuse(net=net, switch_index=k, fuse_type=fuse_list[k])
        sc.calc_sc(net, bus=k, branch_results=True)
        protection_results = calculate_protection_times(net, scenario='sc')
        plot_tripped_grid_protection_device(net, protection_results, sc_bus=k, sc_location=0)


def test_plot_tripped_grid_protection_device2():
    net = fuse_test_net4()
    # assign fuses to switches
    fuse_list = ['HV 25A', 'Siemens NH-2-400', 'Siemens NH-2-250', 'Siemens NH-2-224', 'Siemens NH-2-200',
                 'Siemens NH-1-160', 'Siemens NH-1-125', 'Siemens NH-1-100', 'Siemens NH-1-80']
    for k in range(9):
        Fuse(net=net, switch_index=k, fuse_type=fuse_list[k])

    # perform short circuit calculation at bus 2
    sc.calc_sc(net, bus=3, branch_results=True)

    # calculate protection times
    protection_results = calculate_protection_times(net, scenario='sc')
    plot_tripped_grid_protection_device(net, protection_results, sc_location=0, sc_bus=3)


def test_create_fuse_from_std_type():
    # figure out a clean way to create fuses from std type library
    net = fuse_test_net2()

    # create list of fuse selections, loop through in for loop
    fuse_list = ["HV 63A", "Siemens NH-2-630", "Siemens NH-2-425", "Siemens NH-2-315", "Siemens NH-2-224"]

    for k in range(5):
        Fuse(net=net, switch_index=k, fuse_type=fuse_list[k])
        assert net.protection.at[k, 'object'].fuse_type == fuse_list[k], 'fuse_type should match list selections'


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


def test_calc_prot_times_pp_scenario():
    # test calculate_protection_times() function with power flow scenario
    net = fuse_test_net3()
    Fuse(net=net, switch_index=4, fuse_type="Siemens NH-1-16")  # fuse is underrated
    pp.runpp(net)
    protection_times = calculate_protection_times(net, scenario="pp")
    assert protection_times.trip_melt.at[0] == True, 'trip_melt for switch 4 should be True in pp scenario'


def test_reset_device():
    # test reset_device() method of Fuse class
    net = fuse_test_net3()
    Fuse(net=net, switch_index=0, fuse_type="HV 63A")
    net.protection.at[0, "object"].tripped = True
    assert net.protection.at[0, "object"].tripped == True, 'Fuse should manually trip'

    net.protection.at[0, "object"].reset_device()
    assert net.protection.at[0, "object"].tripped == False, 'Fuse should reset'


def test_has_tripped():
    # test has_tripped() method of Fuse class
    net = fuse_test_net3()
    Fuse(net=net, switch_index=0, fuse_type="HV 63A")
    net.protection.at[0, "object"].tripped = True
    assert net.protection.at[0, "object"].has_tripped() == True, '.has_tripped() should equal True'


def test_status_to_net():
    net = fuse_test_net3()
    Fuse(net=net, switch_index=0, fuse_type="HV 63A")
    net.protection.at[0, "object"].tripped = True
    net.protection.object.at[0].status_to_net(net)
    assert net.switch.closed.at[0] == False, 'switch 0 should be open (Closed = False)'


def test_create_new_std_type():
    # test creating a new fuse std type, adding it to std library, and using it in network
    net = fuse_test_net3()
    new_fuse_data = {'fuse_type': 'New Fuse',
                     'i_rated_a': 15.0,
                     't_avg': 0,
                     't_min': [15, 0.01],
                     't_total': [15, 0.01],
                     'x_avg': 0,
                     'x_min': [20, 2000],
                     'x_total': [50, 5000]}
    create_std_type(net, data=new_fuse_data, name='New Fuse', element="fuse")
    Fuse(net=net, switch_index=4, fuse_type='New Fuse', curve_select=1)
    sc.calc_sc(net, bus=4, branch_results=True)
    protection_results = calculate_protection_times(net, scenario='sc')
    assert protection_results.trip_melt.at[0] == True, 'New Fuse should melt'
    assert protection_results.trip_melt_time_s.at[0] > 0, 'Melt time should be from x_total curve (greater than 0)'


def test_net3():
    # investigate why bus-bus switch isn't working
    net = fuse_test_net3()
    print(net.switch)
    pp.runpp(net)
    print("\n\n\n")
    print(net.res_switch)
    sc.calc_sc(net, bus=4, branch_results=True)
    print("\n\n\n")
    print(net.res_switch_sc)
    assert not np.isnan(net.res_switch.i_ka.at[4]), 'i_ka for switch 4 should not be NaN '
    assert not np.isnan(net.res_switch_sc.ikss_ka.at[4]), 'ikss_ka for switch 4 should not be NaN '


def test_prot_func_tripping():
    # test whether protection_function of Fuse called by calculate_protection_times() updates switch status in net
    net = fuse_test_net3()

    # create list of fuse selections, loop through in for loop
    fuse_list = ["HV 63A", "Siemens NH-2-630", "Siemens NH-2-425", "Siemens NH-2-315", "Siemens NH-2-224"]

    # create fuses at each switch location
    for k in range(5):
        Fuse(net=net, switch_index=k, fuse_type=fuse_list[k])

    # run short-circuit calculation
    sc.calc_sc(net=net, bus=4, branch_results=True)
    df_protection_results = calculate_protection_times(net)

    print(net.switch)


def test_powerflow_simple():
    # test the res_switch output from the pandapower.networks.example_simple()
    net = pp.networks.example_simple()
    pp.runpp(net)
    print(net)
    print(net.switch)
    print(net.res_switch)


def test_json_fuse():
    # test conversion to and from json string, make sure that resulting network still has functional fuse and
    # assert_net_equal

    net = fuse_test_net3()
    fuse_list = ["HV 63A", "Siemens NH-2-630", "Siemens NH-2-425", "Siemens NH-2-315", "Siemens NH-2-224"]
    # create fuses at each switch location
    for k in range(5):
        Fuse(net=net, switch_index=k, fuse_type=fuse_list[k])
    # net2 is copy of net before calc_sc is done
    net2 = pp.from_json_string(pp.to_json(net))
    assert_net_equal(net, net2)

    #pp.runpp(net2)
    sc.calc_sc(net2, bus=3, branch_results=True)

    # net3 is copy of net after calc_sc is run on net
    sc.calc_sc(net, bus=3, branch_results=True)
    net3 = pp.from_json_string(pp.to_json(net))

    protection_results2 = calculate_protection_times(net2, scenario="sc")
    protection_results3 = calculate_protection_times(net3, scenario="sc")

    assert protection_results2.trip_melt.at[3] == True, 'trip_melt Fuse 3 should be True'
    assert np.isclose(protection_results2.trip_melt_time_s.at[3], 0.145108), \
        'melt_time_s3 should be close to 0.145108 seconds for fault at bus 3'
    assert protection_results3.trip_melt.at[3] == True, 'trip_melt Fuse 3 should be True'
    assert np.isclose(protection_results3.trip_melt_time_s.at[3], 0.145108), \
        'melt_time_s3 should be close to 0.145108 seconds for fault at bus 3'


def fuse_test_net3():
    # network with transformer to test HV fuse curve_select
    net = pp.create_empty_network()
    # create buses
    pp.create_buses(net, nr_buses=5, vn_kv=[20, 0.4, 0.4, 0.4, 0.4], index=[0, 1, 2, 3, 4], name=None, type="n",
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


def test_fuse_fault_oc_scenario():
    net = fuse_test_net4()
    print(net.switch)
    assert net.switch.type.at[0] == "fuse"

    # assign fuses to switches
    fuse_list = ['HV 25A', 'Siemens NH-2-400', 'Siemens NH-2-250', 'Siemens NH-2-224', 'Siemens NH-2-200',
                 'Siemens NH-1-160', 'Siemens NH-1-125', 'Siemens NH-1-100', 'Siemens NH-1-80']
    for k in range(9):
        Fuse(net=net, switch_index=k, fuse_type=fuse_list[k])

    # perform short circuit calculation at bus 2
    sc.calc_sc(net, bus=3, branch_results=True)

    # calculate protection times
    protection_results = calculate_protection_times(net, scenario='sc')

    # for fuse with the fastest melting time, melt the corresponding switch
    net.protection.object.at[protection_results.trip_melt_time_s.idxmin()].status_to_net(net)
    assert net.switch.closed.at[3] == False, 'Fuse 3 should melt, switch 3 should be open'

    # after fault is cleared, close the CB at bus 6 so that Load 0 continues to get power
    net.switch.at[9, "closed"] = True

    # perform power flow calculation
    pp.runpp(net)

    # check if any fuses melt after fault has been cleared
    overload_results = calculate_protection_times(net, scenario="pp")
    net.protection.object.at[overload_results.trip_melt_time_s.idxmin()].status_to_net(net)

    pp.runpp(net)
    assert net.switch.closed.at[6] == False, 'Fuse 6 should melt, switch 6 should open'



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


def fuse_test_net4():
    # test network to demonstrate overcurrent protection in tutorial

    net = pp.create_empty_network()

    # create buses
    pp.create_buses(net, nr_buses=9, vn_kv=[20, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                    index=[0, 1, 2, 3, 4, 5, 6, 7, 8], name=None, type="n",
                    geodata=[(0, 0), (0, -2), (-2, -4), (-2, -6), (-2, -8), (2, -4), (2, -6), (2, -8), (2, -10)])

    # create external grid
    pp.create_ext_grid(net, 0, vm_pu=1.0, va_degree=0, s_sc_max_mva=500, s_sc_min_mva=80, rx_max=0.1, rx_min=0.1)

    pp.create_lines(net, from_buses=[1, 2, 3, 1, 5, 6, 7, 6], to_buses=[2, 3, 4, 5, 6, 7, 8, 4],
                    length_km=[0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1], std_type="94-AL1/15-ST1A 0.4")

    net.line["endtemp_degree"] = 250

    # create transformer
    pp.create_transformer(net, hv_bus=0, lv_bus=1, std_type="0.25 MVA 20/0.4 kV")

    # Define trafo fuses
    pp.create_switches(net, buses=[0, 1], elements=[0, 0], et='t', type="fuse")

    # Define line fuses
    pp.create_switches(net, buses=[1, 2, 3, 1, 5, 6, 7], elements=[0, 1, 2, 3, 4, 5, 6], et='l', type="fuse")

    # Define circuit breaker switch
    pp.create_switch(net, bus=6, element=7, et='l', type="CB", closed=False)

    # define load
    pp.create_loads(net, buses=[4, 8], p_mw=[0.1, 0.05], q_mvar=0, const_z_percent=0, const_i_percent=0, name=None,
                    index=[0, 1])

    return net

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
