__author__ = 'jdollichon'

import os

import pytest
import copy
import pandapower.networks as nw
import control
import timeseries
from tests import ppp_dir
epsilon = 0.00000000000001


def test_data_source():
    """
    Testing simply reading from file and checking the data.
    """
    # load file

    fn = os.path.join(ppp_dir, "tests", "timeseries", "test_files", "small_profile.csv")

    my_data_source = timeseries.CsvData(fn, sep=";")
    copy.deepcopy(my_data_source)

    # # print data_sources.time_df
    # for i in xrange(len(data_sources.time_df.index)):
    #     for j in xrange(len(data_sources.time_df.columns)):
    #         print data_sources.get_time_step_values(i, j)
    #         pass

    # check a few of the values
    # (profile_name can be the actual name but also the column number)
    assert my_data_source.get_time_step_value(time_step=1, profile_name="my_profilename") == 0.0
    assert my_data_source.get_time_step_value(time_step=4, profile_name="my_profilename") == 0.0
    assert abs(my_data_source.get_time_step_value(time_step=5, profile_name="my_profilename")
               - -3.97E-1) < epsilon
    assert abs(my_data_source.get_time_step_value(time_step=9, profile_name="constload3")
               - -5.37E-3) < epsilon


def test_datasource_deepcopy():
    # leads to error: No such file or directory: 'control\\tests\\res\\small_profile.csv'
    fn = os.path.join(ppp_dir, "tests", "control", "res", "small_profile.csv")

    my_data_source = timeseries.CsvData(fn, sep=";")

    net = nw.simple_four_bus_system()
    control.ConstControl(net, 'load', 'p_kw', 0, my_data_source, 'P_PV_load_loadbus_1_2')

    # # this is only suitable for manual testing
    # for i in range(10000):
    #     n = copy.deepcopy(net)


def test_data_source_big_profile():
    """
    Testing simply reading from file and checking the data.
    """
    # load file

    fn = os.path.join(ppp_dir, "tests", "timeseries", "test_files", "big_profile.csv")

    my_data_source = timeseries.CsvData(fn, sep=";")

    # # print data_sources.time_df
    # for i in xrange(len(data_sources.time_df.index)):
    #     for j in xrange(len(data_sources.time_df.columns)):
    #         print data_sources.get_time_step_values(i, j)
    #         pass

    # check a few of the values
    # (profile_name can be the actual name but also the column number)
    assert abs(my_data_source.get_time_step_value(time_step=0, profile_name="p_H0")
               - 163.1904948) < epsilon
    assert abs(my_data_source.get_time_step_value(time_step=1, profile_name="p_H0")
               - 151.257003) < epsilon

    assert abs(my_data_source.get_time_step_value(time_step=16065, profile_name="p_H0")
               - 177.6462054) < epsilon
    assert abs(my_data_source.get_time_step_value(time_step=25754, profile_name="p_H0")
               - 179.8798932) < epsilon

    assert abs(my_data_source.get_time_step_value(time_step=35034, profile_name="p_H0")
               - 190.251144) < epsilon
    assert abs(my_data_source.get_time_step_value(time_step=35035, profile_name="p_H0")
               - 177.6432306) < epsilon


#
# def test_data_source_mocking_control_handler():
#
#     # load file to data source
#     fn = os.path.join(os.path.dirname(__file__), 'res\small_profile.csv')
#     my_data_source = ct.CsvData(fn, sep=";")
#
#     # create my net with as many loads as in the datasource
#     net = networks.create_kerber_landnetz_freileitung_1(n_lines=8, p_load_in_kw=5)
#
#     #build controller
#     ctrl_list = []
#     load_idx = range(len(net["load"]))
#     p_ac = 0.85
#
#     # this will be our ConstLoad later ;)
#     const = None
#
#     for load in load_idx:
#         bus = net["load"].loc[load, "bus"].astype(int)
#         p = net["load"].loc[load, "p_kw"]
#         q = net["load"].loc[load, "q_kvar"]
#         rated_power = 1.
#         load_name = net["load"].loc[load, "name"]
#
#         # look up profile name from profile_table
#         profile_table_path = os.path.join(os.path.dirname(__file__), 'res\small_profile_table.csv')
#         profile_table = pd.DataFrame(pd.read_csv(profile_table_path, header=0, sep=';',
#                                                 index_col=0,
#                                                 decimal='.',
#                                                 thousands=None))
#
#         # read profile, strategy etc from profile.csv
#         p_name = profile_table.loc[load_name]["profile"]
#         strat = profile_table.loc[load_name]["strategy"]
#         cos_phi = profile_table.loc[load_name]["cosphi"]
#
#
#         if strat == "cosphi":
#             ctrl_list.append(ct.CosphiPv(bus, p, q, rated_power, p_ac=p_ac, data_source=my_data_source, profile_name=p_name))
#         elif strat == "QPofU":
#             ctrl_list.append(ct.PqofuPv(bus, p, q, rated_power, p_ac=p_ac, data_source=my_data_source, profile_name=p_name))
#         elif strat == "QofU70":
#             ctrl_list.append(ct.QofuLimPv(bus, p, q, rated_power, p_ac=p_ac, data_source=my_data_source, profile_name=p_name))
#         elif strat == "QofU":
#             ctrl_list.append(ct.QofuPv(bus, p, q, rated_power, p_ac=p_ac, data_source=my_data_source, profile_name=p_name))
#         elif strat == "pvnoctrl":
#             ctrl_list.append(ct.PvNoControl(bus, p, q, rated_power, data_source=my_data_source, profile_name=p_name))
#         elif strat == "constload":
#             if const is None:
#                 const = ct.ConstLoadMulti(data_source=my_data_source, cos_phi=cos_phi)
#                 ctrl_list.append(const)
#             const.add_load(net, bus, p, q, name=load_name, profile_name=p_name)
#
#     time = -1
#
#     # mocking the control handler
#     while time <= 7:
#         time += 1
#
#         # call time step
#         for ctrl in ctrl_list:
#             ctrl.time_step(net, time)
#
#         # assert they got new values each time step
#         if time <= 3:
#             for ctrl in ctrl_list:
#                 if isinstance(ctrl, ct.ConstLoadMulti):
#                     npt.assert_almost_equal(ctrl.loads[:, 2], np.array([0.0, 0.0, 0.0]))
#                 else:
#                     npt.assert_almost_equal(ctrl.current_p, 0.0)
#         if time == 4:
#             for ctrl in ctrl_list:
#                 if isinstance(ctrl, ct.ConstLoadMulti):
#                     npt.assert_almost_equal(ctrl.loads[:, 2], np.array([-1.99E-1,-1.99E-1,-3.97E-1]))
#                 elif isinstance(ctrl, ct.PqofuPv):
#                     npt.assert_almost_equal(ctrl.p_kw, -1.99E-1)
#                 else:
#                     npt.assert_almost_equal(ctrl.current_p, -3.97E-1)
#
#         if time == 8:
#             for ctrl in ctrl_list:
#                 if isinstance(ctrl, ct.ConstLoadMulti):
#                     npt.assert_almost_equal(ctrl.loads[:, 2], np.array([-2.69E-2,-2.69E-2,-5.37E-3]))
#                 elif isinstance(ctrl, ct.PqofuPv):
#                     npt.assert_almost_equal(ctrl.p_kw, -2.69E-2)
#                 else:
#                     npt.assert_almost_equal(ctrl.current_p, -5.37E-3)
#
#         pp.runpp(net)
#
#         # call control step
#         for ctrl in ctrl_list:
#             ctrl.control_step(net)
#
# def test_data_source_scale_data_source():
#
#     # load file to data source
#     fn = os.path.join(os.path.dirname(__file__), 'res\small_profile.csv')
#     my_data_source = ct.CsvData(fn, resolution_sec=1, sep=";")
#
#     # create my net with as many loads as in the csv
#     net = networks.create_kerber_landnetz_freileitung_1(n_lines=8, p_load_in_kw=5)
#
#     #build controller
#     ctrl_list = []
#     load_idx = range(len(net["load"]))
#     p_ac = 0.85
#     profile_scale_factor = 2.
#
#     # this will be our ConstLoad later ;)
#     const = None
#
#     for load in load_idx:
#         bus = net["load"].loc[load, "bus"].astype(int)
#         p = net["load"].loc[load, "p_kw"]
#         q = net["load"].loc[load, "q_kvar"]
#         rated_power = 1.
#         load_name = net["load"].loc[load, "name"]
#
#         # look up profile name from profile_table
#         profile_table_path = os.path.join(os.path.dirname(__file__), 'res\small_profile_table.csv')
#         profile_table = pd.DataFrame(pd.read_csv(profile_table_path, header=0, sep=';',
#                                                 index_col=0,
#                                                 decimal='.',
#                                                 thousands=None))
#
#         # read profile, strategy etc from profile
#         p_name = profile_table.loc[load_name]["profile"]
#         strat = profile_table.loc[load_name]["strategy"]
#         cos_phi = profile_table.loc[load_name]["cosphi"]
#
#
#         if strat == "cosphi":
#             ctrl_list.append(ct.CosphiPv(bus, p, q, rated_power, p_ac=p_ac, cos_phi=cos_phi, data_source=my_data_source, profile_name=p_name, profile_scale=profile_scale_factor))
#         elif strat == "QPofU":
#             ctrl_list.append(ct.PqofuPv(bus, p, q, rated_power, p_ac=p_ac, data_source=my_data_source, profile_name=p_name, profile_scale=profile_scale_factor))
#         elif strat == "QofU70":
#             ctrl_list.append(ct.QofuLimPv(bus, p, q, rated_power, p_ac=p_ac, data_source=my_data_source, profile_name=p_name, profile_scale=profile_scale_factor))
#         elif strat == "QofU":
#             ctrl_list.append(ct.QofuPv(bus, p, q, rated_power, p_ac=p_ac, data_source=my_data_source, profile_name=p_name, profile_scale=profile_scale_factor))
#         elif strat == "pvnoctrl":
#             ctrl_list.append(ct.PvNoControl(bus, p, q, rated_power, data_source=my_data_source, profile_name=p_name, profile_scale=profile_scale_factor))
#
#     time = -1
#
#     # mocking the control handler
#     while time <= 7:
#         time += 1
#
#         # call time step
#         for ctrl in ctrl_list:
#             ctrl.time_step(net, time)
#
#         # assert they got new values each time step
#         if time <= 3:
#             for ctrl in ctrl_list:
#                 if isinstance(ctrl, ct.ConstLoadMulti):
#                     npt.assert_almost_equal(ctrl.loads[:, 2], np.array([0.0, 0.0, 0.0])*profile_scale_factor)
#                 else:
#                     npt.assert_almost_equal(ctrl.current_p, 0.0*profile_scale_factor)
#         if time == 4:
#             for ctrl in ctrl_list:
#                 if isinstance(ctrl, ct.ConstLoadMulti):
#                     npt.assert_almost_equal(ctrl.loads[:, 2], np.array([-1.99E-1,-1.99E-1,-3.97E-1])*profile_scale_factor)
#                 elif isinstance(ctrl, ct.PqofuPv):
#                     npt.assert_almost_equal(ctrl.p_kw, -1.99E-1*profile_scale_factor)
#                 elif isinstance(ctrl, ct.QofuLimPv):
#                     limiting_factor = ctrl.p_ac if ctrl.p_ac < ctrl.p_limit else ctrl.p_limit
#                     npt.assert_almost_equal(ctrl.p_kw, np.clip(-3.97E-1*profile_scale_factor, -abs(ctrl.sn_kva*limiting_factor), 0))
#                 else:
#                     npt.assert_almost_equal(ctrl.current_p, -3.97E-1*profile_scale_factor)
#
#         if time == 8:
#             for ctrl in ctrl_list:
#                 if isinstance(ctrl, ct.ConstLoadMulti):
#                     npt.assert_almost_equal(ctrl.loads[:, 2], np.array([-2.69E-2,-2.69E-2,-5.37E-3])*profile_scale_factor)
#                 elif isinstance(ctrl, ct.PqofuPv):
#                     npt.assert_almost_equal(ctrl.p_kw, -2.69E-2*profile_scale_factor)
#                 elif isinstance(ctrl, ct.QofuLimPv):
#                     limiting_factor = ctrl.p_ac if ctrl.p_ac < ctrl.p_limit else ctrl.p_limit
#                     npt.assert_almost_equal(ctrl.p_kw, np.clip(-5.37E-3*profile_scale_factor, -abs(ctrl.sn_kva*limiting_factor), 0))
#                 else:
#                     npt.assert_almost_equal(ctrl.current_p, -5.37E-3*profile_scale_factor)
#
#
# def test_data_source_simulation():
#     """
#     This test doesn't assert anything yet, it rather tests for integration
#     """
#
#     # create DataSource from .csv file
#     fn = os.path.join(os.path.dirname(__file__), 'res\profile.csv')
#     my_data_source = cds.CsvDataSource(fn)
#
#     # create a net from .json file
#     fn = os.path.join(os.path.dirname(__file__), 'res\\net.json')
#     my_net = jfm.load_json_network(fn)
#
#     # create control handler
#     # start_step = where to start (for convenience, usually = 0)
#     # stop_step = steps to simulate, usually the whole profile: len(my_data_source.time_df.index)
#     my_ch = chi.ControlHandler(my_net, start_step=0, stop_step=2)
#
#     # create my controllers and add them
#     profile_table_path = os.path.join(os.path.dirname(__file__), 'res\profile_table.csv')
#     # ch = the ControlHandler we created before
#     # p_ac = the simultaneity factor we use for this study
#     # data_sources = the DataSource we created from the profiles
#     # profile_table_path = path to the profile table, that connects profiles with controller
#     dlh.build_ctrl_with_profile(ch=my_ch, p_ac=0.85, data_sources=my_data_source,
# profile_table_path=profile_table_path)
#
#     # runs the simulation, yielding one final network
#     # to save the result of time steps: implement at the marked location in ControlHandler
#     res = my_ch.handle_controller()
#
#     # On my 2x3.0 GHz, 2GB RAM:
#     # 10000 time steps take 320 sec -> ~20min per month, ~4-5h per year
#     # without saving results


if __name__ == '__main__':
    pytest.main(['-x', '-s', __file__])
    # pytest.main(['-x', __file__])
