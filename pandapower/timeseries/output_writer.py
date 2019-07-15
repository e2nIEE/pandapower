# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import functools
import os
from builtins import str, object
from time import time

import numpy as np
import pandas as pd
import pplog
from misc.utility_functions import mkdirs_if_not_existent
from pandas import DataFrame, isnull

logger = pplog.getLogger(__name__)

__author__ = 'fschaefer'


class OutputWriter(object):
    """
    This class supplies you with methods to store and format your output.
    The general idea is to have a python-dictionary *output* which provides
    a container for arbitrary information you would like to store. By default
    a pandas DataFrame is initialized for the key *Parameters*.

    For each value you want to store you may add a function to the *output_list*
    of the OutputWriter, which contains calculations and return a value to store.
    These values are then stored in the DataFrame mentioned above in a column
    named after the function you implemented.

    A lot of function are already implemented (for full list, see sourcecode).
    If there are any interesting output values missing, feel free to
    add them.

    INPUT:
        **net** - The pandapower format network
        **time_steps** (list) - time_steps to calculate as list

    OPTIONAL:

        **output_path** (string, None) - Path to the file or folder we want to write the output to. Allowed file extensions: *.xls, *.xlsx

        **output_file_type** (string, ".p") - output filetype to use if output_path is not a file. Allowed file extensions: *.xls, *.xlsx, *.csv, *.pickle, *.json

        **csv_seperator** (string, ";") - The seperator used when writing to a csv file

        **write_time** (int, None) - Time to save periodically to disk in minutes. Deactivated by default (=None)

        Note: XLS has a maximum number of 256 rows.
    """

    def __init__(self, net, time_steps=None, output_path=None, output_file_type=".p", write_time=None,
                 csv_seperator=";"):
        self.net = net
        self.csv_seperator = csv_seperator
        self.output_path = output_path
        self.output_file_type = output_file_type

        self.write_time = write_time
        if write_time is not None:
            self.write_time *= 60.0  # convert to seconds

        # init the matrix and the list of output functions
        self.output = dict()
        # internal results stored as numpy arrays in dict. Is created from output_list
        self.np_results = dict()
        # output list contains functools.partial with tables, variables, index...
        self.output_list = []
        # real time is tracked to save results to disk regularly
        self.cur_realtime = time()
        # total time steps to calculate
        self.time_steps = time_steps
        self.init_all()

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        s = "%s with output to %s" % (self.__class__.__name__, self.output_path)
        return s

    def init_all(self):
        if isinstance(self.time_steps, list) or isinstance(self.time_steps, range):
            self.init_timesteps(self.time_steps)
            self._init_np_results()
            self._init_output()
        else:
            logger.debug("Time steps not set at init ")

    def _init_output(self):
        self.output = dict()
        # init parameters
        self.output["Parameters"] = DataFrame(False, index=self.time_steps,
                                              columns=["time_step", "controller_unstable",
                                                       "powerflow_failed"])
        self.output["Parameters"].loc[:, "time_step"] = self.time_steps

    def _init_np_results(self):
        # inits numpy array (contains results)
        self.np_results = dict()
        for partial_func in self.output_list:
            self._init_np_array(partial_func)
            # self._init_np_array(var_name, index, eval_function)

    def _save_to_memmap(self, append):
        raise NotImplementedError("Sorry not implemented yet")

    def _save_seperate(self, append, file_extension):
        for partial in self.output_list:
            table = partial.args[0]
            variable = partial.args[1]
            if table is not "Parameters":
                file_path = os.path.join(self.output_path, table)
                mkdirs_if_not_existent(file_path)
                if append:
                    file_name = str(variable) + "_" + str(self.cur_realtime) + file_extension
                else:
                    file_name = str(variable) + file_extension
                file_path = os.path.join(file_path, file_name)
                data = self.output[self._get_output_name(table, variable)]
                # Todo: this can be done without this if else here, but I don't know how to call function by string. Please somebody help me
                if file_extension == ".json":
                    data.to_json(file_path)
                elif file_extension == ".p":
                    data.to_pickle(file_path)
                elif file_extension == ".xls" or file_extension == ".xlsx":
                    try:
                        data.to_excel(file_path)
                    except ValueError as e:
                        if data.shape[1] > 255:
                            raise ValueError("pandas.to_excel() is not capable to handle big data" +
                                             "with more than 255 columns. Please use other " +
                                             "file_extensions instead, e.g. 'json'.")
                        else:
                            raise ValueError(e)
                elif file_extension == ".csv":
                    data.to_csv(file_path, sep=self.csv_seperator)

    def dump_to_file(self, append=False):
        """
        Save the output matrix to a specific filetype (determined by basename)

           **append** (bool, False) - Option for appending instead of overwriting the file
        """
        file_extension = self.output_file_type
        save_single = False
        self._np_to_pd()
        if self.output_path is not None:
            try:
                if save_single and (file_extension == ".xls" or file_extension == ".xlsx"):
                    self._save_single_xls_sheet(append)
                elif file_extension in [".csv", ".xls", ".xlsx", ".json", ".p"]:
                    self._save_seperate(append, file_extension)
                elif file_extension == ".dat":
                    self._save_to_memmap(append)
                else:
                    raise UserWarning(
                        "Specify output file with .csv, .xls, .xlsx, .p, .json or .dat ending")

                if append:
                    self._init_output()

            except Exception:
                raise

    def dump(self):
        append = False if self.time_step == self.time_steps[-1] else True
        self.dump_to_file(append=append)
        self.cur_realtime = time()  # reset real time counter for next period

    def save_results(self, time_step, pf_converged, ctrl_converged):
        """
        Saves the results of the current time step to a matrix,
        using the output functions in the self.output_list
        """
        # remember the last time step
        self.time_step = time_step

        # add an entry to the output matrix if something failed
        if not pf_converged:
            self.save_nans_to_parameters()
            self.output["Parameters"].loc[time_step, "powerflow_failed"] = True
        elif not ctrl_converged:
            self.output["Parameters"].loc[time_step, "controller_unstable"] = True
        else:
            self.save_to_parameters()

        # if write time is exceeded or it is the last time step, data is written
        if self.write_time is not None:
            if time() - self.cur_realtime > self.write_time:
                self.dump()
        if self.time_step == self.time_steps[-1]:
            self.dump()

    def save_to_parameters(self):
        """
        Saves the results of the current time step to Parameters table,
        using the output functions in the self.output_list
        """
        for of in self.output_list:
            try:
                of()
            except:
                import traceback
                traceback.print_exc()
                logger.error("Error in output function! Stored NaN for '%s' in time-step %i"
                             % (of.__name__, self.time_step))
                self.save_nans_to_parameters()

    def save_nans_to_parameters(self):
        """
        Saves NaNs to for the given time step.
        """
        time_step_idx = self.time_step_lookup[self.time_step]
        for of in self.output_list:
            self.output["Parameters"].loc[time_step_idx, of.__name__] = np.NaN

    def remove_output_variable(self, table, variable):
        """
        Removes a single output from the output variable function stack
        """
        # ToDo: Implement this function

    def log_variable(self, table, variable, index=None, eval_function=None, eval_name=None):
        """
        Adds a variable to log during simulation.
            - table: table where the variable islocated as a string (i.e. "res_bus")
            - variable: variable that should be logged as string (i.e. "p_kw")
            - index: can be either one index or a list of indeces, or a numpy array of indices,
                or a pandas Index, or a pandas Series (e.g. net.load.bus) for which
                the variable will be logged. If no index is given, the variable
                will be logged for all elements in the table
            - eval_function: A function to be applied on the table / variable / index combination.
                For example: pd.min oder pd.max
            - eval_name: the name for an applied function.
                For example: "grid_losses"

        Note: Variable will be written to an extra sheet when writing to Excel
        or to an extra file when writing to csv.
        """

        if np.any(isnull(index)):
            # check how many elements there are in net
            index = self.net[table.split("res_")[-1]].index
        if not hasattr(index, '__iter__'):
            index = [index]
        if isinstance(index, (np.ndarray, pd.Index, pd.Series)):
            index = index.tolist()
        if eval_function is not None and eval_name is None:
            eval_name = "%s.%s.%s.%s" % (table, variable, str(index), eval_function.__name__)
        if eval_function is None and eval_name is not None:
            logger.info("'eval_name' is to give a name in case of evaluation functions. Since " +
                        "no function is given for eval_name '%s', " % eval_name +
                        "eval_name is neglected.")
            eval_name = None

        # var_name = self._get_hash((table, variable, index, eval_function))
        var_name = self._get_output_name(table, variable)
        idx = self._get_same_log_variable_partial_func_idx(table, variable, eval_function,
                                                           eval_name)
        if idx is not None:
            self._append_existing_log_variable_partial_func(idx, index)
        else:
            self._append_output_list(table, variable, index, eval_function, eval_name, var_name)

    def _get_same_log_variable_partial_func_idx(self, table, variable, eval_function, eval_name):
        """ Returns the position index in self.output_list of partial_func which has the same table
        and variable and no evaluation function. """
        if eval_function is None and eval_name is None:
            for i, partial_func in enumerate(self.output_list):
                partial_args = partial_func.args
                match = partial_args[0] == table
                match &= partial_args[1] == variable
                if match:
                    return i

    def _append_existing_log_variable_partial_func(self, idx, index):
        """ Appends the index of existing, same partial_func in output_list. """
        for i in index:
            if i not in self.output_list[idx].args[2]:
                self.output_list[idx].args[2].append(i)

    def _append_output_list(self, table, variable, index, eval_function, eval_name, var_name):
        """ Appends the output_list by an additional partial_func. """
        partial_func = functools.partial(self._log, table, variable, index, eval_function,
                                         eval_name)
        partial_func.__name__ = var_name
        self.output_list.append(partial_func)
        if self.time_steps is not None:
            self._init_np_array(partial_func)

    def add_output(self, output):
        """
        Adds a single output to the list
        """
        logger.warning(
            "Your outputwriter contains deprecated functions, check if the output is right!")
        self.output_list.append(output)

    def _log(self, table, variable, index, eval_function=None, eval_name=None):
        try:
            # ToDo: Create a mask for the numpy array in the beginning and use this one for getting the values. Faster
            if self.net[table].index.equals(index):
                # if index equals all values -> get numpy array directly
                result = self.net[table][variable].values
            else:
                # get by loc (slow)
                result = self.net[table].loc[index, variable].values

            if eval_function is not None:
                result = eval_function(result)

            # save results to numpy array
            time_step_idx = self.time_step_lookup[self.time_step]
            hash_name = self._get_np_name((table, variable, index, eval_function, eval_name))
            self.np_results[hash_name][time_step_idx, :] = result

        except Exception as e:
            logger.error("Error at index %s for %s[%s]: %s" % (index, table, variable, e))

    def _np_to_pd(self):
        # convert numpy arrays (faster so save results) into pd Dataframes (user friendly)
        # intended use: At the end of time series simulation write results to pandas

        for partial_func in self.output_list:
            (table, variable, index, eval_func, eval_name) = partial_func.args
            # res_name = self._get_hash(table, variable)
            res_name = self._get_output_name(table, variable)
            np_name = self._get_np_name(partial_func.args)
            columns = index
            if eval_name is not None:
                columns = [eval_name]
            res_df = DataFrame(self.np_results[np_name], index=self.time_steps, columns=columns)
            if res_name in self.output and eval_name is not None:
                try:
                    self.output[res_name] = pd.concat([self.output[res_name], res_df], axis=1,
                                                      sort=False)
                except TypeError:
                    # pandas legacy < 0.21
                    self.output[res_name] = pd.concat([self.output[res_name], res_df], axis=1)
            else:
                # new dataframe
                self.output[res_name] = res_df

    def _get_output_name(self, table, variable):
        return "%s.%s" % (table, variable)

    def _get_np_name(self, partial_args):
        eval_name = partial_args[4]
        if eval_name is not None:
            return eval_name
        else:
            table = partial_args[0]
            variable = partial_args[1]
            return "%s.%s" % (table, variable)

    def _save_single_xls_sheet(self, append):
        # ToDo: implement save to a single sheet
        raise NotImplementedError("Sorry not implemented yet")

    def init_timesteps(self, time_steps):
        self.time_steps = time_steps
        self.time_step = time_steps[0]
        self.time_step_lookup = {t: idx for idx, t in enumerate(time_steps)}

    def _init_np_array(self, partial_func):
        (table, variable, index, eval_function, eval_name) = partial_func.args
        hash_name = self._get_np_name(partial_func.args)
        n_columns = len(index)
        if eval_function is not None:
            n_columns = 1
        self.np_results[hash_name] = np.zeros((len(self.time_steps), n_columns))

# TODO
# ALL OLD FUNCS:
# =======
#     def max_u_mv(self):
#         mask = (self.net.bus.vn_kv > 1.0) & (self.net.bus.vn_kv < 70.0)
#         mumv = np.amax(self.net.res_bus.loc[mask, "vm_pu"])
#         self.save_under_funcname(mumv)
#
#     def min_u_mv(self):
#         mask = (self.net.bus.vn_kv > 1.0) & (self.net.bus.vn_kv < 70.0)
#         mumv = np.amin(self.net.res_bus.loc[mask, "vm_pu"])
#         self.save_under_funcname(mumv)
#
#     def p_exchange(self):
#         pe = np.sum(self.net.res_ext_grid["p_kw"])
#         self.save_under_funcname(pe)
#
#     def q_exchange(self):
#         qe = np.sum(self.net.res_ext_grid["q_kvar"])
#         self.save_under_funcname(qe)
#
#     def max_u_node(self):
#         mun = np.argmax(self.net.res_bus["vm_pu"], axis=0)
#         self.save_under_funcname(mun)
#
#     def min_u_node(self):
#         mun = np.argmin(self.net.res_bus["vm_pu"], axis=0)
#         self.save_under_funcname(mun)
#
#     def max_u(self):
#         mu = np.amax(self.net.res_bus["vm_pu"], axis=0)
#         self.save_under_funcname(mu)
#
#     def min_u(self):
#         mu = np.amin(self.net.res_bus["vm_pu"], axis=0)
#         self.save_under_funcname(mu)
#
#     def tap_trafo2W(self):
#         tap_trafo2w = self.net.trafo.tap_pos.at[0]
#         self.save_under_funcname(tap_trafo2w)
#
#     def p_trafo3w_mv(self):
#         p_trafo3w = -1 * (self.net.res_trafo3w.p_lv_kw.at[0] +
#                           self.net.res_trafo3w.p_mv_kw.at[0])
#         self.save_under_funcname(p_trafo3w)
#
#     def q_trafo3w_mv(self):
#         q_trafo3w = -1 * (self.net.res_trafo3w.q_lv_kvar.at[0] +
#                           self.net.res_trafo3w.q_mv_kvar.at[0])
#         self.save_under_funcname(q_trafo3w)
#
#     def tap_trafo3w(self):
#         tap_trafo3w = self.net.trafo3w.tap_pos.at[0]
#         self.save_under_funcname(tap_trafo3w)
#
#     def u_mss_trafo3w(self):
#         u_mss_trafo3w = self.net.res_bus.vm_pu.at[self.net.trafo3w.mv_bus.at[0]]
#         self.save_under_funcname(u_mss_trafo3w)
#
#     def grid_losses(self):
#         gl = -np.sum(self.net.res_bus["p_kw"])
#         self.save_under_funcname(gl)
#
#     def tap_pos_usw(self):
#         ti = self.net.trafo.query("100<vn_hv_kv<120").index
#         self.save_under_funcname(self.net.trafo.loc[ti, "tap_pos"].values)
#
#     def tap_voltage_usw(self):
#         ti = self.net.trafo.query("100<vn_hv_kv<120").index
#         if self.net.trafo.loc[ti, "tap_side"].values == 'hv':
#             self.save_under_funcname(self.net.res_trafo.loc[ti, "u_hv_pu"].values)
#         elif self.net.trafo.loc[ti, "tap_side"].values == 'lv':
#             self.save_under_funcname(self.net.res_trafo.loc[ti, "u_lv_pu"].values)
#
#     # TODO: rework
#     # For a year simulation with 1000 Controllers one for loop like the following
#     # increases the calculation time by 5 minutes. It scales approx linear with the number
#     # of Controllers
#
#     def total_load_p(self):
#         self.save_under_funcname(sum(self.net.load.p_kw))
#
#     def total_sgen_p(self):
#         self.save_under_funcname(sum(self.net.sgen.p_kw))
#
#     def total_load_q(self):
#         self.save_under_funcname(sum(self.net.load.q_kvar))
#
#     def total_sgen_q(self):
#         self.save_under_funcname(sum(self.net.sgen.q_kvar))
#
#     def total_p_provision(self):
#         plist = [ctrl.p_kw for ctrl in
#                  ch.get_ctrl_by_type(self.net, control.PvControl) if
#                  self.net.controller.in_service.at[ctrl.index]]
#         self.save_under_funcname(sum(plist))
#
#     def total_q_provision(self):
#         qlist = [ctrl.q_kvar for ctrl in
#                  ch.get_ctrl_by_type(self.net, control.PvControl) if
#                  self.net.controller.in_service.at[ctrl.index]]
#         self.save_under_funcname(sum(qlist))
#
#     def total_p_curtailment(self):
#         pcurt = [ctrl.p_curtailment for ctrl in
#                  ch.get_ctrl_by_type(self.net, control.PvControl) if
#                  self.net.controller.in_service.at[ctrl.index]]
#         self.save_under_funcname(sum(pcurt))
#
#     def maxp_curtailment(self):
#         pcurt = [ctrl.p_curtailment for ctrl in
#                  ch.get_ctrl_by_type(self.net, control.MaxP_Central) if
#                  self.net.controller.in_service.at[ctrl.index]]
#         self.save_under_funcname(sum(sum(pcurt)))
#
#     def maxp_provision(self):
#         pprov = self.net.sgen.p_kw
#         self.save_under_funcname(sum(pprov))
#
#     def maxp_available(self):
#         pav = [ctrl.max_p_kw for ctrl in
#                ch.get_ctrl_by_type(self.net, control.MaxP_Central) +
#                ch.get_ctrl_by_type(self.net, control.PvController) if
#                self.net.controller.in_service.at[ctrl.index]]
#         self.save_under_funcname(sum(pav))
#
#     def maxp_curt_percent(self):
#         for ctrl in ch.get_ctrl_by_type(self.net, control.MaxP_Central):
#             if self.net.controller.in_service.at[ctrl.index] and sum(ctrl.p_available):
#                 pcurtp = [-(-ctrl.net.sgen.p_kw / ctrl.p_available * 100)]
#                 pcurtp = max(max(pcurtp))
#             else:
#                 pcurtp = -100
#
#                 #        if np.isnan(pcurtp):
#                 #            pcurtp=-100
#         self.save_under_funcname(100 + pcurtp)
#
#     def maxp_v_unctrld(self):
#         max_v_unctrld = [ctrl.max_v_unctrld for ctrl in
#                          ch.get_ctrl_by_type(self.net, control.MaxP_Central) if
#                          self.net.controller.in_service.at[ctrl.index]]
#         self.save_under_funcname(max_v_unctrld[0])
#
#     def all_sgen_p_kw(self):
#         logger.error("Output functuion all_sgen_p_kw deprecated. Use log_variable instead")
#         p_kw = self.net.res_sgen["p_kw"]
#         for index, pval in enumerate(p_kw):
#             self.save_under_name(pval, "P_PV_" + str(index))
#
#     def all_sgen_min_p_kw(self):
#         logger.error("Output function all_sgen_min_p_kw deprecated. Use log_variable instead")
#         min_p_kw = self.net.sgen["min_p_kw"]
#         for index, pval in enumerate(min_p_kw):
#             self.save_under_name(pval, "Pmin_PV_" + str(index))
#
#     def all_sgen_q_kvar(self):
#         logger.error("Output functuion all_sgen_q_kvar deprecated. Use log_variable instead")
#         q_kvar = self.net.res_sgen["q_kvar"]
#         for index, qval in enumerate(q_kvar):
#             self.save_under_name(qval, "Q_PV_" + str(index))
#
#     def all_load_p_kw(self):
#         logger.error("Output functuion all_load_p_kw deprecated. Use log_variable instead")
#         p_kw = self.net.res_load["p_kw"]
#         for index, pval in enumerate(p_kw):
#             self.save_under_name(pval, "P_load_" + str(index))
#
#     def all_load_q_kvar(self):
#         logger.error("Output functuion all_load_q_kvar deprecated. Use log_variable instead")
#         q_kvar = self.net.res_load["q_kvar"]
#         for index, qval in enumerate(q_kvar):
#             self.save_under_name(qval, "Q_load_" + str(index))
#
#     def all_controllers_cs_step(self):  # Only relevent for dis_MAS
#         cs_steplist = [ctrl.controlsteplist for ctrl in
#                        ch.get_ctrl_by_type(self.net, control.controller.dis_MAS.MAS_ctrl) if
#                        self.net.controller.in_service.at[ctrl.index]]
#         for index_ctrl, cstep in enumerate(cs_steplist):
#             for index_cs, value in enumerate(cstep):
#                 self.save_under_name(value, "C_STEP" + str(index_cs) + "_CTRL_" + str(index_ctrl))
#
#     def cs_plosses(self):  # Only relevent for dis_MAS
#         cs_p = [ctrl.cs_ploss for ctrl in
#                 ch.get_ctrl_by_type(self.net, control.controller.dis_MAS.MAS_ctrl) if
#                 self.net.controller.in_service.at[ctrl.index]]
#         for index_ctrl, pstep in enumerate(cs_p):
#             for index_cs, value in enumerate(pstep):
#                 self.save_under_name(value, "CS_PLOSS" + str(index_cs) + "_CTRL_" + str(index_ctrl))
#
#     def ts_plosses(self):  # Only relevent for dis_MAS
#         ts_p = [ctrl.ts_ploss for ctrl in
#                 ch.get_ctrl_by_type(self.net, control.controller.dis_MAS.MAS_ctrl) if
#                 self.net.controller.in_service.at[ctrl.index]]
#         for index_ctrl, pstep in enumerate(ts_p):
#             self.save_under_name(pstep, "TS_PLOSS" + "_CTRL_" + str(index_ctrl))
#
#     def cs_line_loading(self):  # Only relevent for dis_MAS
#         cs_lilo = [ctrl.cs_line_loading for ctrl in
#                    ch.get_ctrl_by_type(self.net, control.controller.dis_MAS.MAS_ctrl) if
#                    self.net.controller.in_service.at[ctrl.index]][0]
#         for index_ctrl, lilostep in enumerate(cs_lilo):
#             for index_cs, value in enumerate(lilostep):
#                 self.save_under_name(value,
#                                      "CS_LINE_LOADING" + str(index_cs) + "_CTRL_" + str(index_ctrl))
#
#     def ts_line_loading(self):  # Only relevent for dis_MAS
#         ts_lilo = [ctrl.ts_line_loading for ctrl in
#                    ch.get_ctrl_by_type(self.net, control.controller.dis_MAS.MAS_ctrl) if
#                    self.net.controller.in_service.at[ctrl.index]][0]
#         for index_ctrl, lilostep in enumerate(ts_lilo):
#             self.save_under_name(lilostep, "TS_LINE_LOADING" + "_CTRL_" + str(index_ctrl))
#
#     def all_sgen_cs_q(self):  # Only relevent for dis_MAS
#         cs_q = [ctrl.cs_sgen_Q for ctrl in
#                 ch.get_ctrl_by_type(self.net, control.controller.dis_MAS.MAS_ctrl) if
#                 self.net.controller.in_service.at[ctrl.index] and ctrl.gtype == "Q"]
#         for index_ctrl, qstep in enumerate(cs_q):
#             for index_cs, value in enumerate(qstep):
#                 self.save_under_name(value,
#                                      "CS__SGEN_Q" + str(index_cs) + "_CTRL_" + str(index_ctrl))
#
#     def all_sgen_cs_dfdQ(self):  # Only relevent for dis_MAS
#         cs_dfdQ = [ctrl.cs_sgen_dfdQ for ctrl in
#                    ch.get_ctrl_by_type(self.net, control.controller.dis_MAS.MAS_ctrl) if
#                    self.net.controller.in_service.at[ctrl.index] and ctrl.gtype == "Q"]
#         for index_ctrl, dfdQstep in enumerate(cs_dfdQ):
#             for index_cs, value in enumerate(dfdQstep):
#                 self.save_under_name(value, "CS_dfdQ" + str(index_cs) + "_CTRL_" + str(index_ctrl))
#
#     def all_sgen_cs_sgen_u(self):  # Only relevent for dis_MAS
#         cs_u = [ctrl.cs_sgen_u for ctrl in
#                 ch.get_ctrl_by_type(self.net, control.controller.dis_MAS.MAS_ctrl) if
#                 self.net.controller.in_service.at[ctrl.index] and ctrl.gtype == "Q"]
#         for index_ctrl, ustep in enumerate(cs_u):
#             for index_cs, value in enumerate(ustep):
#                 self.save_under_name(value,
#                                      "CS__SGEN_U" + str(index_cs) + "_CTRL_" + str(index_ctrl))
#
#     def all_sgen_ts_q(self):  # Only relevent for dis_MAS
#         ts_q = [ctrl.ts_sgen_Q for ctrl in
#                 ch.get_ctrl_by_type(self.net, control.controller.dis_MAS.MAS_ctrl) if
#                 self.net.controller.in_service.at[ctrl.index] and ctrl.gtype == "Q"]
#         for index_ctrl, tstep in enumerate(ts_q):
#             self.save_under_name(tstep, "TS__SGEN_Q" + "_CTRL_" + str(index_ctrl))
#
#     def all_sgen_ts_dfdQ(self):  # Only relevent for dis_MAS
#         ts_dfdQ = [ctrl.ts_sgen_dfdQ for ctrl in
#                    ch.get_ctrl_by_type(self.net, control.controller.dis_MAS.MAS_ctrl) if
#                    self.net.controller.in_service.at[ctrl.index] and ctrl.gtype == "Q"]
#         for index_ctrl, tstep in enumerate(ts_dfdQ):
#             self.save_under_name(tstep, "TS__SGEN_dfdQ" + "_CTRL_" + str(index_ctrl))
#
#     def all_sgen_ts_sgen_u(self):  # Only relevent for dis_MAS
#         ts_u = [ctrl.ts_sgen_u for ctrl in
#                 ch.get_ctrl_by_type(self.net, control.controller.dis_MAS.MAS_ctrl) if
#                 self.net.controller.in_service.at[ctrl.index] and ctrl.gtype == "Q"]
#         for index_ctrl, tstep in enumerate(ts_u):
#             self.save_under_name(tstep, "TS__SGEN_U" + "_CTRL_" + str(index_ctrl))
#
#     def all_gen_cs_Q(self):  # Only relevent for dis_MAS
#         cs_q = [ctrl.cs_gen_Q for ctrl in
#                 ch.get_ctrl_by_type(self.net, control.controller.dis_MAS.MAS_ctrl) if
#                 self.net.controller.in_service.at[ctrl.index] and ctrl.gtype == "U"]
#         for index_ctrl, qstep in enumerate(cs_q):
#             for index_cs, value in enumerate(qstep):
#                 self.save_under_name(value,
#                                      "CS__GEN_Q" + str(index_cs) + "_CTRL_" + str(index_ctrl))
#
#     def all_gen_cs_dfdV(self):  # Only relevent for dis_MAS
#         cs_dfdV = [ctrl.cs_gen_dfdV for ctrl in
#                    ch.get_ctrl_by_type(self.net, control.controller.dis_MAS.MAS_ctrl) if
#                    self.net.controller.in_service.at[ctrl.index] and ctrl.gtype == "U"]
#         for index_ctrl, dfdVstep in enumerate(cs_dfdV):
#             for index_cs, value in enumerate(dfdVstep):
#                 self.save_under_name(value,
#                                      "CS__GEN_dfdV" + str(index_cs) + "_CTRL_" + str(index_ctrl))
#
#     def all_gen_cs_u(self):  # Only relevent for dis_MAS
#         cs_u = [ctrl.cs_gen_u for ctrl in
#                 ch.get_ctrl_by_type(self.net, control.controller.dis_MAS.MAS_ctrl) if
#                 self.net.controller.in_service.at[ctrl.index] and ctrl.gtype == "U"]
#         for index_ctrl, ustep in enumerate(cs_u):
#             for index_cs, value in enumerate(ustep):
#                 self.save_under_name(value,
#                                      "CS__GEN_U" + str(index_cs) + "_CTRL_" + str(index_ctrl))
#
#     def all_gen_ts_Q(self):  # Only relevent for dis_MAS
#         ts_q = [ctrl.ts_gen_Q for ctrl in
#                 ch.get_ctrl_by_type(self.net, control.controller.dis_MAS.MAS_ctrl) if
#                 self.net.controller.in_service.at[ctrl.index] and ctrl.gtype == "U"]
#         for index_ctrl, tstep in enumerate(ts_q):
#             self.save_under_name(tstep, "TS__GEN_Q" + "_CTRL_" + str(index_ctrl))
#
#     def all_gen_ts_dfdV(self):  # Only relevent for dis_MAS
#         ts_dfdV = [ctrl.ts_gen_dfdV for ctrl in
#                    ch.get_ctrl_by_type(self.net, control.controller.dis_MAS.MAS_ctrl) if
#                    self.net.controller.in_service.at[ctrl.index] and ctrl.gtype == "U"]
#         for index_ctrl, tstep in enumerate(ts_dfdV):
#             self.save_under_name(tstep, "TS__GEN_dfdV" + "_CTRL_" + str(index_ctrl))
#
#     def all_gen_ts_u(self):  # Only relevent for dis_MAS
#         ts_u = [ctrl.ts_gen_u for ctrl in
#                 ch.get_ctrl_by_type(self.net, control.controller.dis_MAS.MAS_ctrl) if
#                 self.net.controller.in_service.at[ctrl.index] and ctrl.gtype == "U"]
#         for index_ctrl, tstep in enumerate(ts_u):
#             self.save_under_name(tstep, "TS__GEN_U" + "_CTRL_" + str(index_ctrl))
#
#     def optimization_state(self):
#         for ctrl in ch.get_ctrl_by_type(self.net, control.Dyn_Curt):
#             self.save_under_funcname(ctrl.optimized)
#
#     def e_curt(self):
#         elist = [ctrl.e_curt for ctrl in
#                  ch.get_ctrl_by_type(self.net, control.StatCurtPv) if
#                  self.net.controller.in_service.at[ctrl.index]]
#         self.save_under_funcname(sum(elist))
#
#     def e_available(self):
#         elist = [ctrl.e_available for ctrl in
#                  ch.get_ctrl_by_type(self.net, control.StatCurtPv) if
#                  self.net.controller.in_service.at[ctrl.index]]
#         self.save_under_funcname(sum(elist))
#
#     def e_curt_DynCurt(self):
#         e = [ctrl.e_curt for ctrl in
#              ch.get_ctrl_by_type(self.net, control.Dyn_Curt) if
#              self.net.controller.in_service.at[ctrl.index]]
#         self.save_under_funcname(sum(e[0]))
#
#     def e_available_DynCurt(self):
#         e = [ctrl.e_available for ctrl in
#              ch.get_ctrl_by_type(self.net, control.Dyn_Curt) if
#              self.net.controller.in_service.at[ctrl.index]]
#         self.save_under_funcname(sum(e[0]))
#
#     def const_load_p(self):
#         tot = 0
#         for ctrl in ch.get_ctrl_by_type(self.net, control.ConstLoad):
#             if self.net.controller.in_service.at[ctrl.index]:
#                 tot += np.sum(self.net.load.loc[ctrl.profiles.index, "p_kw"] *
#                               self.net.load.loc[ctrl.profiles.index, "scaling"])
#         self.save_under_funcname(tot)
#
#     def pv_no_ctrl_p(self):
#         plist = [ctrl.p_kw for ctrl in
#                  ch.get_ctrl_by_type(self.net, control.NoCtrlPv) if
#                  self.net.controller.in_service.at[ctrl.index]]
#         self.save_under_funcname(sum(plist))
#
#     def pv_qofu_p(self):
#         plist = [ctrl.p_kw for ctrl in ch.get_ctrl_by_type(self.net, control.QofuPv) if
#                  self.net.controller.in_service.at[ctrl.index]]
#         self.save_under_funcname(sum(plist))
#
#     def pv_qqqu_p(self):
#         plist = [ctrl.p_kw for ctrl in ch.get_ctrl_by_type(self.net, control.QQQUPV) if
#                  self.net.controller.in_service.at[ctrl.index]]
#         self.save_under_funcname(sum(plist))
#
#     def pv_qqqu_q(self):
#         qlist = [ctrl.q_kvar for ctrl in
#                  ch.get_ctrl_by_type(self.net, control.QQQUPV) if
#                  self.net.controller.in_service.at[ctrl.index]]
#         self.save_under_funcname(sum(qlist))
#
#     def max_line_loading(self):
#         mll = np.amax(self.net.res_line["loading_percent"])
#         self.save_under_funcname(mll)
#
#     def max_loaded_line(self):
#         mll = np.argmax(self.net.res_line["loading_percent"])
#         self.save_under_funcname(mll)
#
#     def max_trafo_loading(self):
#         mtl = np.amax(self.net.res_trafo["loading_percent"])
#         self.save_under_funcname(mtl)
#
#     def max_loaded_trafo(self):
#         mlt = np.argmax(self.net.res_trafo["loading_percent"])
#         self.save_under_funcname(mlt)
#
#     def save_under_funcname(self, val):
#         self.output["Parameters"].loc[self.time_step, inspect.stack()[1][3]] = val
#
#     def save_under_name(self, val, name):
#         self.output["Parameters"].loc[self.time_step, name] = val
# >>>>>>> develop
