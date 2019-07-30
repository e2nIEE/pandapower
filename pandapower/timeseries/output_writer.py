# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import functools
import os
from time import time

import numpy as np
import pandas as pd

try:
    import pplog
except ImportError:
    import logging as pplog

from pandapower.io_utils import mkdirs_if_not_existent
from pandas import DataFrame, isnull

logger = pplog.getLogger(__name__)

__author__ = 'fschaefer'


class OutputWriter:
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

        **output_path** (string, None) - Path to the file or folder we want to write the output to.
                                        Allowed file extensions: *.xls, *.xlsx

        **output_file_type** (string, ".p") - output filetype to use if output_path is not a file.
                                            Allowed file extensions: *.xls, *.xlsx, *.csv, *.pickle, *.json

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
        pass

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
