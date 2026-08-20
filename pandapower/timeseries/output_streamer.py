# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import os

import pandas as pd

from pandapower.auxiliary import pandapowerNet
from pandapower.io_utils import mkdirs_if_not_existent
from pandapower.timeseries.output_writer import OutputWriter

try:
    import pandaplan.core.pplog as pplog
except ImportError:
    import logging as pplog
logger = pplog.getLogger(__name__)


class OutputStreamer(OutputWriter):
    """
    The OutputStreamer class is used to cyclically store and format specific outputs from a time series calculation.

    For a detailed documentation, please refer to the parent OutputWriter class.

    Currently only xls and csv format is supported.

    Parameters:
        net: The pandapower format network
        time_steps (list): time_steps to calculate as a list (or range)
        output_path (string, None): Path to a folder where the output is written to.
        output_file_type (string, ".p"): output filetype to use. Allowed file extensions: [.xls, .xlsx, .csv, .csv.*,
            .p, .json]

            .. note::

                XLS has a maximum number of 256 rows.

                CSV files can be saved in a compressed format like `.csv.zip`.

        csv_separator (string, ";"): The separator used when writing to a csv file
        write_time (int, None): Time to save periodically to disk in minutes. Deactivated by default
        log_variables (list, None): list of tuples with (table, column) values to  be logged by output writer. Defaults
            are: res_bus.vm_pu and res_line.loading_percent. Additional variables can be added later on with
            ow.log_variable or removed with ow.remove_log_variable

    Example:
        >>> from pandapower.timeseries.output_streamer import OutputStreamer
        >>> import pandapower.networks as nw
        >>> net = nw.simple_four_bus_system()
        >>> ow = OutputStreamer(net) # create an OutputStreamer
        >>> ow.log_variable('res_bus', 'vm_pu') # add logging for bus voltage magnitudes
        >>> ow.log_variable('res_line', 'loading_percent') # add logging for line loadings in percent
        >>> # Getting the cost function slope for each time step:
        >>> def cost_logging(result, n_columns=2):
        >>>     return array([result[i][0][2] for i in range(len(result))])
        >>> ow.log_variable("pwl_cost", "points", eval_function=cost_logging)
    """

    def __init__(
        self,
        net,
        time_steps=None,
        save_interval: int = 0,
        output_path=None,
        output_file_type=".p",
        log_variables=None,
        csv_separator=";",
    ):
        super().__init__(
            net=net,
            time_steps=time_steps,
            output_path=output_path,
            output_file_type=output_file_type,
            log_variables=log_variables,
            csv_separator=csv_separator,
        )
        self.net = net
        self.output_path = output_path
        # defines the interval that is used to save the data to the file. If set to <=0, it behaves like OutputWriter.
        self.save_interval = save_interval
        # initialize time step to 0
        self.time_step = 0
        # initialize the last time step to 0
        self.last_time_step = 0

    def __update_csv_header(self, data: pd.DataFrame, table: str):
        """
        Updates the header of the element's dataframe.

        Parameters:
            data (DataFrame): Data to be updated.
            table (str): Name of the DataFrame table (example: "res_bus")
        """
        element_type = table.split(".")[0].replace("res_", "")
        if element_type in self.net:
            mapping = self.net[element_type].name.to_dict()
            data.rename(columns=mapping, inplace=True)

    def save_results(
        self, net: pandapowerNet, time_step: int, pf_converged: bool, ctrl_converged: bool, recycle_options = None
    ):
        """
        Saves the results of the current time step to a matrix
        and stores it to the disk in a after save_interval time steps.

        Parameters:
            net (pandapowerNet): The pandapower format network.
            time_step (int): Current time step.
            pf_converged (bool): Flag that checks if power flow is converged.
            ctrl_converged (bool): Flat that checks if controllers are converged.
            recycle_options (dict, optional): Recycle options used in OutputWriter. Defaults to None.
        """
        # call original save_results method from OutputWriter
        super().save_results(net, time_step, pf_converged, ctrl_converged, recycle_options=recycle_options)

        if self.save_interval > 0 and time_step < self.time_steps[-1] and (time_step + 1) % self.save_interval == 0:
            self.time_step = time_step + 1
            self.dump_to_file(net)
            self.last_time_step = self.time_step

    def _get_data_since_last_save(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the data DataFrame for the new data since the last dump.

        Parameters:
            data (DataFrame): Data to be filtered.

        Returns:
            DataFrame: Filtered data.
        """
        # Number of rows that have already been saved
        start = self.last_time_step

        if self.time_step == self.time_steps[-1]:
            return data.iloc[start:]

        end = start + self.save_interval

        return data.iloc[start:end]

    def _save_excel(self, file_path: str, data: pd.DataFrame, sheet_name: str = "Sheet1") -> None:
        """
        Saves the new simulation data to an excel file.
        Appends the new data if the file already exists.

        Parameters:
            file_path (str): Path to the excel file
            data (pd.DataFrame): Data to be saved or appended.
            sheet_name (str, optional): Name of the excel sheet. Defaults to "Sheet1".
        """
        try:
            if self.time_step > self.save_interval:
                data = self._get_data_since_last_save(data)
            if not os.path.exists(file_path):
                # create new file if it doesn't exist
                data.to_excel(file_path, sheet_name=sheet_name, index=False)
            else:
                # append to file if it exists
                with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                    # load the existing sheet to find its end
                    try:
                        # set start row to the end of the file
                        start_row = writer.book[sheet_name].max_row
                    except KeyError:
                        # set the start row to 0 in case the sheet doesn't exist
                        start_row = 0

                    # append data to the end without header
                    data.to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=start_row)
        except ValueError as e:
            if data.shape[1] > 255:
                raise ValueError(
                    "pandas.to_excel() is not capable to handle large data"
                    + "with more than 255 columns. Please use other "
                    + "file_extensions instead, e.g. 'json'."
                )
            else:
                raise ValueError(e)

    def _save_csv(self, file_path: str, data: pd.DataFrame, table: str, append: bool = False) -> None:
        """
        Saves the new simulation data to a csv file.

        Parameters:
            file_path (str): Path to the excel file
            data (pd.DataFrame): Data to be saved or appended.
            table (str): Name of the network element.
            append (bool): If True, append only new rows; otherwise rewrite the full output.
        """
        if append:
            data = self._get_data_since_last_save(data)
            header = self.last_time_step == 0
            if header:
                self.__update_csv_header(data, table)
            data.to_csv(file_path, sep=self.csv_separator, mode="a", header=header)
        else:
            self.__update_csv_header(data, table)
            data.to_csv(file_path, sep=self.csv_separator, mode="w", header=True)

    def _save_separate(self, append):

        for partial in self.output_list:
            if isinstance(partial, tuple):
                # if batch output is used
                table = partial[0]
                variable = partial[1]
            else:
                # if output_list contains functools.partial
                table = partial.args[0]
                variable = partial.args[1]
            if table != "Parameters":
                file_path = os.path.join(self.output_path, table)
                mkdirs_if_not_existent(file_path)
                if append:
                    file_name = str(variable) + "_" + str(self.cur_realtime) + self.output_file_type
                else:
                    file_name = str(variable) + self.output_file_type
                file_path = os.path.join(file_path, file_name)
                data = self.output[self._get_output_name(table, variable)]
                # remove rows with zeros only
                data = data.loc[~(data == 0).all(axis=1)]
                if self.output_file_type == ".json":
                    data.to_json(file_path)  # , lines=True, orient="records", mode="a")
                elif self.output_file_type == ".p":
                    data.to_pickle(file_path)
                elif self.output_file_type in [".xls", ".xlsx"]:
                    self._save_excel(file_path, data, table)
                elif "csv" in self.output_file_type.split("."):
                    self._save_csv(file_path, data, table, append=append)
        self.last_time_step = self.time_step
