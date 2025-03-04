# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging
import datetime
import os
import tempfile
import time
from typing import Dict, List
import numpy as np
import pandas as pd


class UCTEParser:
    def __init__(self, path_ucte_file: str = None, config: Dict = None):
        """
        This class parses an UCTE file and loads its content to a dictionary of
        UCTE element type (str) -> UCTE elements (DataFrame)
        :param path_ucte_file: The path to the UCTE file. Optional, default: None. This parameter may be set later.
        :param config: The configuration dictionary. Optional, default: None. This parameter may be set later.
        """
        self.path_ucte_file: str = path_ucte_file
        self.config: Dict = config if isinstance(config, dict) else dict()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ucte_elements = ["##C", "##N", "##L", "##T", "##R", "##TT", "##E"]
        self.data: Dict[str, pd.DataFrame] = dict()
        self.date: datetime.datetime = datetime.datetime.utcnow()

    def parse_file(self, path_ucte_file: str = None) -> bool:
        """
        Parse a UCTE file.
        :param path_ucte_file: The path to the UCTE file. By default the internal field 'self.path_ucte_file' is used.
        :return: True if parsing was successful, False otherwise.
        """
        time_start = time.time()
        # check parameters and needed fields
        if path_ucte_file is not None:
            self.path_ucte_file = path_ucte_file
        if self.path_ucte_file is None:
            self.logger.error(
                "The variable 'path_ucte_file' is None, set it before parsing it!"
            )
            return False
        self.logger.info("Start parsing the file %s." % self.path_ucte_file)
        # parse the date, if its given by the configuration, parse it from there, otherwise from the file string
        if (
            "custom" in self.config.keys()
            and isinstance(self.config["custom"], dict)
            and "date" in self.config["custom"].keys()
        ):
            self._parse_date_str(self.config["custom"]["date"])
        else:
            self._parse_date_str(os.path.basename(self.path_ucte_file)[:13])
        raw_input_dict = dict()
        for ucte_element in self.ucte_elements:
            raw_input_dict[ucte_element] = []
        with open(self.path_ucte_file, "r") as f:
            # current_element contains the UCTE element type which is actually in parse progress, e.g. '##N'
            current_element = ""
            # iterate through the origin input file
            for row in f.readlines():
                row = row.strip()
                if row in self.ucte_elements:
                    # the start of a new UCTE element type in the origin file
                    current_element = row
                elif row.startswith("##C"):
                    # special for comments, because '##C' rows look like '##C 2007.05.01'
                    current_element = "##C"
                    raw_input_dict[current_element].append(row)
                elif row.startswith("##"):
                    self.logger.debug("Skipping row %s" % row)
                else:
                    raw_input_dict[current_element].append(row)
        self._create_df_from_raw(raw_input_dict)
        self.logger.info(
            "Finished parsing file %s in %ss."
            % (self.path_ucte_file, time.time() - time_start)
        )
        return True

    def _parse_date_str(self, date_str: str):
        try:
            self.date = datetime.datetime.strptime(date_str, "%Y%m%d_%H%M")
        except Exception as e:
            self.logger.info(
                f"The given {date_str=} couldn't be parsed as '%Y%m%d_%H%M'.")
            self.date = datetime.datetime.utcnow()

    def _create_df_from_raw(self, raw_input_dict):
        # create DataFrames from the raw_input_dict
        self.data = dict()
        for ucte_element, items in raw_input_dict.items():
            self.data[ucte_element] = pd.DataFrame(items)
        # make sure that at least some empty data exist
        if 0 not in self.data["##C"].columns:
            self.data["##C"][0] = ""
        # split the raw input for each UCTE element type
        self.data["##C"]["comments"] = self.data["##C"][0][:]
        self._split_nodes_from_raw()
        self._split_lines_from_raw()
        self._split_trafos_from_raw()
        self._split_trafos_regulation_from_raw()
        self._split_trafos_specified_parameters_from_raw()
        self._split_exchange_powers_from_raw()
        # drop the raw input columns
        for ucte_element, df in self.data.items():
            if 0 in df.columns:
                df = df.drop(columns=[0], axis=1)
        # set the data types
        dtypes = dict()
        i_t = pd.Int64Dtype()
        dtypes["##N"] = dict(
            {
                "status": i_t,
                "voltage": float,
                "p_load": float,
                "q_load": float,
                "p_gen": float,
                "q_gen": float,
                "min_p_gen": float,
                "max_p_gen": float,
                "min_q_gen": float,
                "max_q_gen": float,
                "static_primary_control": float,
                "p_primary_control": float,
                "three_ph_short_circuit_power": float,
                "x_r_ratio": float,
                "node_type": i_t,
            }
        )
        dtypes["##L"] = dict(
            {"status": i_t, "r": float, "x": float, "b": float, "i": float}
        )
        dtypes["##T"] = dict(
            {
                "status": i_t,
                "voltage1": float,
                "voltage2": float,
                "s": float,
                "r": float,
                "x": float,
                "b": float,
                "g": float,
                "i": float,
            }
        )
        dtypes["##R"] = dict(
            {
                "phase_reg_delta_u": float,
                "phase_reg_n": float,
                "phase_reg_n2": float,
                "phase_reg_u": float,
                "angle_reg_delta_u": float,
                "angle_reg_theta": float,
                "angle_reg_n": float,
                "angle_reg_n2": float,
                "angle_reg_p": float,
            }
        )
        dtypes["##TT"] = dict(
            {
                "tap_position": float,
                "r": float,
                "x": float,
                "delta_u": float,
                "alpha": float,
            }
        )
        dtypes["##E"] = dict({"p": float})
        for ucte_element, one_dtypes in dtypes.items():
            for field, field_type in one_dtypes.items():
                self.data[ucte_element].loc[
                    self.data[ucte_element][field] == "", field
                ] = np.nan
                # for integer: first convert them to float to prevent errors
                if field_type == i_t:
                    self.data[ucte_element][field] = self.data[ucte_element][
                        field
                    ].astype(float)
                self.data[ucte_element][field] = self.data[ucte_element][field].astype(
                    field_type
                )

        # remove '##' at the beginning of each key
        for one_key in list(self.data.keys()):
            self.data[one_key[2:]] = self.data[one_key]
            self.data.pop(one_key)

        del raw_input_dict

    def _split_nodes_from_raw(self):
        element_type = "##N"
        if element_type not in self.data.keys():
            self.logger.warning("No nodes in 'self.data' available! Didn't split them.")
            return
        df = self.data[element_type]
        # if 0 not in df.columns:
        #     df[0] = ""
        df["node"] = df[0].str[0:8].str.strip()
        df["node_name"] = df[0].str[9:21].str.strip()
        df["status"] = df[0].str[22:23].str.strip()
        df["node_type"] = df[0].str[24:25].str.strip()
        df["voltage"] = df[0].str[26:32].str.strip()
        df["p_load"] = df[0].str[33:40].str.strip()
        df["q_load"] = df[0].str[41:48].str.strip()
        df["p_gen"] = df[0].str[49:56].str.strip()
        df["q_gen"] = df[0].str[57:64].str.strip()
        df["min_p_gen"] = df[0].str[65:72].str.strip()
        df["max_p_gen"] = df[0].str[73:80].str.strip()
        df["min_q_gen"] = df[0].str[81:88].str.strip()
        df["max_q_gen"] = df[0].str[89:96].str.strip()
        df["static_primary_control"] = df[0].str[97:102].str.strip()
        df["p_primary_control"] = df[0].str[103:110].str.strip()
        df["three_ph_short_circuit_power"] = df[0].str[111:118].str.strip()
        df["x_r_ratio"] = df[0].str[119:126].str.strip()
        df["type"] = df[0].str[127:128].str.strip()

    def _split_connections_from_raw(self, element_type: str):
        # get the connections to nodes (node1 and node2) from the asset for e.g. ##L or ##T
        self.data[element_type]["node1"] = (
            self.data[element_type][0].str[0:8].str.strip()
        )
        self.data[element_type]["node2"] = (
            self.data[element_type][0].str[9:17].str.strip()
        )

    def _split_lines_from_raw(self):
        element_type = "##L"
        if element_type not in self.data.keys():
            self.logger.warning("No lines in 'self.data' available! Didn't split them.")
            return
        df = self.data[element_type]
        if 0 not in df.columns:
            df[0] = ""
        self._split_connections_from_raw(element_type)
        df["order_code"] = df[0].str[18:19].str.strip()
        df["status"] = df[0].str[20:21].str.strip()
        df["r"] = df[0].str[22:28].str.strip()
        df["x"] = df[0].str[29:35].str.strip()
        df["b"] = df[0].str[36:44].str.strip()
        df["i"] = df[0].str[45:51].str.strip()
        df["name"] = df[0].str[52:64].str.strip()

    def _split_trafos_from_raw(self):
        element_type = "##T"
        if element_type not in self.data.keys():
            self.logger.warning(
                "No transformers in 'self.data' available! Didn't split them."
            )
            return
        df = self.data[element_type]
        if 0 not in df.columns:
            df[0] = ""
        self._split_connections_from_raw(element_type)
        df["order_code"] = df[0].str[18:19].str.strip()
        df["status"] = df[0].str[20:21].str.strip()
        df["voltage1"] = df[0].str[22:27].str.strip()
        df["voltage2"] = df[0].str[28:33].str.strip()
        df["s"] = df[0].str[34:39].str.strip()
        df["r"] = df[0].str[40:46].str.strip()
        df["x"] = df[0].str[47:53].str.strip()
        df["b"] = df[0].str[54:62].str.strip()
        df["g"] = df[0].str[63:69].str.strip()
        df["i"] = df[0].str[70:76].str.strip()
        df["name"] = df[0].str[77:89].str.strip()

    def _split_trafos_regulation_from_raw(self):
        element_type = "##R"
        if element_type not in self.data.keys():
            self.logger.warning(
                "No tap changers in 'self.data' available! Didn't split them."
            )
            return
        df = self.data[element_type]
        if 0 not in df.columns:
            df[0] = ""
        self._split_connections_from_raw(element_type)
        df["order_code"] = df[0].str[18:19].str.strip()
        df["phase_reg_delta_u"] = df[0].str[20:25].str.strip()
        df["phase_reg_n"] = df[0].str[26:28].str.strip()
        df["phase_reg_n2"] = df[0].str[29:32].str.strip()
        df["phase_reg_u"] = df[0].str[33:38].str.strip()
        df["angle_reg_delta_u"] = df[0].str[39:44].str.strip()
        df["angle_reg_theta"] = df[0].str[45:50].str.strip()
        df["angle_reg_n"] = df[0].str[51:53].str.strip()
        df["angle_reg_n2"] = df[0].str[54:57].str.strip()
        df["angle_reg_p"] = df[0].str[58:63].str.strip()
        df["angle_reg_type"] = df[0].str[64:68].str.strip()

    def _split_trafos_specified_parameters_from_raw(self):
        element_type = "##TT"
        if element_type not in self.data.keys():
            self.logger.warning(
                "No specified transformer parameters in 'self.data' available! Didn't split them."
            )
            return
        df = self.data[element_type]
        if 0 not in df.columns:
            df[0] = ""
        self._split_connections_from_raw(element_type)
        df["order_code"] = df[0].str[18:19].str.strip()
        df["tap_position"] = df[0].str[22:25].str.strip()
        df["r"] = df[0].str[26:32].str.strip()
        df["x"] = df[0].str[33:39].str.strip()
        df["delta_u"] = df[0].str[40:45].str.strip()
        df["alpha"] = df[0].str[46:51].str.strip()

    def _split_exchange_powers_from_raw(self):
        element_type = "##E"
        if element_type not in self.data.keys():
            self.logger.warning(
                "No exchange powers in 'self.data' available! Didn't split them."
            )
            return
        df = self.data[element_type]
        if 0 not in df.columns:
            df[0] = ""
        df["country1"] = df[0].str[0:2].str.strip()
        df["country2"] = df[0].str[3:5].str.strip()
        df["p"] = df[0].str[6:13].str.strip()
        df["comment"] = df[0].str[14:26].str.strip()

    def get_fields_dict(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Get an overview about the UCTE element types and their fields.
        :return: A dictionary containing a dictionary with the UCTE element types as keys and for each UCTE element
        type a list of fields and a dictionary with the UCTE element types as keys and for each UCTE element type a
        list of dtypes.
        """
        self.logger.info(
            "Creating a dictionary containing the UCTE element types as keys and for each UCTE element "
            "type a list of fields and dtypes."
        )
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.close()
            ucte_temp = UCTEParser()
            ucte_temp.set_config(dict({"custom": {"date": "20200701_1010"}}))
            ucte_temp.parse_file(path_ucte_file=f.name)
            data = ucte_temp.get_data()
        if os.path.exists(f.name):
            os.remove(f.name)
        return_dict = dict()
        return_dict["element_types"] = dict()
        return_dict["dtypes"] = dict()
        for element_type, df in data.items():
            return_dict["element_types"][element_type] = list(df.columns)
            return_dict["dtypes"][element_type] = [str(x) for x in df.dtypes.values]
        return return_dict

    def set_path_ucte_file(self, path_ucte_file: str):
        self.path_ucte_file = path_ucte_file

    def get_path_ucte_file(self) -> str:
        return self.path_ucte_file

    def set_config(self, config: Dict):
        if isinstance(config, dict):
            self.config = config
        else:
            self.logger.warning(
                "The configuration is not a dictionary! Default configuration is set."
            )
            self.config = dict()

    def get_config(self) -> Dict:
        return self.config

    def get_data(self) -> Dict[str, pd.DataFrame]:
        return self.data

    def get_date(self) -> datetime.datetime:
        return self.date
