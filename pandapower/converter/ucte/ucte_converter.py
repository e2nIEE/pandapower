# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging
import math
import time
from typing import Dict, Union

import numpy as np
import pandapower as pp
import pandapower.auxiliary
import pandas as pd


class UCTE2pandapower:
    def __init__(self):
        """
        Convert UCTE data to pandapower.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.u_d = dict()
        self.net = self._create_empty_network()
        self.net.bus["node_name"] = ""

    @staticmethod
    def _create_empty_network():
        net: pandapower.auxiliary.pandapowerNet = pp.create_empty_network()
        new_columns = {
            "trafo": {
                "tap2_min": int,
                "tap2_max": int,
                "tap2_neutral": int,
                "tap2_pos": int,
                "tap2_step_percent": float,
                "tap2_step_degree": float,
                "tap2_side": str,
                "tap2_changer_type": str,
                "amica_name": str,
            },
            "line": {"amica_name": str},
        }
        for pp_element in new_columns.keys():
            for col, dtype in new_columns[pp_element].items():
                net[pp_element][col] = pd.Series(dtype=dtype)
        return net

    def convert(self, ucte_dict: Dict) -> pandapower.auxiliary.pandapowerNet:
        self.logger.info("Converting UCTE data to a pandapower network.")
        time_start = time.time()
        # create a temporary copy from the origin input data
        self.u_d = dict()
        for ucte_element, items in ucte_dict.items():
            self.u_d[ucte_element] = items.copy()
        # first reset the index to get indices for pandapower
        for ucte_element in self.u_d.keys():
            if ucte_element == "R":
                continue
            self.u_d[ucte_element] = self.u_d[ucte_element].reset_index(level=0)
            self.u_d[ucte_element] = self.u_d[ucte_element].rename(
                columns={"index": "id"}
            )
        # now replace the node1 and node2 columns with the node index at lines, transformers, ...
        merge_nodes = self.u_d["N"][["id", "node"]]
        self.u_d["L"] = pd.merge(
            self.u_d["L"],
            merge_nodes.rename(columns={"node": "node1", "id": "from_bus"}),
            how="left",
            on="node1",
        )
        self.u_d["L"] = pd.merge(
            self.u_d["L"],
            merge_nodes.rename(columns={"node": "node2", "id": "to_bus"}),
            how="left",
            on="node2",
        )
        self.u_d["L"] = self.u_d["L"].drop(columns=["node1", "node2"])
        for one_asset in ["T", "R", "TT"]:
            self.u_d[one_asset] = pd.merge(
                self.u_d[one_asset],
                merge_nodes.rename(columns={"node": "node1", "id": "hv_bus"}),
                how="left",
                on="node1",
            )
            self.u_d[one_asset] = pd.merge(
                self.u_d[one_asset],
                merge_nodes.rename(columns={"node": "node2", "id": "lv_bus"}),
                how="left",
                on="node2",
            )
            self.u_d[one_asset] = self.u_d[one_asset].drop(columns=["node1", "node2"])

        # prepare the element tables
        self._convert_nodes()
        self._convert_loads()
        self._convert_gens()
        self._convert_lines()
        self._convert_impedances()
        self._convert_switches()
        self._convert_trafos()

        # copy data to the element tables of self.net
        self.net = self.set_pp_col_types(self.net)

        # currently, net.bus.name contains the UCTE node name ("Node"), while
        # net.bus.node_name contains the original node name "Node Name". This is changed now:
        cols_in_order = list(pd.Series(self.net.bus.columns).replace("node_name", "ucte_name"))
        self.net.bus = self.net.bus.rename(columns={"name": "ucte_name", "node_name": "name"})[
            cols_in_order]

        self.logger.info(
            "Finished converting the input data to pandapower in %ss."
            % (time.time() - time_start)
        )
        return self.net

    def _copy_to_pp(self, pp_type: str, input_df: pd.DataFrame):
        self.logger.debug(
            "Copy %s datasets to pandapower network with type %s"
            % (input_df.index.size, pp_type)
        )
        if pp_type not in self.net.keys():
            self.logger.warning(
                "Missing pandapower type %s in the pandapower network!" % pp_type
            )
            return
        self.net[pp_type] = pd.concat(
            [
                self.net[pp_type],
                input_df[
                    list(set(self.net[pp_type].columns).intersection(input_df.columns))
                ],
            ],
            ignore_index=True,
            sort=False,
        )

    def _convert_nodes(self):
        self.logger.info("Converting the nodes.")
        nodes = self.u_d[
            "N"
        ]  # Note: Do not use a copy, the columns 'volt_str' and 'node_name' are needed later
        nodes["volt_str"] = nodes["node"].str[6:7]
        volt_map = {
            "0": 750,
            "1": 380,
            "2": 220,
            "3": 150,
            "4": 120,
            "5": 110,
            "6": 70,
            "7": 27,
            "8": 330,
            "9": 500,
            "A": 26,
            "B": 25,
            "C": 24,
            "D": 23,
            "E": 22,
            "F": 21,
            "G": 20,
            "H": 19,
            "I": 18,
            "J": 17,
            "K": 15.7,
            "L": 15,
            "M": 13.7,
            "N": 13,
            "O": 12,
            "P": 11,
            "Q": 9.8,
            "R": 9,
            "S": 8,
            "T": 7,
            "U": 6,
            "V": 5,
            "W": 4,
            "X": 3,
            "Y": 2,
            "Z": 1,
        }
        nodes["vn_kv"] = nodes["volt_str"].map(volt_map)
        nodes["node2"] = nodes["node"].str[:6]
        nodes["grid_area_id"] = nodes["node"].str[:2]
        # drop all voltages at non pu nodes
        nodes.loc[
            (nodes["node_type"] != 2) & (nodes["node_type"] != 3), "voltage"
        ] = np.nan
        nodes = nodes.rename(columns={"node": "name"})
        nodes["in_service"] = True
        self._copy_to_pp("bus", nodes)
        self.logger.info("Finished converting the nodes.")

    def _convert_loads(self):
        self.logger.info("Converting the loads.")
        # select the loads from the nodes and drop not given values
        loads = self.u_d["N"].dropna(subset=["p_load", "q_load"])
        # select all with p != 0 or q != 0
        loads = loads.loc[(loads["p_load"] != 0) | (loads["q_load"] != 0)]
        if not len(loads):
            self.logger.info("Finished converting the loads (no loads existing).")
            return  # Acceleration
        loads = loads.rename(
            columns={"id": "bus", "node": "name", "p_load": "p_mw", "q_load": "q_mvar"}
        )
        # get a new index
        loads = loads.reset_index(level=0, drop=True)
        loads["scaling"] = 1
        loads["in_service"] = True
        loads["const_z_percent"] = 0
        loads["const_i_percent"] = 0
        self._copy_to_pp("load", loads)
        self.logger.info("Finished converting the loads.")

    def _convert_gens(self):
        self.logger.info("Converting the generators.")
        # select the gens from the nodes and drop not given values
        gens = self.u_d["N"].dropna(subset=["p_gen", "q_gen"])
        # select all with p != 0 or q != 0 or voltage != 0
        gens = gens.loc[
            (gens["p_gen"] != 0) | (gens["q_gen"] != 0) | (gens["voltage"] > 0)
        ]
        # change the signing
        gens["p_gen"] = gens["p_gen"] * -1
        gens["q_gen"] = gens["q_gen"] * -1
        gens["min_p_gen"] = gens["min_p_gen"] * -1
        gens["max_p_gen"] = gens["max_p_gen"] * -1
        gens["min_q_gen"] = gens["min_q_gen"] * -1
        gens["max_q_gen"] = gens["max_q_gen"] * -1
        # drop all voltages at non pu nodes
        gens.loc[
            (gens["node_type"] != 2) & (gens["node_type"] != 3), "voltage"
        ] = np.nan
        gens["vm_pu"] = gens["voltage"] / gens["vn_kv"]
        gens = gens.rename(
            columns={
                "id": "bus",
                "node": "name",
                "p_gen": "p_mw",
                "q_gen": "q_mvar",
                "min_p_gen": "min_p_mw",
                "max_p_gen": "max_p_mw",
                "min_q_gen": "min_q_mvar",
                "max_q_gen": "max_q_mvar",
            }
        )
        # get a new index
        gens = gens.reset_index(level=0, drop=True)
        gens["scaling"] = 1
        gens["va_degree"] = 0
        gens["slack_weight"] = 1
        gens["slack"] = False
        gens["current_source"] = True
        gens["in_service"] = True
        self._copy_to_pp("ext_grid", gens.loc[gens["node_type"] == 3])
        self._copy_to_pp("gen", gens.loc[gens["node_type"] == 2])
        self._copy_to_pp(
            "sgen", gens.loc[(gens["node_type"] == 0) | (gens["node_type"] == 1)]
        )
        self.logger.info("Finished converting the generators.")

    def _convert_lines(self):
        self.logger.info("Converting the lines.")
        # get the lines
        # status 9 & 1 stands for equivalent line that can be interpreted as impedance
        # status 7 & 2 stands for busbar coupler that can be interpreted as switches
        lines = self.u_d["L"].loc[self.u_d["L"].status.isin([9, 1, 7, 2]) == False, :]
        # definition of busbar coupler is if r, x, b are all zero, but some statuses are wrong, check for those
        lines = lines.drop(
            lines.loc[(lines.r == 0) & (lines.x == 0) & (lines.b == 0)].index
        )
        # also drop lines with x < 0 as they will be modeled as impedances
        lines = lines.drop(lines.loc[lines.x < 0].index)
        if not len(lines):
            self.logger.info("Finished converting the lines (no lines existing).")
            return  # Acceleration
        # lines = self.u_d['L']
        # create the in_service column from the UCTE status
        in_service_map = dict({0: True, 1: True, 2: True, 7: False, 8: False, 9: False})
        lines["in_service"] = lines["status"].map(in_service_map)
        # i in A to i in kA
        lines["max_i_ka"] = lines["i"] / 1e3
        lines["max_i_ka"] = lines["max_i_ka"].fillna(9999)
        lines["c_nf_per_km"] = 1e3 * lines["b"] / (2 * np.pi * 50)
        lines["g_us_per_km"] = 0
        lines["df"] = 1
        lines["parallel"] = 1
        lines["length_km"] = 1
        self._fill_empty_names(lines)
        self._fill_amica_names(lines, ":line")
        lines.loc[lines.x == 0, "x"] = 0.01
        # rename the columns to the pandapower schema
        lines = lines.rename(
            columns={"r": "r_ohm_per_km", "x": "x_ohm_per_km", "name": "name"}
        )
        self._copy_to_pp("line", lines)
        self.logger.info("Finished converting the lines.")

    def _convert_impedances(self):
        self.logger.info("Converting the impedances.")
        # get the impedances
        # status 9 & 1 stands for equivalent line that can be interpreted as impedance
        status_9_1 = self.u_d["L"].status.isin([9, 1])
        # also all lines with negative x convert to impedances
        negative_x = self.u_d["L"].x < 0
        impedances = self.u_d["L"].loc[status_9_1 | negative_x, :]
        self._set_column_to_type(impedances, "from_bus", int)
        impedances = pd.merge(
            impedances,
            self.u_d["N"][["vn_kv"]],
            how="left",
            left_on="from_bus",
            right_index=True,
        )

        trafos_to_impedances = self._get_trafos_modelled_as_impedances()
        impedances = pd.concat([impedances, trafos_to_impedances])

        # create the in_service column from the UCTE status
        in_service_map = dict({0: True, 1: True, 2: True, 7: False, 8: False, 9: False})
        impedances["in_service"] = impedances["status"].map(in_service_map)
        # Convert ohm/km to per unit (pu)
        impedances["sn_mva"] = 10000  # same as PowerFactory
        impedances["z_ohm"] = impedances["vn_kv"] ** 2 / impedances["sn_mva"]
        impedances["rft_pu"] = impedances["r"] / impedances["z_ohm"]
        impedances["rtf_pu"] = impedances["r"] / impedances["z_ohm"]
        impedances["xft_pu"] = impedances["x"] / impedances["z_ohm"]
        impedances["xtf_pu"] = impedances["x"] / impedances["z_ohm"]
        self._fill_empty_names(impedances)
        self._copy_to_pp("impedance", impedances)
        self.logger.info("Finished converting the impedances.")

    def _get_trafos_modelled_as_impedances(self):
        ### get transformers that will be transformed to impedances and append them ###
        trafos = pd.merge(
            self.u_d["T"],
            self.u_d["R"],
            how="left",
            on=["hv_bus", "lv_bus", "order_code"],
        )

        # check for trafos connecting same voltage levels
        trafos_to_impedances = trafos.loc[
            trafos.loc[:, "0_x"].map(lambda s: s[6])
            == trafos.loc[:, "0_x"].map(lambda s: s[15])
        ]

        trafos_to_impedances = trafos_to_impedances.loc[
            trafos_to_impedances.phase_reg_delta_u.isnull()
        ]
        trafos_to_impedances = trafos_to_impedances.loc[
            trafos_to_impedances.angle_reg_theta.isnull()
        ]
        # calculate iron losses in kW
        trafos_to_impedances["pfe_kw"] = (
            trafos_to_impedances.g * trafos_to_impedances.voltage1**2 / 1e3
        )
        # calculate open loop losses in percent of rated current
        trafos_to_impedances["i0_percent"] = (
            (
                (
                    (trafos_to_impedances.b * 1e-6 * trafos_to_impedances.voltage1**2)
                    ** 2
                    + (
                        trafos_to_impedances.g
                        * 1e-6
                        * trafos_to_impedances.voltage1**2
                    )
                    ** 2
                )
                ** 0.5
            )
            * 100
            / trafos_to_impedances.s
        )
        trafos_to_impedances = trafos_to_impedances.loc[
            (trafos_to_impedances.pfe_kw == 0) & (trafos_to_impedances.i0_percent == 0)
        ]
        # rename the columns to the pandapower schema, as voltages are the same we can take voltage1 as vn_kv
        trafos_to_impedances = trafos_to_impedances.rename(
            columns={
                "hv_bus": "from_bus",
                "lv_bus": "to_bus",
                "voltage1": "vn_kv",
                "0_x": 0,
            }
        )
        return trafos_to_impedances

    def _convert_switches(self):
        self.logger.info("Converting the switches.")
        # get the switches
        # status 7 & 2 stands for busbar coupler that can be interpreted as switch
        switches_by_status = self.u_d["L"].status.isin([7, 2])
        # switches are defined by r, x and b equal 0, but some still have the wrong status (or r, x and b not zero)
        lines_rxb_zero = (
            (self.u_d["L"].r == 0) & (self.u_d["L"].x == 0) & (self.u_d["L"].b == 0)
        )
        switches = self.u_d["L"].loc[lines_rxb_zero | switches_by_status, :]

        # create the in_service column from the UCTE status
        in_service_map = dict({0: True, 1: True, 2: True, 7: False, 8: False, 9: False})
        switches["closed"] = switches["status"].map(in_service_map)
        self._set_column_to_type(switches, "from_bus", int)
        switches["type"] = "LS"
        switches["et"] = "b"
        switches["z_ohm"] = 0
        self._fill_empty_names(switches)
        switches = switches.rename(columns={"from_bus": "bus", "to_bus": "element"})
        self._copy_to_pp("switch", switches)
        self.logger.info("Finished converting the switches.")

    def _convert_trafos(self):
        self.logger.info("Converting the transformers.")
        trafos = pd.merge(
            self.u_d["T"],
            self.u_d["R"],
            how="left",
            on=["hv_bus", "lv_bus", "order_code"],
        )
        if not len(trafos):
            self.logger.info("Finished converting the transformers (no transformers existing).")
            return
        # create the in_service column from the UCTE status
        status_map = dict({0: True, 1: True, 8: False, 9: False})
        trafos["in_service"] = trafos["status"].map(status_map)
        # use same value as in powerfactory for replacing s equals zero values
        trafos.loc[trafos.s == 0, "s"] = 1001
        # calculate the derating factor
        trafos["df"] = trafos["voltage1"] * (trafos["i"] / 1e3) * 3**0.5 / trafos["s"]
        # calculate the relative short-circuit voltage
        trafos["vk_percent"] = (
            np.sign(trafos.x)
            * (abs(trafos.r) ** 2 + abs(trafos.x) ** 2) ** 0.5
            * (trafos.s * 1e3)
            / (10.0 * trafos.voltage1**2)
        )
        # calculate vkr_percent
        trafos["vkr_percent"] = trafos.r * trafos.s * 100 / trafos.voltage1**2
        # calculate iron losses in kW
        trafos["pfe_kw"] = trafos.g * trafos.voltage1**2 / 1e3
        # calculate open loop losses in percent of rated current
        trafos["i0_percent"] = (
            (
                (
                    (trafos.b * 1e-6 * trafos.voltage1**2) ** 2
                    + (trafos.g * 1e-6 * trafos.voltage1**2) ** 2
                )
                ** 0.5
            )
            * 100
            / trafos.s
        )

        # phase and angle regulation have to be split up into 5 cases:
        # only phase regulated -> pr
        # only angle regulated symmetrical model -> ars
        # only angle regulated asymmetrical model -> ara
        # phase and angle regulated symmetrical model -> pars
        # phase and angle regulated asymmetrical model -> para
        # set values for only phase regulated transformers (pr)
        has_phase_values = (
            (~trafos.phase_reg_delta_u.isnull())
            & (~trafos.phase_reg_n.isnull())
            & (~trafos.phase_reg_n2.isnull())
        )
        has_missing_angle_values = (
            trafos.angle_reg_delta_u.isnull()
            | trafos.angle_reg_theta.isnull()
            | trafos.angle_reg_n.isnull()
            | trafos.angle_reg_n2.isnull()
        )
        pr = trafos.loc[has_phase_values & has_missing_angle_values].index

        trafos.loc[pr, "tap_min"] = -trafos["phase_reg_n"]
        trafos.loc[pr, "tap_max"] = trafos["phase_reg_n"]
        trafos.loc[pr, "tap_pos"] = trafos["phase_reg_n2"]
        trafos.loc[pr, "tap_step_percent"] = trafos.loc[pr, "phase_reg_delta_u"].abs()
        trafos.loc[pr, "tap_changer_type"] = "Ratio"

        # set values for only angle regulated transformers symmetrical and asymmetrical
        has_missing_phase_values = (
            trafos.phase_reg_delta_u.isnull()
            & trafos.phase_reg_n.isnull()
            & trafos.phase_reg_n2.isnull()
        )
        has_angle_values = (
            (~trafos.angle_reg_delta_u.isnull())
            & (~trafos.angle_reg_theta.isnull())
            & (~trafos.angle_reg_n.isnull())
            & (~trafos.angle_reg_n2.isnull())
        )
        ar = trafos.loc[has_missing_phase_values & has_angle_values].index

        symm = trafos.angle_reg_type == "SYMM"
        ars = trafos.loc[has_missing_phase_values & has_angle_values & symm].index
        trafos.loc[ars, "tap_min"] = -trafos.loc[ar, "angle_reg_n"]
        trafos.loc[ars, "tap_max"] = trafos.loc[ar, "angle_reg_n"]
        trafos.loc[ars, "tap_pos"] = trafos.loc[ar, "angle_reg_n2"]
        trafos.loc[ars, "tap_step_percent"] = np.nan
        # trafos.loc[ars, 'phase_reg_n'] = trafos.loc[ar, 'angle_reg_n']
        trafos.loc[ars, "tap_changer_type"] = "Ideal"
        trafos.loc[
            ars, "tap_step_degree"
        ] = self._calculate_tap_step_degree_symmetrical(trafos.loc[ars])

        asym = (trafos.angle_reg_type == "ASYM") | (trafos.angle_reg_type == "")
        ara = trafos.loc[has_missing_phase_values & has_angle_values & asym].index
        trafos.loc[ara, "tap2_min"] = -trafos.loc[ar, "angle_reg_n"]
        trafos.loc[ara, "tap2_max"] = trafos.loc[ar, "angle_reg_n"]
        trafos.loc[ara, "tap2_pos"] = trafos.loc[ar, "angle_reg_n2"]
        trafos.loc[ara, "tap2_neutral"] = 0
        trafos.loc[ara, "tap2_step_percent"] = np.nan
        trafos.loc[ara, "tap2_changer_type"] = "Ideal"
        trafos.loc[
            ara, "tap2_step_degree"
        ] = self._calculate_tap_step_degree_asymmetrical(trafos.loc[ara])

        trafos.loc[ara, "tap_min"] = -trafos.loc[ara, "angle_reg_n"]
        trafos.loc[ara, "tap_max"] = trafos.loc[ara, "angle_reg_n"]
        trafos.loc[ara, "tap_pos"] = trafos.loc[ara, "angle_reg_n2"]
        trafos.loc[ara, "tap_changer_type"] = "Ratio"
        trafos.loc[
            ara, "tap_step_percent"
        ] = self._calculate_tap_step_percent_asymmetrical(trafos.loc[ara])

        # get phase and angle regulated transformers symmetrical and asymmetrical
        par = trafos.loc[has_phase_values & has_angle_values].index

        trafos.loc[par, "tap_step_percent"] = trafos.loc[par, "phase_reg_delta_u"].abs()
        trafos.loc[par, "tap_min"] = -trafos.loc[par, "phase_reg_n"]
        trafos.loc[par, "tap_max"] = trafos.loc[par, "phase_reg_n"]
        trafos.loc[par, "tap_pos"] = trafos.loc[par, "phase_reg_n2"]
        trafos.loc[par, "tap_changer_type"] = "Ratio"

        trafos.loc[par, "tap2_min"] = -trafos.loc[par, "angle_reg_n"]
        trafos.loc[par, "tap2_max"] = trafos.loc[par, "angle_reg_n"]
        trafos.loc[par, "tap2_neutral"] = 0
        trafos.loc[par, "tap2_pos"] = trafos.loc[par, "angle_reg_n2"]
        trafos.loc[par, "tap2_step_percent"] = np.nan
        trafos.loc[par, "tap2_changer_type"] = "Ideal"

        pars = trafos.loc[has_phase_values & has_angle_values & symm].index
        trafos.loc[
            pars, "tap2_step_degree"
        ] = self._calculate_tap_step_degree_symmetrical(trafos.loc[pars])

        para = trafos.loc[has_phase_values & has_angle_values & asym].index
        trafos.loc[
            para, "tap2_step_degree"
        ] = self._calculate_tap_step_degree_asymmetrical(trafos.loc[para])
        trafos.loc[para, "tap_step_percent"] = trafos.loc[
            para, "tap_step_percent"
        ] + self._calculate_tap_step_percent_asymmetrical(trafos.loc[para])

        # change signs of tap pos for negative degree or percentage values, since pp only allows positive values
        # trafos.loc[trafos.tap_step_percent < 0, ['tap_pos', 'tap_step_percent']] = trafos.loc[trafos.tap_step_percent < 0, ['tap_pos', 'tap_step_percent']] * -1
        # trafos.loc[trafos.tap_step_degree < 0, ['tap_pos', 'tap_step_degree']] = trafos.loc[trafos.tap_step_degree < 0, ['tap_pos', 'tap_step_degree']] * -1
        # trafos.loc[trafos.tap2_step_degree < 0, ['tap2_pos', 'tap2_step_degree']] = trafos.loc[trafos.tap2_step_degree < 0, ['tap2_pos', 'tap2_step_degree']] * -1

        # set the hv and lv voltage sides to voltage1 and voltage2 (The non-regulated transformer side is currently
        # voltage1, not the hv side!)
        trafos["vn_hv_kv"] = trafos[["voltage1", "voltage2"]].max(axis=1)
        trafos["vn_lv_kv"] = trafos[["voltage1", "voltage2"]].min(axis=1)
        # swap the 'fid_node_start' and 'fid_node_end' if need
        trafos["swap"] = trafos["vn_hv_kv"] != trafos["voltage1"]
        # copy the 'fid_node_start' and 'fid_node_end'
        trafos["hv_bus2"] = trafos["hv_bus"].copy()
        trafos["lv_bus2"] = trafos["lv_bus"].copy()
        trafos.loc[trafos.swap, "hv_bus"] = trafos.loc[trafos.swap, "lv_bus2"]
        trafos.loc[trafos.swap, "lv_bus"] = trafos.loc[trafos.swap, "hv_bus2"]
        # set the tap side, default is lv Correct it for other windings
        trafos["tap_side"] = "lv"
        trafos["tap2_side"] = "lv"
        trafos.loc[trafos.swap, "tap_side"] = "hv"
        trafos.loc[trafos.swap, "tap2_side"] = "hv"
        # now set it to nan for not existing tap changers
        trafos.loc[trafos.phase_reg_n.isnull(), "tap_side"] = None
        trafos["tap_neutral"] = 0
        trafos.loc[trafos.phase_reg_n.isnull(), "tap_neutral"] = np.nan
        trafos["shift_degree"] = 0
        trafos["parallel"] = 1
        self._fill_empty_names(trafos, "0_x")
        self._fill_amica_names(trafos, ":trf", "0_x")
        trafos["tap_changer_type"] = trafos["tap_changer_type"].fillna("Ratio").astype(str)
        trafos["tap2_changer_type"] = trafos["tap2_changer_type"].fillna("Ratio").astype(str)
        # rename the columns to the pandapower schema
        trafos = trafos.rename(columns={"s": "sn_mva"})
        # drop transformers that will be transformed to impedances
        trafos_to_impedances = self._get_trafos_modelled_as_impedances()
        trafos = trafos.drop(trafos_to_impedances.index)

        self._copy_to_pp("trafo", trafos)
        self.logger.info("Finished converting the transformers.")

    def _calculate_tap_step_degree_symmetrical(
        self, input_df: pd.DataFrame
    ) -> pd.Series:
        return input_df["angle_reg_delta_u"].apply(lambda du: 2 * math.atan2(du, 2))

    def _calculate_tap_step_degree_asymmetrical(
        self, input_df: pd.DataFrame
    ) -> pd.Series:
        numerator = input_df["angle_reg_delta_u"] * input_df["angle_reg_theta"].apply(
            lambda t: math.sin(t)
        )
        denominator = 1 + (
            input_df["angle_reg_delta_u"]
            * input_df["angle_reg_theta"].apply(lambda t: math.cos(t))
        )
        return (numerator / denominator).apply(lambda t: math.atan2(t, 1))

    def _calculate_tap_step_percent_asymmetrical(
        self, input_df: pd.DataFrame
    ) -> pd.Series:
        sine_sq = (
            input_df["angle_reg_delta_u"]
            * input_df["angle_reg_theta"].apply(lambda t: math.sin(t))
        ) ** 2
        cosine_sq = (
            input_df["angle_reg_delta_u"]
            * input_df["angle_reg_theta"].apply(lambda t: math.cos(t))
        ) ** 2
        tab_step_percent = (sine_sq + cosine_sq).apply(lambda x: np.sqrt(x)) - 1
        return tab_step_percent

    def _set_column_to_type(self, input_df: pd.DataFrame, column: str, data_type):
        try:
            input_df[column] = input_df[column].astype(data_type)
        except Exception as e:
            self.logger.error(
                "Couldn't set data type %s for column %s!" % (data_type, column)
            )
            self.logger.exception(e)

    def _fill_empty_names(self, input_df: pd.DataFrame, input_column: Union[str, int] = 0):
        """Fills empty names with node1_node2_order-code"""

        def get_name_from_ucte_string(ucte_string: str) -> str:
            return f"{ucte_string[:8].strip()}_{ucte_string[9:17].strip()}_{ucte_string[18]}"

        new_names = input_df.loc[input_df["name"] == "", input_column].map(
            get_name_from_ucte_string
        )
        input_df.loc[input_df["name"] == "", "name"] = new_names

    def _fill_amica_names(
        self, input_df: pd.DataFrame, suffix: str, input_column: Union[str, int] = 0
    ) -> None:
        def get_name_from_ucte_string(ucte_string: str) -> str:
            node1 = ucte_string[:7].replace(" ", "_")
            node2 = ucte_string[9:16].replace(" ", "_")
            order_code = ucte_string[18]
            # for some cases the element name is taken instead of the order code, but not clear when
            # taking always the order code leads to more matches, at least for lines...
            return f"{node1}_{node2}_{order_code}{suffix}"

        amica_names = input_df.loc[:, input_column].map(get_name_from_ucte_string)
        input_df.loc[:, "amica_name"] = amica_names

    def set_pp_col_types(
        self,
        net: Union[pandapower.auxiliary.pandapowerNet, Dict],
        ignore_errors: bool = False,
    ) -> pandapower.auxiliary.pandapowerNet:
        """
        Set the data types for some columns from pandapower assets. This mainly effects bus columns (to int, e.g.
        sgen.bus or line.from_bus) and in_service and other boolean columns (to bool, e.g. line.in_service or gen.slack).
        :param net: The pandapower network to update the data types.
        :param ignore_errors: Ignore problems if set to True (no warnings displayed). Optional, default: False.
        :return: The pandapower network with updated data types.
        """
        time_start = time.time()
        pp_elements = [
            "bus",
            "dcline",
            "ext_grid",
            "gen",
            "impedance",
            "line",
            "load",
            "sgen",
            "shunt",
            "storage",
            "switch",
            "trafo",
            "trafo3w",
            "ward",
            "xward",
        ]
        to_int = ["bus", "element", "to_bus", "from_bus", "hv_bus", "mv_bus", "lv_bus"]
        to_bool = ["in_service", "closed"]
        self.logger.info(
            "Setting the columns data types for buses to int and in_service to bool for the following elements: "
            "%s" % pp_elements
        )
        int_type = int
        bool_type = bool
        for ele in pp_elements:
            self.logger.info("Accessing pandapower element %s." % ele)
            if not hasattr(net, ele):
                if not ignore_errors:
                    self.logger.warning(
                        "Missing the pandapower element %s in the input pandapower network!"
                        % ele
                    )
                continue
            for one_int in to_int:
                if one_int in net[ele].columns:
                    self._set_column_to_type(net[ele], one_int, int_type)
            for one_bool in to_bool:
                if one_bool in net[ele].columns:
                    self._set_column_to_type(net[ele], one_bool, bool_type)
        # some individual things
        if hasattr(net, "sgen"):
            self._set_column_to_type(net["sgen"], "current_source", bool_type)
        if hasattr(net, "gen"):
            self._set_column_to_type(net["gen"], "slack", bool_type)
        if hasattr(net, "shunt"):
            self._set_column_to_type(net["shunt"], "step", int_type)
            self._set_column_to_type(net["shunt"], "max_step", int_type)
        self.logger.info(
            "Finished setting the data types for the pandapower network in %ss."
            % (time.time() - time_start)
        )
        return net
