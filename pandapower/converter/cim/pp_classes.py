# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import logging
from typing import Dict
import json
import pandapower as pp
import pandapower.auxiliary
from . import cim_tools


class PandapowerDiagnostic:
    """
    Create a pandapower diagnostic dictionary with CIM IDs instead of pandapower IDs.
    :param net: The pandapower network.
    :param diagnostic: The pandapower diagnostic. If None a pp.diagnostic(net) will be run. Optional, default: None.
    :return: The pandapower diagnostic with CIM IDs.
    """
    def __init__(self, net: pandapower.auxiliary.pandapowerNet, diagnostic: Dict = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.net = net
        self.diagnostic = diagnostic

    def _rec_replace_pp_diagnostic_with_cim_ids(self, input_obj, element_type: str = None):
        sc = cim_tools.get_pp_net_special_columns_dict()
        element_mapping = dict({
            'bus': 'bus', 'buses': 'bus', 'load': 'load', 'loads': 'load', 'sgen': 'sgen', 'sgens': 'sgen',
            'motor': 'motor', 'motors': 'motor',
            'asymmetric_load': 'asymmetric_load', 'asymmetric_loads': 'asymmetric_load',
            'asymmetric_sgen': 'asymmetric_sgen', 'asymmetric_sgens': 'asymmetric_sgen',
            'storage': 'storage', 'storages': 'storage', 'gen': 'gen', 'gens': 'gen',
            'switch': 'switch', 'switches': 'switch', 'shunt': 'shunt', 'shunts': 'shunt',
            'ext_grid': 'ext_grid', 'ext_grids': 'ext_grid', 'line': 'line', 'lines': 'line',
            'trafo': 'trafo', 'trafos': 'trafo', 'trafo3w': 'trafo3w', 'trafos3w': 'trafo3w',
            'impedance': 'impedance', 'impedances': 'impedance', 'dcline': 'dcline', 'dclines': 'dcline',
            'ward': 'ward', 'wards': 'ward', 'xward': 'xward', 'xwards': 'xward'})
        if isinstance(input_obj, list):
            return_obj = []
            for one_input_obj in input_obj:
                if isinstance(one_input_obj, list) or isinstance(one_input_obj, dict):
                    return_obj.append(self._rec_replace_pp_diagnostic_with_cim_ids(one_input_obj, element_type))
                elif element_type is not None and isinstance(one_input_obj, int):
                    # get the RDF ID direct
                    return_obj.append(self.net[element_type][sc['o_id']].at[one_input_obj])
                elif element_type is not None and isinstance(one_input_obj, tuple):
                    # the first item from the tuple should be the element index
                    one_input_obj = list(one_input_obj)
                    if one_input_obj[0] in self.net[element_type].index.values:
                        one_input_obj[0] = self.net[element_type][sc['o_id']].at[one_input_obj[0]]
                    return_obj.append(tuple(one_input_obj))
                else:
                    # default
                    return_obj.append(one_input_obj)
        elif isinstance(input_obj, dict):
            return_obj = dict()
            for key, item in input_obj.items():
                if isinstance(item, list) or isinstance(item, dict) and key in element_mapping.keys():
                    element_type = element_mapping[key]
                    return_obj[key] = self._rec_replace_pp_diagnostic_with_cim_ids(item, element_type)
                else:
                    # default
                    return_obj[key] = item
        else:
            return_obj = input_obj
        return return_obj

    def replace_pp_diagnostic_with_cim_ids(self) -> Dict:
        """
        Create a pandapower diagnostic dictionary with CIM IDs instead of pandapower IDs.
        :param net: The pandapower network.
        :param diagnostic: The pandapower diagnostic. If None a pp.diagnostic(net) will be run. Optional, default: None.
        :return: The pandapower diagnostic with CIM IDs.
        """
        if self.diagnostic is None:
            self.diagnostic = pp.diagnostic(self.net)
        result_diagnostic = dict()
        for key, item in self.diagnostic.items():
            result_diagnostic[key] = self._rec_replace_pp_diagnostic_with_cim_ids(item)
        # add the CGMES IDs
        if hasattr(self.net, 'CGMES'):
            result_diagnostic['CGMES_IDs'] = dict()
            for one_prf, one_prf_item in self.net['CGMES'].items():
                result_diagnostic['CGMES_IDs'][one_prf] = list(one_prf_item.keys())
        return result_diagnostic

    def serialize(self, diagnostic: Dict, path_to_store: str):
        """
        Serialize a pandapower diagnostic as json to disk.
        :param diagnostic: The pandapower diagnostic dictionary.
        :param path_to_store: The path to store the json on the disk.
        :return:
        """
        self.logger.info("Storing diagnostic to path: %s" % path_to_store)
        with open(path_to_store, mode='w', encoding='UTF-8') as fp:
            json.dump(diagnostic, fp, indent=2, sort_keys=True)
