#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: mrichter
"""
from __future__ import annotations
from typing import List, Dict, Union
import pandas as pd
import json


class ReportContainer:
    pass


class CustomJSONEncoder(json.JSONEncoder):
    pass


class CustomLogger(CustomJSONEncoder):
    pass


class AssetType(CustomLogger):
    pass


class RepairMain(CustomLogger):
    def __init__(self, assets: List[str] = None):
        super().__init__()
        self.assets = assets

    def get_assets(self):
        pass

    def repair(self, data: Dict[str, pd.DataFrame], report_container: ReportContainer = None):
        pass

    def deserialize(self, path_or_json_str: str, report_container: ReportContainer) -> RepairMain:
        pass


class PandapowerRepair(RepairMain):
    def __init__(self, assets: List[AssetType] = None):
        super().__init__(assets)

    def deserialize(self, path_or_json_str: Union[str, PandapowerRepair], report_container: ReportContainer = None) -> \
            PandapowerRepair:
        pass


class CIMRepair(CustomLogger):
    def __init__(self, profiles: Dict[str, RepairMain] = None):
        super().__init__()
        self.profiles = profiles

    def set_profile(self, profile: str, repair_main: RepairMain):
        pass

    def repair(self, cim_dict: Dict[str, Dict[str, pd.DataFrame]], report_container: ReportContainer = None):
        pass

    def deserialize(self, path_or_json_str: Union[str, CIMRepair], report_container: ReportContainer = None) -> \
            CIMRepair:
        pass
