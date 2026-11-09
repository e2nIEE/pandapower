"""

@author: mrichter
"""

from __future__ import annotations

import json

import pandas as pd


class ReportContainer:
    pass


class CustomJSONEncoder(json.JSONEncoder):
    pass


class CustomLogger(CustomJSONEncoder):
    pass


class AssetType(CustomLogger):
    pass


class RepairMain(CustomLogger):
    def __init__(self, assets: list[AssetType] | None = None):
        super().__init__()
        self.assets = assets

    def get_assets(self):
        pass

    def repair(self, data: dict[str, pd.DataFrame], report_container: ReportContainer | None = None):
        pass

    def deserialize(self, path_or_json_str: str, report_container: ReportContainer) -> RepairMain:  # type: ignore[empty-body]
        pass


class PandapowerRepair(RepairMain):
    def __init__(self, assets: list[AssetType] | None = None):
        super().__init__(assets)

    def deserialize(  # type: ignore[empty-body]
        self, path_or_json_str: str | PandapowerRepair, report_container: ReportContainer | None = None
    ) -> PandapowerRepair:
        pass


class CIMRepair(CustomLogger):
    def __init__(self, profiles: dict[str, RepairMain] | None = None):
        super().__init__()
        self.profiles = profiles

    def set_profile(self, profile: str, repair_main: RepairMain):
        pass

    def repair(self, cim_dict: dict[str, dict[str, pd.DataFrame]], report_container: ReportContainer | None = None):
        pass

    def deserialize(  # type: ignore[empty-body]
        self, path_or_json_str: str | CIMRepair, report_container: ReportContainer | None = None
    ) -> CIMRepair:
        pass
