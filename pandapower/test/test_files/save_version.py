# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import os
import numpy as np
import pandas as pd

from pandapower import pp_dir
from pandapower import __version__ as pp_version
from pandapower.control.controller.trafo.DiscreteTapControl import DiscreteTapControl
from pandapower.control.util.auxiliary import create_trafo_characteristic_object
from pandapower.file_io import to_json
from pandapower.networks.create_examples import example_multivoltage
from pandapower.run import runpp

net = example_multivoltage()
DiscreteTapControl(net, 1, 1.02, 1.03)
net["trafo_characteristic_table"] = pd.DataFrame(
    {'id_characteristic': [0, 0, 0], 'step': [-2, 0, 2],
     'voltage_ratio': np.nan, 'angle_deg': np.nan,
     'vk_percent': [3.5, 4, 4.5], "vkr_percent": np.nan,
     'vk_hv_percent': np.nan, 'vkr_hv_percent': np.nan,
     'vk_mv_percent': np.nan, 'vkr_mv_percent': np.nan,
     'vk_lv_percent': np.nan, 'vkr_lv_percent': np.nan})
net.trafo.at[1, 'id_characteristic_table'] = 0
create_trafo_characteristic_object(net)
runpp(net, run_control=True)
to_json(net, os.path.join(pp_dir, "test", "test_files", "old_versions", f"example_{pp_version}.json"))
