# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import os

from pandapower import __version__ as pp_version
from pandapower.control.controller.trafo import DiscreteTapControl
from pandapower.control.util.auxiliary import create_trafo_characteristics
from pandapower.file_io import to_json
from pandapower.networks.create_examples import example_multivoltage
from pandapower.run import runpp

net = example_multivoltage()
DiscreteTapControl(net, 1, 1.02, 1.03)
create_trafo_characteristics(net, "trafo", 1, "vk_percent", [-2, 0, 2], [3.5, 4, 4.5])
runpp(net, run_control=True)
to_json(net, os.path.join("old_versions", "example_%s.json" % pp_version))
