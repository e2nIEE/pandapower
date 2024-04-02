# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapower as pp
import os

net = pp.networks.example_multivoltage()
pp.control.DiscreteTapControl(net, 1, 1.02, 1.03)
pp.control.create_trafo_characteristics(net, "trafo", 1, "vk_percent", [-2, 0, 2], [3.5, 4, 4.5])
pp.runpp(net, run_control=True)
pp.to_json(net, os.path.join("old_versions", "example_%s.json" % pp.__version__))
