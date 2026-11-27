# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import subprocess
import sys


def test_import_without_matplotlib():
    # matplotlib is an optional dependency (pandapower[plotting]), so pandapower and its plotting package have to be
    # importable without it. A None entry in sys.modules makes find_spec() report matplotlib as not installed and
    # makes importing it fail. The import runs in a new interpreter, as pandapower is already imported here.
    code = "import sys; sys.modules['matplotlib'] = None; import pandapower; import pandapower.plotting"
    result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
