import pytest
import os
from pandapower import pp_dir

import pandapower.networks as nw
import pandapower.converter as cv


def test_filter_loadcases():
    test_network_path = os.path.join(pp_dir, "test", "converter", "test_network")
    data = cv.read_csv_data(test_network_path, ";")
    assert data["StudyCases"].shape[0] == 24
    assert data["StudyCases"].shape[1]
    nw.filter_loadcases(data, factors=None)
    assert data["StudyCases"].shape[0] == 6


if __name__ == "__main__":
    if 0:
        pytest.main(["test_simbench_loadcases.py", "-xs"])
    else:
#        test_filter_loadcases()
        pass
