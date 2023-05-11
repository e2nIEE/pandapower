import pandapower as pp

import os
import pytest

from pandapower.converter import from_cim as cim2pp


def test_cim2pp():
    folder_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../tutorials/example_cim"))

    cgmes_files = [os.path.join(folder_path, 'CGMES_v2.4.15_SmallGridTestConfiguration_Boundary_v3.0.0.zip'),
                   os.path.join(folder_path, 'CGMES_v2.4.15_SmallGridTestConfiguration_BaseCase_Complete_v3.0.0.zip')]

    net = cim2pp.from_cim(file_list=cgmes_files, use_GL_or_DL_profile='GL')

    assert 118 == len(net.bus.index)
    assert 115 == len(net.bus_geodata.index)


if __name__ == "__main__":
    pytest.main([__file__])
