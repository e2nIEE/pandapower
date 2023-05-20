## the code below commented is as the same as tutorial of "cim2pp.ipynb" 

# import os
# import pytest
# from pandapower.test import test_path

# from pandapower.converter import from_cim as cim2pp


# def test_cim2pp():
#     folder_path = os.path.join(test_path, "test_files", "example_cim")

#     cgmes_files = [os.path.join(folder_path, 'CGMES_v2.4.15_SmallGridTestConfiguration_Boundary_v3.0.0.zip'),
#                    os.path.join(folder_path, 'CGMES_v2.4.15_SmallGridTestConfiguration_BaseCase_Complete_v3.0.0.zip')]

#     net = cim2pp.from_cim(file_list=cgmes_files, use_GL_or_DL_profile='GL')

#     assert 118 == len(net.bus.index)
#     assert 115 == len(net.bus_geodata.index)


""" this pytest whih fucntion of "find_files" is an optional method which try to be reusable,
the commented code above is more easy and simple for guys focus on prime concept."""

import os
import pytest
from pandapower.file_io import find_files

import pandapower as pp
from pandapower.converter import from_cim

import matplotlib.pyplot as plt

from pandapower.plotting import collections as plot
import pandapower.plotting.colormaps as cmaps

cgmes_files = []

@pytest.fixture()
def detect_cgmes():
    # define keywords of the target files for search
    key_words = ["BaseCase", "Boundary"] # or more detailly and unique like "BaseCase_Complete_v3.0.0" to avoid choose wrong files but in similar name

    # here "../.." is optional, which means it's fine for moving to the parent folder of all files with the absolute path ignored.
    folder_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

    find_files(folder_path, cgmes_files, key_words)
    return cgmes_files
    
@pytest.fixture
def cgmes_plot():

    net = from_cim.from_cim(cgmes_files, use_GL_or_DL_profile='DL')

    assert 118 == len(net.bus.index)
    assert 118 == len(net.bus_geodata.index)

    for f in cgmes_files:
        if not os.path.exists(f):
            raise UserWarning(f"Wrong path specified for the CGMES file {f}")

    pp.runpp(net)
    print(net.res_bus.iloc[0:5]) # print first few bus results
    cmap_list = [(0.9, "blue"), (1.0, "green"), (1.1, "red")]
    cmap, norm = cmaps.cmap_continuous(cmap_list)

    bc = plot.create_bus_collection(net,net.bus.index.values,cmap=cmap)
    lc = plot.create_line_collection(net,net.line.index.values,use_bus_geodata=True)
    tc = plot.create_trafo_collection(net,net.trafo.index.values)
    plot.draw_collections([bc,lc,tc])

def test_detect_cgmes(detect_cgmes: list):
    assert len(cgmes_files) == 2
    assert 'Boundary_v3.0.0' in cgmes_files[0]
    assert 'BaseCase_Complete_v3.0.0' in cgmes_files[1]
    
def test_cgmes_plot(cgmes_plot: None):
    return plt.show()

if __name__ == "__main__":
    pytest.main([__file__])
