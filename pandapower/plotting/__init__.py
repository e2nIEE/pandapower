from pandapower.plotting.collections import *
from pandapower.plotting.colormaps import *
from pandapower.plotting.generic_geodata import *
from pandapower.plotting.powerflow_results import *
from pandapower.plotting.simple_plot import *
from pandapower.plotting.plotly import *
from pandapower.plotting.geo import *
from pandapower.plotting.plotting_toolbox import set_line_geodata_from_bus_geodata
from pandapower.plotting.to_html import to_html

import types

try:
    from matplotlib.backend_bases import GraphicsContextBase, RendererBase

    class GC(GraphicsContextBase):
        def __init__(self):
            super().__init__()
            self.set_capstyle('round')

    def custom_new_gc(self):
        return GC()

    RendererBase.new_gc = types.MethodType(custom_new_gc, RendererBase)

except ImportError:
    pass
