import os

pp_dir: str = os.path.dirname(os.path.realpath(__file__))

import geojson

from pandapower._version import __format_version__, __version__
from pandapower.auxiliary import *  # TODO: some functions shouldn't be available with import pandapower
from pandapower.convert_format import *
from pandapower.create import *
from pandapower.diagnostic import *
from pandapower.file_io import *
from pandapower.groups import *
from pandapower.network import ADict, pandapowerNet
from pandapower.optimal_powerflow import OPFNotConverged
from pandapower.pf.runpp_3ph import runpp_3ph
from pandapower.powerflow import *
from pandapower.run import *
from pandapower.runpm import *
from pandapower.std_types import *

geojson.geometry.DEFAULT_PRECISION = 8


# import pandapower packages
from pandapower import control as control
from pandapower import estimation as estimation
from pandapower import grid_equivalents as grid_equivalents
from pandapower import networks as networks
from pandapower import plotting as plotting
from pandapower import shortcircuit as shortcircuit
from pandapower import timeseries as timeseries
from pandapower import toolbox as toolbox
from pandapower import topology as topology

__all__ = ["ADict", "OPFNotConverged", "__format_version__", "__version__", "pandapowerNet", "runpp_3ph"]
