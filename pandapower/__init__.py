__all__ = ["__format_version__", "__version__", "pp_dir"]

import os

import geojson

from pandapower._version import __format_version__, __version__

geojson.geometry.DEFAULT_PRECISION = 8

pp_dir: str = os.path.dirname(os.path.realpath(__file__))
