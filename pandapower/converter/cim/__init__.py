# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from .cim2pp import from_cim
from .pp2cim import to_cim

__all__ = ["from_cim", "to_cim"]

__version__ = '3.6.10'
