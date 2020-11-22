import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir, "src")
)

import julia  # isort:skip

julia.install(color=True)