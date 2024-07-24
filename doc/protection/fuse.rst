###################################################
Fuse
###################################################

A fuse is a protection device frequently used in low-voltage networks. When exposed to a sufficiently high current, the fuse melts and an arc forms. As the fuse continues to melt, the arc distance increases until it eventually is extinguished, and no more current flows through the fuse.

pandapower allows users to add fuses to networks and analyze their performance in short-circuit and power flow scenarios. It includes a standard library of fuses with rated currents ranging from 16A to 1000A.

Fuses can be created using the Fuse class:

.. autoclass:: pandapower.protection.protection_devices.fuse.Fuse
    :members:
    :class-doc-from: class

To run protection calculations, use the calculate_protection_times function:

.. autofunction:: pandapower.protection.run_protection.calculate_protection_times
    :noindex:

Kindly follow the Fuse tutorial on Github for more details.
https://github.com/e2nIEE/pandapower/blob/develop/tutorials/protection/fuse.ipynb
