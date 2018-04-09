.. _ppDCOPF:

DC Optimal Power Flow
=====================

The dc optimal power flow is a linearized optimization of the grid state. It offers two cost function options, that are fitting special use cases. 
To understand the usage, the OPF tutorial is recommended (see :ref:`tutorial`).
    
.. autofunction:: pandapower.rundcopp

Flexibilities, costs and constraints (except voltage constraints) are handled as in the :ref:`opf`.
Voltage constraints are not considered in the DC OPF, since voltage magnitutes are not part of the 
linearized power flow equations.

.. note::

    If you are interested in the pypower casefile that pandapower is using for power flow, you can find it in net["_ppc_opf"].
    However all necessary informations are written into the pandpower format net, so the pandapower user should not usually have to deal with pypower.

