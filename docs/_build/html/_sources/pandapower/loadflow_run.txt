Run a Powerflow
=====================
.. _ppLoadflow:

A power flow in pandapower is executed by the runpp function. Internally the pandapower power flow is executed with pypower (see :ref:`about pandapower <about>`).


.. note::

    If you are interested in the pypower casefile that pandapower is using for power flow, you can find it in net["_mpc_last_cycle"].
    However all necessary informations are written into the pandpower format net, so the pandapower user should not usually have to deal with pypower.


.. autofunction:: pandapower.runpp

.. warning::
    Neglecting voltage angles is only valid in radial networks! pandapower was developed for distribution networks, which is why omitting the voltage angles is the default. 
    However be aware that voltage angle differences in networks with multiple galvanically coupled external grids lead to balancing power flows between slack nodes.  
    That is why voltage angles always have to be considered in meshed network, such as in the sub-transmission level!
