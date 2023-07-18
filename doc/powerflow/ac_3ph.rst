Asymmetric /Three Phase Power Flow
==================================

pandapower uses Sequence Frame to solve three phase power flow :

.. image:: /pics/flowcharts/pandapower_unbalanced_loadflow.jpg
		:width: 30em
		:alt: alternate Text
		:align: center

.. autofunction:: pandapower.pf.runpp_3ph.runpp_3ph

.. note::

    If you are interested in the pypower casefile that pandapower is using for power flow internally, you can find it in net["_ppc0"],net["_ppc1"], net["_ppc2"].
    However all necessary informations are written into the pandpower format net, so the pandapower user should not usually have to deal with pypower.


Asymmetric Power Flow using power-grid-model
--------------------------------------------

The same function mentioned in :ref:`Balanced AC Power Flow using power-grid-model <pgmpowerflow>`, ie. `pandapower.run.runpp_pgm()` can be used for asymmetric power flow calculation.
The argument `symmetric=False` should be set for this purpose.
