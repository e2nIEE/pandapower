Asymmetric /Three Phase Power Flow
==================================

pandapower uses Sequence Frame to solve three phase power flow :

.. image:: /pics/flowcharts/pandapower_unbalanced_loadflow.jpg
		:width: 30em
		:alt: alternate Text
		:align: center

.. autofunction:: pandapower.pf.runpp_3ph.runpp_3ph

.. note::

	If you are interested in the pypower casefile that pandapower is using for power flow, you can find it in net["_ppc"].
    However all necessary informations are written into the pandpower format net, so the pandapower user should not usually have to deal with pypower.
	