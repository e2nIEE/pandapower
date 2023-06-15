DC Power flow
=====================
.. _ppDCPF:

.. warning::
    To run an AC power flow with DC power flow initialization, use the AC power flow with init="dc".

pandapower uses PYPOWER to solve the DC power flow problem:

.. image:: /pics/flowcharts/pandapower_dc_powerflow.png
		:width: 40em
		:alt: alternate Text
		:align: center
    
.. autofunction:: pandapower.run.rundcpp

.. note::

    If you are interested in the pypower casefile that pandapower is using for power flow, you can find it in net["_ppc"].
    However all necessary informations are written into the pandpower format net, so the pandapower user should not usually have to deal with pypower.
