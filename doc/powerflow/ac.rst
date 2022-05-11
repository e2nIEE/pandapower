Balanced AC Power Flow
=======================

pandapower uses PYPOWER to solve the power flow problem:

.. image:: /pics/flowcharts/pandapower_power flow.png
		:width: 40em
		:alt: alternate Text
		:align: center

.. autofunction:: pandapower.runpp

.. note::

    If you are interested in the pypower casefile that pandapower is using for power flow, you can find it in net["_ppc"].
    However all necessary informations are written into the pandpower format net, so the pandapower user should not usually have to deal with pypower.

If available, the librabry lightsim2grid is used as a backend for power flow simulation instead of the
implementation in pandapower, leading to a boost in performance. The library lightsim2grid is implemented in C++ and
can either be installed with pip install lightsim2grid, or built from source. More about the library and the
installation guide can be found in the `documentation <https://lightsim2grid.readthedocs.io/en/latest/>`_ or
its GitHub `repository <https://github.com/BDonnot/lightsim2grid>`_.

