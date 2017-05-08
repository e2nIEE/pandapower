#############################
Built-in plot functions
#############################

=============================
Simple Plotting
=============================

The function simple_plotly() can be used for simple plotting. For advanced possibilities see the tutorials.

.. _simple_plotly:

.. autofunction:: pandapower.plotting.plotly.simple_plotly

Example plot with mv_oberrhein network from the pandapower.networks package:

.. image:: /pics/simple_plotly_mvoberr_sample.png
	:width: 30em
	:align: center


Example simple plot ::

    from pandapower.plotting.plotly import simple_plotly
    from pandapower.networks import mv_oberrhein
    net = mv_oberrhein()
    simple_plotly(net)


.. image:: /pics/simple_plotly_map_mvoberr_sample.png
	:width: 30em
	:align: center

Example simple plot on a map::

    net = mv_oberrhein()
    simple_plotly(net, on_map=True, projection='epsg:31467')

.. image:: /pics/simple_plotly_mapsatelite_mvoberr_sample.png
	:width: 30em
	:align: center



===============================================
Network coloring according to voltage levels
===============================================

The function vlevel_plotly() is used to plot a network colored and labeled according to voltage levels. For advanced possibilities see the tutorials.

.. _vlevel_plotly:

.. autofunction:: pandapower.plotting.plotly.vlevel_plotly

Example plot with mv_oberrhein network from the pandapower.networks package::

    from pandapower.plotting.plotly import vlevel_plotly
    from pandapower.networks import mv_oberrhein
    net = mv_oberrhein()
    vlevel_plotly(net)

.. image:: /pics/vlevel_plotly_mvoberr_sample.png
	:width: 30em
	:align: center


=============================
Power Flow results
=============================

The function pf_res_plotly() is used to plot a network according to power flow results where a colormap is used to represent line loading and voltage magnitudes. For advanced possibilities see the tutorials.

.. _pf_res_plotly:

.. autofunction:: pandapower.plotting.plotly.pf_res_plotly

Example power flow results plot::

    from pandapower.plotting.plotly import pf_res_plotly
    from pandapower.networks import mv_oberrhein
    net = mv_oberrhein()
    pf_res_plotly(net)


.. image:: /pics/pf_res_plotly_mvoberr_sample.png
	:width: 30em
	:align: center

Power flow results on a map::

    net = mv_oberrhein()
    pf_res_plotly(net, on_map=True, projection='epsg:31467', map_style='dark')

.. image:: /pics/pf_res_plotly_map_mvoberr_sample.png
	:width: 30em
	:align: center


