=============================
Simple Plotting
=============================

Simple Network Plot
==============================

.. _simple_plot:

The function ``simple_plot()`` can be used for simple plotting. For advanced possibilities see
the `tutorial <http://nbviewer.jupyter.org/github/e2nIEE/pandapower/blob/develop/tutorials/plotting_basic.ipynb>`_.

.. autofunction:: pandapower.plotting.simple_plot


Simple Highlighting Plot
==============================

.. _simple_hl_plot:

The function ``simple_hl_plot()`` highlights lines or buses in a simple network plot. The highlighted
elements are displayed in red and enlarged. Additionally, buses and lines can be located directly
in the plot by hovering the mouse over a specific line or bus. The ``name`` and ``index`` will be shown in
a small box: ::

    net = mv_oberrhein()
    ol_lines = net.line.loc[net.line.type=="ol"].index
    ol_buses = net.bus.index[net.bus.index.isin(net.line.from_bus.loc[ol_lines]) |
                             net.bus.index.isin(net.line.to_bus.loc[ol_lines])]

    simple_hl_plot(net, hl_lines=ol_lines, hl_buses=ol_buses)


.. image:: /pics/plotting/simple_hl_plot_mv_obi.png
    :width: 80em
    :align: left


.. autofunction:: pandapower.plotting.simple_hl_plot

