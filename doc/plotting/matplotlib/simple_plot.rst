=============================
Simple Plotting
=============================

Simple Network Plot
==============================

The function simple_plot() can be used for simple plotting. For advanced possibilities see the
`tutorial <https://nbviewer.org/github/e2nIEE/pandapower/blob/develop/tutorials/plotting_basic.ipynb>`__.

.. _simple_plot:

.. autofunction:: pandapower.plotting.simple_plot

When drawing Static Generators by type the used Patches are:


A helper function for angle calculation is provided.
It will use all elements in a network to calculate angles for each patch based on the amount of elements at each bus.

.. autofunction:: pandapower.plotting.calculate_unique_angles

Simple Highlighting Plot
==============================

The function ``simple_plot()`` can also highlight lines or buses in a simple network plot. The highlighted
elements are displayed in red and enlarged. Additionally, buses and lines can be located directly
in the plot by hovering the mouse over a specific line or bus. The ``name`` and ``index`` will be shown in
a small box: ::

    net = mv_oberrhein()
    ol_lines = net.line.loc[net.line.type=="ol"].index
    ol_buses = net.bus.index[net.bus.index.isin(net.line.from_bus.loc[ol_lines]) |
                             net.bus.index.isin(net.line.to_bus.loc[ol_lines])]

    simple_plot(net, hl_lines=ol_lines, hl_buses=ol_buses, enable_hovering=True)


.. image:: /pics/plotting/simple_hl_plot_mv_obi.png
    :width: 80em
    :align: left
