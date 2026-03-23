=============================
Simple Plotting
=============================

The function simple_plot() can be used for simple plotting.
For advanced possibilities see the `tutorial <http://nbviewer.jupyter.org/github/e2nIEE/pandapower/blob/develop/tutorials/plotting_basic.ipynb>`_.

.. _simple_plot:

.. autofunction:: pandapower.plotting.simple_plot

When drawing Static Generators by type the used Patches are:


A helper function for angle calculation is provided.
It will use all elements in a network to calculate angles for each patch based on the amount of elements at each bus.

.. autofunction:: pandapower.plotting.calculate_unique_angles
