====================
Create & Draw Traces
====================


Plotly traces can be created from pandapower networks with the following functions.
The traces can be passed as :code:`additional_traces` (list) to the :func:`simple_plotly` function.

++++++++++
Bus Traces
++++++++++

.. autofunction:: pandapower.plotting.plotly.traces.create_bus_trace


+++++++++++++
Branch Traces
+++++++++++++

.. autofunction:: pandapower.plotting.plotly.traces.create_line_trace

.. autofunction:: pandapower.plotting.plotly.traces.create_trafo_trace


+++++++++++
Draw Traces
+++++++++++

.. autofunction:: pandapower.plotting.plotly.traces.draw_traces


++++++++++++++++++++++++++
Markers with weighted size
++++++++++++++++++++++++++

The function :func:`create_weighted_marker_trace()` can be used to create additional traces with markers
(patches) for one column in one component table.

.. autofunction:: pandapower.plotting.plotly.traces.create_weighted_marker_trace

Example with load and sgen bubbles::

    net = mv_oberrhein()
    net.load.scaling, net.sgen.scaling = 1, 1
    markers_load = create_weighted_marker_trace(net, elm_type="load", color="red",
                                                marker_scaling=100)
    markers_sgen = create_weighted_marker_trace(net, elm_type="sgen", color="green",
                                                marker_scaling=100)
    simple_plotly(net, bus_size=1, additional_traces=[markers_load, markers_sgen])