.. _switch_model:

=============
Switch
=============

Create Function
=====================

.. autofunction:: pandapower.create.create_switch
.. autofunction:: pandapower.create.create_switches

Input Parameters
=====================

*net.switch*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.40\linewidth}|
.. csv-table:: 
   :file: switch_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40

\*necessary for executing a power flow calculation.

   
Electric Model
=================

*Bus-Bus-Switches:*

Two buses that are connected with a closed bus-bus switches are fused internally for the power flow, open bus-bus switches are ignored:

.. image:: switches_bus.png
	:width: 18em
	:alt: alternate Text
	:align: center

This has the following advantages compared to modelling the switch as a small impedance:

    - there is no voltage drop over the switch (ideal switch)
    - no convergence problems due to small impedances / large admittances
    - less buses in the admittance matrix

.. note::

   Because fused buses share one internal node, ``res_switch`` contains NaN for
   bus-bus switches with ``z_ohm=0`` after ``runpp()``.  To compute the power
   flow through these switches via nodal balance, call
   :func:`pandapower.toolbox.compute_switch_flows` after the load flow.
    
*Bus-Element-Switches:*

When the power flow is calculated internally for every open bus-element switch an auxiliary bus is created in the pypower case file. The pypower branch that corresponds to the element is then connected to this bus. This has the following advantages compared to modelling the switch by setting the element out of service:

    - loading current is considered
    - information about switch position is preserved
    - difference between open switch and out of service line (e.g. faulty line) can be modelled

Closed bus-element switches are ignored:

.. image:: switches_element.png
	:width: 30em
	:alt: alternate Text
	:align: center


Result Parameters
==========================
*net.res_switch*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.40\linewidth}|
.. csv-table::
   :file: switch_res.csv
   :delim: ;
   :widths: 10, 10, 40