.. _vsc_bipolar:

==============================================
Bipolar Voltage Source Converter (VSC Bipolar)
==============================================

.. seealso::
	:ref:`Voltage Source Converter (VSC) <vsc>`

The bipolar VSC is a Voltage Source Converter whose DC side is connected between two DC buses, the plus terminal
(bus_dc_plus) and the minus terminal (bus_dc_minus), instead of between one DC bus and ground. The DC current of the
converter flows out of the plus terminal and returns into the minus terminal. The DC voltage set-point is the
voltage difference between the plus and the minus terminal.

A bipolar HVDC station is modelled with two bipolar VSC elements:

- the positive pole between the positive DC bus and the neutral bus,
- the negative pole between the neutral bus and the negative DC bus.

The neutral buses of the stations are connected with a DC line, the dedicated metallic return (DMR). Since the return
currents of the poles enter the neutral bus, the DMR line carries the unbalance current of the poles like any other
DC line: it is zero for a balanced system and equals the difference of the pole currents otherwise (e.g. the full
pole current in monopolar operation, when one pole is out of service).

A DC system which consists only of bipolar VSC is floating, therefore exactly one bus of every DC system has to be
grounded, usually the neutral bus of one station:

.. code:: python

    create_source_dc(net, bus_dc=neutral_bus, vm_pu=0.)

Grounding the neutral buses of both stations would short-circuit the DMR line.

.. seealso::
	:ref:`Unit Systems and Conventions <conventions>`

Create Function
=====================

.. autofunction:: pandapower.create.create_vsc_bipolar

Example
=====================

Bipolar interconnect with DMR, the neutral bus B of the left station is grounded:

.. code:: python

    # DC buses: A (+), B (neutral), C (-) at the left station, D (+), E (neutral), F (-) at the right station
    create_line_dc_from_parameters(net, A, D, length_km=100, r_ohm_per_km=0.0212, max_i_ka=0.963)  # + pole
    create_line_dc_from_parameters(net, C, F, length_km=100, r_ohm_per_km=0.0212, max_i_ka=0.963)  # - pole
    create_line_dc_from_parameters(net, B, E, length_km=100, r_ohm_per_km=0.0212, max_i_ka=0.963)  # DMR

    create_vsc_bipolar(net, ac_left_p, A, B, 0.2, 10, 0.3, control_mode_dc="vm_pu", control_value_dc=1.)
    create_vsc_bipolar(net, ac_left_m, B, C, 0.2, 10, 0.3, control_mode_dc="vm_pu", control_value_dc=1.)
    create_vsc_bipolar(net, ac_right_p, D, E, 0.2, 10, 0.3, control_mode_ac="slack")
    create_vsc_bipolar(net, ac_right_m, E, F, 0.2, 10, 0.3, control_mode_ac="slack")
    create_source_dc(net, bus_dc=B, vm_pu=0.)

    runpp(net)
    net.res_line_dc  # the current of the DMR line is the unbalance current of the poles

Input Parameters
=====================

*net.vsc_bipolar*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.4\linewidth}|
.. csv-table::
   :file: vsc_bipolar_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40

\*necessary for executing a power flow calculation.

Electric Model
=================

The AC side is the same as for the :ref:`VSC <vsc>`. On the DC side the internal DC resistance r_dc_ohm connects the
plus terminal with the internal DC node of the converter, the no-load losses pl_dc_mw are modelled as a conductance
between the plus and the minus terminal. The converter itself is an ideal current source between the internal DC node
and the minus terminal; its power (V_internal - V_minus) * I is balanced with the active power on the AC side.

The parameters describe one converter (one pole), they are not split.

The DC grid is solved with current balance equations, so that neutral buses with a voltage of 0 p.u. are handled
without numerical issues. Before the Newton-Raphson iterations the DC voltages are initialized with a linear no-load
solution (voltage set-points and grounding enforced, no currents), which gives negative start values for the negative
poles and approximately 0 p.u. for the neutral buses.

Several VSC in DC voltage control mode connected to the same terminals share the DC current equally.

Result Parameters
==========================
*net.res_vsc_bipolar*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.40\linewidth}|
.. csv-table::
   :file: vsc_bipolar_res.csv
   :delim: ;
   :widths: 10, 10, 40
