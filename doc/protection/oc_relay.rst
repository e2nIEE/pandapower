
###################################################
Over Current Relay
###################################################

Definite Time Overcurrent Relays :

When the current magnitude is greater than its pick-up value, over-current relays (oc relay) start to operate after a specified time delay.
Here, the relay's operating time is independent of the current magnitude over the pick-up current.
To obtain the appropriate operating time, a time delay mechanism is included.

Running a fault scenario with oc relay protection is carried out with run_fault_scenario_oc function:

.. autofunction:: pandaplan.core.protection.implementation.oc_relay_model.run_fault_scenario_oc

EXAMPLE:

.. code:: python

    import pandaplan.core.protection.implementation.oc_relay_model as oc_protection
    import pandaplan.core.protection.implementation.example_grids as nw
    net = nw.load_6bus_net_directional(open_loop=True)
    trip_decisions= oc_protection.run_fault_scenario_oc(net, sc_line_idx =5,
	 sc_location =0.5,timegrade=[0.0,0.5,0.3])
    print(trip_decisions)                                          