
###################################################
Over Current Relay
###################################################

When the current magnitude exceeds its pick-up value, over-current relays (oc relay) start to operate.

As of now, oc protection module is included with Define Time Over Current Relay (DTOC), Inverse Definite Minimum Time Relay (IDMT) and the combination of DTOC and IDMT (IDTOC).

DTOC relay that operates after a definite time once the current exceeds the pick-up value, the relay's operating time is independent of the current magnitude over the pick-up current.

IDMT relay's operating time is inversely proportional to the fault current; hence higher the fault current shorter the operating time.
To obtain the appropriate operating time for all relays, a time delay mechanism is included.

OC Relays can be created using the OCRelay class


..autoclass:: pandapower.protection.protection_devices.oc_relay.OCRelay
    :members:
    :class-doc-from: class

To run protection calculations, use the calculate_protection_times function:

.. autofunction:: pandapower.protection.run_protection.calculate_protection_times

Kindly follow the tutorial of the Over Current Relay (OC relay) for details:
https://github.com/e2nIEE/pandapower/blob/develop/tutorials/protection/oc_relay.ipynb

Warning! Under pandapower.protection.oc_relay_model.py, oc_relay_settings(), oc_get_trip_decision(), and
run_fault_scenario_oc() are now legacy. In order to maintain compatibility with other protection devices,
please use the new OC Relay class.

.. seealso::
	- *Protective Relays: Their Theory and Practice Volume One* by A. R. van. C. Warrington, 2012.   
                   
