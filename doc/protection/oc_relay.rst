
###################################################
Over Current Relay
###################################################

When the current magnitude exceeds its pick-up value, over-current relays (oc relay) start to operate.

As of now, oc protection module is included with Define Time Over Current Relay (DTOC), Inverse Definite Minimum Time Relay (IDMT) and the combination of DTOC and IDMT (IDTOC).

DTOC relay that operates after a definite time once the current exceeds the pick-up value, the relay's operating time is independent of the current magnitude over the pick-up current.

IDMT relay's operating time is inversely proportional to the fault current; hence higher the fault current shorter the operating time.
To obtain the appropriate operating time for all relays, a time delay mechanism is included.

The oc relay parameters are created using oc_parameters function:

.. autofunction:: pandapower.protection.oc_relay_model.oc_parameters

Running a fault scenario with oc relay protection is carried out with run_fault_scenario_oc function:

.. autofunction:: pandapower.protection.oc_relay_model.run_fault_scenario_oc

EXAMPLE- DTOC Relay:

.. code:: python

    import pandapower.protection.oc_relay_model as oc_protection
    import pandapower.protection.example_grids as nw
    net = nw.dtoc_relay_net()
    relay_settings=oc_protection.oc_parameters(net,time_settings= [0.07, 0.5, 0.3], relay_type='DTOC')
    trip_decisions,net_sc= oc_protection.run_fault_scenario_oc(net,sc_line_id=4,sc_location =0.5,relay_settings)             

EXAMPLE- IDMT Relay:

.. code:: python

    import pandapower.protection.oc_relay_model as oc_protection
    import pandapower.protection.example_grids as nw
    net = nw.idmt_relay_net()
    relay_settings=oc_protection.oc_parameters(net,time_settings= [1,0.5], relay_type='IDMT', curve_type='standard_inverse')
    trip_decisions,net_sc= oc_protection.run_fault_scenario_oc(net,sc_line_id=4,sc_location =0.5,relay_settings)         

EXAMPLE- IDTOC Relay:

.. code:: python

    import pandapower.protection.oc_relay_model as oc_protection
    import pandapower.protection.example_grids as nw
    net = nw.idtoc_relay_net()
    relay_settings=oc_protection.oc_parameters(net,time_settings= [0.07, 0.5, 0.3,1, 0.5], relay_type='IDTOC',curve_type='standard_inverse' )
    trip_decisions,net_sc= oc_protection.run_fault_scenario_oc(net,sc_line_id=4,sc_location =0.5,relay_settings)        

.. seealso::
	- *Protective Relays: Their Theory and Practice Volume One* by A. R. van. C. Warrington, 2012.   
                   