
###################################################
Directional Over Current Relay
###################################################

A directional relay is a special type of over-current relay with directional features.
The directional relay allows the operation (tripping)
of the overcurrent relay when the measured current flows in the selected direction
(forward or reverse) of current flow.

The operating quantity, the input current, and the polarization quantity determine the directional characteristic. 
The polarizing quantity is usually a voltage calculated by rotating the input voltage by the maximum torque angle.

Running a fault scenario using  doc relay  is carried out with run_fault_scenario_doc function:

.. autofunction:: pandapower.protection.implementation.doc_relay_model.run_fault_scenario_doc

EXAMPLE:

.. code:: python

    
    import pandapower.protection.implementation.doc_relay_model as doc_protection
    import pandapower.protection.implementation.example_grids as nw
    net = nw.load_4bus_net(open_loop =False)
    relay_configuration = {"switch_1": [0,"CB_dir","forward",86,45],
                            "switch_2": [1,"CB_dir","forward",86,45], 
                            "switch_3": [2,"CB_dir","forward",86,45],
                            "switch_4": [3,"CB_dir","forward",86,45],
                            "switch_5": [4,"CB_dir", "forward",86,45]}
    trip_decisions = doc_protection.run_fault_scenario_doc(net,sc_line_id =2,sc_location = 0.5,
                     relay_configuration= relay_configuration,tripping_time_auto=[0.07,0.5,0.3])                                          
                                            