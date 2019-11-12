##################################
Controller Simulation
##################################

The control module allows to simulate elements that are controlled based on power flows in power systems. Prominent examples are tap changer controllers that adapt the 
tap changer of a transformer depending on a measured bus voltage, or droop controllers in PV plants that adapt the reactive power depending on the bus voltage.

The control module allows you to simulate these control strategies by either using a predefined controller element that comes with pandapower or building your own 
controller in an object oriented framework. The controller module is closely integrated with the timeseries module, which allows you to run quasi-static timeseries
simulations with controlled elements.

.. toctree:: 
    :maxdepth: 2
    
    control/control_loop
    control/run
    control/controller
    control/tutorials