###################################################
Running a Simulation
###################################################

To run a simulation, simply initialize one or multiple controllers (see ref to controllers). An overview of all controllers registered in the net is given in the controller table:

.. code::

	print(net.controller)
	
Then, run a power flow with the run_control option set to true:

.. code::

	import pandapower as pp
	pp.runpp(net, run_control=True)
	
The runpp function will now run multiple power flow calculations until all registered controllers are converged.

Instead of calling runpp it is also possible to call run_control from the control module directly:


.. code::

	import pandapower.control as control
	control.run_control(net)


By default, this will do the same as runpp with run_control=True. Calling the run_control function however gives you more flexibility to configurate
the controller loop simulation.

.. autofunction:: pandapower.control.run_control

