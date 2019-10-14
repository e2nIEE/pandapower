##################################
Control Loop Simulation
##################################

Within the directory **control** you will find the ``TrafoController``, which simulates a tap-changer-transformer and
the ``ConstControl``, which updates timeseries for a specific element or multiple elements of the same type. 
For example the active power *P* of static generators. The parent class is called ``Controller``. 
The control module is closely interlinked with the **timeseries** module, which  includes the simulation of time 
base operations. The implemented classes ``DataSource`` and ``OutputWriter`` enable the possibilities to run timeseries
and save output data.

.. note::
	These abstractions are implemented object-orientated. If you are not familiar
	with this concept we recommend a quick look at:

	* http://www.python-course.eu/object_oriented_programming.php (english)

	* http://www.python-kurs.eu/klassen.php (german)

**Structure of this chapter:**

.. toctree:: 
    :maxdepth: 2
    
    control/control_loop
    control/building_a_controller
    control/controller
    control/cascade_control
    control/timeseries
    control/tutorials