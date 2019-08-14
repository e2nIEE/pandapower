###################################################
Code Snippets and Tutorials
###################################################

.. _control_tutorials:

The general workflow when approaching simulation of controlled elements within pandapower
could be outlined like this:

    1. Get an overview of your project. Which network elements do you need, what and how should these elements accomplish their task?

    2. Checkout current implementations. Maybe there is a controller that does something similar, so you can save work and also get a point to start with.

    3. Implement your own controller/controlling-strategy.

    4. Setup your simulation and give it a go.

**In case of questions or problems come up, feel free to contact** friederike.meier@iee.fraunhofer.de **or**
jan.wiemer@iee.fraunhofer.de **(for students).**

Single Load-Flow with one Controller
====================================
For introduction purposes an easy example will be described.
The task at hand would be to simulate a Trafo Controller with local continous tap changer voltage control.
First we load a network and define it as ``net`` (if you dont know how, have a look at
:ref:`Pandapower Pro Networks <hpNetworks>`). Next we need one object: an instance of
of a ``ContinuousTapControl``, for example StatCurtPv. We want the transformer with ID 114 to be controlled by this controller, hence we pass ``tid=114``.

.. note::
    Have a look at all transformer IDs by typing ``net.trafo.index`` and chose the ones to be controlled.
	
::

    import pandapower as pp
    import control
    from pandapower.networks import mv_oberrhein
    
    # loading the network with the usecase 'generation'
    net = mv_oberrhein()
    
    # initialising controller
    tol = 1e-6
    trafo_controller = control.ContinuousTapControl(net=net, tid=114, u_set=0.98, tol=tol)
    
    # running a control-loop
    control.run_control(net)

We imported **pandapower** and the **control** module and created the object of a controller we need. You can look up which
parameters are mandatory and which are optional in the constructor of the class you are creating
an instance of. In our example we need to pass a reference to the net, the ID of the controlled
transformer, the voltage setpoint and a calculation tolerance.

.. note::
    I wrote ``import pandapower as pp`` which provides me a handy
    abbreviation ``pp`` for the whole import-reference. These abbreviations have to
    be unique throughout your code.

Now we look at our network that contains our controller. ::

    net.controller
    
The output in the console shows, that the controller is active and has the default values for order and level (we'll look at 
these in more detail shortly). 
Now we run a loadflow-simulation with our controlling unit using the ``control.run_control(net)`` method. 
Have a look at ``net.res_trafo`` to check the results of the transformers. You can compare them with results of a normal loadflow-simulation
by running ``pp.runpp(net)`` and checking ``net.res_trafo`` again. Check the results at the buses and lines in the network aswell for further informations.

Simulating time-series with Controllers
=======================================
If you want to simulate time-series, you may also do so using the controller framework. First you need
a **DataSource** for the profiles loads or pv-plants should be using. Most commonly CSV-files are being used
to provide data values over time, but you could also implement a **DataSource** of your own which e.g.
generates data on-the-fly. In our example we use an instance of **CsvData**. It expects a column named
``time`` containing consecutive timestamps. You may simply use values from zero counting upwards for each
time step or use UNIX-timestamps if you like. Each column contains a profile with a value
for each time step at the corresponding row. You have to pass a datasource as well as the name of the column
a controller should use as profile as depicted below.

::

    import pandapower as pp
    import control
    import timeseries
    
    # loading the network with the usecase 'generation'
    net = pp.networks.mv_oberrhein(scenario='generation')
    
    # loading a timeseries
    ds = timeseries.CsvData("PATH\\FILE.csv", sep=";")
    
    # initialising ConstControl to update values at the regenerative generators
    const = control.ConstControl(net, element='sgen', element_index=net.sgen.index, 
					variable='p_mw',  data_source=ds, profile_name='P_PV_1', level=0)

    # initialising controller
    tol = 1e-6
    trafo_controller = control.ContinuousTapControl(net=net, tid=114, u_set=0.98, tol=tol, level=0)
    
    # starting the timeseries simulation for one day -> 96 15 min values.
    timeseries.run_timeseries(net, time_steps=(0,95))


We created a **DataSource** and passed it to the ``ConstControl``, while also providing the name of the
P-profile. For simplification purposes we used one profile for all generators.
We may want to save certain values at each calculated timestep. In order to do that,
we build an **OutputWriter**.

::

    # initialising the outputwriter to save data
    ow = timeseries.OutputWriter(net)
    ow.log_variable('res_sgen', 'p_kw')
    ow.log_variable('res_bus', 'vm_pu')
    
    # starting the timeseries simulation for one day -> 96 15 min values.
    timeseries.run_timeseries(net, time_steps=(0,95))
    
    # results in ow.output

We created an **OutputWriter** and added a few functions to store values we are intersted in. Have a
look at the implementation of the **OutputWriter** to find out more about saving values during time-series
simulation. Note that the invokation of the simulation differs
from above: we use ``timeseries.run_timeseries()`` and pass on the start- and stop step of the simulation. Results of
the simulation are being stored in a pandas dataframe called ``output`` in the ``OutputWriter``.

Jupyter Notebook Tutorials
==========================
There are a few interactive tutorials to internalize this section:


- `Running a TimeSeries-Simulation <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/time_series.ipynb>`_
- `Running an advanced TS-Simulation <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/time_series_advanced_output.ipynb>`_

.. _ownController:

