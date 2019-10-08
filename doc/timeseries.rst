#############################
Timeseries
#############################
This section covers the timeseries module which includes the simulation of time based operations and is closely 
connected with the control module. The included classes ``DataSource`` and ``OutputWriter`` enable the possibilities to run timeseries 
and save output data. For questions and suggestions contact Florian Schäfer at florian.schaefer@uni-kassel.de.

- `Running a TimeSeries-Simulation <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/time_series.ipynb>`_
- `Running an advanced TS-Simulation <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/time_series_advanced_output.ipynb>`_


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

****************
Datasource
****************

****************
Outputwriter
****************