.. _timeseriesloop:

#############################
Timeseries Module Overview
#############################

When calling the function **run_timeseries** a loop is started iterates over every **time_step**. During each step,
a control loop is started for each controller by **run_control** (see :ref:`controller <controller>` for details). Within
the control loop the element variables are updated by the added controllers.

The following picture shows the time series loop of the time series module:

.. image:: /pics/timeseries/run_timeseries_loop.svg
    :width: 400 px
    :align: center