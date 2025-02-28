#############################
Data Sources
#############################

DataSource Base Class
=====================
A ``DataSource`` object must be given to :ref:`const controllers <ConstControl>` in order to run time series simulations.
The data source contains the values to be set in each time step by the controller. It has the function
:code:`get_time_step_value(time_step)` which reads these values.

.. autoclass:: pandapower.timeseries.data_source.DataSource
    :members:

DataFrame Data Source
=====================
A ``DFData`` object is inherited from ``DataSource`` and contains a DataFrame which stores the time series values.

.. autoclass:: pandapower.timeseries.data_sources.frame_data.DFData
    :members: