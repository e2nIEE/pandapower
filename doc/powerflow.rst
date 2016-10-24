##########################
Power Flow
##########################

.. _ppLoadflow:

The power flow is the most import static network calculation operation. pandapower uses PYPOWER to solve the loadflow problem:

.. image:: /pics/pandapower_loadflow.png
		:width: 40em
		:alt: alternate Text
		:align: center

This section shows you how to run different power flows (AC, DC, OPF), what known problems and caveats there are and how you 
can identify problems using the pandapower diagnostic function.
        
.. toctree:: 
    :maxdepth: 1

    powerflow/run
    powerflow/caveats
    powerflow/diagnostic