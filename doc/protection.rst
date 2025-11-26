##################################
Protection
##################################

Pandapower provides the user with an option to specify a protection scheme (line protection) by 
including protection devices into the network model. 
The developed module can model and analyse fault scenarios both analytically and graphically. 
It also can assist with the coordination of different protective devices.

Based on a relay’s main functionalities, the implemented relay model has three functions:

1. Parametrization (calc_parameters)
2. Measurement (get_measurement_at_relay_location)
3. Tripping decisions (get_trip_decision)

In Parametrization, the characteristic parameters of the relay model are calculated.
These include the tripping thresholds/limits and the tripping times. The parameter 
set is specific for each relay type. In the Measurement phase, a short-circuit event 
at a line is calculated. This is realized by adding a bus at the selected fault location 
and calculating a short circuit at this bus. The fault location is selected by entering the 
fault line index and the location on the line. The Current Transformer (CT) and Voltage Transformer
(VT) functionalities and the relay calculations based on those measurements. 
Tripping decisions are executed by comparing the measurement thresholds with the relay parameters.

The following figure summarizes the functionality of the protection module and how the functions are connected.

.. image:: /pics/protection/protection_flow_chart.png
    :width: 400 px
    :align: center


The functions Parametrization, Measurement, and Tripping decisions are called inside the 
function run_fault_scenario. Additionally, two more files are provided: the utility_functions 
and example_grids. Inside the example_grids function, grids used for the model 
validations are provided. The utility_functions provides standard functions 
for all relay models (e.g., plotting functions, search functions).

As of now pandapower is capable of simulating fault scenario using:

.. toctree:: 
    :maxdepth: 2
    
    protection/oc_relay
    protection/fuse

.. seealso::
	- *Netzschutztechnik-Anlagentechnik für elektrische Verteilungsnetze* by Walter Schossig, Thomas Schossig, 2018.

   
