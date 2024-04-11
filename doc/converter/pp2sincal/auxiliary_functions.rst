######################################
Toolbox with useful Functions
######################################
    
This sections contains a list of functions that are useful to convert
pandapower networks to Sincal. The displayed functions mainly adapt the pandapower model
for an easier follow up conversion.


Scale Geo Coordinates
========================
Scales the pandapower geo-coordinates in order to display the topology in correct proportions.

.. autofunction:: pandapower.converter.sincal.pp2sincal.util.toolbox._scale_geo_data


Unique Naming
========================
Passes a unique name, containing the name of the element plus the index, to each pandapower element
in a network. This is only done, if the names in the name column are not unique.
This name will be used to create the aquivalent sincal element in the process of conversion.

.. autofunction:: pandapower.converter.sincal.pp2sincal.util.toolbox._unique_naming


Number of Elements
========================
Determines the number of elements of a pandapowerNet for each bus.

.. autofunction:: pandapower.converter.sincal.pp2sincal.util.toolbox._number_of_elements


Adapt Area Tile
========================
Adapts the area tile in Sincal to correctly display the pandapowerNet.

.. autofunction:: pandapower.converter.sincal.pp2sincal.util.toolbox._adapt_area_tile


Adapt Geo Coordinates
========================
Adjustment of bus geodata in pandapowerNet to better display transformers in sincal.

.. autofunction:: pandapower.converter.sincal.pp2sincal.util.toolbox._adapt_geo_coordinates


Initialize Voltage Level
========================
Initialize the voltage level in Sincal.

.. autofunction:: pandapower.converter.sincal.pp2sincal.util.toolbox._initialize_voltage_level


Set Calculation Parameters
==========================
Presets the calculation parameters, e.g. load flow method, voltage limits, ...

.. autofunction:: pandapower.converter.sincal.pp2sincal.util.toolbox._set_calc_params
