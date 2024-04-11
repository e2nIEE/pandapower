######################################
Functions for the Conversion
######################################
In this part a list of functions is displayed with basic information
about the used parameters and returned parameters.

Pandapower Preperation
========================
The function prepares the pandapower network model before the conversion.


.. autofunction:: pandapower.converter.sincal.pp2sincal.util.main.pp_preparation

Net Preperation
========================
The function prepares the sincal database object for the conversion.


.. autofunction:: pandapower.converter.sincal.pp2sincal.util.main.net_preparation

Create Bus
========================
Converts the buses from a pandapowerNet.

.. autofunction:: pandapower.converter.sincal.pp2sincal.util.main.create_bus

Create Load
========================
Converts the loads (p,q - Node) from a pandapowerNet.


.. autofunction:: pandapower.converter.sincal.pp2sincal.util.main.create_load

Create Static Generator
========================
Converts a static generators (p,q - Node) from a pandapowerNet.

.. autofunction:: pandapower.converter.sincal.pp2sincal.util.main.create_sgen

Create External Grid
========================
Converts the external grids from a pandapowerNet.

.. autofunction:: pandapower.converter.sincal.pp2sincal.util.main.create_ext_grid

Create Trafo
========================
Converts the transformers from a pandapowerNet.

.. autofunction:: pandapower.converter.sincal.pp2sincal.util.main.create_trafo

Create Line
========================
Converts the lines from a pandapowerNet.

.. autofunction:: pandapower.converter.sincal.pp2sincal.util.main.create_line

Create Switch
========================
Converts the switches from a pandapowerNet.

.. autofunction:: pandapower.converter.sincal.pp2sincal.util.main.create_switch

Create Generator
========================
Converts the generators (p,v - node) from a pandapowerNet.

.. autofunction:: pandapower.converter.sincal.pp2sincal.util.main.create_gen

Create Storage
========================
Converts the storages from a pandapowerNet.

.. autofunction:: pandapower.converter.sincal.pp2sincal.util.main.create_storage

Create DC-Line
========================
Converts the dclines from a pandapowerNet.

.. autofunction:: pandapower.converter.sincal.pp2sincal.util.main.create_dcline
