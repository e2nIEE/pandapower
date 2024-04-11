######################################
Main Function
######################################
    
The main function represents the three major steps needed for a complete conversion of a pandapowerNet to Sincal.
All steps (initialization, conversion, finalization) will be executed automatically after each other. If you want to 
to make calculations or changes between these steps call the methods seperately.


Main Function
========================
Complete conversion of a pandapowerNet to sincal.

.. autofunction:: pandapower.converter.sincal.pp2sincal.pp2sincal.pp2sincal

Convert a simbench Network
===========================
Converts a certain simbench to pandapowerNet using the `pp2sincal` - function. The simbench grids can be selected
using a `simbench_code`. To get further informations checkout out the interactive tutorial https://github.com/e2nIEE/simbench/blob/master/tutorials/simbench_grids_basics_and_usage.ipynb.


.. autofunction:: pandapower.converter.sincal.pp2sincal.pp2sincal.convert_simbench_network

Convert all simbench Networks
==============================
Converts all simbench networks at once.

.. autofunction:: pandapower.converter.sincal.pp2sincal.pp2sincal.convert_all_simbench_networks

Convert simbench networks from a scenario
==========================================


.. autofunction:: pandapower.converter.sincal.pp2sincal.pp2sincal.convert_simbench_networks_scenario

Convert  simbench networks with a level
========================================


.. autofunction:: pandapower.converter.sincal.pp2sincal.pp2sincal.convert_simbench_networks_level
