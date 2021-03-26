#############################
Networks
#############################
.. _ppNetworks:


Besides creating your own grids using pandapower functions, pandapower provides synthetic and
benchmark grids through the networks module.

The pandapower networks module contains example grids, simple test grids, randomly generated
grids, CIGRE test grids, IEEE case files and synthetic low voltage grids from Georg Kerber, Lindner et. al. and Dickert et. al.
If you want to evaluate your algotihms on benchmark grids with corresponding full-year load, generation, and storage profiles
or want to publish your results in a reproducible manner, we recommend the SimBench repository
(`Homepage <https://simbench.de/en/>`_, `GitHub Repository to use SimBench with pandapower <https://github.com/e2nIEE/simbench>`_).

You can find documentation for the individual network modules of pandapower here:

.. toctree::
    :maxdepth: 2

    networks/simbench
    networks/example
    networks/test
    networks/cigre
    networks/mv_oberrhein
    networks/power_system_test_cases
    networks/kerber
    networks/synthetic_voltage_control_lv_networks
    networks/dickert_lv_networks

