#############################
Networks
#############################
.. _ppNetworks:


Besides creating your own grids through the pandapower API, pandapower provides synthetic and
Benchmark networks through the networks module.

The pandapower networks modul contains example networks, simple test networks, randomly generated
networks, CIGRE test networks, IEEE case files and synthetic low voltage networks from Georg Kerber
and Lindner et. al. and Dickert et. al.. For the need of a benchmark grid dataset, we recommend the
new `SimBench <https://simbench.de/en/>`_ grid data.

If you like to test your algotihms with benchmark grids and corresponding full-year load, generation
and storage profiles or want to publish your results, we recommend you to have look to SimBench
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

