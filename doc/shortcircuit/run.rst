====================================
Running a Short-Circuit Calculation
====================================

The short circuit calculation is carried out with the calc_sc function:

.. autofunction:: pandapower.shortcircuit.calc_sc


.. code:: python

    from pandapower.shortcircuit import calc_sc
    from pandapower.networks import mv_oberrhein

    net = mv_oberrhein()
    net.ext_grid["s_sc_min_mva"] = 100
    net.ext_grid["rx_min"] = 0.1

    net.line["endtemp_degree"] = 20
    calc_sc(net, case="min")
    print(net.res_bus_sc)
