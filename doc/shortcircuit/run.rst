====================================
Running a Short-Circuit Calculation
====================================

The short circuit calculation is carried out with the calc_sc function:

.. autofunction:: pandapower.shortcircuit.calc_sc


.. code:: python

    import pandapower.shortcircuit as sc
    import pandapower.networks as nw

    net = nw.mv_oberrhein()
    net.ext_grid["s_sc_min_mva"] = 100
    net.ext_grid["rx_min"] = 0.1

    net.line["endtemp_degree"] = 20
    sc.calc_sc(net, case="min")
    print(net.res_bus_sc)
