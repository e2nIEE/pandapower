############################
Short-Circuit *beta!*
############################

The shortcircuit module is used to calculate short-circuits according to DIN/IEC EN 60909.

.. warning:: The pandapower short circuit module is in beta stadium, so please proceed with caution. If you have any question about the shortcircuit module or want to contribute to its development please contact leon.thurner@uni-kassel.de


The shortcircuit calculation works with the following elements:

    - bus
    - load (neglected)
    - shunt (neglected)
    - sgen
    - gen
    - ext_grid
    - line
    - trafo
    - trafo3w
    - switch

Correction factors for generator and branch elements are implemented as defined in the IEC 60909 standard. The results for all elements are tested against commercial software to ensure that correction factors are correctly applied.

The following currents can be calculated:

   - Ikss
   - Ith (does not yet work for generator)
   - ip (does not yet work for generators)

The to-do list for future implementations is:

    - Implement block transformers correction factor (generators and transformers are now always considered seperately)
    - output branch results
    - ip and ith for generators
    - implement ib and ik
    - implement unsymmetric short-circuit currents (one-phase, two-phase, two-phase-to-earth)

.. autofunction:: pandapower.shortcircuit.runsc



.. code:: python

    import pandapower.shortcircuit as sc
    import pandapower.networks as nw

    net = nw.mv_oberrhein()
    net.ext_grid["s_sc_min_mva"] = 100
    net.ext_grid["rx_min"] = 0.1

    net.line["endtemp_degree"] = 20
    sc.runsc(net, case="min")
    print(net.res_bus_sc)

