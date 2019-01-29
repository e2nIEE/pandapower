.. _update:

============================    
Update to pandapower 2.0
============================

Update saved networks
========================

Resave existing networks ::

    import pandapower as pp
    net = pp.from_pickle("C:\\my_file.p")
    pp.runpp(net)
    pp.to_json(net, "C:\\my_file_2.0.json")
    



Update Transformer parameters
==============================

+--------------------------+---------------------+------------------------+
| pandapower 1.x           | pandapower 2.x      | elements               |
+==========================+=====================+========================+
| tp_side                  | tap_side            | trafo, trafo3w         | 
+--------------------------+---------------------+------------------------+
| tp_mid                   | tap_neutral         | trafo, trafo3w         | 
+--------------------------+---------------------+------------------------+
| tp_min                   | tap_min             | trafo, trafo3w         | 
+--------------------------+---------------------+------------------------+
| tp_max                   | tap_max             | trafo, trafo3w         |
+--------------------------+---------------------+------------------------+
| tp_pos                   | tap_pos             | trafo, trafo3w         |  
+--------------------------+---------------------+------------------------+
| tp_st_percent            | tap_step_percent    | trafo, trafo3w         | 
+--------------------------+---------------------+------------------------+
| tp_st_degree             | tap_step_degree     | trafo, trafo3w         | 
+--------------------------+---------------------+------------------------+
| tp_phase_shifter         | tap_phase_shifter   | trafo, trafo3w         | 
+--------------------------+---------------------+------------------------+
| vsc_percent              | vk_percent          | trafo                  | 
+--------------------------+---------------------+------------------------+
| vscr_percent             | vkr_percent         | trafo                  | 
+--------------------------+---------------------+------------------------+
| vsc_hv_percent           | vk_hv_percent       | trafo3w                | 
+--------------------------+---------------------+------------------------+
| vscr_hv_percent          | vkr_hv_percent      | trafo3w                | 
+--------------------------+---------------------+------------------------+
| vsc_mv_percent           | vk_mv_percent       | trafo3w                | 
+--------------------------+---------------------+------------------------+
| vscr_mv_percent          | vkr_mv_percent      | trafo3w                | 
+--------------------------+---------------------+------------------------+
| vsc_lv_percent           | vk_lv_percent       | trafo3w                | 
+--------------------------+---------------------+------------------------+
| vscr_lv_percent          | vkr_lv_percent      | trafo3w                | 
+--------------------------+---------------------+------------------------+

While a lot of parameters are affected, you will see that there are some clear patterns in the changes.
Code can therefore be udpated to be compatible with pandapower 2.x by searching and replacing as follows:

    - 'tp\_' -------> 'tap\_'
    - '_st_' ------> '_step_'
    - '_mid' -----> '_neutral'
    - 'vsc' ------> 'vk'
    
Update units from kW to MW
===========================

.. |br| raw:: html

   <br />

+-------------------------------------------------------------+-------------------------------------------------------------+
| pandapower 1.x                                              | pandapower 2.x                                              |
+=============================================================+=============================================================+
| :code:`pp.create_load(net, bus=3, p_kw=200, q_kvar=100)`    | :code:`pp.create_load(net, bus=3, p_mw=0.2, q_mvar=0.1)`    |
+-------------------------------------------------------------+-------------------------------------------------------------+
| :code:`bus_sum_kw = net.bus.p_kw.sum()`                     | :code:`bus_sum_kw = net.bus.p_mw.sum()*1e3` or |br|         |
|                                                             | :code:`bus_sum_mw = net.bus.p_mw.sum()`                     |
+-------------------------------------------------------------+-------------------------------------------------------------+
| :code:`net.shunt.p_kw *= 2`                                 | :code:`net.shunt.p_mw*=2`                                   |
+-------------------------------------------------------------+-------------------------------------------------------------+
| :code:`p_from_kw = net.res_line.p_from_kw`                  | :code:`p_from_kw = net.res_line.p_from_mw*1e3` or |br|      |
|                                                             | :code:`p_from_mw = net.res_line.p_from_mw`                  |
+-------------------------------------------------------------+-------------------------------------------------------------+

Update Generation System
===========================

+-------------------------------------------------------------+-------------------------------------------------------------+
| pandapower 1.x                                              | pandapower 2.x                                              |
+=============================================================+=============================================================+
| :code:`pp.create_sgen(net, bus=3, p_kw=-500, q_kvar=100)`   | :code:`pp.create_sgen(net, bus=3, p_mw=0.5, q_mvar=-0.1)`   |
+-------------------------------------------------------------+-------------------------------------------------------------+
| :code:`gen_power_kw = -net.res_gen.p_kw.sum()`              | :code:`gen_power_kw = net.res_gen.p_mw.sum()*1e3` or |br|   |
|                                                             | :code:`gen_power_mw = net.res_gen.p_from_mw.sum()`          |
+-------------------------------------------------------------+-------------------------------------------------------------+


Update Constraints
===========================

+-------------------------------------------------------------+-------------------------------------------------------------+
| pandapower 1.x                                              | pandapower 2.x                                              |
+=============================================================+=============================================================+
| :code:`min_p_kw=-2000` |br|                                 | :code:`min_p_mw=0` |br|                                     |
| :code:`max_p_kw=0`                                          | :code:`max_p_mw=2`                                          |
+-------------------------------------------------------------+-------------------------------------------------------------+
| :code:`min_q_kvar=-300` |br|                                | :code:`min_q_mvar=-0.4` |br|                                |
| :code:`max_q_kvar=400`                                      | :code:`max_q_mvar=0.3`                                      |
+-------------------------------------------------------------+-------------------------------------------------------------+

Update Cost Functions
===========================

+---------------------------------------------------------------------+-------------------------------------------------------------+
| pandapower 1.x                                                      | pandapower 2.x                                              |
+=====================================================================+=============================================================+
| :code:`pp.create_polynomial_costs(net, "gen", 3, [0, 0, -100])`     | :code:`pp.create_poly_costs(net, "gen", 3, c_per_mw=0.1)`   |
+---------------------------------------------------------------------+-------------------------------------------------------------+


Update Measurements
===========================

There have been changes in the measurement table of pandapower grids.
*element* is set to the pandapower index of the measured element, *bus* is not a column amymore.
The new column *side* defines the side of the element at which the measurement is placed.
It can be "from" / "to" for lines, "hv" / "mv" / "lv" for trafo/trafo3w elements and is None for bus measurements.
Explicitly setting a bus index for *side* is still possible.
*type* is renamed to *measurement_type* for additional clarity.
Power measurements are set in MW or MVar now, consistent with the other pandapower tables.
