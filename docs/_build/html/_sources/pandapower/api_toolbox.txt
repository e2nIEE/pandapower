====================
pandapower Toolbox
====================

The pandapower toolbox is a collection of helper functions that are implemented for the pandapower framework. It is
designed for functions of common application that fit nowhere else. Have a look at the available functions to save
yourself the effort of maybe implementing something twice. If you develop some functionality which could be
interesting to other users as well and do not fit into one of the specialized packages, feel welcome to add your
contribution. To improve overview functions are loosely grouped by functionality, please adhere to this notion when
adding your own functions and feel free to open new groups as needed.

.. note::
    If you implement a function that might be useful for others, it is mandatory to add a short docstring to make browsing
    the toolbox practical. Ideally further comments if appropriate and a reference of authorship should be added as well.


Result Information
====================================

.. autofunction:: pandapower.toolbox.lf_info

.. autofunction:: pandapower.toolbox.overloaded_lines

.. autofunction:: pandapower.toolbox.violated_busses

.. autofunction:: pandapower.toolbox.equal_nets

Simulation Setup and Preparation
====================================

.. autofunction:: pandapower.toolbox.convert_format

.. autofunction:: pandapower.toolbox.add_zones_to_elements

.. autofunction:: pandapower.toolbox.create_continuous_bus_index

.. autofunction:: pandapower.toolbox.set_scaling_by_type

Topology Modification
====================================

.. autofunction:: pandapower.toolbox.close_switch_at_line_with_two_open_switches

.. autofunction:: pandapower.toolbox.drop_inactive_elements

.. autofunction:: pandapower.toolbox.drop_busses

.. autofunction:: pandapower.toolbox.drop_trafos

.. autofunction:: pandapower.toolbox.drop_lines

.. autofunction:: pandapower.toolbox.fuse_busses

.. autofunction:: pandapower.toolbox.set_element_status

.. autofunction:: pandapower.toolbox.select_subnet

Item/Element Selection
====================================

.. autofunction:: pandapower.toolbox.get_element_index

.. autofunction:: pandapower.toolbox.next_bus

.. autofunction:: pandapower.toolbox.get_connected_elements

.. autofunction:: pandapower.toolbox.get_connected_busses

.. autofunction:: pandapower.toolbox.get_connected_switches
