###########
Toolbox
###########

The pandapower toolbox is a collection of helper functions that are implemented for the pandapower framework. It is
designed for functions of common application that fit nowhere else. Have a look at the available functions to save
yourself the effort of maybe implementing something twice. If you develop some functionality which could be
interesting to other users as well and do not fit into one of the specialized packages, feel welcome to add your
contribution. To improve overview functions are loosely grouped by functionality, please adhere to this notion when
adding your own functions and feel free to open new groups as needed.

====================================
Comparison
====================================

.. autofunction:: pandapower.toolbox.dataframes_equal

.. autofunction:: pandapower.toolbox.compare_arrays

.. autofunction:: pandapower.toolbox.nets_equal

.. autofunction:: pandapower.toolbox.nets_equal_keys

====================================
Power Factor
====================================

.. autofunction:: pandapower.toolbox.signing_system_value

.. autofunction:: pandapower.toolbox.pq_from_cosphi

.. autofunction:: pandapower.toolbox.cosphi_from_pq

====================================
Result Information
====================================

.. autofunction:: pandapower.toolbox.lf_info

.. autofunction:: pandapower.toolbox.opf_task

.. autofunction:: pandapower.toolbox.switch_info

.. autofunction:: pandapower.toolbox.overloaded_lines

.. autofunction:: pandapower.toolbox.violated_buses

.. autofunction:: pandapower.toolbox.clear_result_tables

.. autofunction:: pandapower.toolbox.res_power_columns

====================================
Item/Element Selection
====================================

.. autofunction:: pandapower.toolbox.get_element_index

.. autofunction:: pandapower.toolbox.get_element_indices

.. autofunction:: pandapower.toolbox.next_bus

.. autofunction:: pandapower.toolbox.get_connected_elements

.. autofunction:: pandapower.toolbox.get_connected_elements_dict

.. autofunction:: pandapower.toolbox.get_connected_buses

.. autofunction:: pandapower.toolbox.get_connected_buses_at_element

.. autofunction:: pandapower.toolbox.get_connected_switches

.. autofunction:: pandapower.toolbox.get_connecting_branches

.. autofunction:: pandapower.toolbox.false_elm_links

.. autofunction:: pandapower.toolbox.false_elm_links_loop

.. autofunction:: pandapower.toolbox.element_bus_tuples

.. autofunction:: pandapower.toolbox.pp_elements

.. autofunction:: pandapower.toolbox.branch_element_bus_dict

.. autofunction:: pandapower.toolbox.count_elements

.. autofunction:: pandapower.toolbox.get_gc_objects_dict

====================================
Data Modification
====================================

.. autofunction:: pandapower.toolbox.add_column_from_node_to_elements

.. autofunction:: pandapower.toolbox.add_column_from_element_to_elements

.. autofunction:: pandapower.toolbox.add_zones_to_elements

.. autofunction:: pandapower.toolbox.reindex_buses

.. autofunction:: pandapower.toolbox.create_continuous_bus_index

.. autofunction:: pandapower.toolbox.reindex_elements

.. autofunction:: pandapower.toolbox.create_continuous_elements_index

.. autofunction:: pandapower.toolbox.set_scaling_by_type

.. autofunction:: pandapower.toolbox.set_data_type_of_columns_to_default

.. autofunction:: pandapower.toolbox.get_inner_branches

====================================
Electric Grid Modification
====================================

.. autofunction:: pandapower.toolbox.select_subnet

.. autofunction:: pandapower.toolbox.merge_nets

.. autofunction:: pandapower.toolbox.set_element_status

.. autofunction:: pandapower.toolbox.set_isolated_areas_out_of_service

.. autofunction:: pandapower.toolbox.repl_to_line

.. autofunction:: pandapower.toolbox.merge_parallel_line

.. autofunction:: pandapower.toolbox.merge_same_bus_generation_plants

.. autofunction:: pandapower.toolbox.close_switch_at_line_with_two_open_switches

.. autofunction:: pandapower.toolbox.fuse_buses

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dropping Elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pandapower.toolbox.drop_elements

.. autofunction:: pandapower.toolbox.drop_elements_simple

.. autofunction:: pandapower.toolbox.drop_buses

.. autofunction:: pandapower.toolbox.drop_trafos

.. autofunction:: pandapower.toolbox.drop_lines

.. autofunction:: pandapower.toolbox.drop_elements_at_buses

.. autofunction:: pandapower.toolbox.drop_switches_at_buses

.. autofunction:: pandapower.toolbox.drop_measurements_at_elements

.. autofunction:: pandapower.toolbox.drop_controllers_at_elements

.. autofunction:: pandapower.toolbox.drop_controllers_at_buses

.. autofunction:: pandapower.toolbox.drop_duplicated_measurements

.. autofunction:: pandapower.toolbox.drop_inner_branches

.. autofunction:: pandapower.toolbox.drop_out_of_service_elements

.. autofunction:: pandapower.toolbox.drop_inactive_elements

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replacing Elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pandapower.toolbox.create_replacement_switch_for_branch

.. autofunction:: pandapower.toolbox.replace_zero_branches_with_switches

.. autofunction:: pandapower.toolbox.replace_impedance_by_line

.. autofunction:: pandapower.toolbox.replace_line_by_impedance

.. autofunction:: pandapower.toolbox.replace_ext_grid_by_gen

.. autofunction:: pandapower.toolbox.replace_gen_by_ext_grid

.. autofunction:: pandapower.toolbox.replace_gen_by_sgen

.. autofunction:: pandapower.toolbox.replace_sgen_by_gen

.. autofunction:: pandapower.toolbox.replace_pq_elmtype

.. autofunction:: pandapower.toolbox.replace_ward_by_internal_elements

.. autofunction:: pandapower.toolbox.replace_xward_by_internal_elements
