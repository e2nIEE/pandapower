###########
Groups
###########

With pandapower version > 2.10.1, functionality to group network elements is available.
Basically, these are helper functions to better handle and interact with multiple elements, even of
different element types, such as buses, generators and lines, at once.

For first steps with group functionality, as often, it is recommended to have a look to the `tutorial <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/group.ipynb>`_.

Plenty of group related functions are presented in the following.

====================================
Create and delete groups
====================================

.. autofunction:: pandapower.create_group

.. autofunction:: pandapower.create_group_from_dict

.. autofunction:: pandapower.drop_group

.. autofunction:: pandapower.drop_group_and_elements

====================================
Adapt group members
====================================

.. autofunction:: pandapower.attach_to_group

.. autofunction:: pandapower.attach_to_groups

.. autofunction:: pandapower.detach_from_group

.. autofunction:: pandapower.detach_from_groups

=================================================
Access group data and evaluate membership
=================================================

.. autofunction:: pandapower.group_name

.. autofunction:: pandapower.group_index

.. autofunction:: pandapower.group_element_index

.. autofunction:: pandapower.group_row

.. autofunction:: pandapower.element_associated_groups

.. autofunction:: pandapower.isin_group

.. autofunction:: pandapower.count_group_elements

=================================================
Compare groups
=================================================

.. autofunction:: pandapower.groups_equal

.. autofunction:: pandapower.compare_group_elements

=================================================
Fix group data
=================================================

.. autofunction:: pandapower.check_unique_group_rows

.. autofunction:: pandapower.remove_not_existing_group_members

.. autofunction:: pandapower.ensure_lists_in_group_element_column

.. autofunction:: pandapower.group_entries_exist_in_element_table

=================================================
Further group functions
=================================================

.. autofunction:: pandapower.set_group_in_service

.. autofunction:: pandapower.set_group_out_of_service

.. autofunction:: pandapower.set_value_to_group

.. autofunction:: pandapower.group_res_p_mw

.. autofunction:: pandapower.group_res_q_mvar

.. autofunction:: pandapower.group_res_power_per_bus

.. autofunction:: pandapower.set_group_reference_column

.. autofunction:: pandapower.return_group_as_net
