###########
Groups
###########

With pandapower version > 2.10.1, functionality to group network elements has been released.
Basically, these are helper functions to better handle and interact with multiple elements, even of
different element types, such as buses, generators and lines, at once.

For first steps with group functionality, as often, it is recommended to have a look to the `tutorial <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/group.ipynb>`_.

Plenty of group related functions are presented in the following.

====================================
CREATE AND DELETE GROUPS
====================================

.. autofunction:: pandapower.create_group

.. autofunction:: pandapower.create_group_from_dict

.. autofunction:: pandapower.drop_group

.. autofunction:: pandapower.drop_group_and_elements

====================================
ADAPT GROUP MEMBERS
====================================

.. autofunction:: pandapower.append_to_group

.. autofunction:: pandapower.drop_from_group

.. autofunction:: pandapower.drop_from_groups

=================================================
ACCESS GROUP DATA AND EVALUATE MEMBERSHIP
=================================================

.. autofunction:: pandapower.group_name

.. autofunction:: pandapower.group_element_index

.. autofunction:: pandapower.group_row

.. autofunction:: pandapower.isin_group

.. autofunction:: pandapower.count_group_elements

=================================================
COMPARE GROUPS
=================================================

.. autofunction:: pandapower.groups_equal

.. autofunction:: pandapower.compare_group_elements

=================================================
FIX GROUP DATA
=================================================

.. autofunction:: pandapower.remove_not_existing_group_members

.. autofunction:: pandapower.ensure_lists_in_group_element_column

.. autofunction:: pandapower.group_entries_exist_in_element_table

=================================================
MAKE USE OF GROUPS AND OTHERS
=================================================

.. autofunction:: pandapower.set_group_in_service

.. autofunction:: pandapower.set_group_out_of_service

.. autofunction:: pandapower.set_value_to_group

.. autofunction:: pandapower.group_res_p_mw

.. autofunction:: pandapower.group_res_q_mvar

.. autofunction:: pandapower.set_group_reference_column

.. autofunction:: pandapower.return_group_as_net
