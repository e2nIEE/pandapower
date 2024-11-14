.. _update:


.. |br| raw:: html

   <br />

============================
Update to pandapower 3.0
============================

Breaking changes are provided with pandapower 3.0 to allow using geo_json functionality and to make namings more consistent.
New features are not explained here.

New format for geo data
==============================

Previously, geo data were stored in tables :code:`net.bus_geodata` and :code:`net.line_geodata`.
Now, these data are stored as geojson strings within the element tables, i.e. :code:`net.bus.geo` and :code:`net.line.geo`.


Renaming
==========

Parameters were renamed for more consistency at controllers :code:`net.controller`.
With version 3.0, controllers refer to net elements via :code:`element` (shorthand for :code:`type`) and :code:`element_index`.
This is established by multiple functions/codings in the pandapower package.
The columns in :code:`net.group` are also changed: To :code:`element_type` and :code:`element_index` to avoid misunderstandings.
A major rework could include the names :code:`element_type` and :code:`element_index` at any place in the package but has not been implemented (yet?) since it would break large parts of package usage.

To sum up, the following changes were made:

    - TrafoController parameters: :code:`trafotable, trafotype` ➔ :code:`element_type`; :code:`tid` ➔ :code:`element_index`
    - Group column: :code:`net.groups.element` ➔ :code:`net.groups.element_index`
