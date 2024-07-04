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

Parameters were renamed for more consistency in the data, vor allem in the controller objects of :code:`net.controller` and in :code:`net.group`.
The new 'standard' to define a net element is provided by :code:`element_type` and :code:`element_index`.

For that, the following changes were made:

    - ConstControl and subclasses: :code:`trafotable, trafotype` ➔ :code:`element_type`
    - TrafoController: :code:`trafotable, trafotype` ➔ :code:`element_type`
    - :code:`net.groups.element` ➔ :code:`net.groups.element_index`
