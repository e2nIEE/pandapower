.. _license:

=========
License
=========

.. highlight:: none


pandapower is published under the following 3-clause BSD license:

    .. literalinclude:: ../../LICENSE

Third-party optional dependencies
---------------------------------

Some optional features depend on external packages that are distributed under their own licenses. They
are installed explicitly by the user and are never vendored into pandapower, so the BSD license above
only covers the pandapower code itself. Notable examples:

* `transformer-thermal-model <https://github.com/alliander-opensource/transformer-thermal-model>`__
  (Mozilla Public License 2.0), used by the transformer temperatures of the temperature dependent power
  flow (``pip install pandapower[tdpf]``).

