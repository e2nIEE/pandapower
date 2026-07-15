==================
OpenDSS
==================

.. _converter_opendss:

The following function imports an `OpenDSS <https://www.epri.com/pages/sa/opendss>`_
feeder into a **balanced (positive-sequence)** pandapower net. It reads the circuit
through ``OpenDSSDirect.py`` (an optional dependency, ``pip install pandapower[opendss]``),
mapping buses, lines (with their LineCode carried through as ``std_type``),
series (bus-to-bus) reactors (the pattern some feeder libraries use to model the
substation's source impedance), two-winding transformers (with their tap changer
imported as an explicit ``tap_pos`` rather than baked into ``vn_hv_kv``/``vn_lv_kv``),
center-tapped split-phase service transformers (collapsed to a two-winding
equivalent), loads, shunt capacitors, switches and the source. With
``import_controllers=True``, each ``RegControl`` is additionally imported as a
``DiscreteTapControl``, so the tap responds to bus voltage during
``pandapower.control.run_control`` instead of staying pinned at the OpenDSS
operating point; line-drop compensation, reverse-mode regulation and time delays
are not modeled and are reported as warnings when encountered.

Positive-sequence is exact for symmetric (e.g. European 3-phase 4-wire) feeders and
a documented approximation for unsymmetrical North-American topology (single-phase
laterals, split-phase), which pandapower's ``runpp_3ph`` cannot represent (see issue
#873). The returned net carries an ``opendss_import`` report with element counts,
per-bus phase counts, the OpenDSS-solved voltages and the approximations made.

.. autofunction:: pandapower.converter.opendss.from_opendss
