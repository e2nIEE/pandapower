===================================================
CIM CGMES to pandapower
===================================================

Converts CIM CGMES 2.4.15 or 3.0 networks to pandapower.

Developed and tested on Python 3.11.

A `tutorial <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/cim2pp.ipynb>`_ as a Jupyter notebook introduces the converter with an example.

Setup
-----
Install the needed dependancies with: ::

    pip install pandapower[converter]

In order to use this converter the following import is needed. ::

    from pandapower.converter.cim import from_cim as cim2pp

For a speed increase it is advisable to install numba into the used python environment. ::

    pip install numba

Using the Converter
--------------------
In order to start the converter the following method is used. At least the location of the CGMES-files that are to be converted must be specified.

.. autofunction:: pandapower.converter.cim.cim2pp.from_cim.from_cim

The recommended way to select the CGMES-files is via the "file_list" parameter.
It accepts a folder of xml- or zip-files, a single zip-file or several zip-files as a list.
For example:

Example of a single zip file ::

    cgmes_files = r'example_cim\CGMES_v2.4.15_RealGridTestConfiguration_v2.zip'

Example of several zip files ::

    cgmes_files = [r'example_cim\CGMES_v2.4.15_SmallGridTestConfiguration_Boundary_v3.0.0.zip',
                   r'example_cim\CGMES_v2.4.15_SmallGridTestConfiguration_BaseCase_Complete_v3.0.0.zip']

Folder of xml or zip files ::

    import os
    curr_xml_dir = 'example_cim\\test'
    cgmes_files = [curr_xml_dir + os.sep + x for x in os.listdir(curr_xml_dir)]

To start the converter, the following line is used. As cgmes_version also '3.0' can be used for cgmes version 3. As a result it returns a pandapower network. ::

    net = cim2pp.from_cim(file_list=cgmes_files, cgmes_version='2.4.15')

In the resulting pandapower-network, the following should be noted:
 - Each component-table (bus, line, trafo, etc.) will get an "origin_id" column which points to the original CIM CGMES UUIDs.
 - If the CGMES model is bus-branch, the pandapower buses will be created from the TopologicalNodes.
 - If the CGMES model is node-breaker, the pandapower buses will be created from the ConnectivityNodes.
 - If the CGMES model has geo-coordinates (in the GL profile) they will be translated to bus.geo and line.geo respectively.
 - If the CGMES model has diagram coordinates (in the DL profile) they will be translated to bus.diagram and line.diagram respectively.
 - If the CGMES model includes measurements, they will be written to the pandapower measurement-table.


**Supported** components from CIM CGMES:

eq profile
 - ControlArea
 - TieFlow
 - ConnectivityNode
 - Bay
 - Substation
 - GeographicalRegion
 - SubGeographicalRegion
 - VoltageLevel
 - BaseVoltage
 - ExternalNetworkInjection
 - ACLineSegment
 - Terminal
 - OperationalLimitSet
 - OperationalLimitType
 - CurrentLimit
 - VoltageLimit
 - DCNode
 - DCEquipmentContainer
 - DCConverterUnit
 - DCLineSegment
 - CsConverter
 - VsConverter
 - DCTerminal
 - ACDCConverterDCTerminal
 - Breaker
 - Disconnector
 - Switch
 - LoadBreakSwitch
 - EnergyConsumer
 - ConformLoad
 - NonConformLoad
 - StationSupply
 - GeneratingUnit
 - WindGeneratingUnit
 - HydroGeneratingUnit
 - SolarGeneratingUnit
 - ThermalGeneratingUnit
 - NuclearGeneratingUnit
 - RegulatingControl
 - SynchronousMachine
 - AsynchronousMachine
 - EnergySource
 - EnergySchedulingType
 - StaticVarCompensator
 - PowerTransformer
 - PowerTransformerEnd
 - TapChangerControl
 - RatioTapChanger
 - PhaseTapChangerLinear
 - PhaseTapChangerAsymmetrical
 - PhaseTapChangerSymmetrical
 - PhaseTapChangerTabular
 - PhaseTapChangerTablePoint
 - RatioTapChangerTable
 - RatioTapChangerTablePoint
 - LinearShuntCompensator
 - NonlinearShuntCompensator
 - NonlinearShuntCompensatorPoint
 - EquivalentBranch
 - EquivalentInjection
 - SeriesCompensator
 - Analog
 - AnalogValue
 - MeasurementValueSource

eq_bd profile
 - ConnectivityNode
 - BaseVoltage
 - Terminal
 - EnergySource
 - EnergySchedulingType

ssh profile
 - ControlArea
 - ExternalNetworkInjection
 - Terminal
 - DCTerminal
 - ACDCConverterDCTerminal
 - CsConverter
 - VsConverter
 - Breaker
 - Disconnector
 - Switch
 - LoadBreakSwitch
 - EnergyConsumer
 - ConformLoad
 - NonConformLoad
 - StationSupply
 - RegulatingControl
 - SynchronousMachine
 - AsynchronousMachine
 - EnergySource
 - StaticVarCompensator
 - TapChangerControl
 - RatioTapChanger
 - PhaseTapChangerLinear
 - PhaseTapChangerAsymmetrical
 - PhaseTapChangerSymmetrical
 - PhaseTapChangerTabular
 - LinearShuntCompensator
 - NonlinearShuntCompensator
 - EquivalentInjection

sv profile
 - SvVoltage
 - SvPowerFlow
 - SvShuntCompensatorSections
 - SvTapStep

tp profile
 - TopologicalNode
 - DCTopologicalNode
 - ConnectivityNode
 - Terminal
 - DCTerminal
 - ACDCConverterDCTerminal

tp_bd profile
 - TopologicalNode
 - ConnectivityNode

dl profile
 - Diagram
 - DiagramObject
 - DiagramObjectPoint

gl profile
 - CoordinateSystem
 - Location
 - PositionPoint


===================================================
pandapower to CIM CGMES
===================================================

Converts a pandapower network back to CIM CGMES 2.4.15 or 3.0 RDF/XML files.

The exporter is the inverse of the importer described above. It is primarily designed for a faithful
*round-trip*: a network that was imported from CGMES carries the original CGMES identifiers (in the
``origin_id`` columns), terminals and topology references, which the exporter reuses to rebuild the
CIM model.

In order to use this converter the following import is needed. ::

    from pandapower.converter.cim import to_cim

Using the Converter
-------------------

.. autofunction:: pandapower.converter.cim.pp2cim.to_cim.to_cim

The exporter always writes the Equipment (EQ), SteadyStateHypothesis (SSH) and Topology (TP)
profiles. Three further profiles are written only when the corresponding data is present:

 - StateVariables (SV) with the power-flow results, when the network has been solved (i.e. the
   ``res_*`` tables are populated, e.g. via ``pandapower.runpp``),
 - DiagramLayout (DL) with the schematic coordinates, when the elements carry a ``diagram`` column,
 - GeographicalLocation (GL) with the geographic coordinates, when the elements carry a ``geo``
   column.

Write the network to a single zip archive (one XML file per profile) ::

    to_cim(net, file_path=r'export\my_grid.zip', cgmes_version='2.4.15')

Write one XML file per profile into a folder ::

    to_cim(net, output_folder=r'export\my_grid', cgmes_version='2.4.15')

Without ``file_path`` or ``output_folder`` no files are written and the in-memory CIM data structure
(profile -> CIM element type -> ``pandas.DataFrame``) is returned for further processing.

Notes on the export
 - Each pandapower element is mapped back to the CIM class stored in its ``origin_class`` column. New
   elements without an ``origin_id`` get a freshly generated UUID.
 - pandapower buses are written as ``TopologicalNode`` (bus-branch model) or ``ConnectivityNode``
   (node-breaker model) depending on their origin. The ``VoltageLevel`` and ``Substation`` containers
   are reconstructed so the bus voltage (and substation/zone) is restored on re-import.
 - Transformer impedances are reconstructed on the HV winding; for three-winding transformers the
   per-winding-pair short-circuit values are inverted to recover the per-end impedances.

**Supported** components for the export:

eq / ssh profile
 - ConnectivityNode, TopologicalNode, BaseVoltage, Terminal
 - Substation, VoltageLevel, SubGeographicalRegion, GeographicalRegion
 - ACLineSegment, Line
 - EnergyConsumer, ConformLoad, NonConformLoad, StationSupply
 - ExternalNetworkInjection
 - SynchronousMachine, GeneratingUnit, RegulatingControl
 - EnergySource
 - Breaker, Disconnector, LoadBreakSwitch, Switch
 - LinearShuntCompensator, NonlinearShuntCompensator, NonlinearShuntCompensatorPoint,
   StaticVarCompensator
 - SeriesCompensator, EquivalentBranch
 - EquivalentInjection
 - PowerTransformer, PowerTransformerEnd
 - RatioTapChanger, RatioTapChangerTable, RatioTapChangerTablePoint
 - PhaseTapChangerLinear, PhaseTapChangerAsymmetrical, PhaseTapChangerSymmetrical
 - PhaseTapChangerTabular, PhaseTapChangerTable, PhaseTapChangerTablePoint
 - OperationalLimitSet, OperationalLimitType, CurrentLimit

sv profile
 - SvVoltage
 - SvTapStep
 - SvShuntCompensatorSections

dl profile
 - Diagram, DiagramObject, DiagramObjectPoint

gl profile
 - CoordinateSystem, Location, PositionPoint

Limitations
-----------
The export targets round-trip fidelity (the re-imported network reproduces the original), not a
byte-identical copy of the source files, and it has not been validated against third-party CGMES
tools or formal CGMES conformance checks. Objects without a preserved ``origin_id`` receive freshly
generated UUIDs.

Approximations (the element is exported, but some detail is simplified):

 - **Two-winding transformer impedance** is placed entirely on the HV winding. This is exact when the
   source already lumps the impedance on one winding (the common ENTSO-E convention); a genuine
   non-equal HV/LV split would round-trip the total impedance but not the per-winding arrangement.
   Three-winding per-winding impedance is reconstructed exactly.
 - **Table-based tap changers** (``PhaseTapChangerTabular`` and table-based ``RatioTapChanger``) are
   rebuilt from the flattened per-step characteristic (``net['trafo_characteristic_table']``): the
   per-step ratio and angle are reconstructed, and the per-step impedance deviation is reconstructed
   for two-winding transformers. For three-winding transformers the per-step impedance deviation is
   treated as tap-independent, which can slightly shift the power-flow result at off-neutral taps.
 - **NonlinearShuntCompensator** is exported with uniform per-section points whose aggregate
   reproduces ``p_mw`` / ``q_mvar``; the original per-section values are not preserved on the net.
 - **Three-winding transformer vector group** (per-winding ``connectionKind``) is not set, because the
   winding split is ambiguous; the two-winding vector group is reconstructed.

Data read by the importer but not (yet) reproduced by the exporter:

 - **Short-circuit data** (e.g. ``SynchronousMachine`` r2/x2 and the derived rdss/xdss, and the
   ``ExternalNetworkInjection`` short-circuit fields) - only the power-flow parameters are written.
 - **Generator metadata**: the ``GeneratingUnit`` subtype (Wind / Hydro / Solar / Thermal / Nuclear),
   ``ReactiveCapabilityCurve`` / ``CurveData``, and ``EnergySchedulingType`` are not written.
 - **TapChangerControl** (tap-changer voltage regulation) and **VoltageLimit** are not written; only
   ``CurrentLimit`` operational limits are exported.
 - **Area objects**: ``ControlArea`` and ``TieFlow`` are not emitted (the ``GeographicalRegion`` /
   ``SubGeographicalRegion`` hierarchy *is* reconstructed, so substation region references resolve).
 - **Measurements** (``Analog`` / ``AnalogValue``) and the **SvPowerFlow** branch results in the SV
   profile are not written (the SV profile writes ``SvVoltage`` / ``SvTapStep`` /
   ``SvShuntCompensatorSections``).
 - **Element families not yet handled**: DC equipment (``DCLineSegment``, ``VsConverter`` /
   ``CsConverter`` and DC nodes/terminals) and ``AsynchronousMachine`` (motors) are not exported.
