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

The exporter writes the Equipment (EQ), SteadyStateHypothesis (SSH) and Topology (TP) profiles. The
StateVariables (SV) profile with the power-flow results is written in addition when the network has
been solved (i.e. the ``res_*`` tables are populated, e.g. via ``pandapower.runpp``).

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
   (node-breaker model) depending on their origin.
 - Transformer impedances are reconstructed on the HV winding; for three-winding transformers the
   per-winding-pair short-circuit values are inverted to recover the per-end impedances.
 - The SV profile is only written for a solved network; otherwise it is omitted.

**Supported** components for the export:

eq / ssh profile
 - ConnectivityNode, TopologicalNode, BaseVoltage, Terminal
 - ACLineSegment
 - EnergyConsumer, ConformLoad, NonConformLoad, StationSupply
 - ExternalNetworkInjection
 - SynchronousMachine, GeneratingUnit, RegulatingControl
 - Breaker, Disconnector, LoadBreakSwitch, Switch
 - LinearShuntCompensator, StaticVarCompensator
 - SeriesCompensator
 - EquivalentInjection
 - PowerTransformer, PowerTransformerEnd
 - RatioTapChanger, PhaseTapChangerLinear, PhaseTapChangerAsymmetrical, PhaseTapChangerSymmetrical

sv profile
 - SvVoltage
 - SvTapStep
 - SvShuntCompensatorSections

Components that are not yet exported include EnergySource, EquivalentBranch, NonlinearShuntCompensator,
PhaseTapChangerTabular and the diagram / geographical (DL / GL) coordinate profiles.
