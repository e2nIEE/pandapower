===================================================
CIM CGMES to pandapower
===================================================

Converts CIM CGMES 2.4.15 networks to pandapower.

Developed and tested on Python 3.8.

A `tutorial <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/cim2pp.ipynb>`_ as a Jupyter notebook introduces the converter with an example.

Setup
-----
In order to use this converter the following import is all that ist needed. ::

    from pandapower.converter import from_cim

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

To start the converter, the following line is used. As a result it returns a pandapower network. ::

    net = from_cim.from_cim(file_list=cgmes_files)

In the resulting pandapower-network, the following should be noted:
 - Each component-table (bus, line, trafo, etc.) will get an "origin_id" column which points to the original CIM CGMES UUIDs.
 - If the CGMES model is bus-branch, the pandapower buses will be created from the TopologicalNodes.
 - If the CGMES model is node-breaker, the pandapower buses will be created from the ConnectivityNodes.
 - If the CGMES model has geo-coordinates (in the GL profile) they will be translated to bus_geodata and line_geodata respectively.
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
