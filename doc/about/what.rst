================================
What is pandapower?
================================

The development of pandapower started as an extension of the widely used power flow solver MATPOWER and its port to python, PYPOWER. 

In PYPOWER, the electric attributes of the network are defined in a casefile in the form of a bus/branch model. The bus/branch model 
formulation is mathematically very close the power flow, which is why it is easy to generate a nodal admittance matrix or other matrices 
needed for the power flow calculation.

In terms of user friendlyness, there are however some significant drawbacks:

- there is no differentiation between lines and transformers. Furthermore, branch impedances have to be defined in per unit, which is usually not a value directly available from cable or transformer data sheets.
- the casefile only contains pure electrical data. Meta information, such as element names, line lenghts or standard types, canot be saved within the datastructure.
- since there is no API for creating the casefile, networks have to be defined by directly building the matrices. 
- the user has to ensure that all bus types (PQ, PV, Slack) are correctly assigned and bus and gen table are coherent.
- power and shunt values can only be assigned as a summed value per bus, the information about individual elements is lost in case of multiple elements at one bus.
- the datastructure is based on matrices, which means deleting one row from the datastructure changes all indices of the following elements.

All these problems make the network definition process prone to errors. pandapower aims to solve these problems by proposing a datastructure
based on pandas using PYPOWER to solve the power flow.

pandapower provides
    - flexible datastructure for comprehensive modeling of electric power systems
    - static electric models for lines, switches, generators, 2/3 winding transformers, ward equivalents etc. 
    - a convenient interface for static and quasi-static power system analysis
    
pandapower allows
    - automized the creation of complex power system models
    - solving three phase AC, DC and optimal power flow problems
    - topological searches in electric networks
    - plotting of structural and/or geographical network plans

pandapower does not yet support:
    - static short circuit calculation (currently in development)
    - unbalanced power flow problems (planned, but not currently in development)
    - RMS simulation (theoretically possible, but not currently in development)
    
pandapower does not, and most likely never will, support:
    - electromagnetic transient simulations
    - dynamic short-circuit simulations
    
If you are interested in contributing to the pandapower project, please contact leon.thurner@uni-kassel.de
