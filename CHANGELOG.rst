Change Log
=============

[1.2.2] - 2017-03-22
--------------------
- [CHANGED] Minor refactoring in pd2ppc
- [ADDED] Technical Report

[1.2.1] - 2017-03-21
--------------------
- [FIXED] Readme for PyPi

[1.2.0] - 2017-03-21
--------------------
- [CHANGED] net.line.imax_ka to net.line.max_i_ka for consistency reasons
- [ADDED] net.line.tp_st_degree for phase shift in trafo tap changers
- [ADDED] sn_kva parameter in create_empty network for per unit system reference power
- [ADDED] parameter parallel for trafo element
- [ADDED] connectivity check for power flow to deal with disconnected network areas
- [ADDED] backward/forward sweep power flow algorithm specially suited for radial and weakly-meshed networks
- [ADDED] linear piece wise and polynomial OPF cost functions 
- [ADDED] possibility to make loads controllable in OPF
- [ADDED] to_json and from_json functions to save/load networks with a JSON format
- [ADDED] generator lookup to allow multiple generators at one bus
- [CHANGED] Initialization of calculate_voltage_angles and init for high voltage networks
- [ADDED] bad data detection for state estimation
- [CHANGED] from_ppc: no detect_trafo anymore, several gen at each node possible
- [CHANGED] validate_from_ppc: improved validation behaviour by means of duplicated gen and branch rearangement
- [ADDED] networks: case33bw, case118, case300, case1354pegase, case2869pegase, case9241pegase, GBreducednetwork, GBnetwork, iceland, cigre_network_mv with_der='all' der
- [ADDED] possibility to add fault impedance for short-circuit current calculation
- [ADDED] branch results for short circuits
- [ADDED] static generator model for short circuits
- [ADDED] three winding transformer model for short circuits
- [FIXED] correctly neglecting shunts and tap changer position for short-circuits
- [ADDED] two phase short-circuit current calculation
- [ADDED] tests for short circuit currents with validation against DIgSILENT PowerFactory


[1.1.1] - 2017-01-12
----------------------
- [ADDED] installation description and pypi files from github
- [ADDED] automatic inversion of active power limits in convert format to account for convention change in version 1.1.0
- [CHANGED] install_requires in setup.py


[1.1.0] - 2017-01-11
----------------------
- [ADDED] impedance element can now be used with unsymetric impedances zij != zji
- [ADDED] dcline element that allows modelling DC lines in PF and OPF
- [ADDED] simple plotting function: call pp.simple_plot(net) to directly plot the network
- [ADDED] measurement table for networks. Enables the definition of measurements for real-time simulations.
- [ADDED] estimation module, which provides state estimation functionality with weighted least squares algorithm
- [ADDED] shortcircuit module in beta version for short-circuit calculation according to IEC-60909
- [ADDED] documentation of model validation and tests
- [ADDED] case14, case24_ieee_rts, case39, case57 networks
- [ADDED] mpc and ppc converter
- [CHANGED] convention for active power limits of generators. Generator with max. feed in of 50kW before: p_min_kw=0, p_max_kw=-50. Now p_max_kw=0, p_min_kw=50
- [ADDED] DC power flow function pp.rundcopp
- [FIXED] bug in create_transformer function for tp_pos parameter
- [FIXED] bug in voltage ratio for low voltage side tap changers
- [FIXED] bug in rated voltage calculation for opf line constraints

[1.0.2] - 2016-11-30
----------------------

- [CHANGED] changed in_service dtype from f8 to bool for shunt, ward, xward
- [CHANGED] included i_from_ka and i_to_ka in net.res_line
- [ADDED] recycle parameter added. ppc, Ybus, _is_elements and bus_lookup can be reused between multiple powerflows if recycle["ppc"] == True, ppc values (P,Q,V) only get updated.
- [FIXED] OPF bugfixes: cost scaling, correct calculation of res_bus.p_kw for sgens
- [ADDED] loadcase added as pypower_extension since unnecessary deepcopies were removed
- [CHANGED] supress warnings parameter removed from loadflow, casting warnings are automatically supressed

[1.0.1] - 2016-11-09
----------------------

- [CHANGED] update short introduction example to include transformer
- [CHANGED] included pypower in setup.py requirements (only pypower, not numpy, scipy etc.)
- [CHANGED] mpc / ppc renamed to ppci / ppc
- [FIXED] MANIFEST.ini includes all relevant doc files and exclude report
- [FIXED] handling of tp_pos parameter in create_trafo and create_trafo3w
- [FIXED] init="result" for open bus-line switches