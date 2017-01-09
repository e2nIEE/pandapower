Change Log
=============

unreleased
----------------------
- [FIXED] bug in create_transformer function for tp_pos parameter
- [ADDED] impedance element can now be used with unsymetric impedances zij != zji
- [ADDED] simple plotting function. Call pp.simple_plot(net) to directly plot the network
- [ADDED] measurement table for networks. Enables the definition of measurements for real-time simulations.
- [ADDED] estimation module, which provides state estimation functionality. The weighted least squares algorithm is used.
- [FIXED] bug in voltage ratio for low voltage side tap changers
- [ADDED] documentation of model validation and tests
- [ADDED] case14, case24_ieee_rts, case39, case57 network added
- [ADDED] mpc and ppc converter
- [ADDED] DC power flow function pp.rundcopp

[1.0.2] - 2016-11-30
----------------------

- [CHANGED] changed in_service dtype from f8 to bool for shunt, ward, xward
- [CHANGED] included i_from_ka and i_to_ka in net.res_line
- [ADDED] recycle parameter added. ppc, Ybus, is_elems and bus_lookup can be reused between multiple powerflows if recycle["ppc"] == True, ppc values (P,Q,V) only get updated.
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