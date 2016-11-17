Change Log
=============

unreleased
--------------
- [ADDED] loadcase added as pypower_extension since unnecessary deepcopies were removed


[1.0.2] - 2016-11-15
----------------------

- [CHANGED] changed in_service dtype from f8 to bool for shunt, ward, xward
- [CHANGED] included i_from_ka and i_to_ka in net.res_line
- [CHANGED] recycle parameter added. ppc, Ybus, is_elems and bus_lookup can be reused between multiple powerflows if recycle["ppc"] == True, ppc values (P,Q,V) only get updated.

[1.0.1] - 2016-11-09
----------------------

- [CHANGED] update short introduction example to include transformer
- [CHANGED] included pypower in setup.py requirements (only pypower, not numpy, scipy etc.)
- [CHANGED] mpc / ppc renamed to ppci / ppc
- [FIXED] MANIFEST.ini includes all relevant doc files and exclude report
- [FIXED] handling of tp_pos parameter in create_trafo and create_trafo3w
- [FIXED] init="result" for open bus-line switches