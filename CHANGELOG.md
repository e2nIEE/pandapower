# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) 
and this project adheres to [Semantic Versioning](http://semver.org/).

## unreleased
### Changed
- changed in_service dtype from f8 to bool for shunt, ward, xward
- included i_from_ka and i_to_ka in net.res_line
- recycle parameter added. ppc, Ybus, is_elems and bus_lookup can be reused between multiple powerflows
  if recycle["ppc"] == True, ppc values (P,Q,V) only get updated.
- loadcase added as pypower_extension since unnecessary deepcopies were removed

## [1.0.1] - 2016-11-09
### Changed
- update short introduction example to include transformer
- included pypower in setup.py requirements (only pypower, not numpy, scipy etc.)
- mpc / ppc renamed to ppci / ppc

### Fixed
- fixed MANIFEST.ini to include all relevant doc files and exclude report
- fixed handling of tp_pos parameter in create_trafo and create_trafo3w
- init="result" fixed for open bus-line switches