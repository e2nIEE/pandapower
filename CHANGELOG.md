# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) 
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]
### Changed
- update short introduction example to include transformer
- included pypower in setup.py requirements (only pypower, not numpy, scipy etc.)
- mpc / ppc renamed to ppci / ppc

### Fixed
- fixed MANIFEST.ini to include all relevant doc files and exclude report
- fixed handling of tp_pos parameter in create_trafo and create_trafo3w
- init="result" fixed for open bus-line switches