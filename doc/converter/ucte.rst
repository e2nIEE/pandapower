===================================================
UCTE-DEF to pandapower
===================================================

panpapower functionality allows to convert grid data from **UCTE** **d**ata **e**xchange **f**ormat for load flow and three phase short circuit studies (UCTE-DEF) to pandapower.
The UCTE-DEF is a simple data format for electric transmission systems for the exchange of grid data introduced by the *Union for the Co-ordination of Transmission of Electricity (UCTE)*, the predecessor of the ENTSO-E.

Using the Converter
--------------------
In order to start the converter the following method is used. Only the location of the UCTE file that should be converted must be specified.

.. autofunction:: pandapower.converter.cim.cim2pp.from_cim.from_cim
