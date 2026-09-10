# -*- coding: utf-8 -*-

"""Entry point for exporting a pandapower net to CIM/CGMES.

:func:`to_cim` rebuilds the CIM data structure from a pandapower net via
:class:`~pandapower.converter.cim.pp2cim.build_cim_net.PpToCimConverter` and
serializes it to RDF/XML profiles with
:class:`~pandapower.converter.cim.cim_writer.CimWriter`.
"""
from __future__ import annotations
import logging
import time
from typing import TYPE_CHECKING, Dict

from pandapower.auxiliary import pandapowerNet
from .build_cim_net import PpToCimConverter
from ..cim_writer import CimWriter

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger('cim.pp2cim.to_cim')


def to_cim(net: pandapowerNet, file_path: str = None, output_folder: str = None,
           cgmes_version: str = '2.4.15', base_name: str = 'pandapower',
           **kwargs) -> Dict[str, Dict[str, "pd.DataFrame"]]:
    """
    Converts a pandapower net to CGMES (CIM) and optionally writes the RDF/XML files.

    This is the inverse of :func:`pandapower.converter.cim.cim2pp.from_cim.from_cim`. It currently
    targets a faithful *round-trip*: a net that was imported from CGMES carries the original CGMES
    identifiers (``origin_id``), terminals and topology references, which are used here to rebuild
    the CIM data structure.

    :param net: The pandapower net to convert.
    :param file_path: If given, write a single zip archive containing one XML per profile.
        Optional, default: None.
    :param output_folder: If given, write one XML file per profile into this folder.
        Optional, default: None.
    :param cgmes_version: The target CGMES version, '2.4.15' or '3.0'. Optional, default: '2.4.15'.
    :param base_name: File name prefix for the written xml files. Optional, default: 'pandapower'.
    :return: The CIM data structure (profile -> CIM element type -> DataFrame).
    """
    time_start = time.time()
    converter = PpToCimConverter(net, cgmes_version=cgmes_version, **kwargs)
    cim = converter.convert_to_cim()
    time_converted = time.time()

    if file_path is not None or output_folder is not None:
        writer = CimWriter(cim, cgmes_version=cgmes_version)
        if output_folder is not None:
            writer.to_files(output_folder, base_name=base_name)
        if file_path is not None:
            writer.to_zip(file_path, base_name=base_name)

    logger.info("Needed time for converting pp -> cim: %s", time_converted - time_start)
    logger.info("Total time: %s", time.time() - time_start)
    return cim
