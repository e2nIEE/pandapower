# -*- coding: utf-8 -*-

"""Serialization of a CIM data structure to CGMES RDF/XML.

:class:`CimWriter` is the inverse of the importer's XML parser: it turns the CIM
data structure (profile -> CIM element type -> DataFrame) into one RDF/XML
document per profile, using the CGMES schema labels and data types to emit the
correct element/attribute tags. It can return the XML as strings, write one file
per profile, or bundle the profiles into a zip archive.
"""
from __future__ import annotations
import logging
import os
import re
import zipfile

import pandas as pd
from lxml import etree

from .cim_tools import get_cim_schema

logger = logging.getLogger(__name__)

# The RDF namespace is identical for all CGMES versions.
RDF_NS = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
# Qualified name of the RDF resource reference attribute.
RDF_RESOURCE = 'rdf:resource'
# The ModelDescription namespace used for the FullModel header.
MD_NS = 'http://iec.ch/TC57/61970-552/ModelDescription/1#'

# Namespace map (prefix -> URI) per CGMES version. The prefixes match the ones used in the
# 'label' fields of the serialized schema (e.g. 'cim:ACLineSegment', 'entsoe:IdentifiedObject.shortName').
NS_DICT = {
    '2.4.15': {
        'cim': 'http://iec.ch/TC57/2013/CIM-schema-cim16#',
        'entsoe': 'http://entsoe.eu/CIM/SchemaExtension/3/1#',
        'md': MD_NS,
        'rdf': RDF_NS,
    },
    '3.0': {
        'cim': 'http://iec.ch/TC57/CIM100#',
        'eu': 'http://iec.ch/TC57/CIM100-European#',
        'md': MD_NS,
        'rdf': RDF_NS,
    },
}

# Default output file name (without extension) per profile.
PROFILE_FILE_SUFFIX = {
    'eq': 'EQ', 'eq_bd': 'EQ_BD', 'ssh': 'SSH', 'sv': 'SV', 'tp': 'TP', 'tp_bd': 'TP_BD',
    'dl': 'DL', 'gl': 'GL',
}

# Classes that are newly *defined* (not updated) inside the SSH/TP/SV profiles and therefore use
# rdf:ID instead of rdf:about. All EQ/EQ_BD/DL/GL classes use rdf:ID; all other SSH classes update
# objects defined in EQ and therefore use rdf:about.
PROFILE_DEFINED_CLASSES = {
    'sv': None,  # None means: every class in this profile is newly defined -> rdf:ID
    'tp': {'TopologicalNode', 'DCTopologicalNode'},
    'tp_bd': {'TopologicalNode'},
}

# FullModel fields (after the CimParser stripped namespaces) that are written as rdf:resource
# references (pointing to urn:uuid: identifiers) instead of literal text.
FULLMODEL_RESOURCE_FIELDS = {'Supersedes', 'DependentOn'}


class CimWriter:
    """
    Serializes a CIM data structure (profile -> CIM element type -> DataFrame, as produced by
    :class:`pandapower.converter.cim.cim_classes.CimParser`) back into CGMES RDF/XML files.

    This is the inverse of :meth:`CimParser._parse_xml_tree`: every DataFrame row becomes one RDF
    element, every non-empty column becomes either a literal child element or an rdf:resource
    reference, driven by the serialized CIM schema (label / data_type templates).
    """

    def __init__(self, cim: dict[str, dict[str, pd.DataFrame]], cgmes_version: str = '2.4.15'):
        """
        :param cim: The CIM data structure to serialize (profile -> element type -> DataFrame).
        :param cgmes_version: The CGMES version, '2.4.15' or '3.0'. Optional, default: '2.4.15'.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cim = cim
        self.cgmes_version = cgmes_version
        if cgmes_version not in NS_DICT:
            raise ValueError(f"Unsupported CGMES version for export: {cgmes_version}. "
                             f"Supported: {list(NS_DICT.keys())}")
        self.ns = NS_DICT[cgmes_version]
        self.cim_schema = get_cim_schema(cgmes_version)

    # ------------------------------------------------------------------ public API

    def to_xml(self) -> dict[str, bytes]:
        """
        Serialize every available profile into an RDF/XML document.

        :return: A dict mapping the profile short name (e.g. 'eq') to the serialized XML bytes.
        """
        result: dict[str, bytes] = {}
        for profile in self.cim:
            if profile not in self.cim_schema and profile != 'eq_bd' and profile != 'tp_bd':
                self.logger.warning("Skipping profile '%s': not part of the CGMES schema.", profile)
                continue
            xml_bytes = self._serialize_profile(profile)
            if xml_bytes is not None:
                result[profile] = xml_bytes
        return result

    def to_files(self, output_folder: str, base_name: str = 'pandapower') -> dict[str, str]:
        """
        Serialize every profile to a separate .xml file inside ``output_folder``.

        :param output_folder: The destination folder (created if it does not exist).
        :param base_name: The file name prefix. Optional, default: 'pandapower'.
        :return: A dict mapping the profile short name to the written file path.
        """
        os.makedirs(output_folder, exist_ok=True)
        written: dict[str, str] = {}
        for profile, xml_bytes in self.to_xml().items():
            suffix = PROFILE_FILE_SUFFIX.get(profile, profile.upper())
            file_path = os.path.join(output_folder, f"{base_name}_{suffix}.xml")
            with open(file_path, 'wb') as f:
                f.write(xml_bytes)
            written[profile] = file_path
        return written

    def to_zip(self, file_path: str, base_name: str = 'pandapower') -> str:
        """
        Serialize every profile into a single zip archive (one .xml per profile).

        :param file_path: The destination .zip path.
        :param base_name: The file name prefix for the contained xml files. Optional, default: 'pandapower'.
        :return: The written zip path.
        """
        with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for profile, xml_bytes in self.to_xml().items():
                suffix = PROFILE_FILE_SUFFIX.get(profile, profile.upper())
                zf.writestr(f"{base_name}_{suffix}.xml", xml_bytes)
        return file_path

    # ------------------------------------------------------------------ serialization

    def _serialize_profile(self, profile: str) -> bytes | None:
        root = etree.Element(self._qname('rdf:RDF'), nsmap=self.ns)

        # write the FullModel header first (if present)
        profile_dict = self.cim[profile]
        wrote_element = False
        if 'FullModel' in profile_dict and isinstance(profile_dict['FullModel'], pd.DataFrame) \
                and not profile_dict['FullModel'].empty:
            self._write_full_model(root, profile_dict['FullModel'])
            wrote_element = True

        schema_profile = self.cim_schema.get(profile, {})
        for class_name, df in profile_dict.items():
            if class_name == 'FullModel':
                continue
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            if class_name not in schema_profile:
                self.logger.warning("Skipping CIM class '%s' in profile '%s': not in schema.",
                                    class_name, profile)
                continue
            self._write_class(root, profile, class_name, df, schema_profile[class_name])
            wrote_element = True

        if not wrote_element:
            return None
        return etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='utf-8')

    def _write_class(self, root, profile: str, class_name: str, df: pd.DataFrame, class_schema: dict):
        class_label = class_schema['label']
        fields = class_schema.get('fields', {})
        use_about = self._uses_about(profile, class_name)
        columns = [c for c in df.columns if c != 'rdfId']

        for _, row in df.iterrows():
            rdf_id = row.get('rdfId')
            if self._is_empty(rdf_id):
                self.logger.warning("Skipping a '%s' element in profile '%s': missing rdfId.",
                                    class_name, profile)
                continue
            element = etree.SubElement(root, self._qname(class_label))
            if use_about:
                element.set(self._qname('rdf:about'), '#' + str(rdf_id))
            else:
                element.set(self._qname('rdf:ID'), str(rdf_id))

            for col in columns:
                if col not in fields:
                    continue
                self._write_field(element, fields[col], row[col])

    def _write_field(self, element, field_schema: dict, value):
        label = field_schema['label']
        data_type = field_schema.get('data_type', '') or ''
        is_resource = RDF_RESOURCE in data_type
        prefix = self._resource_prefix(data_type) if is_resource else ''
        for one_value in self._iter_values(value):
            sub = etree.SubElement(element, self._qname(label))
            if is_resource:
                formatted = self._format_value(one_value)
                # The parser strips the leading '#' from scalar references but keeps it for
                # multi-valued (list) references; normalize so we never emit '##...'.
                if prefix == '#':
                    formatted = formatted.lstrip('#')
                sub.set(self._qname(RDF_RESOURCE), prefix + formatted)
            else:
                sub.text = self._format_value(one_value)

    def _write_full_model(self, root, full_model_df: pd.DataFrame):
        # Use the first FullModel row only (a profile has exactly one).
        row = full_model_df.iloc[0]
        model = etree.SubElement(root, self._qname('md:FullModel'))
        rdf_id = row.get('rdfId')
        if not self._is_empty(rdf_id):
            model.set(self._qname('rdf:about'), self._as_urn(str(rdf_id)))
        for col in full_model_df.columns:
            if col == 'rdfId':
                continue
            value = row[col]
            tag = f'md:Model.{col}'
            for one_value in self._iter_values(value):
                sub = etree.SubElement(model, self._qname(tag))
                if col in FULLMODEL_RESOURCE_FIELDS:
                    sub.set(self._qname(RDF_RESOURCE), self._as_urn(self._format_value(one_value)))
                else:
                    sub.text = self._format_value(one_value)

    @staticmethod
    def _as_urn(value: str) -> str:
        # The parser strips the leading character of scalar 'urn:uuid:...' values (rdf:about /
        # rdf:resource) but keeps multi-valued ones intact; normalize back to a full urn:uuid:.
        if value.startswith('urn:'):
            return value
        if value.startswith('rn:'):
            return 'u' + value
        return 'urn:uuid:' + value

    # ------------------------------------------------------------------ helpers

    def _uses_about(self, profile: str, class_name: str) -> bool:
        if profile not in PROFILE_DEFINED_CLASSES:
            # eq, eq_bd, dl, gl -> objects are defined here (rdf:ID); ssh -> objects are updated (rdf:about)
            return profile == 'ssh'
        defined = PROFILE_DEFINED_CLASSES[profile]
        if defined is None:
            return False  # every class in the profile is newly defined
        return class_name not in defined

    def _qname(self, label: str) -> str:
        prefix, _, local = label.partition(':')
        if prefix not in self.ns:
            raise ValueError(f"Unknown namespace prefix '{prefix}' in label '{label}'.")
        return '{%s}%s' % (self.ns[prefix], local)

    @staticmethod
    def _resource_prefix(data_type: str) -> str:
        # data_type looks like:  rdf:resource="#   or   rdf:resource="http://...#PhaseCode.
        match = re.search(r'rdf:resource="(.*)$', data_type)
        return match.group(1) if match else '#'

    @staticmethod
    def _iter_values(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                if not CimWriter._is_empty(item):
                    yield item
        elif not CimWriter._is_empty(value):
            yield value

    @staticmethod
    def _is_empty(value) -> bool:
        if value is None:
            return True
        if isinstance(value, (list, tuple)):
            return not value
        try:
            return bool(pd.isna(value))
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _format_value(value) -> str:
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if isinstance(value, float) and value.is_integer():
            # avoid writing integers as '5.0' where the source had '5'
            return str(int(value))
        return str(value)
