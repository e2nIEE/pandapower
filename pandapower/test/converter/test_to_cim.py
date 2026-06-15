# -*- coding: utf-8 -*-

"""
Tests for the CGMES serializer (:class:`pandapower.converter.cim.cim_writer.CimWriter`).

The serializer is the inverse of the CGMES parser: it turns the parsed CIM data structure
(profile -> CIM element type -> DataFrame) back into CGMES RDF/XML. These tests verify the
round-trip ``parse -> serialize -> parse`` reproduces the parsed data for every CIM class the
cim2pp converter actually models (the data-structure blueprint).

Known limitation reproduced here: the parser truncates the first entry of multi-valued
rdf:resource references (e.g. ``TopologicalIsland.TopologicalNodes``). Those classes are not part
of the converter blueprint and are therefore excluded from the comparison.
"""

import os
import tempfile
import zipfile

import pandas as pd
import pytest

from pandapower.test import test_path

from pandapower.converter.cim.cim_classes import CimParser
from pandapower.converter.cim.cim_writer import CimWriter

# The CGMES profiles targeted by the exporter (DL/GL coordinates are out of scope for now).
TARGET_PROFILES = ['eq', 'eq_bd', 'ssh', 'sv', 'tp', 'tp_bd']

example_cim_path = os.path.join(test_path, "test_files", "example_cim")


# --------------------------------------------------------------------------- helpers

def _parse(files, cgmes_version):
    parser = CimParser(cgmes_version=cgmes_version)
    parser.parse_files(file_list=[os.path.join(example_cim_path, f) for f in files])
    return parser.get_cim_dict()


def _roundtrip(files, cgmes_version):
    """Parse the given CGMES files, serialize them and parse the serialized output again."""
    cim_original = _parse(files, cgmes_version)
    writer = CimWriter(cim_original, cgmes_version=cgmes_version)
    tmp_dir = tempfile.mkdtemp()
    written = writer.to_files(tmp_dir, base_name="roundtrip")
    parser = CimParser(cgmes_version=cgmes_version)
    parser.parse_files(file_list=list(written.values()))
    return cim_original, parser.get_cim_dict()


def _blueprint(cgmes_version):
    return CimParser(cgmes_version=cgmes_version).get_cim_data_structure()


def _normalize(df, columns):
    df = df[[c for c in columns if c in df.columns]].copy()
    if 'rdfId' in df.columns:
        df = df.sort_values('rdfId').reset_index(drop=True)
    # compare everything as nullable strings so dtypes/NaN do not cause false mismatches
    return df.astype('string').fillna('')


def _assert_blueprint_equal(cim_original, cim_roundtrip, cgmes_version):
    blueprint = _blueprint(cgmes_version)
    compared = 0
    for profile in TARGET_PROFILES:
        if profile not in cim_original or profile not in blueprint:
            continue
        for cls, blueprint_df in blueprint[profile].items():
            original = cim_original[profile].get(cls)
            if not isinstance(original, pd.DataFrame) or original.empty:
                continue
            roundtrip = cim_roundtrip.get(profile, {}).get(cls)
            assert isinstance(roundtrip, pd.DataFrame) and not roundtrip.empty, \
                "class %s in profile %s was lost during serialization" % (cls, profile)
            assert len(original) == len(roundtrip), \
                "row count mismatch for %s/%s: %d -> %d" % (profile, cls, len(original), len(roundtrip))
            columns = list(blueprint_df.columns)
            pd.testing.assert_frame_equal(_normalize(original, columns), _normalize(roundtrip, columns),
                                          check_like=True)
            compared += 1
    assert compared > 0, "no CIM classes were compared - the round-trip produced nothing"


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def minigrid_roundtrip():
    return _roundtrip(['CGMES_v2.4.15_MiniGridTestConfiguration_T1_Complete_v3.zip'], '2.4.15')


@pytest.fixture(scope="module")
def microgrid_nb_roundtrip():
    return _roundtrip(['CGMES_v2.4.15_MicroGridTestConfiguration_T4_BE_NB_Complete_v2.zip'], '2.4.15')


@pytest.fixture(scope="module")
def fullgrid_v2_roundtrip():
    return _roundtrip(['CGMES_v2.4.15_FullGridTestConfiguration_BB_BE_v1.zip',
                       'CGMES_v2.4.15_FullGridTestConfiguration_BD_v1.zip'], '2.4.15')


@pytest.fixture(scope="module")
def fullgrid_v3_roundtrip():
    return _roundtrip(['CGMES_v3.0_FullGrid-Merged_v3.0.2.zip'], '3.0')


# --------------------------------------------------------------------------- round-trip equality

def test_minigrid_roundtrip_blueprint_equal(minigrid_roundtrip):
    _assert_blueprint_equal(*minigrid_roundtrip, cgmes_version='2.4.15')


def test_microgrid_nb_roundtrip_blueprint_equal(microgrid_nb_roundtrip):
    _assert_blueprint_equal(*microgrid_nb_roundtrip, cgmes_version='2.4.15')


def test_fullgrid_v2_roundtrip_blueprint_equal(fullgrid_v2_roundtrip):
    _assert_blueprint_equal(*fullgrid_v2_roundtrip, cgmes_version='2.4.15')


def test_fullgrid_v3_roundtrip_blueprint_equal(fullgrid_v3_roundtrip):
    _assert_blueprint_equal(*fullgrid_v3_roundtrip, cgmes_version='3.0')


# --------------------------------------------------------------------------- profiles / structure

def test_target_profiles_serialized(minigrid_roundtrip):
    _, cim_roundtrip = minigrid_roundtrip
    for profile in ['eq', 'ssh', 'sv', 'tp']:
        assert profile in cim_roundtrip, "profile %s missing from serialized output" % profile


def test_eq_uses_rdf_id_ssh_uses_rdf_about(minigrid_roundtrip):
    # The EQ profile defines equipment (rdf:ID) while the SSH profile updates it (rdf:about). After
    # a round-trip the same Terminal rdfIds must appear in both profiles.
    _, cim_roundtrip = minigrid_roundtrip
    eq_terminals = set(cim_roundtrip['eq']['Terminal']['rdfId'])
    ssh_terminals = set(cim_roundtrip['ssh']['Terminal']['rdfId'])
    assert ssh_terminals
    assert ssh_terminals.issubset(eq_terminals)


def test_full_model_header_roundtrip(minigrid_roundtrip):
    cim_original, cim_roundtrip = minigrid_roundtrip
    for profile in ['eq', 'ssh', 'sv', 'tp']:
        fm_original = cim_original[profile]['FullModel'].iloc[0]
        fm_roundtrip = cim_roundtrip[profile]['FullModel'].iloc[0]
        assert fm_original['rdfId'] == fm_roundtrip['rdfId']
        assert fm_original['scenarioTime'] == fm_roundtrip['scenarioTime']
        # the profile URI(s) must survive so the parser can re-detect the profile
        assert fm_original['profile'] == fm_roundtrip['profile']


def test_aclinesegment_values_roundtrip(minigrid_roundtrip):
    cim_original, cim_roundtrip = minigrid_roundtrip
    original = cim_original['eq']['ACLineSegment'].set_index('rdfId').sort_index()
    roundtrip = cim_roundtrip['eq']['ACLineSegment'].set_index('rdfId').sort_index()
    assert list(original.index) == list(roundtrip.index)
    for col in ['r', 'x', 'bch', 'length', 'BaseVoltage']:
        assert original[col].astype('string').fillna('').tolist() == \
               roundtrip[col].astype('string').fillna('').tolist()


# --------------------------------------------------------------------------- output formats

def test_to_xml_returns_bytes_per_profile(minigrid_roundtrip):
    cim_original, _ = minigrid_roundtrip
    writer = CimWriter(cim_original, cgmes_version='2.4.15')
    xml = writer.to_xml()
    assert {'eq', 'ssh', 'sv', 'tp'}.issubset(xml.keys())
    for profile, content in xml.items():
        assert isinstance(content, bytes)
        assert content.lstrip().startswith(b"<?xml")
        assert b"rdf:RDF" in content


def test_to_zip_is_readable(minigrid_roundtrip):
    cim_original, _ = minigrid_roundtrip
    writer = CimWriter(cim_original, cgmes_version='2.4.15')
    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "export.zip")
    writer.to_zip(zip_path)
    assert os.path.isfile(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert any(name.endswith("_EQ.xml") for name in names)
    assert any(name.endswith("_SSH.xml") for name in names)
    # the produced zip must be parseable again by the CimParser
    parser = CimParser(cgmes_version='2.4.15')
    parser.parse_files(file_list=[zip_path])
    assert not parser.get_cim_dict()['eq']['ACLineSegment'].empty


def test_unsupported_version_raises():
    with pytest.raises(ValueError):
        CimWriter({}, cgmes_version='99.0')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
