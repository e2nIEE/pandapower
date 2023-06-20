import os
import pytest
import math
from pandapower.test import test_path

from pandapower.converter import from_cim as cim2pp


# TODO: gl dl test
@pytest.fixture
def fullgrid():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'CGMES_v2.4.15_FullGridTestConfiguration_BB_BE_v1.zip'),
                   os.path.join(folder_path, 'CGMES_v2.4.15_FullGridTestConfiguration_BD_v1.zip')]

    return cim2pp.from_cim(file_list=cgmes_files, use_GL_or_DL_profile='GL')


@pytest.fixture
def smallgrid():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'CGMES_v2.4.15_SmallGridTestConfiguration_Boundary_v3.0.0.zip'),
                   os.path.join(folder_path, 'CGMES_v2.4.15_SmallGridTestConfiguration_BaseCase_Complete_v3.0.0.zip')]

    return cim2pp.from_cim(file_list=cgmes_files, use_GL_or_DL_profile='GL')


def test_cim2pp(smallgrid):
    assert 118 == len(smallgrid.bus.index)
    assert 115 == len(smallgrid.bus_geodata.index)


def test_fullgrid(fullgrid):
    assert True


def test_fullgrid_line(fullgrid):
    assert 11 == len(fullgrid.line.index)
    assert 'BE-Line_7' == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['name'].item()
    assert None == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['std_type'].item()
    assert 24 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['from_bus'].item()  # TODO:
    assert 3 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['to_bus'].item()  # TODO:
    assert 23.0 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['length_km'].item()
    assert 0.19999999999999998 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['r_ohm_per_km'].item()
    assert 3.0 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['x_ohm_per_km'].item()
    assert 3.0000014794808827 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['c_nf_per_km'].item()
    assert 2.5 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['g_us_per_km'].item()
    assert 1.0620 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['max_i_ka'].item()
    assert 1.0 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['df'].item()
    assert 1.0 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['parallel'].item()
    assert None == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['type'].item()
    assert True == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['in_service'].item()
    assert 'ACLineSegment' == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['origin_class'].item()
    assert '_57ae9251-c022-4c67-a8eb-611ad54c963c' == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['terminal_from'].item()
    assert '_5b2c65b0-68ce-4530-85b7-385346a3b5e1' == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['terminal_to'].item()
    assert math.isnan(fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['r0_ohm_per_km'].item())
    assert math.isnan(fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['x0_ohm_per_km'].item())
    assert math.isnan(fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['c0_nf_per_km'].item())
    assert 0.0 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['g0_us_per_km'].item()
    assert math.isnan(fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['endtemp_degree'].item())
    assert math.isnan(fullgrid.line[fullgrid.line['origin_id'] == '_6052bacf-9eaa-4217-be91-4c7c89e92a52']['max_i_ka'].item())


def test_fullgrid_impedance(fullgrid):
    assert 1 == len(fullgrid.impedance.index)
    assert 'BE_SC_1' == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['name'].item()
    assert 1 == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['from_bus'].item()  # TODO:
    assert 7 == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['to_bus'].item()  # TODO:
    assert 8.181818181818182e-05 == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['rft_pu'].item()
    assert 8.181818181818182e-05 == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['xft_pu'].item()
    assert 8.181818181818182e-05 == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['rtf_pu'].item()
    assert 8.181818181818182e-05 == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['xtf_pu'].item()
    assert 1.0 == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['sn_mva'].item()
    assert True == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['in_service'].item()
    assert 'SeriesCompensator' == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['origin_class'].item()
    assert '_0b2c4a73-e4dd-4445-acc3-1284ad5a8a70' == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['terminal_from'].item()
    assert '_8c735a96-1b4c-a34d-8823-d6124bd87042' == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['terminal_to'].item()
    # assert 'BE_SC_1' == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362'][].item()
    # assert 'BE_SC_1' == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362'][].item()
    # assert 'BE_SC_1' == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362'][].item()
    # assert 'BE_SC_1' == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362'][].item()


def test_fullgrid_gen(fullgrid):
    assert 9 == len(fullgrid.gen.index)
    assert 'BE-G5' == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['name'].item()
    assert 5 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['bus'].item()  # TODO:
    assert 118.0 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['p_mw'].item()
    assert 1.04700 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['vm_pu'].item()
    assert 300.0 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['sn_mva'].item()
    assert -200.0 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['min_q_mvar'].item()
    assert 200.0 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['max_q_mvar'].item()
    assert 1.0 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['scaling'].item()
    assert False == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['slack'].item()
    assert True == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['in_service'].item()
    # assert 'BE-G5' == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['name'].item()
    assert 'Nuclear' == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['type'].item()
    assert 'SynchronousMachine' == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['origin_class'].item()
    assert '_b2dcbf07-4676-774f-ae35-86c1ab695de0' == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['terminal'].item()
    assert 50.0 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['min_p_mw'].item()
    assert 200 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['max_p_mw'].item()
    assert 21.0 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['vn_kv'].item()
    # assert 'BE-G5' == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['name'].item()
    # assert 'BE-G5' == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['name'].item()
    assert 0.850 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['cos_phi'].item()
    assert 0.0 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['pg_percent'].item()


def test_fullgrid_ext_grid(fullgrid):
    assert 1 == len(fullgrid.ext_grid.index)
    assert 'ES_1' == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['name'].item()
    assert 3 == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['bus'].item()  # TODO:
    assert 1 == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['vm_pu'].item()
    assert 0 == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['va_degree'].item()
    # assert 'ES_1' == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['name'].item()
    assert True == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['in_service'].item()
    assert 'EnergySource' == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['origin_class'].item()
    assert '_9835652b-053f-cb44-822e-1e26950d989c' == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['terminal'].item()
    # assert 'ES_1' == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['name'].item()
    # assert 'ES_1' == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['name'].item()
    # assert 'ES_1' == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['name'].item()
    # assert 'ES_1' == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['name'].item()
    # assert 'ES_1' == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['name'].item()
    assert -9.99000 == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['p_mw'].item()
    assert -0.99000 == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['q_mvar'].item()
    # assert 'ES_1' == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['name'].item()
    # assert 'ES_1' == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['name'].item()
    # assert 'ES_1' == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['name'].item()
    # assert 'ES_1' == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['name'].item()
    # assert 'ES_1' == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['name'].item()
    # assert 'ES_1' == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['name'].item()


def test_fullgrid_dcline(fullgrid):
    assert 2 == len(fullgrid.dcline.index)
    assert 'LDC-1230816355' == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['name'].item()
    assert 19 == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['from_bus'].item()  # TODO:
    # assert 19 == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['from_bus'].item()
    assert 0.0 == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['p_mw'].item()
    assert 0.0 == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['loss_percent'].item()
    assert 0.0 == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['loss_mw'].item()
    assert 1.0 == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['vm_from_pu'].item()
    assert 1.0 == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['vm_to_pu'].item()
    # assert 19 == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['from_bus'].item()
    # assert 19 == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['from_bus'].item()
    # assert 19 == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['from_bus'].item()
    # assert 19 == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['from_bus'].item()
    # assert 19 == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['from_bus'].item()
    assert True == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['in_service'].item()
    assert 'DCLineSegment' == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['origin_class'].item()
    assert '_4123e718-716a-4988-bf71-0e525a4422f2' == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['terminal_from'].item()
    assert '_c4c335b5-0405-4539-be10-697f5a3f3e83' == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['terminal_to'].item()


def test_fullgrid_controller(fullgrid):
    assert True  # TODO:

def test_fullgrid_characteristic_temp(fullgrid):
    assert True  # TODO:

def test_fullgrid_characteristic(fullgrid):
    assert True  # TODO:


def test_fullgrid_bus_geodata(fullgrid):
    assert 0 == len(fullgrid.bus_geodata.index)


def test_fullgrid_bus(fullgrid):
    assert 26 == len(fullgrid.bus.index)
    assert 'BE-Busbar_7' == fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['name'].item()
    assert 110.00000 == fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['vn_kv'].item()
    assert 'b' == fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['type'].item()
    assert '' == fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['zone'].item()
    assert True == fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['in_service'].item()
    assert 'TopologicalNode' == fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['origin_class'].item()
    assert 'tp' == fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['origin_profile'].item()
    assert '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35' == fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['cim_topnode'].item()
    assert math.isnan(fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['ConnectivityNodeContainer_id'].item())
    assert math.isnan(fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['substation_id'].item())
    assert 'BBRUS151; BGENT_51' == fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['description'].item()

    assert math.isnan(fullgrid.bus[fullgrid.bus['origin_id'] == '_1098b1c9-dc85-40ce-b65c-39ae02a3afaa']['cim_topnode'].item())
    assert math.isnan(fullgrid.bus[fullgrid.bus['origin_id'] == '_1098b1c9-dc85-40ce-b65c-39ae02a3afaa']['description'].item())


def test_fullgrid_asymmetric_sgen(fullgrid):
    assert 0 == len(fullgrid.asymmetric_sgen.index)  # TODO:


def test_fullgrid_asymmetric_load(fullgrid):
    assert 0 == len(fullgrid.asymmetric_load.index)  # TODO:


if __name__ == "__main__":
    pytest.main([__file__])
