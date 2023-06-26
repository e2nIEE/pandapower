import os
import pytest
import math
import pandas as pd
from pandapower.test import test_path

from pandapower.converter import from_cim as cim2pp


@pytest.fixture(scope="session")
def fullgrid():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'CGMES_v2.4.15_FullGridTestConfiguration_BB_BE_v1.zip'),
                   os.path.join(folder_path, 'CGMES_v2.4.15_FullGridTestConfiguration_BD_v1.zip')]

    return cim2pp.from_cim(file_list=cgmes_files)


@pytest.fixture(scope="session")
def smallgrid_GL():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'CGMES_v2.4.15_SmallGridTestConfiguration_Boundary_v3.0.0.zip'),
                   os.path.join(folder_path, 'CGMES_v2.4.15_SmallGridTestConfiguration_BaseCase_Complete_v3.0.0.zip')]

    return cim2pp.from_cim(file_list=cgmes_files, use_GL_or_DL_profile='GL')


@pytest.fixture(scope="session")
def smallgrid_DL():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'CGMES_v2.4.15_SmallGridTestConfiguration_Boundary_v3.0.0.zip'),
                   os.path.join(folder_path, 'CGMES_v2.4.15_SmallGridTestConfiguration_BaseCase_Complete_v3.0.0.zip')]

    return cim2pp.from_cim(file_list=cgmes_files, use_GL_or_DL_profile='DL')


@pytest.fixture(scope="session")
def realgrid():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'CGMES_v2.4.15_RealGridTestConfiguration_v2.zip')]

    return cim2pp.from_cim(file_list=cgmes_files)


@pytest.fixture(scope="session")
def SimBench_1_HVMVmixed_1_105_0_sw_modified():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'SimBench_1-HVMV-mixed-1.105-0-sw_modified.zip')]

    return cim2pp.from_cim(file_list=cgmes_files)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified(SimBench_1_HVMVmixed_1_105_0_sw_modified):

    assert True


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_load(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 154 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_load.index)
    assert 0.230 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_load.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.load[SimBench_1_HVMVmixed_1_105_0_sw_modified.load['origin_id'] == '_6bcbb329-364a-461e-bdd0-aef5fb25947a'].index]['p_mw'].item()
    assert 0.09090 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_load.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.load[SimBench_1_HVMVmixed_1_105_0_sw_modified.load['origin_id'] == '_6bcbb329-364a-461e-bdd0-aef5fb25947a'].index]['q_mvar'].item()

    assert 3.0 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_load.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.load[SimBench_1_HVMVmixed_1_105_0_sw_modified.load['origin_id'] == '_dcad3c20-024f-4234-83aa-67f03745cec4'].index]['p_mw'].item()
    assert 1.1860 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_load.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.load[SimBench_1_HVMVmixed_1_105_0_sw_modified.load['origin_id'] == '_dcad3c20-024f-4234-83aa-67f03745cec4'].index]['q_mvar'].item()

    assert 34.480 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_load.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.load[SimBench_1_HVMVmixed_1_105_0_sw_modified.load['origin_id'] == '_6184b5be-a494-4ae9-ab24-528a33b19a41'].index]['p_mw'].item()
    assert 13.6270 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_load.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.load[SimBench_1_HVMVmixed_1_105_0_sw_modified.load['origin_id'] == '_6184b5be-a494-4ae9-ab24-528a33b19a41'].index]['q_mvar'].item()


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_line_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line_sc.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_line_est(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line_est.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_line_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line_3ph.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_line(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 194 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.index)
    assert 0.32908338928873615 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_9c2727d3-0232-4352-ac78-b1e4ff562d85'].index]['p_from_mw'].item()
    assert 0.2780429963361799 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_9c2727d3-0232-4352-ac78-b1e4ff562d85'].index]['q_from_mvar'].item()
    assert -0.32900641703695693 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_9c2727d3-0232-4352-ac78-b1e4ff562d85'].index]['p_to_mw'].item()
    assert -0.2883800215714708 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_9c2727d3-0232-4352-ac78-b1e4ff562d85'].index]['q_to_mvar'].item()
    assert 7.697225177921707e-05 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_9c2727d3-0232-4352-ac78-b1e4ff562d85'].index]['pl_mw'].item()
    assert -0.010337025235290898 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_9c2727d3-0232-4352-ac78-b1e4ff562d85'].index]['ql_mvar'].item()
    assert 0.011939852254315474 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_9c2727d3-0232-4352-ac78-b1e4ff562d85'].index]['i_from_ka'].item()
    assert 0.012127162630402383 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_9c2727d3-0232-4352-ac78-b1e4ff562d85'].index]['i_to_ka'].item()
    assert 0.012127162630402383 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_9c2727d3-0232-4352-ac78-b1e4ff562d85'].index]['i_ka'].item()
    assert 1.0416068764183433 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_9c2727d3-0232-4352-ac78-b1e4ff562d85'].index]['vm_from_pu'].item()
    assert -143.8090530125839 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_9c2727d3-0232-4352-ac78-b1e4ff562d85'].index]['va_from_degree'].item()
    assert 1.0414310265920257 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_9c2727d3-0232-4352-ac78-b1e4ff562d85'].index]['vm_to_pu'].item()
    assert -143.80472033332924 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_9c2727d3-0232-4352-ac78-b1e4ff562d85'].index]['va_to_degree'].item()
    assert 5.512346650182902 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_9c2727d3-0232-4352-ac78-b1e4ff562d85'].index]['loading_percent'].item()

    assert 7.351456938401352 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_00279ee7-01a9-4f5e-a99a-6c02f7bebddb'].index]['p_from_mw'].item()
    assert -3.803087780045516 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_00279ee7-01a9-4f5e-a99a-6c02f7bebddb'].index]['q_from_mvar'].item()
    assert -7.343577009323737 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_00279ee7-01a9-4f5e-a99a-6c02f7bebddb'].index]['p_to_mw'].item()
    assert 3.2534705172294767 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_00279ee7-01a9-4f5e-a99a-6c02f7bebddb'].index]['q_to_mvar'].item()
    assert 0.007879929077614811 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_00279ee7-01a9-4f5e-a99a-6c02f7bebddb'].index]['pl_mw'].item()
    assert -0.5496172628160392 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_00279ee7-01a9-4f5e-a99a-6c02f7bebddb'].index]['ql_mvar'].item()
    assert 0.04090204224410598 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_00279ee7-01a9-4f5e-a99a-6c02f7bebddb'].index]['i_from_ka'].item()
    assert 0.03968146590932959 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_00279ee7-01a9-4f5e-a99a-6c02f7bebddb'].index]['i_to_ka'].item()
    assert 0.04090204224410598 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_00279ee7-01a9-4f5e-a99a-6c02f7bebddb'].index]['i_ka'].item()
    assert 1.0621122647814087 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_00279ee7-01a9-4f5e-a99a-6c02f7bebddb'].index]['vm_from_pu'].item()
    assert 7.268678685699929 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_00279ee7-01a9-4f5e-a99a-6c02f7bebddb'].index]['va_from_degree'].item()
    assert 1.0623882310901724 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_00279ee7-01a9-4f5e-a99a-6c02f7bebddb'].index]['vm_to_pu'].item()
    assert 7.109722073381289 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_00279ee7-01a9-4f5e-a99a-6c02f7bebddb'].index]['va_to_degree'].item()
    assert 6.015006212368526 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_00279ee7-01a9-4f5e-a99a-6c02f7bebddb'].index]['loading_percent'].item()


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_impedance_est(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_impedance_est.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_impedance(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_impedance.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_gen_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_gen_sc.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_gen(SimBench_1_HVMVmixed_1_105_0_sw_modified):  # TODO: different net that contains elements?
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_gen.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_ext_grid_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_ext_grid_sc.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_ext_grid_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_ext_grid_3ph.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_ext_grid(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 3 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_ext_grid.index)
    assert -257.1300942171846 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_ext_grid.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid[SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid['origin_id'] == '_5fb17d90-f222-45a2-9e26-05f52a34731f'].index]['p_mw'].item()
    assert 134.7707263896562 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_ext_grid.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid[SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid['origin_id'] == '_5fb17d90-f222-45a2-9e26-05f52a34731f'].index]['q_mvar'].item()


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_bus_dcline(SimBench_1_HVMVmixed_1_105_0_sw_modified):  # TODO: different net that contains elements?
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_dcline.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_bus_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus_sc.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_bus_est(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus_est.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_bus_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus_3ph.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_bus(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 605 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.index)
    assert 1.0491329283083566 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.bus[SimBench_1_HVMVmixed_1_105_0_sw_modified.bus['origin_id'] == '_35ad5aa9-056f-4fc1-86e3-563842550764'].index]['vm_pu'].item()
    assert -144.2919795390245 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.bus[SimBench_1_HVMVmixed_1_105_0_sw_modified.bus['origin_id'] == '_35ad5aa9-056f-4fc1-86e3-563842550764'].index]['va_degree'].item()
    assert -1.770 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.bus[SimBench_1_HVMVmixed_1_105_0_sw_modified.bus['origin_id'] == '_35ad5aa9-056f-4fc1-86e3-563842550764'].index]['p_mw'].item()
    assert 0.09090 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.bus[SimBench_1_HVMVmixed_1_105_0_sw_modified.bus['origin_id'] == '_35ad5aa9-056f-4fc1-86e3-563842550764'].index]['q_mvar'].item()

    assert 1.0568812557235927 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.bus[SimBench_1_HVMVmixed_1_105_0_sw_modified.bus['origin_id'] == '_b7704afe-8d9f-4154-b51c-72f60567f94d'].index]['vm_pu'].item()
    assert 3.616330520902283 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.bus[SimBench_1_HVMVmixed_1_105_0_sw_modified.bus['origin_id'] == '_b7704afe-8d9f-4154-b51c-72f60567f94d'].index]['va_degree'].item()
    assert 0.0 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.bus[SimBench_1_HVMVmixed_1_105_0_sw_modified.bus['origin_id'] == '_b7704afe-8d9f-4154-b51c-72f60567f94d'].index]['p_mw'].item()
    assert 0.0 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.bus[SimBench_1_HVMVmixed_1_105_0_sw_modified.bus['origin_id'] == '_b7704afe-8d9f-4154-b51c-72f60567f94d'].index]['q_mvar'].item()

    assert 1.0471714107338401 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.bus[SimBench_1_HVMVmixed_1_105_0_sw_modified.bus['origin_id'] == '_290a09c9-4b06-4135-aaa7-45db9172e33d'].index]['vm_pu'].item()
    assert -144.16663009041676 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.bus[SimBench_1_HVMVmixed_1_105_0_sw_modified.bus['origin_id'] == '_290a09c9-4b06-4135-aaa7-45db9172e33d'].index]['va_degree'].item()
    assert 0.0 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.bus[SimBench_1_HVMVmixed_1_105_0_sw_modified.bus['origin_id'] == '_290a09c9-4b06-4135-aaa7-45db9172e33d'].index]['p_mw'].item()
    assert 0.0 == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.bus[SimBench_1_HVMVmixed_1_105_0_sw_modified.bus['origin_id'] == '_290a09c9-4b06-4135-aaa7-45db9172e33d'].index]['q_mvar'].item()


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_asymmetric_sgen_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_asymmetric_sgen_3ph.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_asymmetric_sgen(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_asymmetric_sgen.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_asymmetric_load_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_asymmetric_load_3ph.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_asymmetric_load(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_asymmetric_load.index)


def test_test_SimBench_1_HVMVmixed_1_105_0_sw_modified_ext_grid(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 3 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid.index)
    assert 1.0 == SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid[SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid['origin_id'] == '_5fb17d90-f222-45a2-9e26-05f52a34731f']['slack_weight'].item()
    assert '_01ff3523-9c93-45fe-8e35-0d2f4f2' == \
           SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid[SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid['origin_id'] == '_5fb17d90-f222-45a2-9e26-05f52a34731f']['substation'].item()
    assert 0.0 == SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid[SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid['origin_id'] == '_5fb17d90-f222-45a2-9e26-05f52a34731f']['min_p_mw'].item()
    assert 0.0 == SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid[SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid['origin_id'] == '_5fb17d90-f222-45a2-9e26-05f52a34731f']['max_p_mw'].item()
    assert 0.0 == SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid[SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid['origin_id'] == '_5fb17d90-f222-45a2-9e26-05f52a34731f']['min_q_mvar'].item()
    assert 0.0 == SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid[SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid['origin_id'] == '_5fb17d90-f222-45a2-9e26-05f52a34731f']['max_q_mvar'].item()
    assert [[11.3706, 53.601]] == SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid[SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid['origin_id'] == '_5fb17d90-f222-45a2-9e26-05f52a34731f'][
        'coords'].item()


def test_realgrid_sgen(realgrid):
    assert 819 == len(realgrid.sgen.index)
    assert '1149773851' == realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']['name'].item()
    assert '_1362221690_VL_TN1' == realgrid.bus.iloc[realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']['bus'].item()]['origin_id']
    assert 0.0 == realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']['p_mw'].item()
    assert 0.0 == realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']['q_mvar'].item()
    assert 26.53770 == realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']['sn_mva'].item()
    assert 1.0 == realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']['scaling'].item()
    assert not realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']['in_service'].item()
    assert 'Hydro' == realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']['type'].item()
    assert realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']['current_source'].item()
    assert 'SynchronousMachine' == realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']['origin_class'].item()
    assert '_1149773851_HGU_SM_T0' == realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']['terminal'].item()
    assert 0.0 == realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']['k'].item()
    assert math.isnan(realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']['rx'].item())
    assert 20.0 == realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']['vn_kv'].item()
    assert 0.0 == realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']['rdss_ohm'].item()
    assert 0.0 == realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']['xdss_pu'].item()
    assert math.isnan(realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']['lrc_pu'].item())
    assert 'current_source' == realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']['generator_type'].item()

    assert '1994364905' == realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']['name'].item()
    assert '_1129435962_VL_TN1' == realgrid.bus.iloc[realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']['bus'].item()]['origin_id']
    assert 1.00658 == realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']['p_mw'].item()
    assert 1.65901 == realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']['q_mvar'].item()
    assert 49.2443 == realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']['sn_mva'].item()
    assert 1.0 == realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']['scaling'].item()
    assert realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']['in_service'].item()
    assert 'GeneratingUnit' == realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']['type'].item()
    assert realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']['current_source'].item()
    assert 'SynchronousMachine' == realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']['origin_class'].item()
    assert '_1994364905_GU_SM_T0' == realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']['terminal'].item()
    assert 0.0 == realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']['k'].item()
    assert math.isnan(realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']['rx'].item())
    assert 20.0 == realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']['vn_kv'].item()
    assert 0.0 == realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']['rdss_ohm'].item()
    assert 0.0 == realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']['xdss_pu'].item()
    assert math.isnan(realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']['lrc_pu'].item())
    assert 'current_source' == realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']['generator_type'].item()


def test_smallgrid_DL_line_geodata(smallgrid_DL):
    assert 176 == len(smallgrid_DL.line_geodata.index)
    assert [[162.363632, 128.4656], [162.328033, 134.391541], [181.746033, 134.43364]] == \
           smallgrid_DL.line_geodata.iloc[smallgrid_DL.line[smallgrid_DL.line['origin_id'] == '_0447c6f1-c766-11e1-8775-005056c00008'].index]['coords'].item()
    assert [[12.87877, 58.5714264], [12.8923006, 69.33862]] == smallgrid_DL.line_geodata.iloc[smallgrid_DL.line[smallgrid_DL.line['origin_id'] == '_044a5f09-c766-11e1-8775-005056c00008'].index][
        'coords'].item()


def test_smallgrid_DL_bus_geodata(smallgrid_DL):
    assert 118 == len(smallgrid_DL.bus_geodata.index)
    assert 18.5449734 == smallgrid_DL.bus_geodata.iloc[smallgrid_DL.bus[smallgrid_DL.bus['origin_id'] == '_0471bd2a-c766-11e1-8775-005056c00008'].index]['x'].item()
    assert 11.8253975 == smallgrid_DL.bus_geodata.iloc[smallgrid_DL.bus[smallgrid_DL.bus['origin_id'] == '_0471bd2a-c766-11e1-8775-005056c00008'].index]['y'].item()
    assert [[18.5449734, 11.8253975], [18.5449734, 19.41799]] == smallgrid_DL.bus_geodata.iloc[smallgrid_DL.bus[smallgrid_DL.bus['origin_id'] == '_0471bd2a-c766-11e1-8775-005056c00008'].index][
        'coords'].item()


def test_cim2pp(smallgrid_GL):
    assert 118 == len(smallgrid_GL.bus.index)


def test_smallgrid_GL_line_geodata(smallgrid_GL):
    assert 176 == len(smallgrid_GL.line_geodata.index)
    assert [[-0.741597592830658, 51.33917999267578], [-0.9601190090179443, 51.61038589477539], [-1.0638651847839355, 51.73857879638672], [-1.1654152870178223, 52.01515579223633],
            [-1.1700644493103027, 52.199188232421875]] == smallgrid_GL.line_geodata.iloc[smallgrid_GL.line[smallgrid_GL.line['origin_id'] == '_0447c6f1-c766-11e1-8775-005056c00008'].index][
               'coords'].item()
    assert [[-3.339864492416382, 58.50086212158203], [-3.3406713008880615, 58.31454086303711], [-3.6551620960235596, 58.135623931884766], [-4.029672145843506, 57.973060607910156],
            [-4.254667282104492, 57.71146774291992], [-4.405538082122803, 57.53498840332031]] == \
           smallgrid_GL.line_geodata.iloc[smallgrid_GL.line[smallgrid_GL.line['origin_id'] == '_044a5f09-c766-11e1-8775-005056c00008'].index]['coords'].item()


def test_smallgrid_GL_bus_geodata(smallgrid_GL):
    assert 115 == len(smallgrid_GL.bus_geodata.index)
    assert -4.844991207122803 == smallgrid_GL.bus_geodata.iloc[smallgrid_GL.bus[smallgrid_GL.bus['origin_id'] == '_0471bd2a-c766-11e1-8775-005056c00008'].index]['x'].item()
    assert 55.92612075805664 == smallgrid_GL.bus_geodata.iloc[smallgrid_GL.bus[smallgrid_GL.bus['origin_id'] == '_0471bd2a-c766-11e1-8775-005056c00008'].index]['y'].item()
    assert [[-4.844991207122803, 55.92612075805664]] == smallgrid_GL.bus_geodata.iloc[smallgrid_GL.bus[smallgrid_GL.bus['origin_id'] == '_0471bd2a-c766-11e1-8775-005056c00008'].index]['coords'].item()


def test_fullgrid_xward(fullgrid):
    assert 0 == len(fullgrid.xward.index)  # TODO:


def test_fullgrid_ward(fullgrid):
    assert 5 == len(fullgrid.ward.index)
    assert 'BE-Inj-XCA_AL11' == fullgrid.ward[fullgrid.ward['origin_id'] == '_24413233-26c3-4f7e-9f72-4461796938be']['name'].item()
    assert '_d4affe50316740bdbbf4ae9c7cbf3cfd' == fullgrid.bus.iloc[fullgrid.ward[fullgrid.ward['origin_id'] == '_24413233-26c3-4f7e-9f72-4461796938be']['bus'].item()]['origin_id']
    assert -46.816625 == fullgrid.ward[fullgrid.ward['origin_id'] == '_24413233-26c3-4f7e-9f72-4461796938be']['ps_mw'].item()
    assert 79.193778 == fullgrid.ward[fullgrid.ward['origin_id'] == '_24413233-26c3-4f7e-9f72-4461796938be']['qs_mvar'].item()
    assert 0.0 == fullgrid.ward[fullgrid.ward['origin_id'] == '_24413233-26c3-4f7e-9f72-4461796938be']['qz_mvar'].item()
    assert 0.0 == fullgrid.ward[fullgrid.ward['origin_id'] == '_24413233-26c3-4f7e-9f72-4461796938be']['pz_mw'].item()
    assert fullgrid.ward[fullgrid.ward['origin_id'] == '_24413233-26c3-4f7e-9f72-4461796938be']['in_service'].item()
    assert 'EquivalentInjection' == fullgrid.ward[fullgrid.ward['origin_id'] == '_24413233-26c3-4f7e-9f72-4461796938be']['origin_class'].item()
    assert '_53072f42-f77b-47e2-bd9a-e097c910b173' == fullgrid.ward[fullgrid.ward['origin_id'] == '_24413233-26c3-4f7e-9f72-4461796938be']['terminal'].item()


def test_fullgrid_trafo3w(fullgrid):
    assert 1 == len(fullgrid.trafo3w.index)  # TODO: test with more elements
    assert 'BE-TR3_1' == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['name'].item()
    assert None is fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['std_type'].item()
    assert '_ac279ca9-c4e2-0145-9f39-c7160fff094d' == fullgrid.bus.iloc[fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['hv_bus'].item()]['origin_id']
    assert '_99b219f3-4593-428b-a4da-124a54630178' == fullgrid.bus.iloc[fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['mv_bus'].item()]['origin_id']
    assert '_f96d552a-618d-4d0c-a39a-2dea3c411dee' == fullgrid.bus.iloc[fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['lv_bus'].item()]['origin_id']
    assert 650.0 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['sn_hv_mva'].item()
    assert 650.0 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['sn_mv_mva'].item()
    assert 650.0 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['sn_lv_mva'].item()
    assert 380.0 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vn_hv_kv'].item()
    assert 220.0 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vn_mv_kv'].item()
    assert 21.0 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vn_lv_kv'].item()
    assert 15.7560924405895 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vk_hv_percent'].item()
    assert 17.000038713248305 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vk_mv_percent'].item()
    assert 16.752945398532603 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vk_lv_percent'].item()
    assert 0.8394327539433621 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vkr_hv_percent'].item()
    assert 2.400034426828583 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vkr_mv_percent'].item()
    assert 2.369466354325664 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vkr_lv_percent'].item()
    assert 0.0 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['pfe_kw'].item()
    assert 0.05415 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['i0_percent'].item()
    assert 0.0 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['shift_mv_degree'].item()
    assert 0.0 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['shift_lv_degree'].item()
    assert 'mv' == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['tap_side'].item()
    assert 17 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['tap_neutral'].item()
    assert 1.0 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['tap_min'].item()
    assert 33.0 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['tap_max'].item()
    assert 0.6250 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['tap_step_percent'].item()
    assert math.isnan(fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['tap_step_degree'].item())
    assert 17.0 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['tap_pos'].item()
    assert not fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['tap_at_star_point'].item()
    assert fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['in_service'].item()
    assert 'PowerTransformer' == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['origin_class'].item()
    assert '_76e9ca77-f805-40ea-8120-5a6d58416d34' == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['terminal_hv'].item()
    assert '_53fd6693-57e6-482e-8fbe-dcf3531a7ce0' == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['terminal_mv'].item()
    assert '_ca0f7e2e-3442-4ada-a704-91f319c0ebe3' == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['terminal_lv'].item()
    assert '_5f68a129-d5d8-4b71-9743-9ca2572ba26b' == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['PowerTransformerEnd_id_hv'].item()
    assert '_e1f661c0-971d-4ce5-ad39-0ec427f288ab' == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['PowerTransformerEnd_id_mv'].item()
    assert '_2e21d1ef-2287-434c-a767-1ca807cf2478' == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['PowerTransformerEnd_id_lv'].item()
    assert 'RatioTapChanger' == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['tapchanger_class'].item()
    assert '_fe25f43a-7341-446e-a71a-8ab7119ba806' == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['tapchanger_id'].item()
    assert 'YYY' == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vector_group'].item()
    assert isinstance(fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['id_characteristic'].item(), float)
    assert math.isnan(fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vk0_hv_percent'].item())
    assert math.isnan(fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vk0_mv_percent'].item())
    assert math.isnan(fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vk0_lv_percent'].item())
    assert math.isnan(fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vkr0_hv_percent'].item())
    assert math.isnan(fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vkr0_mv_percent'].item())
    assert math.isnan(fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vkr0_lv_percent'].item())
    assert not fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['power_station_unit'].item()
    assert fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['tap_dependent_impedance'].item()
    assert 14 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vk_hv_percent_characteristic'].item()
    assert 16 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vk_mv_percent_characteristic'].item()
    assert 18 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vk_lv_percent_characteristic'].item()
    assert 15 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vkr_hv_percent_characteristic'].item()
    assert 17 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vkr_mv_percent_characteristic'].item()
    assert 19 == fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']['vkr_lv_percent_characteristic'].item()


def test_fullgrid_trafo(fullgrid):
    assert 10 == len(fullgrid.trafo.index)
    assert 'HVDC1_TR2_HVDC2' == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['name'].item()
    assert None is fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['std_type'].item()
    assert '_c142012a-b652-4c03-9c35-aa0833e71831' == fullgrid.bus.iloc[fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['hv_bus'].item()]['origin_id']
    assert '_b01fe92f-68ab-4123-ae45-f22d3e8daad1' == fullgrid.bus.iloc[fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['lv_bus'].item()]['origin_id']
    assert 157.70 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['sn_mva'].item()
    assert 225.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['vn_hv_kv'].item()
    assert 123.90 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['vn_lv_kv'].item()
    assert 12.619851773827161 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['vk_percent'].item()
    assert 0.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['vkr_percent'].item()
    assert 0.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['pfe_kw'].item()
    assert 0.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['i0_percent'].item()
    assert 0.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['shift_degree'].item()
    assert 'hv' == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['tap_side'].item()
    assert 0.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['tap_neutral'].item()
    assert -15.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['tap_min'].item()
    assert 15.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['tap_max'].item()
    assert 1.250 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['tap_step_percent'].item()
    assert math.isnan(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['tap_step_degree'].item())
    assert -2.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['tap_pos'].item()
    assert not fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['tap_phase_shifter'].item()
    assert 1.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['parallel'].item()
    assert 1.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['df'].item()
    assert fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['in_service'].item()
    assert 'PowerTransformer' == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['origin_class'].item()
    assert '_fd64173b-8fb5-4b66-afe5-9a832e6bcb45' == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['terminal_hv'].item()
    assert '_5b52e14e-550a-4084-91cc-14ec5d38e042' == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['terminal_lv'].item()
    assert '_3581a7e1-95c0-4778-a108-1a6740abfacb' == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['PowerTransformerEnd_id_hv'].item()
    assert '_922f1973-62d7-4190-9556-39faa8ca39b8' == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['PowerTransformerEnd_id_lv'].item()
    assert 'RatioTapChanger' == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['tapchanger_class'].item()
    assert '_f6b6428b-d201-4170-89f3-4f630c662b7c' == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['tapchanger_id'].item()
    assert 'YY' == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['vector_group'].item()
    assert isinstance(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['id_characteristic'].item(), float)
    assert math.isnan(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['vk0_percent'].item())
    assert math.isnan(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['vkr0_percent'].item())
    assert math.isnan(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['xn_ohm'].item())
    assert not fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['power_station_unit'].item()
    assert not fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['oltc'].item()
    assert fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['tap_dependent_impedance'].item()
    assert 0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['vkr_percent_characteristic'].item()
    assert 1 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']['vk_percent_characteristic'].item()

    assert 'BE-TR2_6' == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['name'].item()
    assert None is fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['std_type'].item()
    assert '_e44141af-f1dc-44d3-bfa4-b674e5c953d7' == fullgrid.bus.iloc[fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['hv_bus'].item()]['origin_id']
    assert '_5c74cb26-ce2f-40c6-951d-89091eb781b6' == fullgrid.bus.iloc[fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['lv_bus'].item()]['origin_id']
    assert 650.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['sn_mva'].item()
    assert 380.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['vn_hv_kv'].item()
    assert 110.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['vn_lv_kv'].item()
    assert 6.648199321225537 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['vk_percent'].item()
    assert 1.2188364265927978 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['vkr_percent'].item()
    assert 0.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['pfe_kw'].item()
    assert 0.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['i0_percent'].item()
    assert 0.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['shift_degree'].item()
    assert math.isnan(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['tap_side'].item())
    assert pd.isna(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['tap_neutral'].item())
    assert pd.isna(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['tap_min'].item())
    assert pd.isna(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['tap_max'].item())
    assert math.isnan(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['tap_step_percent'].item())
    assert math.isnan(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['tap_step_degree'].item())
    assert math.isnan(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['tap_pos'].item())
    assert not fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['tap_phase_shifter'].item()
    assert 1.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['parallel'].item()
    assert 1.0 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['df'].item()
    assert fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['in_service'].item()
    assert 'PowerTransformer' == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['origin_class'].item()
    assert '_f8f712ea-4c6f-a64d-970f-ffec2af4931c' == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['terminal_hv'].item()
    assert '_6fdc4516-25fc-2f4e-996f-1f590fd5677a' == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['terminal_lv'].item()
    assert '_162712fd-bd8f-2d4d-8ac9-84bf324ef796' == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['PowerTransformerEnd_id_hv'].item()
    assert '_3ee25db5-2305-1d40-a515-01acb2a12e93' == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['PowerTransformerEnd_id_lv'].item()
    assert math.isnan(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['tapchanger_class'].item())
    assert math.isnan(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['tapchanger_id'].item())
    assert 'YY' == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['vector_group'].item()
    assert isinstance(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['id_characteristic'].item(), float)
    assert math.isnan(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['vk0_percent'].item())
    assert math.isnan(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['vkr0_percent'].item())
    assert math.isnan(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['xn_ohm'].item())
    assert not fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['power_station_unit'].item()
    assert not fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['oltc'].item()
    assert fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['tap_dependent_impedance'].item()
    assert 8 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['vkr_percent_characteristic'].item()
    assert 9 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['vk_percent_characteristic'].item()

    assert 1.990 == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_ff3a91ec-2286-a64c-a046-d62bc0163ffe']['tap_step_degree'].item()
    assert 'PhaseTapChangerLinear' == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_ff3a91ec-2286-a64c-a046-d62bc0163ffe']['tapchanger_class'].item()
    assert math.isnan(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_ff3a91ec-2286-a64c-a046-d62bc0163ffe']['id_characteristic'].item())
    assert not fullgrid.trafo[fullgrid.trafo['origin_id'] == '_ff3a91ec-2286-a64c-a046-d62bc0163ffe']['tap_dependent_impedance'].item()
    assert pd.isna(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_ff3a91ec-2286-a64c-a046-d62bc0163ffe']['vkr_percent_characteristic'].item())
    assert pd.isna(fullgrid.trafo[fullgrid.trafo['origin_id'] == '_ff3a91ec-2286-a64c-a046-d62bc0163ffe']['vk_percent_characteristic'].item())


def test_fullgrid_tcsc(fullgrid):
    assert 0 == len(fullgrid.tcsc.index)


def test_fullgrid_switch(fullgrid):
    assert 4 == len(fullgrid.switch.index)
    assert '_5c74cb26-ce2f-40c6-951d-89091eb781b6' == fullgrid.bus.iloc[fullgrid.switch[fullgrid.switch['origin_id'] == '_8a3ad6e1-6e23-b649-880e-4865217501c4']['bus'].item()]['origin_id']
    assert '_c21be5da-d2a6-d94f-8dcb-92e4d6fa48a7' == fullgrid.bus.iloc[fullgrid.switch[fullgrid.switch['origin_id'] == '_8a3ad6e1-6e23-b649-880e-4865217501c4']['element'].item()]['origin_id']
    assert 'b' == fullgrid.switch[fullgrid.switch['origin_id'] == '_8a3ad6e1-6e23-b649-880e-4865217501c4']['et'].item()
    assert 'DS' == fullgrid.switch[fullgrid.switch['origin_id'] == '_8a3ad6e1-6e23-b649-880e-4865217501c4']['type'].item()
    assert fullgrid.switch[fullgrid.switch['origin_id'] == '_8a3ad6e1-6e23-b649-880e-4865217501c4']['closed'].item()
    assert 'BE_DSC_5' == fullgrid.switch[fullgrid.switch['origin_id'] == '_8a3ad6e1-6e23-b649-880e-4865217501c4']['name'].item()
    assert 0.0 == fullgrid.switch[fullgrid.switch['origin_id'] == '_8a3ad6e1-6e23-b649-880e-4865217501c4']['z_ohm'].item()
    assert math.isnan(fullgrid.switch[fullgrid.switch['origin_id'] == '_8a3ad6e1-6e23-b649-880e-4865217501c4']['in_ka'].item())
    assert 'Disconnector' == fullgrid.switch[fullgrid.switch['origin_id'] == '_8a3ad6e1-6e23-b649-880e-4865217501c4']['origin_class'].item()
    assert '_2af7ad2c-062c-1c4f-be3e-9c7cd594ddbb' == fullgrid.switch[fullgrid.switch['origin_id'] == '_8a3ad6e1-6e23-b649-880e-4865217501c4']['terminal_bus'].item()
    assert '_916578a1-7a6e-7347-a5e0-aaf35538949c' == fullgrid.switch[fullgrid.switch['origin_id'] == '_8a3ad6e1-6e23-b649-880e-4865217501c4']['terminal_element'].item()


def test_fullgrid_svc(fullgrid):
    assert 0 == len(fullgrid.svc.index)


def test_fullgrid_storage(fullgrid):
    assert 0 == len(fullgrid.storage.index)


def test_fullgrid_shunt(fullgrid):
    assert 6 == len(fullgrid.shunt.index)
    assert 'BE_S1' == fullgrid.shunt[fullgrid.shunt['origin_id'] == '_d771118f-36e9-4115-a128-cc3d9ce3e3da']['name'].item()
    assert '_f6ee76f7-3d28-6740-aa78-f0bf7176cdad' == fullgrid.bus.iloc[fullgrid.shunt[fullgrid.shunt['origin_id'] == '_d771118f-36e9-4115-a128-cc3d9ce3e3da']['bus'].item()]['origin_id']
    assert -299.99530 == fullgrid.shunt[fullgrid.shunt['origin_id'] == '_d771118f-36e9-4115-a128-cc3d9ce3e3da']['q_mvar'].item()
    assert 0.0 == fullgrid.shunt[fullgrid.shunt['origin_id'] == '_d771118f-36e9-4115-a128-cc3d9ce3e3da']['p_mw'].item()
    assert 110.0 == fullgrid.shunt[fullgrid.shunt['origin_id'] == '_d771118f-36e9-4115-a128-cc3d9ce3e3da']['vn_kv'].item()
    assert 1 == fullgrid.shunt[fullgrid.shunt['origin_id'] == '_d771118f-36e9-4115-a128-cc3d9ce3e3da']['step'].item()
    assert 1 == fullgrid.shunt[fullgrid.shunt['origin_id'] == '_d771118f-36e9-4115-a128-cc3d9ce3e3da']['max_step'].item()
    assert fullgrid.shunt[fullgrid.shunt['origin_id'] == '_d771118f-36e9-4115-a128-cc3d9ce3e3da']['in_service'].item()
    assert 'LinearShuntCompensator' == fullgrid.shunt[fullgrid.shunt['origin_id'] == '_d771118f-36e9-4115-a128-cc3d9ce3e3da']['origin_class'].item()
    assert '_d5e2e58e-ccf6-47d9-b3bb-3088eb7a9b6c' == fullgrid.shunt[fullgrid.shunt['origin_id'] == '_d771118f-36e9-4115-a128-cc3d9ce3e3da']['terminal'].item()


def test_fullgrid_sgen(fullgrid):
    assert 0 == len(fullgrid.sgen.index)


def test_fullgrid_pwl_cost(fullgrid):
    assert 0 == len(fullgrid.pwl_cost.index)


def test_fullgrid_poly_cost(fullgrid):
    assert 0 == len(fullgrid.poly_cost.index)


def test_fullgrid_motor(fullgrid):
    assert 1 == len(fullgrid.motor.index)
    assert 'ASM_1' == fullgrid.motor[fullgrid.motor['origin_id'] == '_2b618292-5fec-af43-ae39-c32566d0a752']['name'].item()
    assert '_f70f6bad-eb8d-4b8f-8431-4ab93581514e' == fullgrid.bus.iloc[fullgrid.motor[fullgrid.motor['origin_id'] == '_2b618292-5fec-af43-ae39-c32566d0a752']['bus'].item()]['origin_id']
    assert math.isnan(fullgrid.motor[fullgrid.motor['origin_id'] == '_2b618292-5fec-af43-ae39-c32566d0a752']['pn_mech_mw'].item())
    assert math.isnan(fullgrid.motor[fullgrid.motor['origin_id'] == '_2b618292-5fec-af43-ae39-c32566d0a752']['loading_percent'].item())
    assert 0.9 == fullgrid.motor[fullgrid.motor['origin_id'] == '_2b618292-5fec-af43-ae39-c32566d0a752']['cos_phi'].item()
    assert 0.9 == fullgrid.motor[fullgrid.motor['origin_id'] == '_2b618292-5fec-af43-ae39-c32566d0a752']['cos_phi_n'].item()
    assert 100.0 == fullgrid.motor[fullgrid.motor['origin_id'] == '_2b618292-5fec-af43-ae39-c32566d0a752']['efficiency_percent'].item()
    assert math.isnan(fullgrid.motor[fullgrid.motor['origin_id'] == '_2b618292-5fec-af43-ae39-c32566d0a752']['efficiency_n_percent'].item())
    assert math.isnan(fullgrid.motor[fullgrid.motor['origin_id'] == '_2b618292-5fec-af43-ae39-c32566d0a752']['lrc_pu'].item())
    assert 225.0 == fullgrid.motor[fullgrid.motor['origin_id'] == '_2b618292-5fec-af43-ae39-c32566d0a752']['vn_kv'].item()
    assert 1.0 == fullgrid.motor[fullgrid.motor['origin_id'] == '_2b618292-5fec-af43-ae39-c32566d0a752']['scaling'].item()
    assert fullgrid.motor[fullgrid.motor['origin_id'] == '_2b618292-5fec-af43-ae39-c32566d0a752']['in_service'].item()
    assert math.isnan(fullgrid.motor[fullgrid.motor['origin_id'] == '_2b618292-5fec-af43-ae39-c32566d0a752']['rx'].item())
    assert 'AsynchronousMachine' == fullgrid.motor[fullgrid.motor['origin_id'] == '_2b618292-5fec-af43-ae39-c32566d0a752']['origin_class'].item()
    assert '_7b71e695-3977-f544-b31f-777cfbbde49b' == fullgrid.motor[fullgrid.motor['origin_id'] == '_2b618292-5fec-af43-ae39-c32566d0a752']['terminal'].item()


def test_fullgrid_measurement(fullgrid):
    assert 0 == len(fullgrid.measurement.index)  # TODO: sv und analogs


def test_fullgrid_load(fullgrid):
    assert 5 == len(fullgrid.load.index)
    assert 'BE_CL_1' == fullgrid.load[fullgrid.load['origin_id'] == '_1324b99a-59ee-0d44-b1f6-15dc0d9d81ff']['name'].item()
    assert '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35' == fullgrid.bus.iloc[fullgrid.load[fullgrid.load['origin_id'] == '_1324b99a-59ee-0d44-b1f6-15dc0d9d81ff']['bus'].item()]['origin_id']
    assert 0.010 == fullgrid.load[fullgrid.load['origin_id'] == '_1324b99a-59ee-0d44-b1f6-15dc0d9d81ff']['p_mw'].item()
    assert 0.010 == fullgrid.load[fullgrid.load['origin_id'] == '_1324b99a-59ee-0d44-b1f6-15dc0d9d81ff']['q_mvar'].item()
    assert 0.0 == fullgrid.load[fullgrid.load['origin_id'] == '_1324b99a-59ee-0d44-b1f6-15dc0d9d81ff']['const_z_percent'].item()
    assert 0.0 == fullgrid.load[fullgrid.load['origin_id'] == '_1324b99a-59ee-0d44-b1f6-15dc0d9d81ff']['const_i_percent'].item()
    assert math.isnan(fullgrid.load[fullgrid.load['origin_id'] == '_1324b99a-59ee-0d44-b1f6-15dc0d9d81ff']['sn_mva'].item())
    assert 1.0 == fullgrid.load[fullgrid.load['origin_id'] == '_1324b99a-59ee-0d44-b1f6-15dc0d9d81ff']['scaling'].item()
    assert fullgrid.load[fullgrid.load['origin_id'] == '_1324b99a-59ee-0d44-b1f6-15dc0d9d81ff']['in_service'].item()
    assert None is fullgrid.load[fullgrid.load['origin_id'] == '_1324b99a-59ee-0d44-b1f6-15dc0d9d81ff']['type'].item()
    assert 'ConformLoad' == fullgrid.load[fullgrid.load['origin_id'] == '_1324b99a-59ee-0d44-b1f6-15dc0d9d81ff']['origin_class'].item()
    assert '_84f6ff75-6bf9-8742-ae06-1481aa3b34de' == fullgrid.load[fullgrid.load['origin_id'] == '_1324b99a-59ee-0d44-b1f6-15dc0d9d81ff']['terminal'].item()


def test_fullgrid_line_geodata(fullgrid):
    assert 0 == len(fullgrid.line_geodata.index)


def test_fullgrid_line(fullgrid):
    assert 11 == len(fullgrid.line.index)
    assert 'BE-Line_7' == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['name'].item()
    assert None is fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['std_type'].item()
    assert '_1fa19c281c8f4e1eaad9e1cab70f923e' == fullgrid.bus.iloc[fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['from_bus'].item()]['origin_id']
    assert '_f70f6bad-eb8d-4b8f-8431-4ab93581514e' == fullgrid.bus.iloc[fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['to_bus'].item()]['origin_id']
    assert 23.0 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['length_km'].item()
    assert 0.19999999999999998 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['r_ohm_per_km'].item()
    assert 3.0 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['x_ohm_per_km'].item()
    assert 3.0000014794808827 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['c_nf_per_km'].item()
    assert 2.5 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['g_us_per_km'].item()
    assert 1.0620 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['max_i_ka'].item()
    assert 1.0 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['df'].item()
    assert 1.0 == fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['parallel'].item()
    assert None is fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['type'].item()
    assert fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']['in_service'].item()
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
    assert 1 == len(fullgrid.impedance.index)  # TODO: test with more elements
    assert 'BE_SC_1' == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['name'].item()
    assert '_514fa0d5-a432-5743-8204-1c8518ffed76' == fullgrid.bus.iloc[fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['from_bus'].item()]['origin_id']
    assert '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35' == fullgrid.bus.iloc[fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['to_bus'].item()]['origin_id']
    assert 8.181818181818182e-05 == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['rft_pu'].item()
    assert 8.181818181818182e-05 == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['xft_pu'].item()
    assert 8.181818181818182e-05 == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['rtf_pu'].item()
    assert 8.181818181818182e-05 == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['xtf_pu'].item()
    assert 1.0 == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['sn_mva'].item()
    assert fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['in_service'].item()
    assert 'SeriesCompensator' == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['origin_class'].item()
    assert '_0b2c4a73-e4dd-4445-acc3-1284ad5a8a70' == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['terminal_from'].item()
    assert '_8c735a96-1b4c-a34d-8823-d6124bd87042' == fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['terminal_to'].item()
    assert math.isnan(fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['rft0_pu'].item())
    assert math.isnan(fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['xft0_pu'].item())
    assert math.isnan(fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['rtf0_pu'].item())
    assert math.isnan(fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']['xtf0_pu'].item())


def test_fullgrid_gen(fullgrid):
    assert len(fullgrid.gen.index) in [7, 9, 10, 11]
    assert 'BE-G5' == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['name'].item()
    assert '_f96d552a-618d-4d0c-a39a-2dea3c411dee' == fullgrid.bus.iloc[fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['bus'].item()]['origin_id']
    assert 118.0 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['p_mw'].item()
    assert 1.04700 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['vm_pu'].item()
    assert 300.0 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['sn_mva'].item()
    assert -200.0 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['min_q_mvar'].item()
    assert 200.0 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['max_q_mvar'].item()
    assert 1.0 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['scaling'].item()
    assert not fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['slack'].item()
    assert fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['in_service'].item()
    assert math.isnan(fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['slack_weight'].item())
    assert 'Nuclear' == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['type'].item()
    assert 'SynchronousMachine' == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['origin_class'].item()
    assert '_b2dcbf07-4676-774f-ae35-86c1ab695de0' == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['terminal'].item()
    assert 50.0 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['min_p_mw'].item()
    assert 200 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['max_p_mw'].item()
    assert 21.0 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['vn_kv'].item()
    assert math.isnan(fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['rdss_ohm'].item())
    assert math.isnan(fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['xdss_pu'].item())
    assert 0.850 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['cos_phi'].item()
    assert 0.0 == fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']['pg_percent'].item()

    assert 1.050 == fullgrid.gen[fullgrid.gen['origin_id'] == '_3a3b27be-b18b-4385-b557-6735d733baf0']['vm_pu'].item()


def test_fullgrid_ext_grid(fullgrid):
    assert 1 == len(fullgrid.ext_grid.index)
    assert 'ES_1' == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['name'].item()
    assert '_f70f6bad-eb8d-4b8f-8431-4ab93581514e' == fullgrid.bus.iloc[fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['bus'].item()]['origin_id']
    assert 1 == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['vm_pu'].item()
    assert 0 == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['va_degree'].item()
    assert math.isnan(fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['slack_weight'].item())
    assert fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['in_service'].item()
    assert 'EnergySource' == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['origin_class'].item()
    assert '_9835652b-053f-cb44-822e-1e26950d989c' == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['terminal'].item()
    assert math.isnan(fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['substation'].item())
    assert math.isnan(fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['min_p_mw'].item())
    assert math.isnan(fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['max_p_mw'].item())
    assert math.isnan(fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['min_q_mvar'].item())
    assert math.isnan(fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['max_q_mvar'].item())
    assert -9.99000 == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['p_mw'].item()
    assert -0.99000 == fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['q_mvar'].item()
    assert math.isnan(fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['s_sc_max_mva'].item())
    assert math.isnan(fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['s_sc_min_mva'].item())
    assert math.isnan(fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['rx_max'].item())
    assert math.isnan(fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['rx_min'].item())
    assert math.isnan(fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['r0x0_max'].item())
    assert math.isnan(fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']['x0x_max'].item())


def test_fullgrid_dcline(fullgrid):
    assert 2 == len(fullgrid.dcline.index)
    assert 'LDC-1230816355' == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['name'].item()
    assert '_27d57afa-6c9d-4b06-93ea-8c88d14af8b1' == fullgrid.bus.iloc[fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['from_bus'].item()]['origin_id']
    assert '_d3d9c515-2ddb-436a-bf17-2f8be2394de3' == fullgrid.bus.iloc[int(fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['to_bus'].item())]['origin_id']
    assert 0.0 == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['p_mw'].item()
    assert 0.0 == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['loss_percent'].item()
    assert 0.0 == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['loss_mw'].item()
    assert 1.0 == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['vm_from_pu'].item()
    assert 1.0 == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['vm_to_pu'].item()
    assert math.isnan(fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['max_p_mw'].item())
    assert math.isnan(fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['min_q_from_mvar'].item())
    assert math.isnan(fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['min_q_to_mvar'].item())
    assert math.isnan(fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['max_q_from_mvar'].item())
    assert math.isnan(fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['max_q_to_mvar'].item())
    assert fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['in_service'].item()
    assert 'DCLineSegment' == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['origin_class'].item()
    assert '_4123e718-716a-4988-bf71-0e525a4422f2' == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['terminal_from'].item()
    assert '_c4c335b5-0405-4539-be10-697f5a3f3e83' == fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']['terminal_to'].item()

    assert math.isnan(fullgrid.dcline[fullgrid.dcline['origin_id'] == '_d7693c6d-58bd-49da-bb24-973a63f9faf1']['to_bus'].item())
    assert math.isnan(fullgrid.dcline[fullgrid.dcline['origin_id'] == '_d7693c6d-58bd-49da-bb24-973a63f9faf1']['loss_mw'].item())
    assert math.isnan(fullgrid.dcline[fullgrid.dcline['origin_id'] == '_d7693c6d-58bd-49da-bb24-973a63f9faf1']['vm_to_pu'].item())
    assert pd.isna(fullgrid.dcline[fullgrid.dcline['origin_id'] == '_d7693c6d-58bd-49da-bb24-973a63f9faf1']['in_service'].item())
    assert math.isnan(fullgrid.dcline[fullgrid.dcline['origin_id'] == '_d7693c6d-58bd-49da-bb24-973a63f9faf1']['terminal_to'].item())


def test_fullgrid_controller(fullgrid):
    assert 8 == len(fullgrid.controller.index)
    for _, obj in fullgrid.controller.iterrows():
        if obj.object.matching_params.get('tid') == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c'].index:
            assert not obj['in_service']
            assert 0.0 == obj['order']
            assert 0 == obj['level']
            assert obj['initial_run']
            assert not obj['recycle'].get('bus_pq')
            assert not obj['recycle'].get('gen')
            assert obj['recycle'].get('trafo')
            assert 11 == obj.object.controlled_bus
            assert 0 == obj.object.controlled_tid
            assert 'lv' == obj.object.side
            assert 15.0 == obj.object.tap_max
            assert -15.0 == obj.object.tap_min
            assert 0.0 == obj.object.tap_neutral
            assert -2.0 == obj.object.tap_pos
            assert 1 == obj.object.tap_side_coeff
            assert 1 == obj.object.tap_sign
            assert math.isnan(obj.object.tap_step_degree)
            assert 1.25 == obj.object.tap_step_percent
            assert 0 == obj.object.tid
            assert math.isnan(obj.object.tol)
            assert '_b01fe92f-68ab-4123-ae45-f22d3e8daad1' == fullgrid.bus.iloc[obj.object.trafobus]['origin_id']
            assert 'trafo' == obj.object.trafotable
            assert '2W' == obj.object.trafotype
            assert math.isnan(obj.object.vm_delta_pu)
            assert math.isnan(obj.object.vm_lower_pu)
            assert None is obj.object.vm_set_pu
            assert math.isnan(obj.object.vm_upper_pu)
        if obj.object.matching_params.get('tid') == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_69a301e8-f6b2-47ad-9f65-52f2eabc9917'].index:
            assert obj['in_service']
            assert 0.0 == obj['order']
            assert 0 == obj['level']
            assert obj['initial_run']
            assert not obj['recycle'].get('bus_pq')
            assert not obj['recycle'].get('gen')
            assert obj['recycle'].get('trafo')
            assert 10 == obj.object.controlled_bus
            assert 3 == obj.object.controlled_tid
            assert 'lv' == obj.object.side
            assert 10.0 == obj.object.tap_max
            assert -10.0 == obj.object.tap_min
            assert 0.0 == obj.object.tap_neutral
            assert 0.0 == obj.object.tap_pos
            assert 1 == obj.object.tap_side_coeff
            assert 1 == obj.object.tap_sign
            assert math.isnan(obj.object.tap_step_degree)
            assert 1.25 == obj.object.tap_step_percent
            assert 3 == obj.object.tid
            assert math.isnan(obj.object.tol)
            assert '_ac772dd8-7910-443f-8af0-a7fca0fb57f9' == fullgrid.bus.iloc[obj.object.trafobus]['origin_id']
            assert 'trafo' == obj.object.trafotable
            assert '2W' == obj.object.trafotype
            assert math.isnan(obj.object.vm_delta_pu)
            assert math.isnan(obj.object.vm_lower_pu)
            assert None is obj.object.vm_set_pu
            assert math.isnan(obj.object.vm_upper_pu)


def test_fullgrid_characteristic_temp(fullgrid):
    assert 8 == len(fullgrid.characteristic_temp.index)


def test_fullgrid_characteristic(fullgrid):
    assert 20 == len(fullgrid.characteristic.index)
    for _, obj in fullgrid.characteristic.iterrows():
        if obj.object.index == fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']['vkr_percent_characteristic'].item():
            assert 'quadratic' == obj.object.kind
            assert [1] == obj.object.x_vals
            assert [1.3405981856094185] == obj.object.y_vals
            break


def test_fullgrid_bus_geodata(fullgrid):
    assert 0 == len(fullgrid.bus_geodata.index)


def test_fullgrid_bus(fullgrid):
    assert 26 == len(fullgrid.bus.index)
    assert 'BE-Busbar_7' == fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['name'].item()
    assert 110.00000 == fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['vn_kv'].item()
    assert 'b' == fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['type'].item()
    assert '' == fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['zone'].item()
    assert fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['in_service'].item()
    assert 'TopologicalNode' == fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['origin_class'].item()
    assert 'tp' == fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['origin_profile'].item()
    assert '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35' == fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['cim_topnode'].item()
    assert math.isnan(fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['ConnectivityNodeContainer_id'].item())
    assert math.isnan(fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['substation_id'].item())
    assert 'BBRUS151; BGENT_51' == fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']['description'].item()

    assert math.isnan(fullgrid.bus[fullgrid.bus['origin_id'] == '_1098b1c9-dc85-40ce-b65c-39ae02a3afaa']['cim_topnode'].item())
    assert math.isnan(fullgrid.bus[fullgrid.bus['origin_id'] == '_1098b1c9-dc85-40ce-b65c-39ae02a3afaa']['description'].item())


def test_fullgrid_asymmetric_sgen(fullgrid):
    assert 0 == len(fullgrid.asymmetric_sgen.index)


def test_fullgrid_asymmetric_load(fullgrid):
    assert 0 == len(fullgrid.asymmetric_load.index)


if __name__ == "__main__":
    pytest.main([__file__])
