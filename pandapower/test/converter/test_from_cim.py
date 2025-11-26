import os
from codecs import ignore_errors

import numpy as np
import pytest
import math
import pandas as pd

from pandapower.test import test_path

from pandapower.converter.cim.cim2pp.from_cim import from_cim
from pandapower.run import runpp

from pandapower.control.util.auxiliary import create_trafo_characteristic_object, create_shunt_characteristic_object


@pytest.fixture(scope="session")
def mini_sc_mod():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'CGMES_v2.4.15_MiniGridTestConfiguration_T1_Complete_v3_mod_in_service_ext_grid.zip')]

    return from_cim(file_list=cgmes_files, ignore_errors=False)

@pytest.fixture(scope="session")
def mini_sc():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'CGMES_v2.4.15_MiniGridTestConfiguration_T1_Complete_v3.zip')]

    return from_cim(file_list=cgmes_files, ignore_errors=False)


@pytest.fixture(scope="session")
def mirco_sc():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'CGMES_v2.4.15_MicroGridTestConfiguration_T4_BE_NB_Complete_v2.zip')]

    return from_cim(file_list=cgmes_files, ignore_errors=False)


@pytest.fixture(scope="session")
def fullgrid_v2():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'CGMES_v2.4.15_FullGridTestConfiguration_BB_BE_v1.zip'),
                   os.path.join(folder_path, 'CGMES_v2.4.15_FullGridTestConfiguration_BD_v1.zip')]

    return from_cim(file_list=cgmes_files)


@pytest.fixture(scope="session")
def fullgrid_v2_spline():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'CGMES_v2.4.15_FullGridTestConfiguration_BB_BE_v1.zip'),
                   os.path.join(folder_path, 'CGMES_v2.4.15_FullGridTestConfiguration_BD_v1.zip')]

    net = from_cim(file_list=cgmes_files)
    create_trafo_characteristic_object(net)
    create_shunt_characteristic_object(net)

    return net


@pytest.fixture(scope="session")
def fullgrid_v3():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'CGMES_v3.0_FullGrid-Merged_v3.0.2.zip')]

    return from_cim(file_list=cgmes_files)


@pytest.fixture(scope="session")
def smallgrid_GL():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'CGMES_v2.4.15_SmallGridTestConfiguration_Boundary_v3.0.0.zip'),
                   os.path.join(folder_path, 'CGMES_v2.4.15_SmallGridTestConfiguration_BaseCase_Complete_v3.0.0.zip')]

    return from_cim(file_list=cgmes_files, use_GL_or_DL_profile='GL')


@pytest.fixture(scope="session")
def smallgrid_DL():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'CGMES_v2.4.15_SmallGridTestConfiguration_Boundary_v3.0.0.zip'),
                   os.path.join(folder_path, 'CGMES_v2.4.15_SmallGridTestConfiguration_BaseCase_Complete_v3.0.0.zip')]

    return from_cim(file_list=cgmes_files, use_GL_or_DL_profile='DL')


@pytest.fixture(scope="session")
def realgrid():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'CGMES_v2.4.15_RealGridTestConfiguration_v2.zip')]

    return from_cim(file_list=cgmes_files)


@pytest.fixture(scope="session")
def SimBench_1_HVMVmixed_1_105_0_sw_modified():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'SimBench_1-HVMV-mixed-1.105-0-sw_modified.zip')]

    return from_cim(file_list=cgmes_files, run_powerflow=True)


@pytest.fixture(scope="session")
def Simbench_1_EHV_mixed__2_no_sw():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'Simbench_1-EHV-mixed--2-no_sw.zip')]

    return from_cim(file_list=cgmes_files, create_measurements='SV', run_powerflow=True)


@pytest.fixture(scope="session")
def example_multivoltage():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'example_multivoltage.zip')]

    net = from_cim(file_list=cgmes_files)
    runpp(net, calculate_voltage_angles="auto")
    return net


@pytest.fixture(scope="session")
def SimBench_1_HVMVmixed_1_105_0_sw_modified_no_load_flow():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'SimBench_1-HVMV-mixed-1.105-0-sw_modified.zip')]

    return from_cim(file_list=cgmes_files)


@pytest.fixture(scope="session")
def fullgrid_node_breaker():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, "CGMES_v2.4.15_FullGridTestConfiguration_NB_BE_v3.zip")]

    return from_cim(file_list=cgmes_files)



def test_micro_sc_ext_grid(mini_sc_mod):
    assert len(mini_sc_mod.ext_grid.index) == 2
    element_0 = mini_sc_mod.ext_grid.iloc[mini_sc_mod.ext_grid[
        mini_sc_mod.ext_grid['origin_id'] == '_089c1945-4101-487f-a557-66c013b748f6'].index]
    assert element_0['s_sc_max_mva'].item() == pytest.approx(658.1793068761733, abs=0.00001)
    assert element_0['s_sc_min_mva'].item() == pytest.approx(0.0, abs=0.00001)
    assert element_0['rx_max'].item() == pytest.approx(0.1, abs=0.00001)
    assert element_0['rx_min'].item() == pytest.approx(0.1, abs=0.00001)
    # sc parameter x0x_max, r0x0_max missing


def test_micro_sc_sgen(mini_sc):
    assert len(mini_sc.sgen.index) == 4
    element_0 = mini_sc.sgen.iloc[mini_sc.sgen[
        mini_sc.sgen['origin_id'] == '_392ea173-4f8e-48fa-b2a3-5c3721e93196'].index]
    assert element_0['k'].item() == pytest.approx(1.571438545258903, abs=0.00001)
    assert element_0['rx'].item() == pytest.approx(0.0163, abs=0.00001)
    assert element_0['vn_kv'].item() == pytest.approx(10.5, abs=0.00001)
    assert element_0['rdss_ohm'].item() == pytest.approx(0.01797075, abs=0.00001)
    assert element_0['xdss_pu'].item() == pytest.approx(0.1, abs=0.00001)
    assert element_0['generator_type'].item() == 'current_source'


def test_micro_sc_trafo(mirco_sc):
    assert len(mirco_sc.trafo.index) == 3
    element_0 = mirco_sc.trafo.iloc[mirco_sc.trafo[
        mirco_sc.trafo['origin_id'] == '_e482b89a-fa84-4ea9-8e70-a83d44790957'].index]
    assert element_0['vk_percent'].item() == pytest.approx(12.000000798749786, abs=0.00001)
    assert element_0['vkr_percent'].item() == pytest.approx(0.2149991967411512, abs=0.00001)
    assert element_0['pfe_kw'].item() == pytest.approx(210.9995411616211, abs=0.00001)
    assert element_0['i0_percent'].item() == pytest.approx(0.4131131902379213, abs=0.00001)
    assert element_0['vector_group'].item() == 'Yy' #TODO: needs addidtional test
    assert element_0['vk0_percent'].item() == pytest.approx(12.000000798749786, abs=0.00001)
    assert element_0['vkr0_percent'].item() == pytest.approx(0.2149991967411512, abs=0.00001)
    assert not element_0['oltc'].item() #TODO: needs addidtional test
    assert not element_0['power_station_unit'].item() #TODO: needs addidtional test
    # sc parameter si0_hv_partial, mag0_rx, mag0_percent missing

def test_micro_sc_trafo3w(mirco_sc):
    assert len(mirco_sc.trafo3w.index) == 1
    element_0 = mirco_sc.trafo3w.iloc[mirco_sc.trafo3w[
        mirco_sc.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246'].index]
    assert element_0['vk_hv_percent'].item() == pytest.approx(15.000000556608706, abs=0.00001)
    assert element_0['vk_mv_percent'].item() == pytest.approx(17.000038713248305, abs=0.00001)
    assert element_0['vk_lv_percent'].item() == pytest.approx(16.000038636114326, abs=0.00001)
    assert element_0['vkr_hv_percent'].item() == pytest.approx(0.8000006007231405, abs=0.00001)
    assert element_0['vkr_mv_percent'].item() == pytest.approx(2.400034426828583, abs=0.00001)
    assert element_0['vkr_lv_percent'].item() == pytest.approx(2.330034201105442, abs=0.00001)
    assert element_0['vk0_hv_percent'].item() == pytest.approx(14.999999802944213, abs=0.00001)
    assert element_0['vk0_mv_percent'].item() == pytest.approx(17.0000679239051, abs=0.00001)
    assert element_0['vk0_lv_percent'].item() == pytest.approx(16.000067933460883, abs=0.00001)
    assert element_0['vkr0_hv_percent'].item() == pytest.approx(0, abs=0.00001)
    assert element_0['vkr0_mv_percent'].item() == pytest.approx(0, abs=0.00001)
    assert element_0['vkr0_lv_percent'].item() == pytest.approx(0, abs=0.00001)
    assert not element_0['power_station_unit'].item() #TODO: needs addidtional test
    assert element_0['vector_group'].item() == 'Yyy' #TODO: needs addidtional test

def test_micro_sc_gen(mirco_sc):
    assert len(mirco_sc.gen.index) == 2
    element_0 = mirco_sc.gen.iloc[mirco_sc.gen[
        mirco_sc.gen['origin_id'] == '_550ebe0d-f2b2-48c1-991f-cebea43a21aa'].index]
    assert element_0['vn_kv'].item() == pytest.approx(21.0, abs=0.00001)
    assert element_0['xdss_pu'].item() == pytest.approx(0.17, abs=0.00001)
    assert element_0['rdss_ohm'].item() == pytest.approx(0.0, abs=0.00001) #TODO: needs addidtional test, docu says should be > 0
    assert element_0['cos_phi'].item() == pytest.approx(0.85, abs=0.00001)
    assert element_0['pg_percent'].item() == pytest.approx(0.0, abs=0.00001) #TODO: needs addidtional test
    # sc parameter power_station_trafo missing!


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_no_load_flow_res_bus(
        SimBench_1_HVMVmixed_1_105_0_sw_modified_no_load_flow):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified_no_load_flow.res_bus.index) == 0


def test_example_multivoltage_res_xward(example_multivoltage):
    assert len(example_multivoltage.res_xward.index) == 2
    element_0 = example_multivoltage.res_xward.iloc[example_multivoltage.xward[
        example_multivoltage.xward['origin_id'] == '_78c751ae-91b7-4d81-8732-670085cf8e94'].index]
    assert element_0['p_mw'].item() == pytest.approx(23.941999999999194, abs=0.0001)
    assert element_0['q_mvar'].item() == pytest.approx(-13.126152038621285, abs=0.0001)
    assert element_0['vm_pu'].item() == pytest.approx(1.0261528781410867, abs=0.000001)
    assert element_0['va_internal_degree'].item() == pytest.approx(-3.1471612276732728, abs=0.0001)
    assert element_0['vm_internal_pu'].item() == pytest.approx(1.02616, abs=0.000001)


def test_example_multivoltage_res_trafo3w(example_multivoltage):
    assert 1 == len(example_multivoltage.res_trafo3w)
    element_0 = example_multivoltage.res_trafo3w.iloc[example_multivoltage.trafo3w[
        example_multivoltage.trafo3w['origin_id'] == '_094b399b-84bb-4c0c-9160-f322ba106b99'].index] #TODO was davon ist der originale wert?
    assert element_0['p_hv_mw'].item() == pytest.approx(8.657017822871058, abs=0.0001)
    assert element_0['p_mv_mw'].item() == pytest.approx(-2.9999999999999982, abs=0.0001)
    assert element_0['p_lv_mw'].item() == pytest.approx(-5.651309159685888, abs=0.0001)
    assert element_0['q_hv_mvar'].item() == pytest.approx(3.8347346868849734, abs=0.0001)
    assert element_0['q_mv_mvar'].item() == pytest.approx(-0.9999999999999539, abs=0.0001)
    assert element_0['q_lv_mvar'].item() == pytest.approx(-2.5351958737584237, abs=0.0001)
    assert element_0['pl_mw'].item() == pytest.approx(0.005708663185171936, abs=0.0001)
    assert element_0['ql_mvar'].item() == pytest.approx(0.2995388131265959, abs=0.0001)
    assert element_0['i_hv_ka'].item() == pytest.approx(0.048934706145584116, abs=0.0001)
    assert element_0['i_mv_ka'].item() == pytest.approx(0.09108067515916406, abs=0.0001)
    assert element_0['i_lv_ka'].item() == pytest.approx(0.35669358848757043, abs=0.0001)
    assert element_0['vm_hv_pu'].item() == pytest.approx(1.015553448688208, abs=0.000001)
    assert element_0['vm_mv_pu'].item() == pytest.approx(1.0022663178330908, abs=0.000001)
    assert element_0['vm_lv_pu'].item() == pytest.approx(1.0025566368417127, abs=0.000001)
    assert element_0['va_hv_degree'].item() == pytest.approx(-4.1106386368211005, abs=0.0001)
    assert element_0['va_mv_degree'].item() == pytest.approx(-5.8681443686929216, abs=0.0001)
    assert element_0['va_lv_degree'].item() == pytest.approx(-5.729162028267157, abs=0.0001)
    assert element_0['va_internal_degree'].item() == pytest.approx(-5.071099920236481, abs=0.0001)
    assert element_0['vm_internal_pu'].item() == pytest.approx(1.0073406631833621, abs=0.000001)
    assert element_0['loading_percent'].item() == pytest.approx(24.712456719781486, abs=0.0001)


def test_example_multivoltage_xward(example_multivoltage):
    assert len(example_multivoltage.xward.index) == 2
    element_0 = example_multivoltage.xward[
        example_multivoltage.xward['origin_id'] == '_78c751ae-91b7-4d81-8732-670085cf8e94']
    assert element_0['name'].item() == 'XWard 1'
    assert example_multivoltage.bus.iloc[element_0['bus'].item()][
               'origin_id'] == '_7783eaac-d397-4b74-9b20-4d10e5861213'
    assert element_0['ps_mw'].item() == pytest.approx(23.942, abs=0.000001)
    assert element_0['qs_mvar'].item() == pytest.approx(-12.24187, abs=0.000001)
    assert element_0['qz_mvar'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['pz_mw'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['r_ohm'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['x_ohm'].item() == pytest.approx(0.1, abs=0.000001)
    assert element_0['vm_pu'].item() == pytest.approx(1.02616, abs=0.000001)
    assert math.isnan(element_0['slack_weight'].item())
    assert element_0['in_service'].item()
    assert element_0['origin_class'].item() == 'EquivalentInjection'
    assert element_0['terminal'].item() == '_044c980b-210d-4ec0-8a99-245dcf1ae5c5'


def test_Simbench_1_EHV_mixed__2_no_sw_res_gen(Simbench_1_EHV_mixed__2_no_sw):
    assert len(Simbench_1_EHV_mixed__2_no_sw.res_gen.index) == pytest.approx(338, abs=0.000001)
    element_0 = Simbench_1_EHV_mixed__2_no_sw.res_gen.iloc[Simbench_1_EHV_mixed__2_no_sw.gen[
        Simbench_1_EHV_mixed__2_no_sw.gen['origin_id'] == '_5b01c8ba-9847-49bc-a1f2-0100ccf7df74'].index]
    assert element_0['p_mw'].item() == pytest.approx(297.0, abs=0.0001)
    assert element_0['q_mvar'].item() == pytest.approx(-5.003710432303805, abs=0.0001)
    assert element_0['va_degree'].item() == pytest.approx(34.87201407457379, abs=0.0001)
    assert element_0['vm_pu'].item() == pytest.approx(1.0920, abs=0.000001)

    element_1 = Simbench_1_EHV_mixed__2_no_sw.res_gen.iloc[Simbench_1_EHV_mixed__2_no_sw.gen[
        Simbench_1_EHV_mixed__2_no_sw.gen['origin_id'] == '_4f01682d-ee27-4f5e-b073-bdc90431d61b'].index]
    assert element_1['p_mw'].item() == pytest.approx(604.0, abs=0.0001)
    assert element_1['q_mvar'].item() == pytest.approx(-37.964890640820215, abs=0.0001)
    assert element_1['va_degree'].item() == pytest.approx(37.18871679516421, abs=0.0001)
    assert element_1['vm_pu'].item() == pytest.approx(1.0679999999999996, abs=0.000001)


def test_Simbench_1_EHV_mixed__2_no_sw_res_dcline(Simbench_1_EHV_mixed__2_no_sw):
    assert 6 == len(Simbench_1_EHV_mixed__2_no_sw.res_dcline.index)
    element_0 = Simbench_1_EHV_mixed__2_no_sw.res_dcline.iloc[Simbench_1_EHV_mixed__2_no_sw.dcline[
        Simbench_1_EHV_mixed__2_no_sw.dcline['origin_id'] == '_ee269f63-6d79-4089-923d-a3a0ee080f92'].index]
    assert element_0['p_from_mw'].item() == pytest.approx(0.0, abs=0.0001)
    assert element_0['q_from_mvar'].item() == pytest.approx(-297.4030725955963, abs=0.0001)
    assert element_0['p_to_mw'].item() == pytest.approx(0.0, abs=0.0001)
    assert element_0['q_to_mvar'].item() == pytest.approx(-124.6603285074234, abs=0.0001)
    assert element_0['pl_mw'].item() == pytest.approx(0.0, abs=0.0001)
    assert element_0['vm_from_pu'].item() == pytest.approx(1.0680, abs=0.000001)
    assert element_0['va_from_degree'].item() == pytest.approx(30.248519550084087, abs=0.0001)
    assert element_0['vm_to_pu'].item() == pytest.approx(1.0680, abs=0.000001)
    assert element_0['va_to_degree'].item() == pytest.approx(33.3178813629858, abs=0.0001)


def test_Simbench_1_EHV_mixed__2_no_sw_measurement(Simbench_1_EHV_mixed__2_no_sw):
    assert len(Simbench_1_EHV_mixed__2_no_sw.measurement.index) == 571
    element_0 = Simbench_1_EHV_mixed__2_no_sw.measurement[
        Simbench_1_EHV_mixed__2_no_sw.measurement['element'] ==
        Simbench_1_EHV_mixed__2_no_sw.bus[Simbench_1_EHV_mixed__2_no_sw.bus[
                                              'origin_id'] == '_1cdc1d88-56de-465b-b1a0-968722f2b287'].index[0]]
    assert element_0['name'].item() == 'EHV Bus 1'
    assert element_0['measurement_type'].item() == 'v'
    assert element_0['element_type'].item() == 'bus'
    assert element_0['value'].item() == pytest.approx(1.0920, abs=0.000001)
    assert element_0['std_dev'].item() == pytest.approx(0.001092, abs=0.000001)
    assert element_0['side'].item() is None


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_xward(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_xward.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_ward(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_ward.index) == 0  # TODO:


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_trafo_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_trafo_sc.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_trafo_est(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_trafo_est.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_trafo_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_trafo_3ph.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_trafo3w_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_trafo3w_sc.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_trafo3w_est(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_trafo3w_est.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_trafo3w(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_trafo3w.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_trafo(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_trafo.index) == 8
    element_0 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_trafo.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.trafo[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.trafo['origin_id'] == '_0e8aabc2-5ddb-4946-9ced-9e1eb20881e9'].index]
    assert element_0['p_hv_mw'].item() == pytest.approx(-128.5650471085923, abs=0.0001)
    assert element_0['q_hv_mvar'].item() == pytest.approx(67.3853631948281, abs=0.0001)
    assert element_0['p_lv_mw'].item() == pytest.approx(128.83323218784219, abs=0.0001)
    assert element_0['q_lv_mvar'].item() == pytest.approx(-56.07814415772255, abs=0.0001)
    assert element_0['pl_mw'].item() == pytest.approx(0.2681850792498892, abs=0.0001)
    assert element_0['ql_mvar'].item() == pytest.approx(11.307219037105554, abs=0.0001)
    assert element_0['i_hv_ka'].item() == pytest.approx(0.2019588628357026, abs=0.0001)
    assert element_0['i_lv_ka'].item() == pytest.approx(0.6978657735602077, abs=0.0001)
    assert element_0['vm_hv_pu'].item() == pytest.approx(1.092, abs=0.000001)
    assert element_0['va_hv_degree'].item() == pytest.approx(0.0, abs=0.0001)
    assert element_0['vm_lv_pu'].item() == pytest.approx(1.0567657300787896, abs=0.000001)
    assert element_0['va_lv_degree'].item() == pytest.approx(4.042276580597498, abs=0.0001)
    assert element_0['loading_percent'].item() == pytest.approx(37.98893926676003, abs=0.0001)

    assert SimBench_1_HVMVmixed_1_105_0_sw_modified.res_trafo.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.trafo[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.trafo['origin_id'] == '_97701d2b-79ef-4fcc-958b-c6628e065798'].index][
               'va_hv_degree'].item() == pytest.approx(4.683488464449254, abs=0.0001)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_tcsc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_tcsc.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_switch_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_switch_sc.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_switch_est(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_switch_est.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_switch(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_switch) == 625  # TODO: test with different net with better values
    element_0 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_switch.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.switch[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.switch['origin_id'] == '_004e90a8-cc0d-43f2-a9eb-c374f9479e8f'].index]
    assert math.isnan(element_0['i_ka'].item())
    assert math.isnan(element_0['loading_percent'].item())

    element_1 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_switch.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.switch[
            SimBench_1_HVMVmixed_1_105_0_sw_modified.switch[
                'origin_id'] == '_00bf6ec7-fb79-4a7b-a3d2-ea35490b067c'].index]
    assert element_1['i_ka'].item() == pytest.approx(0.0, abs=0.0001)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_svc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_svc.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_storage_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_storage_3ph.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_storage(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_storage.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_shunt_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_shunt_3ph.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_shunt(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_shunt.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_sgen_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_sgen_sc.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_sgen_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_sgen_3ph.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_sgen(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_sgen.index) == 205
    element_0 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_sgen.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.sgen[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.sgen['origin_id'] == '_24a34602-5f03-4293-9e5b-bb95201356c6'].index]
    assert element_0['p_mw'].item() == pytest.approx(2.0, abs=0.0001)
    assert element_0['q_mvar'].item() == pytest.approx(0.0, abs=0.0001)

    element_1 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_sgen.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.sgen[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.sgen['origin_id'] == '_00d5cff1-01a8-4193-92c6-0e16088efe9b'].index]
    assert element_1['p_mw'].item() == pytest.approx(149.060, abs=0.0001)
    assert element_1['q_mvar'].item() == pytest.approx(0.0, abs=0.0001)

    element_2 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_sgen.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.sgen[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.sgen['origin_id'] == '_5527d842-0d46-44c6-94ed-f5772fab1393'].index]
    assert element_2['p_mw'].item() == pytest.approx(0.1450, abs=0.0001)
    assert element_2['q_mvar'].item() == pytest.approx(0.0, abs=0.0001)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_motor(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_motor.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_load_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_load_3ph.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_load(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_load.index) == 154
    element_0 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_load.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.load[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.load['origin_id'] == '_6bcbb329-364a-461e-bdd0-aef5fb25947a'].index]
    assert element_0['p_mw'].item() == pytest.approx(0.230, abs=0.0001)
    assert element_0['q_mvar'].item() == pytest.approx(0.09090, abs=0.0001)

    element_1 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_load.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.load[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.load['origin_id'] == '_dcad3c20-024f-4234-83aa-67f03745cec4'].index]
    assert element_1['p_mw'].item() == pytest.approx(3.0, abs=0.0001)
    assert element_1['q_mvar'].item() == pytest.approx(1.1860, abs=0.0001)

    element_2 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_load.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.load[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.load['origin_id'] == '_6184b5be-a494-4ae9-ab24-528a33b19a41'].index]
    assert element_2['p_mw'].item() == pytest.approx(34.480, abs=0.0001)
    assert element_2['q_mvar'].item() == pytest.approx(13.6270, abs=0.0001)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_line_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line_sc.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_line_est(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line_est.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_line_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line_3ph.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_line(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.index) == 194
    element_0 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.line[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_9c2727d3-0232-4352-ac78-b1e4ff562d85'].index]
    assert element_0['p_from_mw'].item() == pytest.approx(0.32908338928873615, abs=0.0001)
    assert element_0['q_from_mvar'].item() == pytest.approx(0.2780429963361799, abs=0.0001)
    assert element_0['p_to_mw'].item() == pytest.approx(-0.32900641703695693, abs=0.0001)
    assert element_0['q_to_mvar'].item() == pytest.approx(-0.2883800215714708, abs=0.0001)
    assert element_0['pl_mw'].item() == pytest.approx(7.697225177921707e-05, abs=0.0001)
    assert element_0['ql_mvar'].item() == pytest.approx(-0.010337025235290898, abs=0.0001)
    assert element_0['i_from_ka'].item() == pytest.approx(0.011939852254315474, abs=0.0001)
    assert element_0['i_to_ka'].item() == pytest.approx(0.012127162630402383, abs=0.0001)
    assert element_0['i_ka'].item() == pytest.approx(0.012127162630402383, abs=0.0001)
    assert element_0['vm_from_pu'].item() == pytest.approx(1.0416068764183433, abs=0.000001)
    assert element_0['va_from_degree'].item() == pytest.approx(-143.8090530125839, abs=0.0001)
    assert element_0['vm_to_pu'].item() == pytest.approx(1.0414310265920257, abs=0.000001)
    assert element_0['va_to_degree'].item() == pytest.approx(-143.80472033332924, abs=0.0001)
    assert element_0['loading_percent'].item() == pytest.approx(5.512346650182902, abs=0.0001)

    element_1 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[
            SimBench_1_HVMVmixed_1_105_0_sw_modified.line[
                'origin_id'] == '_00279ee7-01a9-4f5e-a99a-6c02f7bebddb'].index]
    assert element_1['p_from_mw'].item() == pytest.approx(7.351456938401352, abs=0.0001)
    assert element_1['q_from_mvar'].item() == pytest.approx(-3.803087780045516, abs=0.0001)
    assert element_1['p_to_mw'].item() == pytest.approx(-7.343577009323737, abs=0.0001)
    assert element_1['q_to_mvar'].item() == pytest.approx(3.2534705172294767, abs=0.0001)
    assert element_1['pl_mw'].item() == pytest.approx(0.007879929077614811, abs=0.0001)
    assert element_1['ql_mvar'].item() == pytest.approx(-0.5496172628160392, abs=0.0001)
    assert element_1['i_from_ka'].item() == pytest.approx(0.04090204224410598, abs=0.0001)
    assert element_1['i_to_ka'].item() == pytest.approx(0.03968146590932959, abs=0.0001)
    assert element_1['i_ka'].item() == pytest.approx(0.04090204224410598, abs=0.0001)
    assert element_1['vm_from_pu'].item() == pytest.approx(1.0621122647814087, abs=0.000001)
    assert element_1['va_from_degree'].item() == pytest.approx(7.268678685699929, abs=0.0001)
    assert element_1['vm_to_pu'].item() == pytest.approx(1.0623882310901724, abs=0.000001)
    assert element_1['va_to_degree'].item() == pytest.approx(7.109722073381289, abs=0.0001)
    assert element_1['loading_percent'].item() == pytest.approx(6.015006212368526, abs=0.0001)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_impedance_est(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_impedance_est.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_impedance(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_impedance.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_gen_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_gen_sc.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_gen(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_gen.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_ext_grid_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_ext_grid_sc.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_ext_grid_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_ext_grid_3ph.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_ext_grid(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_ext_grid.index) == 3
    element_0 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_ext_grid.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid[
            SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid[
                'origin_id'] == '_5fb17d90-f222-45a2-9e26-05f52a34731f'].index]
    assert element_0['p_mw'].item() == pytest.approx(-257.1300942171846, abs=0.0001)
    assert element_0['q_mvar'].item() == pytest.approx(134.7707263896562, abs=0.0001)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_bus_dcline(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_dcline.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_bus_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus_sc.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_bus_est(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus_est.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_bus_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus_3ph.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_bus(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.index) == 605
    element_0 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.bus[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.bus['origin_id'] == '_35ad5aa9-056f-4fc1-86e3-563842550764'].index]
    assert element_0['vm_pu'].item() == pytest.approx(1.0491329283083566, abs=0.000001)
    assert element_0['va_degree'].item() == pytest.approx(-144.2919795390245, abs=0.0001)
    assert element_0['p_mw'].item() == pytest.approx(-1.770, abs=0.0001)
    assert element_0['q_mvar'].item() == pytest.approx(0.09090, abs=0.0001)

    element_1 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.bus[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.bus['origin_id'] == '_b7704afe-8d9f-4154-b51c-72f60567f94d'].index]
    assert element_1['vm_pu'].item() == pytest.approx(1.0568812557235927, abs=0.000001)
    assert element_1['va_degree'].item() == pytest.approx(3.616330520902283, abs=0.0001)
    assert element_1['p_mw'].item() == pytest.approx(0.0, abs=0.0001)
    assert element_1['q_mvar'].item() == pytest.approx(0.0, abs=0.0001)

    element_2 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.bus[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.bus['origin_id'] == '_290a09c9-4b06-4135-aaa7-45db9172e33d'].index]
    assert element_2['vm_pu'].item() == pytest.approx(1.0471714107338401, abs=0.000001)
    assert element_2['va_degree'].item() == pytest.approx(-144.16663009041676, abs=0.0001)
    assert element_2['p_mw'].item() == pytest.approx(0.0, abs=0.0001)
    assert element_2['q_mvar'].item() == pytest.approx(0.0, abs=0.0001)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_asymmetric_sgen_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_asymmetric_sgen_3ph.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_asymmetric_sgen(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_asymmetric_sgen.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_asymmetric_load_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_asymmetric_load_3ph.index) == 0


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_asymmetric_load(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_asymmetric_load.index) == 0


def test_test_SimBench_1_HVMVmixed_1_105_0_sw_modified_ext_grid(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert len(SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid.index) == 3
    element_0 = SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid['origin_id'] == '_5fb17d90-f222-45a2-9e26-05f52a34731f']
    assert element_0['slack_weight'].item() == pytest.approx(1.0, abs=0.000001)
    assert element_0['substation'].item() == '_01ff3523-9c93-45fe-8e35-0d2f4f2'
    assert element_0['min_p_mw'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['max_p_mw'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['min_q_mvar'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['max_q_mvar'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['geo'].item() == '{"coordinates": [11.3706, 53.601], "type": "Point"}'


def test_realgrid_sgen(realgrid):
    assert len(realgrid.sgen.index) == 819
    element_0 = realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']
    assert element_0['name'].item() == '1149773851'
    assert realgrid.bus.iloc[element_0['bus'].item()]['origin_id'] == '_1362221690_VL_TN1'
    assert element_0['p_mw'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['q_mvar'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['sn_mva'].item() == pytest.approx(26.53770, abs=0.000001)
    assert element_0['scaling'].item() == pytest.approx(1.0, abs=0.000001)
    assert not element_0['in_service'].item()
    assert element_0['type'].item() == 'Hydro'
    assert element_0['current_source'].item()
    assert element_0['origin_class'].item() == 'SynchronousMachine'
    assert element_0['terminal'].item() == '_1149773851_HGU_SM_T0'
    assert element_0['k'].item() == pytest.approx(0.0, abs=0.000001)
    assert math.isnan(element_0['rx'].item())
    assert element_0['vn_kv'].item() == pytest.approx(20.0, abs=0.000001)
    assert element_0['rdss_ohm'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['xdss_pu'].item() == pytest.approx(0.0, abs=0.000001)
    assert math.isnan(element_0['lrc_pu'].item())
    assert element_0['generator_type'].item() == 'current_source'

    element_1 = realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']
    assert element_1['name'].item() == '1994364905'
    assert realgrid.bus.iloc[element_1['bus'].item()]['origin_id'] == '_1129435962_VL_TN1'
    assert element_1['p_mw'].item() == pytest.approx(1.00658, abs=0.000001)
    assert element_1['q_mvar'].item() == pytest.approx(1.65901, abs=0.000001)
    assert element_1['sn_mva'].item() == pytest.approx(49.2443, abs=0.000001)
    assert element_1['scaling'].item() == pytest.approx(1.0, abs=0.000001)
    assert element_1['in_service'].item()
    assert element_1['type'].item() == 'GeneratingUnit'
    assert element_1['current_source'].item()
    assert element_1['origin_class'].item() == 'SynchronousMachine'
    assert element_1['terminal'].item() == '_1994364905_GU_SM_T0'
    assert element_1['k'].item() == pytest.approx(0.0, abs=0.000001)
    assert math.isnan(element_1['rx'].item())
    assert element_1['vn_kv'].item() == pytest.approx(20.0, abs=0.000001)
    assert element_1['rdss_ohm'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_1['xdss_pu'].item() == pytest.approx(0.0, abs=0.000001)
    assert math.isnan(element_1['lrc_pu'].item())
    assert element_1['generator_type'].item() == 'current_source'


def test_smallgrid_DL_line_geodata(smallgrid_DL):
    assert "diagram" in smallgrid_DL.line.columns
    assert len(smallgrid_DL.line.loc[smallgrid_DL.line.diagram.notna()].index) == 176
    element_0 = smallgrid_DL.line[smallgrid_DL.line['origin_id'] == '_0447c6f1-c766-11e1-8775-005056c00008']
    assert element_0['diagram'].item() == ('{"coordinates": [[162.363632, 128.4656], [162.328033, 134.391541], '
            '[181.746033, 134.43364]], "type": "LineString"}')

    element_1 = smallgrid_DL.line[smallgrid_DL.line['origin_id'] == '_044a5f09-c766-11e1-8775-005056c00008']
    assert element_1[
        'diagram'].item() == '{"coordinates": [[12.87877, 58.5714264], [12.8923006, 69.33862]], "type": "LineString"}'


def test_smallgrid_DL_bus_geodata(smallgrid_DL):
    assert "diagram" in smallgrid_DL.bus.columns
    assert len(smallgrid_DL.bus.loc[smallgrid_DL.bus.diagram.notna()].index) == 118
    element_0 = smallgrid_DL.bus[smallgrid_DL.bus['origin_id'] == '_0471bd2a-c766-11e1-8775-005056c00008']
    assert element_0['diagram'].item() == '{"coordinates": [18.5449734, 11.8253975], "type": "Point"}'

    element_1 = smallgrid_DL.bus[smallgrid_DL.bus['origin_id'] == '_04483c26-c766-11e1-8775-005056c00008']
    assert element_1['diagram'].item() == '{"coordinates": [39.15344, 94.86773], "type": "Point"}'


def test_cim2pp(smallgrid_GL):
    assert len(smallgrid_GL.bus.index) == 118


def test_smallgrid_GL_line_geodata(smallgrid_GL):
    assert "geo" in smallgrid_GL.line.columns
    assert len(smallgrid_GL.line.loc[smallgrid_GL.line.geo.notna()].index) == 176
    element_0 = smallgrid_GL.line.loc[smallgrid_GL.line['origin_id'] == '_0447c6f1-c766-11e1-8775-005056c00008']
    assert element_0['geo'].item() == ('{"coordinates": [[-0.741597592830658, 51.33917999267578], '
            '[-0.9601190090179443, 51.61038589477539], [-1.0638651847839355, 51.73857879638672], '
            '[-1.1654152870178223, 52.01515579223633], '
            '[-1.1700644493103027, 52.199188232421875]], "type": "LineString"}')
    element_1 = smallgrid_GL.line.loc[smallgrid_GL.line['origin_id'] == '_044a5f09-c766-11e1-8775-005056c00008']
    assert element_1['geo'].item() == ('{"coordinates": [[-3.339864492416382, 58.50086212158203], '
            '[-3.3406713008880615, 58.31454086303711], [-3.6551620960235596, 58.135623931884766], '
            '[-4.029672145843506, 57.973060607910156], [-4.254667282104492, 57.71146774291992], '
            '[-4.405538082122803, 57.53498840332031]], "type": "LineString"}')


def test_smallgrid_GL_bus_geodata(smallgrid_GL):
    assert "geo" in smallgrid_GL.bus.columns
    assert len(smallgrid_GL.bus.loc[smallgrid_GL.bus.geo.notna()].index) == 115
    element_0 = smallgrid_GL.bus.loc[smallgrid_GL.bus['origin_id'] == '_0471bd2a-c766-11e1-8775-005056c00008']
    assert element_0['geo'].item() == '{"coordinates": [-4.844991207122803, 55.92612075805664], "type": "Point"}'


def test_fullgrid_xward(fullgrid_v2):
    assert len(fullgrid_v2.xward.index) == 0


def test_fullgrid_ward(fullgrid_v2):
    assert len(fullgrid_v2.ward.index) == 5
    element_0 = fullgrid_v2.ward[fullgrid_v2.ward['origin_id'] == '_24413233-26c3-4f7e-9f72-4461796938be']
    assert element_0['name'].item() == 'BE-Inj-XCA_AL11'
    assert fullgrid_v2.bus.iloc[element_0['bus'].item()]['origin_id'] == '_d4affe50316740bdbbf4ae9c7cbf3cfd'
    assert element_0['ps_mw'].item() == pytest.approx(-46.816625, abs=0.000001)
    assert element_0['qs_mvar'].item() == pytest.approx(79.193778, abs=0.000001)
    assert element_0['qz_mvar'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['pz_mw'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['in_service'].item()
    assert element_0['origin_class'].item() == 'EquivalentInjection'
    assert element_0['terminal'].item() == '_53072f42-f77b-47e2-bd9a-e097c910b173'


def test_fullgrid_trafo3w(fullgrid_v2):
    assert len(fullgrid_v2.trafo3w.index) == 1 # TODO: test with more elements
    element_0 = fullgrid_v2.trafo3w[fullgrid_v2.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']
    assert element_0['name'].item() == 'BE-TR3_1'
    assert None is element_0['std_type'].item()
    assert fullgrid_v2.bus.iloc[element_0['hv_bus'].item()]['origin_id'] == '_ac279ca9-c4e2-0145-9f39-c7160fff094d'
    assert fullgrid_v2.bus.iloc[element_0['mv_bus'].item()]['origin_id'] == '_99b219f3-4593-428b-a4da-124a54630178'
    assert fullgrid_v2.bus.iloc[element_0['lv_bus'].item()]['origin_id'] == '_f96d552a-618d-4d0c-a39a-2dea3c411dee'
    assert element_0['sn_hv_mva'].item() == pytest.approx(650.0, abs=0.000001)
    assert element_0['sn_mv_mva'].item() == pytest.approx(650.0, abs=0.000001)
    assert element_0['sn_lv_mva'].item() == pytest.approx(650.0, abs=0.000001)
    assert element_0['vn_hv_kv'].item() == pytest.approx(380.0, abs=0.000001)
    assert element_0['vn_mv_kv'].item() == pytest.approx(220.0, abs=0.000001)
    assert element_0['vn_lv_kv'].item() == pytest.approx(21.0, abs=0.000001)
    assert element_0['vk_hv_percent'].item() == pytest.approx(15.7560924405895, abs=0.000001)
    assert element_0['vk_mv_percent'].item() == pytest.approx(17.000038713248305, abs=0.000001)
    assert element_0['vk_lv_percent'].item() == pytest.approx(16.752945398532603, abs=0.000001)
    assert element_0['vkr_hv_percent'].item() == pytest.approx(0.8394327539433621, abs=0.000001)
    assert element_0['vkr_mv_percent'].item() == pytest.approx(2.400034426828583, abs=0.000001)
    assert element_0['vkr_lv_percent'].item() == pytest.approx(2.369466354325664, abs=0.000001)
    assert element_0['pfe_kw'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['i0_percent'].item() == pytest.approx(0.05415, abs=0.000001)
    assert element_0['shift_mv_degree'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['shift_lv_degree'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['tap_side'].item() == 'mv'
    assert element_0['tap_neutral'].item() == 17
    assert element_0['tap_min'].item() == pytest.approx(1.0, abs=0.000001)
    assert element_0['tap_max'].item() == pytest.approx(33.0, abs=0.000001)
    assert element_0['tap_step_percent'].item() == pytest.approx(0.6250, abs=0.000001)
    assert math.isnan(element_0['tap_step_degree'].item())
    assert element_0['tap_pos'].item() == pytest.approx(17.0, abs=0.000001)
    assert not element_0['tap_at_star_point'].item()
    assert element_0['in_service'].item()
    assert element_0['origin_class'].item() == 'PowerTransformer'
    assert element_0['terminal_hv'].item() == '_76e9ca77-f805-40ea-8120-5a6d58416d34'
    assert element_0['terminal_mv'].item() == '_53fd6693-57e6-482e-8fbe-dcf3531a7ce0'
    assert element_0['terminal_lv'].item() == '_ca0f7e2e-3442-4ada-a704-91f319c0ebe3'
    assert element_0['PowerTransformerEnd_id_hv'].item() == '_5f68a129-d5d8-4b71-9743-9ca2572ba26b'
    assert element_0['PowerTransformerEnd_id_mv'].item() == '_e1f661c0-971d-4ce5-ad39-0ec427f288ab'
    assert element_0['PowerTransformerEnd_id_lv'].item() == '_2e21d1ef-2287-434c-a767-1ca807cf2478'
    assert element_0['tapchanger_class'].item() == 'RatioTapChanger'
    assert element_0['tapchanger_id'].item() == '_fe25f43a-7341-446e-a71a-8ab7119ba806'
    assert element_0['vector_group'].item() == 'Yyy'
    assert isinstance(element_0['id_characteristic_table'].item(), np.int64)
    assert isinstance(element_0['id_characteristic_table'].dtype, pd.Int64Dtype)
    assert math.isnan(element_0['vk0_hv_percent'].item())
    assert math.isnan(element_0['vk0_mv_percent'].item())
    assert math.isnan(element_0['vk0_lv_percent'].item())
    assert math.isnan(element_0['vkr0_hv_percent'].item())
    assert math.isnan(element_0['vkr0_mv_percent'].item())
    assert math.isnan(element_0['vkr0_lv_percent'].item())
    assert not element_0['power_station_unit'].item()
    assert element_0['tap_dependency_table'].item()
    assert element_0['CurrentLimit.value_hv'].item() == pytest.approx(844.38, abs=0.000001)
    assert element_0['CurrentLimit.value_mv'].item() == pytest.approx(1535.22, abs=0.000001)
    assert element_0['CurrentLimit.value_lv'].item() == pytest.approx(16083.36, abs=0.000001)
    assert element_0['OperationalLimitType.limitType_hv'].item() == 'patlt'
    assert element_0['OperationalLimitType.limitType_mv'].item() == 'patlt'
    assert element_0['OperationalLimitType.limitType_lv'].item() == 'patlt'
    assert element_0['OperationalLimitType.acceptableDuration_hv'].item() == pytest.approx(10.0, abs=0.000001)
    assert element_0['OperationalLimitType.acceptableDuration_mv'].item() == pytest.approx(10.0, abs=0.000001)
    assert element_0['OperationalLimitType.acceptableDuration_lv'].item() == pytest.approx(10.0, abs=0.000001)


def test_fullgrid_trafo3w_spline(fullgrid_v2_spline):
    assert "trafo_characteristic_spline" in fullgrid_v2_spline
    element_0 = fullgrid_v2_spline.trafo3w[
        fullgrid_v2_spline.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']
    cols = ["voltage_ratio_characteristic", "angle_deg_characteristic",
            "vk_hv_percent_characteristic", "vkr_hv_percent_characteristic", "vk_mv_percent_characteristic",
            "vkr_mv_percent_characteristic", "vk_lv_percent_characteristic", "vkr_lv_percent_characteristic"]
    spline_row = fullgrid_v2_spline["trafo_characteristic_spline"][
        fullgrid_v2_spline["trafo_characteristic_spline"]["id_characteristic"] == element_0[
            'id_characteristic_spline'].item()]
    assert element_0['id_characteristic_spline'].item() == 7
    assert not spline_row.empty
    assert spline_row[cols].notna().all(axis=1).item()


def test_fullgrid_trafo(fullgrid_v2):
    assert len(fullgrid_v2.trafo.index) == 10
    element_0 = fullgrid_v2.trafo[fullgrid_v2.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']
    assert element_0['name'].item() == 'HVDC1_TR2_HVDC2'
    assert None is element_0['std_type'].item()
    assert fullgrid_v2.bus.iloc[element_0['hv_bus'].item()]['origin_id'] == '_c142012a-b652-4c03-9c35-aa0833e71831'
    assert fullgrid_v2.bus.iloc[element_0['lv_bus'].item()]['origin_id'] == '_b01fe92f-68ab-4123-ae45-f22d3e8daad1'
    assert element_0['sn_mva'].item() == pytest.approx(157.70, abs=0.000001)
    assert element_0['vn_hv_kv'].item() == pytest.approx(225.0, abs=0.000001)
    assert element_0['vn_lv_kv'].item() == pytest.approx(123.90, abs=0.000001)
    assert element_0['vk_percent'].item() == pytest.approx(12.619851773827161, abs=0.000001)
    assert element_0['vkr_percent'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['pfe_kw'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['i0_percent'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['shift_degree'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['tap_side'].item() == 'hv'
    assert element_0['tap_neutral'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['tap_min'].item() == pytest.approx(-15.0, abs=0.000001)
    assert element_0['tap_max'].item() == pytest.approx(15.0, abs=0.000001)
    assert element_0['tap_step_percent'].item() == pytest.approx(1.250, abs=0.000001)
    assert math.isnan(element_0['tap_step_degree'].item())
    assert element_0['tap_pos'].item() == pytest.approx(-2.0, abs=0.000001)
    assert element_0['tap_changer_type'].item() == "Ratio"
    assert element_0['parallel'].item() == pytest.approx(1.0, abs=0.000001)
    assert element_0['df'].item() == pytest.approx(1.0, abs=0.000001)
    assert element_0['in_service'].item()
    assert element_0['origin_class'].item() == 'PowerTransformer'
    assert element_0['terminal_hv'].item() == '_fd64173b-8fb5-4b66-afe5-9a832e6bcb45'
    assert element_0['terminal_lv'].item() == '_5b52e14e-550a-4084-91cc-14ec5d38e042'
    assert element_0['PowerTransformerEnd_id_hv'].item() == '_3581a7e1-95c0-4778-a108-1a6740abfacb'
    assert element_0['PowerTransformerEnd_id_lv'].item() == '_922f1973-62d7-4190-9556-39faa8ca39b8'
    assert element_0['tapchanger_class'].item() == 'RatioTapChanger'
    assert element_0['tapchanger_id'].item() == '_f6b6428b-d201-4170-89f3-4f630c662b7c'
    assert element_0['vector_group'].item() == 'YNyn'
    assert isinstance(element_0['id_characteristic_table'].item(), np.int64)
    assert isinstance(element_0['id_characteristic_table'].dtype, pd.Int64Dtype)
    assert math.isnan(element_0['vk0_percent'].item())
    assert math.isnan(element_0['vkr0_percent'].item())
    assert math.isnan(element_0['xn_ohm'].item())
    assert not element_0['power_station_unit'].item()
    assert not element_0['oltc'].item()
    assert element_0['tap_dependency_table'].item()
    assert element_0['CurrentLimit.value_hv'].item() == pytest.approx(413.9, abs=0.000001)
    assert element_0['CurrentLimit.value_lv'].item() == pytest.approx(734.9, abs=0.000001)
    assert element_0['OperationalLimitType.limitType_hv'].item() == 'patl'
    assert element_0['OperationalLimitType.limitType_lv'].item() == 'patl'
    assert element_0['OperationalLimitType.acceptableDuration_hv'].item() == pytest.approx(20.0, abs=0.000001)
    assert element_0['OperationalLimitType.acceptableDuration_lv'].item() == pytest.approx(20.0, abs=0.000001)

    element_1 = fullgrid_v2.trafo[fullgrid_v2.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']
    assert element_1['name'].item() == 'BE-TR2_6'
    assert None is element_1['std_type'].item()
    assert fullgrid_v2.bus.iloc[element_1['hv_bus'].item()]['origin_id'] == '_e44141af-f1dc-44d3-bfa4-b674e5c953d7'
    assert fullgrid_v2.bus.iloc[element_1['lv_bus'].item()]['origin_id'] == '_5c74cb26-ce2f-40c6-951d-89091eb781b6'
    assert element_1['sn_mva'].item() == pytest.approx(650.0, abs=0.000001)
    assert element_1['vn_hv_kv'].item() == pytest.approx(380.0, abs=0.000001)
    assert element_1['vn_lv_kv'].item() == pytest.approx(110.0, abs=0.000001)
    assert element_1['vk_percent'].item() == pytest.approx(6.648199321225537, abs=0.000001)
    assert element_1['vkr_percent'].item() == pytest.approx(1.2188364265927978, abs=0.000001)
    assert element_1['pfe_kw'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_1['i0_percent'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_1['shift_degree'].item() == pytest.approx(0.0, abs=0.000001)
    assert None is element_1['tap_side'].item()
    assert pd.isna(element_1['tap_neutral'].item())
    assert pd.isna(element_1['tap_min'].item())
    assert pd.isna(element_1['tap_max'].item())
    assert math.isnan(element_1['tap_step_percent'].item())
    assert math.isnan(element_1['tap_step_degree'].item())
    assert math.isnan(element_1['tap_pos'].item())
    assert pd.isna(element_1['tap_changer_type'].item())
    assert element_1['parallel'].item() == pytest.approx(1.0, abs=0.000001)
    assert element_1['df'].item() == pytest.approx(1.0, abs=0.000001)
    assert element_1['in_service'].item()
    assert element_1['origin_class'].item() == 'PowerTransformer'
    assert element_1['terminal_hv'].item() == '_f8f712ea-4c6f-a64d-970f-ffec2af4931c'
    assert element_1['terminal_lv'].item() == '_6fdc4516-25fc-2f4e-996f-1f590fd5677a'
    assert element_1['PowerTransformerEnd_id_hv'].item() == '_162712fd-bd8f-2d4d-8ac9-84bf324ef796'
    assert element_1['PowerTransformerEnd_id_lv'].item() == '_3ee25db5-2305-1d40-a515-01acb2a12e93'
    assert math.isnan(element_1['tapchanger_class'].item())
    assert math.isnan(element_1['tapchanger_id'].item())
    assert element_1['vector_group'].item() == 'Yy'
    assert isinstance(element_1['id_characteristic_table'].item(), np.int64)
    assert isinstance(element_0['id_characteristic_table'].dtype, pd.Int64Dtype)
    assert math.isnan(element_1['vk0_percent'].item())
    assert math.isnan(element_1['vkr0_percent'].item())
    assert math.isnan(element_1['xn_ohm'].item())
    assert not element_1['power_station_unit'].item()
    assert not element_1['oltc'].item()
    assert element_1['tap_dependency_table'].item()
    assert element_1['CurrentLimit.value_hv'].item() == pytest.approx(844.38, abs=0.000001)
    assert element_1['CurrentLimit.value_lv'].item() == pytest.approx(3070.44, abs=0.000001)
    assert element_1['OperationalLimitType.limitType_hv'].item() == 'patlt'
    assert element_1['OperationalLimitType.limitType_lv'].item() == 'patlt'
    assert element_1['OperationalLimitType.acceptableDuration_hv'].item() == pytest.approx(10.0, abs=0.000001)
    assert element_1['OperationalLimitType.acceptableDuration_lv'].item() == pytest.approx(10.0, abs=0.000001)

    element_2 = fullgrid_v2.trafo[fullgrid_v2.trafo['origin_id'] == '_ff3a91ec-2286-a64c-a046-d62bc0163ffe']
    assert element_2['tap_step_degree'].item() == pytest.approx(1.990, abs=0.000001)
    assert element_2['tapchanger_class'].item() == 'PhaseTapChangerLinear'
    assert element_2['tap_changer_type'].item() == "Ideal"
    assert pd.isna(element_2['id_characteristic_table'].item())
    assert not element_2['tap_dependency_table'].item()
    assert element_2['CurrentLimit.value_hv'].item() == pytest.approx(938.2, abs=0.000001)
    assert element_2['CurrentLimit.value_lv'].item() == pytest.approx(3070.44, abs=0.000001)
    assert element_2['OperationalLimitType.limitType_hv'].item() == 'patl'
    assert element_2['OperationalLimitType.limitType_lv'].item() == 'patlt'
    assert element_2['OperationalLimitType.acceptableDuration_hv'].item() == pytest.approx(10.0, abs=0.000001)
    assert element_2['OperationalLimitType.acceptableDuration_lv'].item() == pytest.approx(10.0, abs=0.000001)


def test_fullgrid_trafo_spline(fullgrid_v2_spline):
    assert "trafo_characteristic_spline" in fullgrid_v2_spline
    element_0 = fullgrid_v2_spline.trafo[
        fullgrid_v2_spline.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']
    cols = ["voltage_ratio_characteristic", "angle_deg_characteristic",
            "vk_percent_characteristic", "vkr_percent_characteristic"]
    spline_row = fullgrid_v2_spline["trafo_characteristic_spline"][
        fullgrid_v2_spline["trafo_characteristic_spline"]["id_characteristic"] == element_0[
            'id_characteristic_spline'].item()]
    assert element_0['id_characteristic_spline'].item() == 0
    assert not spline_row.empty
    assert spline_row[cols].notna().all(axis=1).item()

    element_1 = fullgrid_v2_spline.trafo[
        fullgrid_v2_spline.trafo['origin_id'] == '_ff3a91ec-2286-a64c-a046-d62bc0163ffe']
    assert pd.isna(element_1['id_characteristic_spline'].item())


def test_fullgrid_tcsc(fullgrid_v2):
    assert len(fullgrid_v2.tcsc.index) == 0


def test_fullgrid_switch(fullgrid_v2):
    assert len(fullgrid_v2.switch.index) == 4
    element_0 = fullgrid_v2.switch[fullgrid_v2.switch['origin_id'] == '_8a3ad6e1-6e23-b649-880e-4865217501c4']
    assert fullgrid_v2.bus.iloc[element_0['bus'].item()]['origin_id'] == '_5c74cb26-ce2f-40c6-951d-89091eb781b6'
    assert fullgrid_v2.bus.iloc[element_0['element'].item()]['origin_id'] == '_c21be5da-d2a6-d94f-8dcb-92e4d6fa48a7'
    assert element_0['et'].item() == 'b'
    assert element_0['type'].item() == 'DS'
    assert element_0['closed'].item()
    assert element_0['name'].item() == 'BE_DSC_5'
    assert element_0['z_ohm'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['in_ka'].item() == pytest.approx(0.09999, abs=0.000001)
    assert 'Disconnector' == element_0['origin_class'].item()
    assert element_0['terminal_bus'].item() == '_2af7ad2c-062c-1c4f-be3e-9c7cd594ddbb'
    assert element_0['terminal_element'].item() == '_916578a1-7a6e-7347-a5e0-aaf35538949c'


def test_fullgrid_svc(fullgrid_v2):
    assert len(fullgrid_v2.svc.index) == 0


def test_fullgrid_storage(fullgrid_v2):
    assert len(fullgrid_v2.storage.index) == 0


def test_fullgrid_shunt(fullgrid_v2):
    assert len(fullgrid_v2.shunt.index) == 6
    element_0 = fullgrid_v2.shunt[fullgrid_v2.shunt['origin_id'] == '_d771118f-36e9-4115-a128-cc3d9ce3e3da']
    assert element_0['name'].item() == 'BE_S1'
    assert fullgrid_v2.bus.iloc[element_0['bus'].item()]['origin_id'] == '_f6ee76f7-3d28-6740-aa78-f0bf7176cdad'
    assert element_0['q_mvar'].item() == pytest.approx(-299.99530, abs=0.000001)
    assert element_0['p_mw'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['vn_kv'].item() == pytest.approx(110.0, abs=0.000001)
    assert element_0['step'].item() == 1
    assert element_0['max_step'].item() == 1
    assert element_0['in_service'].item()
    assert element_0['origin_class'].item() == 'LinearShuntCompensator'
    assert element_0['terminal'].item() == '_d5e2e58e-ccf6-47d9-b3bb-3088eb7a9b6c'
    assert not element_0['step_dependency_table'].item()
    assert pd.isna(element_0['id_characteristic_table'].item())


def test_fullgrid_sgen(fullgrid_v2):
    assert len(fullgrid_v2.sgen.index) == 0


def test_fullgrid_pwl_cost(fullgrid_v2):
    assert len(fullgrid_v2.pwl_cost.index) == 0


def test_fullgrid_poly_cost(fullgrid_v2):
    assert len(fullgrid_v2.poly_cost.index) == 0


def test_fullgrid_motor(fullgrid_v2):
    assert len(fullgrid_v2.motor.index) == 1
    element_0 = fullgrid_v2.motor[fullgrid_v2.motor['origin_id'] == '_2b618292-5fec-af43-ae39-c32566d0a752']
    assert element_0['name'].item() == 'ASM_1'
    assert fullgrid_v2.bus.iloc[element_0['bus'].item()]['origin_id'] == '_f70f6bad-eb8d-4b8f-8431-4ab93581514e'
    assert math.isnan(element_0['pn_mech_mw'].item())
    assert math.isnan(element_0['loading_percent'].item())
    assert element_0['cos_phi'].item() == pytest.approx(0.9, abs=0.000001)
    assert element_0['cos_phi_n'].item() == pytest.approx(0.9, abs=0.000001)
    assert element_0['efficiency_percent'].item() == pytest.approx(100.0, abs=0.000001)
    assert math.isnan(element_0['efficiency_n_percent'].item())
    assert math.isnan(element_0['lrc_pu'].item())
    assert element_0['vn_kv'].item() == pytest.approx(225.0, abs=0.000001)
    assert element_0['scaling'].item() == pytest.approx(1.0, abs=0.000001)
    assert element_0['in_service'].item()
    assert math.isnan(element_0['rx'].item())
    assert element_0['origin_class'].item() == 'AsynchronousMachine'
    assert element_0['terminal'].item() == '_7b71e695-3977-f544-b31f-777cfbbde49b'


def test_fullgrid_measurement(fullgrid_v2):
    assert len(fullgrid_v2.measurement.index) == 0  # TODO: analogs


def test_fullgrid_load(fullgrid_v2):
    assert len(fullgrid_v2.load.index) == 5
    element_0 = fullgrid_v2.load[fullgrid_v2.load['origin_id'] == '_1324b99a-59ee-0d44-b1f6-15dc0d9d81ff']
    assert element_0['name'].item() == 'BE_CL_1'
    assert fullgrid_v2.bus.iloc[element_0['bus'].item()]['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35'
    assert element_0['p_mw'].item() == pytest.approx(0.010, abs=0.000001)
    assert element_0['q_mvar'].item() == pytest.approx(0.010, abs=0.000001)
    assert element_0['const_z_p_percent'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['const_i_p_percent'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['const_z_q_percent'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['const_i_q_percent'].item() == pytest.approx(0.0, abs=0.000001)
    assert math.isnan(element_0['sn_mva'].item())
    assert element_0['scaling'].item() == pytest.approx(1.0, abs=0.000001)
    assert element_0['in_service'].item()
    assert None is element_0['type'].item()
    assert element_0['origin_class'].item() == 'ConformLoad'
    assert element_0['terminal'].item() == '_84f6ff75-6bf9-8742-ae06-1481aa3b34de'


def test_fullgrid_line_geodata(fullgrid_v2):
    assert 'geo' in fullgrid_v2.line.columns
    assert len(fullgrid_v2.line.loc[fullgrid_v2.line.geo.notna()].index) == 0
    assert 'diagram' not in fullgrid_v2.line.columns


def test_fullgrid_line(fullgrid_v2):
    assert len(fullgrid_v2.line.index) == 11
    element_0 = fullgrid_v2.line[fullgrid_v2.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']
    assert element_0['name'].item() == 'BE-Line_7'
    assert None is element_0['std_type'].item()
    assert fullgrid_v2.bus.iloc[element_0['from_bus'].item()]['origin_id'] == '_1fa19c281c8f4e1eaad9e1cab70f923e'
    assert fullgrid_v2.bus.iloc[element_0['to_bus'].item()]['origin_id'] == '_f70f6bad-eb8d-4b8f-8431-4ab93581514e'
    assert element_0['length_km'].item() == pytest.approx(23.0, abs=0.000001)
    assert element_0['r_ohm_per_km'].item() == pytest.approx(0.19999999999999998, abs=0.000001)
    assert element_0['x_ohm_per_km'].item() == pytest.approx(3.0, abs=0.000001)
    assert element_0['c_nf_per_km'].item() == pytest.approx(3.0000014794808827, abs=0.000001)
    assert element_0['g_us_per_km'].item() == pytest.approx(2.5, abs=0.000001)
    assert element_0['max_i_ka'].item() == pytest.approx(1.0620, abs=0.000001)
    assert element_0['df'].item() == pytest.approx(1.0, abs=0.000001)
    assert element_0['parallel'].item() == pytest.approx(1.0, abs=0.000001)
    assert None is element_0['type'].item()
    assert element_0['in_service'].item()
    assert element_0['origin_class'].item() == 'ACLineSegment'
    assert element_0['terminal_from'].item() == '_57ae9251-c022-4c67-a8eb-611ad54c963c'
    assert element_0['terminal_to'].item() == '_5b2c65b0-68ce-4530-85b7-385346a3b5e1'
    assert math.isnan(element_0['r0_ohm_per_km'].item())
    assert math.isnan(element_0['x0_ohm_per_km'].item())
    assert math.isnan(element_0['c0_nf_per_km'].item())
    assert element_0['g0_us_per_km'].item() == pytest.approx(0.0, abs=0.000001)
    assert math.isnan(element_0['endtemp_degree'].item())

    element_1 = fullgrid_v2.line[fullgrid_v2.line['origin_id'] == '_6052bacf-9eaa-4217-be91-4c7c89e92a52']
    assert math.isnan(element_1['max_i_ka'].item())


def test_fullgrid_impedance(fullgrid_v2):
    assert len(fullgrid_v2.impedance.index) == 1 # TODO: test with more elements
    element_0 = fullgrid_v2.impedance[fullgrid_v2.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']
    assert element_0['name'].item() == 'BE_SC_1'
    assert fullgrid_v2.bus.iloc[element_0['from_bus'].item()]['origin_id'] == '_514fa0d5-a432-5743-8204-1c8518ffed76'
    assert fullgrid_v2.bus.iloc[element_0['to_bus'].item()]['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35'
    assert element_0['rft_pu'].item() == pytest.approx(8.181818181818182e-05, abs=0.000001)
    assert element_0['xft_pu'].item() == pytest.approx(8.181818181818182e-05, abs=0.000001)
    assert element_0['rtf_pu'].item() == pytest.approx(8.181818181818182e-05, abs=0.000001)
    assert element_0['xtf_pu'].item() == pytest.approx(8.181818181818182e-05, abs=0.000001)
    assert element_0['sn_mva'].item() == pytest.approx(1.0, abs=0.000001)
    assert element_0['in_service'].item()
    assert element_0['origin_class'].item() == 'SeriesCompensator'
    assert element_0['terminal_from'].item() == '_0b2c4a73-e4dd-4445-acc3-1284ad5a8a70'
    assert element_0['terminal_to'].item() == '_8c735a96-1b4c-a34d-8823-d6124bd87042'
    assert math.isnan(element_0['rft0_pu'].item())
    assert math.isnan(element_0['xft0_pu'].item())
    assert math.isnan(element_0['rtf0_pu'].item())
    assert math.isnan(element_0['xtf0_pu'].item())


def test_fullgrid_gen(fullgrid_v2):
    assert len(fullgrid_v2.gen.index) == 8
    element_0 = fullgrid_v2.gen[fullgrid_v2.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']
    assert element_0['name'].item() == 'BE-G5'
    assert fullgrid_v2.bus.iloc[element_0['bus'].item()]['origin_id'] == '_f96d552a-618d-4d0c-a39a-2dea3c411dee'
    assert element_0['p_mw'].item() == pytest.approx(118.0, abs=0.000001)
    assert element_0['vm_pu'].item() == pytest.approx(1.04700, abs=0.000001)
    assert element_0['sn_mva'].item() == pytest.approx(300.0, abs=0.000001)
    assert element_0['min_q_mvar'].item() == pytest.approx(-200.0, abs=0.000001)
    assert element_0['max_q_mvar'].item() == pytest.approx(200.0, abs=0.000001)
    assert element_0['scaling'].item() == pytest.approx(1.0, abs=0.000001)
    assert not element_0['slack'].item()
    assert element_0['in_service'].item()
    assert math.isnan(element_0['slack_weight'].item())
    assert element_0['type'].item() == 'Nuclear'
    assert element_0['origin_class'].item() == 'SynchronousMachine'
    assert element_0['terminal'].item() == '_b2dcbf07-4676-774f-ae35-86c1ab695de0'
    assert element_0['min_p_mw'].item() == pytest.approx(50.0, abs=0.000001)
    assert element_0['max_p_mw'].item() == 200
    assert element_0['vn_kv'].item() == pytest.approx(21.0, abs=0.000001)
    assert math.isnan(element_0['rdss_ohm'].item())
    assert math.isnan(element_0['xdss_pu'].item())
    assert element_0['cos_phi'].item() == pytest.approx(0.850, abs=0.000001)
    assert element_0['pg_percent'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['reactive_capability_curve'].item()
    assert element_0['id_q_capability_characteristic'].item() == 0
    assert element_0['curve_style'].item() == 'straightLineYValues'

    element_1 = fullgrid_v2.gen[fullgrid_v2.gen['origin_id'] == '_3a3b27be-b18b-4385-b557-6735d733baf0']
    assert element_1['vm_pu'].item() == pytest.approx(1.050, abs=0.000001)


def test_full_grid_q_capability_table(fullgrid_v2):
    capa_df = pd.DataFrame({'id_q_capability_curve': {0: 0, 1: 0, 2: 0},
                            'p_mw': {0: -100.0, 1: 0.0, 2: 100.0},
                            'q_min_mvar': {0: -200.0, 1: -300.0, 2: -200.0},
                            'q_max_mvar': {0: 200.0, 1: 300.0, 2: 200.0}})
    pd.testing.assert_frame_equal(fullgrid_v2['q_capability_curve_table'], capa_df, atol=1e-5)


def test_fullgrid_ext_grid(fullgrid_v2):
    assert len(fullgrid_v2.ext_grid.index) == 1
    element_0 = fullgrid_v2.ext_grid[fullgrid_v2.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']
    assert element_0['name'].item() == 'ES_1'
    assert fullgrid_v2.bus.iloc[element_0['bus'].item()]['origin_id'] == '_f70f6bad-eb8d-4b8f-8431-4ab93581514e'
    assert element_0['vm_pu'].item() == 1
    assert element_0['va_degree'].item() == 0
    assert math.isnan(element_0['slack_weight'].item())
    assert element_0['in_service'].item()
    assert element_0['origin_class'].item() == 'EnergySource'
    assert element_0['terminal'].item() == '_9835652b-053f-cb44-822e-1e26950d989c'
    assert math.isnan(element_0['substation'].item())
    assert math.isnan(element_0['min_p_mw'].item())
    assert math.isnan(element_0['max_p_mw'].item())
    assert math.isnan(element_0['min_q_mvar'].item())
    assert math.isnan(element_0['max_q_mvar'].item())
    assert element_0['p_mw'].item() == pytest.approx(-9.99000, abs=0.000001)
    assert element_0['q_mvar'].item() == pytest.approx(-0.99000, abs=0.000001)
    assert math.isnan(element_0['s_sc_max_mva'].item())
    assert math.isnan(element_0['s_sc_min_mva'].item())
    assert math.isnan(element_0['rx_max'].item())
    assert math.isnan(element_0['rx_min'].item())
    assert math.isnan(element_0['r0x0_max'].item())
    assert math.isnan(element_0['x0x_max'].item())


def test_fullgrid_dcline(fullgrid_v2):
    assert len(fullgrid_v2.dcline.index) == 2
    element_0 = fullgrid_v2.dcline[fullgrid_v2.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']
    assert element_0['name'].item() == 'LDC-1230816355'
    assert element_0['description'].item() == 'LDC-1230816355'
    assert fullgrid_v2.bus.iloc[element_0['from_bus'].item()]['origin_id'] == '_27d57afa-6c9d-4b06-93ea-8c88d14af8b1'
    assert fullgrid_v2.bus.iloc[int(element_0['to_bus'].item())]['origin_id'] == '_d3d9c515-2ddb-436a-bf17-2f8be2394de3'
    assert element_0['p_mw'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['loss_percent'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['loss_mw'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['vm_from_pu'].item() == pytest.approx(1.0, abs=0.000001)
    assert element_0['vm_to_pu'].item() == pytest.approx(1.0, abs=0.000001)
    assert math.isnan(element_0['max_p_mw'].item())
    assert math.isnan(element_0['min_q_from_mvar'].item())
    assert math.isnan(element_0['min_q_to_mvar'].item())
    assert math.isnan(element_0['max_q_from_mvar'].item())
    assert math.isnan(element_0['max_q_to_mvar'].item())
    assert element_0['in_service'].item()
    assert element_0['origin_class'].item() == 'DCLineSegment'
    assert element_0['terminal_from'].item() == '_4123e718-716a-4988-bf71-0e525a4422f2'
    assert element_0['terminal_to'].item() == '_c4c335b5-0405-4539-be10-697f5a3f3e83'

    element_0 = fullgrid_v2.dcline[fullgrid_v2.dcline['origin_id'] == '_d7693c6d-58bd-49da-bb24-973a63f9faf1']
    assert math.isnan(element_0['to_bus'].item())
    assert math.isnan(element_0['loss_mw'].item())
    assert math.isnan(element_0['vm_to_pu'].item())
    assert pd.isna(element_0['in_service'].item())
    assert math.isnan(element_0['terminal_to'].item())


def test_fullgrid_controller(fullgrid_v2):
    assert len(fullgrid_v2.controller.index) == 8
    for _, obj in fullgrid_v2.controller.iterrows():
        if obj.object.matching_params.get('tid') == \
                fullgrid_v2.trafo[fullgrid_v2.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c'].index:
            assert not obj['in_service']
            assert obj['order'] == pytest.approx(0.0, abs=0.000001)
            assert obj['level'] == 0
            assert obj['initial_run']
            assert not obj['recycle'].get('bus_pq')
            assert not obj['recycle'].get('gen')
            assert obj['recycle'].get('trafo')
            assert obj.object.controlled_bus == 11
            assert obj.object.controlled_tid == 0
            assert obj.object.side == 'lv'
            assert obj.object.tap_max == pytest.approx(15.0, abs=0.000001)
            assert obj.object.tap_min == pytest.approx(-15.0, abs=0.000001)
            assert obj.object.tap_neutral == pytest.approx(0.0, abs=0.000001)
            assert obj.object.tap_pos == pytest.approx(-2.0, abs=0.000001)
            assert obj.object.tap_side_coeff == 1
            assert obj.object.tap_sign == 1
            assert math.isnan(obj.object.tap_step_degree)
            assert obj.object.tap_step_percent == pytest.approx(1.25, abs=0.000001)
            assert obj.object.tid == 0
            assert math.isnan(obj.object.tol)
            assert fullgrid_v2.bus.iloc[obj.object.trafobus]['origin_id'] == '_b01fe92f-68ab-4123-ae45-f22d3e8daad1'
            assert obj.object.trafotable == 'trafo'
            assert obj.object.trafotype == '2W'
            assert math.isnan(obj.object.vm_delta_pu)
            assert math.isnan(obj.object.vm_lower_pu)
            assert None is obj.object.vm_set_pu
            assert math.isnan(obj.object.vm_upper_pu)
        if obj.object.matching_params.get('tid') == \
                fullgrid_v2.trafo[fullgrid_v2.trafo['origin_id'] == '_69a301e8-f6b2-47ad-9f65-52f2eabc9917'].index:
            assert obj['in_service']
            assert obj['order'] == pytest.approx(0.0, abs=0.000001)
            assert obj['level'] == 0
            assert obj['initial_run']
            assert not obj['recycle'].get('bus_pq')
            assert not obj['recycle'].get('gen')
            assert obj['recycle'].get('trafo')
            assert obj.object.controlled_bus == 10
            assert obj.object.controlled_tid == 3
            assert obj.object.side == 'lv'
            assert obj.object.tap_max == pytest.approx(10.0, abs=0.000001)
            assert obj.object.tap_min == pytest.approx(-10.0, abs=0.000001)
            assert obj.object.tap_neutral == pytest.approx(0.0, abs=0.000001)
            assert obj.object.tap_pos == pytest.approx(0.0, abs=0.000001)
            assert obj.object.tap_side_coeff == 1
            assert obj.object.tap_sign == 1
            assert math.isnan(obj.object.tap_step_degree)
            assert obj.object.tap_step_percent == pytest.approx(1.25, abs=0.000001)
            assert obj.object.tid == 3
            assert math.isnan(obj.object.tol)
            assert fullgrid_v2.bus.iloc[obj.object.trafobus]['origin_id'] == '_ac772dd8-7910-443f-8af0-a7fca0fb57f9'
            assert obj.object.trafotable == 'trafo'
            assert obj.object.trafotype == '2W'
            assert math.isnan(obj.object.vm_delta_pu)
            assert math.isnan(obj.object.vm_lower_pu)
            assert None is obj.object.vm_set_pu
            assert math.isnan(obj.object.vm_upper_pu)


def test_fullgrid_trafo_characteristic_table(fullgrid_v2):
    assert len(fullgrid_v2.trafo_characteristic_table.index) == 8


def test_fullgrid_bus_geodata(fullgrid_v2):
    assert 'geo' in fullgrid_v2.bus.columns
    assert len(fullgrid_v2.bus.loc[fullgrid_v2.bus.geo.notna()].index) == 0
    assert 'diagram' not in fullgrid_v2.bus.columns


def test_fullgrid_bus(fullgrid_v2):
    assert len(fullgrid_v2.bus.index) == 26
    element_0 = fullgrid_v2.bus[fullgrid_v2.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']
    assert element_0['name'].item() == 'BE-Busbar_7'
    assert element_0['vn_kv'].item() == pytest.approx(110.00000, abs=0.000001)
    assert element_0['type'].item() == 'n'
    assert element_0['zone'].item() == 'PP_Brussels'
    assert element_0['in_service'].item()
    assert element_0['origin_class'].item() == 'TopologicalNode'
    assert element_0['origin_profile'].item() == 'tp'
    assert element_0['cim_topnode'].item() == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35'
    assert element_0['ConnectivityNodeContainer_id'].item() == '_8bbd7e74-ae20-4dce-8780-c20f8e18c2e0'
    assert element_0['Substation_id'].item() == '_37e14a0f-5e34-4647-a062-8bfd9305fa9d'
    assert element_0['GeographicalRegion_id'].item() == '_c1d5bfc68f8011e08e4d00247eb1f55e'
    assert element_0['GeographicalRegion_name'].item() == 'BE'
    assert element_0['SubGeographicalRegion_id'].item() == '_c1d5bfc88f8011e08e4d00247eb1f55e'
    assert element_0['SubGeographicalRegion_name'].item() == 'ELIA-Brussels'
    assert math.isnan(element_0['Busbar_id'].item())
    assert math.isnan(element_0['Busbar_name'].item())
    assert element_0['description'].item() == 'BBRUS151; BGENT_51'

    element_1 = fullgrid_v2.bus[fullgrid_v2.bus['origin_id'] == '_1098b1c9-dc85-40ce-b65c-39ae02a3afaa']
    assert math.isnan(element_1['cim_topnode'].item())
    assert math.isnan(element_1['description'].item())

    element_2 = fullgrid_v2.bus[fullgrid_v2.bus['origin_id'] == '_99b219f3-4593-428b-a4da-124a54630178']
    assert element_2['zone'].item() == 'PP_Brussels'
    assert math.isnan(element_2['geo'].item())
    assert element_2['cim_topnode'].item() == '_99b219f3-4593-428b-a4da-124a54630178'
    assert element_2['ConnectivityNodeContainer_id'].item() == '_b10b171b-3bc5-4849-bb1f-61ed9ea1ec7c'
    assert element_2['Substation_id'].item() == '_37e14a0f-5e34-4647-a062-8bfd9305fa9d'
    assert element_2['description'].item() == 'BE_TR_BUS4'
    assert element_2['Busbar_id'].item() == '_5caf27ed-d2f8-458a-834a-6b3193a982e6'
    assert element_2['Busbar_name'].item() == 'BE-Busbar_3'
    assert element_2['GeographicalRegion_id'].item() == '_c1d5bfc68f8011e08e4d00247eb1f55e'
    assert element_2['GeographicalRegion_name'].item() == 'BE'
    assert element_2['SubGeographicalRegion_id'].item() == '_c1d5bfc88f8011e08e4d00247eb1f55e'
    assert element_2['SubGeographicalRegion_name'].item() == 'ELIA-Brussels'


def test_fullgrid_asymmetric_sgen(fullgrid_v2):
    assert len(fullgrid_v2.asymmetric_sgen.index) == 0


def test_fullgrid_asymmetric_load(fullgrid_v2):
    assert len(fullgrid_v2.asymmetric_load.index) == 0


def test_fullgrid_NB_bus(fullgrid_node_breaker):
    assert len(fullgrid_node_breaker.bus.index) == 40
    element_0 = fullgrid_node_breaker.bus[fullgrid_node_breaker.bus['origin_id'] == '_ec6b1f37-6c5a-ac43-a366-019f5bcce2b1']
    assert element_0['name'].item() == 'BB_Disconector_5'
    assert element_0['vn_kv'].item() == pytest.approx(110.00000, abs=0.000001)
    assert element_0['type'].item() == 'n'
    assert element_0['zone'].item() == 'PP_Brussels'
    assert element_0['in_service'].item()
    assert element_0['origin_class'].item() == 'ConnectivityNode'
    assert element_0['origin_profile'].item() == 'eq'
    assert element_0['cim_topnode'].item() == '_5c74cb26-ce2f-40c6-951d-89091eb781b6'
    assert element_0['ConnectivityNodeContainer_id'].item() == '_dfa04cac-2b1c-2d4a-b981-ccc03193809f'
    assert element_0['Substation_id'].item() == '_37e14a0f-5e34-4647-a062-8bfd9305fa9d'
    assert element_0['GeographicalRegion_id'].item() == '_c1d5bfc68f8011e08e4d00247eb1f55e'
    assert element_0['GeographicalRegion_name'].item() == 'BE'
    assert element_0['SubGeographicalRegion_id'].item() == '_c1d5bfc88f8011e08e4d00247eb1f55e'
    assert element_0['SubGeographicalRegion_name'].item() == 'ELIA-Brussels'
    assert math.isnan(element_0['Busbar_id'].item())
    assert math.isnan(element_0['Busbar_name'].item())
    assert element_0['description'].item() == 'BB_Disconector_5'

    element_1 = fullgrid_node_breaker.bus[fullgrid_node_breaker.bus['origin_id'] == '_4836f99b-c6e9-4ee8-a956-b1e3da882d46']
    assert element_1['Busbar_id'].item() == '_64901aec-5a8a-4bcb-8ca7-a3ddbfcd0e6c'
    assert element_1['Busbar_name'].item() == 'BE-Busbar_1'

    element_2 = fullgrid_node_breaker.bus[fullgrid_node_breaker.bus['origin_id'] == '_c38adab3-5168-4004-a83d-28d890dedd36']
    assert element_2['zone'].item() == 'HVDC 1'
    assert math.isnan(element_2['geo'].item())
    assert element_2['cim_topnode'].item() == '_b01fe92f-68ab-4123-ae45-f22d3e8daad1'
    assert element_2['ConnectivityNodeContainer_id'].item() == '_c68f0a24-46cb-42aa-b91d-0b49b8310cc9'
    assert element_2['Substation_id'].item() == '_9df6213f-c5dc-477c-aab4-74721f7d1fdb'
    assert element_2['description'].item() == 'HVDC1_BB'
    assert math.isnan(element_2['Busbar_id'].item())
    assert math.isnan(element_2['Busbar_name'].item())
    assert element_2['GeographicalRegion_id'].item() == '_c1d5bfc68f8011e08e4d00247eb1f55e'
    assert element_2['GeographicalRegion_name'].item() == 'BE'
    assert element_2['SubGeographicalRegion_id'].item() == '_0296d175-5169-48b8-b96a-1cb90f56fe21'
    assert element_2['SubGeographicalRegion_name'].item() == 'HVDC Zone'


def test_fullgrid_NB_switch(fullgrid_node_breaker):
    assert len(fullgrid_node_breaker.switch) == 24

    assert len(fullgrid_node_breaker.switch['closed']) == 24 # all are closed

    element_0 = fullgrid_node_breaker.switch[fullgrid_node_breaker.switch['origin_id'] == '_8a3ad6e1-6e23-b649-880e-4865217501c4']
    assert fullgrid_node_breaker.bus.iloc[element_0['bus'].item()]['origin_id'] == '_ec6b1f37-6c5a-ac43-a366-019f5bcce2b1'
    assert fullgrid_node_breaker.bus.iloc[element_0['element'].item()]['origin_id'] == '_3293fcc7-4962-47df-a7c1-ce150600c388'
    assert element_0['et'].item() == 'b'
    assert element_0['type'].item() == 'DS'
    assert element_0['closed'].item()
    assert element_0['name'].item() == 'BE_DSC_5'
    assert element_0['z_ohm'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_0['in_ka'].item() == pytest.approx(0.09999, abs=0.000001)
    assert element_0['origin_class'].item() == 'Disconnector'
    assert element_0['terminal_bus'].item() == '_2af7ad2c-062c-1c4f-be3e-9c7cd594ddbb'
    assert element_0['terminal_element'].item() == '_916578a1-7a6e-7347-a5e0-aaf35538949c'
    assert element_0['description'].item() == 'BE_DSC_5'

    element_1 = fullgrid_node_breaker.switch[fullgrid_node_breaker.switch['origin_id'] == '_ac77624b-0a92-4a49-bb93-02b131e8857c']
    assert fullgrid_node_breaker.bus.iloc[element_1['bus'].item()]['origin_id'] == '_1695eb20-9044-4133-a3fd-2147f55f170d'
    assert fullgrid_node_breaker.bus.iloc[element_1['element'].item()]['origin_id'] == '_0afe3c6b-c8b5-d946-b05c-e4a8e00a5e6d'
    assert element_1['et'].item() == 'b'
    assert element_1['type'].item() == 'LBS'
    assert element_1['closed'].item()
    assert element_1['name'].item() == 'BE_LB_1'
    assert element_1['z_ohm'].item() == pytest.approx(0.0, abs=0.000001)
    assert element_1['in_ka'].item() == pytest.approx(0.09999, abs=0.000001)
    assert element_1['origin_class'].item() == 'LoadBreakSwitch'
    assert element_1['terminal_bus'].item() == '_1c134839-5bad-124e-93a4-b11663025232'
    assert element_1['terminal_element'].item() == '_ea6bb748-b513-0947-a59b-abd50155dad2'
    assert element_1['description'].item() == 'BE_LB_1'

if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
