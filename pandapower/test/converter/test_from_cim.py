import os
import pytest
import math
import pandas as pd

import pandapower as pp
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

    return cim2pp.from_cim(file_list=cgmes_files, run_powerflow=True)


@pytest.fixture(scope="session")
def Simbench_1_EHV_mixed__2_no_sw():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'Simbench_1-EHV-mixed--2-no_sw.zip')]

    return cim2pp.from_cim(file_list=cgmes_files, create_measurements='SV', run_powerflow=True)


@pytest.fixture(scope="session")
def example_multivoltage():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'example_multivoltage.zip')]

    net = cim2pp.from_cim(file_list=cgmes_files)
    pp.runpp(net, calculate_voltage_angles="auto")
    return net


@pytest.fixture(scope="session")
def SimBench_1_HVMVmixed_1_105_0_sw_modified_no_load_flow():
    folder_path = os.path.join(test_path, "test_files", "example_cim")

    cgmes_files = [os.path.join(folder_path, 'SimBench_1-HVMV-mixed-1.105-0-sw_modified.zip')]

    return cim2pp.from_cim(file_list=cgmes_files)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_no_load_flow_res_bus(
        SimBench_1_HVMVmixed_1_105_0_sw_modified_no_load_flow):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified_no_load_flow.res_bus.index)


def test_example_multivoltage_res_xward(example_multivoltage):
    assert 2 == len(example_multivoltage.res_xward.index)
    element_0 = example_multivoltage.res_xward.iloc[example_multivoltage.xward[
        example_multivoltage.xward['origin_id'] == '_78c751ae-91b7-4d81-8732-670085cf8e94'].index]
    assert pytest.approx(23.941999999999194, abs=0.0001) == element_0['p_mw'].item()
    assert pytest.approx(-13.126152038621285, abs=0.0001) == element_0['q_mvar'].item()
    assert pytest.approx(1.0261528781410867, abs=0.000001) == element_0['vm_pu'].item()
    assert pytest.approx(-3.1471612276732728, abs=0.0001) == element_0['va_internal_degree'].item()
    assert pytest.approx(1.02616, abs=0.000001) == element_0['vm_internal_pu'].item()


def test_example_multivoltage_res_trafo3w(example_multivoltage):
    assert 1 == len(example_multivoltage.res_trafo3w)
    element_0 = example_multivoltage.res_trafo3w.iloc[example_multivoltage.trafo3w[
        example_multivoltage.trafo3w['origin_id'] == '_094b399b-84bb-4c0c-9160-f322ba106b99'].index]
    assert pytest.approx(8.657017822871058, abs=0.0001) == element_0['p_hv_mw'].item()
    assert pytest.approx(-2.9999999999999982, abs=0.0001) == element_0['p_mv_mw'].item()
    assert pytest.approx(-5.651309159685888, abs=0.0001) == element_0['p_lv_mw'].item()
    assert pytest.approx(3.8347346868849734, abs=0.0001) == element_0['q_hv_mvar'].item()
    assert pytest.approx(-0.9999999999999539, abs=0.0001) == element_0['q_mv_mvar'].item()
    assert pytest.approx(-2.5351958737584237, abs=0.0001) == element_0['q_lv_mvar'].item()
    assert pytest.approx(0.005708663185171936, abs=0.0001) == element_0['pl_mw'].item()
    assert pytest.approx(0.2995388131265959, abs=0.0001) == element_0['ql_mvar'].item()
    assert pytest.approx(0.048934706145584116, abs=0.0001) == element_0['i_hv_ka'].item()
    assert pytest.approx(0.09108067515916406, abs=0.0001) == element_0['i_mv_ka'].item()
    assert pytest.approx(0.35669358848757043, abs=0.0001) == element_0['i_lv_ka'].item()
    assert pytest.approx(1.015553448688208, abs=0.000001) == element_0['vm_hv_pu'].item()
    assert pytest.approx(1.0022663178330908, abs=0.000001) == element_0['vm_mv_pu'].item()
    assert pytest.approx(1.0025566368417127, abs=0.000001) == element_0['vm_lv_pu'].item()
    assert pytest.approx(-4.1106386368211005, abs=0.0001) == element_0['va_hv_degree'].item()
    assert pytest.approx(-5.8681443686929216, abs=0.0001) == element_0['va_mv_degree'].item()
    assert pytest.approx(-5.729162028267157, abs=0.0001) == element_0['va_lv_degree'].item()
    assert pytest.approx(-5.071099920236481, abs=0.0001) == element_0['va_internal_degree'].item()
    assert pytest.approx(1.0073406631833621, abs=0.000001) == element_0['vm_internal_pu'].item()
    assert pytest.approx(24.712456719781486, abs=0.0001) == element_0['loading_percent'].item()


def test_example_multivoltage_xward(example_multivoltage):
    assert 2 == len(example_multivoltage.xward.index)
    element_0 = example_multivoltage.xward[
        example_multivoltage.xward['origin_id'] == '_78c751ae-91b7-4d81-8732-670085cf8e94']
    assert 'XWard 1' == element_0['name'].item()
    assert '_7783eaac-d397-4b74-9b20-4d10e5861213' == example_multivoltage.bus.iloc[
        element_0['bus'].item()]['origin_id']
    assert 23.942 == element_0['ps_mw'].item()
    assert -12.24187 == element_0['qs_mvar'].item()
    assert 0.0 == element_0['qz_mvar'].item()
    assert 0.0 == element_0['pz_mw'].item()
    assert 0.0 == element_0['r_ohm'].item()
    assert 0.1 == element_0['x_ohm'].item()
    assert 1.02616 == element_0['vm_pu'].item()
    assert math.isnan(element_0['slack_weight'].item())
    assert element_0['in_service'].item()
    assert 'EquivalentInjection' == element_0['origin_class'].item()
    assert '_044c980b-210d-4ec0-8a99-245dcf1ae5c5' == element_0['terminal'].item()


def test_Simbench_1_EHV_mixed__2_no_sw_res_gen(Simbench_1_EHV_mixed__2_no_sw):
    assert 338 == len(Simbench_1_EHV_mixed__2_no_sw.res_gen.index)
    element_0 = Simbench_1_EHV_mixed__2_no_sw.res_gen.iloc[Simbench_1_EHV_mixed__2_no_sw.gen[
        Simbench_1_EHV_mixed__2_no_sw.gen['origin_id'] == '_5b01c8ba-9847-49bc-a1f2-0100ccf7df74'].index]
    assert pytest.approx(297.0, abs=0.0001) == element_0['p_mw'].item()
    assert pytest.approx(-5.003710432303805, abs=0.0001) == element_0['q_mvar'].item()
    assert pytest.approx(34.87201407457379, abs=0.0001) == element_0['va_degree'].item()
    assert pytest.approx(1.0920, abs=0.000001) == element_0['vm_pu'].item()

    element_1 = Simbench_1_EHV_mixed__2_no_sw.res_gen.iloc[Simbench_1_EHV_mixed__2_no_sw.gen[
        Simbench_1_EHV_mixed__2_no_sw.gen['origin_id'] == '_4f01682d-ee27-4f5e-b073-bdc90431d61b'].index]
    assert pytest.approx(604.0, abs=0.0001) == element_1['p_mw'].item()
    assert pytest.approx(-37.964890640820215, abs=0.0001) == element_1['q_mvar'].item()
    assert pytest.approx(37.18871679516421, abs=0.0001) == element_1['va_degree'].item()
    assert pytest.approx(1.0679999999999996, abs=0.000001) == element_1['vm_pu'].item()


def test_Simbench_1_EHV_mixed__2_no_sw_res_dcline(Simbench_1_EHV_mixed__2_no_sw):
    assert 6 == len(Simbench_1_EHV_mixed__2_no_sw.res_dcline.index)
    element_0 = Simbench_1_EHV_mixed__2_no_sw.res_dcline.iloc[Simbench_1_EHV_mixed__2_no_sw.dcline[
        Simbench_1_EHV_mixed__2_no_sw.dcline['origin_id'] == '_ee269f63-6d79-4089-923d-a3a0ee080f92'].index]
    assert pytest.approx(0.0, abs=0.0001) == element_0['p_from_mw'].item()
    assert pytest.approx(-297.4030725955963, abs=0.0001) == element_0['q_from_mvar'].item()
    assert pytest.approx(0.0, abs=0.0001) == element_0['p_to_mw'].item()
    assert pytest.approx(-124.6603285074234, abs=0.0001) == element_0['q_to_mvar'].item()
    assert pytest.approx(0.0, abs=0.0001) == element_0['pl_mw'].item()
    assert pytest.approx(1.0680, abs=0.000001) == element_0['vm_from_pu'].item()
    assert pytest.approx(30.248519550084087, abs=0.0001) == element_0['va_from_degree'].item()
    assert pytest.approx(1.0680, abs=0.000001) == element_0['vm_to_pu'].item()
    assert pytest.approx(33.3178813629858, abs=0.0001) == element_0['va_to_degree'].item()


def test_Simbench_1_EHV_mixed__2_no_sw_measurement(Simbench_1_EHV_mixed__2_no_sw):
    assert 571 == len(Simbench_1_EHV_mixed__2_no_sw.measurement.index)
    element_0 = Simbench_1_EHV_mixed__2_no_sw.measurement[
        Simbench_1_EHV_mixed__2_no_sw.measurement['element'] ==
        Simbench_1_EHV_mixed__2_no_sw.bus[Simbench_1_EHV_mixed__2_no_sw.bus[
                                              'origin_id'] == '_1cdc1d88-56de-465b-b1a0-968722f2b287'].index[0]]
    assert 'EHV Bus 1' == element_0['name'].item()
    assert 'v' == element_0['measurement_type'].item()
    assert 'bus' == element_0['element_type'].item()
    assert 1.0920 == element_0['value'].item()
    assert 0.001092 == element_0['std_dev'].item()
    assert None is element_0['side'].item()


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_xward(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_xward.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_ward(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_ward.index)  # TODO:


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_trafo_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_trafo_sc.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_trafo_est(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_trafo_est.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_trafo_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_trafo_3ph.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_trafo3w_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_trafo3w_sc.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_trafo3w_est(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_trafo3w_est.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_trafo3w(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_trafo3w.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_trafo(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 8 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_trafo.index)
    element_0 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_trafo.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.trafo[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.trafo['origin_id'] == '_0e8aabc2-5ddb-4946-9ced-9e1eb20881e9'].index]
    assert pytest.approx(-128.5650471085923, abs=0.0001) == element_0['p_hv_mw'].item()
    assert pytest.approx(67.3853631948281, abs=0.0001) == element_0['q_hv_mvar'].item()
    assert pytest.approx(128.83323218784219, abs=0.0001) == element_0['p_lv_mw'].item()
    assert pytest.approx(-56.07814415772255, abs=0.0001) == element_0['q_lv_mvar'].item()
    assert pytest.approx(0.2681850792498892, abs=0.0001) == element_0['pl_mw'].item()
    assert pytest.approx(11.307219037105554, abs=0.0001) == element_0['ql_mvar'].item()
    assert pytest.approx(0.2019588628357026, abs=0.0001) == element_0['i_hv_ka'].item()
    assert pytest.approx(0.6978657735602077, abs=0.0001) == element_0['i_lv_ka'].item()
    assert pytest.approx(1.092, abs=0.000001) == element_0['vm_hv_pu'].item()
    assert pytest.approx(0.0, abs=0.0001) == element_0['va_hv_degree'].item()
    assert pytest.approx(1.0567657300787896, abs=0.000001) == element_0['vm_lv_pu'].item()
    assert pytest.approx(4.042276580597498, abs=0.0001) == element_0['va_lv_degree'].item()
    assert pytest.approx(37.98893926676003, abs=0.0001) == element_0['loading_percent'].item()

    assert pytest.approx(4.683488464449254, abs=0.0001) == SimBench_1_HVMVmixed_1_105_0_sw_modified.res_trafo.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.trafo[SimBench_1_HVMVmixed_1_105_0_sw_modified.trafo['origin_id'] == '_97701d2b-79ef-4fcc-958b-c6628e065798'].index]['va_hv_degree'].item()


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_tcsc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_tcsc.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_switch_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_switch_sc.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_switch_est(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_switch_est.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_switch(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 625 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_switch)  # TODO: test with different net with better values
    element_0 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_switch.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.switch[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.switch['origin_id'] == '_004e90a8-cc0d-43f2-a9eb-c374f9479e8f'].index]
    assert math.isnan(element_0['i_ka'].item())
    assert math.isnan(element_0['loading_percent'].item())

    element_1 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_switch.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.switch[
            SimBench_1_HVMVmixed_1_105_0_sw_modified.switch[
                'origin_id'] == '_00bf6ec7-fb79-4a7b-a3d2-ea35490b067c'].index]
    assert pytest.approx(0.0, abs=0.0001) == element_1['i_ka'].item()


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_svc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_svc.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_storage_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_storage_3ph.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_storage(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_storage.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_shunt_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_shunt_3ph.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_shunt(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_shunt.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_sgen_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_sgen_sc.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_sgen_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_sgen_3ph.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_sgen(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 205 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_sgen.index)
    element_0 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_sgen.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.sgen[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.sgen['origin_id'] == '_24a34602-5f03-4293-9e5b-bb95201356c6'].index]
    assert pytest.approx(2.0, abs=0.0001) == element_0['p_mw'].item()
    assert pytest.approx(0.0, abs=0.0001) == element_0['q_mvar'].item()

    element_1 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_sgen.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.sgen[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.sgen['origin_id'] == '_00d5cff1-01a8-4193-92c6-0e16088efe9b'].index]
    assert pytest.approx(149.060, abs=0.0001) == element_1['p_mw'].item()
    assert pytest.approx(0.0, abs=0.0001) == element_1['q_mvar'].item()

    element_2 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_sgen.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.sgen[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.sgen['origin_id'] == '_5527d842-0d46-44c6-94ed-f5772fab1393'].index]
    assert pytest.approx(0.1450, abs=0.0001) == element_2['p_mw'].item()
    assert pytest.approx(0.0, abs=0.0001) == element_2['q_mvar'].item()


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_motor(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_motor.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_load_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_load_3ph.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_load(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 154 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_load.index)
    element_0 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_load.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.load[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.load['origin_id'] == '_6bcbb329-364a-461e-bdd0-aef5fb25947a'].index]
    assert pytest.approx(0.230, abs=0.0001) == element_0['p_mw'].item()
    assert pytest.approx(0.09090, abs=0.0001) == element_0['q_mvar'].item()

    element_1 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_load.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.load[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.load['origin_id'] == '_dcad3c20-024f-4234-83aa-67f03745cec4'].index]
    assert pytest.approx(3.0, abs=0.0001) == element_1['p_mw'].item()
    assert pytest.approx(1.1860, abs=0.0001) == element_1['q_mvar'].item()

    element_2 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_load.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.load[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.load['origin_id'] == '_6184b5be-a494-4ae9-ab24-528a33b19a41'].index]
    assert pytest.approx(34.480, abs=0.0001) == element_2['p_mw'].item()
    assert pytest.approx(13.6270, abs=0.0001) == element_2['q_mvar'].item()


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_line_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line_sc.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_line_est(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line_est.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_line_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line_3ph.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_line(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 194 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.index)
    element_0 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.line[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line['origin_id'] == '_9c2727d3-0232-4352-ac78-b1e4ff562d85'].index]
    assert pytest.approx(0.32908338928873615, abs=0.0001) == element_0['p_from_mw'].item()
    assert pytest.approx(0.2780429963361799, abs=0.0001) == element_0['q_from_mvar'].item()
    assert pytest.approx(-0.32900641703695693, abs=0.0001) == element_0['p_to_mw'].item()
    assert pytest.approx(-0.2883800215714708, abs=0.0001) == element_0['q_to_mvar'].item()
    assert pytest.approx(7.697225177921707e-05, abs=0.0001) == element_0['pl_mw'].item()
    assert pytest.approx(-0.010337025235290898, abs=0.0001) == element_0['ql_mvar'].item()
    assert pytest.approx(0.011939852254315474, abs=0.0001) == element_0['i_from_ka'].item()
    assert pytest.approx(0.012127162630402383, abs=0.0001) == element_0['i_to_ka'].item()
    assert pytest.approx(0.012127162630402383, abs=0.0001) == element_0['i_ka'].item()
    assert pytest.approx(1.0416068764183433, abs=0.000001) == element_0['vm_from_pu'].item()
    assert pytest.approx(-143.8090530125839, abs=0.0001) == element_0['va_from_degree'].item()
    assert pytest.approx(1.0414310265920257, abs=0.000001) == element_0['vm_to_pu'].item()
    assert pytest.approx(-143.80472033332924, abs=0.0001) == element_0['va_to_degree'].item()
    assert pytest.approx(5.512346650182902, abs=0.0001) == element_0['loading_percent'].item()

    element_1 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_line.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.line[
            SimBench_1_HVMVmixed_1_105_0_sw_modified.line[
                'origin_id'] == '_00279ee7-01a9-4f5e-a99a-6c02f7bebddb'].index]
    assert pytest.approx(7.351456938401352, abs=0.0001) == element_1['p_from_mw'].item()
    assert pytest.approx(-3.803087780045516, abs=0.0001) == element_1['q_from_mvar'].item()
    assert pytest.approx(-7.343577009323737, abs=0.0001) == element_1['p_to_mw'].item()
    assert pytest.approx(3.2534705172294767, abs=0.0001) == element_1['q_to_mvar'].item()
    assert pytest.approx(0.007879929077614811, abs=0.0001) == element_1['pl_mw'].item()
    assert pytest.approx(-0.5496172628160392, abs=0.0001) == element_1['ql_mvar'].item()
    assert pytest.approx(0.04090204224410598, abs=0.0001) == element_1['i_from_ka'].item()
    assert pytest.approx(0.03968146590932959, abs=0.0001) == element_1['i_to_ka'].item()
    assert pytest.approx(0.04090204224410598, abs=0.0001) == element_1['i_ka'].item()
    assert pytest.approx(1.0621122647814087, abs=0.000001) == element_1['vm_from_pu'].item()
    assert pytest.approx(7.268678685699929, abs=0.0001) == element_1['va_from_degree'].item()
    assert pytest.approx(1.0623882310901724, abs=0.000001) == element_1['vm_to_pu'].item()
    assert pytest.approx(7.109722073381289, abs=0.0001) == element_1['va_to_degree'].item()
    assert pytest.approx(6.015006212368526, abs=0.0001) == element_1['loading_percent'].item()


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_impedance_est(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_impedance_est.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_impedance(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_impedance.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_gen_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_gen_sc.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_gen(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_gen.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_ext_grid_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_ext_grid_sc.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_ext_grid_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_ext_grid_3ph.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_ext_grid(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 3 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_ext_grid.index)
    element_0 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_ext_grid.iloc[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid[
            SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid[
                'origin_id'] == '_5fb17d90-f222-45a2-9e26-05f52a34731f'].index]
    assert pytest.approx(-257.1300942171846, abs=0.0001) == element_0['p_mw'].item()
    assert pytest.approx(134.7707263896562, abs=0.0001) == element_0['q_mvar'].item()


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_bus_dcline(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_dcline.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_bus_sc(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus_sc.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_bus_est(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus_est.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_bus_3ph(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 0 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus_3ph.index)


def test_SimBench_1_HVMVmixed_1_105_0_sw_modified_res_bus(SimBench_1_HVMVmixed_1_105_0_sw_modified):
    assert 605 == len(SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.index)
    element_0 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.bus[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.bus['origin_id'] == '_35ad5aa9-056f-4fc1-86e3-563842550764'].index]
    assert pytest.approx(1.0491329283083566, abs=0.000001) == element_0['vm_pu'].item()
    assert pytest.approx(-144.2919795390245, abs=0.0001) == element_0['va_degree'].item()
    assert pytest.approx(-1.770, abs=0.0001) == element_0['p_mw'].item()
    assert pytest.approx(0.09090, abs=0.0001) == element_0['q_mvar'].item()

    element_1 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.bus[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.bus['origin_id'] == '_b7704afe-8d9f-4154-b51c-72f60567f94d'].index]
    assert pytest.approx(1.0568812557235927, abs=0.000001) == element_1['vm_pu'].item()
    assert pytest.approx(3.616330520902283, abs=0.0001) == element_1['va_degree'].item()
    assert pytest.approx(0.0, abs=0.0001) == element_1['p_mw'].item()
    assert pytest.approx(0.0, abs=0.0001) == element_1['q_mvar'].item()

    element_2 = SimBench_1_HVMVmixed_1_105_0_sw_modified.res_bus.iloc[SimBench_1_HVMVmixed_1_105_0_sw_modified.bus[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.bus['origin_id'] == '_290a09c9-4b06-4135-aaa7-45db9172e33d'].index]
    assert pytest.approx(1.0471714107338401, abs=0.000001) == element_2['vm_pu'].item()
    assert pytest.approx(-144.16663009041676, abs=0.0001) == element_2['va_degree'].item()
    assert pytest.approx(0.0, abs=0.0001) == element_2['p_mw'].item()
    assert pytest.approx(0.0, abs=0.0001) == element_2['q_mvar'].item()


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
    element_0 = SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid[
        SimBench_1_HVMVmixed_1_105_0_sw_modified.ext_grid['origin_id'] == '_5fb17d90-f222-45a2-9e26-05f52a34731f']
    assert 1.0 == element_0['slack_weight'].item()
    assert '_01ff3523-9c93-45fe-8e35-0d2f4f2' == element_0['substation'].item()
    assert 0.0 == element_0['min_p_mw'].item()
    assert 0.0 == element_0['max_p_mw'].item()
    assert 0.0 == element_0['min_q_mvar'].item()
    assert 0.0 == element_0['max_q_mvar'].item()
    assert [[11.3706, 53.601]] == element_0['coords'].item()


def test_realgrid_sgen(realgrid):
    assert 819 == len(realgrid.sgen.index)
    element_0 = realgrid.sgen[realgrid.sgen['origin_id'] == '_1149773851_HGU_SM']
    assert '1149773851' == element_0['name'].item()
    assert '_1362221690_VL_TN1' == realgrid.bus.iloc[element_0['bus'].item()]['origin_id']
    assert 0.0 == element_0['p_mw'].item()
    assert 0.0 == element_0['q_mvar'].item()
    assert 26.53770 == element_0['sn_mva'].item()
    assert 1.0 == element_0['scaling'].item()
    assert not element_0['in_service'].item()
    assert 'Hydro' == element_0['type'].item()
    assert element_0['current_source'].item()
    assert 'SynchronousMachine' == element_0['origin_class'].item()
    assert '_1149773851_HGU_SM_T0' == element_0['terminal'].item()
    assert 0.0 == element_0['k'].item()
    assert math.isnan(element_0['rx'].item())
    assert 20.0 == element_0['vn_kv'].item()
    assert 0.0 == element_0['rdss_ohm'].item()
    assert 0.0 == element_0['xdss_pu'].item()
    assert math.isnan(element_0['lrc_pu'].item())
    assert 'current_source' == element_0['generator_type'].item()

    element_1 = realgrid.sgen[realgrid.sgen['origin_id'] == '_1994364905_GU_SM']
    assert '1994364905' == element_1['name'].item()
    assert '_1129435962_VL_TN1' == realgrid.bus.iloc[element_1['bus'].item()]['origin_id']
    assert 1.00658 == element_1['p_mw'].item()
    assert 1.65901 == element_1['q_mvar'].item()
    assert 49.2443 == element_1['sn_mva'].item()
    assert 1.0 == element_1['scaling'].item()
    assert element_1['in_service'].item()
    assert 'GeneratingUnit' == element_1['type'].item()
    assert element_1['current_source'].item()
    assert 'SynchronousMachine' == element_1['origin_class'].item()
    assert '_1994364905_GU_SM_T0' == element_1['terminal'].item()
    assert 0.0 == element_1['k'].item()
    assert math.isnan(element_1['rx'].item())
    assert 20.0 == element_1['vn_kv'].item()
    assert 0.0 == element_1['rdss_ohm'].item()
    assert 0.0 == element_1['xdss_pu'].item()
    assert math.isnan(element_1['lrc_pu'].item())
    assert 'current_source' == element_1['generator_type'].item()


def test_smallgrid_DL_line_geodata(smallgrid_DL):
    assert 176 == len(smallgrid_DL.line_geodata.index)
    element_0 = smallgrid_DL.line_geodata.iloc[
        smallgrid_DL.line[smallgrid_DL.line['origin_id'] == '_0447c6f1-c766-11e1-8775-005056c00008'].index]
    assert [[162.363632, 128.4656], [162.328033, 134.391541], [181.746033, 134.43364]] == element_0['coords'].item()

    element_1 = smallgrid_DL.line_geodata.iloc[
        smallgrid_DL.line[smallgrid_DL.line['origin_id'] == '_044a5f09-c766-11e1-8775-005056c00008'].index]
    assert [[12.87877, 58.5714264], [12.8923006, 69.33862]] == element_1['coords'].item()


def test_smallgrid_DL_bus_geodata(smallgrid_DL):
    assert 118 == len(smallgrid_DL.bus_geodata.index)
    element_0 = smallgrid_DL.bus_geodata.iloc[
        smallgrid_DL.bus[smallgrid_DL.bus['origin_id'] == '_0471bd2a-c766-11e1-8775-005056c00008'].index]
    assert 18.5449734 == element_0['x'].item()
    assert 11.8253975 == element_0['y'].item()
    assert [[18.5449734, 11.8253975], [18.5449734, 19.41799]] == element_0['coords'].item()


def test_cim2pp(smallgrid_GL):
    assert 118 == len(smallgrid_GL.bus.index)


def test_smallgrid_GL_line_geodata(smallgrid_GL):
    assert 176 == len(smallgrid_GL.line_geodata.index)
    element_0 = smallgrid_GL.line_geodata.iloc[
        smallgrid_GL.line[smallgrid_GL.line['origin_id'] == '_0447c6f1-c766-11e1-8775-005056c00008'].index]
    assert [[-0.741597592830658, 51.33917999267578],
            [-0.9601190090179443, 51.61038589477539],
            [-1.0638651847839355, 51.73857879638672],
            [-1.1654152870178223, 52.01515579223633],
            [-1.1700644493103027, 52.199188232421875]] == element_0['coords'].item()
    element_1 = smallgrid_GL.line_geodata.iloc[
        smallgrid_GL.line[smallgrid_GL.line['origin_id'] == '_044a5f09-c766-11e1-8775-005056c00008'].index]
    assert [[-3.339864492416382, 58.50086212158203],
            [-3.3406713008880615, 58.31454086303711],
            [-3.6551620960235596, 58.135623931884766],
            [-4.029672145843506, 57.973060607910156],
            [-4.254667282104492, 57.71146774291992],
            [-4.405538082122803, 57.53498840332031]] == element_1['coords'].item()


def test_smallgrid_GL_bus_geodata(smallgrid_GL):
    assert 115 == len(smallgrid_GL.bus_geodata.index)
    element_0 = smallgrid_GL.bus_geodata.iloc[
        smallgrid_GL.bus[smallgrid_GL.bus['origin_id'] == '_0471bd2a-c766-11e1-8775-005056c00008'].index]
    assert -4.844991207122803 == element_0['x'].item()
    assert 55.92612075805664 == element_0['y'].item()
    assert math.isnan(element_0['coords'].item())


def test_fullgrid_xward(fullgrid):
    assert 0 == len(fullgrid.xward.index)


def test_fullgrid_ward(fullgrid):
    assert 5 == len(fullgrid.ward.index)
    element_0 = fullgrid.ward[fullgrid.ward['origin_id'] == '_24413233-26c3-4f7e-9f72-4461796938be']
    assert 'BE-Inj-XCA_AL11' == element_0['name'].item()
    assert '_d4affe50316740bdbbf4ae9c7cbf3cfd' == fullgrid.bus.iloc[element_0['bus'].item()]['origin_id']
    assert -46.816625 == element_0['ps_mw'].item()
    assert 79.193778 == element_0['qs_mvar'].item()
    assert 0.0 == element_0['qz_mvar'].item()
    assert 0.0 == element_0['pz_mw'].item()
    assert element_0['in_service'].item()
    assert 'EquivalentInjection' == element_0['origin_class'].item()
    assert '_53072f42-f77b-47e2-bd9a-e097c910b173' == element_0['terminal'].item()


def test_fullgrid_trafo3w(fullgrid):
    assert 1 == len(fullgrid.trafo3w.index)  # TODO: test with more elements
    element_0 = fullgrid.trafo3w[fullgrid.trafo3w['origin_id'] == '_84ed55f4-61f5-4d9d-8755-bba7b877a246']
    assert 'BE-TR3_1' == element_0['name'].item()
    assert None is element_0['std_type'].item()
    assert '_ac279ca9-c4e2-0145-9f39-c7160fff094d' == fullgrid.bus.iloc[element_0['hv_bus'].item()]['origin_id']
    assert '_99b219f3-4593-428b-a4da-124a54630178' == fullgrid.bus.iloc[element_0['mv_bus'].item()]['origin_id']
    assert '_f96d552a-618d-4d0c-a39a-2dea3c411dee' == fullgrid.bus.iloc[element_0['lv_bus'].item()]['origin_id']
    assert 650.0 == element_0['sn_hv_mva'].item()
    assert 650.0 == element_0['sn_mv_mva'].item()
    assert 650.0 == element_0['sn_lv_mva'].item()
    assert 380.0 == element_0['vn_hv_kv'].item()
    assert 220.0 == element_0['vn_mv_kv'].item()
    assert 21.0 == element_0['vn_lv_kv'].item()
    assert 15.7560924405895 == element_0['vk_hv_percent'].item()
    assert 17.000038713248305 == element_0['vk_mv_percent'].item()
    assert 16.752945398532603 == element_0['vk_lv_percent'].item()
    assert 0.8394327539433621 == element_0['vkr_hv_percent'].item()
    assert 2.400034426828583 == element_0['vkr_mv_percent'].item()
    assert 2.369466354325664 == element_0['vkr_lv_percent'].item()
    assert 0.0 == element_0['pfe_kw'].item()
    assert 0.05415 == element_0['i0_percent'].item()
    assert 0.0 == element_0['shift_mv_degree'].item()
    assert 0.0 == element_0['shift_lv_degree'].item()
    assert 'mv' == element_0['tap_side'].item()
    assert 17 == element_0['tap_neutral'].item()
    assert 1.0 == element_0['tap_min'].item()
    assert 33.0 == element_0['tap_max'].item()
    assert 0.6250 == element_0['tap_step_percent'].item()
    assert math.isnan(element_0['tap_step_degree'].item())
    assert 17.0 == element_0['tap_pos'].item()
    assert not element_0['tap_at_star_point'].item()
    assert element_0['in_service'].item()
    assert 'PowerTransformer' == element_0['origin_class'].item()
    assert '_76e9ca77-f805-40ea-8120-5a6d58416d34' == element_0['terminal_hv'].item()
    assert '_53fd6693-57e6-482e-8fbe-dcf3531a7ce0' == element_0['terminal_mv'].item()
    assert '_ca0f7e2e-3442-4ada-a704-91f319c0ebe3' == element_0['terminal_lv'].item()
    assert '_5f68a129-d5d8-4b71-9743-9ca2572ba26b' == element_0['PowerTransformerEnd_id_hv'].item()
    assert '_e1f661c0-971d-4ce5-ad39-0ec427f288ab' == element_0['PowerTransformerEnd_id_mv'].item()
    assert '_2e21d1ef-2287-434c-a767-1ca807cf2478' == element_0['PowerTransformerEnd_id_lv'].item()
    assert 'RatioTapChanger' == element_0['tapchanger_class'].item()
    assert '_fe25f43a-7341-446e-a71a-8ab7119ba806' == element_0['tapchanger_id'].item()
    assert 'YYY' == element_0['vector_group'].item()
    assert isinstance(element_0['id_characteristic'].item(), float)
    assert math.isnan(element_0['vk0_hv_percent'].item())
    assert math.isnan(element_0['vk0_mv_percent'].item())
    assert math.isnan(element_0['vk0_lv_percent'].item())
    assert math.isnan(element_0['vkr0_hv_percent'].item())
    assert math.isnan(element_0['vkr0_mv_percent'].item())
    assert math.isnan(element_0['vkr0_lv_percent'].item())
    assert not element_0['power_station_unit'].item()
    assert element_0['tap_dependent_impedance'].item()
    assert 14 == element_0['vk_hv_percent_characteristic'].item()
    assert 16 == element_0['vk_mv_percent_characteristic'].item()
    assert 18 == element_0['vk_lv_percent_characteristic'].item()
    assert 15 == element_0['vkr_hv_percent_characteristic'].item()
    assert 17 == element_0['vkr_mv_percent_characteristic'].item()
    assert 19 == element_0['vkr_lv_percent_characteristic'].item()


def test_fullgrid_trafo(fullgrid):
    assert 10 == len(fullgrid.trafo.index)
    element_0 = fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c']
    assert 'HVDC1_TR2_HVDC2' == element_0['name'].item()
    assert None is element_0['std_type'].item()
    assert '_c142012a-b652-4c03-9c35-aa0833e71831' == fullgrid.bus.iloc[element_0['hv_bus'].item()]['origin_id']
    assert '_b01fe92f-68ab-4123-ae45-f22d3e8daad1' == fullgrid.bus.iloc[element_0['lv_bus'].item()]['origin_id']
    assert 157.70 == element_0['sn_mva'].item()
    assert 225.0 == element_0['vn_hv_kv'].item()
    assert 123.90 == element_0['vn_lv_kv'].item()
    assert 12.619851773827161 == element_0['vk_percent'].item()
    assert 0.0 == element_0['vkr_percent'].item()
    assert 0.0 == element_0['pfe_kw'].item()
    assert 0.0 == element_0['i0_percent'].item()
    assert 0.0 == element_0['shift_degree'].item()
    assert 'hv' == element_0['tap_side'].item()
    assert 0.0 == element_0['tap_neutral'].item()
    assert -15.0 == element_0['tap_min'].item()
    assert 15.0 == element_0['tap_max'].item()
    assert 1.250 == element_0['tap_step_percent'].item()
    assert math.isnan(element_0['tap_step_degree'].item())
    assert -2.0 == element_0['tap_pos'].item()
    assert not element_0['tap_phase_shifter'].item()
    assert 1.0 == element_0['parallel'].item()
    assert 1.0 == element_0['df'].item()
    assert element_0['in_service'].item()
    assert 'PowerTransformer' == element_0['origin_class'].item()
    assert '_fd64173b-8fb5-4b66-afe5-9a832e6bcb45' == element_0['terminal_hv'].item()
    assert '_5b52e14e-550a-4084-91cc-14ec5d38e042' == element_0['terminal_lv'].item()
    assert '_3581a7e1-95c0-4778-a108-1a6740abfacb' == element_0['PowerTransformerEnd_id_hv'].item()
    assert '_922f1973-62d7-4190-9556-39faa8ca39b8' == element_0['PowerTransformerEnd_id_lv'].item()
    assert 'RatioTapChanger' == element_0['tapchanger_class'].item()
    assert '_f6b6428b-d201-4170-89f3-4f630c662b7c' == element_0['tapchanger_id'].item()
    assert 'YY' == element_0['vector_group'].item()
    assert isinstance(element_0['id_characteristic'].item(), float)
    assert math.isnan(element_0['vk0_percent'].item())
    assert math.isnan(element_0['vkr0_percent'].item())
    assert math.isnan(element_0['xn_ohm'].item())
    assert not element_0['power_station_unit'].item()
    assert not element_0['oltc'].item()
    assert element_0['tap_dependent_impedance'].item()
    assert 0 == element_0['vkr_percent_characteristic'].item()
    assert 1 == element_0['vk_percent_characteristic'].item()

    element_1 = fullgrid.trafo[fullgrid.trafo['origin_id'] == '_99f55ee9-2c75-3340-9539-b835ec8c5994']
    assert 'BE-TR2_6' == element_1['name'].item()
    assert None is element_1['std_type'].item()
    assert '_e44141af-f1dc-44d3-bfa4-b674e5c953d7' == fullgrid.bus.iloc[element_1['hv_bus'].item()]['origin_id']
    assert '_5c74cb26-ce2f-40c6-951d-89091eb781b6' == fullgrid.bus.iloc[element_1['lv_bus'].item()]['origin_id']
    assert 650.0 == element_1['sn_mva'].item()
    assert 380.0 == element_1['vn_hv_kv'].item()
    assert 110.0 == element_1['vn_lv_kv'].item()
    assert 6.648199321225537 == element_1['vk_percent'].item()
    assert 1.2188364265927978 == element_1['vkr_percent'].item()
    assert 0.0 == element_1['pfe_kw'].item()
    assert 0.0 == element_1['i0_percent'].item()
    assert 0.0 == element_1['shift_degree'].item()
    assert None is element_1['tap_side'].item()
    assert pd.isna(element_1['tap_neutral'].item())
    assert pd.isna(element_1['tap_min'].item())
    assert pd.isna(element_1['tap_max'].item())
    assert math.isnan(element_1['tap_step_percent'].item())
    assert math.isnan(element_1['tap_step_degree'].item())
    assert math.isnan(element_1['tap_pos'].item())
    assert not element_1['tap_phase_shifter'].item()
    assert 1.0 == element_1['parallel'].item()
    assert 1.0 == element_1['df'].item()
    assert element_1['in_service'].item()
    assert 'PowerTransformer' == element_1['origin_class'].item()
    assert '_f8f712ea-4c6f-a64d-970f-ffec2af4931c' == element_1['terminal_hv'].item()
    assert '_6fdc4516-25fc-2f4e-996f-1f590fd5677a' == element_1['terminal_lv'].item()
    assert '_162712fd-bd8f-2d4d-8ac9-84bf324ef796' == element_1['PowerTransformerEnd_id_hv'].item()
    assert '_3ee25db5-2305-1d40-a515-01acb2a12e93' == element_1['PowerTransformerEnd_id_lv'].item()
    assert math.isnan(element_1['tapchanger_class'].item())
    assert math.isnan(element_1['tapchanger_id'].item())
    assert 'YY' == element_1['vector_group'].item()
    assert isinstance(element_1['id_characteristic'].item(), float)
    assert math.isnan(element_1['vk0_percent'].item())
    assert math.isnan(element_1['vkr0_percent'].item())
    assert math.isnan(element_1['xn_ohm'].item())
    assert not element_1['power_station_unit'].item()
    assert not element_1['oltc'].item()
    assert element_1['tap_dependent_impedance'].item()
    assert 8 == element_1['vkr_percent_characteristic'].item()
    assert 9 == element_1['vk_percent_characteristic'].item()

    element_2 = fullgrid.trafo[fullgrid.trafo['origin_id'] == '_ff3a91ec-2286-a64c-a046-d62bc0163ffe']
    assert 1.990 == element_2['tap_step_degree'].item()
    assert 'PhaseTapChangerLinear' == element_2['tapchanger_class'].item()
    assert math.isnan(element_2['id_characteristic'].item())
    assert not element_2['tap_dependent_impedance'].item()
    assert pd.isna(element_2['vkr_percent_characteristic'].item())
    assert pd.isna(element_2['vk_percent_characteristic'].item())


def test_fullgrid_tcsc(fullgrid):
    assert 0 == len(fullgrid.tcsc.index)


def test_fullgrid_switch(fullgrid):
    assert 4 == len(fullgrid.switch.index)
    element_0 = fullgrid.switch[fullgrid.switch['origin_id'] == '_8a3ad6e1-6e23-b649-880e-4865217501c4']
    assert '_5c74cb26-ce2f-40c6-951d-89091eb781b6' == fullgrid.bus.iloc[element_0['bus'].item()]['origin_id']
    assert '_c21be5da-d2a6-d94f-8dcb-92e4d6fa48a7' == fullgrid.bus.iloc[element_0['element'].item()]['origin_id']
    assert 'b' == element_0['et'].item()
    assert 'DS' == element_0['type'].item()
    assert element_0['closed'].item()
    assert 'BE_DSC_5' == element_0['name'].item()
    assert 0.0 == element_0['z_ohm'].item()
    assert math.isnan(element_0['in_ka'].item())
    assert 'Disconnector' == element_0['origin_class'].item()
    assert '_2af7ad2c-062c-1c4f-be3e-9c7cd594ddbb' == element_0['terminal_bus'].item()
    assert '_916578a1-7a6e-7347-a5e0-aaf35538949c' == element_0['terminal_element'].item()


def test_fullgrid_svc(fullgrid):
    assert 0 == len(fullgrid.svc.index)


def test_fullgrid_storage(fullgrid):
    assert 0 == len(fullgrid.storage.index)


def test_fullgrid_shunt(fullgrid):
    assert 6 == len(fullgrid.shunt.index)
    element_0 = fullgrid.shunt[fullgrid.shunt['origin_id'] == '_d771118f-36e9-4115-a128-cc3d9ce3e3da']
    assert 'BE_S1' == element_0['name'].item()
    assert '_f6ee76f7-3d28-6740-aa78-f0bf7176cdad' == fullgrid.bus.iloc[element_0['bus'].item()]['origin_id']
    assert -299.99530 == element_0['q_mvar'].item()
    assert 0.0 == element_0['p_mw'].item()
    assert 110.0 == element_0['vn_kv'].item()
    assert 1 == element_0['step'].item()
    assert 1 == element_0['max_step'].item()
    assert element_0['in_service'].item()
    assert 'LinearShuntCompensator' == element_0['origin_class'].item()
    assert '_d5e2e58e-ccf6-47d9-b3bb-3088eb7a9b6c' == element_0['terminal'].item()


def test_fullgrid_sgen(fullgrid):
    assert 1 == len(fullgrid.sgen.index)


def test_fullgrid_pwl_cost(fullgrid):
    assert 0 == len(fullgrid.pwl_cost.index)


def test_fullgrid_poly_cost(fullgrid):
    assert 0 == len(fullgrid.poly_cost.index)


def test_fullgrid_motor(fullgrid):
    assert 1 == len(fullgrid.motor.index)
    element_0 = fullgrid.motor[fullgrid.motor['origin_id'] == '_2b618292-5fec-af43-ae39-c32566d0a752']
    assert 'ASM_1' == element_0['name'].item()
    assert '_f70f6bad-eb8d-4b8f-8431-4ab93581514e' == fullgrid.bus.iloc[element_0['bus'].item()]['origin_id']
    assert math.isnan(element_0['pn_mech_mw'].item())
    assert math.isnan(element_0['loading_percent'].item())
    assert 0.9 == element_0['cos_phi'].item()
    assert 0.9 == element_0['cos_phi_n'].item()
    assert 100.0 == element_0['efficiency_percent'].item()
    assert math.isnan(element_0['efficiency_n_percent'].item())
    assert math.isnan(element_0['lrc_pu'].item())
    assert 225.0 == element_0['vn_kv'].item()
    assert 1.0 == element_0['scaling'].item()
    assert element_0['in_service'].item()
    assert math.isnan(element_0['rx'].item())
    assert 'AsynchronousMachine' == element_0['origin_class'].item()
    assert '_7b71e695-3977-f544-b31f-777cfbbde49b' == element_0['terminal'].item()


def test_fullgrid_measurement(fullgrid):
    assert 0 == len(fullgrid.measurement.index)  # TODO: analogs


def test_fullgrid_load(fullgrid):
    assert 5 == len(fullgrid.load.index)
    element_0 = fullgrid.load[fullgrid.load['origin_id'] == '_1324b99a-59ee-0d44-b1f6-15dc0d9d81ff']
    assert 'BE_CL_1' == element_0['name'].item()
    assert '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35' == fullgrid.bus.iloc[element_0['bus'].item()]['origin_id']
    assert 0.010 == element_0['p_mw'].item()
    assert 0.010 == element_0['q_mvar'].item()
    assert 0.0 == element_0['const_z_percent'].item()
    assert 0.0 == element_0['const_i_percent'].item()
    assert math.isnan(element_0['sn_mva'].item())
    assert 1.0 == element_0['scaling'].item()
    assert element_0['in_service'].item()
    assert None is element_0['type'].item()
    assert 'ConformLoad' == element_0['origin_class'].item()
    assert '_84f6ff75-6bf9-8742-ae06-1481aa3b34de' == element_0['terminal'].item()


def test_fullgrid_line_geodata(fullgrid):
    assert 0 == len(fullgrid.line_geodata.index)


def test_fullgrid_line(fullgrid):
    assert 11 == len(fullgrid.line.index)
    element_0 = fullgrid.line[fullgrid.line['origin_id'] == '_a16b4a6c-70b1-4abf-9a9d-bd0fa47f9fe4']
    assert 'BE-Line_7' == element_0['name'].item()
    assert None is element_0['std_type'].item()
    assert '_1fa19c281c8f4e1eaad9e1cab70f923e' == fullgrid.bus.iloc[element_0['from_bus'].item()]['origin_id']
    assert '_f70f6bad-eb8d-4b8f-8431-4ab93581514e' == fullgrid.bus.iloc[element_0['to_bus'].item()]['origin_id']
    assert 23.0 == element_0['length_km'].item()
    assert 0.19999999999999998 == element_0['r_ohm_per_km'].item()
    assert 3.0 == element_0['x_ohm_per_km'].item()
    assert 3.0000014794808827 == element_0['c_nf_per_km'].item()
    assert 2.5 == element_0['g_us_per_km'].item()
    assert 1.0620 == element_0['max_i_ka'].item()
    assert 1.0 == element_0['df'].item()
    assert 1.0 == element_0['parallel'].item()
    assert None is element_0['type'].item()
    assert element_0['in_service'].item()
    assert 'ACLineSegment' == element_0['origin_class'].item()
    assert '_57ae9251-c022-4c67-a8eb-611ad54c963c' == element_0['terminal_from'].item()
    assert '_5b2c65b0-68ce-4530-85b7-385346a3b5e1' == element_0['terminal_to'].item()
    assert math.isnan(element_0['r0_ohm_per_km'].item())
    assert math.isnan(element_0['x0_ohm_per_km'].item())
    assert math.isnan(element_0['c0_nf_per_km'].item())
    assert 0.0 == element_0['g0_us_per_km'].item()
    assert math.isnan(element_0['endtemp_degree'].item())

    element_1 = fullgrid.line[fullgrid.line['origin_id'] == '_6052bacf-9eaa-4217-be91-4c7c89e92a52']
    assert math.isnan(element_1['max_i_ka'].item())


def test_fullgrid_impedance(fullgrid):
    assert 1 == len(fullgrid.impedance.index)  # TODO: test with more elements
    element_0 = fullgrid.impedance[fullgrid.impedance['origin_id'] == '_3619970b-7c3d-bf4f-b499-fb0a99efb362']
    assert 'BE_SC_1' == element_0['name'].item()
    assert '_514fa0d5-a432-5743-8204-1c8518ffed76' == fullgrid.bus.iloc[element_0['from_bus'].item()]['origin_id']
    assert '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35' == fullgrid.bus.iloc[element_0['to_bus'].item()]['origin_id']
    assert 8.181818181818182e-05 == element_0['rft_pu'].item()
    assert 8.181818181818182e-05 == element_0['xft_pu'].item()
    assert 8.181818181818182e-05 == element_0['rtf_pu'].item()
    assert 8.181818181818182e-05 == element_0['xtf_pu'].item()
    assert 1.0 == element_0['sn_mva'].item()
    assert element_0['in_service'].item()
    assert 'SeriesCompensator' == element_0['origin_class'].item()
    assert '_0b2c4a73-e4dd-4445-acc3-1284ad5a8a70' == element_0['terminal_from'].item()
    assert '_8c735a96-1b4c-a34d-8823-d6124bd87042' == element_0['terminal_to'].item()
    assert math.isnan(element_0['rft0_pu'].item())
    assert math.isnan(element_0['xft0_pu'].item())
    assert math.isnan(element_0['rtf0_pu'].item())
    assert math.isnan(element_0['xtf0_pu'].item())


def test_fullgrid_gen(fullgrid):
    assert len(fullgrid.gen.index) in [7, 9, 10, 11]
    element_0 = fullgrid.gen[fullgrid.gen['origin_id'] == '_55d4aae2-0d4b-4248-bc90-1193f3499fa0']
    assert 'BE-G5' == element_0['name'].item()
    assert '_f96d552a-618d-4d0c-a39a-2dea3c411dee' == fullgrid.bus.iloc[element_0['bus'].item()]['origin_id']
    assert 118.0 == element_0['p_mw'].item()
    assert 1.04700 == element_0['vm_pu'].item()
    assert 300.0 == element_0['sn_mva'].item()
    assert -200.0 == element_0['min_q_mvar'].item()
    assert 200.0 == element_0['max_q_mvar'].item()
    assert 1.0 == element_0['scaling'].item()
    assert not element_0['slack'].item()
    assert element_0['in_service'].item()
    assert math.isnan(element_0['slack_weight'].item())
    assert 'Nuclear' == element_0['type'].item()
    assert 'SynchronousMachine' == element_0['origin_class'].item()
    assert '_b2dcbf07-4676-774f-ae35-86c1ab695de0' == element_0['terminal'].item()
    assert 50.0 == element_0['min_p_mw'].item()
    assert 200 == element_0['max_p_mw'].item()
    assert 21.0 == element_0['vn_kv'].item()
    assert math.isnan(element_0['rdss_ohm'].item())
    assert math.isnan(element_0['xdss_pu'].item())
    assert 0.850 == element_0['cos_phi'].item()
    assert 0.0 == element_0['pg_percent'].item()

    element_1 = fullgrid.gen[fullgrid.gen['origin_id'] == '_3a3b27be-b18b-4385-b557-6735d733baf0']
    assert 1.050 == element_1['vm_pu'].item()


def test_fullgrid_ext_grid(fullgrid):
    assert 1 == len(fullgrid.ext_grid.index)
    element_0 = fullgrid.ext_grid[fullgrid.ext_grid['origin_id'] == '_484436ac-0c91-6743-8db9-91daf8ec5182']
    assert 'ES_1' == element_0['name'].item()
    assert '_f70f6bad-eb8d-4b8f-8431-4ab93581514e' == fullgrid.bus.iloc[element_0['bus'].item()]['origin_id']
    assert 1 == element_0['vm_pu'].item()
    assert 0 == element_0['va_degree'].item()
    assert math.isnan(element_0['slack_weight'].item())
    assert element_0['in_service'].item()
    assert 'EnergySource' == element_0['origin_class'].item()
    assert '_9835652b-053f-cb44-822e-1e26950d989c' == element_0['terminal'].item()
    assert math.isnan(element_0['substation'].item())
    assert math.isnan(element_0['min_p_mw'].item())
    assert math.isnan(element_0['max_p_mw'].item())
    assert math.isnan(element_0['min_q_mvar'].item())
    assert math.isnan(element_0['max_q_mvar'].item())
    assert -9.99000 == element_0['p_mw'].item()
    assert -0.99000 == element_0['q_mvar'].item()
    assert math.isnan(element_0['s_sc_max_mva'].item())
    assert math.isnan(element_0['s_sc_min_mva'].item())
    assert math.isnan(element_0['rx_max'].item())
    assert math.isnan(element_0['rx_min'].item())
    assert math.isnan(element_0['r0x0_max'].item())
    assert math.isnan(element_0['x0x_max'].item())


def test_fullgrid_dcline(fullgrid):
    assert 2 == len(fullgrid.dcline.index)
    element_0 = fullgrid.dcline[fullgrid.dcline['origin_id'] == '_70a3750c-6e8e-47bc-b1bf-5a568d9733f7']
    assert 'LDC-1230816355' == element_0['name'].item()
    assert '_27d57afa-6c9d-4b06-93ea-8c88d14af8b1' == fullgrid.bus.iloc[element_0['from_bus'].item()]['origin_id']
    assert '_d3d9c515-2ddb-436a-bf17-2f8be2394de3' == fullgrid.bus.iloc[int(element_0['to_bus'].item())]['origin_id']
    assert 0.0 == element_0['p_mw'].item()
    assert 0.0 == element_0['loss_percent'].item()
    assert 0.0 == element_0['loss_mw'].item()
    assert 1.0 == element_0['vm_from_pu'].item()
    assert 1.0 == element_0['vm_to_pu'].item()
    assert math.isnan(element_0['max_p_mw'].item())
    assert math.isnan(element_0['min_q_from_mvar'].item())
    assert math.isnan(element_0['min_q_to_mvar'].item())
    assert math.isnan(element_0['max_q_from_mvar'].item())
    assert math.isnan(element_0['max_q_to_mvar'].item())
    assert element_0['in_service'].item()
    assert 'DCLineSegment' == element_0['origin_class'].item()
    assert '_4123e718-716a-4988-bf71-0e525a4422f2' == element_0['terminal_from'].item()
    assert '_c4c335b5-0405-4539-be10-697f5a3f3e83' == element_0['terminal_to'].item()

    element_0 = fullgrid.dcline[fullgrid.dcline['origin_id'] == '_d7693c6d-58bd-49da-bb24-973a63f9faf1']
    assert math.isnan(element_0['to_bus'].item())
    assert math.isnan(element_0['loss_mw'].item())
    assert math.isnan(element_0['vm_to_pu'].item())
    assert pd.isna(element_0['in_service'].item())
    assert math.isnan(element_0['terminal_to'].item())


def test_fullgrid_controller(fullgrid):
    assert 8 == len(fullgrid.controller.index)
    for _, obj in fullgrid.controller.iterrows():
        if obj.object.matching_params.get('tid') == \
                fullgrid.trafo[fullgrid.trafo['origin_id'] == '_4ebd554c-1cdb-4f3d-8dd0-dfd4bda8e18c'].index:
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
        if obj.object.matching_params.get('tid') == \
                fullgrid.trafo[fullgrid.trafo['origin_id'] == '_69a301e8-f6b2-47ad-9f65-52f2eabc9917'].index:
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
        if obj.object.index == \
                fullgrid.trafo[fullgrid.trafo['origin_id'] ==
                               '_99f55ee9-2c75-3340-9539-b835ec8c5994']['vkr_percent_characteristic'].item():
            assert [1] == obj.object.x_vals
            assert [1.3405981856094185] == obj.object.y_vals
            break


def test_fullgrid_bus_geodata(fullgrid):
    assert 0 == len(fullgrid.bus_geodata.index)


def test_fullgrid_bus(fullgrid):
    assert 26 == len(fullgrid.bus.index)
    element_0 = fullgrid.bus[fullgrid.bus['origin_id'] == '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35']
    assert 'BE-Busbar_7' == element_0['name'].item()
    assert 110.00000 == element_0['vn_kv'].item()
    assert 'b' == element_0['type'].item()
    assert '' == element_0['zone'].item()
    assert element_0['in_service'].item()
    assert 'TopologicalNode' == element_0['origin_class'].item()
    assert 'tp' == element_0['origin_profile'].item()
    assert '_4c66b132-0977-1e4c-b9bb-d8ce2e912e35' == element_0['cim_topnode'].item()
    assert math.isnan(element_0['ConnectivityNodeContainer_id'].item())
    assert math.isnan(element_0['substation_id'].item())
    assert 'BBRUS151; BGENT_51' == element_0['description'].item()

    element_1 = fullgrid.bus[fullgrid.bus['origin_id'] == '_1098b1c9-dc85-40ce-b65c-39ae02a3afaa']
    assert math.isnan(element_1['cim_topnode'].item())
    assert math.isnan(element_1['description'].item())


def test_fullgrid_asymmetric_sgen(fullgrid):
    assert 0 == len(fullgrid.asymmetric_sgen.index)


def test_fullgrid_asymmetric_load(fullgrid):
    assert 0 == len(fullgrid.asymmetric_load.index)


if __name__ == "__main__":
    pytest.main([__file__])
