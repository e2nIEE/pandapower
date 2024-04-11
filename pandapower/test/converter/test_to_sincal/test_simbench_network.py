import os

import numpy as np
import pytest

from pandapower.converter.sincal.pp2sincal.util.main import initialize, convert_pandapower_net, finalize
from pandapower.test import test_path
from pandapower.test.converter.test_to_sincal.result_comparison import compare_results

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

try:
    import simbench
except ImportError:
    logger.warning(r'you need to install the package "simbench" first')

try:
    import win32com.client
except:
    logger.warning(r'seems like you are not on a windows machine')

try:
    simulation = win32com.client.Dispatch("Sincal.Simulation")
except:
    logger.warning("Sincal not found. Install Sincal first.")
    simulation = None

import importlib

scenarios = [0, 1, 2]

plotting = True


@pytest.mark.skipif(importlib.util.find_spec('simbench') is None or simulation is None,
                    reason=r'you need to install the package "simbench" first. '
                           r'If you installed simbench and the test is still skipped, you need a sincal instance')
@pytest.mark.slow
@pytest.mark.parametrize("scenario", scenarios)
def test_all_simbench_networks(scenario):
    codes = simbench.collect_all_simbench_codes(scenario=scenario)
    for simbench_code in codes:
        if ('complete' in simbench_code) | ('EHVHVMVLV' in simbench_code):
            continue
        print(simbench_code)
        net_pp = simbench.get_simbench_net(simbench_code)
        output_folder = os.path.join(test_path, 'converter', 'test_to_sincal',
                                     'results', 'simbench', 'scenario_' + str(scenario))
        file_name = simbench_code + '.sin'
        use_active_net = False
        use_ui = False
        sincal_interaction = False
        net, app, sim, doc = initialize(output_folder, file_name, use_active_net, use_ui, sincal_interaction)
        convert_pandapower_net(net, net_pp, doc, plotting)
        diff_u, diff_deg = compare_results(net, net_pp, sim)
        assert (all(np.isclose(np.zeros(len(diff_u)), diff_u, atol=1 * 10 ** -6)))
        assert (all(np.isclose(np.zeros(len(diff_deg)), diff_deg, atol=1 * 10 ** -5)))
        finalize(net, output_folder, file_name, app, sim, doc, sincal_interaction)

@pytest.mark.skipif(importlib.util.find_spec('simbench') is None or simulation is None,
                    reason=r'you need to install the package "simbench" first. '
                           r'If you installed simbench and the test is still skipped, you need a sincal instance')
@pytest.mark.slow
def test_all_simbench_networks_scen_0():
    codes = simbench.collect_all_simbench_codes(scenario=0)
    for simbench_code in codes:
        if ('complete' in simbench_code) | ('EHVHVMVLV' in simbench_code):
            continue
        print(simbench_code)
        net_pp = simbench.get_simbench_net(simbench_code)
        output_folder = os.path.join(test_path, 'converter', 'test_to_sincal', 'results', 'simbench', 'scenario_0')
        file_name = simbench_code + '.sin'
        use_active_net = False
        use_ui = False
        sincal_interaction = False
        net, app, sim, doc = initialize(output_folder, file_name, use_active_net, use_ui, sincal_interaction)
        convert_pandapower_net(net, net_pp, doc, plotting)
        diff_u, diff_deg = compare_results(net, net_pp, sim)
        assert (all(np.isclose(np.zeros(len(diff_u)), diff_u, atol=1 * 10 ** -6)))
        assert (all(np.isclose(np.zeros(len(diff_deg)), diff_deg, atol=1 * 10 ** -6)))
        finalize(net, output_folder, file_name, app, sim, doc, sincal_interaction)

@pytest.mark.skipif(importlib.util.find_spec('simbench') is None or simulation is None,
                    reason=r'you need to install the package "simbench" first. '
                           r'If you installed simbench and the test is still skipped, you need a sincal instance')
@pytest.mark.slow
def test_all_simbench_networks_scen_1():
    codes = simbench.collect_all_simbench_codes(scenario=1)
    for simbench_code in codes:
        if ('complete' in simbench_code) | ('EHVHVMVLV' in simbench_code):
            continue
        print(simbench_code)
        net_pp = simbench.get_simbench_net(simbench_code)
        output_folder = os.path.join(test_path, 'converter', 'test_to_sincal', 'results', 'simbench', 'scenario_1')
        file_name = simbench_code + '.sin'
        use_active_net = False
        use_ui = False
        sincal_interaction = False
        net, app, sim, doc = initialize(output_folder, file_name, use_active_net, use_ui, sincal_interaction)
        convert_pandapower_net(net, net_pp, doc, plotting)
        diff_u, diff_deg = compare_results(net, net_pp, sim)
        assert (all(np.isclose(np.zeros(len(diff_u)), diff_u, atol=1 * 10 ** -6)))
        assert (all(np.isclose(np.zeros(len(diff_deg)), diff_deg, atol=1 * 10 ** -6)))
        finalize(net, output_folder, file_name, app, sim, doc, sincal_interaction)

@pytest.mark.skipif(importlib.util.find_spec('simbench') is None or simulation is None,
                    reason=r'you need to install the package "simbench" first. '
                           r'If you installed simbench and the test is still skipped, you need a sincal instance')
@pytest.mark.slow
def test_all_simbench_networks_scen_2():
    codes = simbench.collect_all_simbench_codes(scenario=2)
    for simbench_code in codes:
        if ('complete' in simbench_code) | ('EHVHVMVLV' in simbench_code):
            continue
        print(simbench_code)
        net_pp = simbench.get_simbench_net(simbench_code)
        output_folder = os.path.join(test_path, 'converter', 'test_to_sincal', 'results', 'simbench', 'scenario_2')
        file_name = simbench_code + '.sin'
        use_active_net = False
        use_ui = False
        sincal_interaction = False
        net, app, sim, doc = initialize(output_folder, file_name, use_active_net, use_ui, sincal_interaction)
        convert_pandapower_net(net, net_pp, doc, plotting)
        diff_u, diff_deg = compare_results(net, net_pp, sim)
        assert (all(np.isclose(np.zeros(len(diff_u)), diff_u, atol=1 * 10 ** -6)))
        assert (all(np.isclose(np.zeros(len(diff_deg)), diff_deg, atol=1 * 10 ** -6)))
        finalize(net, output_folder, file_name, app, sim, doc, sincal_interaction)

@pytest.mark.skipif(importlib.util.find_spec('simbench') is None or simulation is None,
                    reason=r'you need to install the package "simbench" first. '
                           r'If you installed simbench and the test is still skipped, you need a sincal instance')
@pytest.mark.parametrize("scenario", scenarios)
def test_all_lv_simbench_networks(scenario):
    codes = simbench.collect_all_simbench_codes(scenario=scenario)
    for simbench_code in codes:
        if simbench_code.split('-')[1] != 'LV':
            continue
        print(simbench_code)
        net_pp = simbench.get_simbench_net(simbench_code)
        output_folder = os.path.join(test_path, 'converter', 'test_to_sincal',
                                     'results', 'simbench', 'scenario_' + str(scenario))
        file_name = simbench_code + '.sin'
        use_active_net = False
        use_ui = False
        sincal_interaction = False
        net, app, sim, doc = initialize(output_folder, file_name, use_active_net, use_ui, sincal_interaction)
        convert_pandapower_net(net, net_pp, doc, plotting)
        diff_u, diff_deg = compare_results(net, net_pp, sim)
        assert (all(np.isclose(np.zeros(len(diff_u)), diff_u, atol=1 * 10 ** -6)))
        assert (all(np.isclose(np.zeros(len(diff_deg)), diff_deg, atol=1 * 10 ** -6)))
        finalize(net, output_folder, file_name, app, sim, doc, sincal_interaction)

@pytest.mark.skipif(importlib.util.find_spec('simbench') is None or simulation is None,
                    reason=r'you need to install the package "simbench" first. '
                           r'If you installed simbench and the test is still skipped, you need a sincal instance')
@pytest.mark.parametrize("scenario", scenarios)
def test_all_mv_simbench_networks(scenario):
    codes = simbench.collect_all_simbench_codes(scenario=scenario)
    for simbench_code in codes:
        if simbench_code.split('-')[1] != 'MV':
            continue
        print(simbench_code)
        net_pp = simbench.get_simbench_net(simbench_code)
        output_folder = os.path.join(test_path, 'converter', 'test_to_sincal',
                                     'results', 'simbench', 'scenario_' + str(scenario))
        file_name = simbench_code + '.sin'
        use_active_net = False
        use_ui = False
        sincal_interaction = False
        net, app, sim, doc = initialize(output_folder, file_name, use_active_net, use_ui, sincal_interaction)
        convert_pandapower_net(net, net_pp, doc, plotting)
        diff_u, diff_deg = compare_results(net, net_pp, sim)
        assert (all(np.isclose(np.zeros(len(diff_u)), diff_u, atol=1 * 10 ** -6)))
        assert (all(np.isclose(np.zeros(len(diff_deg)), diff_deg, atol=1 * 10 ** -6)))
        finalize(net, output_folder, file_name, app, sim, doc, sincal_interaction)

@pytest.mark.skipif(importlib.util.find_spec('simbench') is None or simulation is None,
                    reason=r'you need to install the package "simbench" first. '
                           r'If you installed simbench and the test is still skipped, you need a sincal instance')
@pytest.mark.parametrize("scenario", scenarios)
def test_all_hv_simbench_networks(scenario):
    codes = simbench.collect_all_simbench_codes(scenario=scenario)
    for simbench_code in codes:
        if simbench_code.split('-')[1] != 'HV':
            continue
        print(simbench_code)
        net_pp = simbench.get_simbench_net(simbench_code)
        output_folder = os.path.join(test_path, 'converter', 'test_to_sincal',
                                     'results', 'simbench', 'scenario_' + str(scenario))
        file_name = simbench_code + '.sin'
        use_active_net = False
        use_ui = False
        sincal_interaction = False
        net, app, sim, doc = initialize(output_folder, file_name, use_active_net, use_ui, sincal_interaction)
        convert_pandapower_net(net, net_pp, doc, plotting)
        diff_u, diff_deg = compare_results(net, net_pp, sim)
        assert (all(np.isclose(np.zeros(len(diff_u)), diff_u, atol=1 * 10 ** -6)))
        assert (all(np.isclose(np.zeros(len(diff_deg)), diff_deg, atol=1 * 10 ** -5)))
        finalize(net, output_folder, file_name, app, sim, doc, sincal_interaction)

@pytest.mark.skipif(importlib.util.find_spec('simbench') is None or simulation is None,
                    reason=r'you need to install the package "simbench" first. '
                           r'If you installed simbench and the test is still skipped, you need a sincal instance')
@pytest.mark.parametrize("scenario", scenarios)
def test_ehv_wo_sw_simbench_network(scenario):
    codes = simbench.collect_all_simbench_codes(scenario=scenario)
    for simbench_code in codes:
        splitted_code = simbench_code.split('-')
        if (splitted_code[1] != 'EHV') or (splitted_code[5] != 'no_sw'):
            continue
        print(simbench_code)
        net_pp = simbench.get_simbench_net(simbench_code)
        output_folder = os.path.join(test_path, 'converter', 'test_to_sincal',
                                     'results', 'simbench', 'scenario_' + str(scenario))
        file_name = simbench_code + '.sin'
        use_active_net = False
        use_ui = False
        sincal_interaction = False
        net, app, sim, doc = initialize(output_folder, file_name, use_active_net, use_ui, sincal_interaction)
        convert_pandapower_net(net, net_pp, doc, plotting)
        diff_u, diff_deg = compare_results(net, net_pp, sim)
        assert (all(np.isclose(np.zeros(len(diff_u)), diff_u, atol=1 * 10 ** -6)))
        assert (all(np.isclose(np.zeros(len(diff_deg)), diff_deg, atol=1 * 10 ** -6)))
        finalize(net, output_folder, file_name, app, sim, doc, sincal_interaction)

@pytest.mark.skipif(importlib.util.find_spec('simbench') is None or simulation is None,
                    reason=r'you need to install the package "simbench" first. '
                           r'If you installed simbench and the test is still skipped, you need a sincal instance')
@pytest.mark.slow
@pytest.mark.parametrize("scenario", scenarios)
def test_ehv_sw_simbench_network(scenario):
    codes = simbench.collect_all_simbench_codes(scenario=scenario)
    for simbench_code in codes:
        splitted_code = simbench_code.split('-')
        if (splitted_code[1] != 'EHV') or (splitted_code[5] != 'sw'):
            continue
        print(simbench_code)
        net_pp = simbench.get_simbench_net(simbench_code)
        output_folder = os.path.join(test_path, 'converter', 'test_to_sincal',
                                     'results', 'simbench', 'scenario_' + str(scenario))
        file_name = simbench_code + '.sin'
        use_active_net = False
        use_ui = False
        sincal_interaction = False
        net, app, sim, doc = initialize(output_folder, file_name, use_active_net, use_ui, sincal_interaction)
        convert_pandapower_net(net, net_pp, doc, plotting)
        diff_u, diff_deg = compare_results(net, net_pp, sim)
        assert (all(np.isclose(np.zeros(len(diff_u)), diff_u, atol=1 * 10 ** -6)))
        assert (all(np.isclose(np.zeros(len(diff_deg)), diff_deg, atol=1 * 10 ** -6)))
        finalize(net, output_folder, file_name, app, sim, doc, sincal_interaction)

if __name__ == '__main__':
    pytest.main([__file__, "-xs", "-m", "not slow"])