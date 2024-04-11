import importlib
import os

import numpy as np
import pytest
import win32com.client

from pandapower.converter.sincal.pp2sincal.pp2sincal import convert_simbench_network
from pandapower.converter.sincal.pp2sincal.util.main import finalize, pp_preparation, initialize
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
    simulation = win32com.client.Dispatch("Sincal.Simulation")
except:
    logger.warning("Sincal not found. Install Sincal first.")
    simulation = None


@pytest.mark.skipif(importlib.util.find_spec('simbench') is None or simulation is None,
                    reason=r'you need to install the package "simbench" first. '
                           r'If you installed simbench and the test is still skipped, you need a sincal instance')
def test_convert_simbench_network():
    # Simbench Code and Path
    simbench_code = '1-HV-mixed--0-sw'
    output_folder = os.path.join(test_path, 'converter', 'test_to_sincal', 'results', 'main', 'scenario_0')
    file_name = simbench_code + '.sin'

    use_active_net = False
    delete_files = True
    plotting = True
    sincal_interaction = False
    use_ui = False

    # Convert Simbench Network using main-Function
    convert_simbench_network(simbench_code, output_folder, use_active_net=use_active_net, plotting=plotting,
                             use_ui=use_ui, sincal_interaction=sincal_interaction, delete_files=delete_files)

    # initialize
    net, app, sim, doc = initialize(output_folder, file_name, use_active_net=use_active_net, use_ui=use_ui,
                                    sincal_interaction=sincal_interaction, delete_files=False)

    # Get pandapowerNet for Result Comparison
    net_pp = simbench.get_simbench_net(simbench_code)
    pp_preparation(net_pp)

    # Compare Results
    diff_u, diff_deg = compare_results(net, net_pp, sim)

    assert (all(np.isclose(np.zeros(len(diff_u)), diff_u, atol=1 * 10 ** -6)))
    assert (all(np.isclose(np.zeros(len(diff_deg)), diff_deg, atol=1 * 10 ** -6)))

    finalize(net, output_folder, file_name, app, sim, doc, sincal_interaction=True)


if __name__ == '__main__':
    pytest.main([__file__, "-xs", "-m", "not slow"])
