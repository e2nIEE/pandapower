import numpy as np
import pandas as pd

import pandapower as pp

import os
import pytest

from pandapower.converter.powerfactory.validate import validate_pf_conversion
from pandapower.converter.powerfactory.export_pfd_to_pp import import_project, from_pfd

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

try:
    import powerfactory as pf
    PF_INSTALLED = True
except ImportError:
    PF_INSTALLED = False
    logger.info('could not import powerfactory, converter test pf2pp not possible')


def get_tol():
    # tolerances for validation
    # the model of trafo3w for tap changer at terminals is not very exact, therefore lower
    # tolerances
    tol = {
        'diff_vm': 1e-6,
        'diff_va': 1e-4,
        'line_diff': 1e-2,
        'trafo_diff': 1e-2,
        'trafo3w_diff': 1e-2,
        'sgen_p_diff_is': 1e-5,
        'sgen_q_diff_is': 1e-5,
        'gen_p_diff_is': 1e-5,
        'gen_q_diff_is': 1e-5,
        'ward_p_diff_is': 1e-5,
        'ward_q_diff_is': 1e-5,
        'load_p_diff_is': 1e-5,
        'load_q_diff_is': 1e-5,
        'ext_grid_p_diff': 1e-3,
        'ext_grid_q_diff': 1e-3
    }
    return tol


@pytest.mark.skipif(not PF_INSTALLED, reason='powerfactory must be installed')
def test_pf_export():
    # init PowerFactory
    app = pf.GetApplication()

    # first, import the test grid to powerfactory
    path = os.path.join(pp.pp_dir, 'test', 'converter', 'testfiles', 'test_export.pfd')
    prj = import_project(path, app, 'TEST_PF_CONVERTER', import_folder='TEST_IMPORT', clear_import_folder=True)
    prj_name = prj.GetFullName()

    net = from_pfd(app, prj_name=prj_name)

    all_diffs = validate_pf_conversion(net, tolerance_mva=1e-9)

    tol = get_tol()

    for key, diff in all_diffs.items():
        delta = diff.abs().max()
        assert delta < tol[key], "%s has too high difference: %f > %f" % (key, delta, tol[key])


@pytest.mark.xfail(reason="implementation of the trafo3w data model is not completely consistent with PowerFactory")
@pytest.mark.skipif(not PF_INSTALLED, reason='powerfactory must be installed')
def test_pf_export_trafo3w():
    app = pf.GetApplication()
    # import the 3W-Trafo test grid to powerfactory
    # todo: at the moment the 3W-Trafo model is not accurate enough, here testing with lower tol
    path = os.path.join(pp.pp_dir, 'test', 'converter', 'testfiles', 'test_trafo3w.pfd')
    prj = import_project(path, app, 'TEST_PF_CONVERTER', import_folder='TEST_IMPORT', clear_import_folder=True)
    prj_name = prj.GetFullName()

    net = from_pfd(app, prj_name=prj_name)

    all_diffs = validate_pf_conversion(net, tolerance_mva=1e-9)
    tol = get_tol()

    # doesn't pass yet due to trafo3w implementation
    # trafo3w implementation is not very accurate
    for key, diff in all_diffs.items():
        delta = diff.abs().max()
        assert delta < tol[key], "%s has too high difference: %f > %f" % (key, delta, tol[key])


def test_trafo_tap2_results():
    path = os.path.join(pp.pp_dir, 'test', 'converter', 'testfiles', 'trafo_tap_model.json')
    net = pp.from_json(path)
    all_diffs = validate_pf_conversion(net, tolerance_mva=1e-9)
    tol = 2e-7
    for key, diff in all_diffs.items():
        delta = diff.abs().max()
        assert delta < tol, "%s has too high difference: %f > %f" % (key, delta, tol)


if __name__ == '__main__':
    pytest.main(['-xs', __file__])
