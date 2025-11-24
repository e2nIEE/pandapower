# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import os

import pandas as pd
import pytest
from packaging import version as vs

from pandapower import pp_dir, from_json, convert_format, runpp

folder = os.path.join(pp_dir, "test", "test_files", "old_versions")
found_versions = [file.split("_")[1].split(".json")[0] for _, _, files
                  in os.walk(folder) for file in files]


@pytest.mark.slow
@pytest.mark.parametrize("version", found_versions)
def test_convert_format(version):
    filename = os.path.join(folder, "example_%s.json" % version)
    if not os.path.isfile(filename):
        raise ValueError("File for version %s does not exist" % version)
    try:
        net = from_json(filename, convert=False)
        if ('version' in net) and (vs.parse(str(net.version)) > vs.parse('2.0.1')):
            _ = from_json(filename, elements_to_deserialize=['bus', 'load'])
    except:
        raise UserWarning("Can not load network saved in pandapower version %s" % version)
    vm_pu_old = net.res_bus["vm_pu"].copy()
    convert_format(net)
    try:
        runpp(net, run_control="controller" in net and len(net.controller) > 0)
    except:
        raise UserWarning("Can not run power flow in network "
                          "saved with pandapower version %s" % version)
    if not np.allclose(vm_pu_old.values, net.res_bus["vm_pu"].values):
        raise UserWarning("Power flow results mismatch "
                          "with pandapower version %s" % version)


def test_convert_format_pq_bus_meas():
    net = from_json(os.path.join(folder, "example_2.3.1.json"), convert=False)
    net = convert_format(net)
    runpp(net)

    bus_p_meas = net.measurement.query("element_type=='bus' and measurement_type=='p'").set_index("element", drop=True)
    assert np.allclose(net.res_bus.p_mw, bus_p_meas["value"])


def test_convert_format_controller():
    net = from_json(os.path.join(folder, "example_2.3.0.json"), convert=True)
    controller = net.controller.object.iloc[0]
    assert not hasattr(controller, "net")


def test_convert_format_characteristics():
    net = from_json(os.path.join(folder, "example_2.13.0.1.json"), convert=True)
    assert hasattr(net.characteristic.at[0, "object"], "interpolator_kind")
    assert hasattr(net.characteristic.at[0, "object"], "kwargs")
    assert not hasattr(net.characteristic.at[0, "object"], "kind")
    assert not hasattr(net.characteristic.at[0, "object"], "fill_value")
    runpp(net)


def test_convert_format_adding_characteristic_columns():
    net = from_json(os.path.join(folder, "example_3.1.2.json"), convert=True)
    assert 'id_characteristic_table' in net.shunt
    assert 'step_dependency_table' in net.shunt
    assert 'id_q_capability_characteristic' in net.sgen
    assert 'reactive_capability_curve' in net.sgen
    assert 'curve_style' in net.sgen
    assert 'id_q_capability_characteristic' in net.gen
    assert 'reactive_capability_curve' in net.gen
    assert 'curve_style' in net.gen
    net = from_json(os.path.join(folder, "example_3.1.2.json"), convert=False)
    assert 'id_characteristic_table' not in net.shunt
    assert 'step_dependency_table' not in net.shunt
    assert 'id_q_capability_characteristic' not in net.sgen
    assert 'reactive_capability_curve' not in net.sgen
    assert 'curve_style' not in net.sgen
    assert 'id_q_capability_characteristic' not in net.gen
    assert 'reactive_capability_curve' not in net.gen
    assert 'curve_style' not in net.gen


def test_convert_format_renaming_characteristic_table():
    capa_df = pd.DataFrame({'id_q_capability_curve': {0: 0, 1: 0, 2: 0},
                            'p_mw': {0: -100.0, 1: 0.0, 2: 100.0},
                            'q_min_mvar': {0: -200.0, 1: -300.0, 2: -200.0},
                            'q_max_mvar': {0: 200.0, 1: 300.0, 2: 200.0}})
    capa_df['id_q_capability_curve'] = capa_df['id_q_capability_curve'].astype('Int64')
    net = from_json(os.path.join(folder, "example_3.1.2_capability.json"), convert=True)
    assert 'q_capability_curve_table' in net
    assert 'q_capability_characteristic' in net
    assert 'q_capability_curve_characteristic' not in net
    pd.testing.assert_frame_equal(net['q_capability_curve_table'], capa_df, atol=1e-5)
    net = from_json(os.path.join(folder, "example_3.1.2_capability.json"), convert=False)
    assert 'q_capability_curve_table' in net
    assert 'q_capability_characteristic' not in net
    assert 'q_capability_curve_characteristic' in net
    pd.testing.assert_frame_equal(net['q_capability_curve_table'], capa_df, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
