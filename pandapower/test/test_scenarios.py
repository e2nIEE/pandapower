# -*- coding: utf-8 -*-
__author__ = 'fmeier'
"""Run a series of tests.
"""

import pytest
import pandapower as pp
import pandas as pd
import pandapower.test as pst
import pandas.util.testing as pdt
from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.test.result_test_network_generator import add_test_bus_bus_switch


# 2 gen 2 ext_grid missing

def test_2gen_1ext_grid():
    net = pst.create_test_network2()
    net.shunt.q_kvar *= -1
    pp.create_gen(net, 2, p_kw=-100)
# Testing 2 gen and 1 ext grid >> Läuft in py3
    pp.runpp(
        net,
        init='dc',
        calculate_voltage_angles=True,
        trafo_shift=True,
        VERBOSE=False)

    ref_gen = pd.DataFrame({"p_kw": [-100., -100.],
                            "q_kvar": [447.397232056, 51.8152713776],
                            "va_degree": [0.242527288986, -143.558157703],
                            "vm_pu": [1.0, 1.0]})

    ref_bus = pd.DataFrame({"vm_pu": [1.000000, 0.956422, 1.000000, 1.000000],
                            "va_degree": [0.000000, -145.536429154, -143.558157703, 0.242527288986],
                            "p_kw": [61.87173, 30.00000, -100.00000, 0.00000],
                            "q_kvar": [-470.929980278, 2.000000, 21.8152713776, 447.397232056]
                            }).ix[:, [2, 3, 0, 1]]

    ref_ext_grid = pd.DataFrame(
        {"p_kw": 61.87173, "q_kvar": -470.927898}, index=[0])

    for element in ['ext_grid', 'bus', 'gen']:
        net['res_' + element].sort_index(axis=1, inplace=True)
        eval('ref_' + element + '.sort_index(axis=1, inplace=True)')

        eval('pdt.assert_frame_equal(net.res_' + element + ', ref_' + element + ')')


def test_0gen_2ext_grid():
    net = pst.create_test_network2()
    net.shunt.q_kvar *= -1
    pp.create_ext_grid(net, 1)
    net.gen = net.gen.drop(0)

# testing 2 ext grid and 0 gen, both EG on same trafo side >> läuft in py 3

    net.ext_grid.in_service.at[1] = False
    pp.create_ext_grid(net, 3)

    pp.runpp(
        net,
        init='dc',
        calculate_voltage_angles=True,
        trafo_shift=True,
        VERBOSE=False)

    ref_bus = pd.DataFrame({"vm_pu": [1.000000, 0.932225, 0.976965, 1.000000],
                            "va_degree": [0.000000, -155.719283, -153.641832, 0.000000],
                            "p_kw": [-0.000000, 30.000000, 0.000000, -32.993015],
                            "q_kvar": [4.08411026001, 2.000000, -28.6340014753, 27.437210083]
                            }).ix[:, [2, 3, 0, 1]]

    ref_ext_grid = pd.DataFrame({"p_kw": [-0.000000, -132.993015],
                                 "q_kvar": [4.08411026001, 27.437210083]}, index=[0, 2])

    for element in ['ext_grid', 'bus']:
        net['res_' + element].sort_index(axis=1, inplace=True)
        eval('ref_' + element + '.sort_index(axis=1, inplace=True)')

        eval('pdt.assert_frame_equal(net.res_' + element + ', ref_' + element + ')')


def test_0gen_2ext_grid_decoupled():
    net = pst.create_test_network2()
    net.gen = net.gen.drop(0)
    net.shunt.q_kvar *= -1
    pp.create_ext_grid(net, 1)
    net.ext_grid.in_service.at[1] = False
    pp.create_ext_grid(net, 3)
    net.ext_grid.in_service.at[2] = False
    auxbus = pp.create_bus(net, name="bus1", vn_kv=10.)
    pp.create_std_type(net, {"type": "cs", "r_ohm_per_km": 0.876,  "q_mm2": 35.0,
                             "endtmp_deg": 160.0, "c_nf_per_km": 260.0,
                             "imax_ka": 0.123, "x_ohm_per_km": 0.1159876}, 
                             name="NAYSEY 3x35rm/16 6/10kV" , element="line")
    pp.create_line(net, 0, auxbus, 1, name="line_to_decoupled_grid",
                   std_type="NAYSEY 3x35rm/16 6/10kV") #NAYSEY 3x35rm/16 6/10kV
    pp.create_ext_grid(net, auxbus)
    pp.create_switch(net, auxbus, 2, et="l", closed=0, type="LS")
    pp.runpp(
        net,
        init='dc',
        calculate_voltage_angles=True,
        VERBOSE=False)

    ref_bus = pd.DataFrame({"vm_pu": [1.000000, 0.930961, 0.975764, 0.998865, 1.0],
                            "va_degree": [0.000000, -155.752225311, -153.669395244, -0.0225931152895, 0.0],
                            "p_kw": [-133.158732, 30.000000, 0.000000, 100.000000, 0.000000],
                            "q_kvar": [39.5843982697, 2.000000, -28.5636406913, 0.000000, 0.000000]
                            })
    ref_ext_grid = pd.DataFrame({"p_kw": [-133.158732, -0.000000],
                                 "q_kvar": [39.5843982697, -0.000000]}, index=[0, 3])

    for element in ['ext_grid', 'bus']:
        net['res_' + element].sort_index(axis=1, inplace=True)
        eval('ref_' + element + '.sort_index(axis=1, inplace=True)')

        eval('pdt.assert_frame_equal(net.res_' + element + ', ref_' + element + ')')





def test_bus_bus_switch_at_eg():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, name="bus1")
    b2 = pp.create_bus(net, name="bus2")
    b3 = pp.create_bus(net, name="bus3")

    pp.create_ext_grid(net, b1)

    pp.create_switch(net, b1, et="b", element=1)
    pp.create_line(net, b2, b3, 1, name="line1",
                   std_type="NAYY 4x150 SE")

    pp.create_load(net, b3, p_kw=10, q_kvar=0, name="load1")

    runpp_with_consistency_checks(net)


def test_bb_switch():
    net = pp.create_empty_network()
    net = add_test_bus_bus_switch(net)
    runpp_with_consistency_checks(net)


if __name__ == "__main__":
    pytest.main(["test_scenarios.py"])
    
#    pytest.main(["test_scenarios.py::test_bus_bus_switch_at_eg"])
#    pytest.main(["test_scenarios.py::test_generation_only"])
    
#    test_full_grid_without_shunts()
#    test_full_grid_with_shunts()
#    test_2gen_1ext_grid()
#    test_0gen_2ext_grid_decoupled()
#    test_0gen_2ext_grid()
#    test_2gen_2ext_grid()
#    test_bus_bus_switch_at_eg()
#    test_bb_switch()
#    test_generation_only()

# TODO scaling faktoren
# TODO trafo stufensteller test
