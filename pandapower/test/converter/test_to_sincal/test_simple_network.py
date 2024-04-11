import os
import numpy as np
import pandapower as pp
import pytest
from pandapower.converter.sincal.pp2sincal.util.main import initialize, finalize, pp_preparation, convert_pandapower_net, net_preparation
from pandapower.test import test_path
from pandapower.test.converter.test_to_sincal.result_comparison import compare_results

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

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

plotting = True


@pytest.fixture
def simple_network_pandapower_1():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 1, name='Station1')
    b2 = pp.create_bus(net, 1, name='Station2')
    pp.create_line_from_parameters(net, b1, b2, 0.5, 0.1, 0.4, 0.0, 1.0)
    pp.create_load(net, b2, 1)
    pp.create_ext_grid(net, b1)
    return net, 'net1'


@pytest.fixture
def simple_network_pandapower_2():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 20, name='Station1')
    b2 = pp.create_bus(net, 0.4, name='Station2')
    b3 = pp.create_bus(net, 0.4, name='Station3')
    pp.create_line_from_parameters(net, b2, b3, 0.2, 0.1, 0.4, 0.0, 1.0)
    pp.create_load(net, b3, 0.3)
    pp.create_transformer_from_parameters(net, b1, b2, 0.63, 20, 0.4, 0, 8, 0, 0)
    pp.create_ext_grid(net, b1)
    return net, 'net2'


@pytest.fixture
def breaker_network_pandapower_1():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 20, name='Station1')
    b2 = pp.create_bus(net, 0.4, name='Station2')
    b3 = pp.create_bus(net, 0.4, name='Station3')
    b4 = pp.create_bus(net, 0.4, name='Station4')
    b5 = pp.create_bus(net, 0.4, name='Station5')
    b6 = pp.create_bus(net, 0.4, name='Station6')
    pp.create_line_from_parameters(net, b5, b6, 0.1, 0.1, 0.4, 0.0, 1.0)
    pp.create_line_from_parameters(net, b4, b5, 0.1, 0.1, 0.4, 0.0, 1.0)
    pp.create_line_from_parameters(net, b3, b5, 0.1, 0.1, 0.4, 0.0, 1.0)
    pp.create_switch(net, b2, b3, et='b')
    pp.create_switch(net, b2, b4, et='b', closed=False)
    pp.create_load(net, b6, 0.3)
    pp.create_transformer_from_parameters(net, b1, b2, 0.63, 20, 0.4, 0, 8, 0, 0)
    pp.create_ext_grid(net, b1)
    return net, 'breaker1'


@pytest.fixture
def breaker_network_pandapower_2():
    net_pp = pp.create_empty_network()

    # Create Buses for External Connection
    b1 = pp.create_bus(net_pp, 20, name='Station')
    b2 = pp.create_bus(net_pp, 0.4, name='Station')
    b3 = pp.create_bus(net_pp, 0.4, name='Station')
    b4 = pp.create_bus(net_pp, 0.4, name='Station')
    b5 = pp.create_bus(net_pp, 0.4, name='Station')
    b6 = pp.create_bus(net_pp, 0.4, name='Station')

    # External Grid
    pp.create_ext_grid(net_pp, b1)
    net_pp.ext_grid['Element_ID'] = None
    # Trafo
    t1 = pp.create_transformer_from_parameters(net_pp, b1, b2, 0.63, 20, 0.4, 0, 8, 0, 0, name='Trafo1')
    t2 = pp.create_transformer_from_parameters(net_pp, b1, b3, 0.63, 20, 0.4, 0, 8, 0, 0, name='Trafo2')
    net_pp.trafo['Element_ID'] = None

    # Line
    l1 = pp.create_line_from_parameters(net_pp, b3, b4, 0.1, 0.1, 0.4, 0.0, 1.0, name="Line1")
    l2 = pp.create_line_from_parameters(net_pp, b2, b4, 0.1, 0.1, 0.4, 0.0, 1.0, name="Line2")
    l3 = pp.create_line_from_parameters(net_pp, b2, b5, 0.1, 0.1, 0.4, 0.0, 1.0, name="Line3")
    _ = pp.create_line_from_parameters(net_pp, b5, b6, 0.1, 0.1, 0.4, 0.0, 1.0, name="Line4")
    _ = pp.create_line_from_parameters(net_pp, b4, b6, 0.1, 0.1, 0.4, 0.0, 1.0, name="Line5")
    net_pp.line['Element_ID'] = None

    # Switches
    pp.create_switch(net_pp, bus=b1, element=t1, et='t')
    pp.create_switch(net_pp, bus=b1, element=t2, et='t', closed=False)
    pp.create_switch(net_pp, bus=b3, element=l1, et='l')
    pp.create_switch(net_pp, bus=b2, element=l2, et='l')
    pp.create_switch(net_pp, bus=b2, element=l3, et='l', closed=False)
    net_pp.switch['Element_ID'] = None

    # Load
    pp.create_load(net_pp, b6, 0.3, name="L1")
    net_pp.load['Element_ID'] = None

    return net_pp, 'breaker2'


@pytest.fixture
def gen_network_pandapower():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 20, name='Station1')
    b2 = pp.create_bus(net, 0.4, name='Station2')
    b3 = pp.create_bus(net, 0.4, name='Station3')
    b4 = pp.create_bus(net, 0.4, name='Station4')
    pp.create_line_from_parameters(net, b2, b3, 0.2, 0.1, 0.4, 0.0, 1.0)
    pp.create_line_from_parameters(net, b3, b4, 0.2, 0.1, 0.4, 0.0, 1.0)
    pp.create_load(net, b3, 0.3)
    pp.create_gen(net, b4, 0.3, 1.09)
    pp.create_sgen(net, b4, 0.4, 0.1)
    pp.create_transformer_from_parameters(net, b1, b2, 0.63, 20, 0.4, 0, 8, 0, 0)
    pp.create_ext_grid(net, b1)
    return net, 'gen'


@pytest.fixture
def storage_network_pandapower():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 20, name='Station1')
    b2 = pp.create_bus(net, 0.4, name='Station2')
    b3 = pp.create_bus(net, 0.4, name='Station3')
    b4 = pp.create_bus(net, 0.4, name='Station4')
    pp.create_line_from_parameters(net, b2, b3, 0.2, 0.1, 0.4, 0.0, 1.0)
    pp.create_line_from_parameters(net, b3, b4, 0.2, 0.1, 0.4, 0.0, 1.0)
    pp.create_storage(net, b3, 0.3, 10)
    pp.create_storage(net, b4, -0.2, 10)
    pp.create_transformer_from_parameters(net, b1, b2, 0.63, 20, 0.4, 0, 8, 0, 0)
    pp.create_ext_grid(net, b1)
    return net, 'storage'


@pytest.fixture
def dcline_network_pandapower():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 1, name='Station1')
    b2 = pp.create_bus(net, 1, name='Station2')
    b3 = pp.create_bus(net, 1, name='Station3')
    pp.create_line_from_parameters(net, b1, b3, 0.2, 0.1, 0.4, 0.0, 1.0)
    pp.create_dcline(net, b1, b2, 1, 0, 0, 1., 1.01)
    pp.create_dcline(net, b2, b3, 1, 0, 0, 1.01, 1.02)
    pp.create_load(net, b2, 1)
    pp.create_ext_grid(net, b1)
    pp.create_ext_grid(net, b2, 1.01)
    return net, 'dcline'
@pytest.mark.skipif(simulation is None, reason=r'you need a sincal instance!')
def test_simple_network_1(simple_network_pandapower_1):
    net_pp, name = simple_network_pandapower_1
    output_folder = os.path.join(test_path, 'converter', 'test_to_sincal', 'results', 'simple')
    file_name = name + '.sin'
    use_active_net = False
    sincal_interaction = False
    use_ui = False
    net, app, sim, doc = initialize(output_folder, file_name, use_active_net, use_ui, sincal_interaction)
    _ = pp_preparation(net_pp)
    net_preparation(net, net_pp, doc)

    net.SetParameter("NetworkLevel", 1)
    station1 = net.CreateNode("Station1")
    station2 = net.CreateNode("Station2")
    load = net.CreateElement("Load", "L1", station2)
    ext_grid = net.CreateElement("Infeeder", "Ext_Grid_1", station1)
    line = net.CreateElement('Line', 'Line1', station1, station2)

    load.SetValue('P', 1)
    ext_grid.SetValue('Flag_Lf', 3)
    ext_grid.SetValue('u', 100)
    line.SetValue('l', 0.5)
    line.SetValue('Ith', 1.0)

    station1.CreateGraphic(net_pp.bus_geodata.at[0, 'x'], net_pp.bus_geodata.at[0, 'y'])
    station2.CreateGraphic(net_pp.bus_geodata.at[1, 'x'], net_pp.bus_geodata.at[1, 'y'])
    load.CreateGraphic(net_pp.bus_geodata.at[1, 'x'], net_pp.bus_geodata.at[1, 'y'])
    ext_grid.CreateGraphic(net_pp.bus_geodata.at[0, 'x'], net_pp.bus_geodata.at[0, 'y'])
    line.CreateGraphic()

    load.Update()
    ext_grid.Update()
    line.Update()

    diff_u, diff_deg = compare_results(net, net_pp, sim)

    assert (all(np.isclose(np.zeros(len(diff_u)), diff_u, atol=1 * 10 ** -6)))
    assert (all(np.isclose(np.zeros(len(diff_deg)), diff_deg, atol=1 * 10 ** -6)))

    finalize(net, output_folder, file_name, app, sim, doc, sincal_interaction)

@pytest.mark.skipif(simulation is None, reason=r'you need a sincal instance!')
def test_simple_network_2(simple_network_pandapower_2):
    net_pp, name = simple_network_pandapower_2
    output_folder = os.path.join(test_path, 'converter', 'test_to_sincal', 'results', 'simple')
    file_name = name + '.sin'
    use_active_net = False
    sincal_interaction = False
    use_ui = False
    net, app, sim, doc = initialize(output_folder, file_name, use_active_net, use_ui, sincal_interaction)
    _ = pp_preparation(net_pp)
    net_preparation(net, net_pp, doc)

    vls = dict()
    for vn_kv in [0.4, 20]:
        vl = net.CreateObject('VoltageLevel')
        vl.SetValue('Name', 'Level_' + str(vn_kv))
        vl.SetValue('Un', vn_kv)
        vl.SetValue('Uop', vn_kv)
        vl.Update()
        vls[vn_kv] = vl.GetValue('VoltLevel_ID')
    net.SetParameter("NetworkLevel", vls[20])
    station1 = net.CreateNode("Station1")
    net.SetParameter("NetworkLevel", vls[0.4])
    station2 = net.CreateNode("Station2")
    station3 = net.CreateNode("Station3")
    load = net.CreateElement("Load", "L1", station3)
    ext_grid = net.CreateElement("Infeeder", "Ext_Grid_1", station1)
    line = net.CreateElement('Line', 'Line1', station2, station3)
    trafo = net.CreateElement('TwoWindingTransformer', 'Trafo1', station1, station2)

    load.SetValue('P', 0.3)
    ext_grid.SetValue('Flag_Lf', 3)
    ext_grid.SetValue('u', 100)
    line.SetValue('l', 0.2)
    line.SetValue('Ith', 1.0)
    trafo.SetValue('Sn', 0.63)

    station1.CreateGraphic(net_pp.bus_geodata.at[0, 'x'], net_pp.bus_geodata.at[0, 'y'])
    station2.CreateGraphic(net_pp.bus_geodata.at[1, 'x'], net_pp.bus_geodata.at[1, 'y'])
    station3.CreateGraphic(net_pp.bus_geodata.at[2, 'x'], net_pp.bus_geodata.at[2, 'y'])
    load.CreateGraphic(net_pp.bus_geodata.at[2, 'x'], net_pp.bus_geodata.at[2, 'y'])
    ext_grid.CreateGraphic(net_pp.bus_geodata.at[0, 'x'], net_pp.bus_geodata.at[0, 'y'])
    line.CreateGraphic()
    trafo.CreateGraphic()

    load.Update()
    ext_grid.Update()
    line.Update()
    trafo.Update()

    diff_u, diff_deg = compare_results(net, net_pp, sim)

    assert (all(np.isclose(np.zeros(len(diff_u)), diff_u, atol=1 * 10 ** -6)))
    assert (all(np.isclose(np.zeros(len(diff_deg)), diff_deg, atol=1 * 10 ** -6)))

    finalize(net, output_folder, file_name, app, sim, doc, sincal_interaction)

@pytest.mark.skipif(simulation is None, reason=r'you need a sincal instance!')
def test_breaker_network_1(breaker_network_pandapower_1):
    net_pp, name = breaker_network_pandapower_1
    output_folder = os.path.join(test_path, 'converter', 'test_to_sincal', 'results', 'simple')
    file_name = name + '.sin'
    use_active_net = False
    sincal_interaction = False
    use_ui = False
    net, app, sim, doc = initialize(output_folder, file_name, use_active_net, use_ui, sincal_interaction)
    _ = pp_preparation(net_pp)
    net_preparation(net, net_pp, doc)

    vls = dict()
    for vn_kv in [0.4, 20]:
        vl = net.CreateObject('VoltageLevel')
        vl.SetValue('Name', 'Level_' + str(vn_kv))
        vl.SetValue('Un', vn_kv)
        vl.SetValue('Uop', vn_kv)
        vl.Update()
        vls[vn_kv] = vl.GetValue('VoltLevel_ID')
    net.SetParameter("NetworkLevel", vls[20])
    station1 = net.CreateNode("Station1")
    net.SetParameter("NetworkLevel", vls[0.4])
    station2 = net.CreateNode("Station2")
    station3 = net.CreateNode("Station3")
    station4 = net.CreateNode("Station4")
    station5 = net.CreateNode("Station5")
    station6 = net.CreateNode("Station6")
    load = net.CreateElement("Load", "L1", station6)
    ext_grid = net.CreateElement("Infeeder", "Ext_Grid_1", station1)
    line1 = net.CreateElement('Line', 'Line1', station2, station3)
    line2 = net.CreateElement('Line', 'Line1', station2, station4)
    line3 = net.CreateElement('Line', 'Line1', station3, station5)
    line4 = net.CreateElement('Line', 'Line1', station4, station5)
    line5 = net.CreateElement('Line', 'Line1', station5, station6)
    trafo = net.CreateElement('TwoWindingTransformer', 'Trafo1', station1, station2)

    load.SetValue('P', 0.3)
    ext_grid.SetValue('Flag_Lf', 3)
    ext_grid.SetValue('u', 100)
    line5.SetValue('l', 0.1)
    line5.SetValue('Ith', 1.0)
    line4.SetValue('l', 0.1)
    line4.SetValue('Ith', 1.0)
    line3.SetValue('l', 0.1)
    line3.SetValue('Ith', 1.0)
    line2.SetValue('Flag_LineTyp', 3)
    line2.SetValue('Flag_State', 0)
    line1.SetValue('Flag_LineTyp', 3)
    line2.SetValue('l', 0.0)
    line1.SetValue('l', 0.0)
    trafo.SetValue('Sn', 0.63)

    station1.CreateGraphic(net_pp.bus_geodata.at[0, 'x'], net_pp.bus_geodata.at[0, 'y'])
    station2.CreateGraphic(net_pp.bus_geodata.at[1, 'x'], net_pp.bus_geodata.at[1, 'y'])
    station3.CreateGraphic(net_pp.bus_geodata.at[2, 'x'], net_pp.bus_geodata.at[2, 'y'])
    station4.CreateGraphic(net_pp.bus_geodata.at[3, 'x'], net_pp.bus_geodata.at[3, 'y'])
    station5.CreateGraphic(net_pp.bus_geodata.at[4, 'x'], net_pp.bus_geodata.at[4, 'y'])
    station6.CreateGraphic(net_pp.bus_geodata.at[5, 'x'], net_pp.bus_geodata.at[5, 'y'])
    load.CreateGraphic(net_pp.bus_geodata.at[5, 'x'], net_pp.bus_geodata.at[5, 'y'])
    ext_grid.CreateGraphic(net_pp.bus_geodata.at[0, 'x'], net_pp.bus_geodata.at[0, 'y'])
    line1.CreateGraphic()
    line2.CreateGraphic()
    line3.CreateGraphic()
    line4.CreateGraphic()
    line5.CreateGraphic()
    trafo.CreateGraphic()

    load.Update()
    ext_grid.Update()
    line1.Update()
    line2.Update()
    line3.Update()
    line4.Update()
    line5.Update()
    trafo.Update()

    diff_u, diff_deg = compare_results(net, net_pp, sim)

    assert (all(np.isclose(np.zeros(len(diff_u)), diff_u, atol=1 * 10 ** -6)))
    assert (all(np.isclose(np.zeros(len(diff_deg)), diff_deg, atol=1 * 10 ** -6)))

    finalize(net, output_folder, file_name, app, sim, doc, sincal_interaction)

@pytest.mark.skipif(simulation is None, reason=r'you need a sincal instance!')
def test_breaker_network_2(breaker_network_pandapower_2):
    net_pp, name = breaker_network_pandapower_2

    # Initialize SIncal Environment
    output_folder = os.path.join(test_path, 'converter', 'test_to_sincal', 'results', 'simple')
    file_name = name + '.sin'
    use_active_net = False
    sincal_interaction = False
    use_ui = False
    net, app, sim, doc = initialize(output_folder, file_name, use_active_net, use_ui, sincal_interaction)
    _ = pp_preparation(net_pp)
    net_preparation(net, net_pp, doc)

    # Define Voltage Levels
    vls = dict()
    for vn_kv in net_pp.bus.vn_kv.unique():
        vl = net.CreateObject('VoltageLevel')
        vl.SetValue('Name', 'Level_' + str(vn_kv))
        vl.SetValue('Un', vn_kv)
        vl.SetValue('Uop', vn_kv)
        vl.Update()
        vls[vn_kv] = vl.GetValue('VoltLevel_ID')

    # Network Level
    net.SetParameter("NetworkLevel", vls[20])
    station1 = net.CreateNode("bus0")
    net.SetParameter("NetworkLevel", vls[0.4])

    # Create Nodes
    station2 = net.CreateNode("bus1")
    station3 = net.CreateNode("bus2")
    station4 = net.CreateNode("bus3")
    station5 = net.CreateNode("bus4")
    station6 = net.CreateNode("bus5")

    station1.CreateGraphic(net_pp.bus_geodata.at[0, 'x'], net_pp.bus_geodata.at[0, 'y'])
    station2.CreateGraphic(net_pp.bus_geodata.at[1, 'x'], net_pp.bus_geodata.at[1, 'y'])
    station3.CreateGraphic(net_pp.bus_geodata.at[2, 'x'], net_pp.bus_geodata.at[2, 'y'])
    station4.CreateGraphic(net_pp.bus_geodata.at[3, 'x'], net_pp.bus_geodata.at[3, 'y'])
    station5.CreateGraphic(net_pp.bus_geodata.at[4, 'x'], net_pp.bus_geodata.at[4, 'y'])
    station6.CreateGraphic(net_pp.bus_geodata.at[5, 'x'], net_pp.bus_geodata.at[5, 'y'])

    # Create Load
    load = net.CreateElement("Load", "L1", station6)
    load.SetValue('P', 0.3)
    load.CreateGraphic(net_pp.bus_geodata.at[5, 'x'], net_pp.bus_geodata.at[5, 'y'])
    load.Update()

    # Create ExtGrid
    ext_grid = net.CreateElement("Infeeder", "Ext_Grid_1", station1)
    ext_grid.SetValue('Flag_Lf', 3)
    ext_grid.SetValue('u', 100)
    ext_grid.CreateGraphic(net_pp.bus_geodata.at[0, 'x'], net_pp.bus_geodata.at[0, 'y'])
    ext_grid.Update()

    # Create Lines
    line1 = net.CreateElement('Line', 'Line1', station2, station4)
    line2 = net.CreateElement('Line', 'Line2', station3, station4)
    line3 = net.CreateElement('Line', 'Line3', station3, station5)
    line4 = net.CreateElement('Line', 'Line4', station4, station6)
    line5 = net.CreateElement('Line', 'Line5', station5, station6)

    # Saving Element_ID at pandapower Grid for later processing
    l_EID = line1.GetValue("Element_ID")
    net_pp.line.Element_ID.at[0] = l_EID
    l_EID = line2.GetValue("Element_ID")
    net_pp.line.Element_ID.at[1] = l_EID
    l_EID = line3.GetValue("Element_ID")
    net_pp.line.Element_ID.at[2] = l_EID
    l_EID = line4.GetValue("Element_ID")
    net_pp.line.Element_ID.at[3] = l_EID
    l_EID = line5.GetValue("Element_ID")
    net_pp.line.Element_ID.at[4] = l_EID

    line1.SetValue('l', 0.1)
    line1.SetValue('Ith', 1.0)
    line2.SetValue('l', 0.1)
    line2.SetValue('Ith', 1.0)
    line3.SetValue('l', 0.1)
    line3.SetValue('Ith', 1.0)
    line4.SetValue('l', 0.1)
    line4.SetValue('Ith', 1.0)
    line5.SetValue('l', 0.1)
    line5.SetValue('Ith', 1.0)
    line1.CreateGraphic()
    line1.Update()
    line2.CreateGraphic()
    line2.Update()
    line3.CreateGraphic()
    line3.Update()
    line4.CreateGraphic()
    line4.Update()
    line5.CreateGraphic()
    line5.Update()

    # Create Trafo
    trafo1 = net.CreateElement('TwoWindingTransformer', 'Trafo1', station1, station2)
    trafo2 = net.CreateElement('TwoWindingTransformer', 'Trafo2', station1, station3)

    # Saving Element_ID at pandapower Grid for later processing
    t_EID = trafo1.GetValue("Element_ID")
    net_pp.trafo.Element_ID.at[0] = t_EID
    t_EID = trafo2.GetValue("Element_ID")
    net_pp.trafo.Element_ID.at[1] = t_EID

    trafo1.SetValue('Sn', 0.63)
    trafo2.SetValue('Sn', 0.63)
    trafo1.CreateGraphic()
    trafo1.Update()
    trafo2.CreateGraphic()
    trafo2.Update()

    # Create Breaker
    # Trafo Breaker et='t'
    br_tr1 = net.CreateObject('Breaker')
    br_tr1.SetValue("Breaker.Flag_State", 1)  # 0: Open, 1: Closed
    br_tr1.SetValue("Name", "switch_" + str(0))
    br_tr2 = net.CreateObject('Breaker')
    br_tr2.SetValue("Breaker.Flag_State", 0)  # 0: Open, 1: Closed
    br_tr2.SetValue("Name", "switch_" + str(1))

    T_EID = net_pp.trafo.Element_ID.at[0]
    tr1 = net.GetCommonObject("TwoWindingTransformer", T_EID)  # 1
    T_TID = tr1.GetValue("Terminal1.Terminal_ID")
    br_tr1.SetValue("Terminal_ID", T_TID)
    T_EID = net_pp.trafo.Element_ID.at[1]
    tr2 = net.GetCommonObject("TwoWindingTransformer", T_EID)  # 1
    T_TID = tr2.GetValue("Terminal1.Terminal_ID")
    br_tr2.SetValue("Terminal_ID", T_TID)

    br_tr1.CreateGraphic()
    br_tr1.Update()
    br_tr2.CreateGraphic()
    br_tr2.Update()

    # Line Breaker et='l'
    br_l1 = net.CreateObject('Breaker')
    br_l1.SetValue("Breaker.Flag_State", 1)  # 0: Open, 1: Closed
    br_l1.SetValue("Name", "switch_" + str(2))
    br_l2 = net.CreateObject('Breaker')
    br_l2.SetValue("Breaker.Flag_State", 1)  # 0: Open, 1: Closed
    br_l2.SetValue("Name", "switch_" + str(3))
    br_l3 = net.CreateObject('Breaker')
    br_l3.SetValue("Breaker.Flag_State", 0)  # 0: Open, 1: Closed
    br_l3.SetValue("Name", "switch_" + str(4))

    l_EID = net_pp.line.Element_ID.at[0]
    l1 = net.GetCommonObject("Line", l_EID)  # Get Element ID

    l_TID = l1.GetValue("Terminal1.Terminal_ID")  # Get Terminal ID
    br_l1.SetValue("Terminal_ID", l_TID)

    l_EID = net_pp.line.Element_ID.at[1]
    l2 = net.GetCommonObject("Line", l_EID)  # Get Element ID

    l_TID = l2.GetValue("Terminal1.Terminal_ID")  # Get Terminal ID
    br_l2.SetValue("Terminal_ID", l_TID)

    l_EID = net_pp.line.Element_ID.at[2]
    l3 = net.GetCommonObject("Line", l_EID)  # Get Element ID

    l_TID = l3.GetValue("Terminal1.Terminal_ID")  # Get Terminal ID
    br_l3.SetValue("Terminal_ID", l_TID)

    br_l1.CreateGraphic()
    br_l1.Update()
    br_l2.CreateGraphic()
    br_l2.Update()
    br_l3.CreateGraphic()
    br_l3.Update()

    # Comparison
    diff_u, diff_deg = compare_results(net, net_pp, sim)

    assert (all(np.isclose(np.zeros(len(diff_u)), diff_u, atol=1 * 10 ** -6)))
    assert (all(np.isclose(np.zeros(len(diff_deg)), diff_deg, atol=1 * 10 ** -6)))

    finalize(net, output_folder, file_name, app, sim, doc, sincal_interaction)

@pytest.mark.skipif(simulation is None, reason=r'you need a sincal instance!')
def test_gen_network(gen_network_pandapower):
    net_pp, name = gen_network_pandapower
    output_folder = os.path.join(test_path, 'converter', 'test_to_sincal', 'results', 'simple')
    file_name = name + '.sin'
    use_active_net = False
    sincal_interaction = False
    use_ui = False
    net, app, sim, doc = initialize(output_folder, file_name, use_active_net, use_ui, sincal_interaction)
    _ = pp_preparation(net_pp)
    net_preparation(net, net_pp, doc)

    vls = dict()
    for vn_kv in [0.4, 20]:
        vl = net.CreateObject('VoltageLevel')
        vl.SetValue('Name', 'Level_' + str(vn_kv))
        vl.SetValue('Un', vn_kv)
        vl.SetValue('Uop', vn_kv)
        vl.Update()
        vls[vn_kv] = vl.GetValue('VoltLevel_ID')
    net.SetParameter("NetworkLevel", vls[20])
    station1 = net.CreateNode("Station1")
    net.SetParameter("NetworkLevel", vls[0.4])
    station2 = net.CreateNode("Station2")
    station3 = net.CreateNode("Station3")
    station4 = net.CreateNode("Station4")
    load = net.CreateElement("Load", "L1", station3)
    sgen = net.CreateElement("DCInfeeder", "DC1", station4)
    gen = net.CreateElement("SynchronousMachine", "GEN1", station4)
    ext_grid = net.CreateElement("Infeeder", "Ext_Grid_1", station1)
    line1 = net.CreateElement('Line', 'Line1', station2, station3)
    line2 = net.CreateElement('Line', 'Line2', station3, station4)
    trafo = net.CreateElement('TwoWindingTransformer', 'Trafo1', station1, station2)
    load.SetValue('P', 0.3)
    sgen.SetValue('P', 0.4)
    sgen.SetValue('fP', 1.)
    sgen.SetValue('Q', 0.1)
    sgen.SetValue('fQ', 1.)
    gen.SetValue('P', 0.3)
    gen.SetValue('fP', 1.)
    gen.SetValue('Flag_Lf', 11)
    gen.SetValue('u', 1.09 * 100)
    ext_grid.SetValue('Flag_Lf', 3)
    ext_grid.SetValue('u', 100)
    line1.SetValue('l', 0.2)
    line1.SetValue('Ith', 1.0)
    line2.SetValue('l', 0.2)
    line2.SetValue('Ith', 1.0)
    trafo.SetValue('Sn', 0.63)

    station1.CreateGraphic(net_pp.bus_geodata.at[0, 'x'], net_pp.bus_geodata.at[0, 'y'])
    station2.CreateGraphic(net_pp.bus_geodata.at[1, 'x'], net_pp.bus_geodata.at[1, 'y'])
    station3.CreateGraphic(net_pp.bus_geodata.at[2, 'x'], net_pp.bus_geodata.at[2, 'y'])
    station4.CreateGraphic(net_pp.bus_geodata.at[3, 'x'], net_pp.bus_geodata.at[3, 'y'])
    load.CreateGraphic(net_pp.bus_geodata.at[2, 'x'], net_pp.bus_geodata.at[2, 'y'])
    ext_grid.CreateGraphic(net_pp.bus_geodata.at[0, 'x'], net_pp.bus_geodata.at[0, 'y'])
    sgen.CreateGraphic(net_pp.bus_geodata.at[3, 'x'], net_pp.bus_geodata.at[3, 'y'])
    gen.CreateGraphic(net_pp.bus_geodata.at[3, 'x'], net_pp.bus_geodata.at[3, 'y'])
    line1.CreateGraphic()
    line2.CreateGraphic()
    trafo.CreateGraphic()

    load.Update()
    ext_grid.Update()
    sgen.Update()
    gen.Update()
    line1.Update()
    line2.Update()
    trafo.Update()

    diff_u, diff_deg = compare_results(net, net_pp, sim)

    assert (all(np.isclose(np.zeros(len(diff_u)), diff_u, atol=1 * 10 ** -6)))
    assert (all(np.isclose(np.zeros(len(diff_deg)), diff_deg, atol=1 * 10 ** -6)))

    finalize(net, output_folder, file_name, app, sim, doc, sincal_interaction)

test_networks = pytest.mark.parametrize(
    "input",
    [
        pytest.lazy_fixture('simple_network_pandapower_1'),
        pytest.lazy_fixture('simple_network_pandapower_2'),
        pytest.lazy_fixture('breaker_network_pandapower_1'),
        pytest.lazy_fixture('breaker_network_pandapower_2'),
        pytest.lazy_fixture('gen_network_pandapower'),
        pytest.lazy_fixture('storage_network_pandapower'),
        pytest.lazy_fixture('dcline_network_pandapower'),
    ],
)

@pytest.mark.skipif(simulation is None, reason=r'you need a sincal instance!')
@test_networks
def test_convert_simple_network(input):
    net_pp, name = input
    output_folder = os.path.join(test_path, 'converter', 'test_to_sincal', 'results', 'simple')
    file_name = 'conv_' + name + '.sin'
    use_active_net = False
    sincal_interaction = False
    use_ui = False
    net, app, sim, doc = initialize(output_folder, file_name, use_active_net, use_ui, sincal_interaction)
    convert_pandapower_net(net, net_pp, doc)

    diff_u, diff_deg = compare_results(net, net_pp, sim)

    assert (all(np.isclose(np.zeros(len(diff_u)), diff_u, atol=1 * 10 ** -6)))
    assert (all(np.isclose(np.zeros(len(diff_deg)), diff_deg, atol=1 * 10 ** -6)))

    finalize(net, output_folder, file_name, app, sim, doc, sincal_interaction)


if __name__ == '__main__':
    pytest.main([__file__, "-x"])
