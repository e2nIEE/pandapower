import copy
import time

import numpy as np
import pandas as pd

import pandapower as pp
from pandapower.converter.sincal.pp2sincal.util.finalization import write_to_net, close_net
from pandapower.converter.sincal.pp2sincal.util.initialization import create_simulation_environment, initialize_net
from pandapower.converter.sincal.pp2sincal.util.toolbox import _unique_naming, _number_of_elements, _scale_geo_data, \
    _adapt_area_tile, \
    _initialize_voltage_level, _set_calc_params, _adapt_geo_coordinates

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def pp_preparation(net_pp):
    '''
    Preperation of the pandapower network in order to convert it to Sincal. Each element gets a
    unique name. The function returns the number of elements which enables a nice display of the different elements.

    :param net_pp: The pandapower network that shall be converted
    :type net_pp: pandapowerNet
    :return: Number of elements
    :rtype: Tuple
    '''
    pp.set_user_pf_options(net_pp, trafo_model='pi')
    pp.runpp(net_pp, calculate_voltage_angles=True)
    _unique_naming(net_pp)
    elements = _number_of_elements(net_pp)
    _scale_geo_data(net_pp)
    return elements


def net_preparation(net, net_pp, doc):
    '''
    Prepares the user interface from Sincal to display the converted pandapower network in a correct
    manner. It returns the adapted Sincal network and the different voltage levels a bus can be assigned to.

    :param net: Sincal electrical Database Object
    :type net: object
    :param net_pp: The pandapower network that shall be converted
    :type net_pp: pandapowerNet
    :param doc: Sincal Document (Automation Object)
    :type doc: object
    :return: (net, voltage_level_dict)
    :rtype: Tuple
    '''
    _adapt_area_tile(net, net_pp)
    if not doc is None:
        doc.Reload()
    time.sleep(0.5)
    _adapt_geo_coordinates(net, net_pp)
    voltage_level_dict = _initialize_voltage_level(net, net_pp)
    _set_calc_params(net)
    return net, voltage_level_dict


def initialize(output_folder, file_name, use_active_net=False, use_ui=True,
               sincal_interaction=False, delete_files=True):
    '''
    The function returns the needed sincal object for further adjustments.

    :param output_folder: Output folder of the converted network
    :type output_folder: string
    :param file_name: File name
    :type file_name: string
    :param use_active_net: Use the active and open Sincal user interface
    :type use_active_net: boolean
    :param use_ui: Set True, if you want to open the Sincal user interface
    :type use_ui: boolean
    :param sincal_interaction: Flag if Sincal interaction is wished. Prevents closing of the Sincal network model \
    after conversion.
    :type sincal_interaction: boolean
    :param delete_files: Delete already contained .sin - files in the output folder
    :type delete_files: boolean
    :return: (Sincal Database Object, Sincal Application Object, Sincal Document)
    :rtype: Tuple
    '''
    sim, doc, app = create_simulation_environment(output_folder, file_name, use_active_net, use_ui,
                                                  sincal_interaction, delete_files)
    net = initialize_net(sim, output_folder, file_name)
    return net, app, sim, doc


def finalize(net, output_folder, file_name, app, sim, doc, sincal_interaction=False):
    '''
    The function finalizes the conversion process by closing the sincal network model (net)
    and frees all variables.

    :param net: Sincal electrical Database Object
    :type net: object
    :param output_folder: Output folder of the converted network
    :type output_folder: string
    :param file_name: File Name
    :type file_name: string
    :param app: Sincal Application Object
    :type app: object
    :param sim: Simulation Object
    :type sim: object
    :param doc: Sincal Document (Automation Object)
    :type doc: object
    :param sincal_interaction: Flag if Sincal interaction is wished. Prevents closing of the Sincal network model \
    after conversion.
    :type sincal_interaction: boolean
    :return: None
    :rtype: None
    '''
    write_to_net(net, doc)
    if not sincal_interaction:
        close_net(app, sim, doc, output_folder, file_name)


def convert_pandapower_net(net, net_pp, doc, plotting=True):
    '''
    Conversion of a pandapowerNet to a sincal network.

    :param net: Sincal electrical Database Object
    :type net: object
    :param net_pp: The pandapower network that shall be converted
    :type net_pp: pandapowerNet
    :param doc: Sincal Document (Automation Object)
    :type doc: object
    :param plotting: Flag to graphically display elements in Sincal
    :type plotting: boolean
    :return: None
    :rtype: None
    '''

    elements = pp_preparation(net_pp)
    net, voltage_level_dict = net_preparation(net, net_pp, doc)
    create_bus(net, net_pp, voltage_level_dict, plotting)
    create_load(net, net_pp, elements, plotting)
    create_sgen(net, net_pp, elements, plotting)
    create_ext_grid(net, net_pp, elements, plotting)
    create_gen(net, net_pp, elements, plotting)
    create_storage(net, net_pp, elements, plotting)
    create_dcline(net, net_pp, elements, plotting)
    create_trafo(net, net_pp, plotting)
    create_line(net, net_pp, plotting)
    create_switch(net, net_pp, plotting)


def create_bus(net, net_pp, voltage_level_dict, plotting=True, buses=None, buses_geodata=None):
    '''
    The function creates an electrical aquivalent element of the chosen pandapower-buses in the
    Sincal electrical Database Object (net). If plotting=True a graphical element will also be created.

    :param net: Sincal electrical Database Object
    :type net: object
    :param net_pp: The pandapower network that shall be converted
    :type net_pp: pandapowerNet
    :param voltage_level_dict:
    :type voltage_level_dict:
    :param plotting: Flag to graphically display Elements
    :type plotting: boolean
    :param buses: DataFrame with buses to convert, if None all buses will be converted.
    :type buses: DataFrame
    :param buses_geodata: DataFrame with buses_geodata to convert, if None all buses_geodata will be converted.
    :type buses_geodata: DataFrame, None
    :return: None
    :rtype: None
    '''
    if buses is None:
        buses = net_pp.bus
    if buses_geodata is None:
        buses_geodata = net_pp.bus_geodata
    zipped = net_pp.bus_geodata.x.astype(str) + net_pp.bus_geodata.y.astype(str)
    _, idx, no = np.unique(zipped, return_counts=True, return_index=True)
    buses['Node_ID'] = None
    for (idx, bus), (_, geo) in zip(buses.iterrows(), buses_geodata.iterrows()):
        net.SetParameter("NetworkLevel", voltage_level_dict[bus.vn_kv])
        b = net.CreateNode(bus['Sinc_Name'])
        n_id = b.GetValue("Node_ID")
        buses.Node_ID.at[idx] = n_id

        b.SetValue('Node_ID', bus.name)
        t = {'n': 1, 'b': 1, 'm': 3, 'db': 2, 'auxiliary': 3}[bus.type]
        b.SetValue('Flag_Type', t)
        if plotting:
            tile = net.GetCommonObject("GraphicAreaTile", 1)
            factor = tile.GetValue('ScalePaper')
            b.CreateGraphic(geo.x, geo.y)
            if bus.type == 'db':
                gnode = net.GetCommonObject("GraphicNode", n_id)
                coord_x = gnode.GetValue('NodeStartX')
                gnode.SetValue("SymType", 3)
                gnode.SetValue('NodeStartX', coord_x - 0.005 * factor)
                gnode.SetValue('NodeEndX', coord_x + 0.005 * factor)
                gnode.Update()
            gtext = net.GetCommonObject("GraphicText", n_id)
            pos = gtext.GetValue("Pos2")
            pos += pos + 0.015 * factor / 4
            gtext.SetValue("Pos2", pos)
            gtext.SetValue("Visible", 0)
            gtext.Update()
        b.Update()
    if len(net_pp.bus) > 500:
        net.Close()
        net.OpenEx()


def create_load(net, net_pp, elements, plotting=True, loads=None):
    '''
    The function creates an electrical aquivalent element of the chosen pandapower-loads in the
    sincal electrical Database Object (net). If plotting=True a graphical element will also be created.

    :param net: Sincal electrical Database Object
    :type net: object
    :param net_pp: The pandapower network that shall be converted
    :type net_pp: pandapowerNet
    :param elements: Number of connected elements to all busses
    :type elements: Tuple
    :param plotting: Flag to graphically display Elements
    :type plotting: boolean
    :param loads: DataFrame with loads to convert, if None all loads will be converted.
    :type loads: DataFrame, None
    :return: None
    :rtype: None
    '''
    if loads is None:
        loads = net_pp.load
    loads['Element_ID'] = None
    for idx, load in loads.iterrows():
        ld = net.CreateElement('Load', load['Sinc_Name'], net_pp.bus.loc[load.bus, 'Sinc_Name'])
        e_id = ld.GetValue("Element_ID")
        loads.Element_ID.at[idx] = e_id

        ld.SetValue('P', load.p_mw)
        ld.SetValue('Q', load.q_mvar)
        ld.SetValue('fP', load.scaling)
        ld.SetValue('fQ', load.scaling)
        geo = net_pp.bus_geodata.loc[load.bus, :]
        if plotting:
            tile = net.GetCommonObject("GraphicAreaTile", 1)
            factor = tile.GetValue('ScalePaper')
            x = 0.015 * np.sin(np.deg2rad(elements[0][load.bus] * elements[1][load.bus] + 100))
            y = np.sqrt(0.015 ** 2 - x ** 2)
            if (elements[0][load.bus] * elements[1][load.bus] + 10 > 180) and \
                    (elements[0][load.bus] * elements[1][load.bus] + 10 < 360):
                y = - y
            ld.CreateGraphic(geo.x + x * factor, geo.y + y * factor)
            t_id = ld.GetValue('Terminal1.Terminal_ID')
            gterminal = net.GetCommonObject("GraphicTerminal", t_id)
            gterminal.SetValue('SwtNodePos', 15)
            gterminal.SetValue('SwtFactor', 20)
            gterminal.Update()
            gt_id = gterminal.GetValue('GraphicText_ID')
            gtext = net.GetCommonObject("GraphicText", gt_id)
            gtext.SetValue("Pos1", x * factor / 2)
            gtext.SetValue("Pos2", y * factor / 2)
            gtext.SetValue("Visible", 0)
            gtext.Update()
            gelement = net.GetCommonObject("GraphicElement", e_id)
            gelement.SetValue('SymbolSize', 60)
            gelement.Update()
            gt_id = gelement.GetValue('GraphicText_ID1')
            gtexte = net.GetCommonObject("GraphicText", gt_id)
            pos = gtexte.GetValue("Pos2")
            gtexte.SetValue("Pos2", pos + 0.015 * factor / 4)
            gtexte.Update()
            elements[1][load.bus] += 1
        ld.Update()
    if len(net_pp.load) > 500:
        net.Close()
        net.OpenEx()


def create_sgen(net, net_pp, elements, plotting=True, sgens=None, pv=False):
    '''
    The function creates an electrical aquivalent element of the chosen pandapower-sgens in the
    sincal electrical Database Object (net). If plotting=True a graphical element will also be created.

    :param net: Sincal electrical Database Object
    :type net: object
    :param net_pp: The pandapower network that shall be converted
    :type net_pp: pandapowerNet
    :param elements: Number of connected elements to all busses
    :type elements: Tuple
    :param plotting: Flag to graphically display Elements
    :type plotting: boolean
    :param sgens: DataFrame with sgens to convert, if None all sgens will be converted.
    :type sgens: DataFrame, None
    :return: None
    :rtype: None
    '''
    if sgens is None:
        sgens = net_pp.sgen
    sgens['Element_ID'] = None
    v_max = net_pp.res_bus.vm_pu.max()
    v_min = net_pp.res_bus.vm_pu.min()
    for idx, sgen in sgens.iterrows():
        s = net.CreateElement('DCInfeeder', sgen['Sinc_Name'], net_pp.bus.loc[sgen.bus, 'Sinc_Name'])
        e_id = s.GetValue("Element_ID")
        sgens.Element_ID.at[idx] = e_id
        s.SetValue('Sn', sgen.sn_mva)
        s.SetValue('P', sgen.p_mw)
        if pv:
            s.SetValue('Flag_Lf', 4)
            s.SetValue('u', sgen.vm_pu * 100)
        else:
            s.SetValue('fP', sgen.scaling)
            s.SetValue('Q', sgen.q_mvar)
            s.SetValue('fQ', sgen.scaling)
        s.SetValue('Umax_Inverter', v_max * 100)
        s.SetValue('Umin_Inverter', v_min * 100)
        if plotting:
            tile = net.GetCommonObject("GraphicAreaTile", 1)
            factor = tile.GetValue('ScalePaper')
            x = 0.015 * np.sin(np.deg2rad(elements[0][sgen.bus] * elements[1][sgen.bus] + 100))
            y = np.sqrt(0.015 ** 2 - x ** 2)
            if (elements[0][sgen.bus] * elements[1][sgen.bus] + 10 > 180) and \
                    (elements[0][sgen.bus] * elements[1][sgen.bus] + 10 < 360):
                y = - y
            geo = net_pp.bus_geodata.loc[sgen.bus, :]
            s.CreateGraphic(geo.x + x * factor, geo.y + y * factor)
            t_id = s.GetValue('Terminal1.Terminal_ID')
            gterminal = net.GetCommonObject("GraphicTerminal", t_id)
            gterminal.SetValue('SwtNodePos', 15)
            gterminal.SetValue('SwtFactor', 20)
            gterminal.Update()
            gt_id = gterminal.GetValue('GraphicText_ID')
            gtext = net.GetCommonObject("GraphicText", gt_id)
            gtext.SetValue("Pos1", x * factor / 2)
            gtext.SetValue("Pos2", y * factor / 2)
            gtext.Update()
            gelement = net.GetCommonObject("GraphicElement", e_id)
            gelement.SetValue('SymbolSize', 30)
            gelement.Update()
            gt_id = gelement.GetValue('GraphicText_ID1')
            gtexte = net.GetCommonObject("GraphicText", gt_id)
            pos = gtexte.GetValue("Pos2")
            gtexte.SetValue("Pos2", pos + 0.015 * factor / 4)
            gtexte.Update()
            elements[1][sgen.bus] += 1
        s.Update()
    if len(net_pp.sgen) > 500:
        net.Close()
        net.OpenEx()


def create_ext_grid(net, net_pp, elements, plotting=True, ext_grids=None):
    '''
    The function creates an electrical aquivalent element of the chosen pandapower-ext_grids in the
    sincal electrical Database Object (net). If plotting=True a graphical element will also be created.

    :param net: Sincal electrical Database Object
    :type net: object
    :param net_pp: The pandapower network that shall be converted
    :type net_pp: pandapowerNet
    :param elements: Number of connected elements to all busses
    :type elements: Tuple
    :param plotting: Flag to graphically display Elements
    :type plotting: boolean
    :param ext_grids: DataFrame with ext_grids to convert, if None all ext_grids will be converted.
    :type ext_grids: DataFrame, None
    :return: None
    :rtype: None
    '''
    if ext_grids is None:
        ext_grids = net_pp.ext_grid
    ext_grids['Element_ID'] = None
    for idx, ext_grid in ext_grids.iterrows():
        ext = net.CreateElement('Infeeder', ext_grid['Sinc_Name'], net_pp.bus.loc[ext_grid.bus, 'Sinc_Name'])
        e_id = ext.GetValue("Element_ID")
        ext_grids.Element_ID.at[idx] = e_id

        ext.SetValue('Flag_Lf', 3)
        ext.SetValue('u', ext_grid.vm_pu * 100)
        if plotting:
            tile = net.GetCommonObject("GraphicAreaTile", 1)
            factor = tile.GetValue('ScalePaper')
            x = 0.015 * np.cos(np.deg2rad(elements[0][ext_grid.bus] * elements[1][ext_grid.bus] + 10))
            y = np.sqrt(0.015 ** 2 - x ** 2)
            if (elements[0][ext_grid.bus] * elements[1][ext_grid.bus] + 10 > 180) and \
                    (elements[0][ext_grid.bus] * elements[1][ext_grid.bus] + 10 < 360):
                y = - y
            geo = net_pp.bus_geodata.loc[ext_grid.bus, :]
            ext.CreateGraphic(geo.x + x * factor, geo.y + y * factor)
            t_id = ext.GetValue('Terminal1.Terminal_ID')
            gterminal = net.GetCommonObject("GraphicTerminal", t_id)
            gterminal.SetValue('SwtNodePos', 15)
            gterminal.SetValue('SwtFactor', 20)
            gterminal.Update()
            gt_id = gterminal.GetValue('GraphicText_ID')
            gtext = net.GetCommonObject("GraphicText", gt_id)
            gtext.SetValue("Pos1", x * factor / 2)
            gtext.SetValue("Pos2", y * factor / 2)
            gtext.Update()
            gelement = net.GetCommonObject("GraphicElement", e_id)
            gt_id = gelement.GetValue('GraphicText_ID1')
            gtexte = net.GetCommonObject("GraphicText", gt_id)
            pos = gtexte.GetValue("Pos2")
            gtexte.SetValue("Pos2", pos + 0.015 * factor / 4)
            gtexte.Update()
            elements[1][ext_grid.bus] += 1
        ext.Update()
    if len(net_pp.ext_grid) > 500:
        net.Close()
        net.OpenEx()


def create_trafo(net, net_pp, plotting=False, trafos=None):
    '''
    The function creates an electrical aquivalent element of the chosen pandapower-trafos in the
    sincal electrical Database Object (net). If plotting=True a graphical element will also be created.

    :param net: Sincal electrical Database Object
    :type net: object
    :param net_pp: The pandapower network that shall be converted
    :type net_pp: pandapowerNet
    :param plotting: Flag to graphically display Elements
    :type plotting: boolean
    :param trafos: DataFrame with trafos to convert, if None all trafos will be converted.
    :type trafos: DataFrame, None
    :return: None
    :rtype: None
    '''
    if trafos is None:
        trafos = net_pp.trafo
    trafos['Element_ID'] = None
    bus_ocr = pd.Series(np.ones(len(net_pp.bus)), index=net_pp.bus.index, name='sh')
    for idx, trafo in trafos.iterrows():
        t = net.CreateElement('TwoWindingTransformer', trafo['Sinc_Name'], net_pp.bus.loc[trafo.hv_bus, 'Sinc_Name'],
                              net_pp.bus.loc[trafo.lv_bus, 'Sinc_Name'])
        e_id = t.GetValue("Element_ID")
        trafos.Element_ID.at[idx] = e_id

        t.SetValue('Un1', trafo.vn_hv_kv)
        t.SetValue('Un2', trafo.vn_lv_kv)
        t.SetValue('Sn', trafo.sn_mva)
        t.SetValue('uk', trafo.vk_percent)
        t.SetValue('ur', trafo.vkr_percent)
        t.SetValue('Vfe', trafo.pfe_kw)
        t.SetValue('i0', trafo.i0_percent)
        t.SetValue('rohl', trafo.tap_min)
        t.SetValue('rohm', trafo.tap_neutral)
        t.SetValue('AddRotate', trafo.shift_degree)
        t.SetValue('rohu', trafo.tap_max)
        t.SetValue('roh', trafo.tap_pos)
        t.SetValue('ukr', trafo.tap_step_percent)
        t.SetValue('Flag_fr', trafo.df)
        if not trafo.tap_side is None:
            t.SetValue('Flag_ConNode', {'hv': 1, 'lv': 2}[trafo.tap_side])
            t.SetValue('Flag_Input', 4099)
        if not trafo.std_type is None:
            t.SetValue('Description', 'std_type: ' + trafo.std_type)
        if plotting:
            tile = net.GetCommonObject("GraphicAreaTile", 1)
            factor = tile.GetValue('ScalePaper')
            t.CreateGraphic()
            geo_hv = net_pp.bus_geodata.loc[trafo.hv_bus, :]
            geo_lv = net_pp.bus_geodata.loc[trafo.lv_bus, :]
            t_id = t.GetValue('Terminal1.Terminal_ID')
            gterminal = net.GetCommonObject("GraphicTerminal", t_id)
            gterminal.SetValue('SwtNodePos', 15)
            gterminal.SetValue('SwtFactor', 20)
            hb = trafo.hv_bus
            nth = net_pp.trafo.loc[(net_pp.trafo.hv_bus == hb) | (net_pp.trafo.lv_bus == hb)].index
            if len(nth) > 1:
                n_id = net_pp.bus.loc[hb, 'Node_ID']
                gn = net.GetCommonObject("GraphicNode", n_id)
                nxs = gn.GetValue('NodeStartX')
                nxe = gn.GetValue('NodeEndX')
                nx = (nxe + nxs) / 2
                gterminal.SetValue('PosX', nx + bus_ocr.loc[hb] * 0.02 * factor)
                bus_ocr.loc[hb] += 1
            gterminal.Update()
            gt_id = gterminal.GetValue('GraphicText_ID')
            gtext = net.GetCommonObject("GraphicText", gt_id)
            x = 0.1 * (geo_lv.x - geo_hv.x)
            y = 0.1 * (geo_lv.y - geo_hv.y)
            gtext.SetValue("Pos1", x)
            gtext.SetValue("Pos2", y)
            gtext.Update()

            t_id = t.GetValue('Terminal2.Terminal_ID')
            gterminal = net.GetCommonObject("GraphicTerminal", t_id)
            gterminal.SetValue('SwtNodePos', 15)
            gterminal.SetValue('SwtFactor', 20)
            lb = trafo.lv_bus
            ntl = net_pp.trafo.loc[(net_pp.trafo.hv_bus == lb) | (net_pp.trafo.lv_bus == lb)].index
            if len(ntl) > 1:
                n_id = net_pp.bus.loc[lb, 'Node_ID']
                gn = net.GetCommonObject("GraphicNode", n_id)
                nxs = gn.GetValue('NodeStartX')
                nxe = gn.GetValue('NodeEndX')
                nx = (nxe + nxs) / 2
                gterminal.SetValue('PosX', nx + bus_ocr.loc[lb] * 0.02 * factor)
                bus_ocr.loc[lb] += 1
            gterminal.Update()
            gt_id = gterminal.GetValue('GraphicText_ID')
            gtext = net.GetCommonObject("GraphicText", gt_id)
            x = 0.1 * (geo_hv.x - geo_lv.x)
            y = 0.1 * (geo_hv.y - geo_lv.y)
            gtext.SetValue("Pos1", x)
            gtext.SetValue("Pos2", y)
            gtext.Update()
            gelement = net.GetCommonObject("GraphicElement", e_id)
            gt_id = gelement.GetValue('GraphicText_ID1')
            gtexte = net.GetCommonObject("GraphicText", gt_id)
            pos = gtexte.GetValue("Pos2")
            gtexte.SetValue("Pos2", pos + 0.015 * factor / 4)
            gtexte.Update()
        t.Update()
    bus_ocr = bus_ocr.loc[bus_ocr != 0]
    for b in bus_ocr.index:
        if bus_ocr.loc[b] <= 1:
            continue
        n_id = net_pp.bus.loc[b, 'Node_ID']
        gn = net.GetCommonObject("GraphicNode", n_id)
        coords_x = gn.GetValue('NodeStartX')
        coorde_x = gn.GetValue('NodeEndX')
        coord_x = (coorde_x + coords_x) / 2
        gn.SetValue("SymType", 3)
        gn.SetValue('NodeStartX', coord_x - (bus_ocr.loc[b] - 1) * 0.02 * factor)
        gn.SetValue('NodeEndX', coord_x + (bus_ocr.loc[b] - 1) * 0.02 * factor)
        gn.Update()
    if len(net_pp.trafo) > 500:
        net.Close()
        net.OpenEx()


def create_line(net, net_pp, plotting=True, lines=None):
    '''
    The function creates an electrical aquivalent element of the chosen pandapower-lines in the
    sincal electrical Database Object (net). If plotting=True a graphical element will also be created.

    :param net: Sincal electrical Database Object
    :type net: object
    :param net_pp: The pandapower network that shall be converted
    :type net_pp: pandapowerNet
    :param plotting: Flag to graphically display Elements
    :type plotting: boolean
    :param lines: DataFrame with lines to convert, if None all lines will be converted.
    :type lines: DataFrame, None
    :return: None
    :rtype: None
    '''
    if lines is None:
        lines = net_pp.line
    lines['Element_ID'] = None
    bus_ocr = dict()
    for b in net_pp.bus.index:
        to_bus = net_pp.line.loc[net_pp.line.from_bus == b, 'to_bus']
        from_bus = net_pp.line.loc[net_pp.line.to_bus == b, 'from_bus']
        busses = np.concatenate([from_bus, to_bus])
        uni_bus, counts = np.unique(busses, return_counts=True)
        if not len(uni_bus):
            continue
        uni_bus = pd.Series(np.zeros(len(uni_bus)), index=uni_bus.tolist())
        bus_ocr[b] = dict(uni_bus)
    for idx, line in lines.iterrows():
        ln = net.CreateElement('Line', line['Sinc_Name'], net_pp.bus.loc[line.from_bus, 'Sinc_Name'],
                               net_pp.bus.loc[line.to_bus, 'Sinc_Name'])
        e_id = ln.GetValue("Element_ID")
        lines.Element_ID.at[idx] = e_id

        ln.SetValue('l', line.length_km)
        ln.SetValue('r', line.r_ohm_per_km)
        ln.SetValue('x', line.x_ohm_per_km)
        ln.SetValue('c', line.c_nf_per_km)
        ln.SetValue('Ith', line.max_i_ka)
        ln.SetValue('Flag_lineTyp', line.type)
        ln.SetValue('Flag_ParSys', line.parallel)
        ln.SetValue('Flag_fr', line.df)
        if not line.std_type is None:
            ln.SetValue('Description', 'std_type: ' + line.std_type)
        if plotting:
            tile = net.GetCommonObject("GraphicAreaTile", 1)
            factor = tile.GetValue('ScalePaper')
            bus_geo_from = net_pp.bus_geodata.loc[line.from_bus]
            bus_geo_to = net_pp.bus_geodata.loc[line.to_bus]
            ln.CreateGraphic()
            t_id = ln.GetValue('Terminal1.Terminal_ID')
            gterminal = net.GetCommonObject("GraphicTerminal", t_id)
            gterminal.SetValue('SwtNodePos', 15)
            gterminal.SetValue('SwtFactor', 20)
            fb = line.from_bus
            nlf = net_pp.line.loc[(net_pp.line.from_bus == fb) | (net_pp.line.to_bus == fb)].index
            if len(nlf) > 1:
                n_id = net_pp.bus.loc[fb, 'Node_ID']
                gn = net.GetCommonObject("GraphicNode", n_id)
                nxs = gn.GetValue('NodeStartX')
                nxe = gn.GetValue('NodeEndX')
                nx = (nxe + nxs) / 2
                tb = line.to_bus
                if bus_ocr[fb][tb] % 2:
                    gterminal.SetValue('PosX', nx - round(bus_ocr[fb][tb] / 2 + 0.1) * 0.02 * factor + 0.01 * factor)
                else:
                    gterminal.SetValue('PosX', nx + bus_ocr[fb][tb] / 2 * 0.02 * factor + 0.01 * factor)
                bus_ocr[fb][tb] += 1
            gterminal.Update()
            gt_id = gterminal.GetValue('GraphicText_ID')
            gtext = net.GetCommonObject("GraphicText", gt_id)
            x = 0.1 * (bus_geo_to.x - bus_geo_from.x)
            y = 0.1 * (bus_geo_to.y - bus_geo_from.y)
            gtext.SetValue("Pos1", x)
            gtext.SetValue("Pos2", y)
            gtext.SetValue("Visible", 0)
            gtext.Update()

            t_id = ln.GetValue('Terminal2.Terminal_ID')
            gterminal = net.GetCommonObject("GraphicTerminal", t_id)
            gterminal.SetValue('SwtNodePos', 15)
            gterminal.SetValue('SwtFactor', 20)
            gt_id = gterminal.GetValue('GraphicText_ID')
            gtext = net.GetCommonObject("GraphicText", gt_id)
            x = 0.1 * (bus_geo_from.x - bus_geo_to.x)
            y = 0.1 * (bus_geo_from.y - bus_geo_to.y)
            gtext.SetValue("Pos1", x)
            gtext.SetValue("Pos2", y)
            gtext.SetValue("Visible", 0)
            gtext.Update()
            gelement = net.GetCommonObject("GraphicElement", e_id)
            gt_id = gelement.GetValue('GraphicText_ID1')
            gtexte = net.GetCommonObject("GraphicText", gt_id)
            pos = gtexte.GetValue("Pos2")
            gtexte.SetValue("Pos2", pos + 0.015 * factor / 4)
            gtexte.Update()
            tb = line.to_bus
            nlt = net_pp.line.loc[(net_pp.line.from_bus == tb) | (net_pp.line.to_bus == tb)].index
            if len(nlt) > 1:
                n_id = net_pp.bus.loc[tb, 'Node_ID']
                gn = net.GetCommonObject("GraphicNode", n_id)
                nxs = gn.GetValue('NodeStartX')
                nxe = gn.GetValue('NodeEndX')
                nx = (nxe + nxs) / 2
                fb = line.from_bus
                if bus_ocr[tb][fb] % 2:
                    gterminal.SetValue('PosX', nx - (round(bus_ocr[tb][fb] / 2 + 0.1)) * 0.02 * factor + 0.01 * factor)
                else:
                    gterminal.SetValue('PosX', nx + bus_ocr[tb][fb] / 2 * 0.02 * factor + 0.01 * factor)
                bus_ocr[tb][fb] += 1
            gterminal.Update()
        ln.Update()
    for b in bus_ocr.keys():
        n_id = net_pp.bus.loc[b, 'Node_ID']
        gn = net.GetCommonObject("GraphicNode", n_id)
        coords_x = gn.GetValue('NodeStartX')
        coorde_x = gn.GetValue('NodeEndX')
        coord_x = (coorde_x + coords_x) / 2
        maximum = max(bus_ocr[b].values())
        comp = (round(maximum / 2 + 0.1) - 1) * 0.02 * factor + 0.01 * factor
        if (maximum == 1) and (coord_x + comp > coorde_x):
            continue
        elif coord_x + comp > coorde_x:
            gn.SetValue('NodeStartX', coord_x - comp)
            gn.SetValue('NodeEndX', coord_x + comp)
        gn.SetValue("SymType", 3)
        gn.Update()
    if len(net_pp.line) > 500:
        net.Close()
        net.OpenEx()


def create_switch(net, net_pp, plotting=True, switches=None):
    '''
    The function creates an electrical aquivalent element of the chosen pandapower-switches in the
    sincal electrical Database Object (net). If plotting=True a graphical element will also be created.

    :param net: Sincal electrical Database Object
    :type net: object
    :param net_pp: The pandapower network that shall be converted
    :type net_pp: pandapowerNet
    :param plotting: Flag to graphically display Elements
    :type plotting: boolean
    :param switches: DataFrame with switches to convert, if None all switches will be converted.
    :type switches: DataFrame, None
    :return: None
    :rtype: None
    '''
    if switches is None:
        switches = net_pp.switch
    switches['Element_ID'] = None
    for idx, switch in switches.iterrows():

        # Model Bus-Bus Switches as Lines with Type "Connector" (3)
        if switch.et == 'b':
            cnnctr = net.CreateElement('Line', 'switch_' + str(idx), net_pp.bus.loc[switch.bus, 'Sinc_Name'],
                                       net_pp.bus.loc[switch.element, 'Sinc_Name'])
            # Saving Element_ID
            e_id = cnnctr.GetValue("Element_ID")
            switches.Element_ID.at[idx] = e_id

            # Parametrization
            cnnctr.SetValue('l', 0)
            cnnctr.SetValue('Flag_LineTyp', 3)  # Setting Type as Connector

            # Create Breaker
            brk_b = net.CreateObject('Breaker')
            brk_b.SetValue("Flag_State", int(switch.closed))
            brk_b.SetValue("Name", "switch_" + str(idx))

            b_tid = cnnctr.GetValue("Terminal1.Terminal_ID")
            brk_b.SetValue("Terminal_ID", b_tid)

            if plotting:
                tile = net.GetCommonObject("GraphicAreaTile", 1)
                factor = tile.GetValue('ScalePaper')
                cnnctr.CreateGraphic()
                t_id = cnnctr.GetValue('Terminal1.Terminal_ID')
                gterminal = net.GetCommonObject("GraphicTerminal", t_id)
                gt_id = gterminal.GetValue('GraphicText_ID')
                gtext = net.GetCommonObject("GraphicText", gt_id)
                gtext.SetValue("Visible", 0)
                gtext.Update()
                gtext = net.GetCommonObject("GraphicText", gt_id - 1)
                gtext.SetValue("Visible", 0)
                gtext.Update()
                t_id = cnnctr.GetValue('Terminal2.Terminal_ID')
                gterminal = net.GetCommonObject("GraphicTerminal", t_id)
                gt_id = gterminal.GetValue('GraphicText_ID')
                gtext = net.GetCommonObject("GraphicText", gt_id)
                gtext.SetValue("Visible", 0)
                gtext.Update()
                gelement = net.GetCommonObject("GraphicElement", e_id)
                gt_id = gelement.GetValue('GraphicText_ID1')
                gtexte = net.GetCommonObject("GraphicText", gt_id)
                pos = gtexte.GetValue("Pos2")
                gtexte.SetValue("Pos2", pos + 0.015 * factor / 4)
                gtexte.Update()

                brk_b.CreateGraphic()
                brk_b.SetValue("GraphicAddTerminal.SymType", 3)  # 3: Circle Symbol
                b_tid = brk_b.GetValue("Breaker_ID")
                gaddterm = net.GetCommonObject('GraphicAddTerminal', b_tid)
                gaddterm.SetValue('SymNodePos', 50)
                gt_id = gaddterm.GetValue('GraphicText_ID')
                gtext = net.GetCommonObject("GraphicText", gt_id)
                gtext.SetValue("Visible", 0)
                pos = gtext.GetValue("Pos2")
                gtext.SetValue("Pos2", pos + 0.015 * factor / 4)
                gtext.Update()
                gaddterm.Update()
            cnnctr.Update()
            brk_b.Update()

        # Model Bus-Line Switches as Breaker
        elif switch.et == 'l':
            brk_l = net.CreateObject('Breaker')

            # Saving Element_ID
            e_id = brk_l.GetValue("Element_ID")
            switches.Element_ID.at[idx] = e_id

            # Parametrization
            brk_l.SetValue("Flag_State", int(switch.closed))  # 0: Open, 1: Closed
            brk_l.SetValue("Name", "switch_" + str(idx))

            # Chose correct Terminal (from_bus = Terminal1 | to_bus = Terminal2)
            line_idx = switch.element
            bus_idx = switch.bus

            l_eid = net_pp.line.Element_ID.at[line_idx]
            ln = net.GetCommonObject("Line", l_eid)  # Get Element ID

            l_tid1 = ln.GetValue("Terminal1.Terminal_ID")
            l_tid2 = ln.GetValue("Terminal2.Terminal_ID")
            if bus_idx == net_pp.line.from_bus.at[line_idx]:
                brk_l.SetValue("Terminal_ID", l_tid1)
            else:
                brk_l.SetValue("Terminal_ID", l_tid2)
            if plotting:
                tile = net.GetCommonObject("GraphicAreaTile", 1)
                factor = tile.GetValue('ScalePaper')
                brk_l.CreateGraphic()
                brk_l.SetValue("GraphicAddTerminal.SymType", 3)  # 3: Circle Symbol
                b_tid = brk_l.GetValue("Breaker_ID")
                gaddterm = net.GetCommonObject('GraphicAddTerminal', b_tid)
                gaddterm.SetValue('SymNodePos', 50)
                gt_id = gaddterm.GetValue('GraphicText_ID')
                gtext = net.GetCommonObject("GraphicText", gt_id)
                gtext.SetValue("Visible", 0)
                pos = gtext.GetValue("Pos2")
                gtext.SetValue("Pos2", pos + 0.015 * factor / 4)
                gtext.Update()
                gaddterm.Update()
            brk_l.Update()

        # Model Bus-Trafo Switches as Breaker
        elif switch.et == 't':
            brk_t = net.CreateObject('Breaker')

            # Saving Element_ID
            e_id = brk_t.GetValue("Element_ID")
            switches.Element_ID.at[idx] = e_id

            # Parametrization
            brk_t.SetValue("Flag_State", int(switch.closed))  # 0: Open, 1: Closed
            brk_t.SetValue("Name", "switch_" + str(idx))

            # Chose correct Terminal (hv_bus = Terminal1 | lv_bus = Terminal2)
            trafo_idx = switch.element
            bus_idx = switch.bus

            t_eid = net_pp.trafo.Element_ID.at[trafo_idx]
            tr = net.GetCommonObject("TwoWindingTransformer", t_eid)  # Get Element ID

            t_tid1 = tr.GetValue("Terminal1.Terminal_ID")
            t_tid2 = tr.GetValue("Terminal2.Terminal_ID")
            if bus_idx == net_pp.trafo.hv_bus.at[trafo_idx]:
                brk_t.SetValue("Terminal_ID", t_tid1)
            else:
                brk_t.SetValue("Terminal_ID", t_tid2)
            if plotting:
                tile = net.GetCommonObject("GraphicAreaTile", 1)
                factor = tile.GetValue('ScalePaper')
                brk_t.CreateGraphic()
                brk_t.SetValue("GraphicAddTerminal.SymType", 3)  # 3: Circle Symbol
                b_tid = brk_t.GetValue("Breaker_ID")
                gaddterm = net.GetCommonObject("GraphicAddTerminal", b_tid)
                gaddterm.SetValue('SymNodePos', 50)
                gt_id = gaddterm.GetValue('GraphicText_ID')
                gtext = net.GetCommonObject("GraphicText", gt_id)
                gtext.SetValue("Visible", 0)
                pos = gtext.GetValue("Pos2")
                gtext.SetValue("Pos2", pos + 0.015 * factor / 4)
                gtext.Update()
                gaddterm.Update()
            brk_t.Update()
    if len(net_pp.switch) > 500:
        net.Close()
        net.OpenEx()


def create_gen(net, net_pp, elements, plotting=True, gens=None):
    '''
    The function creates an electrical aquivalent element of the chosen pandapower-gens in the
    sincal electrical Database Object (net). If plotting=True a graphical element will also be created.

    :param net: Sincal electrical Database Object
    :type net: object
    :param net_pp: The pandapower network that shall be converted
    :type net_pp: pandapowerNet
    :param elements: Number of connected elements to all busses
    :type elements: Tuple
    :param plotting: Flag to graphically display Elements
    :type plotting: boolean
    :param gens: DataFrame with gens to convert, if None all gens will be converted.
    :type gens: DataFrame, None
    :return: None
    :rtype: None
    '''
    if gens is None:
        gens = net_pp.gen
    gens['Element_ID'] = None
    for idx, gen in gens.iterrows():
        g = net.CreateElement('SynchronousMachine', gen['Sinc_Name'], net_pp.bus.loc[gen.bus, 'Sinc_Name'])
        e_id = g.GetValue("Element_ID")
        gens.Element_ID.at[idx] = e_id

        g.SetValue('Sn', gen.sn_mva)
        g.SetValue('Flag_Lf', 11)
        g.SetValue('P', gen.p_mw)
        g.SetValue('u', gen.vm_pu * 100)
        if plotting:
            tile = net.GetCommonObject("GraphicAreaTile", 1)
            factor = tile.GetValue('ScalePaper')
            geo = net_pp.bus_geodata.loc[gen.bus, :]
            x = 0.015 * np.sin(np.deg2rad(elements[0][gen.bus] * elements[1][gen.bus] + 90))
            y = np.sqrt(0.015 ** 2 - x ** 2)
            if (elements[0][gen.bus] * elements[1][gen.bus] > 180) and \
                    (elements[0][gen.bus] * elements[1][gen.bus] < 360):
                y = - y
            g.CreateGraphic(geo.x + x * factor, geo.y + y * factor)
            t_id = g.GetValue('Terminal1.Terminal_ID')
            gterminal = net.GetCommonObject("GraphicTerminal", t_id)
            gterminal.SetValue('SwtNodePos', 15)
            gterminal.SetValue('SwtFactor', 20)
            gterminal.Update()
            gt_id = gterminal.GetValue('GraphicText_ID')
            gtext = net.GetCommonObject("GraphicText", gt_id)
            gtext.SetValue("Pos1", x * factor / 2)
            gtext.SetValue("Pos2", y * factor / 2)
            gtext.Update()
            elements[1][gen.bus] += 1
            gelement = net.GetCommonObject("GraphicElement", e_id)
            gt_id = gelement.GetValue('GraphicText_ID1')
            gtexte = net.GetCommonObject("GraphicText", gt_id)
            pos = gtexte.GetValue("Pos2")
            gtexte.SetValue("Pos2", pos + 0.015 * factor / 4)
            gtexte.Update()
        g.Update()
    if len(net_pp.gen) > 500:
        net.Close()
        net.OpenEx()


def create_storage(net, net_pp, elements, plotting=True, storages=None):
    '''
    The function creates an electrical aquivalent element of the chosen pandapower-storages in the
    sincal electrical Database Object (net). If plotting=True a graphical element will also be created.

    :param net: Sincal electrical Database Object
    :type net: object
    :param net_pp: The pandapower network that shall be converted
    :type net_pp: pandapowerNet
    :param elements: Number of connected elements to all busses
    :type elements: Tuple
    :param plotting: Flag to graphically display Elements
    :type plotting: boolean
    :param storages: DataFrame with storages to convert, if None all storages will be converted.
    :type storages: DataFrame, None
    :return: None
    :rtype: None
    '''
    if storages is None:
        storages = net_pp.storage
    storage_loads = storages.loc[storages.p_mw >= 0]
    storage_sgens = copy.deepcopy(storages.loc[storages.p_mw < 0])
    storage_sgens.p_mw.values[:] = np.abs(storage_sgens.p_mw.values)
    create_load(net, net_pp, elements, plotting, storage_loads)
    create_sgen(net, net_pp, elements, plotting, storage_sgens)
    if len(net_pp.storage) > 500:
        net.Close()
        net.OpenEx()


def create_dcline(net, net_pp, elements, plotting=True, dcline=None):
    '''
    The function creates an electrical aquivalent element of the chosen pandapower-dclines in the
    sincal electrical Database Object (net). If plotting=True a graphical element will also be created.

    :param net: Sincal electrical Database Object
    :type net: object
    :param net_pp: The pandapower network that shall be converted
    :type net_pp: pandapowerNet
    :param elements: Number of connected elements to all busses
    :type elements: Tuple
    :param plotting: Flag to graphically display Elements
    :type plotting: boolean
    :param dcline: DataFrame with dcline to convert, if None all dcline will be converted.
    :type dcline: DataFrame, None
    :return: None
    :rtype: None
    '''
    if dcline is None:
        dcline = net_pp.dcline
    dcline_from = copy.deepcopy(dcline)
    res_dcline_from = net_pp.res_dcline.loc[dcline_from.index, :]
    dcline_from['vm_pu'] = dcline_from['vm_from_pu']
    dcline_from['bus'] = dcline_from['from_bus']
    dcline_from['sn_mva'] = np.sqrt(dcline_from['p_mw'] ** 2 + res_dcline_from['q_from_mvar'] ** 2)
    dcline_from['Sinc_Name'] = 'from_' + dcline_from['Sinc_Name']

    create_sgen(net, net_pp, elements, plotting, dcline_from, True)

    dcline_to = copy.deepcopy(dcline)
    res_dcline_to = net_pp.res_dcline.loc[dcline_to.index, :]
    dcline_to['vm_pu'] = dcline_to['vm_to_pu']
    dcline_to['bus'] = dcline_to['to_bus']
    dcline_to['sn_mva'] = np.sqrt(dcline_to['p_mw'] ** 2 + res_dcline_to['q_to_mvar'] ** 2)
    dcline_to['Sinc_Name'] = 'to_' + dcline_to['Sinc_Name']

    create_sgen(net, net_pp, elements, plotting, dcline_to, True)
    if len(net_pp.dcline) > 500:
        net.Close()
        net.OpenEx()
