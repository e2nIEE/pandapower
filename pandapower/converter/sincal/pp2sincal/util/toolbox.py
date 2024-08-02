import copy
from datetime import date

import geopandas as gpd
import numpy as np
import pandapower as pp
import pandas as pd
from networkx import shortest_path
from pandapower import plotting
from pandapower.topology import create_nxgraph
from shapely import geometry

def _unique_naming(net_pp):
    '''
    Passes a unique name to each pandapower element. Name contains element name and index, if the name in the name \
    column is not unique.

    :param net_pp: The pandapower network that shall be converted
    :type net_pp: pandapowerNet
    :return: None
    :rtype: None
    '''
    for element in ['bus', 'line', 'dcline', 'load', 'sgen', 'asymmetric_sgen',
                    'gen', 'ext_grid', 'trafo', 'storage', 'switch']:
        net_pp[element]["Sinc_Name"] = None

        names = net_pp[element].name
        if (len(set(names)) != len(names)) or any(names == None):
            net_pp[element].loc[:, 'Sinc_Name'] = element + net_pp[element].index.astype(str).values
        else:
            net_pp[element].loc[:, 'Sinc_Name'] = names


def _number_of_elements(net_pp):
    '''
    Calculates the number of elements.

    :param net_pp: Pandapower Network Model
    :type net_pp: pandapowerNet
    :return: number of elements
    :rtype: Tuple
    '''
    noe = dict.fromkeys(net_pp['bus'].index, 0)
    cnoe = copy.deepcopy(noe)
    for idx in net_pp['bus'].index:
        for ele in ['load', 'sgen', 'ext_grid', 'gen', 'storage', 'dcline', 'asymmetric_sgen']:
            element = pp.get_connected_elements(net_pp, ele, idx)
            noe[idx] += len(element)
        noe[idx] = 360 / noe[idx] if noe[idx] != 0 else 0
    return noe, cnoe


def _scaling(net, net_pp):
    '''
    Adapts the area tile in Sincal referring to the pandapower geo-coordinates.

    :param net: Sincal electrical Database Object
    :type net: object
    :param net_pp: The pandapower network that shall be converted
    :type net_pp: pandapowerNet
    :return: None
    :rtype: None
    '''
    tile = net.GetCommonObject("GraphicAreaTile", 1)
    tile.SetValue('Flag', 1)
    fac = 100
    tile.SetValue('ScalePaper', fac)
    tile.SetValue('Scale2', 3)
    if not len(net_pp.bus_geodata):
        plotting.create_generic_coordinates(net_pp)
        # Offset to better display topology in PSS Sincal
        x = net_pp.bus_geodata.x.values + 50
        y = net_pp.bus_geodata.y.values + 50
    else:
        x = net_pp.bus_geodata.x.values
        y = net_pp.bus_geodata.y.values
    coordinates = pd.Series(list(zip(x, y)), name='geometry')
    coordinates = coordinates.apply(geometry.Point)
    coordinates = gpd.GeoDataFrame(coordinates)
    coordinates.crs = {'init': 'epsg:4326'}
    coordinates = coordinates.to_crs(epsg=31467)
    x_orig = coordinates['geometry'].x.values
    y_orig = coordinates['geometry'].y.values
    x = x_orig - x_orig.min()
    y = y_orig - y_orig.min()

    net_pp.bus_geodata.x = (x + x.mean() / 5)
    net_pp.bus_geodata.y = (y + y.mean() / 5)

    if 'line_geodata' in net_pp:
        coordinates = net_pp.line_geodata.coords
        coordinates.name = 'geometry'
        coordinates = coordinates.apply(geometry.LineString)
        coordinates = gpd.GeoDataFrame(coordinates, crs='epsg:4326')
        coordinates = coordinates.to_crs(epsg=31467)
        coordinates = coordinates.geometry.apply(lambda e: list(e.coords))
        coordinates = coordinates.apply(lambda e: list(np.array(e) - [x_orig.min() - x.mean() / 5,
                                                                      y_orig.min() - y.mean() / 5]))
        net_pp.line_geodata.coords.values[:] = coordinates.values
    width = x.max() + 2 * x.mean() / 5
    height = y.max() + 2 * y.mean() / 5
    tile.SetValue('AreaWidth', width)
    tile.SetValue('AreaHeight', height)
    tile.Update()


def _adapt_geo_coordinates(net, net_pp):
    '''
    Adjusts bus geodata for better graphical display of transformers.

    :param net: Sincal electrical Database Object
    :type net: object
    :param net_pp: Pandapower Network Model
    :type net_pp: pandapowerNet
    :return: None
    :rtype: None
    '''
    tile = net.GetCommonObject("GraphicAreaTile", 1)
    factor = tile.GetValue('ScalePaper')
    busses_geo = pd.Series(list(zip(net_pp.bus_geodata['x'].values, net_pp.bus_geodata['y'].values)),
                           index=net_pp.bus.index)
    line_busses_geo_df = pd.DataFrame(np.array([
        net_pp.bus_geodata.loc[net_pp.line.from_bus.values]['y'].values,
        net_pp.bus_geodata.loc[net_pp.line.to_bus.values]['y'].values]).T,
                                      columns=['y_from', 'y_to'],
                                      index=list(net_pp.line.index))
    uni, c = np.unique(net_pp.bus_geodata[['x', 'y']].values, axis=0, return_counts=True)
    mg = create_nxgraph(net_pp)
    if len(net_pp.switch):
        multi_wo_sw = 1
    else:
        multi_wo_sw = 0
    for entry, count in zip(uni, c):
        if count == 1:
            continue
        busses = busses_geo[(net_pp.bus_geodata['x'].values == entry[0]) &
                            (net_pp.bus_geodata['y'].values == entry[1])]
        bus_hv, idx_hv = np.where(net_pp.trafo.hv_bus.values == np.array(busses.index)[:, np.newaxis])
        bus_lv, idx_lv = np.where(net_pp.trafo.lv_bus.values == np.array(busses.index)[:, np.newaxis])
        net_pp.bus_geodata.loc[busses.index[bus_hv], 'y'] += 0.055 * factor
        net_pp.bus_geodata.loc[busses.index[bus_hv], 'x'] += 0.02 * factor * np.arange(1, len(bus_hv) + 1) * multi_wo_sw
        lv = np.where(net_pp.trafo.hv_bus.values == np.array(busses.index)[bus_hv][:, np.newaxis])
        net_pp.bus_geodata.loc[net_pp.trafo.lv_bus.iloc[lv[1]], 'y'] += 0.005 * factor * multi_wo_sw
        uni_lv, idx_lv = np.unique(lv[1], return_index=True)
        net_pp.bus_geodata.loc[net_pp.trafo.lv_bus.iloc[uni_lv[idx_lv]], 'x'] += 0.02 * factor * np.arange(1, len(
            uni_lv) + 1) * multi_wo_sw
        bus_trafo = np.concatenate([bus_hv, bus_lv])

        bus_from, idx_from = np.where(net_pp.line.from_bus.values == np.array(busses.index)[:, np.newaxis])
        bus_to, idx_to = np.where(net_pp.line.to_bus.values == np.array(busses.index)[:, np.newaxis])
        y_from_line = line_busses_geo_df.loc[idx_to, 'y_from']
        y_to_line = line_busses_geo_df.loc[idx_from, 'y_to']
        bus_line = np.concatenate([bus_from, bus_to])
        y_line = np.concatenate([y_to_line, y_from_line])
        if not len(y_line):
            continue
        rest = set(busses.index) - set(busses.index[bus_line]) - set(busses.index[bus_trafo])

        multi_res_up = 0
        multi_res_down = 0

        down = y_line <= np.mean(y_line)
        up = y_line > np.mean(y_line)

        if len(bus_trafo):
            #down = np.ones(len(y_line), dtype=bool)
            #up = np.zeros(len(y_line), dtype=bool)
            for b in rest:
                res = shortest_path(mg, b, busses.index[bus_lv[0]])
                res = np.where(res == np.array(busses.index)[bus_hv][:, np.newaxis])[0]
                if len(res):
                    net_pp.bus_geodata.loc[b, 'y'] += 0.06 * factor
                    net_pp.bus_geodata.loc[b, 'x'] += multi_res_up * 0.02 * factor + 0.01 * factor
                    multi_res_up += 1
                else:
                    net_pp.bus_geodata.loc[b, 'x'] += multi_res_down * 0.02 * factor + 0.01 * factor
                    multi_res_down += 1
            add_up = 0.06 * factor
        else:
            #down = y_line <= np.mean(y_line)
            #up = y_line > np.mean(y_line)
            for b in rest:
                res = shortest_path(mg, b, busses.index[bus_line[np.argmin(y_line)]])
                res = np.where(list(rest) == np.array(res)[1:, np.newaxis])[0]
                if len(res):
                    net_pp.bus_geodata.loc[b, 'y'] += 0.005 * factor
                    net_pp.bus_geodata.loc[b, 'x'] -= multi_res_up * 0.02 * factor + 0.01 * factor
                    multi_res_up += 1
                else:
                    net_pp.bus_geodata.loc[b, 'x'] -= multi_res_down * 0.02 * factor + 0.01 * factor
                    multi_res_down += 1
            add_up = 0.005 * factor
        if len(net_pp.switch) or not len(bus_trafo):
            u_down = np.unique(y_line[down])
            u_up = np.unique(y_line[up])
            net_pp.bus_geodata.loc[busses.index[bus_line[down]], 'y'] -= 0.005 * factor
            net_pp.bus_geodata.loc[busses.index[bus_line[up]], 'y'] += 0.005 * factor + add_up
            for uup in u_up:
                data_up = net_pp.bus_geodata.loc[busses.index[bus_line[y_line == uup]], 'x']
                multi = np.arange(0, len(data_up))
                data_up.loc[:] -= multi * 0.02 * factor
                net_pp.bus_geodata.loc[busses.index[bus_line[y_line == uup]], 'x'] = data_up
            for udown in u_down:
                data_down = net_pp.bus_geodata.loc[busses.index[bus_line[y_line == udown]], 'x']
                multi = np.arange(0, len(data_down))
                data_down.loc[:] -= multi * 0.02 * factor
                net_pp.bus_geodata.loc[busses.index[bus_line[y_line == udown]], 'x'] = data_down


def _initialize_voltage_level(net, net_pp):
    '''
    Initializes the voltage level in Sincal.

    :param net: Sincal electrical Database Object
    :type net: object
    :param net_pp: The pandapower network that shall be converted
    :type net_pp: pandapowerNet
    :return: Sincal Voltage Levels
    :rtype: object
    '''
    vn_kvs = set(net_pp.bus.vn_kv)
    vls = dict()
    for i, vn_kv in enumerate(vn_kvs):
        vl = net.CreateObject('VoltageLevel')
        vl.SetValue('Name', 'Level_' + str(vn_kv))
        vl.SetValue('Un', vn_kv)
        vl.SetValue('Uop', vn_kv)
        vl.Update()
        vls[vn_kv] = vl.GetValue('VoltLevel_ID')
    return vls


def _set_calc_params(net):
    '''
    Sets the correct voltage limits.

    :param net: Sincal electrical Database Object
    :type net: object
    :return: None
    :rtype: None
    '''
    cp = net.getCommonObject('CalcParameter', 1)
    cp.SetValue('uul', 1.25 * 100)
    cp.SetValue('ull', 0.75 * 100)
    cp.SetValue('Flag_LFmet', 2)
    cp.SetValue('T_SV', 0.1)
    cp.SetValue('PNB', 0.0001)
    cp.SetValue('VDN', 0.0001)
    today = date.today().strftime('%d.%m.%Y') + ' 00:00'
    cp.SetValue('LC_StartDate', today)
    cp.SetValue('Flag_XLV', 0)
    cp.SetValue('Flag_Det_ST', 0)
    cp.SetValue('Flag_PhiUlf', 0)
    cp.SetValue('Flag_Unit', 0)
    cp.Update()


def _get_vector_group(vector_group, shift_degree):
    if shift_degree % 30 != 0:
        raise (UserWarning('trafo shift degrees different from multitudes of 30 are not supported yet. Yours is %s'
                           % shift_degree))
    ttyp = vector_group + str(int(shift_degree / 30))
    ttyp = ttyp.lower()
    if ttyp == 'yy0':
        return 6
    elif ttyp == 'ynd5':
        return 26
    elif ttyp == 'yzn5':
        return 30
    elif ttyp == 'dyn5':
        return 24
    else:
        raise (UserWarning('trafo vector group %s not implemented yet' % ttyp))
