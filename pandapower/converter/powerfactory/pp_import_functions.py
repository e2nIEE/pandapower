import bisect
import math
import numbers
import re
from itertools import combinations

import numpy as np
import pandapower as pp
from pandapower.auxiliary import ADict
from pandas import DataFrame, Series

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


# make wrapper for GetAttribute
def ga(element, attr):
    return element.GetAttribute(attr)


# import network to pandapower:
def from_pf(dict_net, pv_as_slack=True, pf_variable_p_loads='plini', pf_variable_p_gen='pgini',
            flag_graphics='GPS', tap_opt="nntap", export_controller=True, handle_us="Deactivate",
            max_iter=None, is_unbalanced=False):
    logger.debug("__name__: %s" % __name__)
    logger.debug('started from_pf')
    logger.info(logger.__dict__)

    flag_graphics = flag_graphics if flag_graphics in ['GPS', 'no geodata'] else 'graphic objects'

    logger.debug('collecting grid')
    grid_name = dict_net['ElmNet'].loc_name
    base_sn_mva = dict_net['global_parameters']['base_sn_mva']
    net = pp.create_empty_network(grid_name, sn_mva=base_sn_mva)
    pp.results.reset_results(net, mode="pf_3ph")
    if max_iter is not None:
        pp.set_user_pf_options(net, max_iteration=max_iter)
    logger.info('creating grid %s' % grid_name)
    if 'res_switch' not in net.keys():
        net['res_switch'] = DataFrame(columns=['pf_closed', 'pf_in_service'], dtype='bool')

    logger.debug('creating buses')
    # create buses:
    global bus_dict
    bus_dict = {}
    global grf_map
    grf_map = dict_net.get('graphics', {})
    logger.debug('the graphic mapping is: %s' % grf_map)

    # ist leider notwendig
    n = 0
    for n, bus in enumerate(dict_net['ElmTerm'], 1):
        create_bus(net=net, item=bus, flag_graphics=flag_graphics, is_unbalanced=is_unbalanced)
    if n > 0: logger.info('imported %d buses' % n)

    logger.debug('creating external grids')
    # create external networks:
    n = 0
    for n, ext_net in enumerate(dict_net['ElmXnet'], 1):
        create_ext_net(net=net, item=ext_net, pv_as_slack=pv_as_slack, is_unbalanced=is_unbalanced)
    if n > 0: logger.info('imported %d external grids' % n)

    logger.debug('creating loads')
    # create loads:
    n = 0
    for n, load in enumerate(dict_net['ElmLod'], 1):
        try:
            create_load(net=net, item=load, pf_variable_p_loads=pf_variable_p_loads,
                        dict_net=dict_net, is_unbalanced=is_unbalanced)
        except RuntimeError as err:
            logger.debug('load failed at import and was not imported: %s' % err)
    if n > 0: logger.info('imported %d loads' % n)

    logger.debug('creating lv loads')
    # create loads:
    n = 0
    for n, load in enumerate(dict_net['ElmLodlv'], 1):
        try:
            create_load(net=net, item=load, pf_variable_p_loads=pf_variable_p_loads,
                        dict_net=dict_net, is_unbalanced=is_unbalanced)
        except RuntimeError as err:
            logger.warning('load failed at import and was not imported: %s' % err)
    if n > 0: logger.info('imported %d lv loads' % n),

    logger.debug('creating mv loads')
    # create loads:
    n = 0
    for n, load in enumerate(dict_net['ElmLodmv'], 1):
        try:
            create_load(net=net, item=load, pf_variable_p_loads=pf_variable_p_loads,
                        dict_net=dict_net, is_unbalanced=is_unbalanced)
        except RuntimeError as err:
            logger.error('load failed at import and was not imported: %s' % err)
    if n > 0: logger.info('imported %d mv loads' % n)

#    logger.debug('sum loads: %.3f' % sum(net.load.loc[net.load.in_service, 'p_mw']))

    logger.debug('creating static generators')
    # create static generators:
    n = 0
    for n, gen in enumerate(dict_net['ElmGenstat'], 1):
        try:
            create_sgen_genstat(net=net, item=gen, pv_as_slack=pv_as_slack,
                                pf_variable_p_gen=pf_variable_p_gen, dict_net=dict_net, is_unbalanced=is_unbalanced)
        except RuntimeError as err:
            logger.debug('sgen failed at import and was not imported: %s' % err)
    if n > 0: logger.info('imported %d static generators' % n)

    logger.debug('creating pv generators as static generators')
    # create pv generators:
    n = 0
    for n, pv in enumerate(dict_net['ElmPvsys'], 1):
        create_sgen_genstat(net=net, item=pv, pv_as_slack=pv_as_slack,
                            pf_variable_p_gen=pf_variable_p_gen, dict_net=dict_net, is_unbalanced=is_unbalanced)
    if n > 0: logger.info('imported %d pv generators' % n)

    logger.debug('creating asynchronous machines')
    # create asynchronous machines:
    n = 0
    for n, asm in enumerate(dict_net['ElmAsm'], n):
        create_sgen_asm(net=net, item=asm, pf_variable_p_gen=pf_variable_p_gen, dict_net=dict_net)
    if n > 0: logger.info('imported %d asynchronous machines' % n)

    logger.debug('creating synchronous machines')
    # create synchronous machines:
    n = 0
    for n, gen in enumerate(dict_net['ElmSym'], n):
        create_sgen_sym(net=net, item=gen, pv_as_slack=pv_as_slack,
                        pf_variable_p_gen=pf_variable_p_gen, dict_net=dict_net)
    if n > 0: logger.info('imported %d synchronous machines' % n)

    logger.debug('creating transformers')
    # create trafos:
    n = 0
    for n, trafo in enumerate(dict_net['ElmTr2'], 1):
        create_trafo(net=net, item=trafo, tap_opt=tap_opt, export_controller=export_controller,
                     is_unbalanced=is_unbalanced)
    if n > 0: logger.info('imported %d trafos' % n)

    logger.debug('creating 3W-transformers')
    # create 3w-trafos:
    n = 0
    for n, trafo in enumerate(dict_net['ElmTr3'], 1):
        create_trafo3w(net=net, item=trafo, tap_opt=tap_opt)
    if n > 0:
        logger.info('imported %d 3w-trafos' % n)
        pp.set_user_pf_options(net, trafo3w_losses='star')

    logger.debug('creating switches (couplings)')
    # create switches (ElmCoup):
    n = 0
    for n, coup in enumerate(dict_net['ElmCoup'], 1):
        create_coup(net=net, item=coup)
    if n > 0: logger.info('imported %d coups' % n)

    logger.debug('creating fuses (as couplings)')
    # create fuses (RelFuse):
    n = 0
    for n, fuse in enumerate(dict_net['RelFuse'], 1):
        create_coup(net=net, item=fuse, is_fuse=True)
    if n > 0: logger.info('imported %d fuses' % n)

    # create shunts (ElmShnt):
    n = 0
    for n, shunt in enumerate(dict_net['ElmShnt'], 1):
        create_shunt(net=net, item=shunt)
    if n > 0: logger.info('imported %d shunts' % n)

    # create zpu (ElmZpu):
    n = 0
    for n, zpu in enumerate(dict_net['ElmZpu'], 1):
        create_zpu(net=net, item=zpu)
    if n > 0: logger.info('imported %d impedances' % n)

    # create series impedance (ElmSind):
    n = 0
    for n, sind in enumerate(dict_net['ElmSind'], 1):
        create_sind(net=net, item=sind)
    if n > 0: logger.info('imported %d SIND' % n)

    # create vac (ElmVac):
    n = 0
    for n, vac in enumerate(dict_net['ElmVac'], 1):
        create_vac(net=net, item=vac)
    if n > 0: logger.info('imported %d VAC' % n)

    # logger.debug('creating switches')
    # # create switches (StaSwitch):
    # n = 0
    # for switch in dict_net['StaSwitch']:
    #     create_switch(net=net, item=switch)
    #     n += 1
    # logger.info('imported %d switches' % n)

    for idx, row in net.trafo.iterrows():
        propagate_bus_coords(net, row.lv_bus, row.hv_bus)

    for idx, row in net.switch[net.switch.et == 'b'].iterrows():
        propagate_bus_coords(net, row.bus, row.element)

    # we do lines last because of propagation of coordinates
    logger.debug('creating lines')
    # create lines:
    global line_dict
    line_dict = {}
    n = 0
    for n, line in enumerate(dict_net['ElmLne'], 0):
        create_line(net=net, item=line, flag_graphics=flag_graphics, corridor=n, is_unbalanced=is_unbalanced)
    logger.info('imported %d lines' % (len(net.line.line_idx.unique())) if len(net.line) else 0)
    net.line['section_idx'] = 0
    if dict_net['global_parameters']["iopt_tem"] == 1:
        pp.set_user_pf_options(net, consider_line_temperature=True)

    if len(dict_net['ElmLodlvp']) > 0:
        lvp_dict = get_lvp_for_lines(dict_net)
        logger.debug(lvp_dict)
        split_all_lines(net, lvp_dict)

    remove_folder_of_std_types(net)

    ### don't import the ElmLodlvp for now...
    # logger.debug('creating lv partial loads')
    # # create loads:
    # n = 0
    # for n, load in enumerate(dict_net['ElmLodlvp'], 1):
    #     create_load(net=net, item=load, pf_variable_p_loads=pf_variable_p_loads)
    # if n > 0: logger.info('imported %d lv partial loads' % n)

    # # here we import the partial LV loads that are part of lines because of line section
    # coordinates
    # logger.debug('creating lv partial loads')
    # # create loads:
    # n = 0
    # for n, load in enumerate(dict_net['ElmLodlvp'], 1):
    #     try:
    #         create_load(net=net, item=load, pf_variable_p_loads=pf_variable_p_loads)
    #     except NotImplementedError:
    #         logger.debug('load %s not imported because it is not contained in ElmLod' % load)
    # if n > 0: logger.info('imported %d lv partial loads' % n)

    # if len(dict_net['ElmLodlvp']) > 0:
    #     n = 0
    #     for line in dict_net['ElmLne']:
    #         partial_loads = line.GetContents('*.ElmLodlvp')
    #         partial_loads.sort(key=lambda x: x.lneposkm)
    #         for load in partial_loads:
    #             create_load(net=net, item=load, pf_variable_p_loads=pf_variable_p_loads)
    #             n += 1
    #     logger.info('imported %d lv partial loads' % n)

    if handle_us == "Deactivate":
        logger.info('deactivating unsupplied elements')
        pp.set_isolated_areas_out_of_service(net)
    elif handle_us == "Drop":
        logger.info('dropping inactive elements')
        pp.drop_inactive_elements(net)
    elif handle_us != "Nothing":
        raise ValueError("handle_us should be 'Deactivate', 'Drop' or 'Nothing', "
                         "received: %s" % handle_us)

    if is_unbalanced:
        pp.add_zero_impedance_parameters(net)

    logger.info('imported net')
    return net


def get_graphic_object(item):
    try:
        graphic_object = grf_map[item]
    except KeyError as err:
        logger.warning('graphic object missing for element %s: %s' % (item, err))
        return None
    else:
        return graphic_object


def add_additional_attributes(item, net, element, element_id, attr_list=None, attr_dict=None):
    """
    Adds additonal atributes from powerfactory such as sernum or for_name

    @param item: powerfactory item
    @param net: pp net
    @param element: pp element namme (str). e.g. bus, load, sgen
    @param element_id: element index in pp net
    @param attr_list: list of attribtues to add. e.g. ["sernum", "for_name"]
    @param attr_dict: names of an attribute in powerfactory and in pandapower
    @return:
    """
    if attr_dict is None:
        attr_dict = {k: k for k in attr_list}

    for attr in attr_dict.keys():
        if '.' in attr:
            # go in the object chain of a.b.c.d until finally get the chr_name
            obj = item
            for a in attr.split('.'):
                if hasattr(obj, 'HasAttribute') and obj.HasAttribute(a):
                    obj = ga(obj, a)
            if obj is not None and isinstance(obj, str):
                net[element].loc[element_id, attr_dict[attr]] = obj

        elif item.HasAttribute(attr):
            chr_name = ga(item, attr)
            if chr_name is not None:
                if isinstance(chr_name, (str, numbers.Number)):
                    net[element].loc[element_id, attr_dict[attr]] = chr_name
                elif isinstance(chr_name, list):
                    if len(chr_name) > 1:
                        raise NotImplementedError(f"attribute {attr} is a list with more than 1 items - not supported.")
                    elif len(chr_name) == 0:
                        continue
                    net[element].loc[element_id, attr_dict[attr]] = chr_name[0]


def create_bus(net, item, flag_graphics, is_unbalanced):
    # add geo data
    if flag_graphics == 'GPS':
        x = ga(item, 'e:GPSlon')
        y = ga(item, 'e:GPSlat')
    elif flag_graphics == 'graphic objects':
        graphic_object = get_graphic_object(item)
        if graphic_object:
            x = ga(graphic_object, 'rCenterX')
            y = ga(graphic_object, 'rCenterY')
            # add gr coord data
        else:
            x, y = 0, 0
    else:
        x, y = 0, 0

    # only values > 0+-1e-3 are entered into the bus_geodata
    if x > 1e-3 or y > 1e-3:
        geodata = (x, y)
    else:
        geodata = None

    usage = ["b", "m", "n"]
    params = {
        'name': item.loc_name,
        'vn_kv': item.uknom,
        'in_service': not bool(item.outserv),
        'type': usage[item.iUsage],
        'geodata': geodata
    }

    logger.debug('>> creating bus <%s>' % params['name'])
    try:
        params['zone'] = item.Grid.loc_name.split('.ElmNet')[0]
    except AttributeError:
        params['zone'] = item.cpGrid.loc_name.split('.ElmNet')[0]

    bid = pp.create_bus(net, **params)
    # add the bus to the bus dictionary
    bus_dict[item] = bid

    get_pf_bus_results(net, item, bid, is_unbalanced)

    substat_descr = ''
    if item.HasAttribute('cpSubstat'):
        substat = item.cpSubstat
        if substat is not None:
            logger.debug('adding substat %s to descr of bus %s (#%d)' %
                         (substat, params['name'], bid))
            substat_descr = substat.loc_name
        else:
            logger.debug("bus has no substat description")
    else:
        logger.debug('bus %s is not part of any substation' %
                     params['name'])

    if len(item.desc) > 0:
        descr = ' \n '.join(item.desc)
    elif item.fold_id:
        descr = item.fold_id.loc_name
    else:
        descr = ''

    logger.debug('adding descr <%s> to bus' % descr)

    net.bus.at[bid, "description"] = descr
    net.bus.at[bid, "substat"] = substat_descr
    net.bus.at[bid, "folder_id"] = item.fold_id.loc_name

    add_additional_attributes(item, net, "bus", bid, attr_dict={"for_name": "equipment", "cimRdfId": "origin_id"},
                              attr_list=["sernum", "chr_name", "cpSite.loc_name"])

    # add geo data
    if flag_graphics == 'GPS':
        x = ga(item, 'e:GPSlon')
        y = ga(item, 'e:GPSlat')
    elif flag_graphics == 'graphic objects':
        graphic_object = get_graphic_object(item)
        if graphic_object:
            x = ga(graphic_object, 'rCenterX')
            y = ga(graphic_object, 'rCenterY')
            # add gr coord data
        else:
            x, y = 0, 0
    else:
        x, y = 0, 0

    # only values > 0+-1e-3 are entered into the bus_geodata
    if x > 1e-3 or y > 1e-3:
        net.bus_geodata.loc[bid, 'x'] = x
        net.bus_geodata.loc[bid, 'y'] = y

def get_pf_bus_results(net, item, bid, is_unbalanced):
    bus_type = None
    result_variables = None
    if is_unbalanced:
        bus_type = "res_bus_3ph"
        result_variables = {
          "pf_vm_a_pu": "m:u:A",
          "pf_va_a_degree": "m:phiu:A",
          "pf_vm_b_pu": "m:u:B",
          "pf_va_b_degree": "m:phiu:B",
          "pf_vm_c_pu": "m:u:C",
          "pf_va_c_degree": "m:phiu:C",
        }
    else:
        bus_type = "res_bus"
        result_variables = {
               "pf_vm_pu": "m:u",
               "pf_va_degree": "m:phiu"
               }

    for res_var_pp, res_var_pf in result_variables.items():
        res = np.nan
        if item.HasResults(0):
            res = ga(item, res_var_pf)
        net[bus_type].at[bid, res_var_pp] = res


# # This one deletes all the results :(
# # Don't use it
# def find_bus_index_in_net(item, net=None):
#     foreign_key = int(ga(item, 'for_name'))
#     return foreign_key


# Is unfortunately not that safe :(
# Don't use it
# def find_bus_index_in_net(item, net):
#     usage = ["b", "m", "n"]
#     # to be sure that the bus is the correct one
#     name = ga(item, 'loc_name')
#     bus_type = usage[ga(item, 'iUsage')]
#     logger.debug('looking for bus <%s> in net' % name)
#
#     if item.HasAttribute('cpSubstat'):
#         substat = ga(item, 'cpSubstat')
#         if substat is not None:
#             descr = ga(substat, 'loc_name')
#             logger.debug('bus <%s> has substat, descr is <%s>' % (name, descr))
#         else:
#             # omg so ugly :(
#             descr = ga(item, 'desc')
#             descr = descr[0] if len(descr) > 0 else ""
#             logger.debug('substat is none, descr of bus <%s> is <%s>' % (name, descr))
#     else:
#         descr = ga(item, 'desc')
#         descr = descr[0] if len(descr) > 0 else ""
#         logger.debug('no attribute "substat", descr of bus <%s> is <%s>' % (name, descr))
#
#     try:
#         zone = ga(item, 'Grid')
#         zone_name = ga(zone, 'loc_name').split('.ElmNet')[0]
#         logger.debug('zone "Grid" found: <%s>' % zone_name)
#     except:
#         zone = ga(item, 'cpGrid')
#         zone_name = ga(zone, 'loc_name').split('.ElmNet')[0]
#         logger.debug('zone "cpGrid" found: <%s>' % zone_name)
#
#     temp_df_a = net.bus[net.bus.zone == zone_name]
#     temp_df_b = temp_df_a[temp_df_a.type == bus_type]
#     temp_df_c = temp_df_b[temp_df_a.description == descr]
#     bus_index = temp_df_c[temp_df_b.name == name].index.values[0]
#     logger.debug('bus index in net of bus <%s> is <%d>' % (name, bus_index))
#
#     return bus_index


def find_bus_index_in_net(pf_bus, net):
    # bid = bus_dict.get(pf_bus, -1)
    # i want key error
    bid = bus_dict[pf_bus]
    return bid


def get_connection_nodes(net, item, num_nodes):
    buses = []
    for i in range(num_nodes):
        try:
            pf_bus = item.GetNode(i)
        except Exception as err:
            logger.error('GetNode failed for %s' % item)
            logger.error(err)
            pf_bus = None
        if pf_bus is None:
            if num_nodes == 1:
                logger.error(f"{item} has no connection node")
                raise IndexError
            buses.append(None)
        else:
            logger.debug("got bus %s" % pf_bus.loc_name)
            pp_bus = find_bus_index_in_net(pf_bus, net)
            if num_nodes == 1:
                return pp_bus
            buses.append(pp_bus)

    if all([b is None for b in buses]):
        logger.error("Element %s is Disconnected: buses are %s" %
                     (item.loc_name, buses))
        raise IndexError
    elif None in buses:
        logger.debug('exising buses: %s' % buses)
        existing_bus = (set(buses) - {None}).pop()
        name = net.bus.at[existing_bus, "name"] + "_aux"
        new_buses = []
        # determine the voltage needed
        # check if trafo
        v = []
        pf_class = item.GetClassName()
        logger.warning("object %s of class %s is not properly connected - creating auxiliary buses."
                       " check if the auxiliary buses have been created with correct voltages" % (
                           item, pf_class))

        if pf_class == "ElmTr2":
            v.append(ga(item, 't:utrn_h'))
            v.append(ga(item, 't:utrn_l'))
        elif pf_class == "ElmTr3":
            v.append(ga(item, 't:utrn3_h'))
            v.append(ga(item, 't:utrn3_m'))
            v.append(ga(item, 't:utrn3_l'))
        else:
            v = [net.bus.vn_kv.at[existing_bus] for _ in buses]

        # the order of buses must be the same as the order of voltage values
        # imo this could be more robust, because we don't know in what order item.GetNode(i)
        # actually returns the values, we can only rely on PF that it always is hv, mv, lv etc.
        for b, vv in zip(buses, v):
            if b is None:
                aux_bus = pp.create_bus(net, vv, type="n", name=name)
                new_buses.append(aux_bus)
                logger.debug("Created new bus '%s' for disconected line " % name)
            else:
                new_buses.append(b)
        return tuple(new_buses)
    else:
        return tuple(buses)


def import_switch(item, idx_cubicle):
    logger.debug('importing switch for %s (%d)' % (item.loc_name, idx_cubicle))
    switch_types = {"cbk": "CB", "sdc": "LBS", "swt": "LS", "dct": "DS"}
    cub = item.GetCubicle(idx_cubicle)
    if cub is None:
        return None, None, None
    switches = cub.GetContents('*.StaSwitch')
    if len(switches) > 1:
        logger.error('more then 1 switch found for %s: %s' % (item, switches))
    if len(switches) != 0:
        switch = switches[0]
        switch_in_service = not bool(switch.outserv) if switch.HasAttribute('outserv') else True
        switch_name = switch.cDisplayName
        if not switch.HasAttribute('isclosed'):
            logger.warning('switch %s does not have the attribute isclosed!!!' % switch)
        switch_is_closed = bool(switch.on_off) and bool(switch.isclosed) and switch_in_service
        switch_usage = switch_types.get(switch.aUsage, 'unknown')
        return switch_is_closed, switch_usage, switch_name
    else:
        return None, None, None


def create_connection_switches(net, item, number_switches, et, buses, elements):
    # False if open, True if closed, None if no switch
    logger.debug('creating connection switches')
    for i in range(number_switches):
        switch_is_closed, switch_usage, switch_name = import_switch(item, i)
        logger.debug('switch closed: %s, switch_usage: %s' % (switch_is_closed, switch_usage))
        if switch_is_closed is not None:
            cd = pp.create_switch(net, bus=buses[i], element=elements[i], et=et,
                                  closed=switch_is_closed, type=switch_usage, name=switch_name)
            net.res_switch.loc[cd, ['pf_closed', 'pf_in_service']] = switch_is_closed, True


def get_coords_from_buses(net, from_bus, to_bus, **kwargs):
    coords = []
    if from_bus in net.bus_geodata.index:
        x1, y1 = net.bus_geodata.loc[from_bus, ['x', 'y']]
        has_coords = True
    else:
        x1, y1 = np.nan, np.nan
        has_coords = False

    if to_bus in net.bus_geodata.index:
        x2, y2 = net.bus_geodata.loc[to_bus, ['x', 'y']]
        has_coords = True
    else:
        x2, y2 = np.nan, np.nan
        has_coords = False

    if has_coords:
        coords = [[x1, y1], [x2, y2]]
        logger.debug('got coords from buses: %s' % coords)
    else:
        logger.debug('no coords for line between buses %d and %d' % (from_bus, to_bus))
    return coords


def get_coords_from_item(item):
    # function reads geodata from item directly (for lines this is in item.GPScoords)
    coords = item.GPScoords
    try:
        # lat / lon must be switched in my example (karlsruhe). Check if this is always right
        c = tuple((x, y) for [y, x] in coords)
    except ValueError:
        try:
            c = tuple((x, y) for [y, x, z] in coords)
        except ValueError:
            c = []
    return c


def get_coords_from_grf_object(item):
    logger.debug('getting coords from gr_obj')
    # center_x = ga(grf_map[item], 'rCenterX')
    # center_y = ga(grf_map[item], 'rCenterY')
    # coords = [[center_x, center_y]]

    graphic_object = get_graphic_object(item)
    if graphic_object:
        coords = []
        cons = graphic_object.GetContents('*.IntGrfcon')
        cons.sort(key=lambda x: x.iDatConNr)
        for c in cons:
            con_nr = c.iDatConNr
            x_list = c.rX
            y_list = c.rY
            coords_list = list(list(t) for t in zip(x_list, y_list) if t != (-1, -1))
            if con_nr == 0:
                coords_list = coords_list[::-1]
            coords.extend(coords_list)
        if len(coords) == 0:
            coords = [[graphic_object.rCenterX, graphic_object.rCenterY]] * 2
        logger.debug('extracted line coords from graphic object: %s' % coords)
        # net.line_geodata.loc[lid, 'coords'] = coords
    else:
        coords = []

    return coords


def create_line(net, item, flag_graphics, corridor, is_unbalanced):
    params = {'parallel': item.nlnum, 'name': item.loc_name}
    logger.debug('>> creating line <%s>' % params['name'])
    logger.debug('line <%s> has <%d> parallel lines' % (params['name'], params['parallel']))

    logger.debug('asked for buses')
    # here: implement situation if line not connected

    try:
        params['bus1'], params['bus2'] = get_connection_nodes(net, item, 2)
    except IndexError:
        logger.debug("Cannot add Line '%s': not connected" % params['name'])
        return
    except:
        logger.error("Error while exporting Line '%s'" % params['name'])
        return

    line_sections = item.GetContents('*.ElmLnesec')
    # geodata
    if flag_graphics == 'no geodata':
        coords = []
    elif flag_graphics == 'GPS':
        if len(item.GPScoords) > 0:
            coords = get_coords_from_item(item)
        else:
            coords = get_coords_from_buses(net, params['bus1'], params['bus2'])
    else:
        coords = get_coords_from_grf_object(item)

    if len(line_sections) == 0:
        if coords:
            params["geodata"] = coords
        logger.debug('line <%s> has no sections' % params['name'])
        lid = create_line_normal(net=net, item=item, is_unbalanced=is_unbalanced, **params)
        sid_list = [lid]
        line_dict[item] = sid_list
        logger.debug('created line <%s> with index <%d>' % (params['name'], lid))

    else:
        logger.debug('line <%s> has sections' % params['name'])
        sid_list = create_line_sections(net=net, item_list=line_sections, line=item,
                                        coords=coords, is_unbalanced=is_unbalanced, **params)
        line_dict[item] = sid_list
        logger.debug('created <%d> line sections for line <%s>' % (len(sid_list), params['name']))

    net.line.loc[sid_list, "line_idx"] = corridor
    net.line.loc[sid_list, "folder_id"] = item.fold_id.loc_name
    net.line.loc[sid_list, "equipment"] = item.for_name
    create_connection_switches(net, item, 2, 'l', (params['bus1'], params['bus2']),
                               (sid_list[0], sid_list[-1]))

    logger.debug('line <%s> created' % params['name'])


def point_len(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def calc_len_coords(coords):
    tot_len = 0
    for i in range(len(coords) - 1):
        tot_len += point_len(coords[i], coords[i + 1])
    return tot_len


def cut_coords_segment(p1, p2, split_len):
    # finds the point where the line segment is cut in two
    # use scale_factor before calling this
    if split_len == 0:
        logger.debug('split_len=0: return %s' % p1)
        return p1
    tot_len = point_len(p1, p2)
    if split_len == tot_len:
        logger.debug('split_len=tot_len (%.3f): return %s' % (tot_len, p2))
        return p2
    x1, y1 = p1
    x2, y2 = p2
    x_k = x2 * split_len / tot_len + x1 * (tot_len - split_len) / tot_len
    y_k = y2 * split_len / tot_len + y1 * (tot_len - split_len) / tot_len
    logger.debug('cut coords segment: p1=%s, p2=%s, split_len=%.3f, x_k = %.3f, y_k = %.3f' %
                 (p1, p2, split_len, x_k, y_k))
    return x_k, y_k


def get_section_coords(coords, sec_len, start_len, scale_factor):
    tol = 1e-6
    # get the starting point of section
    logger.debug('calculating section coords: sec_len=%.3f, start_len=%.3f, scale_factor=%.3f' %
                 (sec_len, start_len, scale_factor))
    if abs(sec_len) < tol and scale_factor == 0:
        logger.debug('cannot do anything: sec_len and scale factor are 0')
        sec_coords = [coords[0], coords[1]]
        return sec_coords
    elif scale_factor == 0:
        logger.error('scale factor==0')
        sec_coords = [coords[0], coords[1]]
        return sec_coords
    elif abs(sec_len) < tol:
        logger.debug('section length is 0')

    len_i = 0
    i = 0
    sec_coords = []
    logger.debug('len coords: %d, coords: %s' % (len(coords), coords))
    # find starting point
    for i in range(len(coords) - 1):
        len_i += point_len(coords[i], coords[i + 1])
        logger.debug('i: %d, len_i: %.3f' % (i, len_i * scale_factor))
        # catch if line has identical coords
        if not len_i:
            sec_coords = coords
            return sec_coords

        if len_i * scale_factor > start_len or abs(len_i * scale_factor - start_len) <= tol:
            logger.debug('len_i>start_len: cut coords segment')
            logger.debug('coords[i]: %s, coods[i+1]: %s' % (coords[i], coords[i + 1]))
            if start_len == 0:
                sec_coords = [[coords[i][0], coords[i][1]]]
            else:
                x_k, y_k = cut_coords_segment(coords[i], coords[i + 1],
                                              start_len / scale_factor +
                                              point_len(coords[i], coords[i + 1]) - len_i)
                sec_coords = [[x_k, y_k]]
            logger.debug('found start of the section: %s' % sec_coords)
            break
    else:
        logger.error(
            'could not find start of section: len_i = %.7f, start_len = %.7f' % (
                len_i * scale_factor, start_len))
        logger.debug('delta: %f' % (len_i * scale_factor - start_len))

    # keep adding points until encounter the end of line
    len_j = 0
    k = 0
    for j in range(i + 1, len(coords)):
        try:
            len_j += point_len(sec_coords[k], coords[j])
        except IndexError:
            logger.error(f"{j=}, {i=}, {k=}")
        if len_j <= sec_len / scale_factor:
            sec_coords.append(coords[j])
            k += 1
        else:
            # cut line between coords[i] and coords[i+1]
            x_k, y_k = cut_coords_segment(sec_coords[k], coords[j],
                                          sec_len / scale_factor +
                                          point_len(sec_coords[k], coords[j]) - len_j)
            sec_coords.append([x_k, y_k])
            break
    logger.debug('calculated sec_coords: %s' % sec_coords)
    return sec_coords


def segment_buses(net, bus1, bus2, num_sections, line_name):  # , sec_len, start_len, coords):
    yield bus1
    m = 1
    # if coords:
    #     if bus1 not in net.bus_geodata.index:
    #         logger.warning('bus1 not in coords, bus: %d, line: %s' % (bus1, line_name))
    #     if bus2 not in net.bus_geodata.index:
    #         logger.warning('bus2 not in coords, bus: %d, line: %s' % (bus2, line_name))
    #     x1, y1 = net.bus_geodata.loc[bus1, ['x', 'y']]
    #     x2, y2 = net.bus_geodata.loc[bus2, ['x', 'y']]
    #     # tot_len = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    #     tot_len = calc_len_coords(coords)
    #     scale_factor = tot_len / sum(sec_len)
    #     split_len = 0

    while m < num_sections:
        bus_name = "%s (Muff %u)" % (line_name, m)
        vn_kv = net.bus.at[bus1, "vn_kv"]
        zone = net.bus.at[bus1, "zone"]
        k = pp.create_bus(net, name=bus_name, type='ls', vn_kv=vn_kv, zone=zone)

        # if coords:
        #     split_len += sec_len[m - 1] * scale_factor
        #
        #     x_k, y_k = cut_coords_segment([x1, y1], [x2, y2], split_len)
        #     net.bus_geodata.loc[k, ['x', 'y']] = x_k, y_k
        #     if x_k == 0 or y_k == 0:
        #         logger.warning('bus %d has 0 coords, bus1: %d, bus2: %d' % k, bus1, bus2)

        if "description" in net.bus:
            net.bus.at[k, "description"] = u""
        yield k
        yield k
        m += 1
    else:
        yield bus2


def create_line_sections(net, item_list, line, bus1, bus2, coords, parallel, is_unbalanced, **kwargs):
    sid_list = []
    line_name = line.loc_name
    item_list.sort(key=lambda x: x.index)  # to ensure they are in correct order

    if line.HasResults(-1):  # -1 for 'c' results (whatever that is...)
        line_loading = ga(line, 'c:loading')
    else:
        line_loading = np.nan

    sec_len = [sec.dline for sec in item_list]
    # start_len = [sec.rellen for sec in item_list]

    buses_gen = segment_buses(net, bus1=bus1, bus2=bus2, num_sections=len(item_list),
                              line_name=line_name)

    for item in item_list:
        name = line_name
        section_name = item.loc_name
        bus1 = next(buses_gen)
        bus2 = next(buses_gen)
        sid = create_line_normal(net=net, item=item, bus1=bus1, bus2=bus2, name=name,
                                 parallel=parallel, is_unbalanced=is_unbalanced)
        sid_list.append(sid)
        net.line.at[sid, "section"] = section_name
        net.res_line.at[sid, "pf_loading"] = line_loading

        if coords:
            try:
                scaling_factor = sum(sec_len) / calc_len_coords(coords)
                sec_coords = get_section_coords(coords, sec_len=item.dline, start_len=item.rellen,
                                                scale_factor=scaling_factor)
                net.line_geodata.loc[sid, 'coords'] = sec_coords
                # p1 = sec_coords[0]
                # p2 = sec_coords[-1]
                net.bus_geodata.loc[bus2, ['x', 'y']] = sec_coords[-1]
            except ZeroDivisionError:
                logger.warning("Could not generate geodata for line !!")

    return sid_list


def create_line_normal(net, item, bus1, bus2, name, parallel, is_unbalanced, geodata=None):
    pf_type = item.typ_id
    std_type, type_created = create_line_type(net=net, item=pf_type,
                                              cable_in_air=item.inAir if item.HasAttribute(
                                                  'inAir') else False)

    params = {
        'from_bus': bus1,
        'to_bus': bus2,
        'name': name,
        'in_service': not bool(item.outserv),
        'length_km': item.dline,
        'df': item.fline,
        'parallel': parallel,
        'alpha': pf_type.alpha if pf_type is not None else None,
        'temperature_degree_celsius': pf_type.tmax if pf_type is not None else None,
        'geodata': geodata
    }

    if std_type is not None: #and not is_unbalanced:delete later
        params["std_type"] = std_type
        logger.debug('creating normal line with type <%s>' % std_type)
        lid = pp.create_line(net, **params)
    else:
        logger.debug('creating normal line <%s> from parameters' % name)
        r_ohm, x_ohm, c_nf = item.R1, item.X1, item.C1
        r0_ohm, x0_ohm, c0_nf = item.R0, item.X0, item.C0


        if r_ohm == 0 and x_ohm == 0 and c_nf == 0:
            logger.error('Incomplete data for line "%s": missing type and '
                         'missing parameters R, X, C' % name)
        if r0_ohm == 0 and x0_ohm == 0 and c0_nf == 0:
            logger.error('Incomplete data for line "%s": missing type and '
                         'missing parameters R0, X0, C0' % name)
        params.update({
            'r_ohm_per_km': r_ohm / params['length_km'],
            'x_ohm_per_km': x_ohm / params['length_km'],
            'c_nf_per_km': c_nf / params['length_km'] * 1e3,  # internal unit for C in PF is uF
            'r0_ohm_per_km': r0_ohm / params['length_km'],
            'x0_ohm_per_km': x0_ohm / params['length_km'],
            'c0_nf_per_km': c0_nf / params['length_km'] * 1e3,  # internal unit for C in PF is uF,
            'max_i_ka': item.Inom if item.Inom != 0 else 1e-3,
            'alpha': pf_type.alpha if pf_type is not None else None
        })

        coupling = item.c_ptow
        if coupling is not None:
            coupling_type = coupling.GetClassName()
            if coupling_type == 'TypCabsys':
                # line is part of "Cable System"
                params['type'] = 'cs'
            elif coupling_type == 'ElmTow':
                params['type'] = 'ol'
            else:
                params['type'] = None
        else:
            params['type'] = None

        lid = pp.create_line_from_parameters(net=net, **params)

    net.line.loc[lid, 'description'] = ' \n '.join(item.desc) if len(item.desc) > 0 else ''
    if hasattr(item, "cimRdfId"):
        chr_name = item.cimRdfId
        if chr_name is not None and len(chr_name) > 0:
            net["line"].loc[lid, 'origin_id'] = chr_name[0]

    get_pf_line_results(net, item, lid, is_unbalanced)

    return lid


def get_pf_line_results(net, item, lid, is_unbalanced):
    line_type = None
    result_variables = None
    if is_unbalanced:
        line_type = "res_line_3ph"
        result_variables = {
          "pf_i_a_from_ka": "m:I:bus1:A",
          "pf_i_a_to_ka": "m:I:bus2:A",
          "pf_i_b_from_ka": "m:I:bus1:B",
          "pf_i_b_to_ka": "m:I:bus2:B",
          "pf_i_c_from_ka": "m:I:bus1:C",
          "pf_i_c_to_ka": "m:I:bus2:C",
          "pf_i_n_from_ka": "m:I0x3:bus1",
          "pf_i_n_to_ka": "m:I0x3:bus2",
          "pf_loading_percent": "c:loading",
        }
    else:
        line_type = "res_line"
        result_variables = {
               "pf_loading": "c:loading"
               }

    for res_var_pp, res_var_pf in result_variables.items():
        res = np.nan
        if item.HasResults(-1): # -1 for 'c' results (whatever that is...)
            res = ga(item, res_var_pf)
        net[line_type].at[lid, res_var_pp] = res


def create_line_type(net, item, cable_in_air=False):
    # return False if no line type has been created
    # return True if a new line type has been created

    if item is None:
        logger.warning('create_line_type: no item given! Be sure you can deal with None!')
        return None, False

    # pf_folder = item.fold_id.split('IntPrjfolder')[-1]
    pf_folder = item.fold_id.loc_name
    name = "%s\\%s" % (pf_folder, item.loc_name) if not cable_in_air else "%s\\%s_%s" % (
        pf_folder, item.loc_name, 'air')
    if pp.std_type_exists(net, name):
        logger.debug('type <%s> exists already' % name)
        return name, False

    line_or_cable = 'cs' if item.cohl_ == 0 else 'ol'

    max_i_ka = item.sline if not cable_in_air else item.InomAir
    type_data = {
        "r_ohm_per_km": item.rline,
        "x_ohm_per_km": item.xline,
        "c_nf_per_km": item.cline*item.frnom/50 * 1e3,  # internal unit for C in PF is uF
        "q_mm2": item.qurs,
        "max_i_ka": max_i_ka if max_i_ka != 0 else 1e-3,
        "endtemp_degree": item.rtemp,
        "type": line_or_cable,
        "r0_ohm_per_km": item.rline0,
        "x0_ohm_per_km": item.xline0,
        "c0_nf_per_km": item.cline0*item.frnom/50 * 1e3,  # internal unit for C in PF is uF
        "alpha": item.alpha
    }
    pp.create_std_type(net, type_data, name, "line")
    logger.debug('>> created line type <%s>' % name)

    return name, True


def monopolar_in_service(item):
    in_service = not bool(item.outserv)

    # False if open, True if closed, None if no switch
    switch_is_closed, _ , _ = import_switch(item, 0)
    if switch_is_closed is not None:
        logger.debug('element in service: <%s>, switch is closed: <%s>' %
                     (in_service, switch_is_closed))
        # if switch is open, in_sevice becomes False
        in_service = in_service and switch_is_closed
    return in_service


def create_ext_net(net, item, pv_as_slack, is_unbalanced):
    name = item.loc_name
    logger.debug('>> creating ext_grid <%s>' % name)

    try:
        bus1 = get_connection_nodes(net, item, 1)
    except IndexError:
        logger.error("Cannot add Xnet '%s': not connected" % name)
        return

    logger.debug('found bus <%d> in net' % bus1)

    in_service = monopolar_in_service(item)

    # implement kW!..
    p_mw = item.pgini
    q_mvar = item.qgini
    logger.debug('p_mw = %.3f, q_mvar = %.3f' % (p_mw, q_mvar))

    # implementation for other elements that should be created as xnet
    s_max = item.snss if item.HasAttribute('snss') else item.Pmax_uc \
        if item.HasAttribute('Pmax_uc') else np.nan
    # needed change in create.py: line 570 - added sk_min_mva in list of params
    s_min = item.snssmin if item.HasAttribute('snssmin') else item.Pmin_uc \
        if item.HasAttribute('Pmin_uc') else np.nan

    rx_max = item.rntxn if item.HasAttribute('rntxn') else np.nan
    rx_min = item.rntxnmin if item.HasAttribute('rntxnmin') else np.nan

    vm_set_pu = item.usetp
    phi = item.phiini
    node_type = item.bustp if item.HasAttribute('bustp') else np.nan

    # create...
    if node_type == 'PQ':
        logger.debug('node type is "PQ", creating sgen')
        xid = pp.create_sgen(net, bus1, p_mw=p_mw, q_mvar=q_mvar, name=name,
                             in_service=in_service)
        elm = 'sgen'
    elif node_type == 'PV' and not pv_as_slack:
        logger.debug('node type is "PV" and pv_as_slack is False, creating gen')
        xid = pp.create_gen(net, bus1, p_mw=p_mw, vm_pu=vm_set_pu, name=name,
                            in_service=in_service)
        elm = 'gen'
    else:
        logger.debug('node type is <%s>, pv_as_slack=%s, creating ext_grid' % (node_type,
                                                                               pv_as_slack))
        xid = pp.create_ext_grid(net, bus=bus1, name=name, vm_pu=vm_set_pu,
                                 va_degree=phi, s_sc_max_mva=s_max,
                                 s_sc_min_mva=s_min, rx_max=rx_max, rx_min=rx_min,
                                 in_service=in_service)
        try:
            net.ext_grid.loc[xid, "r0x0_max"] = item.r0tx0
            net.ext_grid.loc[xid, "x0x_max"] = item.x0tx1
            net.ext_grid.loc[xid, "r0x0_min"] = item.r0tx0min
            net.ext_grid.loc[xid, "x0x_min"] = item.x0tx1min
        except AttributeError:
            pass
        elm = 'ext_grid'

    get_pf_ext_grid_results(net, item, xid, is_unbalanced)

    # if item.HasResults(0):  # 'm' results...
    #     # sm:r, sm:i don't work...
    #     logger.debug('<%s> has results' % name)
    #     net['res_' + elm].at[xid, "pf_p"] = ga(item, 'm:P:bus1')
    #     net['res_' + elm].at[xid, "pf_q"] = ga(item, 'm:Q:bus1')
    # else:
    #     net['res_' + elm].at[xid, "pf_p"] = np.nan
    #     net['res_' + elm].at[xid, "pf_q"] = np.nan

    # logger.debug('added pf_p and pf_q to {} {}: {}'.format(elm, xid, net['res_' + elm].loc[
    #     xid, ["pf_p", 'pf_q']].values))

    net[elm].loc[xid, 'description'] = ' \n '.join(item.desc) if len(item.desc) > 0 else ''
    add_additional_attributes(item, net, element=elm, element_id=xid, attr_list=['cpSite.loc_name'])

    return xid

def get_pf_ext_grid_results(net, item, xid, is_unbalanced):
    ext_grid_type = None
    result_variables = None
    if is_unbalanced:
        ext_grid_type = "res_ext_grid_3ph"
        result_variables = {
          "pf_p_a": "m:P:bus1:A",
          "pf_q_a": "m:Q:bus1:A",
          "pf_p_b": "m:P:bus1:B",
          "pf_q_b": "m:Q:bus1:B",
          "pf_p_c": "m:P:bus1:C",
          "pf_q_c": "m:Q:bus1:C",
        }
    else:
        ext_grid_type = "res_ext_grid"
        result_variables = {
               "pf_p": "m:P:bus1",
               "pf_q": "m:Q:bus1"
               }

    for res_var_pp, res_var_pf in result_variables.items():
        res = np.nan
        if item.HasResults(0):
            res = ga(item, res_var_pf)
        net[ext_grid_type].at[xid, res_var_pp] = res


# def extract_partial_loads_from_lv_load(net, item):
#     part_lods = item.GetContents('*.ElmLodlvp')
#     logger.debug('%s' % part_lods)
#     for elm in part_lods:
#         pass
#         # create_load(net, elm, use_nominal_power)

def map_power_var(pf_var, map_var):
    """
    Returns additional variables from pf_variable_p_xxxx
    Args:
        pf_var: pf_variable_p_xxxx
        map_var: 'q' or 's'

    Returns: pf_variable as string

    """

    vars = {
        'q': {
            'plini': 'qlini',
            'plini_a': 'qlini_a',
            'm:P:bus1': 'm:Q:bus1',
            'pgini': 'qgini',
            'pgini_a': 'qgini_a'
        },
        's': {
            'plini': 'slini',
            'plini_a': 'slini_a',
            'm:P:bus1': 'm:S:bus1',
            'pgini': 'sgini',
            'pgini_a': 'sgini_a'
        },
        'sn': {
            'plini': 'sgn',
            'plini_a': 'sgn',
            'm:P:bus1': 'sgn',
            'pgini': 'sgn',
            'pgini_a': 'sgn'
        }
    }

    return vars[map_var][pf_var]


def map_type_var(pf_load_type):
    load_type = {
            "3PH PH-E" : "wye",
            "3PH-'YN'" : "wye",
            "3PH-'D'": "delta"
            }
    return load_type[pf_load_type]


def map_sgen_type_var(pf_sgen_type):
    sgen_type = {
            0: "wye",
            1: "wye",
            2: "wye"
            }
    return sgen_type[pf_sgen_type]


def get_power_multiplier(item, var):
    if item.outserv:
        return 1.
    if var == "m:P:bus1" and not item.HasResults():
        return 1.
        # raise UserWarning(f"{item} does not have results - cannot get power multiplier")
    exponent = item.GetAttributeUnit(var)
    if exponent.startswith('k'):
        multiplier = 1e-3
    elif exponent.startswith('M'):
        multiplier = 1
    else:
        raise UserWarning("unknown exponent %s" % exponent)
    # print(item.loc_name, exponent, multiplier)
    return multiplier


def ask_load_params(item, pf_variable_p_loads, dict_net, variables):
    multiplier = get_power_multiplier(item, pf_variable_p_loads)
    params = ADict()
    if pf_variable_p_loads == 'm:P:bus1' and not item.HasResults(0):
        raise RuntimeError('load %s does not have results and is ignored' % item.loc_name)
    if 'p_mw' in variables:
        params.p_mw = ga(item, pf_variable_p_loads) * multiplier
    if 'q_mvar' in variables:
        params.q_mvar = ga(item, map_power_var(pf_variable_p_loads, 'q')) * multiplier
    if 'sn_mva' in variables:
        params.sn_mva = ga(item, map_power_var(pf_variable_p_loads, 's')) * multiplier

        kap = -1 if item.pf_recap == 1 else 1
        try:
            params.q_mvar = kap * np.sqrt(params.sn_mva ** 2 - params.p_mw ** 2)
        except Exception as err:
            logger.error(
                'While creating load, error occurred for calculation of q_mvar: %s, %s' %
                (params, err))
            raise err
        logger.debug('load parameters: %s' % params)

    global_scaling = dict_net['global_parameters']['global_load_scaling']
    params.scaling = global_scaling * item.scale0 \
                            if pf_variable_p_loads == 'plini' else 1
    if item.HasAttribute('zonefact'):
        params.scaling *= item.zonefact

    # p_mw = p_mw, q_mvar = q_mvar, scaling = scaling

    return params

def ask_unbalanced_load_params(item, pf_variable_p_loads, dict_net, variables):
    params = ADict()
    if pf_variable_p_loads == 'm:P:bus1' and not item.HasResults(0):
        raise RuntimeError('load %s does not have results and is ignored' % item.loc_name)
    if 'p_mw' in variables:
        params.p_a_mw = ga(item, pf_variable_p_loads+"r")
        params.p_b_mw = ga(item, pf_variable_p_loads+"s")
        params.p_c_mw = ga(item, pf_variable_p_loads+"t")
    if 'q_mvar' in variables:
        params.q_a_mvar = ga(item, map_power_var(pf_variable_p_loads, 'q')+"r")
        params.q_b_mvar = ga(item, map_power_var(pf_variable_p_loads, 'q')+"s")
        params.q_c_mvar = ga(item, map_power_var(pf_variable_p_loads, 'q')+"t")
    if 'sn_mva' in variables:
        params.sn_a_mva = ga(item, map_power_var(pf_variable_p_loads, 's')+"r")
        params.sn_b_mva = ga(item, map_power_var(pf_variable_p_loads, 's')+"s")
        params.sn_c_mva = ga(item, map_power_var(pf_variable_p_loads, 's')+"t")

        kap = -1 if item.pf_recap == 1 else 1
        try:
            params.q_mvar = kap * np.sqrt(params.sn_mva ** 2 - params.p_mw ** 2)
        except Exception as err:
            logger.error(
                'While creating load, error occurred for calculation of q_mvar: %s, %s' %
                (params, err))
            raise err
        logger.debug('load parameters: %s' % params)

    global_scaling = dict_net['global_parameters']['global_load_scaling']
    params.scaling = global_scaling * item.scale0 \
                            if pf_variable_p_loads == 'plini' else 1
    if item.HasAttribute('zonefact'):
        params.scaling *= item.zonefact

    return params


def find_section(load, sections):
    tot_len = 0
    for s in sections:
        tot_len += s.dline
        if tot_len >= load.lneposkm:
            break
    else:
        raise RuntimeError('could not find section for load %s' % load)
    return s


def make_split_dict(line):
    # für jede einzelne line
    sections = line.GetContents('*.ElmLnesec')
    loads = line.GetContents('*.ElmLodlvp')
    if len(loads) > 0:
        loads.sort(key=lambda x: x.lneposkm)
    else:
        return {}

    split_dict = {}
    if len(sections) > 0:
        sections.sort(key=lambda x: x.index)
        for load in loads:
            section = find_section(load, sections)
            split_dict[section] = split_dict.get(section, []).append(load)

    else:
        for load in loads:
            split_dict[line] = split_dict.get(line, []).append(load)
    return split_dict


def split_line_add_bus(net, split_dict):
    for line, loads in split_dict.items():
        # here we throw the stones
        if len(loads) == 0:
            continue
        loads = loads.sort(key=lambda x: x.lneposkm)
        lix = line_dict[line]
        coords = net.line.at[lix, 'coords']
        tot_len = calc_len_coords(coords)
        scale_factor = line.dline / tot_len

        start_len = 0
        list_coords = []
        len_sections = []
        list_bus_coords = []
        temp_len = 0
        for load in loads:
            sec_len = load.lneposkm - start_len
            temp_len += sec_len
            sec_coords = get_section_coords(coords, sec_len, start_len, scale_factor)
            list_coords.append(sec_coords)
            len_sections.append(sec_len)
            list_bus_coords.append(sec_coords[-1])
            start_len = load.lneposkm

        ## not sure if this is necessary
        # if temp_len < line.dline:
        #     # the last piece of line
        #     sec_len = line.dline - start_len
        #     sec_coords = get_section_coords(coords, sec_len, start_len, scale_factor)
        #     list_coords.append(sec_coords)
        #     len_sections.append(sec_len)

        # it's time to collect the stones
        # get bus voltage
        vn_kv = net.bus.at[net.line.at[lix, 'from_bus'], 'vn_kv']

        for i in range(len(len_sections)):
            # create bus
            name = 'Muff Partial Load'
            bus = pp.create_bus(net, name=name, vn_kv=vn_kv, geodata=list_bus_coords[i])
            # create new line
            from_bus = net.line.at[lix, 'from_bus']
            to_bus = net.line.at[lix, 'to_bus']
            std_type = net.line.at[lix, 'std_type']
            name = net.line.at[lix, 'name']
            new_lix = pp.create_line(net, from_bus=from_bus, to_bus=to_bus,
                                     length_km=len_sections[i],
                                     std_type=std_type, name=name)
            # change old line
            net.line.at[lix, 'to_bus'] = bus
            net.line.at[lix, 'length_km'] = net.line.at[lix, 'length_km'] - len_sections[i]
            pass


def split_line_add_bus_old(net, item, parent):
    # get position at line
    # find line section
    previous_sec_len = 0
    sections = parent.GetContents('*.ElmLnesec')
    sections.sort(key=lambda x: x.index)  # to ensure they are in correct order
    if len(sections) == 0:
        # cool! no sections - split the line
        sec = parent
        has_sections = False
        logger.debug('line has no sections')
    else:
        has_sections = True
        logger.debug('line has %d sections' % len(sections))
        for s in sections:
            logger.debug('section start: %s, load pos: %s, section end: %s' % (
                s.rellen, item.lneposkm, s.dline))
            if s.rellen <= item.lneposkm <= s.rellen + s.dline:
                sec = s
                logger.debug('found section: %s' % sec)
                break
            else:
                previous_sec_len += s.dline
        else:
            raise RuntimeError("could not find section where ElmLodlvp %s belongs" % item.loc_name)

    # found section in powerfactory
    # at this point the section can be split by other loads and its length can vary
    # now find section in pandapower net
    if has_sections:
        sid = net.line.loc[
            (net.line.name == parent.loc_name) & (net.line.section == sec.loc_name)].index
        logger.debug('index of section in net: %s' % sid)
    else:
        sid = net.line.loc[(net.line.name == parent.loc_name)].index
        logger.debug('index of line in net: %s' % sid)
    # check
    if len(sid) > 1:
        # section_idx is 0, 1, ...
        for m in range(len(sid)):
            # find the correct section for lodlvp
            temp_lines = net.line.loc[sid]
            logger.debug('temp_lines: %s' % temp_lines)
            temp_sec_len = float(temp_lines.loc[temp_lines.section_idx == m, 'length_km'])
            logger.debug('temp_sec_len of sec nr. %d: %.3f' % (m, temp_sec_len))
            if (temp_sec_len + previous_sec_len) >= item.lneposkm:
                # temp_section = temp_lines.query('section_idx == @m')
                # sid = temp_lines[temp_lines.section_idx == m].index.values[0]
                sid = sid[m]
                logger.debug('found section for creating lodlvp: %d' % sid)
                break
            else:
                previous_sec_len += temp_sec_len
        else:
            raise RuntimeError(
                "could not find line or section where ElmLodlvp %s belongs: multiple indices "
                "found in net and none of them is good" % item.loc_name)
    elif len(sid) == 0:
        raise RuntimeError(
            "could not find line or section where ElmLodlvp %s belongs: no index found in net" %
            item.loc_name)
    else:
        sid = sid.values[0]
        logger.debug('index is unique: %d' % sid)

    # new line lengths
    tot_len = net.line.at[sid, 'length_km']
    sec_len_a = item.lneposkm - previous_sec_len
    sec_len_b = tot_len - sec_len_a
    logger.debug('total length: %.3f, a: %.3f, b:%.3f' % (tot_len, sec_len_a, sec_len_b))
    if sec_len_b < 0:
        raise RuntimeError('incorrect length for section %s: %.3f' % (sec, sec_len_b))

    # get coords
    if sid in net.line_geodata.index.values:
        logger.debug('line has coords')
        coords = net.line_geodata.at[sid, 'coords']
        logger.debug('old geodata of line %d: %s' % (sid, coords))

        # get coords for 2 split lines
        coords_len = calc_len_coords(coords)
        scale_factor = parent.dline / coords_len  # scale = real_len / coords_len
        coords_a = get_section_coords(coords, sec_len_a, 0, scale_factor)
        coords_b = get_section_coords(coords, sec_len_b, sec_len_a, scale_factor)
        logger.debug('new coords: %s; %s' % (coords_a, coords_b))

        # get bus coords
        bus_coords = tuple(coords_b[0])
        logger.debug('new bus coords: %.3f, %.3f' % bus_coords)
    else:
        logger.debug('line has no coords')
        bus_coords = None
        coords_a = None
        coords_b = None

    if sec_len_b > 0:
        # create new bus
        vn_kv = net.bus.at[net.line.at[sid, 'from_bus'], 'vn_kv']
        name = 'LodLV-%s' % item.loc_name
        bus = pp.create_bus(net, vn_kv=vn_kv, name=name, geodata=bus_coords, type='n')
        net.bus.loc[bus, 'description'] = 'Partial load %s = %.3f kW' % (item.loc_name, item.plini)
        logger.debug('created new bus in net: %s' % net.bus.loc[bus])

        # create new line
        lid = pp.create_line(net, from_bus=bus, to_bus=net.line.at[sid, 'to_bus'],
                             length_km=sec_len_b,
                             std_type=net.line.at[sid, 'std_type'],
                             name=net.line.at[sid, 'name'], df=net.line.at[sid, 'df'])
        net.line.at[lid, 'section'] = net.line.at[sid, 'section']
        net.line_geodata.loc[lid, 'coords'] = coords_b
        if not net.line.loc[sid, 'section_idx']:
            net.line.loc[sid, 'section_idx'] = 0

        net.line.loc[lid, 'section_idx'] = net.line.at[sid, 'section_idx'] + 1

        logger.debug('old line: %s' % net.line.loc[sid])
        logger.debug('new line: %s' % net.line.loc[lid])

        net.line.at[sid, 'to_bus'] = bus
        net.line.at[sid, 'length_km'] = sec_len_a
        net.line_geodata.loc[sid, 'coords'] = coords_a
        logger.debug('changed: %s' % net.line.loc[sid])
    else:
        # no new bus/line are created: take the to_bus
        bus = net.line.at[sid, 'to_bus']
    return bus


def create_load(net, item, pf_variable_p_loads, dict_net, is_unbalanced):
    # params collects the input parameters for the create function
    params = ADict()
    bus_is_known = False
    params.name = item.loc_name
    load_class = item.GetClassName()
    logger.debug('>> creating load <%s.%s>' % (params.name, load_class))

    is_unbalanced = item.i_sym

    ask = ask_unbalanced_load_params if is_unbalanced else ask_load_params

    if load_class == 'ElmLodlv':
        # if bool(ga(item, 'e:cHasPartLod')):
        #     logger.info('ElmLodlv %s has partial loads - skip' % item.loc_name)
        #     part_lods = item.GetContents('*.ElmLodlvp')
        #     logger.debug('%s' % part_lods)
        #     return
        # else:
        #     params.update(ask(item, pf_variable_p_loads, 'p_mw', 'sn_mva'))
        try:
            params.update(ask(item, pf_variable_p_loads, dict_net=dict_net,
                                          variables=('p_mw', 'sn_mva')))
        except Exception as err:
            logger.error("m:P:bus1 and m:Q:bus1 should be used with ElmLodlv")
            logger.error('While creating load %s, error occurred for '
                         'calculation of q_mvar: %s, %s' % (item, params, err))
            raise err

    elif load_class == 'ElmLodmv':
        params.update(ask(item, pf_variable_p_loads=pf_variable_p_loads,
                                      dict_net=dict_net, variables=('p_mw', 'sn_mva')))

    elif load_class == 'ElmLod':
        params.update(ask(item, pf_variable_p_loads=pf_variable_p_loads,
                                      dict_net=dict_net, variables=('p_mw', 'q_mvar')))

    ### for now - don't import ElmLodlvp
    elif load_class == 'ElmLodlvp':
        parent = item.fold_id
        parent_class = parent.GetClassName()
        logger.debug('parent class name of ElmLodlvp: %s' % parent_class)
        if parent_class == 'ElmLodlv':
            raise NotImplementedError('ElmLodlvp as not part of ElmLne not implemented')
        elif parent_class == 'ElmLne':
            logger.debug('creating load that is part of line %s' % parent)
            params.update(ask(item, pf_variable_p_loads=pf_variable_p_loads,
                                          dict_net=dict_net, variables=('p_mw', 'sn_mva')))
            params.name += '(%s)' % parent.loc_name
            split_dict = make_split_dict(parent)
            # todo remake this
            params.bus = split_line_add_bus(net, split_dict)
            bus_is_known = True
            logger.debug('created bus <%d> in net and changed lines' % params.bus)
        else:
            raise NotImplementedError('ElmLodlvp as part of %s not implemented' % parent)

    else:
        logger.warning(
            'item <%s> not imported - <%s> not implemented yet!' % (item.loc_name, load_class))
        raise RuntimeError('Load <%s> of type <%s> not implemented!' % (item.loc_name, load_class))

    # implement negative load as sgen:
    # if p_mw < 0:
    #     create_sgen_load(net=net, item=item)
    #     return

    if not bus_is_known:
        try:
            params.bus = get_connection_nodes(net, item, 1)
            logger.debug('found bus <%d> in net' % params.bus)
        except IndexError:
            logger.error("Cannot add Load '%s': not connected" % params.name)
            return

    params.in_service = not bool(item.outserv)

    if load_class != 'ElmLodlvp':
        params.in_service = monopolar_in_service(item)

    logger.debug('parameters: %s' % params)

    if is_unbalanced:
        pf_load_type = item.phtech
        params.type = map_type_var(pf_load_type)
    # create...
    try:
        # net, bus, p_mw, q_mvar=0, sn_mva=np.nan, name=None, scaling=1., index=None,
        # in_service=True, type=None
        if is_unbalanced:
            ld = pp.create_asymmetric_load(net, **params)
        else:
            ld = pp.create_load(net, **params)
        logger.debug('created load with index <%d>' % ld)
    except Exception as err:
        logger.error('While creating %s.%s, error occured: %s' % (params.name, load_class, err))
        raise err

    load_type = None

    if is_unbalanced:
        load_type = "asymmetric_load"
    else:
        load_type = "load"

    net[load_type].loc[ld, 'description'] = ' \n '.join(item.desc) if len(item.desc) > 0 else ''
    attr_list = ["sernum", "chr_name", 'cpSite.loc_name']
    if load_class == 'ElmLodlv':
        attr_list.extend(['pnight', 'cNrCust', 'cPrCust', 'UtilFactor', 'cSmax', 'cSav', 'ccosphi'])
    add_additional_attributes(item, net, load_type, ld, attr_dict={"for_name": "equipment"}, attr_list=attr_list)
    get_pf_load_results(net, item, ld, is_unbalanced)
#    if not is_unbalanced:
#        if item.HasResults(0):  # 'm' results...
#            logger.debug('<%s> has results' % params.name)
#            net["res_load"].at[ld, "pf_p"] = ga(item, 'm:P:bus1')
#            net["res_load"].at[ld, "pf_q"] = ga(item, 'm:Q:bus1')
#        else:
#            net["res_load"].at[ld, "pf_p"] = np.nan
#            net["res_load"].at[ld, "pf_q"] = np.nan

    logger.debug('created load <%s> at index <%d>' % (params.name, ld))

def get_pf_load_results(net, item, ld, is_unbalanced):
    load_type = None
    result_variables = None
    if is_unbalanced:
        load_type = "res_asymmetric_load_3ph"
        result_variables = {
            "pf_p_a": "m:P:bus1:A",
            "pf_p_b": "m:P:bus1:B",
            "pf_p_c": "m:P:bus1:C",
            "pf_q_a": "m:Q:bus1:A",
            "pf_q_b": "m:Q:bus1:B",
            "pf_q_c": "m:Q:bus1:C",
            }
    else:
        load_type = "res_load"
        result_variables = {
               "pf_p": "m:P:bus1",
               "pf_q": "m:Q:bus1"
               }

    for res_var_pp, res_var_pf in result_variables.items():
        res = np.nan
        if item.HasResults(0):
            res = ga(item, res_var_pf) * get_power_multiplier(item, res_var_pf)
        net[load_type].at[ld, res_var_pp] = res




def ask_gen_params(item, pf_variable_p_gen, *vars):
    multiplier = get_power_multiplier(item, pf_variable_p_gen)
    params = ADict()
    if pf_variable_p_gen == 'm:P:bus1' and not item.HasResults(0):
        raise RuntimeError('generator %s does not have results and is ignored' % item.loc_name)
    if 'p_mw' in vars:
        params.p_mw = ga(item, pf_variable_p_gen) * multiplier
    if 'q_mvar' in vars:
        params.q_mvar = ga(item, map_power_var(pf_variable_p_gen, 'q')) * multiplier
    if 'sn_mva' in vars:
        params.sn_mva = ga(item, map_power_var(pf_variable_p_gen, 'sn')) * multiplier

    params.scaling = item.scale0 if pf_variable_p_gen == 'pgini' else 1
    # p_mw = p_mw, q_mvar = q_mvar, scaling = scaling

    return params

def ask_unbalanced_sgen_params(item, pf_variable_p_sgen, *vars):

    params = ADict()
    if pf_variable_p_sgen == 'm:P:bus1' and not item.HasResults(0):
        raise RuntimeError('sgen %s does not have results and is ignored' % item.loc_name)

    technology = item.phtech
    if technology in [0, 1]: # (0-1: 3PH)
        if 'p_mw' in vars:
            params.p_a_mw = ga(item, pf_variable_p_sgen)/3
            params.p_b_mw = ga(item, pf_variable_p_sgen)/3
            params.p_c_mw = ga(item, pf_variable_p_sgen)/3
        if 'q_mvar' in vars:
            params.q_a_mvar = ga(item, map_power_var(pf_variable_p_sgen, 'q'))/3
            params.q_b_mvar = ga(item, map_power_var(pf_variable_p_sgen, 'q'))/3
            params.q_c_mvar = ga(item, map_power_var(pf_variable_p_sgen, 'q'))/3
    elif technology in [2, 3, 4]: # (2-4: 1PH)
        if 'p_mw' in vars:
            params.p_a_mw = ga(item, pf_variable_p_sgen)
            params.p_b_mw = 0
            params.p_c_mw = 0
        if 'q_mvar' in vars:
            params.q_a_mvar = ga(item, map_power_var(pf_variable_p_sgen, 'q'))
            params.q_b_mvar = 0
            params.q_c_mvar = 0

    if 'sn_mva' in vars:
        params.sn_mva = ga(item, map_power_var(pf_variable_p_sgen, 's'))

    params.scaling = item.scale0 if pf_variable_p_sgen == 'pgini' else 1
    return params


def create_sgen_genstat(net, item, pv_as_slack, pf_variable_p_gen, dict_net, is_unbalanced):
    params = ADict()
    categories = {"wgen": "WKA", "pv": "PV", "reng": "REN", "stg": "SGEN"}
    params.name = item.loc_name
    logger.debug('>> creating genstat <%s>' % params)

    av_mode = item.av_mode
    is_reference_machine = bool(item.ip_ctrl)

    ask = ask_unbalanced_sgen_params if is_unbalanced else ask_gen_params

    if is_reference_machine or (av_mode == 'constv' and pv_as_slack):
        logger.info('Genstat <%s> to be imported as external grid' % params.name)
        logger.debug('genstat parameters: %s' % params)
        sg = create_ext_net(net, item=item, pv_as_slack=pv_as_slack, is_unbalanced=is_unbalanced)
        element = 'ext_grid'
    else:
        try:
            params.bus = get_connection_nodes(net, item, 1)
        except:
            logger.error("Cannot add Sgen '%s': not connected" % params.name)
            return

        params.update(ask(item, pf_variable_p_gen, 'p_mw', 'q_mvar', 'sn_mva'))
        logger.debug('genstat parameters: ' % params)

        params.in_service = monopolar_in_service(item)

        # category (wind, PV, etc):
        if item.GetClassName() == 'ElmPvsys':
            cat = 'PV'
        else:
            try:
                cat = categories[item.aCategory]
            except KeyError:
                cat = None
                logger.debug('sgen <%s> with category <%s> imported as <%s>' %
                               (params.name, item.aCategory, cat))
        # parallel units:
        ngnum = item.ngnum
        logger.debug('%d parallel generators of type %s' % (ngnum, cat))

        for param in params.keys():
            if any(param.startswith(prefix) for prefix in ["p_", "q_", "sn_"]):
                params[param] *= ngnum
#        params.p_mw *= ngnum
#        params.q_mvar *= ngnum
#        params.sn_mva *= ngnum
        if is_unbalanced:
            pf_sgen_type = item.phtech
            params.type = map_sgen_type_var(pf_sgen_type)
        else:
            params.type = cat

        # create...
        if av_mode == 'constv':
            logger.debug('av_mode: %s - creating as gen' % av_mode)
            params.vm_pu = item.usetp
            del params['q_mvar']
            sg = pp.create_gen(net, **params)
            element = 'gen'
        else:
            if is_unbalanced:
                sg = pp.create_asymmetric_sgen(net, **params)
                element = "asymmetric_sgen"
            else:
                sg = pp.create_sgen(net, **params)
                element = 'sgen'
    logger.debug('created sgen at index <%d>' % sg)

    net[element].at[sg, 'description'] = ' \n '.join(item.desc) if len(item.desc) > 0 else ''
    add_additional_attributes(item, net, element, sg, attr_dict={"for_name": "equipment"},
                              attr_list=["sernum", "chr_name", "cpSite.loc_name"])
    net[element].at[sg, 'scaling'] = dict_net['global_parameters']['global_generation_scaling'] * item.scale0
    get_pf_sgen_results(net, item, sg, is_unbalanced, element=element)

    logger.debug('created genstat <%s> as element <%s> at index <%d>' % (params.name, element, sg))


    ###########################

#    if is_unbalanced:
#        pf_sgen_type = item.phtech
#        params.type = map_type_var(pf_sgen_type)
#    # create...
#    try:
#        # net, bus, p_mw, q_mvar=0, sn_mva=np.nan, name=None, scaling=1., index=None,
#        # in_service=True, type=None
#        if is_unbalanced:
#            sg = pp.create_asymmetric_sgen(net, **params)
#            logger.info("CREATING UNBALANCED SGEN")
#        else:
#            logger.info("CREATING BALANCED SGEN")
#            sg = pp.create_sgen_genstat(net, **params)
#        logger.debug('created sgen with index <%d>' % sg)
#    except Exception as err:
#        logger.error('While creating %s.%s, error occured: %s' % (params.name, sgen_class, err))
#        raise err
#
#    sgen_type = None
#
#    if is_unbalanced:
#        sgen_type = "asymmetric_sgen"
#    else:
#        sgen_type = "sgen"
#
#    net[sgen_type].loc[sg, 'description'] = ' \n '.join(item.desc) if len(item.desc) > 0 else ''
#    attr_list = ["sernum", "for_name", "chr_name", 'cpSite.loc_name']
#    if sgen_class == 'ElmGenstat':
#        attr_list.extend(['pnight', 'cNrCust', 'cPrCust', 'UtilFactor', 'cSmax', 'cSav', 'ccosphi'])
#    add_additional_attributes(item, net, sgen_type, sg, attr_list=attr_list)
#    get_pf_sgen_results(net, item, sg, is_unbalanced)
#
#
#    logger.debug('created sgen <%s> at index <%d>' % (params.name, sg))

def get_pf_sgen_results(net, item, sg, is_unbalanced, element='sgen'):
    result_variables = None

    if is_unbalanced:
        technology = item.phtech
        sgen_type = "res_asymmetric_sgen_3ph"

        if technology in [0, 1]:
            result_variables = {
                "pf_p_a": "m:P:bus1:A",
                "pf_p_b": "m:P:bus1:B",
                "pf_p_c": "m:P:bus1:C",
                "pf_q_a": "m:Q:bus1:A",
                "pf_q_b": "m:Q:bus1:B",
                "pf_q_c": "m:Q:bus1:C",
                }
        elif technology in [2, 3, 4]:
            result_variables = {
                "pf_p_a": "m:P:bus1:A",
                "pf_p_b": None,
                "pf_p_c": None,
                "pf_q_a": "m:Q:bus1:A",
                "pf_q_b": None,
                "pf_q_c": None,
                }
    else:
        sgen_type = "res_%s" % element
        result_variables = {
               "pf_p": "m:P:bus1",
               "pf_q": "m:Q:bus1"
               }

    for res_var_pp, res_var_pf in result_variables.items():
        res = np.nan
        if item.HasResults(0):
            if res_var_pf is not None:
                res = ga(item, res_var_pf) * get_power_multiplier(item, res_var_pf)
            else:
                res = np.nan
        net[sgen_type].at[sg, res_var_pp] = res

def create_sgen_neg_load(net, item, pf_variable_p_loads, dict_net):
    raise UserWarning('not used')
    params = ADict()
    #    categories = {"wgen": "WKA", "pv": "PV", "reng": "REN", "stg": "SGEN"}
    # let the category be PV:
    params.type = None
    params.name = item.loc_name
    logger.debug('>> implementing negative load <%s> as sgen' % params.name)
    try:
        params.bus = get_connection_nodes(net, item, 1)
    except IndexError:
        logger.error("Cannot add Sgen '%s': not connected" % params.name)
        return

    params.update(ask_load_params(item, pf_variable_p_loads=pf_variable_p_loads,
                                  dict_net=dict_net, variables=('p_mw', 'q_mvar')))
    # rated S:
    params.sn_mva = math.sqrt(params.p_mw ** 2 + params.q_mvar ** 2)

    logger.debug('negative load parameters: %s' % params)

    params.in_service = monopolar_in_service(item)

    # create...
    sg = pp.create_sgen(net, **params)

    net.sgen.loc[sg, 'description'] = ' \n '.join(item.desc) if len(item.desc) > 0 else ''
    add_additional_attributes(item, net, "sgen", sg, attr_dict={"for_name": "equipment"},
                              attr_list=["sernum", "chr_name", "cpSite.loc_name"])

    if item.HasResults(0):  # 'm' results...
        logger.debug('<%s> has results' % params.name)
        net.res_sgen.at[sg, "pf_p"] = -ga(item, 'm:P:bus1')
        net.res_sgen.at[sg, "pf_q"] = -ga(item, 'm:Q:bus1')
    else:
        net.res_sgen.at[sg, "pf_p"] = np.nan
        net.res_sgen.at[sg, "pf_q"] = np.nan

    logger.debug('created load <%s> as sgen at index <%d>' % (params.name, sg))


def create_sgen_sym(net, item, pv_as_slack, pf_variable_p_gen, dict_net):
    categories = {"wgen": "WKA", "pv": "PV", "reng": "REN", "stg": "SGEN"}
    name = item.loc_name
    sid = None
    element = None
    logger.debug('>> creating synchronous machine <%s>' % name)
    av_mode = item.av_mode
    is_reference_machine = bool(item.ip_ctrl)
    is_motor = bool(item.i_mot)
    global_scaling = dict_net['global_parameters']['global_motor_scaling'] if is_motor else \
        dict_net['global_parameters']['global_generation_scaling']
    multiplier = get_power_multiplier(item, pf_variable_p_gen)

    if is_reference_machine or (av_mode == 'constv' and pv_as_slack):
        logger.info('synchronous machine <%s> to be imported as external grid' % name)
        logger.debug('ref. machine: %d, av_mode: %s, pv as slack: %s' %
                     (is_reference_machine, av_mode, pv_as_slack))
        sid = create_ext_net(net, item=item, pv_as_slack=pv_as_slack, is_unbalanced=False)
        net.ext_grid.loc[sid, 'p_disp_mw'] = -item.pgini * multiplier
        net.ext_grid.loc[sid, 'q_disp_mvar'] = -item.qgini * multiplier
        logger.debug('created ext net with sid <%d>', sid)
        element = 'ext_grid'
    else:
        try:
            bus1 = get_connection_nodes(net, item, 1)
        except IndexError:
            logger.error("Cannot add Sgen '%s': not connected" % name)
            return

        logger.debug('sgen <%s> is a %s' % (name, {True: 'motor', False: 'generator'}[is_motor]))

        in_service = monopolar_in_service(item)

        # category (wind, PV, etc):
        try:
            cat = categories[item.aCategory]
        except KeyError:
            cat = 'SGEN'
            logger.debug('sgen <%s> with category <%s> imported as <%s>' %
                           (name, item.aCategory, cat))

        # parallel units:
        ngnum = item.ngnum
        logger.debug('%d parallel generators' % ngnum)

        p_mw = ngnum * item.pgini * multiplier

        logger.debug('av_mode: %s' % av_mode)
        if av_mode == 'constv':
            logger.debug('creating sym %s as gen' % name)
            vm_pu = item.usetp
            sid = pp.create_gen(net, bus=bus1, p_mw=p_mw, vm_pu=vm_pu,
                                name=name, type=cat, in_service=in_service, scaling=global_scaling)
            element = 'gen'
        elif av_mode == 'constq':
            q_mvar = ngnum * item.qgini * multiplier
            sid = pp.create_sgen(net, bus=bus1, p_mw=p_mw, q_mvar=q_mvar,
                                 name=name, type=cat, in_service=in_service, scaling=global_scaling)
            element = 'sgen'

        if sid is None or element is None:
            logger.error('Error! Sgen not created')
        logger.debug('sym <%s>: p_mw = %.3f' % (name, p_mw))
        logger.debug('created sgen at index <%s>' % sid)

    net[element].loc[sid, 'description'] = ' \n '.join(item.desc) if len(item.desc) > 0 else ''
    add_additional_attributes(item, net, element, sid, attr_dict={"for_name": "equipment"},
                              attr_list=["sernum", "chr_name", "cpSite.loc_name"])

    if item.HasResults(0):  # 'm' results...
        logger.debug('<%s> has results' % name)
        net['res_' + element].at[sid, "pf_p"] = ga(item, 'm:P:bus1') * multiplier
        net['res_' + element].at[sid, "pf_q"] = ga(item, 'm:Q:bus1') * multiplier
    else:
        net['res_' + element].at[sid, "pf_p"] = np.nan
        net['res_' + element].at[sid, "pf_q"] = np.nan

    logger.debug('created genstat <%s> at index <%d>' % (name, sid))


def create_sgen_asm(net, item, pf_variable_p_gen, dict_net):
    is_motor = bool(item.i_mot)
    global_scaling = dict_net['global_parameters']['global_motor_scaling'] if is_motor else \
        dict_net['global_parameters']['global_generation_scaling']

    multiplier = get_power_multiplier(item, pf_variable_p_gen)
    p_res = ga(item, 'pgini') * multiplier
    q_res = ga(item, 'qgini') * multiplier
    if item.HasResults(0):
        q_res = ga(item, 'm:Q:bus1') / global_scaling * multiplier
    else:
        logger.warning('reactive power for asynchronous generator is not exported properly '
                       '(advanced modelling of asynchronous generators not implemented)')

    logger.debug('p_res: %.3f, q_res: %.3f' % (p_res, q_res))

    in_service = monopolar_in_service(item)

    logger.debug('in_service: %s' % in_service)

    try:
        bus = get_connection_nodes(net, item, 1)
    except IndexError:
        logger.error("Cannot add Sgen asm '%s': not connected" % item.loc_name)
        return

    params = {
        'name': item.loc_name,
        'bus': bus,
        'p_mw': item.pgini * multiplier,
        'q_mvar': item.qgini * multiplier if item.bustp == 'PQ' else q_res,
        'in_service': in_service,
        'scaling': global_scaling
    }

    logger.debug('params: %s' % params)

    sid = pp.create_sgen(net, **params)

    net.sgen.loc[sid, 'description'] = ' \n '.join(item.desc) if len(item.desc) > 0 else ''
    add_additional_attributes(item, net, "sgen", sid, attr_dict={"for_name": "equipment", "cimRdfId": "origin_id"},
                              attr_list=["sernum", "chr_name", "cpSite.loc_name"])

    if item.HasResults(0):
        net.res_sgen.at[sid, 'pf_p'] = ga(item, 'm:P:bus1') * multiplier
        net.res_sgen.at[sid, 'pf_q'] = ga(item, 'm:Q:bus1') * multiplier
    else:
        net.res_sgen.at[sid, 'pf_p'] = np.nan
        net.res_sgen.at[sid, 'pf_q'] = np.nan


def create_trafo_type(net, item):
    # return False if no line type has been created
    # return True if a new line type has been created

    logger.debug('>> creating trafo type')
    if item is None:
        logger.error('no item given!')
        return None, False

    pf_folder = item.fold_id.loc_name
    name = "%s\\%s" % (pf_folder, item.loc_name)
    if pp.std_type_exists(net, name):
        logger.debug('trafo type <%s> already exists' % name)
        return name, False

    type_data = {
        "sn_mva": item.strn,
        "vn_hv_kv": item.utrn_h,
        "vn_lv_kv": item.utrn_l,
        "vk_percent": item.uktr,
        "vkr_percent": item.uktrr,
        "pfe_kw": item.pfe,
        "i0_percent": item.curmg,
        "shift_degree": item.nt2ag * 30,
        "vector_group": item.vecgrp[:-1],
        "vk0_percent": item.uk0tr,
        "vkr0_percent": item.ur0tr,
        "mag0_percent": item.zx0hl_n,
        "mag0_rx": item.rtox0_n,
        "si0_hv_partial": item.zx0hl_h
    }

    if item.itapch:
        logger.debug('trafo <%s> has tap changer' % name)
        type_data.update({
            "tap_side": ['hv', 'lv', 'ext'][item.tap_side],  # 'ext' not implemented
            # see if it is an ideal phase shifter or a complex phase shifter
            # checking tap_step_percent because a nonzero value for ideal phase shifter can be stored in the object
            "tap_step_percent": item.dutap if item.tapchtype != 1 else 0,
            "tap_step_degree": item.dphitap if item.tapchtype == 1 else item.phitr,
            "tap_phase_shifter": True if item.tapchtype == 1 else False,
            "tap_max": item.ntpmx,
            "tap_min": item.ntpmn,
            "tap_neutral": item.nntap0
        })
        if item.tapchtype == 2:
            logger.warning("trafo %s has symmetrical tap changer (tap changer at both hv and "
                           "lv side) - not implemented, importing as asymmetrical tap changer at "
                           "side %s. Results will differ." % (item.loc_name, type_data['tap_side']))

    # In PowerFactory, if the first tap changer is absent, the second is also, even if the check was there
    if item.itapch and item.itapch2:
        logger.debug('trafo <%s> has tap2 changer' % name)
        type_data.update({
            "tap2_side": ['hv', 'lv', 'ext'][item.tap_side2],  # 'ext' not implemented
            # see if it is an ideal phase shifter or a complex phase shifter
            # checking tap_step_percent because a nonzero value for ideal phase shifter can be stored in the object
            "tap2_step_percent": item.dutap2 if item.tapchtype2 != 1 else 0,
            "tap2_step_degree": item.dphitap2 if item.tapchtype2 == 1 else item.phitr2,
            "tap2_phase_shifter": True if item.tapchtype2 == 1 else False,
            "tap2_max": item.ntpmx2,
            "tap2_min": item.ntpmn2,
            "tap2_neutral": item.nntap02
        })
        if item.tapchtype2 == 2:
            logger.warning("trafo %s has symmetrical tap2 changer (tap2 changer at both hv and "
                           "lv side) - not implemented, importing as asymmetrical tap2 changer at "
                           "side %s. Results will differ." % (item.loc_name, type_data['tap2_side']))

    if 'tap_side' in type_data.keys() and (type_data.get('tap_side') == 'ext' or type_data.get('tap_side') == 'ext'):
        logger.warning('controlled node of trafo "EXT" not implemented (type <%s>)' % name)
    pp.create_std_type(net, type_data, name, "trafo")
    logger.debug('created trafo type <%s> with params: %s' % (name, type_data))
    return name, True


def create_trafo(net, item, export_controller=True, tap_opt="nntap", is_unbalanced=False):
    name = item.loc_name  # type: str
    logger.debug('>> creating trafo <%s>' % name)
    in_service = not bool(item.outserv)  # type: bool

    # figure out the connection terminals
    try:
        bus1, bus2 = get_connection_nodes(net, item, 2)  # type: int
    except IndexError:
        logger.error("Cannot add Trafo '%s': not connected" % name)
        return

    propagate_bus_coords(net, bus1, bus2)

    if not net.bus.vn_kv[bus1] >= net.bus.vn_kv[bus2]:
        logger.error('trafo <%s>: violated condition of HV >= LV!' % name)
    # assert net.bus.vn_kv[bus1] >= net.bus.vn_kv[bus2]

    # figure out trafo type
    pf_type = item.typ_id
    if pf_type is None:
        logger.error('cannot create transformer <%s>: missing type' % name)
        return
    std_type, type_created = create_trafo_type(net=net, item=pf_type)

    # figure out current tap position
    tap_pos = np.nan
    if pf_type.itapch:
        if tap_opt == "nntap":
            tap_pos = ga(item, "nntap")
            logger.debug("got tap %f from nntap" % tap_pos)

        elif tap_opt == "c:nntap":
            tap_pos = ga(item, "c:nntap")
            logger.debug("got tap %f from c:nntap" % tap_pos)
        else:
            raise ValueError('could not read current tap position: tap_opt = %s' % tap_opt)

    tap_pos2 = np.nan
    # In PowerFactory, if the first tap changer is absent, the second is also, even if the check was there
    if pf_type.itapch and pf_type.itapch2:
        if tap_opt == "nntap":
            tap_pos2 = ga(item, "nntap2")
        elif tap_opt == "c:nntap":
            tap_pos2 = ga(item, "c:nntap2")

    if std_type is not None:
        tid = pp.create_transformer(net, hv_bus=bus1, lv_bus=bus2, name=name,
                                    std_type=std_type, tap_pos=tap_pos,
                                    in_service=in_service, parallel=item.ntnum, df=item.ratfac, tap2_pos=tap_pos2)
        logger.debug('created trafo at index <%d>' % tid)
    else:
        logger.info("Create Trafo 3ph")
        tid = pp.create_transformer_from_parameters(net, hv_bus=bus1, lv_bus=bus2, name=name,
                                    tap_pos=tap_pos,
                                    in_service=in_service, parallel=item.ntnum, df=item.ratfac,
                                    sn_mva=pf_type.strn, vn_hv_kv=pf_type.utrn_h, vn_lv_kv=pf_type.utrn_l,
                                    vk_percent=pf_type.uktr, vkr_percent=pf_type.uktrr,
                                    pfe_kw=pf_type.pfe, i0_percent=pf_type.curmg,
                                    vector_group=pf_type.vecgrp[:-1], vk0_percent=pf_type.uk0tr,
                                    vkr0_percent=pf_type.ur0tr, mag0_percent=pf_type.zx0hl_n,
                                    mag0_rx=pf_type.rtox0_n, si0_hv_partial=pf_type.zx0hl_h,
                                    shift_degree=pf_type.nt2ag * 30, tap2_pos=tap_pos2)

    # add value for voltage setpoint
    net.trafo.loc[tid, 'tap_set_vm_pu'] = item.usetp

    net.trafo.loc[tid, 'description'] = ' \n '.join(item.desc) if len(item.desc) > 0 else ''


    get_pf_trafo_results(net, item, tid, is_unbalanced)

    # adding switches
    # False if open, True if closed, None if no switch
    create_connection_switches(net, item, 2, 't', (bus1, bus2), (tid, tid))

    # adding tap changer
    if export_controller and item.HasAttribute('ntrcn') and item.HasAttribute('i_cont') \
            and item.ntrcn == 1:
        import pandapower.control as control
        if item.t2ldc == 0:
            logger.debug('tap controller of trafo <%s> at hv' % name)
            side = 'hv'
        else:
            logger.debug('tap controller of trafo <%s> at lv' % name)
            side = 'lv'
        if item.i_cont == 1:
            vm_set_pu = item.usetp
            logger.debug('trafo <%s> has continuous tap controller with vm_set_pu = %.3f, side = %s' %
                         (name, vm_set_pu, side))
            try:
                tap_changer = control.ContinuousTapControl(net, tid, side=side, vm_set_pu=vm_set_pu)
            except BaseException as err:
                logger.error('error while creating continuous tap controller at trafo <%s>' % name)
                logger.error('Error: %s' % err)
                tap_changer = None
            else:
                logger.debug('created discrete tap controller at trafo <%s>' % name)
        else:
            vm_lower_pu = item.usp_low
            vm_upper_pu = item.usp_up
            logger.debug('trafo <%s> has discrete tap controller with '
                         'u_low = %.3f, u_up = %.3f, side = %s' % (name, vm_lower_pu, vm_upper_pu, side))
            try:
                control.DiscreteTapControl(net, tid, side=side, vm_lower_pu=vm_lower_pu, vm_upper_pu=vm_upper_pu)
            except BaseException as err:
                logger.error('error while creating discrete tap controller at trafo <%s>' % name)
                logger.error('Error: %s' % err)
            else:
                logger.debug('created discrete tap controller at trafo <%s>' % name)
    else:
        logger.debug('trafo <%s> has no tap controller' % name)

    add_additional_attributes(item, net, element='trafo', element_id=tid,
                              attr_dict={'e:cpSite.loc_name': 'site', 'for_name': 'equipment', "cimRdfId": "origin_id"})
    if pf_type.itapzdep:
        x_points = (net.trafo.at[tid, "tap_min"], net.trafo.at[tid, "tap_neutral"], net.trafo.at[tid, "tap_max"])
        vk_min, vk_neutral, vk_max = pf_type.uktmn, net.trafo.at[tid, "vk_percent"], pf_type.uktmx
        vkr_min, vkr_neutral, vkr_max = pf_type.ukrtmn, net.trafo.at[tid, "vkr_percent"], pf_type.ukrtmx
        #todo
        #vk0_min, vk0_max = pf_type.uk0tmn, pf_type.uk0tmx
        #vkr0_min, vkr0_max = pf_type.uk0rtmn, pf_type.uk0rtmx
        pp.control.create_trafo_characteristics(net, trafotable="trafo", trafo_index=tid, variable="vk_percent",
                                                x_points=x_points, y_points=(vk_min, vk_neutral, vk_max))
        pp.control.create_trafo_characteristics(net, trafotable="trafo", trafo_index=tid, variable="vkr_percent",
                                                x_points=x_points, y_points=(vkr_min, vkr_neutral, vkr_max))


def get_pf_trafo_results(net, item, tid, is_unbalanced):
    trafo_type = None
    result_variables = None
    if is_unbalanced:
        trafo_type = "res_trafo_3ph"
        result_variables = {
          "pf_i_a_hv_ka": "m:I:bushv:A",
          "pf_i_a_lv_ka": "m:I:buslv:A",
          "pf_i_b_hv_ka": "m:I:bushv:B",
          "pf_i_b_lv_ka": "m:I:buslv:B",
          "pf_i_c_hv_ka": "m:I:bushv:C",
          "pf_i_c_lv_ka": "m:I:buslv:C",
          "pf_i_n_hv_ka": "m:I0x3:bushv",
          "pf_i_n_lv_ka": "m:I0x3:buslv",
          "pf_loading_percent": "c:loading",
        }
    else:
        trafo_type = "res_trafo"
        result_variables = {
               "pf_loading": "c:loading"
               }

    for res_var_pp, res_var_pf in result_variables.items():
        res = np.nan
        if item.HasResults(-1): # -1 for 'c' results (whatever that is...)
            res = ga(item, res_var_pf)
        net[trafo_type].at[tid, res_var_pp] = res


def create_trafo3w(net, item, tap_opt='nntap'):
    # not tested properly yet...
    logger.debug('importing 3W-trafo <%s>' % item.loc_name)
    pf_type = item.typ_id

    try:
        bus1, bus2, bus3 = get_connection_nodes(net, item, 3)
    except IndexError:
        logger.error("Cannot add Trafo3W '%s': not connected" % item.loc_name)
        return

    logger.debug('%s; %s; %s' % (bus1, bus2, bus3))
    if not (net.bus.vn_kv.at[bus1] >= net.bus.vn_kv.at[bus2] >= net.bus.vn_kv.at[bus3]):
        logger.error('trafo <%s>: violated condition of HV > LV!' % item.loc_name)
    # assert net.bus.vn_kv[bus1] > net.bus.vn_kv[bus2] >= net.bus.vn_kv[bus3]
    else:
        logger.debug('bus voltages OK')
    params = {
        'name': item.loc_name,
        'hv_bus': bus1,
        'mv_bus': bus2,
        'lv_bus': bus3,
        'sn_hv_mva': pf_type.strn3_h,
        'sn_mv_mva': pf_type.strn3_m,
        'sn_lv_mva': pf_type.strn3_l,
        'vn_hv_kv': pf_type.utrn3_h,
        'vn_mv_kv': pf_type.utrn3_m,
        'vn_lv_kv': pf_type.utrn3_l,
        'vk_hv_percent': pf_type.uktr3_h,
        'vk_mv_percent': pf_type.uktr3_m,
        'vk_lv_percent': pf_type.uktr3_l,
        'vkr_hv_percent': pf_type.uktrr3_h,
        'vkr_mv_percent': pf_type.uktrr3_m,
        'vkr_lv_percent': pf_type.uktrr3_l,

        'vk0_hv_percent': pf_type.uk0hm,
        'vk0_mv_percent': pf_type.uk0ml,
        'vk0_lv_percent': pf_type.uk0hl,
        'vkr0_hv_percent': pf_type.ur0hm,
        'vkr0_mv_percent': pf_type.ur0ml,
        'vkr0_lv_percent': pf_type.ur0hl,
        'vector_group': re.sub(r'\d+', '', pf_type.vecgrp),

        'pfe_kw': pf_type.pfe,
        'i0_percent': pf_type.curm3,
        'shift_mv_degree': -(pf_type.nt3ag_h - pf_type.nt3ag_m) * 30,
        'shift_lv_degree': -(pf_type.nt3ag_h - pf_type.nt3ag_l) * 30,
        'tap_at_star_point': pf_type.itapos == 0,
        'in_service': not bool(item.outserv)
    }

    if params['tap_at_star_point']:
        logger.warning('%s: implementation for tap changer at star point is not finalized - it can '
                       'lead to wrong results for voltage' % item.loc_name)

    if item.nt3nm != 1:
        logger.warning("trafo3w %s has parallel=%d, this is not implemented. "
                       "Calculation results will be incorrect." % (item.loc_name, item.nt3nm))

    if item.HasAttribute('t:du3tp_h'):
        steps = [pf_type.du3tp_h, pf_type.du3tp_m, pf_type.du3tp_l]
        side = np.nonzero(steps)[0]
        if len(side) > 1:
            logger.warning("pandapower currently doesn't support 3w transformer with"
                           "multiple tap changers")
        elif len(side) == 1:
            ts = ["h", "m", "l"][side[0]]
            # figure out current tap position
            if tap_opt == "nntap":
                tap_pos = ga(item, 'n3tap_' + ts)
                logger.debug("got tap %f from n3tap" % tap_pos)

            elif tap_opt == "c:nntap":
                tap_pos = ga(item, "c:n3tap_" + ts)
                logger.debug("got tap %f from c:n3tap" % tap_pos)
            else:
                raise ValueError('could not read current tap position: tap_opt = %s' % tap_opt)
            params.update({
                'tap_side': ts + 'v',  # hv, mv, lv
                'tap_step_percent': ga(item, 't:du3tp_' + ts),
                'tap_step_degree': ga(item, 't:ph3tr_' + ts),
                'tap_min': ga(item, 't:n3tmn_' + ts),
                'tap_max': ga(item, 't:n3tmx_' + ts),
                'tap_neutral': ga(item, 't:n3tp0_' + ts),
                'tap_pos': tap_pos
            })

    logger.debug('collected params: %s' % params)
    logger.debug('creating trafo3w from parameters')
    tid = pp.create_transformer3w_from_parameters(net, **params)  # type:int

    # adding switches
    # False if open, True if closed, None if no switch
    create_connection_switches(net, item, 3, 't3', (bus1, bus2, bus3), (tid, tid, tid))

    logger.debug('successfully created trafo3w from parameters: %d' % tid)
    # testen
    # net.trafo3w.loc[tid, 'tap_step_degree'] = ga(item, 't:ph3tr_h')

    # adding switches
    # False if open, True if closed, None if no switch
    # Switches for Trafos-3W are not implemented in the load flow!
    # create_connection_switches(net, item, 3, 't3', (bus1, bus2, bus3), (tid, tid, tid))
    # logger.debug('created connection switches for trafo 3w successfully')
    add_additional_attributes(item, net, element='trafo3w', element_id=tid,
                              attr_dict={'cpSite.loc_name': 'site', 'for_name': 'equipment',
                                         'typ_id.loc_name': 'std_type', 'usetp': 'vm_set_pu',
                                         "cimRdfId": "origin_id"})

    # assign loading from power factory results
    if item.HasResults(-1):  # -1 for 'c' results (whatever that is...)
        logger.debug('trafo3w <%s> has results' % item.loc_name)
        loading = ga(item, 'c:loading')
        net.res_trafo3w.at[tid, "pf_loading"] = loading
    else:
        net.res_trafo3w.at[tid, "pf_loading"] = np.nan

    # TODO Implement the tap changer controller for 3-winding transformer

    if pf_type.itapzdep:
        x_points = (net.trafo3w.at[tid, "tap_min"], net.trafo3w.at[tid, "tap_neutral"], net.trafo3w.at[tid, "tap_max"])
        side = net.trafo3w.at[tid, "tap_side"]
        vk_min = ga(pf_type, f"uktr3mn_{side[0]}")
        vk_neutral = net.trafo3w.at[tid, f"vk_{side}_percent"]
        vk_max = ga(pf_type, f"uktr3mx_{side[0]}")
        vkr_min = ga(pf_type, f"uktrr3mn_{side[0]}")
        vkr_neutral = net.trafo3w.at[tid, f"vkr_{side}_percent"]
        vkr_max = ga(pf_type, f"uktrr3mx_{side[0]}")
        # todo zero-sequence parameters (must be implemented in build_branch first)
        pp.control.create_trafo_characteristics(net, trafotable="trafo3w", trafo_index=tid,
                                                variable=f"vk_{side}_percent", x_points=x_points,
                                                y_points=(vk_min, vk_neutral, vk_max))
        pp.control.create_trafo_characteristics(net, trafotable="trafo3w", trafo_index=tid,
                                                variable=f"vkr_{side}_percent", x_points=x_points,
                                                y_points=(vkr_min, vkr_neutral, vkr_max))


def propagate_bus_coords(net, bus1, bus2):
    pass
    # if bus1 in net.bus_geodata.index and bus2 not in net.bus_geodata.index:
    #     net.bus_geodata.loc[bus2, ['x', 'y']] = net.bus_geodata.loc[bus1, ['x', 'y']]
    # elif bus2 in net.bus_geodata.index and bus1 not in net.bus_geodata.index:
    #     net.bus_geodata.loc[bus1, ['x', 'y']] = net.bus_geodata.loc[bus2, ['x', 'y']]


def create_coup(net, item, is_fuse=False):
    switch_types = {"cbk": "CB", "sdc": "LBS", "swt": "LS", "dct": "DS"}
    name = item.loc_name
    logger.debug('>> creating coup <%s>' % name)

    try:
        bus1, bus2 = get_connection_nodes(net, item, 2)
    except IndexError:
        logger.error("Cannot add Coup '%s': not connected" % name)
        return

    propagate_bus_coords(net, bus1, bus2)
    if not item.HasAttribute('isclosed') and not is_fuse:
        logger.error('switch %s does not have the attribute isclosed!' % item)
    switch_is_closed = bool(item.on_off) \
                       and (bool(item.isclosed) if item.HasAttribute('isclosed') else True)
    in_service = not bool(item.outserv) if item.HasAttribute('outserv') else True
    switch_is_closed = switch_is_closed and in_service
    switch_usage = switch_types.get(item.aUsage, 'unknown')

    cd = pp.create_switch(net, name=name, bus=bus1, element=bus2, et='b',
                          closed=switch_is_closed,
                          type=switch_usage)

    add_additional_attributes(item, net, element='switch', element_id=cd,
                              attr_list=['cpSite.loc_name'], attr_dict={"cimRdfId": "origin_id"})

    logger.debug('created switch at index <%d>, closed = %s, usage = %s' %
                 (cd, switch_is_closed, switch_usage))

    net.res_switch.loc[cd, ['pf_closed', 'pf_in_service']] = bool(item.on_off) and (
        bool(item.isclosed) if item.HasAttribute('isclosed') else True), in_service


# # false approach, completely irrelevant
# def create_switch(net, item):
#     switch_types = {"cbk": "CB", "sdc": "LBS", "swt": "LS", "dct": "DS"}
#     name = ga(item, 'loc_name')
#     logger.debug('>> creating switch <%s>' % name)
#
#     pf_bus1 = item.GetNode(0)
#     pf_bus2 = item.GetNode(1)
#
#     # here: implement situation if line not connected
#     if pf_bus1 is None or pf_bus2 is None:
#         logger.error("Cannot add Switch '%s': not connected" % name)
#         return
#
#     bus1 = find_bus_index_in_net(pf_bus1, net)
#     bus2 = find_bus_index_in_net(pf_bus2, net)
#     logger.debug('switch %s connects buses <%d> and <%d>' % (name, bus1, bus2))
#
#     switch_is_closed = bool(ga(item, 'on_off'))
#     switch_usage = switch_types[ga(item, 'aUsage')]
#
#     cd = pp.create_switch(net, name=name, bus=bus1, element=bus2, et='b',
# closed=switch_is_closed, type=switch_usage)
#     logger.debug('created switch at index <%d>, closed = %s, usage = %s' % (cd,
# switch_is_closed, switch_usage))


def create_shunt(net, item):
    try:
        bus = get_connection_nodes(net, item, 1)
    except IndexError:
        logger.error("Cannot add Shunt '%s': not connected" % item.loc_name)
        return

    multiplier = get_power_multiplier(item, 'Qact')
    params = {
        'name': item.loc_name,
        'bus': bus,
        'in_service': monopolar_in_service(item),
        'vn_kv': item.ushnm,
        'q_mvar': item.Qact * multiplier
    }

    if item.shtype == 1:
        # Shunt is an R-L element
        params['q_mvar'] = item.qrean * multiplier
        p_mw = (item.ushnm ** 2 * item.rrea / (item.rrea ** 2 + item.xrea ** 2)) * multiplier
        sid = pp.create_shunt(net, p_mw=p_mw, **params)
    elif item.shtype == 2:
        # Shunt is a capacitor bank
        loss_factor = item.tandc
        sid = pp.create_shunt_as_capacitor(net, loss_factor=loss_factor, **params)
    else:
        # Shunt is an element of R-L-C (0), R-L (1), R-L-C, Rp (3), R-L-C1-C2, Rp (4)
        logger.warning('Importing of shunt elements that represent anything but capacitor banks '
                       'is not implemented correctly, the results will be inaccurate')
        if item.HasResults(0):
            p_mw = ga(item, 'm:P:bus1') * multiplier
        elif item.HasAttribute('c:PRp'):
            p_mw = ga(item, 'c:PRp') / 1e3 + ga(item, 'c:PL') / 1e3
        else:
            logger.warning("Shunt element %s is not implemented, p_mw is set to 0, results will be incorrect!" % item.loc_name)
            p_mw = 0
        sid = pp.create_shunt(net, p_mw=p_mw, **params)

    add_additional_attributes(item, net, element='shunt', element_id=sid,
                              attr_list=['cpSite.loc_name'], attr_dict={"cimRdfId": "origin_id"})

    if item.HasResults(0):
        net.res_shunt.loc[sid, 'pf_p'] = ga(item, 'm:P:bus1') * multiplier
        net.res_shunt.loc[sid, 'pf_q'] = ga(item, 'm:Q:bus1') * multiplier
    else:
        net.res_shunt.loc[sid, 'pf_p'] = np.nan
        net.res_shunt.loc[sid, 'pf_q'] = np.nan


def _add_shunt_to_impedance_bus(net, item, bus):
    pp.create_shunt(net, bus, -item.bi_pu*net.sn_mva, p_mw=-item.gi_pu*net.sn_mva)


def create_zpu(net, item):
    try:
        (bus1, bus2) = get_connection_nodes(net, item, 2)
    except IndexError:
        logger.error("Cannot add ZPU '%s': not connected" % item.loc_name)
        return
    logger.debug('bus1 = %d, bus2 = %d' % (bus1, bus2))

    # net, from_bus, to_bus, r_pu, x_pu, sn_Mva, name=None, in_service=True, index=None
    params = {
        'name': item.loc_name,
        'from_bus': bus1,
        'to_bus': bus2,
        'rft_pu': item.r_pu,
        'xft_pu': item.x_pu,
        'rtf_pu': item.r_pu_ji,
        'xtf_pu': item.x_pu_ji,
        'sn_mva': item.Sn,
        'in_service': not bool(item.outserv)
    }

    logger.debug('params = %s' % params)
    xid = pp.create_impedance(net, **params)
    add_additional_attributes(item, net, element='impedance', element_id=xid, attr_list=["cpSite.loc_name"],
                              attr_dict={"cimRdfId": "origin_id"})

    # create shunts at the buses connected to the impedance
    if ~np.isclose(item.gi_pu, 0) or ~np.isclose(item.bi_pu, 0):
        _add_shunt_to_impedance_bus(net, item, bus1)
    if ~np.isclose(item.gj_pu, 0) or ~np.isclose(item.bj_pu, 0):
        _add_shunt_to_impedance_bus(net, item, bus2)


def create_vac(net, item):
    """
    not tested yet

    """
    try:
        bus = get_connection_nodes(net, item, 1)
    except IndexError:
        logger.error("Cannot add VAC '%s': not connected" % item.loc_name)
        return

    params = {
        'name': item.loc_name,
        'bus': bus,
        'ps_mw': item.Pload - item.Pgen,
        'qs_mvar': item.Qload - item.Qgen,
        'pz_mw': item.Pzload,
        'qz_mvar': item.Qzload,
        'in_service': not bool(item.outserv)
    }

    if item.itype == 3:
        # extended ward
        params.update({
            'r_ohm': item.Rext,
            'x_ohm': item.Xext,
            'vm_pu': item.usetp
        })

        if params['x_ohm'] == 0:
            params['x_ohm'] = 1e-6
            logger.warning("Element %s has x_ohm == 0, setting to 1e-6. Check impedance of the "
                           "created xward" % item.loc_name)

        xid = pp.create_xward(net, **params)
        elm = 'xward'

    elif item.itype == 2:
        # ward
        xid = pp.create_ward(net, **params)
        elm = 'ward'

    elif item.itype == 0:
        # voltage source
        params.update({
            'vm_pu': item.usetp,
            'va_degree': item.phisetp,
        })
        xid = pp.create_ext_grid(net, **params)
        elm = 'ext_grid'
    else:
        raise NotImplementedError(
            'Could not import %s: element type <%d> for AC Voltage Source not implemented' % (
                params['name'], item.itype))

    if item.HasResults(0):  # -1 for 'c' results (whatever that is...)
        net['res_%s' % elm].at[xid, "pf_p"] = -ga(item, 'm:P:bus1')
        net['res_%s' % elm].at[xid, "pf_q"] = -ga(item, 'm:Q:bus1')
    else:
        net['res_%s' % elm].at[xid, "pf_p"] = np.nan
        net['res_%s' % elm].at[xid, "pf_q"] = np.nan

    add_additional_attributes(item, net, element=elm, element_id=xid, attr_list=["cpSite.loc_name"],
                              attr_dict={"cimRdfId": "origin_id"})

    logger.debug('added pf_p and pf_q to {} {}: {}'.format(elm, xid, net['res_' + elm].loc[
        xid, ["pf_p", 'pf_q']].values))


def create_sind(net, item):
    # series reactor is modelled as per-unit impedance, values in Ohm are calculated into values in
    # per unit at creation
    try:
        (bus1, bus2) = get_connection_nodes(net, item, 2)
    except IndexError:
        logger.error("Cannot add Sind '%s': not connected" % item.loc_name)
        return

    sind = pp.create_series_reactor_as_impedance(net, from_bus=bus1, to_bus=bus2, r_ohm=item.rrea,
                                                 x_ohm=item.xrea, sn_mva=item.Sn,
                                                 name=item.loc_name,
                                                 in_service=not bool(item.outserv))

    logger.debug('created series reactor %s as per unit impedance at index %d' %
                 (net.impedance.at[sind, 'name'], sind))


def split_line_at_length(net, line, length_pos):
    bus1, bus2 = net.line.loc[line, ['from_bus', 'to_bus']]
    if length_pos == net.line.at[line, 'length_km']:
        bus = bus2
    elif length_pos == 0:
        bus = bus1
    else:
        bus_name = "%s (Muff %u)" % (net.line.at[line, 'name'], length_pos)
        vn_kv = net.bus.at[bus1, "vn_kv"]
        zone = net.bus.at[bus1, "zone"]

        bus = pp.create_bus(net, name=bus_name, type='ls', vn_kv=vn_kv, zone=zone)

        net.line.at[line, 'to_bus'] = bus
        old_length = net.line.at[line, 'length_km']
        new_length = old_length - length_pos
        net.line.at[line, 'length_km'] = length_pos
        std_type = net.line.at[line, 'std_type']
        name = net.line.at[line, 'name']

        new_line = pp.create_line(net, from_bus=bus, to_bus=bus2, length_km=new_length,
                                  std_type=std_type, name=name, df=net.line.at[line, 'df'],
                                  parallel=net.line.at[line, 'parallel'],
                                  in_service=net.line.at[line, 'in_service'])

        if 'max_loading_percent' in net.line.columns:
            net.line.loc[new_line, 'max_loading_percent'] = net.line.at[line, 'max_loading_percent']

        if 'line_geodata' in net.keys() and line in net.line_geodata.index.values:
            coords = net.line_geodata.at[line, 'coords']

            scaling_factor = old_length / calc_len_coords(coords)
            sec_coords_a = get_section_coords(coords, sec_len=length_pos, start_len=0.,
                                              scale_factor=scaling_factor)

            sec_coords_b = get_section_coords(coords, sec_len=new_length, start_len=length_pos,
                                              scale_factor=scaling_factor)

            net.line_geodata.loc[line, 'coords'] = sec_coords_a
            net.line_geodata.loc[new_line, 'coords'] = sec_coords_b

            net.bus_geodata.loc[bus, ['x', 'y']] = sec_coords_b[0]

    return bus


def get_lodlvp_length_pos(line_item, lod_item):
    sections = line_item.GetContents('*.ElmLnesec')
    if len(sections) > 0:
        sections.sort(lambda x: x.index)
        sections_start = [s.rellen for s in sections]
        sections_end = [s.rellen + s.dline for s in sections]
    else:
        sections_start = [0]
        sections_end = [line_item.dline]

    loads = line_item.GetContents('*.ElmLodlvp')
    if len(loads) > 0:
        loads.sort(lambda x: x.rellen)
        loads_start = [l.rellen for l in loads]
    else:
        loads_start = [0]

    pos_sec_idx = bisect.bisect(sections_end, lod_item.rellen)
    pos_load_idx = bisect.bisect(loads_start, lod_item.rellen)

    pos = max(sections_end[pos_sec_idx - 1], loads_start[pos_load_idx - 1])

    return lod_item.rellen - pos


def get_next_line(net, line):
    name = net.line.at[line, 'name']
    to_bus = net.line.at[line, 'to_bus']
    next_line = net.line.loc[(net.line.name == name) & (net.line.from_bus == to_bus)].index

    return next_line


# def get_section_for_lodlvp(net, line_item, lod_item):
#     linepos = lod_item.rellen
#
#     cum_len = 0
#     while cum_len < linepos:
#         line =


# for ElmLodlvp - splits line at the partial load, creates new bus, sets up coordinates
def split_line(net, line_idx, pos_at_line, line_item):
    tol = 1e-6
    line_length = net.line.at[line_idx, 'length_km']
    logger.debug("line length: %.3f" % line_length)
    if pos_at_line < tol:
        bus_i = net.line.at[line_idx, 'from_bus']
        logger.debug('bus_i: %s' % bus_i)
        net.bus.at[bus_i, 'type'] = 'n'
        return bus_i
    elif abs(pos_at_line - line_length) < tol:
        bus_j = net.line.at[line_idx, 'to_bus']
        logger.debug('bus_j: %s' % bus_j)
        net.bus.at[bus_j, 'type'] = 'n'
        return bus_j
    elif (pos_at_line - line_length) > tol:
        raise ValueError(
            'Position at line is higher than the line length itself! Line length: %.7f, position at line: %.7f (line: \n%s)' % (
                # line_length, pos_at_line, line_item.loc_name))
                line_length, pos_at_line, net.line.loc[line_dict[line_item]]))
    else:
        logger.debug('getting split position')
        name = net.line.at[line_idx, 'name']
        bus_i = net.line.at[line_idx, 'from_bus']
        bus_j = net.line.at[line_idx, 'to_bus']
        u = net.bus.at[bus_i, 'vn_kv']

        new_bus = pp.create_bus(net, name="Partial Load", vn_kv=u, type='n')
        logger.debug("created new split bus %s" % new_bus)

        line_type = net.line.at[line_idx, 'std_type']
        len_a = pos_at_line
        len_b = line_length - pos_at_line

        net.line.at[line_idx, 'length_km'] = len_a

        # connect the existing line to the new bus
        net.line.at[line_idx, 'to_bus'] = new_bus

        new_line = pp.create_line(net, new_bus, bus_j, len_b, line_type, name=name)
        # change the connection of the bus-line switch to the new line
        sw = net.switch.query("et=='l' & bus==@bus_j & element==@line_idx").index
        if len(sw) > 0:
            if len(sw) > 1:
                raise RuntimeError(
                    'found too many switches to fix for line %s: \n%s' % (
                        line_item, net.switch.loc[sw]))
            net.switch.loc[sw, 'element'] = new_line

        line_dict[line_item].append(new_line)

        net.line.at[new_line, 'section'] = "%s_1" % net.line.at[line_idx, 'section']
        net.line.at[new_line, 'order'] = net.line.at[line_idx, 'order'] + 1
        net.res_line.at[new_line, 'pf_loading'] = net.res_line.at[line_idx, 'pf_loading']

        if line_idx in net.line_geodata.index.values:
            logger.debug('setting new coords')
            set_new_coords(net, new_bus, line_idx, new_line, line_length, pos_at_line)

        return new_bus


def calc_segment_length(x1, y1, x2, y2):
    delta_x = float(x2) - float(x1)
    delta_y = float(y2) - float(y1)
    return (delta_x ** 2 + delta_y ** 2) ** 0.5


def get_scale_factor(length_line, coords):
    if any(coords) is np.nan:
        return np.nan
    temp_len = 0
    num_coords = len(coords)
    for i in range(num_coords - 1):
        x1 = float(coords[i][0])
        y1 = float(coords[i][1])

        x2 = float(coords[i + 1][0])
        y2 = float(coords[i + 1][1])
        temp_len += calc_segment_length(x1, y1, x2, y2)
    return temp_len / length_line if length_line != 0 else 0


def break_coords_sections(coords, section_length, scale_factor_length):
    section_length *= scale_factor_length
    # breaks coordinates into 2 parts (chops the line section away)
    if any(coords) is np.nan:
        return [[np.nan, np.nan]], [[np.nan, np.nan]]

    num_coords = len(coords)
    if num_coords < 2:
        return [[np.nan, np.nan]], [[np.nan, np.nan]]
    # define scale

    sum_len, delta_len, x1, y1, x2, y2 = tuple([0] * 6)
    i = 0
    for i in range(num_coords - 1):
        x1 = float(coords[i][0])
        y1 = float(coords[i][1])

        x2 = float(coords[i + 1][0])
        y2 = float(coords[i + 1][1])

        delta_len = calc_segment_length(x1, y1, x2, y2)
        sum_len += delta_len
        if sum_len >= section_length:
            break

    a = section_length - (sum_len - delta_len)
    b = sum_len - section_length
    x0 = a * x2 / delta_len + b * x1 / delta_len
    y0 = a * y2 / delta_len + b * y1 / delta_len

    section_coords = coords[0:i + 1] + [[x0, y0]]
    new_coords = [[x0, y0]] + coords[(i + 1)::]
    return section_coords, new_coords


# set up new coordinates for line sections that are split by the new bus of the ElmLodlvp
def set_new_coords(net, bus_id, line_idx, new_line_idx, line_length, pos_at_line):
    line_coords = net.line_geodata.at[line_idx, 'coords']
    logger.debug('got coords for line %s' % line_idx)

    scale_factor_length = get_scale_factor(line_length, line_coords)
    section_coords, new_coords = break_coords_sections(line_coords, pos_at_line,
                                                       scale_factor_length)

    logger.debug('calculated new coords: %s, %s ' % (section_coords, new_coords))

    net.line_geodata.at[line_idx, 'coords'] = section_coords
    net.line_geodata.at[new_line_idx, 'coords'] = new_coords

    net.bus_geodata.at[bus_id, 'x'] = new_coords[0][0]
    net.bus_geodata.at[bus_id, 'y'] = new_coords[0][1]


# gather info about ElmLodlvp in a dict
def get_lvp_for_lines(dict_net):
    logger.debug(dict_net['lvp_params'])

    def calc_p_q(lvp, lvp_params):
        lvp_type = lvp.typ_id

        # if lvp_type is not None:
        #     cos_fix = lvp.coslini_a
        #
        #     s_var = lvp.cSav
        #     cos_var = lvp.ccosphi
        # else:
        #     cos_fix = lvp_params['cosfix']
        #
        #     s_var = lvp.cSav
        #     cos_var = lvp_params['cosvar']

        # s_fix = lvp_params['Sfix'] * lvp.NrCust + lvp.slini_a

        # p_fix = s_fix * cos_fix
        # q_fix = s_fix * np.sin(np.arccos(cos_fix))
        #
        # p_var = s_var * cos_var
        # q_var = s_var * np.sin(np.arccos(cos_var))

        if lvp_type is not None:
            s_fix_global = 0
            cos_fix_global = lvp.coslini_a
        else:
            s_fix_global = lvp_params['Sfix'] * lvp.NrCust
            cos_fix_global = lvp_params['cosfix']

        s_fix_local = lvp.slini_a
        cos_fix_local = lvp.coslini_a

        s_var_local = lvp.cSav
        cos_var_local = lvp.ccosphi

        p_fix = s_fix_local * cos_fix_local + s_fix_global * cos_fix_global
        q_fix = s_fix_local * np.sin(np.arccos(cos_fix_local)) + s_fix_global * np.sin(
            np.arccos(cos_fix_global))

        p_var = s_var_local * cos_var_local
        q_var = s_var_local * np.sin(np.arccos(cos_var_local))

        scale_p_night = lvp_params['scPnight'] / 100
        p_night = lvp.pnight_a * scale_p_night

        # logger.debug(
        #     f"load: {lvp.loc_name}, s_fix: {s_fix}, cos_fix: {cos_fix}, s_var: {s_var}, cos_var: {cos_var}, "
        #     f"p_night: {p_night}, scale_p_night: {scale_p_night}")

        p = p_fix + p_var + p_night
        q = q_fix + q_var

        return p, q

    line_items = dict_net['ElmLne']
    # choose ElmLodlvp that are part of lines
    lvp_items = [lvp for lvp in dict_net['ElmLodlvp'] if lvp.fold_id.GetClassName() == 'ElmLne']
    logger.debug(lvp_items)

    lvp_dict = {}
    for line in line_items:
        temp_loads = [lvp for lvp in lvp_items if lvp.fold_id == line]
        logger.debug('line: %s , loads: %s' % (line, temp_loads))

        if len(temp_loads) == 0:
            continue

        # {'line': [(load.ElmLodlvp, position_at_line, (p_mw, q_mvar))]}
        lvp_dict[line] = [(lvp, lvp.lneposkm, calc_p_q(lvp, dict_net['lvp_params']))
                          for lvp in temp_loads]

        lvp_dict[line].sort(key=lambda tup: tup[1])
    return lvp_dict


# find position of ElmLodlvp at the section
def get_pos_at_sec(net, lvp_dict, line_item, load_item):
    val = lvp_dict[line_item]
    pos_at_line = 0

    for load_item_for, pos_at_line_for, _ in val:
        if load_item_for == load_item:
            pos_at_line = pos_at_line_for
            break

    # line_sections = net.line[net.line.name == line_item.loc_name].sort_values(by='order')
    line_sections = net.line.loc[line_dict[line_item]].sort_values(by='order')
    logger.debug('line sections:\n%s' % line_sections)

    tot_length = 0
    sec_length = 0
    section = 1

    for section in line_sections.index:
        sec_length = line_sections.at[section, 'length_km']
        tot_length += sec_length
        logger.debug(
            "section: %s, sec_length: %s, tot_length: %s" % (section, sec_length, tot_length))
        if tot_length >= pos_at_line:
            break
    else:
        logger.warning(
            'possibly wrong section found: %s of %s for %s (tot_length=%s, pas_at_line=%s)' % (
                section, line_item, load_item, tot_length, pos_at_line))

    # section_name = line_sections[(line_sections.index == section)]['Name'].values[0]
    pos_at_sec = sec_length + pos_at_line - tot_length

    return section, pos_at_sec


# write order of sections
def write_line_order(net):
    net.line['order'] = ''
    line_names = net.line.name.unique()

    for n in line_names:
        k = 1000
        for i, row in net.line[net.line.name == n].iterrows():
            net.line.at[i, 'order'] = k
            k += 1000


# split all lines and create loads in place of ElmLodlvp
def split_all_lines(net, lvp_dict):
    write_line_order(net)
    # for idx in net.line.index:
    for line_item, val in lvp_dict.items():
        logger.debug(line_item)
        # for load_idx, pos_at_line, _, _ in val:
        #     section, pos_at_sec = get_pos_at_sec(net, net_dgs, lvp_dict, line, load_idx)
        #     pas[load_idx] = {'section':section, 'pos': pos_at_sec}
        # print('line: %s, val: %s' % (line, val))
        # val = [(92, 1, 0.025, 0.1), (91, 2, 0.031, 0.2), (90, 2, 0.032, 0.3)]
        for load_item, pos_at_line, (p, q) in val:
            logger.debug(load_item)
            ## calculate at once and then read from dict - not good approach! don't do it
            # section, pos_at_sec = get_pos_at_sec(net, net_dgs, lvp_dict, line, load_idx)
            # section = pas[load_idx]['section']
            # pos_at_sec = pas[load_idx]['pos']
            section, pos_at_sec = get_pos_at_sec(net, lvp_dict, line_item, load_item)
            logger.debug("section: %s, pos_at_sec: %s" % (section, pos_at_sec))
            logger.debug("%s" % net.line.at[section, 'in_service'])
            if not net.line.at[section, 'in_service']:
                print('line %s skipped because it is not in service' % net.line.at[section, 'name'])
                continue
            new_bus = split_line(net, section, pos_at_sec, line_item=line_item)
            logger.debug("new_bus: %s" % new_bus)
            net.bus.at[new_bus, 'description'] = 'Partial Line LV-Load %.2f kW' % p

            if p >= 0 or True:
                # TODO: set const_i_percent to 100 after the pandapower bug is fixed
                new_load = pp.create_load(net, new_bus, name=load_item.loc_name, p_mw=p, q_mvar=q,
                                          const_i_percent=0)
                logger.debug('created load %s' % new_load)
                net.res_load.at[new_load, 'pf_p'] = p
                net.res_load.at[new_load, 'pf_q'] = q
            else:
                # const I not implemented for sgen...
                new_load = pp.create_sgen(net, new_bus, name=load_item.loc_name, p_mw=p, q_mvar=q)
                logger.debug('created sgen %s' % new_load)
                net.res_sgen.at[new_load, 'pf_p'] = p
                net.res_sgen.at[new_load, 'pf_q'] = q


def remove_folder_of_std_types(net):
    """
    Removes the folder name from all standard types that do not have duplicates, or where
    duplicates have the same parameters
    """
    for element in ["line", "trafo", "trafo3w"]:
        std_types = pp.available_std_types(net, element=element).index
        reduced_std_types = {name.split("\\")[-1] for name in std_types}
        for std_type in reduced_std_types:
            all_types = [st for st in std_types if st.split('\\')[-1] == std_type]
            if len(all_types) > 1:
                types_equal = [
                    pp.load_std_type(net, type1, element) == pp.load_std_type(net, type2, element)
                    for type1, type2 in combinations(all_types, 2)]
                if not all(types_equal):
                    continue
            for st in all_types:
                net.std_types[element][std_type] = net.std_types[element].pop(st)
                net[element].std_type = net[element].std_type.replace(st, std_type)
