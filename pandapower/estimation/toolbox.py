# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandas as pd

import pandapower as pp
from pandapower.pd2ppc import _pd2ppc
from pandapower.run import rundcpp

def _get_bus_ppc_mapping(net):
    try:
        rundcpp(net)
    except:
        pass

    bus_with_elements = set(net.load.bus).union(set(net.sgen.bus)).union(
                    set(net.shunt.bus)).union(set(net.gen.bus)).union(
                    set(net.ext_grid.bus)).union(set(net.ward.bus)).union(
                    set(net.xward.bus))

    bus_ppc = pd.DataFrame(data=net._pd2ppc_lookups['bus'], columns=["bus_ppc"])
    bus_ppc['bus_with_elements'] = bus_ppc.index.isin(bus_with_elements).astype(int)   
    ppc_bus_with_elements = bus_ppc.groupby('bus_ppc')['bus_with_elements'].sum()
    bus_ppc.loc[:, 'elements_in_cluster'] = ppc_bus_with_elements[bus_ppc['bus_ppc'].values].values 
    return bus_ppc

def set_bb_switch_impedance(net, z_ohm=0.1, prevent_fusing_bus_with_elements = False):
    """
     Assuming a substation with multiple buses connected to each other with bb switch
     if multiple of them have an element (load, sgen usw.) then it caused problem on 
     e.g. State Estimation, since they will be automatically fused in ppc
     A possiblity would be to properly seperate the substation to multiple clusters (ppc) bus
     according to the number of buses with elements, in order to estimate the p, q on all 
     the busses. 
     These method implemented a forward and backward substitution:
     1. Find the mapping of pd2ppc bus, and identify the ppc bus, which contains multiple bus with elements
     2. Skip the first bus with elements with the same ppc bus, and finding all the switch connected to the
       rest pandapower bus and set impedance
     3. Update the pd2ppc lookup
     4. if it caused some bus without elements to detach from any of the bus with connected element, then the 
       switch caused it will be reset to fuse mode
     5. Iterate this process until all buses with elements are seperated to their own ppc bus, while no bus is 
       detached in order to get the voltage on bus right
    :param net: pandapower net
    :param r_ohm: float r_ohm to be setted for the switch
    :param prevent_fusing_bus_with_elements: Bool deactivate set switch impedance to all switches, actuvat
        to set on the selected switch
    :return: None
    """
    if 'z_ohm' in net.switch:
        net.switch['z_ohm_ori'] = net.switch['z_ohm']
    if not prevent_fusing_bus_with_elements:
        net.switch.loc[:, 'z_ohm'] = z_ohm
    else:
        lookup = _get_bus_ppc_mapping(net)
        for _ in range(lookup.elements_in_cluster.max()-1):     
            bus_to_be_handled = lookup[((lookup['elements_in_cluster']>=2)&\
                                        lookup['bus_with_elements'])]
            bus_to_be_handled = bus_to_be_handled[bus_to_be_handled['bus_ppc'].duplicated(keep='first')].index
            imp_switch_sel = (net.switch.et=='b')&(net.switch.closed)&\
                (net.switch.bus.isin(bus_to_be_handled)|net.switch.element.isin(bus_to_be_handled))
            net.switch.loc[imp_switch_sel, 'z_ohm'] = z_ohm
            
            #check if fused buses were isolated by the switching type change
            lookup_discon = _get_bus_ppc_mapping(net)
            imp_switch = net.switch.loc[imp_switch_sel, :]
            detached_bus = lookup_discon.loc[lookup_discon.elements_in_cluster==0, :].index
            # Find the cause of isolated bus, if caused by the 
            switch_to_be_fused = (imp_switch.bus.isin(bus_to_be_handled)&imp_switch.element.isin(detached_bus))|\
                                 (imp_switch.bus.isin(detached_bus)&imp_switch.element.isin(bus_to_be_handled))
            if not switch_to_be_fused.values.any():
                return
            net.switch.loc[imp_switch[switch_to_be_fused].index, 'z_ohm'] = 0  
            lookup = _get_bus_ppc_mapping(net)
            if lookup.elements_in_cluster.max() == 1:
                return


def reset_bb_switch_impedance(net):
    """
    Reset the z_ohm of the switch to its original state, undo the operation by set_bb_switch_impedance 
    :param net: pandapower net
    :return: None
    """
    if "z_ohm_ori" in net.switch:
        net.switch["z_ohm"] = net.switch["z_ohm_ori"] 
        net.switch.drop("z_ohm_ori", axis=1, inplace=True)
        
        
def add_virtual_meas_from_loadflow(net, v_std_dev=0.001, p_std_dev=0.03, q_std_dev=0.03):
    bus_meas_types = {'v': 'vm_pu', 'p': 'p_mw', 'q': 'q_mvar'}
    branch_meas_type = {'line':{'side': ('from', 'to'), 
                                'meas_type': ('p_mw', 'q_mvar')},
                        'trafo':{'side': ('hv', 'lv'), 
                                 'meas_type': ('p_mw', 'q_mvar')},
                        'trafo3w': {'side': ('hv', 'mv', 'lv'), 
                                    'meas_type': ('p_mw', 'q_mvar')}} 
    for bus_ix, bus_res in net.res_bus.iterrows():
        for meas_type in bus_meas_types.keys():
            meas_value = float(bus_res[bus_meas_types[meas_type]])
            if meas_type in ('p', 'q'):
                meas_value *= -1
                pp.create_measurement(net, meas_type=meas_type, element_type='bus', element=bus_ix,
                                      value=meas_value, std_dev=1)
            else:
                pp.create_measurement(net, meas_type=meas_type, element_type='bus', element=bus_ix,
                                      value=meas_value, std_dev=1)
            
    for br_type in branch_meas_type.keys():
        if not net['res_'+br_type].empty:
            for br_ix, br_res in net['res_'+br_type].iterrows():
                for side, meas_type in zip(branch_meas_type[br_type]['side'],
                                           branch_meas_type[br_type]['meas_type']):
                    pp.create_measurement(net, meas_type=meas_type[0], element_type=br_type, 
                                          element=br_ix, side=side,
                                          value=br_res[meas_type[0]+'_'+side+meas_type[1:]], std_dev=1)

    add_virtual_meas_error(net, v_std_dev, p_std_dev, q_std_dev)

                   
def add_virtual_meas_error(net, v_std_dev, p_std_dev, q_std_dev):
    assert not net.measurement.empty
    
    r = np.random.rand(net.measurement.shape[0]) * 2 - 1 # random error in range from -1, 1
    p_meas_mask = net.measurement.measurement_type=="p"
    q_meas_mask = net.measurement.measurement_type=="q"    
    v_meas_mask = net.measurement.measurement_type=="v" 
    
    net.measurement.loc[p_meas_mask, 'value'] += r[p_meas_mask.values] * p_std_dev
    net.measurement.loc[p_meas_mask, 'std_dev'] = p_std_dev
    net.measurement.loc[q_meas_mask, 'value'] += r[q_meas_mask.values] * q_std_dev
    net.measurement.loc[q_meas_mask, 'std_dev'] = q_std_dev
    net.measurement.loc[v_meas_mask, 'value'] += r[v_meas_mask.values] * v_std_dev
    net.measurement.loc[v_meas_mask, 'std_dev'] = v_std_dev


if __name__ == "__main__":
    import pandapower.networks as pn
    from pandapower.estimation.state_estimation import estimate
    
    net = pn.case57()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net)
    estimate(net, algorithm="opt", init="slack", estimator='wls')
#    estimate(net)
    
    assert np.isclose(net.res_bus_est.vm_pu, net.res_bus_power_flow.vm_pu, atol=0.01).all()
    
    
    