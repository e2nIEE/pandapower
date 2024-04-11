import pandapower as pp
from pandapower.converter.sincal.pp2sincal.util.initialization import initialize_calculation

def compare_results(net, net_pp, sim):
    '''
    Comparison between pandapower and sincal loadflow calculation.

    :param net: Sincal electrical Database Object
    :type net: object
    :param net_pp: The pandapower network that shall be converted
    :type net_pp: pandapowerNet
    :param sim: Simulation Object
    :type sim: object
    :return: voltage/angle -difference
    :rtype: Tuple
    '''
    initialize_calculation(net, sim)
    pp.set_user_pf_options(net_pp, trafo_model='pi')
    pp.runpp(net_pp, calculate_voltage_angles=True)
    diff_u = []
    diff_deg = []

    for i in net_pp.bus.index:
        bus = net.GetCommonObject("Node", net_pp.bus.at[i, 'Sinc_Name'])
        res = net.GetCommonObject("LFNodeResult", bus.GetValue('Node_ID'))
        res_sincal_u = res.GetValue('U_Un') / 100
        res_sincal_deg = res.GetValue('phi_rot')
        res_pp_u = net_pp.res_bus.at[i, 'vm_pu']
        res_pp_deg = net_pp.res_bus.at[i, 'va_degree']
        diff_u += [res_pp_u - res_sincal_u]
        diff_deg += [res_pp_deg - res_sincal_deg]
    return diff_u, diff_deg
