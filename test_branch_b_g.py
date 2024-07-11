import pandapower as pp
import numpy as np

def test_3w_net():
    net = pp.create_empty_network()
    buses=pp.create_buses(net, 3, [110, 20, 10])
    pp.create_ext_grid(net, buses[0])
    pp.create_transformer3w(net, buses[0], buses[1], buses[2], "63/25/38 MVA 110/20/10 kV")
    pp.create_load(net, buses[1], p_mw=20)
    pp.create_load(net, buses[2], p_mw=30, q_mvar=5)
    return net

def test_line_net():
    net = pp.create_empty_network()
    buses=pp.create_buses(net, 2, [110, 110])
    pp.create_ext_grid(net, buses[0])
    pp.create_line(net, buses[0], buses[1], std_type="243-AL1/39-ST1A 110.0", length_km=10)
    pp.create_load(net, buses[1], p_mw=20, q_mvar=10)
    return net

def test_2w_net():
    net = pp.create_empty_network()
    buses=pp.create_buses(net, 2, [110, 20])
    pp.create_ext_grid(net, buses[0])
    pp.create_transformer(net, buses[0], buses[1], "40 MVA 110/20 kV")
    pp.create_load(net, buses[1], p_mw=20)
    return net

if __name__ == "__main__":
    net = test_3w_net()
    # net = test_line_net()
    # net = test_2w_net()
    pp.runpp(net)
    
    # tr3w=net.trafo3w

    print(net.res_bus)
    # print(net.res_trafo3w)

    ppc=net._ppc
    
    from pandapower.converter.pypower.to_ppc import to_ppc
    ppc_test=to_ppc(net)
    
    
    


