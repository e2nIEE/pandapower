from pandapower.create import (
    create_empty_network,
    create_bus,
    create_ext_grid,
    create_line_from_parameters,
    create_transformer_from_parameters,
    create_gen,
    create_load
)

def facts_case_study_grid():
    net = create_empty_network()

    b1 = create_bus(net, name="B1", vn_kv=18)
    b2 = create_bus(net, name="B2", vn_kv=16.5)
    b3 = create_bus(net, name="B3", vn_kv=230)
    b4 = create_bus(net, name="B4", vn_kv=230)
    b5 = create_bus(net, name="B5", vn_kv=230)
    b6 = create_bus(net, name="B6", vn_kv=230)
    b7 = create_bus(net, name="B7", vn_kv=230)
    b8 = create_bus(net, name="B8", vn_kv=230)

    create_ext_grid(net, bus=b1, vm_pu=1, va_degree=0)

    create_line_from_parameters(net, name="L1", from_bus=b3, to_bus=b4, length_km=30, r_ohm_per_km=0.049,
                                x_ohm_per_km=0.136, g_us_per_km=0, c_nf_per_km=142, max_i_ka=1.5)
    create_line_from_parameters(net, name="L2", from_bus=b3, to_bus=b4, length_km=30, r_ohm_per_km=0.049,
                                x_ohm_per_km=0.136, g_us_per_km=0, c_nf_per_km=142, max_i_ka=1.5)
    create_line_from_parameters(net, name="L3", from_bus=b4, to_bus=b5, length_km=100, r_ohm_per_km=0.081,
                                x_ohm_per_km=0.312, g_us_per_km=0, c_nf_per_km=11, max_i_ka=1.5)
    create_line_from_parameters(net, name="L4", from_bus=b4, to_bus=b6, length_km=100, r_ohm_per_km=0.081,
                                x_ohm_per_km=0.312, g_us_per_km=0, c_nf_per_km=11, max_i_ka=1.5)
    create_line_from_parameters(net, name="L5", from_bus=b5, to_bus=b7, length_km=220, r_ohm_per_km=0.081,
                                x_ohm_per_km=0.312, g_us_per_km=0, c_nf_per_km=11, max_i_ka=1.5)
    create_line_from_parameters(net, name="L6", from_bus=b6, to_bus=b8, length_km=140, r_ohm_per_km=0.081,
                                x_ohm_per_km=0.312, g_us_per_km=0, c_nf_per_km=11, max_i_ka=1.5)
    create_line_from_parameters(net, name="L7", from_bus=b5, to_bus=b6, length_km=180, r_ohm_per_km=0.081,
                                x_ohm_per_km=0.312, g_us_per_km=0, c_nf_per_km=11, max_i_ka=1.5)
    create_line_from_parameters(net, name="L8", from_bus=b7, to_bus=b8, length_km=180, r_ohm_per_km=0.081,
                                x_ohm_per_km=0.312, g_us_per_km=0, c_nf_per_km=11, max_i_ka=1.5)

    # create_line_from_parameters(net,name="L9",from_bus=3,to_bus=4,length_km=100, r_ohm_per_km=0.312,
    # x_ohm_per_km=0.312,g_us_per_km=0,c_nf_per_km=11)

    create_transformer_from_parameters(net, name="trafo1", hv_bus=b8, lv_bus=b1, sn_mva=192, vn_hv_kv=230,
                                       vn_lv_kv=18, vkr_percent=0, vector_group="Yy0", pfe_kw=0, vk_percent=12,
                                       i0_percent=0)
    create_transformer_from_parameters(net, name="trafo2", hv_bus=b3, lv_bus=b2, sn_mva=500, vn_hv_kv=230,
                                       vn_lv_kv=16.5, vkr_percent=0, vector_group="Yy0", pfe_kw=0, vk_percent=16,
                                       i0_percent=0)

    create_gen(net, bus=b2, p_mw=500, vm_pu=1)
    # create_sgen(net,bus = 2, p_mw=500,name="WT")
    #
    create_load(net, bus=b4, p_mw=130, q_mvar=50)
    create_load(net, bus=b5, p_mw=120, q_mvar=50)
    create_load(net, bus=b6, p_mw=80, q_mvar=25)
    create_load(net, bus=b7, p_mw=50, q_mvar=25)

    return net

