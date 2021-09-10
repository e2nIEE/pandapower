import pandapower as pp


def test_grid():
    net = pp.create_empty_network()
    # std_type = "243-AL1/39-ST1A 110.0"
    # r = 0.1188
    std_type = "490-AL1/64-ST1A 220.0"
    r = 0.059
    s_base = 1000
    v_base = 220
    z_base = v_base ** 2 / s_base

    pp.create_buses(net, 5, 220)

    pp.create_line(net, 0, 1, 0.84e-2 * z_base / r, std_type, name="1-2")
    pp.create_line(net, 0, 3, 0.84e-2 * z_base / r, std_type, name="1-4")
    pp.create_line(net, 1, 2, 0.67e-2 * z_base / r, std_type, name="2-3")
    pp.create_line(net, 1, 4, 0.42e-2 * z_base / r, std_type, name="2-5")
    pp.create_line(net, 2, 3, 0.67e-2 * z_base / r, std_type, name="3-4")
    pp.create_line(net, 3, 4, 0.42e-2 * z_base / r, std_type, name="4-5")

    pp.create_ext_grid(net, 0, 1.05, name="G1")
    pp.create_sgen(net, 0, 200, name="R1")
    pp.create_sgen(net, 1, 250, name="R2")
    pp.create_gen(net, 2, 600, 1.05, name="G3")
    pp.create_gen(net, 4, 300, 1.05, name="G5")

    pp.create_load(net, 1, 600, 240)
    pp.create_load(net, 3, 1000, 400)
    pp.create_load(net, 4, 400, 160)

    return net
