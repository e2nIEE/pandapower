def _get_bus_results(net, ppc_0, ppc_1, ppc_2, bus):


    bus_lookup = net._pd2ppc_lookups["bus"]
    ppc_index = bus_lookup[net.bus.index]

    fault = net["_options"]["fault"]  # LG / LL / LLG / LLL

    ppc_sequence = {0: ppc_0, 1: ppc_1, 2: ppc_2, "": ppc_1}
    # -------------- STANDARD PANDAPOWER PART -------------------
    if fault == "LG":
        net.res_bus_sc["ikss_ka"] = ppc_0["bus"][ppc_index, IKSSV] + ppc_1["bus"][ppc_index, IKSSC]
        net.res_bus_sc["skss_mw"] = ppc_0["bus"][ppc_index, SKSS]
        sequence_relevant = range(3)

    elif fault == "LLG":
        sequence_relevant = range(3)

    elif fault == "LL":
        net.res_bus_sc["ikss_ka"] = ppc_1["bus"][ppc_index, IKSSV] + ppc_1["bus"][ppc_index, IKSSC]
        net.res_bus_sc["skss_mw"] = ppc_1["bus"][ppc_index, SKSS]
        sequence_relevant = range(3)

    else:  # LLL (three-phase)
        net.res_bus_sc["ikss_ka"] = ppc_1["bus"][ppc_index, IKSSV] + ppc_1["bus"][ppc_index, IKSSC]
        net.res_bus_sc["skss_mw"] = ppc_1["bus"][ppc_index, SKSS]
        sequence_relevant = ("",)

    # ---- R/X impedance part stays same ----
    for sequence in sequence_relevant:
        ppc_s = ppc_sequence[sequence]

        if fault == "LL":
            fault_ohm_factor = 2 if sequence in [1, 2] else 1
            net.res_bus_sc[f"rk{sequence}_ohm"] = (ppc_s["bus"][ppc_index, R_EQUIV_OHM] +
                                                   net["_options"]["r_fault_ohm"] / fault_ohm_factor)
            net.res_bus_sc[f"xk{sequence}_ohm"] = (ppc_s["bus"][ppc_index, X_EQUIV_OHM] +
                                                   net["_options"]["x_fault_ohm"] / fault_ohm_factor)
        else:
            net.res_bus_sc[f"rk{sequence}_ohm"] = ppc_s["bus"][ppc_index, R_EQUIV_OHM]
            net.res_bus_sc[f"xk{sequence}_ohm"] = ppc_s["bus"][ppc_index, X_EQUIV_OHM]

        # infinity replacement
        baseZ = ppc_s["bus"][ppc_index, BASE_KV] ** 2 / ppc_s["baseMVA"]
        net.res_bus_sc.loc[net.res_bus_sc[f"xk{sequence}_ohm"] / baseZ > 1e9, f"xk{sequence}_ohm"] = np.inf
        net.res_bus_sc.loc[net.res_bus_sc[f"rk{sequence}_ohm"] / baseZ > 1e9, f"rk{sequence}_ohm"] = np.inf

    # standard ip / ith
    if net._options["ip"]:
        net.res_bus_sc["ip_ka"] = ppc_1["bus"][ppc_index, IP]
    if net._options["ith"]:
        net.res_bus_sc["ith_ka"] = ppc_1["bus"][ppc_index, ITH]

    # ------------- NEW PART: ABC STRUJE I SNAGE -----------------

    # extract sequence currents (complex)
    skss_abc_mva = np.full((len(ppc_index), 3), np.nan, dtype=np.float64)
    ikss_abc_ka = np.full((len(ppc_index), 3), np.nan, dtype=np.float64)

    i_1_ka_0 = ppc_0['bus'][:, IKSSV] * np.exp(1j * np.deg2rad(ppc_0['bus'][:, PHI_IKSSV_DEGREE].real))[:, np.newaxis]
    i_1_ka_1 = ppc_1['bus'][:, IKSSV] * np.exp(1j * np.deg2rad(ppc_1['bus'][:, PHI_IKSSV_DEGREE].real))[:, np.newaxis]
    i_1_ka_2 = ppc_2['bus'][:, IKSSV] * np.exp(1j * np.deg2rad(ppc_2['bus'][:, PHI_IKSSV_DEGREE].real))[:, np.newaxis]

    # TODO check results with sgen
    # short-ciruit for inverter-based generation (current source)
    i_2_ka_0 = ppc_0['bus'][:, IKSSC] * np.exp(1j * np.deg2rad(ppc_0['bus'][:, PHI_IKSSV_DEGREE].real))[:, np.newaxis]
    i_2_ka_1 = ppc_1['bus'][:, IKSSC] * np.exp(1j * np.deg2rad(ppc_1['bus'][:, PHI_IKSSV_DEGREE].real))[:, np.newaxis]
    i_2_ka_2 = ppc_2['bus'][:, IKSSC] * np.exp(1j * np.deg2rad(ppc_2['bus'][:, PHI_IKSSV_DEGREE].real))[:, np.newaxis]

    i_1_012_ka = np.stack([i_1_ka_0, i_1_ka_1, i_1_ka_2], 2)
    i_2_012_ka = np.stack([i_2_ka_0, i_2_ka_1, i_2_ka_2], 2)

    i_1_abc_ka = np.apply_along_axis(sequence_to_phase, 2, i_1_012_ka)
    i_2_abc_ka = np.apply_along_axis(sequence_to_phase, 2, i_2_012_ka)

    i_total_abc_ka = i_1_abc_ka + i_2_abc_ka

    i_total_abc_ka_abs = np.zeros((len(bus), i_total_abc_ka.shape[2]))

    v_pu_0 = ppc_0["internal"]["V_ikss"][bus][:, np.newaxis]
    v_pu_1 = ppc_1["internal"]["V_ikss"][bus][:, np.newaxis]
    v_pu_2 = ppc_2["internal"]["V_ikss"][bus][:, np.newaxis]

    v_012_pu = np.stack([v_pu_0, v_pu_1, v_pu_2], 2)
    v_abc_pu = np.apply_along_axis(sequence_to_phase, 2, v_012_pu)
    # v_abc_pu = sequence_to_phase(np.vstack([v_pu_0, v_pu_1, v_pu_2]))
    v_abc_pu[np.abs(v_abc_pu) < 1e-5] = 0

    # this is inefficient because it copies data to fit into a shape, better to use a slice,
    # and even better to find how to use sequence-based powers:
    baseV = ppc_1['bus'][bus, BASE_KV][:, np.newaxis]
    # baseV = ppc_1["internal"]["baseV"][bus][:, np.newaxis]
    # v_base_kv = np.stack([baseV, baseV, baseV], 2)

    skss_abc_mva_phase = i_total_abc_ka_abs * baseV / np.sqrt(3)
    skss_abc_mva[np.ix_(bus, [0, 1, 2, ])] = skss_abc_mva_phase
    ikss_abc_ka[np.ix_(bus, [0, 1, 2, ])] = i_total_abc_ka_abs

    for i, phase in enumerate(['a', 'b', 'c']):
        net.res_bus_sc[f'ikss_{phase}_ka'] = ikss_abc_ka[:, i]  # ikss values
        net.res_bus_sc[f'skss_{phase}_mw'] = skss_abc_mva[:, i]  # skss values


    # ------------- RESTRICT TO ONLY USED BUSES -------------
    net.res_bus_sc = net.res_bus_sc.loc[bus, :]