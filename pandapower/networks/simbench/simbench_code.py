from pandapower.converter.simbench.auxiliary import ensure_iterability


__author__ = "smeinecke"


def complete_data_sb_code(scenario):
    return "1-complete_data-mixed-all-%s-sw" % str(scenario)


def complete_grid_sb_code(scenario):
    return "1-EHVHVMVLV-mixed-all-%s-sw" % str(scenario)


def collect_all_simbench_codes(version=1, hv_level=None, lv_level=None, hv_type=None, lv_grid=None,
                               scenario=None, breaker_rep=None, all_data=True, shortened=False,
                               **kwargs):
    """ Returns a list of all possible SimBench Codes, considering given fixed sb_code_parameters.
        **kwargs are ignored.
    """
    pos_hv_level = ["EHV", "HV", "MV", "LV"]
    pos_lv_level = ["HV", "MV", "LV", ""]
    # determine all hv_level - lv_level combinations and store them into "v_level"
    pos_v_level = []
    for i, hv in enumerate(pos_hv_level):
        pos_v_level += [(hv, pos_lv_level[i])]
        if i < 3:
            pos_v_level += [(hv, pos_lv_level[3])]

    # define possible code names for each parameter
    hv_level = pos_hv_level if hv_level is None else ensure_iterability(hv_level)
    lv_level = pos_lv_level if lv_level is None else ensure_iterability(lv_level)
    v_level = [vl for vl in pos_v_level if vl[0] in hv_level and vl[1] in lv_level]
    if hv_type is None:
        hv_type = {"EHV": ["mixed"],
                   "HV": ["mixed", "urban"],
                   "MV": ["rural", "semiurb", "urban", "comm"],
                   "LV": ["rural1", "rural2", "rural3", "semiurb4", "semiurb5", "urban6"]}
    elif not isinstance(hv_type, dict):
        new_hv_type = dict()
        for hv in hv_level:
            new_hv_type[hv] = ensure_iterability(hv_type)
        hv_type = new_hv_type
    if not shortened:
        lv_grid = {"EHV": {"mixed": ["all", "1", "2"]},
                   "HV": {"mixed": ["all", "1.105", "2.102", "4.101"],
                          "urban": ["all", "2.203", "3.201", "4.201"]},
                   "MV": {"rural": ["all", "1.108", "2.107", "4.101"],
                          "semiurb": ["all", "3.202", "4.201", "5.220"],
                          "urban": ["all", "5.303", "6.305", "6.309"],
                          "comm": ["all", "3.403", "4.416", "5.401"]}} if lv_grid is None \
            else lv_grid
    else:
        lv_grid = {"EHV": {"mixed": ["all", "1", "2"]},
                   "HV": {"mixed": ["all", "1.102", "2.101", "4.101"],
                          "urban": ["all", "2.202", "3.201", "4.201"]},
                   "MV": {"rural": ["all", "1.101", "2.102", "4.101"],
                          "semiurb": ["all", "3.202", "4.201", "5.201"],
                          "urban": ["all", "5.301", "6.302", "6.302"],
                          "comm": ["all", "2.402", "4.401", "6.402"]}} if lv_grid is None \
            else lv_grid
    scenario = ["0", "1", "2"] if scenario is None else [
        str(i) for i in ensure_iterability(scenario)]
    breaker_rep = ["sw", "no_sw"] if breaker_rep is None else ensure_iterability(breaker_rep)

    # determine all possible SimBench Codes
    all_simbench_codes = []
    if all_data:
        all_simbench_codes += [complete_data_sb_code(sc) for sc in scenario]
        all_simbench_codes += [complete_grid_sb_code(sc) for sc in scenario]
    # starting with all grid data code
    for hv in hv_level:
        lvs = [v[1] for v in v_level if v[0] == hv]
        for lv in lvs:
            types = hv_type[hv]
            for type_ in types:
                lv_grids = lv_grid[hv][type_] if len(lv) else [""]
                for grid in lv_grids:
                    for scen in scenario:
                        for breaker in breaker_rep:
                            all_simbench_codes += [str(version)+"-"+hv+lv+"-"+type_+"-"+grid +
                                                   "-"+scen+"-"+breaker]
    return all_simbench_codes


def get_parameters_from_simbench_code(sb_code):
    """ Converts a SimBench Code into flag parameters, describing a SimBench grid selection. """
    sb_code_split = sb_code.split("-")
    version = int(sb_code_split[0])
    if sb_code_split[1] != "complete_data":
        hv_level = sb_code_split[1].split("V")[0] + "V"
        lv_level = sb_code_split[1].split(hv_level)[1]
    else:
        hv_level = sb_code_split[1]
        lv_level = ""
    hv_type = sb_code_split[2]
    try:
        lv_grid = int(sb_code_split[3])
    except ValueError:
        lv_grid = sb_code_split[3]
    scenario = sb_code_split[4]
    breaker_rep = False if "no" in sb_code_split[5] else True
    return [version, hv_level, lv_level, hv_type, lv_grid, scenario, breaker_rep]


def get_simbench_code_from_parameters(sb_code_parameters):
    """ Converts flag parameters, describing a SimBench grid selection, into the unique regarding
    SimBench Code. """
    switch_param = "no_sw" if not sb_code_parameters[6] else "sw"
    sb_code = str(sb_code_parameters[0])+"-"+sb_code_parameters[1]+sb_code_parameters[2]+"-" + \
        sb_code_parameters[3]+"-"+str(sb_code_parameters[4])+"-"+str(sb_code_parameters[5])+"-" + \
        switch_param
    return sb_code


def get_simbench_code_and_parameters(sb_code_info):
    """ Detects whether sb_code_info are parameters or is code and returns both. """
    if isinstance(sb_code_info, list):
        assert len(sb_code_info) == 7
        return get_simbench_code_from_parameters(sb_code_info), sb_code_info
    elif isinstance(sb_code_info, str):
        return sb_code_info, get_parameters_from_simbench_code(sb_code_info)
    else:
        raise ValueError("'sb_code_info' needs to be simbench code or simbench code parameters.")


if __name__ == '__main__':
    if 1:
        all_ = collect_all_simbench_codes()
        sb_code_parameters = get_parameters_from_simbench_code(all_[8])
        sb_code = get_simbench_code_from_parameters(sb_code_parameters)
    else:
        pass
