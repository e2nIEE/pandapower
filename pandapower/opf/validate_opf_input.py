import pandas as pd


def _check_necessary_opf_parameters(net, logger):
    # Check if all necessary parameters are given:
    opf_col = {
        'ext_grid': pd.Series(['min_p_kw', 'max_p_kw', 'min_q_kvar', 'max_q_kvar']),
        'gen': pd.Series(['min_p_kw', 'max_p_kw', 'min_q_kvar', 'max_q_kvar']),
        'sgen': pd.Series(['min_p_kw', 'max_p_kw', 'min_q_kvar', 'max_q_kvar']),
        'load': pd.Series(['min_p_kw', 'max_p_kw', 'min_q_kvar', 'max_q_kvar']),
        'storage': pd.Series(['min_p_kw', 'max_p_kw', 'min_q_kvar', 'max_q_kvar']),
        'dcline': pd.Series(['max_p_kw', 'min_q_from_kvar', 'min_q_to_kvar', 'max_q_from_kvar',
                             'max_q_to_kvar'])}
    missing_val = []
    error = False
    for element_type in opf_col.keys():
        if len(net[element_type]):
            missing_col = opf_col[element_type].loc[~opf_col[element_type].isin(
                net[element_type].columns)].values
            # --- ensure "controllable" as column
            controllable = True
            if element_type in ['gen', 'sgen', 'load', 'storage']:
                if 'controllable' not in net[element_type].columns:
                    if element_type == 'gen':
                        net[element_type]['controllable'] = True
                        logger.info("'controllable' has been added to gen as True.")
                    else:
                        net[element_type]['controllable'] = False
                        logger.info("'controllable' has been added to %s as False." % element_type)
                        controllable = False
                else:
                    if element_type == 'gen':
                        net[element_type].controllable.fillna(True, inplace=True)
                    else:  # 'sgen', 'load'
                        net[element_type].controllable.fillna(False, inplace=True)
                        if not net[element_type].controllable.any():
                            controllable = False
            # --- logging for missing data in element tables with controllables
            if controllable:
                if bool(len(missing_col)):
                    if element_type != "ext_grid":
                        logger.error("These columns are missing in " + element_type + ": " +
                                     str(missing_col))
                        error = True
                    else:  # "ext_grid" -> no error due to missing columns at ext_grid
                        logger.info("These missing columns in ext_grid are considered in OPF as " +
                                    "+- 1000 TW.: " + str(missing_col))
                # determine missing values
                for lim_col in set(opf_col[element_type]) - set(missing_col):
                    if element_type in ['gen', 'sgen', 'load', 'storage']:
                        controllables = net[element_type].loc[net[element_type].controllable].index
                    else:  # 'ext_grid', 'dcline'
                        controllables = net[element_type].index
                    if net[element_type][lim_col].loc[controllables].isnull().any():
                        missing_val.append(element_type)
                        break
    if missing_val:
        logger.info("These elements have missing power constraint values, which are considered " +
                    "in OPF as +- 1000 TW: " + str(missing_val))

    # voltage limits: no error due to missing voltage limits
    if 'min_vm_pu' in net.bus.columns:
        if net.bus.min_vm_pu.isnull().any():
            logger.info("There are missing bus.min_vm_pu values, which are considered in OPF as " +
                        "0.0 pu.")
    else:
        logger.info("min_vm_pu is missing in bus table. In OPF these limits are considered as " +
                    "0.0 pu.")
    if 'max_vm_pu' in net.bus.columns:
        if net.bus.max_vm_pu.isnull().any():
            logger.info("There are missing bus.max_vm_pu values, which are considered in OPF as " +
                        "2.0 pu.")
    else:
        logger.info("max_vm_pu is missing in bus table. In OPF these limits are considered as " +
                    "2.0 pu.")

    if error:
        raise KeyError("OPF parameters are not set correctly. See error log.")
