import pandas as pd


def _check_necessary_opf_parameters(net, logger):
    # Check if all necessary parameters are given:
    opf_col = {
        'ext_grid': pd.Series(['min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar']),
        'gen': pd.Series(['min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar']),
        'sgen': pd.Series(['min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar']),
        'load': pd.Series(['min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar']),
        'storage': pd.Series(['min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar']),
        'dcline': pd.Series(['max_p_mw', 'min_q_from_mvar', 'min_q_to_mvar', 'max_q_from_mvar',
                             'max_q_to_mvar'])}
    missing_val = []
    error = False
    for element_type, columns in opf_col.items():
        if len(net[element_type]):

            # --- determine controllables
            if element_type in ["ext_grid", "dcline"]:
                controllables = net[element_type].index
            else:
                if "controllable" in net[element_type].columns:
                    if element_type == 'gen':
                        net[element_type].controllable.fillna(True, inplace=True)
                    else:  # 'sgen', 'load', 'storage'
                        net[element_type].controllable.fillna(False, inplace=True)
                    controllables = net[element_type].index[net[element_type].controllable.astype(
                        bool)]
                else:
                    controllables = net[element_type].index if element_type == 'gen' else []

            missing_col = columns.loc[~columns.isin(net[element_type].columns)].values
            na_col = [col for col in set(columns)-set(missing_col) if
                      net[element_type][col].isnull().any()]

            # --- logging for missing data in element tables with controllables
            if len(controllables) and len(missing_col):
                if element_type != "ext_grid":
                    logger.error("These columns are missing in " + element_type + ": " +
                                 str(missing_col))
                    error = True
                else:  # "ext_grid" -> no error due to missing columns at ext_grid
                    logger.debug("These missing columns in ext_grid are considered in OPF as " +
                                 "+- 1000 TW.: " + str(missing_col))
            # determine missing values
            if len(na_col):
                missing_val.append(element_type)

    if missing_val:
        logger.info("These elements have missing power constraint values, which are considered " +
                    "in OPF as +- 1000 TW: " + str(missing_val))

    # voltage limits: no error due to missing voltage limits
    if 'min_vm_pu' in net.bus.columns:
        if net.bus.min_vm_pu.isnull().any():
            logger.info("There are missing bus.min_vm_pu values, which are considered in OPF as " +
                        "0.0 pu.")
    else:
        logger.info("'min_vm_pu' is missing in bus table. In OPF these limits are considered as " +
                    "0.0 pu.")
    if 'max_vm_pu' in net.bus.columns:
        if net.bus.max_vm_pu.isnull().any():
            logger.info("There are missing bus.max_vm_pu values, which are considered in OPF as " +
                        "2.0 pu.")
    else:
        logger.info("'max_vm_pu' is missing in bus table. In OPF these limits are considered as " +
                    "2.0 pu.")

    if error:
        raise KeyError("OPF parameters are not set correctly. See error log.")
