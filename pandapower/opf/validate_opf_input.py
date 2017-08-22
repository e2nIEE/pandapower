import pandas as pd


def _check_necessary_opf_columns(net, logger):
    # Check if all necessary parameters are given:
    opf_col = {
        'ext_grid': pd.Series(['min_p_kw', 'max_p_kw', 'min_q_kvar', 'max_q_kvar']),
        'gen': pd.Series(['min_p_kw', 'max_p_kw', 'min_q_kvar', 'max_q_kvar']),
        'sgen': pd.Series(['min_p_kw', 'max_p_kw', 'min_q_kvar', 'max_q_kvar']),
        'load': pd.Series(['min_p_kw', 'max_p_kw', 'min_q_kvar', 'max_q_kvar']),
        'dcline': pd.Series(['max_p_kw', 'min_q_from_kvar', 'min_q_to_kvar', 'max_q_from_kvar',
                             'max_q_to_kvar'])}
    error = False
    for element_type in opf_col.keys():
        if len(net[element_type]):
            missing_col = opf_col[element_type].loc[~opf_col[element_type].isin(
                net[element_type].columns)].values
            controllable = True
            if element_type in ['gen', 'sgen', 'load']:
                if 'controllable' not in net[element_type].columns:
                    if element_type == 'gen':
                        net[element_type]['controllable'] = True
                        logger.info("'controllable' has been added to gen as True.")
                    else:
                        net[element_type]['controllable'] = False
                        logger.info("'controllable' has been added to %s as False." % element_type)
                        controllable = False
                else:  # 'ext_grid', 'dcline'
                    if element_type == 'gen':
                        net[element_type].controllable.fillna(True, inplace=True)
                    else:
                        net[element_type].controllable.fillna(False, inplace=True)
                        if not net[element_type].controllable.any():
                            controllable = False
            if bool(len(missing_col)) & controllable:
                if element_type != "ext_grid":
                    logger.error("These columns are missing in " + element_type + ": " +
                                 str(['%s' % col for col in missing_col]))
                    error = True
                else:
                    logger.info("In ext_grid these columns are missing: " +
                                str(['%s' % col for col in missing_col]) + ". In OPF they are " +
                                "considered as +- 1000 TW.")
    if error:
        raise KeyError("OPF parameters are not set correctly. See error log.")

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
