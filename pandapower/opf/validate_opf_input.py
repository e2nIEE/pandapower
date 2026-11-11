import pandas as pd


def _check_necessary_opf_parameters(net, logger):
    # Check if all necessary parameters are given:
    opf_col = {
        "ext_grid": pd.Series(["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]),
        "gen": pd.Series(["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]),
        "sgen": pd.Series(["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]),
        "load": pd.Series(["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]),
        "storage": pd.Series(["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]),
        "dcline": pd.Series(["max_p_mw", "min_q_from_mvar", "min_q_to_mvar", "max_q_from_mvar", "max_q_to_mvar"]),
    }
    missing_val = []
    error = False
    for element_type, columns in opf_col.items():
        if len(net[element_type]):
            # --- determine controllables
            if element_type in ["ext_grid", "dcline"]:
                controllables = net[element_type].index
            else:
                if "controllable" in net[element_type].columns:
                    net[element_type].controllable = net[element_type].controllable.fillna(element_type == "gen")
                    controllables = net[element_type].index[net[element_type].controllable.astype(bool)]
                else:
                    controllables = net[element_type].index if element_type == "gen" else []

            # --- logging for missing data in element tables with controllables
            if len(controllables):
                missing_col = columns.loc[~columns.isin(net[element_type].columns)].values
                na_col = [
                    col
                    for col in set(columns) - set(missing_col)
                    if net[element_type][col].loc[controllables].isnull().any()
                ]

                if len(missing_col):
                    if element_type == "ext_grid":  # no error due to missing columns
                        logger.debug(
                            f"These missing columns in {element_type} are considered in OPF as +- 1000 TW.: {missing_col}"
                        )
                    else:
                        logger.error(f"These columns are missing in {element_type}: {missing_col}")
                        error = True

                # determine missing values
                if len(na_col):
                    missing_val.append(element_type)

    if missing_val:
        logger.info(
            f"These elements have missing power constraint values, "
            f"which are considered in OPF as +- 1000 TW: {missing_val}"
        )

    # voltage limits: no error due to missing voltage limits
    if "min_vm_pu" in net.bus.columns:
        if net.bus.min_vm_pu.isnull().any():
            logger.info("There are missing bus.min_vm_pu values, which are considered in OPF as 0.0 pu.")
    else:
        logger.info("'min_vm_pu' is missing in bus table. In OPF these limits are considered as 0.0 pu.")
    if "max_vm_pu" in net.bus.columns:
        if net.bus.max_vm_pu.isnull().any():
            logger.info("There are missing bus.max_vm_pu values, which are considered in OPF as 2.0 pu.")
    else:
        logger.info("'max_vm_pu' is missing in bus table. In OPF these limits are considered as 2.0 pu.")

    # branch voltage angle difference limits: no error due to missing limits
    for element in ("line", "trafo", "trafo3w", "impedance"):
        for col in ("min_angle_degree", "max_angle_degree"):
            if element not in net or col not in net[element].columns or not len(net[element]):
                continue
            values = net[element][col]
            if values.isnull().all():
                logger.info(f"All {element}.{col} values are missing, which means unconstrained angle differences.")
            elif (values == 0).any():
                logger.warning(
                    f"{element}.{col} contains 0, which is interpreted as 'no limit' (MATPOWER convention), not as a "
                    "0 degree limit. Use a small epsilon instead of 0 to enforce a limit."
                )

    if error:
        raise KeyError("OPF parameters are not set correctly. See error log.")

    # --- log multiple costs to elements
    cost_check_df = pd.concat([net.poly_cost[["element", "et"]], net.poly_cost[["element", "et"]]], ignore_index=True)
    cost_check_df["power_type"] = ["p"] * net.poly_cost.shape[0] + ["q"] * net.poly_cost.shape[0]
    cost_check_df = pd.concat([cost_check_df, net.pwl_cost[["element", "et", "power_type"]]], ignore_index=True)
    if cost_check_df.duplicated().any():
        raise UserWarning("There are multiple costs to one or multiple elements.")
