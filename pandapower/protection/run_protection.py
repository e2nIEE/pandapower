import pandapower.shortcircuit as sc
import pandas as pd


def calculate_protection_times(net, scenario="sc"):
    """
    Calculate protection times for short-circuit and power-flow scenarios

    INPUT:
        **net** - pandapower network

        **scenario** (string, 'sc') - specify what type of scenario to calculate for: 'sc' for short-circuit, 'pp'
            for power flow. Default is 'sc'

    OUTPUT:

        **df_protection_results** - pandas dataframe containing results

    """

    if scenario != "sc" and scenario != "pp":
        raise ValueError("scenario must be either sc or op")

    protection_devices = net.protection.loc[net.protection.in_service, "object"].values
    protection_results = []

    for p in protection_devices:
        protection_result = p.protection_function(net=net, scenario=scenario)
        protection_results.append(protection_result)

    df_protection_results = pd.DataFrame.from_dict(protection_results)
    return df_protection_results
