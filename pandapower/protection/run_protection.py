import pandapower.shortcircuit as sc
import pandas as pd


def calculate_protection_times(net_sc):
    # this function takes net_sc that has already been acted upon by sc.calc_sc (net_sc) and returns a dataframe
    # containing the activation time for each protection device

    protection_devices = net_sc.protection.query("in_service").object.values
    protection_results = []

    for p in protection_devices:
        protection_result = p.protection_function(net_sc)
        protection_results.append(protection_result)

    df_protection_results = pd.DataFrame.from_dict(protection_results)
    print(df_protection_results)
    return df_protection_results

def run_protection(net, fault_bus):
    # similar logic as the control loop
    # simulates cascading faults
    # FINISH IMPLEMENTING ONLY AFTER ALL OTHER METHODS AND FUNCTIONS ARE DONE
    sc.calc_sc(net, bus=fault_bus)
    pt = calculate_protection_times(net)

    protection_devices = net.protection.query("in_service").object.values
    # here: function call to calculate times of every device
    # add sorting by tripping time and group by tripping time (multiple devices can have same reaction time - same to order and level in controller)

    converged = True
    while True:
        for p in protection_devices:
            if not p.has_tripped():
                p.protection_function(net)
                converged = False
                # add break?

        if converged:
            # no protection devices tripped
            break

        # if any of the protection devices tripped we re-analyze the grid
        sc.calc_sc(net, bus=fault_bus)
        pt = calculate_protection_times(net)

