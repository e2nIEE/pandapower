import pandapower as pp
from pandapower.file_io import from_json
from pandapower import pp_dir
from pandapower.shortcircuit import calc_sc
import os
import pandas as pd
from pandapower.test.shortcircuit.sce_tests.functions_tests import check_pattern
import numpy as np
from tqdm import tqdm

# Toleranzen für relevante Größen
tolerances = {"ikss_ka": 1e-4, "skss_mw": 1e-4, "rk_ohm": 1e-5, "xk_ohm": 1e-5,
              "vm_pu": 1e-4, "va_degree": 1e-2, "p_mw": 1e-4, "q_mvar": 1e-4, "ikss_degree": 1e-3}

faults = ["LLL", "LL", "LG", "LLG"]
cases = ["min", "max"]
fault_ohm_values = [(0.0, 0.0), (5.0, 5.0)]
lv_tol_percent = 10  # 6
fault_locations = [2]
gen_bus = 3

def initialize_current_source_test(net):

    # Set generators and synchronous generators as out of service initially
    net.sgen.in_service = False
    net.gen.in_service = False

    # Configure generator parameters
    net.gen['k'] = 6
    net.sgen['k'] = 6
    net.gen["current_source"] = False

    # Set one generator as in service and as a current source
    net.gen.loc[net.gen.bus == gen_bus, 'in_service'] = True
    net.gen.loc[net.gen.bus == gen_bus, 'current_source'] = True

    return net

def compare_sc_results(net, branch=False):
    all_differences = []

    for r_fault_ohm, x_fault_ohm in fault_ohm_values:
        for fault in faults:
            for case in cases:
                for fault_location in fault_locations:
                    calc_sc(net, fault=fault, case=case, bus=fault_location, return_all_currents=False,
                            branch_results=branch, ip=False, r_fault_ohm=r_fault_ohm, x_fault_ohm=x_fault_ohm,
                            lv_tol_percent=lv_tol_percent)

                    if branch:
                        net.res_line_sc["name"] = net.line.name
                        net.res_line_sc.sort_values(by="name", inplace=True)
                        result_df = net.res_line_sc
                    else:
                        net.res_bus_sc["name"] = net.bus.name
                        net.res_bus_sc.sort_values(by="name", inplace=True)
                        result_df = net.res_bus_sc

                    # Modify the in_service status for the generator and synchronous generator
                    net.gen.loc[net.gen.bus == gen_bus, 'in_service'] = False
                    net.sgen.loc[net.sgen.bus == gen_bus, 'in_service'] = True

                    # Run modified short-circuit calculation
                    calc_sc(net, fault=fault, case=case, bus=fault_location, return_all_currents=False,
                            branch_results=branch, ip=False, r_fault_ohm=r_fault_ohm, x_fault_ohm=x_fault_ohm,
                            lv_tol_percent=lv_tol_percent)

                    if branch:
                        net.res_line_sc["name"] = net.line.name
                        net.res_line_sc.sort_values(by="name", inplace=True)
                        modified_pf_results = net.res_line_sc
                        compare_ids = net.line.name

                    else:
                        net.res_bus_sc["name"] = net.bus.name
                        net.res_bus_sc.sort_values(by="name", inplace=True)
                        modified_pf_results = net.res_bus_sc
                        compare_ids = net.bus.name

                    element_id_column = "name"

                    # Modify the in_service status for the generator and synchronous generator to have the initial status
                    net.gen.loc[net.gen.bus == gen_bus, 'in_service'] = True
                    net.sgen.loc[net.sgen.bus == gen_bus, 'in_service'] = False

                    for element in compare_ids:
                        for column in result_df.columns:
                            if column == element_id_column:
                                continue
                            column_key = check_pattern(column)
                            if column_key not in tolerances:
                                continue
                            if branch and column_key == "ikss_ka" and column == "ikss_ka":
                                continue
                            try:
                                pp_val = result_df.loc[result_df[element_id_column] == element, column].values[0]
                                pf_val = \
                                modified_pf_results.loc[modified_pf_results[element_id_column] == element, column].values[0]

                                if column_key.endswith("degree"):
                                    diff = (pp_val - pf_val + 180) % 360 - 180
                                else:
                                    diff = pp_val - pf_val

                                diff_perc = 1 - pp_val / pf_val if pf_val != 0 else np.nan
                                diff_perc = diff_perc * 100
                                tol = tolerances[column_key]
                                status = "OK" if abs(diff) <= tol else "Exceeds tolerance"

                                all_differences.append({
                                    "Element": element,
                                    "Fault Type": fault,
                                    "Case": case,
                                    "r_fault_ohm": r_fault_ohm,
                                    "x_fault_ohm": x_fault_ohm,
                                    "Quantity": column,
                                    "gen as current source": pp_val,
                                    "sgen results": pf_val,
                                    "Difference": np.round(diff, 4),
                                    "Difference_perc": np.round(diff_perc, 4),
                                    "Status": status
                                })
                            except Exception as e:
                                continue
    return pd.DataFrame(all_differences)

def generate_summary_tables(net_names, fault_locations, detailed=False):
    bus_summary = []
    branch_summary = []

    current_keys = ["ikss_ka", "ikss_degree", "ikss_a_ka", "ikss_b_ka", "ikss_c_ka",
                    "ikss_a_degree", "ikss_b_degree", "ikss_c_degree",
                    "ikss_a_from_ka", "ikss_b_from_ka", "ikss_c_from_ka",
                    "ikss_a_to_ka", "ikss_b_to_ka", "ikss_c_to_ka"]

    impedance_keys = ["rk_ohm", "xk_ohm", "rk0_ohm", "rk1_ohm", "rk2_ohm",
                      "xk0_ohm", "xk1_ohm", "xk2_ohm"]

    voltage_keys = ["vm_pu", "va_degree", "vm_a_from_pu", "vm_b_from_pu", "vm_c_from_pu",
                    "vm_a_to_pu", "vm_b_to_pu", "vm_c_to_pu",
                    "va_a_from_degree", "va_b_from_degree", "va_c_from_degree",
                    "va_a_to_degree", "va_b_to_degree", "va_c_to_degree"]

    combinations = [(net, loc) for net in net_names for loc in fault_locations]

    for net_name, fault_location in tqdm(combinations, desc="generate_summary", unit="grid"):
        try:
            net = from_json(
                os.path.join(pp.pp_dir, "test", "shortcircuit", "sce_tests", "test_grids", "wp_2.2_2.4", net_name + ".json"))
            net = initialize_current_source_test(net)
            diff_df = compare_sc_results(net, branch=False)
            diff_df_branch = compare_sc_results(net, branch=True)

            if diff_df is None and diff_df_branch is None:
                continue

            # bus
            if not diff_df.empty:
                if detailed:
                    grouped_bus = diff_df.groupby(["Fault Type", "Case", "r_fault_ohm", "x_fault_ohm"])
                    for group_keys, group_df in grouped_bus:
                        current_ok = all(group_df[group_df["Quantity"].isin(current_keys)]["Status"] == "OK")
                        impedance_ok = all(group_df[group_df["Quantity"].isin(impedance_keys)]["Status"] == "OK")
                        overall_ok = all([current_ok, impedance_ok])
                        bus_summary.append({
                            "name": net_name,
                            "location": fault_location,
                            "fault_type": group_keys[0],
                            "case": group_keys[1],
                            "rx_fault_ohm": str(group_keys[2]),
                            "current ok": True if current_ok else False,
                            "impedance ok": True if impedance_ok else False,
                            "total ok": True if overall_ok else False
                        })
                else:
                    for fault_type, group_df in diff_df.groupby("Fault Type"):
                        current_ok = all(group_df[group_df["Quantity"].isin(current_keys)]["Status"] == "OK")
                        impedance_ok = all(group_df[group_df["Quantity"].isin(impedance_keys)]["Status"] == "OK")
                        overall_ok = all([current_ok, impedance_ok])
                        bus_summary.append({
                            "name": net_name,
                            "location": fault_location,
                            "fault_type": fault_type,
                            "current ok": True if current_ok else False,
                            "impedance ok": True if impedance_ok else False,
                            "total ok": True if overall_ok else False
                        })

            # branch
            if not diff_df_branch.empty:
                if detailed:
                    grouped_branch = diff_df_branch.groupby(["Fault Type", "Case", "r_fault_ohm", "x_fault_ohm"])
                    for group_keys, group_df in grouped_branch:
                        current_ok = all(group_df[group_df["Quantity"].isin(current_keys)]["Status"] == "OK")
                        voltage_ok = all(group_df[group_df["Quantity"].isin(voltage_keys)]["Status"] == "OK")
                        overall_ok = all([current_ok, voltage_ok])
                        branch_summary.append({
                            "name": net_name,
                            "location": fault_location,
                            "fault_type": group_keys[0],
                            "case": group_keys[1],
                            "rx_fault_ohm": str(group_keys[2]),
                            "current ok": True if current_ok else False,
                            "voltage ok": True if voltage_ok else False,
                            "total ok": True if overall_ok else False
                        })
                else:
                    for fault_type, group_df in diff_df_branch.groupby("Fault Type"):
                        current_ok = all(group_df[group_df["Quantity"].isin(current_keys)]["Status"] == "OK")
                        voltage_ok = all(group_df[group_df["Quantity"].isin(voltage_keys)]["Status"] == "OK")
                        overall_ok = all([current_ok, voltage_ok])
                        branch_summary.append({
                            "name": net_name,
                            "location": fault_location,
                            "fault_type": fault_type,
                            "current ok": True if current_ok else False,
                            "voltage ok": True if voltage_ok else False,
                            "total ok": True if overall_ok else False
                        })

        except Exception as e:
            print(f"error for {net_name}, {fault_location}: {e}")
            continue

    return pd.DataFrame(bus_summary), pd.DataFrame(branch_summary)

# Define the network name
net_name = r"1_four_bus_radial_grid_gen.json"

# Load the network from a JSON file
net = from_json(os.path.join(pp.pp_dir, "test", "shortcircuit", "sce_tests", "test_grids", "wp_2.2_2.4", net_name))
net = initialize_current_source_test(net)

diff_df_bus = compare_sc_results(net, branch=False)
diff_df_branch = compare_sc_results(net, branch=True)
print("Errors in bus_results found:", len(diff_df_bus.loc[diff_df_bus.Status != "OK"]))
print("Errors in branch_results found:", len(diff_df_branch.loc[diff_df_branch.Status != "OK"]))

testfiles_gen_path = os.path.join(pp_dir, 'test', 'shortcircuit', 'sce_tests', 'test_grids', 'wp_2.2_2.4')
net_names_gen = [f[:-5] for f in os.listdir(testfiles_gen_path) if f.endswith(".json")]

## detailed overview for all grids
names = [name for name in net_names_gen if name.endswith("_gen")]
df_bus, df_branch = generate_summary_tables(names, fault_locations, detailed=True)

## simple overview for all grids
names = [name for name in net_names_gen if name.endswith("_gen")]
df_bus_simple, df_branch_simple = generate_summary_tables(names, fault_locations, detailed=False)