import json
import pandas as pd

'''
def analyze_bus_errors(file1_path, file2_path):
    # Load first file
    with open(file1_path, 'r', encoding='utf-8') as f:
        data1 = json.load(f)

    # Load second file
    with open(file2_path, 'r', encoding='utf-8') as f:
        data2 = json.load(f)

    # Combine both files
    all_data = data1 + data2

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Remove duplicates
    df_unique = df.drop_duplicates()

    print(f"Total records before removing duplicates: {len(df)}")
    print(f"Total records after removing duplicates: {len(df_unique)}")
    print(f"Duplicates removed: {len(df) - len(df_unique)}\n")

    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total tests: {len(df_unique)}")
    print(f"Tests with current errors (current ok = False): {len(df_unique[df_unique['current ok'] == False])}")
    print(f"Tests with impedance errors (impedance ok = False): {len(df_unique[df_unique['impedance ok'] == False])}")
    print(f"Total failed tests (total ok = False): {len(df_unique[df_unique['total ok'] == False])}")
    print(f"Total passed tests (total ok = True): {len(df_unique[df_unique['total ok'] == True])}")

    print("\n" + "=" * 80)
    print("ANALYSIS BY FAULT TYPE")
    print("=" * 80)

    fault_types = df_unique['fault_type'].unique()

    for fault in fault_types:
        df_fault = df_unique[df_unique['fault_type'] == fault]

        current_errors = len(df_fault[df_fault['current ok'] == False])
        impedance_errors = len(df_fault[df_fault['impedance ok'] == False])
        total_failed = len(df_fault[df_fault['total ok'] == False])
        total_passed = len(df_fault[df_fault['total ok'] == True])

        print(f"\n{fault} fault:")
        print(f"  Total tests: {len(df_fault)}")
        print(f"  Current errors: {current_errors}")
        print(f"  Impedance errors: {impedance_errors}")
        print(f"  Total failed: {total_failed}")
        print(f"  Total passed: {total_passed}")

        # Networks with errors for this fault type
        failed_nets = df_fault[df_fault['total ok'] == False]['name'].unique()
        print(f"  Networks with errors ({len(failed_nets)}):")
        for net in failed_nets:
            net_data = df_fault[(df_fault['name'] == net) & (df_fault['total ok'] == False)]
            curr_err = len(net_data[net_data['current ok'] == False])
            imp_err = len(net_data[net_data['impedance ok'] == False])
            print(f"    - {net}: {curr_err} current errors, {imp_err} impedance errors")


# Usage
analyze_bus_errors('df_bus_simple_saved_10.json', 'df_branch_saved_10.json')'''




def analyze_branch_errors(file1_path, file2_path):
    # Load first file
    with open(file1_path, 'r', encoding='utf-8') as f:
        data1 = json.load(f)

    # Load second file
    with open(file2_path, 'r', encoding='utf-8') as f:
        data2 = json.load(f)

    # Combine both files
    all_data = data1 + data2

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Remove duplicates
    df_unique = df.drop_duplicates()

    print(f"Total records before removing duplicates: {len(df)}")
    print(f"Total records after removing duplicates: {len(df_unique)}")
    print(f"Duplicates removed: {len(df) - len(df_unique)}\n")

    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total tests: {len(df_unique)}")
    print(f"Tests with current errors (current ok = False): {len(df_unique[df_unique['current ok'] == False])}")
    print(f"Tests with voltage errors (voltage ok = False): {len(df_unique[df_unique['voltage ok'] == False])}")
    print(f"Total failed tests (total ok = False): {len(df_unique[df_unique['total ok'] == False])}")
    print(f"Total passed tests (total ok = True): {len(df_unique[df_unique['total ok'] == True])}")

    print("\n" + "=" * 80)
    print("ANALYSIS BY FAULT TYPE")
    print("=" * 80)

    fault_types = df_unique['fault_type'].unique()

    for fault in fault_types:
        df_fault = df_unique[df_unique['fault_type'] == fault]

        current_errors = len(df_fault[df_fault['current ok'] == False])
        voltage_errors = len(df_fault[df_fault['voltage ok'] == False])
        total_failed = len(df_fault[df_fault['total ok'] == False])
        total_passed = len(df_fault[df_fault['total ok'] == True])

        print(f"\n{fault} fault:")
        print(f"  Total tests: {len(df_fault)}")
        print(f"  Current errors: {current_errors}")
        print(f"  Voltage errors: {voltage_errors}")
        print(f"  Total failed: {total_failed}")
        print(f"  Total passed: {total_passed}")

        # Networks with errors for this fault type
        failed_nets = df_fault[df_fault['total ok'] == False]['name'].unique()
        print(f"  Networks with errors ({len(failed_nets)}):")
        for net in failed_nets:
            net_data = df_fault[(df_fault['name'] == net) & (df_fault['total ok'] == False)]
            curr_err = len(net_data[net_data['current ok'] == False])
            volt_err = len(net_data[net_data['voltage ok'] == False])
            print(f"    - {net}: {curr_err} current errors, {volt_err} voltage errors")


# Usage
analyze_branch_errors('df_branch_simple_saved_10.json', 'dvukovic/df_branch_saved_10.json')