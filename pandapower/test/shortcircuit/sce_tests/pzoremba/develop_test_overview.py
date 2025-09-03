"""
This code is used to provide an overview of pandapower tests. It saves all error messages in a dictionary "error_logs"
for each module tested individually in a dataframe.
Additionally, it provides an overview of the share of valid tests and tests yet to be fixed.
"""


import pandas as pd
from functools import wraps
import sys
import os
import importlib
import inspect

# -- ensuring a normal function is imported instead of a pytest function which cannot be accessed remotely --
def fake_fixture(*dargs, **dkwargs):
    def decorator(func):
        return func
    if dargs and callable(dargs[0]) and not dkwargs:
        return dargs[0]
    return decorator

import pytest
pytest.fixture = fake_fixture

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


# -- determining whether the test function creates the net themselves --
def is_net_function(func):
    try:
        src = inspect.getsource(func)
    except OSError:
        return False

    if "create_empty_network" not in src:
        return False

    if func.__name__.startswith("test_"):
        return False
    if "assert" in src:
        return False

    if "return net" not in src:
        return False
    return True


# -- importing all grid functions automatically, apart from the grids created within test functions --
def import_net_generators(module_names, base_path=None):
    if base_path is None:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    if base_path not in sys.path:
        sys.path.append(base_path)

    fixtures = {}
    for module_name in module_names:
        module = importlib.import_module(module_name)

        for name, func in inspect.getmembers(module, inspect.isfunction):
            if is_net_function(func):
                try:
                    net = func()
                    fixtures[name] = net
                except Exception as e:
                    print(f"Could not load grid {name}: {e}")
    return fixtures


# -- catching and saving error messages --
def log_errors(func, error_log, fixtures):
    @wraps(func)
    def wrapper(*args, **kwargs):
        error_occurred = False
        try:
            sig = inspect.signature(func)
            bound_args = {}
            for name in sig.parameters:
                if name in fixtures:
                    bound_args[name] = fixtures[name]

            func(**bound_args)
        except Exception as e:
            error_occurred = True
            error_type = type(e).__name__
            error_msg = str(e)

            mask = (error_log["function"] == func.__name__) & \
                   (error_log["error_type"] == error_type) & \
                   (error_log["error_msg"] == error_msg)
            if mask.any():
                error_log.loc[mask, "count"] += 1
            else:
                error_log.loc[len(error_log)] = [func.__name__, error_type, error_msg, 1]

            print(f"Error in function '{func.__name__}': {error_type}: {error_msg}")
        finally:
            if not error_occurred:
                error_log.loc[len(error_log)] = [func.__name__, None, None, 0]
    return wrapper

# todo .py files automatisch hinzufügen #################################################################
# -- modules containing the test functions --
module_names = [
    "test_gen", "test_impedance", "test_all_currents", "test_iec60909_4",
    "test_lg", "test_meshing_detection", "test_min_branch_results",
    "test_motor", "test_ring", "test_sc_multi_bus", "test_sc_single_bus",
    "test_sc_voltage", "test_sgen", "test_trafo3w", "test_transformer"
]

# -- using the before defined function on importing the grids --
fixtures = import_net_generators(module_names)


# -- importing the test functions --
def import_functions_from_modules(module_names, base_path=None):
    if base_path is None:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    if base_path not in sys.path:
        sys.path.append(base_path)

    all_functions = {}
    for module_name in module_names:
        module = importlib.import_module(module_name)
        funcs_in_module = {}
        for name, func in inspect.getmembers(module, inspect.isfunction):
            funcs_in_module[name] = func
        all_functions[module_name] = funcs_in_module
    return all_functions


test_functions_dict = import_functions_from_modules(module_names)

# -- saving error logs for each function in dataframe -- all dataframes are to be saved in dictionary "error_logs" --
decorated_functions = {}
error_logs = {}

for module_name, funcs in test_functions_dict.items():
    error_logs[module_name] = pd.DataFrame(columns=["function", "error_type", "error_msg", "count"])

    decorated_funcs = {}
    for func_name, func in funcs.items():
        if "test" in func_name:
            decorated_funcs[func_name] = log_errors(func, error_logs[module_name], fixtures)
    decorated_functions[module_name] = decorated_funcs

# -- running test functions --
for module_name, funcs in decorated_functions.items():
    for func_name, func in funcs.items():
        print(f"Teste Funktion: {module_name}.{func_name}")
        func()



# -- percentage of valid tests --
total_tests = 0
valid_tests = 0
invalid_tests = 0

for df in error_logs.values():
    total_tests += len(df)
    valid_tests += df["error_type"].isna().sum()
    invalid_tests += df["error_type"].notna().sum()

if total_tests > 0:
    valid_pct = (valid_tests / total_tests) * 100
    invalid_pct = (invalid_tests / total_tests) * 100
else:
    valid_pct = invalid_pct = 0

print("\n______________________________________________________________________________________________________________")
print(f"\nOverview:")
print(f"Valid Tests: {valid_tests} ({valid_pct:.0f}%)")
print(f"Invalid Tests: {invalid_tests} ({invalid_pct:.0f}%)")
print("\nAll error messages are saved in the dictionary 'error_logs' for each module individually.")