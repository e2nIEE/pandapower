import pandas as pd
from functools import wraps
import sys
import os
import importlib
import inspect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


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



def log_errors(func, error_log):
    @wraps(func)
    def wrapper(*args, **kwargs):
        error_occurred = False
        try:
            func()
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



# --- choosing modules to import ---
module_names = ["test_gen", "test_impedance"]
test_functions_dict = import_functions_from_modules(module_names)

# --- error messages will be saved in the dictionary "error_logs" for each module separately ---
decorated_functions = {}
error_logs = {}

for module_name, funcs in test_functions_dict.items():
    error_logs[module_name] = pd.DataFrame(columns=["function", "error_type", "error_msg", "count"])

    decorated_funcs = {}
    for func_name, func in funcs.items():
        if "test" in func_name:
            decorated_funcs[func_name] = log_errors(func, error_logs[module_name])
    decorated_functions[module_name] = decorated_funcs

# --- in case you do not want to use the dataframe, console output is also available ---
for module_name, funcs in decorated_functions.items():
    for func_name, func in funcs.items():
        print(f"Teste Funktion: {module_name}.{func_name}")
        func()

