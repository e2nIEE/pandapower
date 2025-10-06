import pandera.pandas as pa

import importlib.util
import os
from pathlib import Path

from pandapower import pandapowerNet


def get_dtypes(schema: pa.DataFrameSchema):
    return {name: col.dtype.type for name, col in schema.columns.items() if schema.columns[name].required}
    # return {name: dtype.type for name, dtype in schema.dtype.items()} # faster but not working for nonetype


def validate_dataframes_for_network(net: pandapowerNet):
    """
    Validate dataframes using schemas from schema folder

    Args:
        net: pandapower Network
    """

    for element in net:
        schema_path = Path(os.getcwd() + os.sep + "pandapower" + os.sep + "network_schema") / f"{element}.py"

        if not os.path.exists(schema_path):
            continue

        # Dynamic import
        spec = importlib.util.spec_from_file_location(element, schema_path)
        schema_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(schema_module)

        # Assume schema is named 'schema' in each file
        schema = schema_module.schema

        try:
            schema.validate(net[element])
        except Exception as e:
            print(element)
            print(e)


def create_docu_csv_from_schema(schema):
    import pandas as pd
    import json
    schema_json = schema.to_json()
    schema_dict = json.loads(schema_json)

    columns_info = []

    def get_checks(checks):
        if checks is None:
            return ''

        for check in checks:
            if check['options']['check_name'] == 'greater_than':
                return f'>{check["value"]}'
            elif check['options']['check_name'] == 'greater_than_or_equal_to':
                return f'>={check["value"]}'
            elif check['options']['check_name'] == 'less_than':
                return f'<{check["value"]}'
            elif check['options']['check_name'] == 'less_than_or_equal_to':
                return f'<={check["value"]}'
            elif check['options']['check_name'] == 'in_range':
                return f'[{check["min_value"]}, {check["max_value"]}]'
            elif check['options']['check_name'] == 'isin':
                return f'{check["value"]}'
        return ''

    for col_name, col_details in schema_dict['columns'].items():
        columns_info.append({
            'Parameter': col_name,
            'Datatype': col_details.get('dtype', ''),
            'Value Range': get_checks(col_details.get('checks')),
            'nullable': col_details.get('nullable', True),
            'required': col_details.get('required', False),
            'Explanation': col_details.get('description', ''),
        })

    # Create CSV with column metadata
    df = pd.DataFrame(columns_info)

    pd.set_option('display.max_columns', None)
    print(df)
    df.to_csv('column_schema.csv', index=False)