import pandera.pandas as pa

import importlib.util
import os
from pathlib import Path

from pandapower import pandapowerNet


def get_dtypes(schema: pa.DataFrameSchema):
    return {name: col.dtype.type for name, col in schema.columns.items()}


def validate_dataframes_for_network(net: pandapowerNet):
    """
    Validate dataframes using schemas from schema folder

    Args:
        net: pandapower Network
    """

    for element in net:
        schema_path = Path(os.getcwd() + os.sep + 'pandapower' + os.sep + 'network_schema') / f"{element}.py"

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
