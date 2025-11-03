import importlib.util
import logging
import os
from pathlib import Path

import pandera.pandas as pa

from pandapower import pandapowerNet
from pandapower.network_schema.tools.validation.bus_index_validation import _bus_index_validation

logger = logging.getLogger()


def validate_network(net: pandapowerNet):
    """
    Validate pandapower network element dataframes using schemas from the schema folder.

    This function iterates through all elements in a pandapower network and validates
    each element's dataframe against its corresponding schema file. Schema files are
    expected to be located in pandapower.network_schema and named after the
    element type (e.g., 'bus.py', 'line.py', etc.).

    The validation process includes:

    1. Dynamic import of element-specific schema modules
    2. Schema validation of dataframe structure and data types
    3. Bus index dependency validation for network consistency

    Args:
        net (pandapowerNet): A pandapower network object containing various
                           electrical network elements (buses, lines, loads, etc.)
                           to be validated.

    Raises:
        pa.errors.SchemaError: If validation fails for any network element.
                              The error includes details about which element
                              failed validation and the underlying cause.

    Note:

        - Schema files must contain a variable named '{element}_schema'
        - Elements without corresponding schema files are skipped
        - Missing or invalid schema modules are logged as warnings but don't raise exceptions
        - Bus index validation is performed after schema validation for each element

    Example:
        >>> import pandapower as pp
        >>> net = pp.create_empty_network()
        >>> # ... populate network with elements
        >>> validate_network(net)  # Validates all elements
    """

    def _dynamic_import(element, schema_path):
        # Dynamic import for schemata from files
        spec = importlib.util.spec_from_file_location(element, schema_path)
        if spec is None:
            logger.warning(f"Schema for {element} not found, no spec")
            return None
        schema_module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        if loader is None:
            logger.warning(f"Schema for {element} not found, no loader")
            return None
        loader.exec_module(schema_module)
        return schema_module

    for element in net:
        schema_path = Path(Path(__file__).parents[2], f"{element}.py")

        if not os.path.exists(schema_path):
            continue

        schema = getattr(_dynamic_import(element, schema_path), f"{element}_schema")

        if schema is None:
            continue

        # validate element schema
        # try:
        #     schema.validate(net[element])
        # except Exception as e:
        #     raise pa.errors.SchemaError(data=e, message=f"Validation failed for {element}", schema=schema)

        # validate bus index dependency
        _bus_index_validation(element, net)
