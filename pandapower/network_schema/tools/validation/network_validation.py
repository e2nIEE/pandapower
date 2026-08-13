import importlib.util
import logging
import os
from pathlib import Path

import pandera.pandas as pa

from pandapower import pandapowerNet
from pandapower.network_schema.tools.validation.bus_index_validation import _bus_index_validation

logger = logging.getLogger()


def validate_network(net: pandapowerNet, groups_to_validate: str | set[str] | None = None):
    """
    Validate pandapower network element dataframes using schemas from the schema folder.

    This function iterates through all elements in a pandapower network and validates
    each element's dataframe against its corresponding schema file. Schema files are
    expected to be located in pandapower.network_schema and named after the
    element type (e.g., 'bus.py', 'line.py', etc.).

    The validation process includes:

    1. Dynamic import of element-specific schema modules
    2. Schema validation of dataframe structure and data types
    3. Optional group dependency validation (e.g., opf, sc, 3ph)
    4. Bus index dependency validation for network consistency

    Args:
        net (pandapowerNet): A pandapower network object containing various
                           electrical network elements (buses, lines, loads, etc.)
                           to be validated.
        groups_to_validate (str | list[str] | None): Optional group(s) to validate.
            If provided, validates the specified dependency groups in addition to
            the base schema. Valid groups depend on the element schema (e.g., "opf",
            "sc", "3ph" for ext_grid). If None, validates only the base schema.

    Raises:
        pa.errors.SchemaError: If validation fails for any network element.
                              The error includes details about which element
                              failed validation and the underlying cause.

    Note:

        - Schema files must contain a variable named '{element}_schema'
        - Elements without corresponding schema files are skipped
        - Missing or invalid schema modules are logged as warnings but don't raise exceptions
        - Bus index validation is performed after schema validation for each element
        - When groups_to_validate is specified, validates the base schema AND the specified groups

    Example:
        >>> import pandapower as pp
        >>> net = pp.pandapowerNet(name="validate_network")
        >>> # ... populate network with elements
        >>> validate_network(net)  # Validates base schema only
        >>> validate_network(net, groups_to_validate="sc")  # Validates + sc group
        >>> validate_network(net, groups_to_validate=["sc", "3ph"])  # Validates + both groups
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

    # Normalize groups_to_validate to a set
    if groups_to_validate is None:
        groups_set = None
    elif isinstance(groups_to_validate, str):
        groups_set = {
            groups_to_validate,
        }
    else:
        groups_set = groups_to_validate

    for element in net.keys():
        schema_path = Path(Path(__file__).parents[2], f"{element}.py")

        if not os.path.exists(schema_path):
            continue

        schema_module = _dynamic_import(element, schema_path)
        if schema_module is None:
            continue

        schema = getattr(schema_module, f"{element}_schema", None)
        if schema is None:
            continue

        df = net[element]
        reset_required: set[str] = set()
        try:
            if groups_set is not None:
                # Get all group checks at once for all specified groups
                cols = [
                    name
                    for name, col in schema.columns.items()
                    if col.metadata is not None and list(groups_set & set(col.metadata.keys()))
                ]
                for col_name in cols:
                    if not schema.columns[col_name].required:
                        reset_required.add(col_name)
                        schema.columns[col_name].required = True
            schema.validate(df)
        except Exception as e:
            msg = f"Validation failed for {element}"
            if groups_set is not None:
                msg += f" (groups: {groups_set})"
            raise pa.errors.SchemaError(data=e, message=msg, schema=schema)
        finally:  # ensure the schema is reset even if exception is raised.
            # reset the schema
            for col_name in reset_required:
                schema.columns[col_name].required = False

        # validate bus index dependency
        _bus_index_validation(element, schema, net)
