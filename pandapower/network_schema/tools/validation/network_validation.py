import copy
import logging
from dataclasses import dataclass, field

import pandas as pd
import pandera.pandas as pa

from pandapower import pandapowerNet
from pandapower.network_schema.tools.helper import get_element_schema
from pandapower.network_schema.tools.validation.bus_index_validation import build_foreign_key_index_checks

logger = logging.getLogger()

#: Networks that are excluded from validation because their data intentionally violates the schema.
#: Keys are network names (``net.name``), values are the reason for the exclusion.
NETWORKS_EXCLUDED_FROM_VALIDATION = {
    "case11_iwamoto": (
        "The case11_iwamoto data does not contain line ratings (max_i_ka is NaN) and is a "
        "deliberately ill-conditioned case that cannot be solved by a normal power flow, "
        "so the network is not a valid pandapower network in the sense of the schema."
    ),
}

# number of failing rows that are listed per issue before the output is truncated
MAX_REPORTED_FAILURE_CASES = 5


@dataclass(frozen=True)
class ValidationIssue:
    """
    A single, human readable validation issue found in one element dataframe.

    Printing a :class:`ValidationIssue` (or a list / dict of them) shows the
    ``message``, so collected errors can be passed to logging or to a user
    facing report without further formatting.

    Attributes:
        element: name of the network element table the issue belongs to (e.g. "line")
        message: human readable description of the issue
        reason: pandera reason code of the issue (e.g. "DATAFRAME_CHECK")
        column: affected column, if the issue is limited to a single column
        check: name of the violated check, if available
        failure_cases: raw failure cases reported by pandera, if available
    """

    element: str
    message: str
    reason: str = ""
    column: str | None = None
    check: str | None = None
    failure_cases: pd.DataFrame | str | bool | None = field(default=None, repr=False)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return self.message


def _format_index(index) -> str:
    """Format an element index for the report, dropping a pointless ``.0`` of float indices."""
    if isinstance(index, float) and index.is_integer():
        return str(int(index))
    return str(index)


def _describe_failure_cases(failure_cases: pd.DataFrame, limit: int = MAX_REPORTED_FAILURE_CASES) -> str:
    """
    Turn the failure cases of a single pandera check into a compact, readable string.

    Only the first ``limit`` entries are listed, the rest is summarized as a count.

    Args:
        failure_cases: dataframe as reported in ``SchemaError.failure_cases``
        limit: maximum number of entries to list

    Returns:
        readable representation, e.g. ``"-1.0 (index 0), 0.0 (index 3) and 2 more"``
    """
    total = len(failure_cases)
    shown = failure_cases.head(limit)
    values = [f"{row['failure_case']!s} (index {_format_index(row['index'])})" for _, row in shown.iterrows()]
    remainder = total - limit
    if remainder > 0:
        values.append(f"... and {remainder} more")
    return ", ".join(values)


def _check_name(check) -> str:
    """Return the readable name of a pandera check."""
    error = getattr(check, "error", None)
    if error:
        return str(error)
    name = getattr(check, "name", None)
    if name:
        return str(name)
    return str(check)


def _dtype_name(dtype) -> str:
    """
    Return the bare name of a dtype, e.g. ``float64`` for ``"dtype('float64')"``.

    Pandera reports the expected dtype of a column as ``"dtype('float64')"``, which is
    unnecessarily noisy in a user facing message.

    Args:
        dtype: dtype as reported by pandera, either as ``dtype`` object or as string

    Returns:
        the dtype name without the surrounding ``dtype(...)``
    """
    text = str(dtype)
    if text.startswith("dtype(") and text.endswith(")"):
        text = text[len("dtype(") : -1].strip("'\"")
    return text



def _build_message(element: str, error: pa.errors.SchemaError) -> str:
    """
    Create a user readable message for a single pandera ``SchemaError``.

    Args:
        element: name of the element table the error belongs to
        error: the pandera error raised for one check or schema level rule

    Returns:
        one sentence describing what is wrong, and where
    """
    reason = getattr(error.reason_code, "name", str(error.reason_code))
    column = error.column_name
    check = _check_name(error.check)
    failure_cases = error.failure_cases

    if reason == "COLUMN_NOT_IN_DATAFRAME":
        return f"{element}: required column '{failure_cases}' is missing"

    if reason == "WRONG_DATATYPE":
        return f"{element}: column '{column}' has dtype '{_dtype_name(failure_cases)}', expected '{_dtype_name(check)}'"

    if reason == "SERIES_CONTAINS_NULLS":
        where = _describe_failure_cases(failure_cases) if isinstance(failure_cases, pd.DataFrame) else failure_cases
        return f"{element}: column '{column}' contains null values: {where}"

    if reason == "DATAFRAME_CHECK":
        if not isinstance(failure_cases, pd.DataFrame):
            # schema level checks (e.g. column group dependencies) report a single
            # boolean instead of the failing rows - the check text is the message
            return f"{element}: {check}"
        if check.startswith("foreign key"):
            return (
                f"{element}: column '{column}' references indices that do not exist in the "
                f"referenced table: {_describe_failure_cases(failure_cases)}"
            )
        return f"{element}: column '{column}' violates {check}: {_describe_failure_cases(failure_cases)}"

    if reason == "CHECK_ERROR":
        # failure cases contain a full traceback - only the first line is useful
        first_line = str(failure_cases).splitlines()[0] if failure_cases is not None else "unknown error"
        return f"{element}: check {check} on column '{column}' could not be evaluated: {first_line}"

    if reason == "SCHEMA_COMPONENT_CHECK":
        return f"{element}: {check}"

    return f"{element}: {reason} in column '{column}' ({check})"


def _issues_from_schema_errors(element: str, errors: pa.errors.SchemaErrors) -> list[ValidationIssue]:
    """
    Convert a pandera ``SchemaErrors`` (raised with ``lazy=True``) into issues.

    Args:
        element: name of the element table
        errors: the collected pandera errors

    Returns:
        one :class:`ValidationIssue` per violated check
    """
    issues = []
    for error in errors.schema_errors:
        issues.append(
            ValidationIssue(
                element=element,
                message=_build_message(element, error),
                reason=getattr(error.reason_code, "name", str(error.reason_code)),
                column=error.column_name,
                check=_check_name(error.check),
                failure_cases=error.failure_cases,
            )
        )
    return issues


def _issue_from_exception(element: str, error: Exception) -> ValidationIssue:
    """
    Convert an exception raised outside of pandera's validation into an issue.

    This is used for errors that happen while preparing the validation, e.g. when a
    referenced element table is missing, to keep the collected output complete.

    Args:
        element: name of the element table
        error: the exception that occurred

    Returns:
        a :class:`ValidationIssue` describing the exception
    """
    return ValidationIssue(
        element=element,
        message=f"{element}: validation could not be performed: {type(error).__name__}: {error}",
        reason=type(error).__name__,
        failure_cases=str(error),
    )


def format_validation_errors(errors: dict[str, list[ValidationIssue]]) -> str:
    """
    Format collected validation errors as a single readable string.

    Args:
        errors: result of :func:`validate_network` with ``collect_errors=True``

    Returns:
        one line per issue, grouped by element, or an empty string if there are no errors

    Example:
        >>> errors = validate_network(net, collect_errors=True)
        >>> print(format_validation_errors(errors))
        line: column 'length_km' violates greater_than(0): -1.0 (index 0)
    """
    return "\n".join(str(issue) for issues in errors.values() for issue in issues)


def validate_network(
    net: pandapowerNet,
    groups_to_validate: str | list[str] | set[str] | None = None,
    collect_errors: bool = False,
) -> dict[str, list[ValidationIssue]] | None:
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
        groups_to_validate (str | list[str] | set[str] | None): Optional group(s) to validate.
            If provided, validates the specified dependency groups in addition to
            the base schema. Valid groups depend on the element schema (e.g., "opf",
            "sc", "3ph" for ext_grid). A single group can be passed as a string, several
            groups as a list or set. If None, validates only the base schema.
        collect_errors (bool): If False (default) the first element that fails validation
            raises a ``SchemaError``. If True, all elements are validated and all issues
            are collected instead of raising, so that a complete picture of the network
            is returned.

    Raises:
        pa.errors.SchemaError: If validation fails for any network element and
                              ``collect_errors`` is False. The error includes details
                              about which element failed validation and the underlying cause.

    Returns:
        None if ``collect_errors`` is False and the validation passed.
        If ``collect_errors`` is True, a dict mapping the element name to a list of
        :class:`ValidationIssue`. Each issue carries a readable ``message`` and printing
        the issues (or using :func:`format_validation_errors`) gives a user facing report.
        The dict is empty if all elements are valid.

    Note:

        - Schema files must contain a variable named '{element}_schema'
        - Elements without corresponding schema files are skipped
        - Missing or invalid schema modules are logged as warnings but don't raise exceptions
        - Bus index validation is performed after schema validation for each element
        - When groups_to_validate is specified, validates the base schema AND the specified groups
        - Networks listed in NETWORKS_EXCLUDED_FROM_VALIDATION (matched by ``net.name``) are
          skipped entirely, since their data is known to violate the schema on purpose

    Example:
        >>> import pandapower as pp
        >>> net = pp.pandapowerNet(name="validate_network")
        >>> # ... populate network with elements
        >>> validate_network(net)  # Validates base schema only
        >>> validate_network(net, groups_to_validate="sc")  # Validates + sc group
        >>> validate_network(net, groups_to_validate=["sc", "3ph"])  # Validates + both groups
        >>> # collect all issues instead of raising on the first one
        >>> errors = validate_network(net, collect_errors=True)
        >>> for element, issues in errors.items():
        ...     for issue in issues:
        ...         print(issue)
    """
    net_name = net.get("name")
    if net_name in NETWORKS_EXCLUDED_FROM_VALIDATION:
        logger.info(f"Skipping validation of network '{net_name}': {NETWORKS_EXCLUDED_FROM_VALIDATION[net_name]}")
        return

    # Normalize groups_to_validate to a set
    if groups_to_validate is None:
        groups_set = None
    elif isinstance(groups_to_validate, str):
        groups_set = {
            groups_to_validate,
        }
    else:
        groups_set = set(groups_to_validate)

    collected_errors: dict[str, list[ValidationIssue]] = {}

    for element in net:
        schema = copy.deepcopy(get_element_schema(element))
        if schema is None:
            continue

        df = net[element]
        reset_required: set[str] = set()
        try:
            # validate bus index dependency
            build_foreign_key_index_checks(schema, net)
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
            schema.validate(df, lazy=collect_errors)
        except pa.errors.SchemaErrors as e:
            # only raised with lazy=True, i.e. when collect_errors is True
            collected_errors[element] = _issues_from_schema_errors(element, e)
        except pa.errors.SchemaError as e:
            # if collect_errors is False, keep the behaviour of raising on the first issue
            msg = f"Validation failed for {element}"
            if groups_set is not None:
                msg += f" (groups: {groups_set})"
            raise pa.errors.SchemaError(data=e, message=msg, schema=schema) from e
        except Exception as e:
            if not collect_errors:
                msg = f"Validation failed for {element}"
                if groups_set is not None:
                    msg += f" (groups: {groups_set})"
                raise pa.errors.SchemaError(data=e, message=msg, schema=schema) from e
            collected_errors[element] = [_issue_from_exception(element, e)]
        finally:  # ensure the schema is reset even if exception is raised.
            # reset the schema
            for col_name in reset_required:
                schema.columns[col_name].required = False

    if collect_errors:
        return collected_errors
    return None
