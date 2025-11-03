import pandas as pd
import pandera.pandas as pa


def create_column_group_dependency_validation_func(columns):
    """
    Creates a validator function that ensures column group dependency.

    Returns a validator that checks whether the specified columns are either
    all present or all absent in a DataFrame. This is useful for validating
    that related columns exist together as a group.

    Args:
        columns (list): List of column names that must exist together as a group.
                       Duplicates are automatically handled.

    Returns:
        callable: A validator function that takes a DataFrame and returns True
                 if the column dependency is satisfied, False otherwise.

    Examples:
        >>> validator = create_column_group_dependency_validation_func(['lat', 'lon', 'altitude'])
        >>>
        >>> # Valid: All columns present
        >>> df1 = pd.DataFrame({'lat': [1], 'lon': [2], 'altitude': [3], 'other': [4]})
        >>> validator(df1)  # Returns True
        >>>
        >>> # Valid: No columns present
        >>> df2 = pd.DataFrame({'name': ['John'], 'age': [25]})
        >>> validator(df2)  # Returns True
        >>>
        >>> # Invalid: Only some columns present
        >>> df3 = pd.DataFrame({'lat': [1], 'lon': [2], 'name': ['John']})
        >>> validator(df3)  # Returns False

    Note:
        The validator returns True when either:
        - None of the specified columns exist in the DataFrame
        - All of the specified columns exist in the DataFrame

        It returns False when only a subset of the columns exist.
    """

    def validator(df):
        df_columns = set(df.columns)
        target_columns = set(columns)
        present_columns = target_columns & df_columns

        # Either all present or none present
        return len(present_columns) in {0, len(target_columns)}

    return validator


def create_column_dependency_checks_from_metadata(names: list, schema_columns: dict) -> list:
    """
    Create dependency validation checks for columns based on their metadata.

    This function generates Pandera checks that validate column dependencies by examining
    column metadata. For each specified dependency name, it identifies columns that have
    this dependency marked in their metadata and creates corresponding validation checks.

    Args:
        names (list): List of dependency names to check for in column metadata.
        schema_columns (dict): Dictionary mapping column names to their schema objects,
                             where each schema object has a metadata attribute containing
                             dependency information.

    Returns:
        list: List of Pandera Check objects that validate column group dependencies.
              Each check ensures that dependent columns are present in the dataframe.

    Example:
        >>> names = ['required_group', 'optional_group']
        >>> schema_cols = {
        ...     'col1': pa.Column(metadata={'required_group': True}),
        ...     'col2': pa.Column(metadata={'required_group': True}),
        ...     'col3': pa.Column(metadata={'optional_group': True})
        ... }
        >>> checks = create_column_dependency_checks_from_metadata(names, schema_cols)
        >>> len(checks)
        3

    Note:
        - Only columns with metadata containing the specified dependency names are processed
        - Each check uses the validate_column_group_dependency function for validation

        - Error messages indicate which columns have dependency violations
    """
    checks = []
    for name in names:
        for col in [
            col_name
            for col_name, col_schema in schema_columns.items()
            if col_schema.metadata
            and col_schema.metadata.get(name)  # Extract column names that have opf=True in their metadata.
        ]:
            checks.append(
                pa.Check(
                    create_column_group_dependency_validation_func(col),
                    error=f"{name} columns have dependency violations. Please ensure {col} columns are present in the dataframe.",
                )
            )
    return checks
