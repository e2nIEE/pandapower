import pandas as pd
import pandera.pandas as pa


def create_column_group_dependency_validation_func(columns):
    """
    Creates a validator function that ensures column group dependency.

    Returns a validator that checks whether the specified columns are either
    all present and properly filled, or all absent in a DataFrame. This is
    useful for validating that related columns exist together as a group and
    contain valid data.

    Args:
        columns (list): List of column names that must exist together as a group.
                       Duplicates are automatically handled.

    Returns:
        callable: A validator function that takes a DataFrame and returns True
                 if the column dependency is satisfied, False otherwise.

    Examples:
        >>> validator = create_column_group_dependency_validation_func(['lat', 'lon', 'altitude'])
        >>>
        >>> # Valid: All columns present and filled
        >>> df1 = pd.DataFrame({'lat': [1.0], 'lon': [2.0], 'altitude': [3.0], 'other': [4]})
        >>> validator(df1)  # Returns True
        >>>
        >>> # Valid: No columns present
        >>> df2 = pd.DataFrame({'name': ['John'], 'age': [25]})
        >>> validator(df2)  # Returns True
        >>>
        >>> # Invalid: Only some columns present
        >>> df3 = pd.DataFrame({'lat': [1.0], 'lon': [2.0], 'name': ['John']})
        >>> validator(df3)  # Returns False
        >>>
        >>> # Invalid: All columns present but some have null values
        >>> df4 = pd.DataFrame({'lat': [1.0, pd.NA], 'lon': [2.0, 3.0], 'altitude': [3.0, 4.0]})
        >>> validator(df4)  # Returns False

    Note:
        The validator returns True when either:

        - None of the specified columns exist in the DataFrame
        - All of the specified columns exist AND all rows have non-null values

          in all these columns

        It returns False when:

        - Only a subset of the columns exist
        - All columns exist but at least one row has null/NA values in any column

    """

    def validator(df):
        df_columns = set(df.columns)
        target_columns = set(columns)
        present_columns = target_columns & df_columns

        # Case 1: None of the columns are present - valid
        if len(present_columns) == 0:
            return True

        # Case 2: Not all columns are present - invalid
        if len(present_columns) != len(target_columns):
            return False

        # Case 3: All columns are present - check if they're consistently filled
        # For each row, either all target columns should be NA or none should be NA
        target_cols_list = list(target_columns)
        na_mask = df[target_cols_list].isna()

        # Count NA values per row across target columns
        na_counts = na_mask.sum(axis=1)

        # Valid if each row has either 0 NAs (all filled) or len(target_columns) NAs (all empty)
        valid_counts = (na_counts == 0) | (na_counts == len(target_columns))

        return all(valid_counts)

    return validator


def create_column_dependency_checks_from_metadata(names: list, schema_columns: dict) -> list[pa.Check]:
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
        cols = [
            col_name
            for col_name, col_schema in schema_columns.items()
            if getattr(col_schema, "metadata", None) and col_schema.metadata.get(name)
        ]
        if len(cols) > 0:
            checks.append(
                pa.Check(
                    create_column_group_dependency_validation_func(cols),
                    error=f"{name} columns have dependency violations. Please ensure columns {cols} are present in the dataframe.",
                )
            )
    return checks
