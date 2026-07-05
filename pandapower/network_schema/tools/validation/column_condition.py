import pandera as pa
import numpy as np


def create_lower_equals_column_check(first_element: str, second_element: str) -> pa.Check:
    """
    Create a Pandera check that validates one column is less than or equal to another.

    This check verifies that values in the first column are less than or equal to
    corresponding values in the second column. The check passes (returns True) if:

    - The first column value is <= the second column value
    - Either column contains NaN values

    - Either column is missing from the DataFrame

    Parameters
    ----------
    first_element : str
        Name of the first column (should contain lower or equal values).
    second_element : str
        Name of the second column (should contain higher or equal values).

    Returns
    -------
    pa.Check
        A Pandera Check object that can be applied to a DataFrame schema.

    Examples
    --------
    >>> check = create_lower_equals_column_check("min_value", "max_value")
    >>> schema = pa.DataFrameSchema({
    ...     "min_value": pa.Column(float, checks=[check]),
    ...     "max_value": pa.Column(float)
    ... })
    """
    return pa.Check(
        lambda df: (
            df[first_element].fillna(-np.inf) <= df[second_element].fillna(np.inf)
            if all(col in df.columns for col in [first_element, second_element])
            else True
        ),
        error=f"Column '{first_element}' must be <= column '{second_element}'",
    )
