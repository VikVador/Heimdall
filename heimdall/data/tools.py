r"""A collection of tools designed for data module."""

__all__ = [
    "assert_date_format",
]

import re


def assert_date_format(date_string: str) -> None:
    r"""Assert that a date string follows the expected format.

    Arguments:
        date_string : Date to validate, expected format 'YYYY-MM-DD'.
    """

    pattern = r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"
    if not re.match(pattern, date_string):
        raise ValueError("ERROR - The format is incorrect, it should be YYYY-MM-DD.")
