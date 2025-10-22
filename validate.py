"""
validate.py

Module for validating and formatting SQL queries.
Ensures queries are safe (non-destructive) and well-formatted.
"""

import re
from typing import Tuple
import sqlparse
from sqlparse.exceptions import SQLParseError

# Flag potentially destructive statements
_FORBIDDEN = re.compile(
    r'\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|EXEC|MERGE)\b',
    re.IGNORECASE
)


def validate_sql(query: str) -> Tuple[bool, str, str]:
    """
    Validate and format a generated SQL query.

    Args:
        query (str): The SQL query string to validate.

    Returns:
        Tuple[bool, str, str]:
            - is_valid: True if query is safe and parseable
            - message: Validation or error message
            - formatted_sql: Formatted SQL string (empty if invalid)
    """
    if not query or not query.strip():
        return False, "Empty query", ""

    if _FORBIDDEN.search(query):
        return False, "Forbidden or potentially destructive statement detected", ""

    try:
        formatted = sqlparse.format(query, reindent=True, keyword_case='upper')
        parsed = sqlparse.parse(formatted)
        if not parsed:
            return False, "Unable to parse SQL", formatted
        return True, "OK", formatted
    except SQLParseError as e:
        return False, f"SQL parse/format error: {e}", ""
