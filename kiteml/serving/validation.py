"""serving/validation.py — Input validation utilities for the serving layer."""

from typing import Any


def validate_predict_request(
    data: Any,
    feature_names: list[str],
) -> tuple[bool, list[str]]:
    """
    Lightweight validation of a prediction request payload.

    Parameters
    ----------
    data : list of dict or dict
    feature_names : list of str

    Returns
    -------
    (is_valid, errors)
    """
    errors: list[str] = []

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        return False, ["Request 'data' must be a list of dicts or a single dict."]

    for i, row in enumerate(data):
        if not isinstance(row, dict):
            errors.append(f"Row {i} is not a dict.")
            continue
        missing = [f for f in feature_names if f not in row]
        if missing:
            errors.append(f"Row {i} missing features: {missing}")

    return len(errors) == 0, errors
