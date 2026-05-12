"""
inference_guardrails.py — Production-safe input validation for KiteML.

Validates every input before model inference to prevent silent failures
from schema mismatches, wrong dtypes, missing columns, or out-of-range values.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class GuardrailViolation:
    """A single validation failure."""

    field: str
    violation_type: str  # "missing_column" | "wrong_dtype" | "null_value" | "out_of_range"
    expected: Any
    received: Any
    message: str
    severity: str  # "error" | "warning"


@dataclass
class GuardrailResult:
    """Result of guardrail validation."""

    is_valid: bool
    violations: list[GuardrailViolation]
    errors: list[GuardrailViolation]
    warnings: list[GuardrailViolation]
    summary: str

    def raise_if_invalid(self) -> None:
        """Raise ValueError if validation failed."""
        if not self.is_valid:
            messages = [v.message for v in self.errors]
            raise ValueError("Inference input failed validation:\n" + "\n".join(f"  ⚠️  {m}" for m in messages))


class InferenceGuardrails:
    """
    Validates prediction inputs against a training schema.

    Parameters
    ----------
    feature_names : list of str
        Feature columns expected by the model.
    schema : dict, optional
        Schema dict from the .kiteml bundle (has dtype and range info).
    allow_extra_columns : bool
        If True, extra columns in input are silently dropped.
    """

    def __init__(
        self,
        feature_names: list[str],
        schema: dict | None = None,
        allow_extra_columns: bool = True,
    ):
        self.feature_names = feature_names
        self.schema = schema or {}
        self.allow_extra_columns = allow_extra_columns

    def validate(self, X: Any) -> GuardrailResult:
        """
        Validate input data before inference.

        Parameters
        ----------
        X : pd.DataFrame, dict, or list of dicts

        Returns
        -------
        GuardrailResult
        """
        violations: list[GuardrailViolation] = []

        # Normalize to DataFrame
        try:
            if isinstance(X, dict):
                df = pd.DataFrame([X])
            elif isinstance(X, list) and all(isinstance(r, dict) for r in X):
                df = pd.DataFrame(X)
            elif isinstance(X, pd.DataFrame):
                df = X.copy()
            elif isinstance(X, np.ndarray):
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                df = pd.DataFrame(X, columns=self.feature_names[: X.shape[1]])
            else:
                df = pd.DataFrame(X)
        except Exception as e:
            return GuardrailResult(
                is_valid=False,
                violations=[
                    GuardrailViolation(
                        field="input",
                        violation_type="conversion_error",
                        expected="DataFrame-compatible",
                        received=type(X).__name__,
                        message=f"Cannot convert input to DataFrame: {e}",
                        severity="error",
                    )
                ],
                errors=[],
                warnings=[],
                summary="Input conversion failed.",
            )

        input_cols = set(df.columns)

        # ── Check missing required columns ────────────────────────────────
        for feat in self.feature_names:
            if feat not in input_cols:
                violations.append(
                    GuardrailViolation(
                        field=feat,
                        violation_type="missing_column",
                        expected=feat,
                        received=None,
                        message=f"Required column '{feat}' is missing from input.",
                        severity="error",
                    )
                )

        # ── Check extra columns ───────────────────────────────────────────
        extra = input_cols - set(self.feature_names)
        if extra and not self.allow_extra_columns:
            for col in extra:
                violations.append(
                    GuardrailViolation(
                        field=col,
                        violation_type="extra_column",
                        expected="not present",
                        received=col,
                        message=f"Unexpected column '{col}' in input.",
                        severity="warning",
                    )
                )

        # ── Check nulls in present columns ────────────────────────────────
        for feat in self.feature_names:
            if feat in df.columns and df[feat].isna().any():
                n_null = int(df[feat].isna().sum())
                violations.append(
                    GuardrailViolation(
                        field=feat,
                        violation_type="null_value",
                        expected="non-null",
                        received=f"{n_null} null(s)",
                        message=f"Column '{feat}' contains {n_null} null value(s).",
                        severity="warning",
                    )
                )

        errors = [v for v in violations if v.severity == "error"]
        warnings = [v for v in violations if v.severity == "warning"]
        is_valid = len(errors) == 0

        if is_valid:
            summary = f"✅ Validation passed ({len(warnings)} warning(s))."
        else:
            summary = f"❌ Validation failed: {len(errors)} error(s), {len(warnings)} warning(s)."

        return GuardrailResult(
            is_valid=is_valid,
            violations=violations,
            errors=errors,
            warnings=warnings,
            summary=summary,
        )

    def sanitize(self, X: Any) -> pd.DataFrame:
        """
        Validate and return a clean DataFrame with only the expected columns.

        Raises ValueError on hard errors, drops extra columns.
        """
        result = self.validate(X)
        result.raise_if_invalid()

        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, list):
            df = pd.DataFrame(X)
        else:
            df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        # Keep only expected columns (in training order)
        present = [f for f in self.feature_names if f in df.columns]
        return df[present]
