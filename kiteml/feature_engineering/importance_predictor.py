"""
importance_predictor.py — Predicts estimated importance and confidence for candidate features.
"""

from typing import Sequence

from kiteml.feature_engineering.strategy import FETransformType


class FeatureImportancePredictor:
    """
    Feature Importance Predictor (Flagship Feature for Story 4.2).

    Estimates the predictive value and confidence of candidate engineered features
    prior to physical dataset transformation.
    """

    def predict(
        self,
        source_columns: Sequence[str],
        transform_type: FETransformType | str,
        generated_name: str,
    ) -> tuple[float, float, str]:
        """
        Predict estimated importance (0.0 to 1.0), confidence (0.0 to 1.0), and reasoning text.

        Returns
        -------
        tuple[float, float, str]
            (estimated_importance, confidence, rationale)
        """
        sources_lower = [c.lower() for c in source_columns]
        tt = transform_type.value if hasattr(transform_type, "value") else str(transform_type)

        # 1. Price x Quantity Domain Heuristic
        if len(sources_lower) == 2:
            s1, s2 = sources_lower[0], sources_lower[1]
            if (
                ("price" in s1 or "cost" in s1 or "amount" in s1) and ("qty" in s2 or "quantity" in s2 or "count" in s2)
            ) or (
                ("price" in s2 or "cost" in s2 or "amount" in s2) and ("qty" in s1 or "quantity" in s1 or "count" in s1)
            ):
                return (
                    0.95,
                    0.95,
                    "Multiplicative interaction between price and quantity captures total financial revenue.",
                )

            if ("height" in s1 and "weight" in s2) or ("height" in s2 and "weight" in s1):
                return (
                    0.92,
                    0.90,
                    "Ratio interaction between height and weight approximates body mass index (BMI).",
                )

        # 2. Datetime Component Extractions
        if tt.startswith("datetime_"):
            if "weekday" in tt or "month" in tt:
                return (
                    0.88,
                    0.92,
                    f"Calendar feature ({tt}) captures strong cyclical temporal patterns.",
                )
            return (
                0.82,
                0.90,
                f"Calendar component ({tt}) extracts temporal trend signals.",
            )

        # 3. Skewness Log/Sqrt Transformations
        if tt in ("log_transform", "sqrt_transform"):
            return (
                0.84,
                0.88,
                f"Transformation ({tt}) stabilizes variance and compresses right-skewed distribution.",
            )

        # 4. Text Length Statistics
        if tt.startswith("text_"):
            return (
                0.78,
                0.85,
                f"Text metric ({tt}) captures document length and structural complexity.",
            )

        # 5. Categorical Frequency
        if tt == "category_frequency":
            return (
                0.75,
                0.80,
                "Category frequency encoding preserves category prevalence without one-hot explosion.",
            )

        # Default fallback
        return (
            0.65,
            0.75,
            f"Candidate engineered feature derived via {tt}.",
        )
