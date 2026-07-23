"""
pipeline.py — Master Validation Pipeline Orchestrator for KiteML.
"""

from typing import Any

from kiteml.validation.hooks import ValidationHookSystem
from kiteml.validation.utils import timer
from kiteml.validation.validation_manager import ValidationManager
from kiteml.validation.validation_summary import ValidationSummary


class ValidationPipeline:
    """
    Master Validation Pipeline orchestrating Dataset, Target, Schema, and Data Quality validators
    with fail-fast protection, lifecycle hooks, and unified reporting.
    """

    def __init__(
        self,
        manager: ValidationManager | None = None,
        hooks: ValidationHookSystem | None = None,
    ) -> None:
        self.manager = manager or ValidationManager()
        self.hooks = hooks or ValidationHookSystem()

    def validate(
        self,
        dataframe: Any,
        target: str | None = None,
        problem_type: str | None = None,
        fail_fast: bool = True,
        **kwargs: Any,
    ) -> ValidationSummary:
        """
        Execute validation pipeline on a dataset.

        Parameters
        ----------
        dataframe : Any
            Dataset to validate (DataFrame or file path loaded prior).
        target : str, optional
            Target column name.
        problem_type : str, optional
            'classification' or 'regression'.
        fail_fast : bool
            If True, stop pipeline immediately upon first critical/error failure.
        **kwargs : Any

        Returns
        -------
        ValidationSummary
        """
        self.hooks.trigger_hook(
            "before_validation",
            dataframe=dataframe,
            target=target,
            problem_type=problem_type,
        )

        validator_results: dict[str, Any] = {}
        all_messages: list[Any] = []
        recommendations: list[str] = []
        pipeline_passed = True
        halted_early = False

        validators = self.manager.get_validators()

        with timer() as t:
            for validator in validators:
                res = validator.validate(
                    dataframe,
                    target=target,
                    problem_type=problem_type,
                    **kwargs,
                )
                val_dict = res.to_dict()
                validator_results[validator.name] = val_dict
                all_messages.extend(res.messages)

                # Extract recommendations
                for msg in res.messages:
                    if msg.suggestion and msg.suggestion not in recommendations:
                        recommendations.append(msg.suggestion)

                self.hooks.trigger_hook(
                    "after_validator",
                    validator_name=validator.name,
                    result=res,
                )

                # Fail-fast check
                if not res.passed:
                    pipeline_passed = False
                    if fail_fast:
                        halted_early = True
                        self.hooks.trigger_hook(
                            "on_validation_failure",
                            validator_name=validator.name,
                            result=res,
                        )
                        break

        total_elapsed = t.get("elapsed", 0.0)

        # Aggregate summary metrics
        total_checks = sum(len(d.get("messages", [])) for d in validator_results.values())
        warning_count = sum(1 for m in all_messages if m.severity == "warning")
        error_count = sum(1 for m in all_messages if m.severity == "error")
        critical_count = sum(1 for m in all_messages if m.severity == "critical")
        passed_checks = max(0, total_checks - (warning_count + error_count + critical_count))

        # Health Score & Grade from QualityValidator or DatasetValidator
        health_score = 100
        health_grade = "A+"
        health_rating = "★★★★★ Excellent"

        if "QualityValidator" in validator_results:
            q_stats = validator_results["QualityValidator"].get("statistics", {})
            health_score = q_stats.get("health_score", 100)
            health_grade = q_stats.get("health_grade", "A+")
            health_rating = q_stats.get("health_rating", "★★★★★ Excellent")
        elif "DatasetValidator" in validator_results and not pipeline_passed:
            health_score = 0
            health_grade = "Needs Attention"
            health_rating = "★☆☆☆☆ Unusable"

        ready_for_training = pipeline_passed and not halted_early

        summary = ValidationSummary(
            passed=pipeline_passed,
            health_score=health_score,
            health_grade=health_grade,
            health_rating=health_rating,
            total_checks=total_checks,
            passed_checks=passed_checks,
            warning_count=warning_count,
            error_count=error_count,
            critical_count=critical_count,
            execution_time=total_elapsed,
            validator_results=validator_results,
            recommendations=recommendations,
            ready_for_training=ready_for_training,
        )

        self.hooks.trigger_hook("after_validation", summary=summary)

        return summary
