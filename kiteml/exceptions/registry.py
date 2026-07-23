"""
registry.py — ErrorRegistry global catalog repository for KiteML.
"""

from typing import Any

from kiteml.exceptions import codes
from kiteml.exceptions.metadata import ErrorDefinition


class ErrorRegistry:
    """Central repository storing all registered ErrorDefinition objects."""

    def __init__(self) -> None:
        self._catalog: dict[str, ErrorDefinition] = {}
        self._populate_default_catalog()

    def register(self, definition: ErrorDefinition) -> None:
        """Register an ErrorDefinition in the catalog."""
        if not isinstance(definition, ErrorDefinition):
            raise TypeError(f"Expected ErrorDefinition, got {type(definition)}")
        self._catalog[definition.code] = definition

    def get(self, code: str) -> ErrorDefinition | None:
        """Retrieve ErrorDefinition by code string."""
        return self._catalog.get(code)

    def contains(self, code: str) -> bool:
        """Return True if code is in the catalog."""
        return code in self._catalog

    def all_definitions(self) -> list[ErrorDefinition]:
        """Return all registered ErrorDefinition objects."""
        return list(self._catalog.values())

    def categories(self) -> list[str]:
        """Return list of all unique categories in catalog."""
        return sorted({d.category for d in self._catalog.values()})

    def validate_integrity(self) -> bool:
        """Validate that all registered error codes are unique and valid format."""
        code_list = list(self._catalog.keys())
        if len(code_list) != len(set(code_list)):
            return False
        return all(code.startswith("KML-") for code in code_list)

    def _populate_default_catalog(self) -> None:
        """Populate the standard initial KiteML error catalog."""
        defs = [
            # Dataset
            ErrorDefinition(
                codes.KML_D001,
                "Dataset Not Provided",
                "Dataset",
                message_template="Dataset parameter is missing.",
                default_suggestion="Provide a valid DataFrame or file path.",
                documentation_slug="kml-d001",
                tags=["dataset", "input", "missing"],
            ),
            ErrorDefinition(
                codes.KML_D002,
                "Dataset File Not Found",
                "Dataset",
                message_template="Dataset file '{path}' was not found.",
                default_suggestion="Verify file path existence.",
                documentation_slug="kml-d002",
                tags=["dataset", "file", "path"],
            ),
            ErrorDefinition(
                codes.KML_D003,
                "Empty Dataset",
                "Dataset",
                message_template="Dataset is empty.",
                default_suggestion="Provide a dataset containing rows and columns.",
                documentation_slug="kml-d003",
                tags=["dataset", "empty"],
            ),
            ErrorDefinition(
                codes.KML_D004,
                "Zero Rows Dataset",
                "Dataset",
                message_template="Dataset contains zero rows.",
                default_suggestion="Verify dataset content before training.",
                documentation_slug="kml-d004",
                tags=["dataset", "rows", "empty"],
            ),
            ErrorDefinition(
                codes.KML_D005,
                "Zero Columns Dataset",
                "Dataset",
                message_template="Dataset contains zero columns.",
                default_suggestion="Verify feature columns exist.",
                documentation_slug="kml-d005",
                tags=["dataset", "columns", "empty"],
            ),
            ErrorDefinition(
                codes.KML_D006,
                "Unsupported Format",
                "Dataset",
                message_template="Unsupported file format '{format}'.",
                default_suggestion="Provide a CSV, Parquet, JSON, or Excel file.",
                documentation_slug="kml-d006",
                tags=["dataset", "format"],
            ),
            ErrorDefinition(
                codes.KML_D007,
                "Dataset Loading Failed",
                "Dataset",
                message_template="Failed to load dataset: {reason}",
                default_suggestion="Ensure file permissions and format validity.",
                documentation_slug="kml-d007",
                tags=["dataset", "load"],
            ),
            ErrorDefinition(
                codes.KML_D008,
                "Corrupted Dataset",
                "Dataset",
                message_template="Dataset file appears corrupted or unparseable.",
                default_suggestion="Re-download or export dataset file.",
                documentation_slug="kml-d008",
                tags=["dataset", "corrupted"],
            ),
            # Target
            ErrorDefinition(
                codes.KML_T001,
                "Target Not Specified",
                "Target",
                message_template="Target column was not specified.",
                default_suggestion="Explicitly specify target='column_name'.",
                documentation_slug="kml-t001",
                tags=["target", "missing"],
            ),
            ErrorDefinition(
                codes.KML_T002,
                "Target Column Not Found",
                "Target",
                message_template="Target column '{target}' was not found.",
                default_suggestion="Choose one of the available columns.",
                documentation_slug="kml-t002",
                tags=["target", "columns"],
            ),
            ErrorDefinition(
                codes.KML_T003,
                "Target Contains Only Missing Values",
                "Target",
                message_template="Target column contains 100% missing values.",
                default_suggestion="Impute or select a valid target column.",
                documentation_slug="kml-t003",
                tags=["target", "missing", "nan"],
            ),
            ErrorDefinition(
                codes.KML_T004,
                "Target Contains One Unique Value",
                "Target",
                message_template="Target column contains only 1 unique class/value.",
                default_suggestion="Ensure target contains at least 2 distinct classes.",
                documentation_slug="kml-t004",
                tags=["target", "constant"],
            ),
            ErrorDefinition(
                codes.KML_T005,
                "Invalid Regression Target",
                "Target",
                message_template="Invalid regression target datatype or values.",
                default_suggestion="Ensure target is numeric for regression tasks.",
                documentation_slug="kml-t005",
                tags=["target", "regression"],
            ),
            ErrorDefinition(
                codes.KML_T006,
                "Invalid Classification Target",
                "Target",
                message_template="Invalid classification target distribution.",
                default_suggestion="Ensure target contains discrete class labels.",
                documentation_slug="kml-t006",
                tags=["target", "classification"],
            ),
            ErrorDefinition(
                codes.KML_T007,
                "Identifier Used As Target",
                "Target",
                message_template="Target column appears to be an identifier.",
                default_suggestion="Select a valid target feature instead of ID.",
                documentation_slug="kml-t007",
                tags=["target", "identifier"],
            ),
            # Schema
            ErrorDefinition(
                codes.KML_S001,
                "Duplicate Columns",
                "Schema",
                message_template="Dataset contains duplicate column names: {cols}.",
                default_suggestion="Rename or remove duplicate columns.",
                documentation_slug="kml-s001",
                tags=["schema", "duplicate"],
            ),
            ErrorDefinition(
                codes.KML_S002,
                "Empty Column Names",
                "Schema",
                message_template="Dataset contains empty or whitespace column names.",
                default_suggestion="Provide non-empty header names.",
                documentation_slug="kml-s002",
                tags=["schema", "empty"],
            ),
            ErrorDefinition(
                codes.KML_S003,
                "Unsupported Datatype",
                "Schema",
                message_template="Unsupported datatype '{dtype}' in column '{col}'.",
                default_suggestion="Cast column to numeric or categorical type.",
                documentation_slug="kml-s003",
                tags=["schema", "datatype"],
            ),
            ErrorDefinition(
                codes.KML_S004,
                "Mixed Datatype",
                "Schema",
                message_template="Mixed datatypes detected in column '{col}'.",
                default_suggestion="Clean column so all values share one type.",
                documentation_slug="kml-s004",
                tags=["schema", "mixed"],
            ),
            ErrorDefinition(
                codes.KML_S005,
                "Constant Feature",
                "Schema",
                message_template="Feature column '{col}' is constant.",
                default_suggestion="Remove constant features prior to training.",
                documentation_slug="kml-s005",
                tags=["schema", "constant"],
            ),
            ErrorDefinition(
                codes.KML_S006,
                "Identifier Feature",
                "Schema",
                message_template="Feature column '{col}' is a unique identifier.",
                default_suggestion="Drop identifier column to prevent overfitting.",
                documentation_slug="kml-s006",
                tags=["schema", "identifier"],
            ),
            ErrorDefinition(
                codes.KML_S007,
                "High Cardinality",
                "Schema",
                message_template="High cardinality in categorical feature '{col}'.",
                default_suggestion="Group rare categories or apply target encoding.",
                documentation_slug="kml-s007",
                tags=["schema", "cardinality"],
            ),
            ErrorDefinition(
                codes.KML_S008,
                "Infinite Numeric Values",
                "Schema",
                message_template="Feature column '{col}' contains infinite values.",
                default_suggestion="Replace inf/nan values with finite numbers.",
                documentation_slug="kml-s008",
                tags=["schema", "infinite"],
            ),
            # Validation
            ErrorDefinition(
                codes.KML_V001,
                "Validation Failed",
                "Validation",
                message_template="Dataset validation failed with errors.",
                default_suggestion="Review and resolve dataset validation messages.",
                documentation_slug="kml-v001",
                tags=["validation", "pipeline"],
            ),
            ErrorDefinition(
                codes.KML_V002,
                "Validation Pipeline Interrupted",
                "Validation",
                message_template="Validation pipeline halted by fail-fast check.",
                default_suggestion="Fix critical dataset errors.",
                documentation_slug="kml-v002",
                tags=["validation", "fail_fast"],
            ),
            ErrorDefinition(
                codes.KML_V003,
                "Dataset Health Below Threshold",
                "Validation",
                message_template="Dataset health score {score}/100 is below acceptable threshold.",
                default_suggestion="Clean missing data and outliers.",
                documentation_slug="kml-v003",
                tags=["validation", "health"],
            ),
            ErrorDefinition(
                codes.KML_V004,
                "Critical Validation Error",
                "Validation",
                message_template="Critical error during validation rule evaluation: {rule}.",
                default_suggestion="Check dataset integrity.",
                documentation_slug="kml-v004",
                tags=["validation", "critical"],
            ),
            # Preprocessing
            ErrorDefinition(
                codes.KML_P001,
                "Encoding Failed",
                "Preprocessing",
                message_template="Categorical encoding failed for feature '{col}'.",
                default_suggestion="Check categorical column formatting.",
                documentation_slug="kml-p001",
                tags=["preprocessing", "encoding"],
            ),
            ErrorDefinition(
                codes.KML_P002,
                "Scaling Failed",
                "Preprocessing",
                message_template="Feature scaling failed.",
                default_suggestion="Check numerical features for extreme values or inf.",
                documentation_slug="kml-p002",
                tags=["preprocessing", "scaling"],
            ),
            ErrorDefinition(
                codes.KML_P003,
                "Feature Transformation Failed",
                "Preprocessing",
                message_template="Feature transformation pipeline step failed.",
                default_suggestion="Verify preprocessor input schema.",
                documentation_slug="kml-p003",
                tags=["preprocessing", "transform"],
            ),
            ErrorDefinition(
                codes.KML_P004,
                "Missing Preprocessing Pipeline",
                "Preprocessing",
                message_template="Preprocessing pipeline is not fitted.",
                default_suggestion="Fit preprocessor on training data first.",
                documentation_slug="kml-p004",
                tags=["preprocessing", "fitted"],
            ),
            # Training
            ErrorDefinition(
                codes.KML_M001,
                "Model Training Failed",
                "Training",
                message_template="Training failed for model '{model}'.",
                default_suggestion="Check hyperparameters or data format.",
                documentation_slug="kml-m001",
                tags=["training", "fit"],
            ),
            ErrorDefinition(
                codes.KML_M002,
                "Cross Validation Failed",
                "Training",
                message_template="Cross-validation failed during model selection.",
                default_suggestion="Verify sample count and class balance.",
                documentation_slug="kml-m002",
                tags=["training", "cv"],
            ),
            ErrorDefinition(
                codes.KML_M003,
                "Model Convergence Failed",
                "Training",
                message_template="Model failed to converge.",
                default_suggestion="Increase max_iter or scale input features.",
                documentation_slug="kml-m003",
                tags=["training", "convergence"],
            ),
            ErrorDefinition(
                codes.KML_M004,
                "Model Selection Failed",
                "Training",
                message_template="All candidate models failed during selection.",
                default_suggestion="Fix dataset errors before selection.",
                documentation_slug="kml-m004",
                tags=["training", "selection"],
            ),
            # Prediction
            ErrorDefinition(
                codes.KML_I001,
                "Prediction Failed",
                "Prediction",
                message_template="Inference failed: {reason}",
                default_suggestion="Ensure input data matches training schema.",
                documentation_slug="kml-i001",
                tags=["prediction", "inference"],
            ),
            ErrorDefinition(
                codes.KML_I002,
                "Feature Mismatch",
                "Prediction",
                message_template="Feature mismatch during inference: expected {expected}, got {got}.",
                default_suggestion="Pass matching feature columns.",
                documentation_slug="kml-i002",
                tags=["prediction", "mismatch"],
            ),
            ErrorDefinition(
                codes.KML_I003,
                "Model Not Loaded",
                "Prediction",
                message_template="Model bundle is not loaded or fitted.",
                default_suggestion="Load a valid trained model bundle.",
                documentation_slug="kml-i003",
                tags=["prediction", "loaded"],
            ),
            ErrorDefinition(
                codes.KML_I004,
                "Invalid Inference Input",
                "Prediction",
                message_template="Invalid inference input shape or data type.",
                default_suggestion="Pass DataFrame or dictionary matching schema.",
                documentation_slug="kml-i004",
                tags=["prediction", "input"],
            ),
            # Deployment
            ErrorDefinition(
                codes.KML_DP001,
                "Package Generation Failed",
                "Deployment",
                message_template="Failed to create deployment bundle.",
                default_suggestion="Check destination path permissions.",
                documentation_slug="kml-dp001",
                tags=["deployment", "package"],
            ),
            ErrorDefinition(
                codes.KML_DP002,
                "ONNX Export Failed",
                "Deployment",
                message_template="ONNX export failed.",
                default_suggestion="Ensure skl2onnx dependencies are installed.",
                documentation_slug="kml-dp002",
                tags=["deployment", "onnx"],
            ),
            ErrorDefinition(
                codes.KML_DP003,
                "FastAPI Generation Failed",
                "Deployment",
                message_template="FastAPI code generation failed.",
                default_suggestion="Check deployment configuration.",
                documentation_slug="kml-dp003",
                tags=["deployment", "fastapi"],
            ),
            ErrorDefinition(
                codes.KML_DP004,
                "Docker Export Failed",
                "Deployment",
                message_template="Dockerfile export failed.",
                default_suggestion="Verify target export directory.",
                documentation_slug="kml-dp004",
                tags=["deployment", "docker"],
            ),
            # CLI
            ErrorDefinition(
                codes.KML_C001,
                "Invalid Command",
                "CLI",
                message_template="Unrecognized CLI command '{command}'.",
                default_suggestion="Run 'kiteml --help' to see valid commands.",
                documentation_slug="kml-c001",
                tags=["cli", "command"],
            ),
            ErrorDefinition(
                codes.KML_C002,
                "Missing CLI Argument",
                "CLI",
                message_template="Missing required argument '{arg}'.",
                default_suggestion="Provide required argument.",
                documentation_slug="kml-c002",
                tags=["cli", "argument"],
            ),
            ErrorDefinition(
                codes.KML_C003,
                "Invalid Option",
                "CLI",
                message_template="Invalid command option '{option}'.",
                default_suggestion="Check command line flag spelling.",
                documentation_slug="kml-c003",
                tags=["cli", "option"],
            ),
            ErrorDefinition(
                codes.KML_C004,
                "CLI Execution Failed",
                "CLI",
                message_template="CLI execution error: {details}",
                default_suggestion="Check terminal output for details.",
                documentation_slug="kml-c004",
                tags=["cli", "execution"],
            ),
            # Configuration
            ErrorDefinition(
                codes.KML_CFG001,
                "Invalid Configuration",
                "Configuration",
                message_template="Invalid setting '{setting}'.",
                default_suggestion="Check configuration dictionary key/value.",
                documentation_slug="kml-cfg001",
                tags=["configuration", "invalid"],
            ),
            ErrorDefinition(
                codes.KML_CFG002,
                "Missing Configuration",
                "Configuration",
                message_template="Missing required configuration parameter.",
                default_suggestion="Provide required configuration.",
                documentation_slug="kml-cfg002",
                tags=["configuration", "missing"],
            ),
            ErrorDefinition(
                codes.KML_CFG003,
                "Unsupported Version",
                "Configuration",
                message_template="Unsupported configuration version '{version}'.",
                default_suggestion="Upgrade KiteML to latest version.",
                documentation_slug="kml-cfg003",
                tags=["configuration", "version"],
            ),
        ]
        for d in defs:
            self.register(d)


# Global singleton instance
global_error_registry = ErrorRegistry()
