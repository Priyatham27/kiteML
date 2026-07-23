# KiteML Error Catalog Reference

Complete reference of all standard KiteML error codes (`KML-XXXNNN`).

| Error Code | Name | Category | Severity | Message Template | Default Suggestion |
| ---------- | ---- | -------- | -------- | ---------------- | ------------------ |
| `KML-D001` | Dataset Not Provided | Dataset | ERROR | `Dataset parameter is missing.` | Provide a valid DataFrame or file path. |
| `KML-D002` | Dataset File Not Found | Dataset | ERROR | `Dataset file '{path}' was not found.` | Verify file path existence. |
| `KML-D003` | Empty Dataset | Dataset | ERROR | `Dataset is empty.` | Provide a dataset containing rows and columns. |
| `KML-D004` | Zero Rows Dataset | Dataset | ERROR | `Dataset contains zero rows.` | Verify dataset content before training. |
| `KML-D005` | Zero Columns Dataset | Dataset | ERROR | `Dataset contains zero columns.` | Verify feature columns exist. |
| `KML-D006` | Unsupported Format | Dataset | ERROR | `Unsupported file format '{format}'.` | Provide a CSV, Parquet, JSON, or Excel file. |
| `KML-D007` | Dataset Loading Failed | Dataset | ERROR | `Failed to load dataset: {reason}` | Ensure file permissions and format validity. |
| `KML-D008` | Corrupted Dataset | Dataset | ERROR | `Dataset file appears corrupted or unparseable.` | Re-download or export dataset file. |
| `KML-T001` | Target Not Specified | Target | ERROR | `Target column was not specified.` | Explicitly specify target='column_name'. |
| `KML-T002` | Target Column Not Found | Target | ERROR | `Target column '{target}' was not found.` | Choose one of the available columns. |
| `KML-T003` | Target Contains Only Missing Values | Target | ERROR | `Target column contains 100% missing values.` | Impute or select a valid target column. |
| `KML-T004` | Target Contains One Unique Value | Target | ERROR | `Target column contains only 1 unique class/value.` | Ensure target contains at least 2 distinct classes. |
| `KML-T005` | Invalid Regression Target | Target | ERROR | `Invalid regression target datatype or values.` | Ensure target is numeric for regression tasks. |
| `KML-T006` | Invalid Classification Target | Target | ERROR | `Invalid classification target distribution.` | Ensure target contains discrete class labels. |
| `KML-T007` | Identifier Used As Target | Target | ERROR | `Target column appears to be an identifier.` | Select a valid target feature instead of ID. |
| `KML-S001` | Duplicate Columns | Schema | ERROR | `Dataset contains duplicate column names: {cols}.` | Rename or remove duplicate columns. |
| `KML-S002` | Empty Column Names | Schema | ERROR | `Dataset contains empty or whitespace column names.` | Provide non-empty header names. |
| `KML-S003` | Unsupported Datatype | Schema | ERROR | `Unsupported datatype '{dtype}' in column '{col}'.` | Cast column to numeric or categorical type. |
| `KML-S004` | Mixed Datatype | Schema | ERROR | `Mixed datatypes detected in column '{col}'.` | Clean column so all values share one type. |
| `KML-S005` | Constant Feature | Schema | ERROR | `Feature column '{col}' is constant.` | Remove constant features prior to training. |
| `KML-S006` | Identifier Feature | Schema | ERROR | `Feature column '{col}' is a unique identifier.` | Drop identifier column to prevent overfitting. |
| `KML-S007` | High Cardinality | Schema | ERROR | `High cardinality in categorical feature '{col}'.` | Group rare categories or apply target encoding. |
| `KML-S008` | Infinite Numeric Values | Schema | ERROR | `Feature column '{col}' contains infinite values.` | Replace inf/nan values with finite numbers. |
| `KML-V001` | Validation Failed | Validation | ERROR | `Dataset validation failed with errors.` | Review and resolve dataset validation messages. |
| `KML-V002` | Validation Pipeline Interrupted | Validation | ERROR | `Validation pipeline halted by fail-fast check.` | Fix critical dataset errors. |
| `KML-V003` | Dataset Health Below Threshold | Validation | ERROR | `Dataset health score {score}/100 is below acceptable threshold.` | Clean missing data and outliers. |
| `KML-V004` | Critical Validation Error | Validation | ERROR | `Critical error during validation rule evaluation: {rule}.` | Check dataset integrity. |
| `KML-P001` | Encoding Failed | Preprocessing | ERROR | `Categorical encoding failed for feature '{col}'.` | Check categorical column formatting. |
| `KML-P002` | Scaling Failed | Preprocessing | ERROR | `Feature scaling failed.` | Check numerical features for extreme values or inf. |
| `KML-P003` | Feature Transformation Failed | Preprocessing | ERROR | `Feature transformation pipeline step failed.` | Verify preprocessor input schema. |
| `KML-P004` | Missing Preprocessing Pipeline | Preprocessing | ERROR | `Preprocessing pipeline is not fitted.` | Fit preprocessor on training data first. |
| `KML-M001` | Model Training Failed | Training | ERROR | `Training failed for model '{model}'.` | Check hyperparameters or data format. |
| `KML-M002` | Cross Validation Failed | Training | ERROR | `Cross-validation failed during model selection.` | Verify sample count and class balance. |
| `KML-M003` | Model Convergence Failed | Training | ERROR | `Model failed to converge.` | Increase max_iter or scale input features. |
| `KML-M004` | Model Selection Failed | Training | ERROR | `All candidate models failed during selection.` | Fix dataset errors before selection. |
| `KML-I001` | Prediction Failed | Prediction | ERROR | `Inference failed: {reason}` | Ensure input data matches training schema. |
| `KML-I002` | Feature Mismatch | Prediction | ERROR | `Feature mismatch during inference: expected {expected}, got {got}.` | Pass matching feature columns. |
| `KML-I003` | Model Not Loaded | Prediction | ERROR | `Model bundle is not loaded or fitted.` | Load a valid trained model bundle. |
| `KML-I004` | Invalid Inference Input | Prediction | ERROR | `Invalid inference input shape or data type.` | Pass DataFrame or dictionary matching schema. |
| `KML-DP001` | Package Generation Failed | Deployment | ERROR | `Failed to create deployment bundle.` | Check destination path permissions. |
| `KML-DP002` | ONNX Export Failed | Deployment | ERROR | `ONNX export failed.` | Ensure skl2onnx dependencies are installed. |
| `KML-DP003` | FastAPI Generation Failed | Deployment | ERROR | `FastAPI code generation failed.` | Check deployment configuration. |
| `KML-DP004` | Docker Export Failed | Deployment | ERROR | `Dockerfile export failed.` | Verify target export directory. |
| `KML-C001` | Invalid Command | CLI | ERROR | `Unrecognized CLI command '{command}'.` | Run 'kiteml --help' to see valid commands. |
| `KML-C002` | Missing CLI Argument | CLI | ERROR | `Missing required argument '{arg}'.` | Provide required argument. |
| `KML-C003` | Invalid Option | CLI | ERROR | `Invalid command option '{option}'.` | Check command line flag spelling. |
| `KML-C004` | CLI Execution Failed | CLI | ERROR | `CLI execution error: {details}` | Check terminal output for details. |
| `KML-CFG001` | Invalid Configuration | Configuration | ERROR | `Invalid setting '{setting}'.` | Check configuration dictionary key/value. |
| `KML-CFG002` | Missing Configuration | Configuration | ERROR | `Missing required configuration parameter.` | Provide required configuration. |
| `KML-CFG003` | Unsupported Version | Configuration | ERROR | `Unsupported configuration version '{version}'.` | Upgrade KiteML to latest version. |