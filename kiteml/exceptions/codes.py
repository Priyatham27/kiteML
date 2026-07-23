"""
codes.py — Error code constants catalog for KiteML exceptions.
"""

# Base category default codes
KML_ERR_GENERIC = "KML-E000"
KML_ERR_DATASET = "KML-D000"
KML_ERR_TARGET = "KML-T000"
KML_ERR_SCHEMA = "KML-S000"
KML_ERR_VALIDATION = "KML-V000"
KML_ERR_PREPROCESSING = "KML-P000"
KML_ERR_TRAINING = "KML-M000"
KML_ERR_PREDICTION = "KML-I000"
KML_ERR_DEPLOYMENT = "KML-DP000"
KML_ERR_CLI = "KML-C000"
KML_ERR_CONFIG = "KML-CFG000"

# Dataset (KML-D*)
KML_D001 = "KML-D001"  # Dataset not provided
KML_D002 = "KML-D002"  # Dataset file not found
KML_D003 = "KML-D003"  # Empty dataset
KML_D004 = "KML-D004"  # Dataset contains zero rows
KML_D005 = "KML-D005"  # Dataset contains zero columns
KML_D006 = "KML-D006"  # Unsupported dataset format
KML_D007 = "KML-D007"  # Dataset loading failed
KML_D008 = "KML-D008"  # Corrupted dataset

# Target (KML-T*)
KML_T001 = "KML-T001"  # Target not specified
KML_T002 = "KML-T002"  # Target column not found
KML_T003 = "KML-T003"  # Target contains only missing values
KML_T004 = "KML-T004"  # Target contains one unique value
KML_T005 = "KML-T005"  # Invalid regression target
KML_T006 = "KML-T006"  # Invalid classification target
KML_T007 = "KML-T007"  # Identifier used as target

# Schema (KML-S*)
KML_S001 = "KML-S001"  # Duplicate columns
KML_S002 = "KML-S002"  # Empty column names
KML_S003 = "KML-S003"  # Unsupported datatype
KML_S004 = "KML-S004"  # Mixed datatype
KML_S005 = "KML-S005"  # Constant feature
KML_S006 = "KML-S006"  # Identifier feature
KML_S007 = "KML-S007"  # High cardinality
KML_S008 = "KML-S008"  # Infinite numeric values

# Validation (KML-V*)
KML_V001 = "KML-V001"  # Validation failed
KML_V002 = "KML-V002"  # Validation pipeline interrupted
KML_V003 = "KML-V003"  # Dataset health below threshold
KML_V004 = "KML-V004"  # Critical validation error

# Preprocessing (KML-P*)
KML_P001 = "KML-P001"  # Encoding failed
KML_P002 = "KML-P002"  # Scaling failed
KML_P003 = "KML-P003"  # Feature transformation failed
KML_P004 = "KML-P004"  # Missing preprocessing pipeline

# Training / Model (KML-M*)
KML_M001 = "KML-M001"  # Model training failed
KML_M002 = "KML-M002"  # Cross-validation failed
KML_M003 = "KML-M003"  # Model convergence failed
KML_M004 = "KML-M004"  # Model selection failed

# Prediction / Inference (KML-I*)
KML_I001 = "KML-I001"  # Prediction failed
KML_I002 = "KML-I002"  # Feature mismatch
KML_I003 = "KML-I003"  # Model not loaded
KML_I004 = "KML-I004"  # Invalid inference input

# Deployment (KML-DP*)
KML_DP001 = "KML-DP001"  # Package generation failed
KML_DP002 = "KML-DP002"  # ONNX export failed
KML_DP003 = "KML-DP003"  # FastAPI generation failed
KML_DP004 = "KML-DP004"  # Docker export failed

# CLI (KML-C*)
KML_C001 = "KML-C001"  # Invalid command
KML_C002 = "KML-C002"  # Missing CLI argument
KML_C003 = "KML-C003"  # Invalid option
KML_C004 = "KML-C004"  # CLI execution failed

# Configuration (KML-CFG*)
KML_CFG001 = "KML-CFG001"  # Invalid configuration
KML_CFG002 = "KML-CFG002"  # Missing configuration
KML_CFG003 = "KML-CFG003"  # Unsupported version
