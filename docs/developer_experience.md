# KiteML Developer Experience Guide

Every KiteML error, warning, and recommendation is designed to teach the user how to fix the problem.

## Architecture

```text
User API / kiteml.train()
          │
          ▼
     DXPipeline
          │
 ┌────────┼────────┬────────┐
 ▼        ▼        ▼        ▼
Context Error   Warning Suggestion
Builder Manager Manager Manager
 │        │        │        │
 └────────┴────────┴────────┘
          │
          ▼
     Diagnostics
```

## Subsystems

1. **Exceptions Framework (`kiteml.exceptions`)**:
   - Structured exception classes (`DatasetError`, `TargetError`, `SchemaError`, etc.) inheriting from `KiteMLError`.
   - `KML-XXXNNN` error codes with catalog metadata.
   - Multi-format error renderers (Terminal, Text, JSON, HTML, Markdown).

2. **Warning Framework (`kiteml.warnings`)**:
   - Structured `KiteMLWarning` classes.
   - `KML-W-XXXNNN` warning catalog with severity levels (`INFO`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
   - Configurable `WarningPolicy` (`ignore`, `info`, `warn`, `error` escalation).

3. **Context-Aware Suggestions Engine (`kiteml.suggestions`)**:
   - Fuzzy string matching (`match_column_name`) for column typos (`prcie` -> `Price`).
   - Domain providers for Target, Schema, Validation, Training, Deployment, and Performance guidance.
   - Explainable suggestions with a `Why?` section detailing rationale.

4. **DX Pipeline & Diagnostics (`kiteml.pipeline`)**:
   - `DXPipeline` orchestrating context, exceptions, warnings, and suggestions.
   - `Result.diagnostics()` summary box for execution feedback.
