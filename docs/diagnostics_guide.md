# KiteML Diagnostics Guide

Execution Diagnostics provide a complete summary of training execution, validation health, warnings, and suggestions.

## Usage

```python
result = kiteml.train(df, target="price")
print(result.diagnostics())
```

## Output Format

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🪁 KiteML Diagnostics

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Status             SUCCESS
  Errors             0
  Warnings           1
  Suggestions        3
  Validation         Passed
  Training           Completed
  Execution Time     1.24 sec
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
