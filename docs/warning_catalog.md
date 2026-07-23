# KiteML Warning Catalog Reference

Complete reference of all standard KiteML warning codes (`KML-W-XXXNNN`).

| Warning Code | Name | Category | Severity | Message Template | Default Recommendation |
| ------------ | ---- | -------- | -------- | ---------------- | ---------------------- |
| `KML-W-D001` | High Missing Values | Dataset | MEDIUM | `High missing values in column '{col}' ({ratio:.0%}).` | Consider imputation or feature removal. |
| `KML-W-D002` | Duplicate Rows | Dataset | LOW | `Dataset contains {count} duplicate rows.` | Review and remove duplicate rows if unintentional. |
| `KML-W-S001` | Constant Feature | Schema | HIGH | `Feature '{col}' has 0 variance (constant).` | Remove constant feature prior to modeling. |
| `KML-W-S002` | High Cardinality | Schema | MEDIUM | `Feature '{col}' has high cardinality ({count} unique values).` | Apply target encoding or frequency encoding. |
| `KML-W-V001` | Dataset Health Below Recommendation | Validation | HIGH | `Dataset health score ({score}/100) is below recommended threshold.` | Inspect quality report and address high severity issues. |
| `KML-W-M001` | Slow Convergence | Training | LOW | `Model '{model}' convergence was slow.` | Increase max_iter or scale numerical features. |
| `KML-W-I001` | Unknown Inference Category | Prediction | LOW | `Unknown category '{val}' in column '{col}' during inference.` | Value was imputed with default category. |
| `KML-W-DP001` | Large Deployment Bundle | Deployment | MEDIUM | `Deployment bundle size ({size_mb:.1f} MB) is large.` | Consider model pruning or compression. |