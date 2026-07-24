# KiteML Documentation Style Guide (`docs-spec/style-guide.md`)

This guide defines strict Markdown, code snippet, docstring, and visualization standards for all KiteML documentation.

---

## 1. Writing Principles & Voice

- **Tone**: Clear, authoritative, developer-friendly, and precise.
- **Perspective**: Use active voice and second-person perspective ("you can configure...", "train your model with...").
- **Clarity**: Avoid fluff or vague marketing claims. Back up statements with concrete code snippets or metric descriptions.

---

## 2. Markdown Formatting Standards

### Headings
- Use a single `#` (H1) heading per document corresponding to the page title.
- Nest subheadings strictly in order (`##`, `###`, `####`). Do not skip levels.

### Code Blocks & Annotations
- Always specify explicit language identifiers: `python`, `bash`, `yaml`, `json`, `text`, or `mermaid`.
- For multi-tab instructions (CLI vs Python API), use MkDocs tabbed blocks:

```markdown
=== "Python API"

    ```python
    from kiteml import train

    result = train("data.csv", target="label")
    ```

=== "CLI"

    ```bash
    kiteml train data.csv --target label
    ```
```

### Admonitions (Callouts)
Use standard MkDocs Material admonition syntax for notes, tips, warnings, and dangerous operations:

- `!!! note "Title"`: Background context, default behavior details, or technical facts.
- `!!! tip "Title"`: Performance advice, optimization tricks, or workflow shortcuts.
- `!!! warning "Title"`: Potential pitfalls, schema mismatches, or deprecated features.
- `!!! danger "Title"`: Breaking changes, data loss risks, or security considerations.

---

## 3. Python Docstring Format

All Python source code docstrings must strictly adhere to the **Google Python Docstring Style** to ensure seamless parsing by `mkdocstrings[python]`:

```python
def train(
    data: str | pd.DataFrame,
    target: str,
    problem_type: str | None = None,
    test_size: float = 0.2,
    scale: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> Result:
    """Train an optimal machine learning model with automated preprocessing and selection.

    Args:
        data: Path to a CSV/JSON/Parquet file or an existing pandas DataFrame.
        target: Name of the target column to predict.
        problem_type: Optional explicit problem type ('classification' or 'regression').
            If None, inferred automatically from target distribution.
        test_size: Proportion of dataset allocated to validation (0.0 to 1.0).
        scale: If True, applies standard scaling to numerical features.
        random_state: Seed for reproducible train/test splitting.
        verbose: If True, outputs diagnostic execution logs to standard output.

    Returns:
        Result: Container instance holding the best model, metrics, and report.

    Raises:
        TargetNotFoundError: If `target` is not present in `data`.
        EmptyDatasetError: If dataset contains zero rows.

    Example:
        >>> from kiteml import train
        >>> res = train("housing.csv", target="price", problem_type="regression")
        >>> print(res.summary())
    """
```

---

## 4. Diagram Standards (Mermaid)

- Use ```mermaid custom fences for all system architecture, execution flow, and sequence diagrams.
- Quote node labels containing parentheses or special characters (`id["Label (Extra)"]`).
- Avoid raw HTML formatting inside Mermaid node labels.
