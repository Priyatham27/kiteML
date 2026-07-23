"""
docs_generator.py — Automated documentation generator for KiteML error catalog, warnings, and suggestions.
"""

from pathlib import Path
from typing import Any

from kiteml.exceptions import ErrorCatalog
from kiteml.warnings import global_warning_registry


class DocGenerator:
    """
    Automated Documentation Generator for KiteML Developer Experience subsystem.
    Dynamically generates Markdown tables and guides directly from the code catalog.
    """

    def generate_error_catalog_md(self) -> str:
        """Generate markdown reference table for Error Catalog."""
        defs = ErrorCatalog.all_definitions()

        lines = [
            "# KiteML Error Catalog Reference",
            "",
            "Complete reference of all standard KiteML error codes (`KML-XXXNNN`).",
            "",
            "| Error Code | Name | Category | Severity | Message Template | Default Suggestion |",
            "| ---------- | ---- | -------- | -------- | ---------------- | ------------------ |",
        ]

        for d in defs:
            msg = d.message_template.replace("|", "\\|")
            sug = (d.default_suggestion or "").replace("|", "\\|")
            sev = d.severity.value if hasattr(d.severity, "value") else str(d.severity)
            lines.append(f"| `{d.code}` | {d.name} | {d.category} | {sev} | `{msg}` | {sug} |")

        return "\n".join(lines)

    def generate_warning_catalog_md(self) -> str:
        """Generate markdown reference table for Warning Catalog."""
        defs = global_warning_registry.all_definitions()

        lines = [
            "# KiteML Warning Catalog Reference",
            "",
            "Complete reference of all standard KiteML warning codes (`KML-W-XXXNNN`).",
            "",
            "| Warning Code | Name | Category | Severity | Message Template | Default Recommendation |",
            "| ------------ | ---- | -------- | -------- | ---------------- | ---------------------- |",
        ]

        for d in defs:
            msg = d.message_template.replace("|", "\\|")
            rec = (d.default_recommendation or "").replace("|", "\\|")
            sev = d.severity.value if hasattr(d.severity, "value") else str(d.severity)
            lines.append(f"| `{d.code}` | {d.name} | {d.category} | {sev} | `{msg}` | {rec} |")

        return "\n".join(lines)

    def generate_all_docs(self, output_dir: str = "docs") -> dict[str, str]:
        """Generate and save all documentation files to output_dir."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        docs = {
            "error_catalog.md": self.generate_error_catalog_md(),
            "warning_catalog.md": self.generate_warning_catalog_md(),
        }

        for filename, content in docs.items():
            file_path = out_path / filename
            file_path.write_text(content, encoding="utf-8")

        return docs
