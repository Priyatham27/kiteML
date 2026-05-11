#!/usr/bin/env python3
"""
verify_install.py — KiteML Installation Verification

Verifies that KiteML is correctly installed and functional.
Designed to run in a fresh virtual environment after `pip install kiteml`.

Usage:
  python scripts/verify_install.py
"""

import importlib
import subprocess
import sys


def check(label: str, condition: bool) -> bool:
    """Print a pass/fail check result."""
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def main() -> int:
    print("\n" + "=" * 50)
    print("  KiteML Installation Verification")
    print("=" * 50 + "\n")

    results = []

    # 1. Import check
    try:
        import kiteml
        results.append(check("Import kiteml", True))
    except ImportError as e:
        results.append(check(f"Import kiteml: {e}", False))
        print("\n  CRITICAL: kiteml cannot be imported. Aborting.")
        return 1

    # 2. Version check
    version = getattr(kiteml, "__version__", None)
    results.append(check(f"Version: {version}", version is not None))

    # 3. Core API imports
    core_imports = [
        ("kiteml.core", "train"),
        ("kiteml.config", "DEFAULT_RANDOM_STATE"),
        ("kiteml.output.result", "Result"),
        ("kiteml.preprocessing.pipeline", None),
        ("kiteml.evaluation.metrics", None),
        ("kiteml.cli.main", "main"),
    ]

    for module_name, attr in core_imports:
        try:
            mod = importlib.import_module(module_name)
            if attr:
                assert hasattr(mod, attr), f"Missing attribute '{attr}'"
            results.append(check(f"Import {module_name}" + (f".{attr}" if attr else ""), True))
        except Exception as e:
            results.append(check(f"Import {module_name}: {e}", False))

    # 4. Optional module imports (non-fatal)
    optional_modules = [
        "kiteml.intelligence.explainability",
        "kiteml.deployment.onnx_export",
        "kiteml.deployment.docker_export",
        "kiteml.monitoring",
        "kiteml.experiments",
        "kiteml.plugins.sdk",
        "kiteml.governance",
    ]

    for module_name in optional_modules:
        try:
            importlib.import_module(module_name)
            results.append(check(f"Import {module_name}", True))
        except ImportError:
            results.append(check(f"Import {module_name} (optional, skipped)", True))
        except Exception as e:
            results.append(check(f"Import {module_name}: {e}", False))

    # 5. CLI entrypoint check
    try:
        result = subprocess.run(
            [sys.executable, "-m", "kiteml.cli.main", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        results.append(check(f"CLI --version: {result.stdout.strip()}", result.returncode == 0))
    except Exception as e:
        results.append(check(f"CLI --version: {e}", False))

    # 6. Dependencies check
    required_deps = ["pandas", "scikit-learn", "numpy", "joblib"]
    for dep in required_deps:
        dep_import = dep.replace("-", "_").replace(" ", "_")
        if dep_import == "scikit_learn":
            dep_import = "sklearn"
        try:
            mod = importlib.import_module(dep_import)
            ver = getattr(mod, "__version__", "unknown")
            results.append(check(f"Dependency {dep} ({ver})", True))
        except ImportError:
            results.append(check(f"Dependency {dep} MISSING", False))

    # Summary
    passed = sum(results)
    total = len(results)
    failed = total - passed

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("\n  KiteML installation is VERIFIED.\n")
        return 0
    else:
        print(f"\n  WARNING: {failed} check(s) failed.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
