#!/usr/bin/env python3
"""
build_docs.py — KiteML Documentation Builder

Builds and optionally serves the MkDocs documentation locally.

Usage:
  python scripts/build_docs.py          # Build only
  python scripts/build_docs.py --serve  # Build and serve locally
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Build KiteML documentation")
    parser.add_argument("--serve", action="store_true", help="Serve docs locally after building")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    parser.add_argument("--port", type=int, default=8000, help="Port for local server (default: 8000)")
    args = parser.parse_args()

    # Check mkdocs is installed
    try:
        import mkdocs  # noqa: F401
    except ImportError:
        print("ERROR: mkdocs is not installed.")
        print("  Install with: pip install -r requirements/docs.txt")
        sys.exit(1)

    if args.serve:
        print(f"\nServing docs at http://localhost:{args.port}")
        cmd = ["mkdocs", "serve", "--dev-addr", f"0.0.0.0:{args.port}"]
        if args.strict:
            cmd.append("--strict")
        subprocess.run(cmd, cwd=ROOT)
    else:
        print("\nBuilding documentation...")
        cmd = ["mkdocs", "build"]
        if args.strict:
            cmd.append("--strict")
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: Build failed:\n{result.stderr}")
            sys.exit(1)
        print("  Documentation built successfully!")
        print(f"  Output: {ROOT / 'site'}")


if __name__ == "__main__":
    main()
