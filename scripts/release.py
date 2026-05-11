#!/usr/bin/env python3
"""
release.py — KiteML Release Automation Script

Automates the full release workflow:
  1. Bump version (major/minor/patch)
  2. Update __init__.py and pyproject.toml
  3. Update CHANGELOG.md with new version header
  4. Build sdist + wheel
  5. Validate with twine
  6. Create git commit + tag

Usage:
  python scripts/release.py patch     # 1.0.0 -> 1.0.1
  python scripts/release.py minor     # 1.0.0 -> 1.1.0
  python scripts/release.py major     # 1.0.0 -> 2.0.0
  python scripts/release.py 1.2.3     # Explicit version
"""

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT_FILE = ROOT / "kiteml" / "__init__.py"
PYPROJECT_FILE = ROOT / "pyproject.toml"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"


def get_current_version() -> str:
    """Read the current version from kiteml/__init__.py."""
    content = INIT_FILE.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    if not match:
        print("ERROR: Could not find __version__ in __init__.py")
        sys.exit(1)
    return match.group(1)


def bump_version(current: str, bump_type: str) -> str:
    """Calculate the next version based on bump type."""
    parts = current.split(".")
    if len(parts) != 3:
        print(f"ERROR: Version '{current}' is not valid semver (x.y.z)")
        sys.exit(1)

    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        # Explicit version string
        if not re.match(r"^\d+\.\d+\.\d+$", bump_type):
            print(f"ERROR: '{bump_type}' is not a valid version or bump type")
            sys.exit(1)
        return bump_type


def update_init(new_version: str) -> None:
    """Update __version__ in kiteml/__init__.py."""
    content = INIT_FILE.read_text(encoding="utf-8")
    updated = re.sub(
        r'__version__\s*=\s*"[^"]+"',
        f'__version__ = "{new_version}"',
        content,
    )
    INIT_FILE.write_text(updated, encoding="utf-8")
    print(f"  Updated {INIT_FILE.relative_to(ROOT)}")


def update_pyproject(new_version: str) -> None:
    """Update version in pyproject.toml."""
    content = PYPROJECT_FILE.read_text(encoding="utf-8")
    updated = re.sub(
        r'^version\s*=\s*"[^"]+"',
        f'version = "{new_version}"',
        content,
        flags=re.MULTILINE,
    )
    PYPROJECT_FILE.write_text(updated, encoding="utf-8")
    print(f"  Updated {PYPROJECT_FILE.relative_to(ROOT)}")


def update_changelog(new_version: str) -> None:
    """Add a new version header to CHANGELOG.md."""
    if not CHANGELOG_FILE.exists():
        print("  CHANGELOG.md not found, skipping")
        return

    content = CHANGELOG_FILE.read_text(encoding="utf-8")
    today = date.today().isoformat()
    new_entry = f"## [{new_version}] -- {today}\n\n### Added\n\n- _Fill in release notes here_\n\n---\n\n"

    # Insert after the [Unreleased] section
    marker = "## [Unreleased]"
    if marker in content:
        content = content.replace(
            marker,
            f"{marker}\n\n_Nothing yet. Stay tuned!_\n\n---\n\n{new_entry}",
        )
    else:
        # Insert after the first ---
        idx = content.find("---", content.find("---") + 1)
        if idx != -1:
            content = content[:idx] + f"---\n\n{new_entry}" + content[idx + 3:]

    CHANGELOG_FILE.write_text(content, encoding="utf-8")
    print(f"  Updated {CHANGELOG_FILE.relative_to(ROOT)}")


def build_package() -> None:
    """Build sdist and wheel distributions."""
    print("\nBuilding distributions...")
    dist_dir = ROOT / "dist"
    if dist_dir.exists():
        import shutil
        shutil.rmtree(dist_dir)

    result = subprocess.run(
        [sys.executable, "-m", "build"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Build failed:\n{result.stderr}")
        sys.exit(1)
    print("  Build successful!")

    # List built files
    for f in sorted(dist_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name}  ({size_kb:.1f} KB)")


def validate_package() -> None:
    """Run twine check on built distributions."""
    print("\nValidating package...")
    result = subprocess.run(
        [sys.executable, "-m", "twine", "check", "dist/*"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        shell=True,
    )
    if result.returncode != 0:
        print(f"WARNING: twine check reported issues:\n{result.stdout}\n{result.stderr}")
    else:
        print("  Validation passed!")


def create_git_tag(version: str) -> None:
    """Create a git commit and tag for the release."""
    print(f"\nCreating git tag v{version}...")
    try:
        subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"release: v{version}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "tag", "-a", f"v{version}", "-m", f"Release v{version}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        print(f"  Created tag v{version}")
        print("  Push with: git push origin main --tags")
    except subprocess.CalledProcessError as e:
        print(f"  Git operations failed (non-fatal): {e}")


def main():
    parser = argparse.ArgumentParser(description="KiteML Release Automation")
    parser.add_argument(
        "bump",
        choices=["major", "minor", "patch"],
        nargs="?",
        default=None,
        help="Version bump type (major/minor/patch) or explicit version",
    )
    parser.add_argument("--version", type=str, help="Explicit version to set")
    parser.add_argument("--no-build", action="store_true", help="Skip building distributions")
    parser.add_argument("--no-tag", action="store_true", help="Skip git tag creation")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without making changes")

    args = parser.parse_args()

    bump_type = args.version or args.bump
    if not bump_type:
        parser.print_help()
        sys.exit(1)

    current = get_current_version()
    new_version = bump_version(current, bump_type)

    print(f"\n{'=' * 50}")
    print(f"  KiteML Release: {current} -> {new_version}")
    print(f"{'=' * 50}")

    if args.dry_run:
        print("\n  [DRY RUN] No changes will be made.")
        return

    print("\nUpdating version files...")
    update_init(new_version)
    update_pyproject(new_version)
    update_changelog(new_version)

    if not args.no_build:
        build_package()
        validate_package()

    if not args.no_tag:
        create_git_tag(new_version)

    print(f"\n{'=' * 50}")
    print(f"  Release v{new_version} prepared!")
    print(f"{'=' * 50}")
    print("\n  Next steps:")
    print("    1. Review CHANGELOG.md and fill in release notes")
    print("    2. Push: git push origin main --tags")
    print("    3. GitHub Actions will handle PyPI publishing")
    print(f"    4. Verify: pip install kiteml=={new_version}")


if __name__ == "__main__":
    main()
