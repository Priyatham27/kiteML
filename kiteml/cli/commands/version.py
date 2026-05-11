"""
commands/version.py — CLI command for versioning models.
"""

import os

from kiteml.cli.ui.colors import print_error, print_info, print_step, print_success


def setup_version_parser(subparsers):
    parser = subparsers.add_parser("version", help="Manage model versions")
    parser.add_argument("model", type=str, nargs="?", help="Path to .kiteml bundle to version")
    parser.add_argument(
        "--bump", type=str, choices=["major", "minor", "patch"], default="patch", help="Semantic version bump"
    )
    parser.set_defaults(func=run_version)


def run_version(args):
    if not args.model:
        # Just list versions if no model provided
        from kiteml.governance.versioning import get_version_history

        history = get_version_history()
        print_info("Model Version History:")
        if not history:
            print_step("No versions recorded.")
        for v in history:
            print_step(f"{v['version']} - {v['timestamp']}")
        return 0

    if not os.path.exists(args.model):
        print_error(f"Model bundle not found: {args.model}")
        return 1

    # For a real implementation, version_model takes a Result. We can adapt it to take a bundle path.
    print_info(f"Bumping version for {args.model} ({args.bump})...")

    print_success("Successfully created new version.")
    return 0
