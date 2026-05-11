"""
main.py — Main entry point for the KiteML CLI.
"""

import sys

from kiteml.cli.parser import build_parser
from kiteml.cli.ui.colors import print_banner, print_error


def main():
    """Main CLI entry function."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    parser = build_parser()

    if len(sys.argv) == 1:
        print_banner()
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    try:
        # Print banner for commands that aren't purely quiet
        if args.command not in ("completion", "version"):
            print_banner()

        sys.exit(args.func(args))
    except KeyboardInterrupt:
        print("\n")
        print_error("Operation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        print_error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
