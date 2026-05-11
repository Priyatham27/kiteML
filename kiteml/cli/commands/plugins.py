"""
commands/plugins.py — CLI command for managing KiteML plugins.
"""

from kiteml.cli.ui.colors import print_header, print_info, print_step


def setup_plugins_parser(subparsers):
    parser = subparsers.add_parser("plugins", help="Manage KiteML extensions and plugins")
    sub = parser.add_subparsers(dest="plugin_cmd")

    list_p = sub.add_parser("list", help="List installed plugins")
    list_p.set_defaults(func=run_plugins_list)

    parser.set_defaults(func=lambda args: parser.print_help() if not getattr(args, "plugin_cmd", None) else None)


def run_plugins_list(args):
    print_header("Installed KiteML Plugins")
    # For now, it's a mock since plugins architecture isn't fully implemented
    print_info("No plugins installed.")
    print_step("To install plugins, use pip install kiteml-<plugin-name>")
    return 0
