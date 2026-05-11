"""
parser.py — Main CLI parser configuration for KiteML.
"""

import argparse

from kiteml.cli.commands.serve import setup_serve_parser
from kiteml.cli.commands.train import setup_train_parser


def build_parser() -> argparse.ArgumentParser:
    """Build the main argparse parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="kiteml",
        description="KiteML — Intelligent Machine Learning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--version", action="version", version="KiteML 0.1.0")

    subparsers = parser.add_subparsers(title="commands", dest="command", help="Available commands")

    from kiteml.cli.commands.benchmark import setup_benchmark_parser
    from kiteml.cli.commands.dashboard import setup_dashboard_parser
    from kiteml.cli.commands.doctor import setup_doctor_parser
    from kiteml.cli.commands.experiment import setup_experiment_parser
    from kiteml.cli.commands.export import setup_export_parser
    from kiteml.cli.commands.init import setup_init_parser
    from kiteml.cli.commands.monitor import setup_monitor_parser
    from kiteml.cli.commands.playground import setup_playground_parser
    from kiteml.cli.commands.plugins import setup_plugins_parser
    from kiteml.cli.commands.predict import setup_predict_parser
    from kiteml.cli.commands.profile import setup_profile_parser
    from kiteml.cli.commands.version import setup_version_parser

    # Register commands
    setup_train_parser(subparsers)
    setup_serve_parser(subparsers)
    setup_predict_parser(subparsers)
    setup_profile_parser(subparsers)
    setup_doctor_parser(subparsers)
    setup_monitor_parser(subparsers)
    setup_export_parser(subparsers)
    setup_init_parser(subparsers)
    setup_experiment_parser(subparsers)
    setup_version_parser(subparsers)
    setup_benchmark_parser(subparsers)
    setup_dashboard_parser(subparsers)
    setup_plugins_parser(subparsers)
    setup_playground_parser(subparsers)

    return parser
