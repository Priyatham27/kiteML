"""
cli.py - Command-line interface for KiteML.

Usage:
    kiteml train data.csv --target column_name
"""

import argparse
import sys

from kiteml.core import train


def main():
    parser = argparse.ArgumentParser(
        prog="kiteml",
        description="KiteML - Train ML models with a single command.",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model on a dataset.")
    train_parser.add_argument("data", type=str, help="Path to dataset (CSV, Excel, JSON, Parquet).")
    train_parser.add_argument("--target", "-t", required=True, help="Target column name.")
    train_parser.add_argument(
        "--type", "-T", choices=["classification", "regression"], default=None, help="Problem type."
    )
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2).")
    train_parser.add_argument("--no-scale", action="store_true", help="Disable feature scaling.")
    train_parser.add_argument("--save", "-s", type=str, default=None, help="Path to save the trained model.")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        result = train(
            data=args.data,
            target=args.target,
            problem_type=args.type,
            test_size=args.test_size,
            scale=not args.no_scale,
            random_state=args.seed,
            verbose=True,
        )

        if args.save:
            result.save_model(args.save)


if __name__ == "__main__":
    main()
