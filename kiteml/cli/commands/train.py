"""
commands/train.py — CLI command for training models.
"""

import os

from kiteml.cli.ui.colors import print_error, print_info, print_step, print_success
from kiteml.core import train


def setup_train_parser(subparsers):
    parser = subparsers.add_parser("train", aliases=["tr"], help="Train a new ML model from a dataset")
    parser.add_argument("data", type=str, help="Path to the training dataset (CSV/Parquet)")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument(
        "--problem-type",
        type=str,
        choices=["classification", "regression"],
        help="Problem type (auto-detected if omitted)",
    )
    parser.add_argument("--export", type=str, help="Path to export the .kiteml bundle (e.g., model.kiteml)")
    parser.add_argument("--dashboard", action="store_true", help="Generate an HTML dashboard after training")
    parser.add_argument("--experiment", type=str, help="Track this run in a named experiment")
    parser.set_defaults(func=run_train)


def run_train(args):
    if not os.path.exists(args.data):
        print_error(f"Dataset not found: {args.data}")
        return 1

    print_info(f"Loading data from {args.data}...")

    try:
        result = train(data=args.data, target=args.target, problem_type=args.problem_type, verbose=False)
    except Exception as e:
        print_error(f"Training failed: {e}")
        return 1

    print_step(f"Training completed. Winner: {result.model_name}")
    print_step(f"Problem Type: {result.problem_type}")

    # Print metrics
    metrics = result.metrics.__dict__ if hasattr(result.metrics, "__dict__") else result.metrics
    score = getattr(result, "score", None)
    if score is not None:
        print_info(f"Primary Score: {score:.4f}")

    if args.export:
        print_info(f"Packaging model to {args.export}...")
        result.package(args.export)
        print_step("Model packaged successfully.")

    if args.dashboard:
        dash_path = f"{result.model_name}_dashboard.html"
        result.generate_dashboard(dash_path)
        print_step(f"Dashboard generated: {dash_path}")

    if args.experiment:
        result.experiment(experiment_name=args.experiment)
        print_step(f"Run tracked in experiment: {args.experiment}")

    print_success("Training pipeline finished!")
    return 0
