"""
commands/experiment.py — CLI command for tracking experiments.
"""
import argparse

from kiteml.cli.ui.colors import print_step, print_info, print_error, print_header


def setup_experiment_parser(subparsers):
    parser = subparsers.add_parser("experiments", aliases=["exp"], help="Manage KiteML experiments")
    exp_subparsers = parser.add_subparsers(dest="exp_command")
    
    # List
    list_p = exp_subparsers.add_parser("list", help="List all runs in an experiment")
    list_p.add_argument("name", type=str, default="default", nargs="?", help="Experiment name")
    list_p.set_defaults(func=run_experiment_list)
    
    parser.set_defaults(func=lambda args: parser.print_help() if not getattr(args, "exp_command", None) else None)


def run_experiment_list(args):
    from kiteml.experiments.tracker import list_runs
    
    runs = list_runs(args.name)
    print_header(f"Experiment Runs: {args.name}")
    
    if not runs:
        print_info("No runs found.")
        return 0
        
    for r in runs:
        score = r.metrics.get('score', r.metrics.get('accuracy', r.metrics.get('r2_score', 'N/A')))
        if isinstance(score, float):
            score = f"{score:.4f}"
            
        print_step(f"Run {r.run_id[:8]} | {r.model_name} | Score: {score}")
        
    return 0
