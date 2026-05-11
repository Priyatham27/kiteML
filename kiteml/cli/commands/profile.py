"""
commands/profile.py — CLI command for dataset intelligence.
"""
import argparse
import os

from kiteml.cli.ui.colors import print_step, print_info, print_error, print_success, print_header
import pandas as pd


def setup_profile_parser(subparsers):
    parser = subparsers.add_parser("profile", help="Generate intelligence profile for a dataset")
    parser.add_argument("data", type=str, help="Path to input data (CSV/Parquet)")
    parser.add_argument("--target", type=str, help="Target column (optional)")
    parser.add_argument("--html", type=str, help="Path to export HTML profile")
    parser.set_defaults(func=run_profile)


def run_profile(args):
    if not os.path.exists(args.data):
        print_error(f"Input data not found: {args.data}")
        return 1

    print_info(f"Loading data from {args.data}...")
    try:
        df = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_parquet(args.data)
        from kiteml.profiling.inspector import DatasetInspector
        from kiteml.profiling.html_export import export_html
        
        inspector = DatasetInspector(df, target=args.target)
        profile = inspector.profile()
        
        print_header("Dataset Profile")
        print_step(f"Rows: {profile.n_rows:,} | Columns: {profile.n_cols:,}")
        if args.target:
            print_step(f"Target: '{args.target}' (Type: {profile.target_type})")
        print_step(f"Quality Score: {profile.quality_score:.1f}/100")
        
        if profile.warnings:
            print_header("Data Quality Warnings")
            for w in profile.warnings:
                from kiteml.cli.ui.colors import print_warning
                print_warning(w)
                
        if args.html:
            export_html(profile, args.html)
            print_success(f"HTML profile exported to {args.html}")
            
    except Exception as e:
        print_error(f"Profiling failed: {e}")
        return 1

    return 0
