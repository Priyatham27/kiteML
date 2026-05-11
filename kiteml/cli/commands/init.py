"""
commands/init.py — CLI command for initializing a new KiteML project.
"""

import os

from kiteml.cli.ui.colors import print_error, print_header, print_step, print_success


def setup_init_parser(subparsers):
    parser = subparsers.add_parser("init", help="Initialize a new KiteML project directory")
    parser.add_argument("project_name", type=str, help="Name of the new project directory")
    parser.set_defaults(func=run_init)


def run_init(args):
    project_dir = args.project_name
    if os.path.exists(project_dir):
        print_error(f"Directory '{project_dir}' already exists.")
        return 1

    print_header(f"Initializing KiteML Project: {project_dir}")

    # Create directories
    dirs = ["data", "notebooks", "experiments", "configs", "models"]

    for d in dirs:
        os.makedirs(os.path.join(project_dir, d), exist_ok=True)

    # Create template train.py
    train_py = os.path.join(project_dir, "train.py")
    with open(train_py, "w") as f:
        f.write(
            '"""\nKiteML Training Script\n"""\n\nimport pandas as pd\nfrom kiteml.core import train\n\n# Load your data\n# df = pd.read_csv("data/dataset.csv")\n\n# Train model\n# result = train(df, target="target_column")\n# result.package("models/model.kiteml")\n'
        )

    print_step("Created project structure")
    print_step("Generated train.py template")

    print_success(f"Project '{project_dir}' ready! Run `cd {project_dir}` to begin.")
    return 0
