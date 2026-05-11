"""
commands/playground.py — CLI command for instant experimentation.
"""

import os
import urllib.request

from kiteml.cli.ui.colors import print_error, print_header, print_info, print_step, print_success

# Curated datasets mapping
DATASETS = {
    "titanic": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "customer_churn": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
}


def setup_playground_parser(subparsers):
    parser = subparsers.add_parser("playground", help="Download a sample dataset and scaffold a notebook")
    parser.add_argument("dataset", type=str, choices=list(DATASETS.keys()), help="Name of the sample dataset")
    parser.set_defaults(func=run_playground)


def run_playground(args):
    dataset_name = args.dataset
    url = DATASETS[dataset_name]

    print_header(f"🪁 KiteML Playground: {dataset_name}")

    os.makedirs("data", exist_ok=True)
    os.makedirs("notebooks", exist_ok=True)

    csv_path = f"data/{dataset_name}.csv"
    print_info(f"Downloading {dataset_name} dataset...")

    try:
        urllib.request.urlretrieve(url, csv_path)
        print_step(f"Saved dataset to {csv_path}")
    except Exception as e:
        print_error(f"Failed to download dataset: {e}")
        return 1

    # Generate a starter notebook (as a simple Python script for now if jupyter not installed)
    notebook_path = f"notebooks/{dataset_name}_starter.py"
    with open(notebook_path, "w") as f:
        f.write(f'"""\nStarter script for {dataset_name}\n"""\n')
        f.write("import pandas as pd\nfrom kiteml.core import train\n\n")
        f.write(f'df = pd.read_csv("../{csv_path}")\n')
        f.write("print(df.head())\n\n")
        f.write("# Uncomment to train:\n")
        f.write('# result = train(df, target="YOUR_TARGET_COLUMN")\n')
        f.write("# result.generate_dashboard()\n")

    print_step(f"Generated starter script at {notebook_path}")
    print_success("Playground ready! Run the script or open in Jupyter to start exploring.")
    return 0
