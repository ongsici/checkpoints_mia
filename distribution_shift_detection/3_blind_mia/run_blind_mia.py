import os
import pickle
import pandas as pd
from bag_of_words import bag_of_words_basic
from greedy_selection import greedy_selection_basic
from date_detection import date_detection_basic
from contextlib import redirect_stdout
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--attack", choices=["bow", "greedy", "date"], default="bow")
parser.add_argument("--olmo_subset", choices=["wiki", "stack", "pes2o"], default="wiki")
parser.add_argument("--dataset_type", choices=["olmo"], default="olmo")
args = parser.parse_args()

# Configuration
olmo_subset = args.olmo_subset
dataset_type = args.dataset_type
base_dataset_dir = f"../data/{olmo_subset}"
output_dir = f"bow_results/{olmo_subset}" if args.attack == "bow" else f"greedy_results/{olmo_subset}" if args.attack == "greedy" else f"date_results/{olmo_subset}"
os.makedirs(output_dir, exist_ok=True)

# Sequence lengths
seq_lens = [64, 128, 256, 512, 1024, 2048]

for seq_len in seq_lens:
    dataset_path = f"{base_dataset_dir}/{olmo_subset}_seqlen{seq_len}.pkl"
    output_path = f"{output_dir}/{olmo_subset}_seqlen{seq_len}_results.txt"
    
    # Skip if dataset doesn't exist
    if not os.path.exists(dataset_path):
        print(f"Skipping {dataset_path} (not found)")
        continue
    
    with open(dataset_path, "rb") as f:
        df = pickle.load(f)

    X = df['truncated_text'].astype(str).tolist()
    y = df['label'].astype(int).tolist()
    members = df[df['label'] == 1]['truncated_text'].astype(str).tolist()
    nonmembers = df[df['label'] == 0]['truncated_text'].astype(str).tolist()

    dataset_name = f"pile_{dataset_type}_{olmo_subset}_seqlen{seq_len}"
    fpr_budget = 1
    plot_roc = False
    hypersearch = True

    # Redirect stdout to file
    with open(output_path, "w") as fout:
        with redirect_stdout(fout):
            print(f"Running bag-of-words for seq_len={seq_len}")
            if args.attack == "bow":
                bag_of_words_basic(X, y, dataset_name, fpr_budget, plot_roc, hypersearch)
            elif args.attack == "greedy":
                greedy_selection_basic(member_lines=members, nonmember_lines=nonmembers, dataset_name=dataset_name, fpr_budget=fpr_budget, plot_roc=plot_roc)
            elif args.attack == "date":
                date_detection_basic(X, y, dataset_name, fpr_budget, plot_roc)
            else:
                raise ValueError(f"Unknown attack type: {args.attack}")

