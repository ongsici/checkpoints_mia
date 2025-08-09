import pickle
import pandas as pd
import yaml
from utils import run_reservoir_sampling, tokenize_df
from transformers import LlamaTokenizer
from tqdm import tqdm
from collections import defaultdict

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Assign config variables
MIN_TARGET_SAMPLES = cfg["MIN_TARGET_SAMPLES"]
OVERSAMPLE_EXTRA = cfg["OVERSAMPLE_EXTRA"]
MIN_SEQ_LEN = cfg["MIN_SEQ_LEN"]
TRAIN_GROUP = cfg["TRAIN_GROUP"]
SEED = cfg["SEED"]
DATE = cfg["DATE"]
OUTPUT_FILE = cfg["OUTPUT_FILE"]
LLAMA_MODEL_NAME = cfg["LLAMA_MODEL_NAME"]
ALL_GROUPS = cfg["ALL_GROUPS"]
LOGFILE_NAME = cfg["LOGFILE_TEMPLATE"].format(DATE=DATE)
TRAIN_GROUP = cfg["TRAIN_GROUP"]

print('==============================================')
print('SCRIPT INFORMATION RESERVOIR SAMPLING')
print('==============================================\n')

print(f'num samples per group: {MIN_TARGET_SAMPLES + OVERSAMPLE_EXTRA}')
print(f'random seed: {SEED}')
print(f'output file: {OUTPUT_FILE}')
print(f'log file: {LOGFILE_NAME}')
print(f'selected groups: {ALL_GROUPS}')

seen_sample_ids = defaultdict(set)
qualified_sample_counts = defaultdict(int) 
group_pass_threshold = set()
all_samples = []

round = 1

selected_groups = ALL_GROUPS.copy()

while selected_groups:
    print(f'============================================')
    print(f'Running round {round} of reservoir sampling')
    print(f'============================================')

    for group in selected_groups:
        print(f"{group}: {len(seen_sample_ids[group])} seen samples")

    # Determine how many more samples needed per group
    target_per_group_dict = {
        group: max(0, MIN_TARGET_SAMPLES - qualified_sample_counts[group]) + OVERSAMPLE_EXTRA
        for group in selected_groups
    }

    # Perform reservoir sampling
    reservoirs = run_reservoir_sampling(selected_groups,
                                            target_per_group_dict, 
                                            seen_sample_ids, 
                                            train_group=TRAIN_GROUP,
                                            seed=SEED + round)
    # Flatten results
    batch_samples = []
    for key, res in reservoirs.items():
        batch_samples.extend(res)

    df_batch = pd.DataFrame(batch_samples)
    if df_batch.empty:
        print("No new samples were retrieved. Stopping early.")
        break

    df_batch["pile_set_name"] = df_batch["meta"].apply(lambda x: x.get("pile_set_name"))
    df_batch.drop(columns="meta", inplace=True)

    counts = df_batch['pile_set_name'].value_counts()
    print("Samples collected this round per group:")
    for group in selected_groups:
        print(f"  {group}: {counts.get(group, 0)} samples")

    # Tokenize
    print("Tokenizing batch...")
    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL_NAME, torch_dtype="auto")
    tokenizer.pad_token = tokenizer.eos_token
    df_batch = tokenize_df(df_batch, tokenizer)

    # Filter to long sequences
    df_filtered = df_batch[df_batch["seq_len"] >= MIN_SEQ_LEN]
    grouped = df_filtered.groupby("pile_set_name")

    for name, group in grouped:
        qualified_sample_counts[name] += len(group)
        if qualified_sample_counts[name] >= MIN_TARGET_SAMPLES:
            if name not in group_pass_threshold:
                print(f'Group {name} now has enough qualified samples: {qualified_sample_counts[name]}')
            group_pass_threshold.add(name)
        else:
            print(f'{name} has {qualified_sample_counts[name]} qualified samples (still needs {MIN_TARGET_SAMPLES - qualified_sample_counts[name]})')

    all_samples.append(df_filtered)

    # Update the group list for next round
    selected_groups = [g for g in ALL_GROUPS if g not in group_pass_threshold]

    round += 1

print("Saving final dataset...")
df_final = pd.concat(all_samples, ignore_index=True)
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(df_final, f)
