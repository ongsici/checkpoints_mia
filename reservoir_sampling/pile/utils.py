
import hashlib
from collections import defaultdict
from datasets import load_dataset
import random
from tqdm import tqdm

tqdm.pandas()

def tokenize_text(tokenizer, text):
    """Tokenize a single text input using the provided tokenizer."""
    tokens = tokenizer.encode_plus(
        text,
        return_tensors="pt",
        padding="longest"
    )
    input_ids = tokens["input_ids"][:, 1:].tolist()[0]
    attention_mask = tokens["attention_mask"][:, 1:].tolist()[0]
    return input_ids, attention_mask


def tokenize_df(df, tokenizer):
    """Tokenize a DataFrame column with text data."""
    df["llama_input_ids"], df["llama_attention_mask"] = zip(*df["text"].progress_apply(
        lambda t: tokenize_text(tokenizer, t)
    ))
    df["seq_len"] = df["llama_input_ids"].apply(len)
    return df

def get_sample_id(sample):
    """Generate a unique ID (hash) from sample text."""
    return hashlib.sha256(sample["text"].encode("utf-8")).hexdigest()


def run_reservoir_sampling(selected_groups, target_per_group_dict, seen_sample_ids, train_group=True, seed=42):
    """
    Perform reservoir sampling on the streaming dataset, filtering by group and text length.
    Avoid duplicates using seen_sample_ids set.
    """

    random.seed(seed)

    if train_group:
        stream = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True
        )
        print(f'Loaded training dataset stream')
    else:
        stream = load_dataset(
            "monology/pile-uncopyrighted",
            data_files={"test": "test.jsonl.zst"},
            split="test",
            streaming=True
        )
        print('Loaded test dataset stream')
    
    print(f"Stream object id: {id(stream)}")

    reservoirs = {key: [] for key in selected_groups}
    counters = defaultdict(int)


    for sample in stream:
        key = sample["meta"]["pile_set_name"]

        if key not in selected_groups:
            continue
        if len(sample["text"]) <= 5000:
            continue

        sample_id = get_sample_id(sample)
        if sample_id in seen_sample_ids[key]:
            continue

        counters[key] += 1
        count = counters[key]

        if key not in reservoirs:
            reservoirs[key] = []

        reservoir = reservoirs[key]
        k = target_per_group_dict[key]

        if count <= k:
            reservoir.append(sample)
        else:
            j = random.randrange(count)
            if j < k:
                reservoir[j] = sample

    for key, reservoir in reservoirs.items():
        for sample in reservoir:
            sample_id = get_sample_id(sample)
            seen_sample_ids[key].add(sample_id)

    return reservoirs