import yaml
import pandas as pd
from utils import run_rnn_classifier

def get_config():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    return cfg

def read_dataset(DATASET_SEQ_LENS, PILE_SUBSET, MODEL):
    dataset = {}

    for seq_len in DATASET_SEQ_LENS:
        dataset_path = cfg["DATASET_SEQ_LEN_TEMPLATE"].format(PILE_SUBSET=PILE_SUBSET, SEQ_LEN=seq_len, MODEL=MODEL)
        dataset[seq_len] =  pd.read_pickle(dataset_path)

    return dataset

if __name__ == "__main__":
    cfg = get_config()
    
    NUM_EPOCHS = cfg["NUM_EPOCHS"]
    BATCH_SIZE = cfg["BATCH_SIZE"]
    PATIENCE = cfg["PATIENCE"]
    NUM_FOLDS = cfg["NUM_FOLDS"]
    PILE_SUBSET = cfg["PILE_SUBSET"]
    MODEL = cfg["MODEL"]
    DATASET_SEQ_LENS = cfg["DATASET_SEQ_LENS"]

    dataset = read_dataset(DATASET_SEQ_LENS, PILE_SUBSET, MODEL)

    run_rnn_classifier(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        num_folds=NUM_FOLDS,
        df=dataset
    )