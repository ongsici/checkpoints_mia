import copy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch 


class LossTraceDataset(Dataset):
    def __init__(self, dataframe):
        self.loss_traces = dataframe['loss_traces'].tolist()
        self.labels = dataframe['label'].tolist()
        self.extra_feats = dataframe[['lt_iqr_0.25_0.75', 'clfa_normalized', 'loss_trace_slope']].values

    def __len__(self):
        return len(self.loss_traces)

    def __getitem__(self, idx):
        trace = torch.tensor(self.loss_traces[idx], dtype=torch.float32).unsqueeze(1)  
        extra_feat = torch.tensor(self.extra_feats[idx], dtype=torch.float32)          
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return trace, extra_feat, label
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, rnn_outputs):  # rnn_outputs: [batch, seq_len, hidden]
        # Compute attention scores
        attn_energies = self.attn(rnn_outputs).squeeze(-1)  # [batch, seq_len]
        attn_weights = torch.softmax(attn_energies, dim=1)  # [batch, seq_len]
        # Weighted sum of RNN outputs
        context = torch.sum(rnn_outputs * attn_weights.unsqueeze(-1), dim=1)  # [batch, hidden]
        return context, attn_weights


class AttnRNNWithExtraFeatures(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        rnn_cls = nn.GRU if cfg["rnn_type"] == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            cfg["input_size"], 
            cfg["hidden_size"], 
            batch_first=True, 
            bidirectional=cfg.get("bidirectional", False)
        )
        self.attention = Attention(cfg["hidden_size"] * (2 if cfg.get("bidirectional", False) else 1))
        self.dropout = nn.Dropout(cfg.get("dropout", 0.0))
        self.fc = nn.Linear(cfg["hidden_size"] * (2 if cfg.get("bidirectional", False) else 1) + cfg["extra_feat_dim"], 1)

    def forward(self, trace, extra_feat):
        rnn_out, _ = self.rnn(trace)
        context, attn_weights = self.attention(rnn_out)
        combined = torch.cat((context, extra_feat), dim=1)
        out = self.fc(self.dropout(combined))
        return torch.sigmoid(out).squeeze(1), attn_weights


def run_rnn_classifier(model_cfg, num_epochs, batch_size, patience, num_folds, df):
    # Hyperparameters
    NUM_EPOCHS = num_epochs
    BATCH_SIZE = batch_size
    PATIENCE = patience
    NUM_FOLDS = num_folds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initial train+val / test split
    trainval_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    trainval_dataset = LossTraceDataset(trainval_df)
    test_dataset = LossTraceDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 2. K-Fold on train+val
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_df)):
        print(f"\n🚀 Fold {fold + 1}/{NUM_FOLDS}")

        # Subset datasets
        train_subset = Subset(trainval_dataset, train_idx)
        val_subset = Subset(trainval_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

        # Initialize model, optimizer, loss
        model = AttnRNNWithExtraFeatures(model_cfg).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0

            for traces, extras, labels in train_loader:
                traces, extras, labels = traces.to(device), extras.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs, _ = model(traces, extras)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * traces.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for traces, extras, labels in val_loader:
                    traces, extras, labels = traces.to(device), extras.to(device), labels.to(device)

                    outputs, _ = model(traces, extras)
                    val_loss += criterion(outputs, labels.float()).item() * traces.size(0)
                    all_preds.append(outputs.cpu())
                    all_labels.append(labels.cpu())

            val_loss /= len(val_loader.dataset)
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            try:
                auc = roc_auc_score(all_labels, all_preds)
            except ValueError:
                auc = float('nan')

            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, AUC = {auc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"⏹️ Early stopping at epoch {epoch + 1}")
                    break

        # Save and evaluate best model on test set
        model.load_state_dict(best_model_state)
        model.eval()
        test_loss = 0.0
        test_preds, test_labels = [], []

        with torch.no_grad():
            for traces, extras, labels in test_loader:
                traces, extras, labels = traces.to(device), extras.to(device), labels.to(device)

                outputs, _ = model(traces, extras)
                test_loss += criterion(outputs, labels.float()).item() * traces.size(0)
                test_preds.append(outputs.cpu())
                test_labels.append(labels.cpu())

        test_loss /= len(test_loader.dataset)
        test_preds = torch.cat(test_preds).numpy()
        test_labels = torch.cat(test_labels).numpy()

        try:
            test_auc = roc_auc_score(test_labels, test_preds)
        except ValueError:
            test_auc = float('nan')

        print(f"🧪 Fold {fold + 1} Test Loss = {test_loss:.4f}, Test AUC = {test_auc:.4f}")

        fold_metrics.append({
            'fold': fold + 1,
            'val_loss': best_val_loss,
            'val_auc': auc,
            'test_loss': test_loss,
            'test_auc': test_auc
        })

    # Summary
    print("\n✅ Cross-validation complete!")
    for m in fold_metrics:
        print(f"Fold {m['fold']}: Val Loss = {m['val_loss']:.4f}, Val AUC = {m['val_auc']:.4f}, Test Loss = {m['test_loss']:.4f}, Test AUC = {m['test_auc']:.4f}")

    avg_test_loss = np.mean([m['test_loss'] for m in fold_metrics])
    avg_test_auc = np.nanmean([m['test_auc'] for m in fold_metrics])
    std_test_auc = np.nanstd([m['test_auc'] for m in fold_metrics], ddof=1)
    print(f"\n📊 Average Test Loss: {avg_test_loss:.4f}")
    print(f"📈 Average Test AUC: {avg_test_auc:.4f}")
    print(f"Test AUC Std Dev: {std_test_auc:.4f}")