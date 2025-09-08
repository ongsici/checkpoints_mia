import pandas as pd
import numpy as np


## LT-IQR

def compute_iqr_features(df, loss_columns, modify_df = False, q1=0.25, q2=0.75):
    df["loss_traces"] = df[loss_columns].values.tolist()
    loss_traces_subset = df[loss_columns].values.tolist()

    iqr_values = [np.quantile(x, q2) - np.quantile(x, q1) for x in loss_traces_subset]

    if modify_df:
        df["loss_traces"] = loss_traces_subset
        col_name = f"lt_iqr_{q1:.2f}_{q2:.2f}"
        df[col_name] = iqr_values
    
    return iqr_values, col_name if modify_df else None


## LT-Slope

def compute_loss_trace_slope(loss_trace):
    """
    Compute the slope φ(x, y) of a loss trace using the formula:
        φ = sum[(l_s - mean_l) * (s - mean_s)] / sum[(s - mean_s)^2]
    where s = 1, 2, ..., S
    """
    loss_trace = np.array(loss_trace)
    S = len(loss_trace)
    s = np.arange(1, S + 1)
    
    mean_l = np.mean(loss_trace)
    mean_s = np.mean(s)
    
    numerator = np.sum((loss_trace - mean_l) * (s - mean_s))
    denominator = np.sum((s - mean_s) ** 2)
    
    return numerator / denominator

## LT-Mean

def compute_loss_trace_mean(loss_trace):
    loss_trace = np.array(loss_trace)
    return np.mean(loss_trace)

## CLFA

def compute_clfa_row(row, loss_columns):
    losses = np.array(row[loss_columns], dtype=float)
    diffs = np.abs(np.diff(losses))
    return np.sum(diffs)

def compute_clfa_normalized(df, loss_columns, modify_df=False, col_name='clfa_normalized'):
    clfa_values = df.apply(lambda row: compute_clfa_row(row, loss_columns), axis=1)

    min_clfa = clfa_values.min()
    max_clfa = clfa_values.max()

    clfa_normalized = (clfa_values - min_clfa) / ((max_clfa - min_clfa) + 1e-10)
    
    if modify_df:
        df['clfa'] = clfa_values
        df[col_name] = clfa_normalized
    
    return clfa_normalized


## FOD Mean

def compute_first_order_stats(row, loss_columns):
    losses = np.array(row[loss_columns], dtype=float)
    diffs = np.diff(losses)  
    mean_diff = np.mean(diffs)
    abs_mean_diff = np.mean(np.abs(diffs))
    return pd.Series({
        'fod_mean_diff': mean_diff,
        'fod_abs_mean_diff': abs_mean_diff
    })


## EMA-IQR and EMA-Slope

def compute_ema_features(row, loss_columns, alphas=[0.1, 0.2, 0.3, 0.4, 0.5], return_type='trace'):
    losses = row[loss_columns].astype(float).values
    results = {}

    for alpha in alphas:
        ema = losses[0]
        ema_trace = [ema]
        for loss in losses[1:]:
            ema = (1-alpha) * loss + alpha * ema
            ema_trace.append(ema)

        if return_type == 'last':
            results[f'ema_last_alpha_{alpha}'] = ema_trace[-1]
        elif return_type == 'mean':
            results[f'ema_mean_alpha_{alpha}'] = np.mean(ema_trace)
        elif return_type == 'trace':
            results[f'ema_trace_alpha_{alpha}'] = ema_trace
        else:
            raise ValueError("return_type must be 'last', 'mean', or 'trace'")
    
    return results

# S2Conv

def compute_steps_to_convergence_row(row, loss_columns, epsilon=0.05, final_fraction=0.1):
    losses = np.array(row[loss_columns], dtype=float)
    n_steps = len(losses)
    steps = np.arange(n_steps)

    cutoff = int(n_steps * (1 - final_fraction))
    L_final = losses[cutoff:].mean()

    threshold = (1 + epsilon) * L_final

    idx = np.where(losses <= threshold)[0]
    if len(idx) == 0:
        return steps[-1]
    else:
        return int(steps[idx[0]])