import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, auc
import math
from mia import *


k_values = list(range(0,144000,1000))
loss_columns = [f'loss_{k}' for k in k_values]

CB_color_cycle = ['#377eb8', '#4daf4a','#f781bf',
                '#a65628','#ff7f00', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

## Normalised Loss Traces

def get_normalised_losses(curr_loss, first_loss, final_loss):
    return (curr_loss - final_loss) / (first_loss - final_loss)

def normalize_loss_dfs(dataset_dict: dict, loss_columns: list) -> dict:
    normalised_dataset_dict = {}
    keep_cols = ["label", "llama_ppl", "llama_loss", "ratio", "mink_prob"]
    
    for key, df in dataset_dict.items():
        first_loss = df[loss_columns[0]]
        final_loss = df[loss_columns[-1]]

        norm_df = df[loss_columns].apply(
            lambda col: get_normalised_losses(col, first_loss, final_loss)
        )

        norm_df[keep_cols] = df[keep_cols]

        normalised_dataset_dict[key] = norm_df

    
    return normalised_dataset_dict

## Plotting Loss Curves

def plot_loss_curves(datasets, dataset_name, loss_columns=loss_columns, k_values=k_values):
    fig, axes = plt.subplots(nrows=math.ceil(len(datasets) / 2), ncols=2, figsize=(12, 18))
    axes = axes.flatten()

    for i, (title, df) in enumerate(datasets.items()):
        means_0 = df[df.label == 0][loss_columns].mean()
        stds_0 = df[df.label == 0][loss_columns].std()

        means_1 = df[df.label == 1][loss_columns].mean()
        stds_1 = df[df.label == 1][loss_columns].std()

        ax = axes[i]
        ax.errorbar(k_values, means_0, yerr=stds_0, label='Non-members', fmt='-o', color='red', capsize=4)
        ax.errorbar(k_values, means_1, yerr=stds_1, label='Members', fmt='-o', color='green', capsize=4)

        ax.set_title(f"Mean ± Std Loss ({title})")
        ax.set_xlabel('Training steps')
        ax.set_ylabel('Loss')
        ax.grid(True)
        ax.legend()

    # Hide the unused subplot (6th one)
    if len(datasets) < len(axes):
        axes[-1].axis('off')

    for i in range(len(datasets), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(f'Loss Curves for {dataset_name} subset of The Pile')
    plt.tight_layout()
    plt.show()


def plot_indv_loss_curves(dataset_dict, k_values, dataset_name):
    fig, axes = plt.subplots(nrows=math.ceil(len(dataset_dict) / 2), ncols=2, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, (title, df) in enumerate(dataset_dict.items()):

        ax = axes[i]
        for idx, row in df.iterrows():
            y = [row[f"loss_{k}"] for k in k_values]
            color = 'blue' if row['label'] == 1 else 'orange'
            ax.plot([k/1000 for k in k_values], y, color=color, alpha=0.5)
        if i==5:
            ax.plot([], [], color='blue', label='Member')
            ax.plot([], [], color='orange', label='Non-member')

        ax.set_title(f"{title}", fontsize=22)
        if i % 2 == 0:
            ax.set_ylabel('Normalised Losses', fontsize=22)
        if i > 3:
            ax.set_xlabel('Training steps (k)', fontsize=22)
        
        ax.set_yscale('log')
        ax.tick_params(axis='x', labelsize=22)
        ax.tick_params(axis='y', labelsize=22)
        # ax.set_xscale('log')
        # ax.grid(True)
        ax.legend()

    # plt.suptitle(f"Loss across training steps for {dataset_name}", fontsize=20)
    plt.tight_layout()
    plt.show()


## Plotting Baseline Attack ROC

def plot_roc_curves(datasets, dataset_name, target_fpr, loss_columns=loss_columns):
    
    fig, axes = plt.subplots(nrows=math.ceil(len(datasets) / 2), ncols=2, figsize=(12, 18))
    axes = axes.flatten()

    loss_aucs = {}
    loss_tpr_at_target = {}

    minkprob_aucs = {}
    minkprob_tpr_at_target = {} 

    ratio_aucs = {}
    ratio_tpr_at_target = {}

    for i, (title, df) in enumerate(datasets.items()):

        y_true = df['label']
        y_score_loss = df[loss_columns[-1]]
        y_score_minkprob = df['mink_prob']
        y_score_ratio = df['ratio']

        # LOSS attack
        fpr_loss, tpr_loss, _ = roc_curve(y_true, -y_score_loss, pos_label=1)
        roc_auc_loss = auc(fpr_loss, tpr_loss)
        tpr_loss_at_target = np.interp(target_fpr, fpr_loss, tpr_loss)
        loss_aucs[title] = roc_auc_loss
        loss_tpr_at_target[title] = tpr_loss_at_target

        # Min-K% Prob attack
        fpr_minkprob, tpr_minkprob, _ = roc_curve(y_true, y_score_minkprob, pos_label=1)
        roc_auc_minkprob = auc(fpr_minkprob, tpr_minkprob)
        tpr_minkprob_at_target = np.interp(target_fpr, fpr_minkprob, tpr_minkprob)
        minkprob_aucs[title] = roc_auc_minkprob
        minkprob_tpr_at_target[title] = tpr_minkprob_at_target

        # Ratio attack
        fpr_ratio, tpr_ratio, _ = roc_curve(y_true, -y_score_ratio)
        roc_auc_ratio = auc(fpr_ratio, tpr_ratio)
        tpr_ratio_at_target = np.interp(target_fpr, fpr_ratio, tpr_ratio)
        ratio_aucs[title] = roc_auc_ratio
        ratio_tpr_at_target[title] = tpr_ratio_at_target
        
        ax = axes[i]
        ax.plot(fpr_loss, tpr_loss, color='blue', label=f'LOSS (AUC = {roc_auc_loss:.3f})')
        ax.axhline(y=tpr_loss_at_target, color='blue', linestyle='--', label=f'LOSS TPR @ {target_fpr * 100}% FPR ({title} = {tpr_loss_at_target:.4f})')
        ax.plot(fpr_minkprob, tpr_minkprob, color='orange', label=f'Min-K% Prob (AUC = {roc_auc_minkprob:.3f})')
        ax.axhline(y=tpr_minkprob_at_target, color='orange', linestyle='--', label=f'LOSS TPR @ {target_fpr * 100}% FPR ({title} = {tpr_minkprob_at_target:.4f})')
        ax.plot(fpr_ratio, tpr_ratio, color='green', label=f'Ratio (AUC = {roc_auc_ratio:.3f})')
        ax.axhline(y=tpr_ratio_at_target, color='green', linestyle='--', label=f'Ratio TPR @ {target_fpr * 100}% FPR ({title} = {tpr_ratio_at_target:.4f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')

        ax.set_title(f"{title}")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.grid(True)
        ax.legend(loc='lower right')

    # Hide the unused subplot (6th one)
    if len(datasets) < len(axes):
        axes[-1].axis('off')

    plt.suptitle(f"ROC Plots for {dataset_name} - LOSS and Min-K Prob Attacks")
    plt.tight_layout()
    plt.show()

    return loss_aucs, loss_tpr_at_target, minkprob_aucs, minkprob_tpr_at_target, ratio_aucs, ratio_tpr_at_target


## Plot LT-IQR ROC

def plot_roc_iqr_single_plot(dataset_dict, loss_columns, dataset_name, target_fpr=0.01, modify_df = False, q1=0.25, q2=0.75):
    fig = plt.figure(figsize=(8,6))
    colors = plt.cm.tab10.colors

    seq_len_aucs = {}
    seq_len_tpr_at_target = {}

    for idx, (title, df) in enumerate(dataset_dict.items()):
  
        iqr_values, iqr_col= compute_iqr_features(df, loss_columns, modify_df, q1, q2)
        y_true = df['label']
        y_score = np.array(iqr_values)
        color = colors[idx % len(colors)] 

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        tpr_at_target = np.interp(target_fpr, fpr, tpr)
        seq_len_aucs[title] = roc_auc
        seq_len_tpr_at_target[title] = tpr_at_target

        plt.plot(fpr, tpr, label=f'{title} (AUC = {roc_auc:.3f})')
        # plt.axhline(y=tpr_at_target, color=color, linestyle='--', label = f'TPR @ {target_fpr * 100}% FPR ({title}) ={tpr_at_target:.4f}')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'ROC Curve for LT-IQR of {dataset_name} (q1={q1}, q2={q2})')
    plt.grid(True)
    plt.legend(loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()

    return seq_len_aucs, seq_len_tpr_at_target


## Plot LT-Slope ROC

def plot_loss_trace_slope(dataset_dict, loss_columns, dataset_name, target_fpr=0.01, modify_df = False):
    fig = plt.figure(figsize=(8,6))
    colors = plt.cm.tab10.colors

    seq_len_aucs = {}
    seq_len_tpr_at_target = {}

    for idx, (title, df) in enumerate(dataset_dict.items()):

        selected_loss_trace_values = df[loss_columns].values.tolist()
        slope_values = [compute_loss_trace_slope(trace) for trace in selected_loss_trace_values]

        if modify_df:
            df['loss_trace_slope'] = slope_values 

        y_true = df['label']
        y_score = np.array(slope_values)
        color = colors[idx % len(colors)] 

        fpr, tpr, _ = roc_curve(y_true, -y_score)
        roc_auc = auc(fpr, tpr)
        tpr_at_target = np.interp(target_fpr, fpr, tpr)

        seq_len_aucs[title] = roc_auc
        seq_len_tpr_at_target[title] = tpr_at_target

        plt.plot(fpr, tpr, color=color, label=f'{title} (AUC = {roc_auc:.3f})')
        # plt.axhline(y=tpr_at_target, color=color, linestyle='--', label = f'TPR @ {target_fpr * 100}% FPR ({title}) ={tpr_at_target:.4f}')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for LT-Slope of {dataset_name}')
    plt.grid(True)
    plt.legend(loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()

    return seq_len_aucs, seq_len_tpr_at_target

## Plot LT-Mean ROC

def plot_loss_trace_mean(dataset_dict, loss_columns, dataset_name, target_fpr=0.01, modify_df= False):
    fig = plt.figure(figsize=(8,6))
    colors = plt.cm.tab10.colors

    seq_len_aucs = {}
    seq_len_tpr_at_target = {}

    for idx, (title, df) in enumerate(dataset_dict.items()):

        selected_loss_trace_values = df[loss_columns].values.tolist()
        mean_values = [compute_loss_trace_mean(trace) for trace in selected_loss_trace_values]

        if modify_df:
            df['loss_trace_mean'] = mean_values 

        y_true = df['label']
        y_score = np.array(mean_values)
        color = colors[idx % len(colors)] 

        fpr, tpr, _ = roc_curve(y_true, -y_score)
        roc_auc = auc(fpr, tpr)
        tpr_at_target = np.interp(target_fpr, fpr, tpr)

        seq_len_aucs[title] = roc_auc
        seq_len_tpr_at_target[title] = tpr_at_target

        plt.plot(fpr, tpr, color=color, label=f'{title} (AUC = {roc_auc:.3f})')
        # plt.axhline(y=tpr_at_target, color=color, linestyle='--', label = f'TPR @ {target_fpr * 100}% FPR ({title}) ={tpr_at_target:.4f}')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for LT-Mean of {dataset_name}')
    plt.grid(True)
    plt.legend(loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()

    return seq_len_aucs, seq_len_tpr_at_target

## Plot CLFA ROC and Distribution 

def plot_clfa_normalized(dataset_dict, loss_columns, dataset_name, target_fpr=0.01, modify_df=False, colors=CB_color_cycle, col_name='clfa_normalized'):
    fig = plt.figure(figsize=(8,6))

    seq_len_aucs = {}
    seq_len_tpr_at_target = {}

    for idx, (title, df) in enumerate(dataset_dict.items()):

        clfa_normalized = compute_clfa_normalized(df, loss_columns, modify_df, col_name)

        y_true = df['label']
        y_score = clfa_normalized
        color = colors[idx % len(colors)] 

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        tpr_at_target = np.interp(target_fpr, fpr, tpr)
        seq_len_aucs[title] = roc_auc
        seq_len_tpr_at_target[title] = tpr_at_target

        plt.plot(fpr, tpr, color=color, label=f'{title} (AUC = {roc_auc:.3f})')
        # plt.axhline(y=tpr_at_target, color=color, linestyle='--', label = f'TPR @ {target_fpr * 100}% FPR ({title}) = {tpr_at_target:.4f}')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'ROC Curve for CLFA of {dataset_name}', fontsize=16)
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.show()

    return seq_len_aucs, seq_len_tpr_at_target

def plot_clfa_norm_distribution(dataset_dict, dataset_name):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 14))
    axes = axes.flatten()

    all_values = np.concatenate([df['clfa_normalized'].values for df in dataset_dict.values()])
    min_val = np.min(all_values)
    max_val = np.max(all_values)

    bins = np.linspace(min_val, max_val, 51)  

    for i, (title, df) in enumerate(dataset_dict.items()):
        ax = axes[i]

        sns.histplot(
            data=df[df['label'] == 1],
            x='clfa_normalized',
            bins=bins, color='blue', ax=ax, alpha=0.5
        )
        sns.histplot(
            data=df[df['label'] == 0],
            x='clfa_normalized',
            bins=bins, color='orange', ax=ax, alpha=0.5
        )

        if i==5:
            ax.plot([], [], color='blue', label='Member')
            ax.plot([], [], color='orange', label='Non-member')

        ax.set_title(f"seq_len={title}", fontsize=22)
        ax.set_xlabel('Normalized CLFA', fontsize=22)
        ax.set_ylabel('Frequency', fontsize=22)
        ax.tick_params(axis='x', labelsize=22)
        ax.tick_params(axis='y', labelsize=22)
        ax.legend(fontsize=22)

    # Hide any unused subplots
    for i in range(len(dataset_dict), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


## Plot FOD Mean ROC

def plot_fod(dataset_dict, dataset_name, loss_columns, target_fpr=0.01):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    fig.suptitle(f'ROC Curves for FOD Features ({dataset_name})', fontsize=16)
    
    colors = plt.cm.tab10.colors

    fod_mean_aucs = {}
    fod_mean_tpr_at_target = {}
    fod_abs_mean_aucs = {}
    fod_abs_mean_tpr_at_target = {}

    for i, (title, df) in enumerate(dataset_dict.items()):

        fod_stats = df.apply(lambda row: compute_first_order_stats(row, loss_columns), axis=1)
        df['fod_mean'] = fod_stats['fod_mean_diff']
        df['fod_abs_mean'] = fod_stats['fod_abs_mean_diff']

        for ax, metric in zip(axes, ['fod_mean', 'fod_abs_mean']):
            y_true = df['label']
            y_score = df[metric]

            if metric == 'fod_mean':
                y_score = -y_score

            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            tpr_at_target = np.interp(target_fpr, fpr, tpr)
            if metric == 'fod_mean':
                fod_mean_aucs[title] = roc_auc
                fod_mean_tpr_at_target[title] = tpr_at_target
            else:
                fod_abs_mean_aucs[title] = roc_auc
                fod_abs_mean_tpr_at_target[title] = tpr_at_target

            ax.plot(fpr, tpr, label=f'{title} (AUC={roc_auc:.3f})', color=colors[i % len(colors)])
            ax.axhline(y=tpr_at_target, color = colors[i % len(colors)], linestyle='--', label=f'TPR @ {target_fpr * 100}% FPR ({title}) = {tpr_at_target:.4f}')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.set_title(f'{metric}')
            ax.grid(True)

    axes[0].legend(loc='upper left', fontsize='small')
    axes[1].legend(loc='upper left', fontsize='small')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return fod_mean_aucs, fod_mean_tpr_at_target, fod_abs_mean_aucs, fod_abs_mean_tpr_at_target

## Plot EMA-IQR and EMA-Slope ROC

def compute_and_plot_ema_df(
    dataset_dict,
    dataset_ema_dict,
    dataset_name,
    loss_columns,
    target_fpr=0.01,
    modify_df=False,
    alphas=[0.1, 0.2, 0.3, 0.4, 0.5],
    q1=0.25,
    q2=0.75,
):
    fig_iqr, axes_iqr = plt.subplots(nrows=math.ceil(len(dataset_dict) / 2), ncols=2, figsize=(14, 20), sharey=True, sharex=True)
    fig_slope, axes_slope = plt.subplots(nrows=math.ceil(len(dataset_dict) / 2), ncols=2, figsize=(14, 20), sharey=True, sharex=True)
    axes_iqr = axes_iqr.flatten()
    axes_slope = axes_slope.flatten()

    fig_iqr.suptitle(f'ROC Curves for EMA-IQR (q1={q1}, q2={q2}) for {dataset_name}')
    fig_slope.suptitle(f'ROC Curves for EMA-Slope for {dataset_name}')
    colors = plt.cm.tab10.colors

    iqr_seq_len_alpha_aucs = dict()
    iqr_seq_len_alpha_tprs = dict()
    slope_seq_len_alpha_aucs = dict()
    slope_seq_len_alpha_tprs = dict()

    for i, (title, df) in enumerate(dataset_dict.items()):
        ax_iqr = axes_iqr[i]
        ax_slope = axes_slope[i]
        iqr_seq_len_alpha_aucs[title] = dict()
        iqr_seq_len_alpha_tprs[title] = dict()
        slope_seq_len_alpha_aucs[title] = dict()
        slope_seq_len_alpha_tprs[title] = dict()

        ema_df = df.copy()

        # Compute EMA traces for each row using the specified loss columns
        ema_traces_df = df.apply(
            lambda row: compute_ema_features(row, loss_columns=loss_columns, alphas=alphas, return_type='trace'),
            axis=1
        ).apply(pd.Series)

        ema_df = pd.concat([ema_df, ema_traces_df], axis=1)

        y_true = df['label'].values

        for j, alpha in enumerate(alphas):
            col = f'ema_trace_alpha_{alpha}'

            # EMA-IQR
            ema_df["loss_traces"] = ema_df[col] 
            iqr_values, iqr_col_name = compute_iqr_features(ema_df, loss_columns="loss_traces", modify_df=True, q1=q1, q2=q2)
            y_score_iqr = ema_df[iqr_col_name].values

            fpr_iqr, tpr_iqr, _ = roc_curve(y_true, y_score_iqr)
            roc_auc_iqr = auc(fpr_iqr, tpr_iqr)
            tpr_at_target_iqr = np.interp(target_fpr, fpr_iqr, tpr_iqr)
            iqr_seq_len_alpha_aucs[title][alpha] = roc_auc_iqr
            iqr_seq_len_alpha_tprs[title][alpha] = tpr_at_target_iqr

            ax_iqr.plot(fpr_iqr, tpr_iqr, label=f'α={alpha} (AUC={roc_auc_iqr:.3f})', color=colors[j])
            ax_iqr.axhline(y=tpr_at_target_iqr, color=colors[j], linestyle='--', linewidth=0.8,
                           label=f'TPR @ {target_fpr * 100}% FPR ({title}) = {tpr_at_target_iqr:.4f} ')

            # EMA-Slope 
            slope_col = f'slope_alpha_{alpha}'
            ema_df[slope_col] = ema_df[col].apply(compute_loss_trace_slope)
            y_score_slope = -ema_df[slope_col].values
            fpr_slope, tpr_slope, _ = roc_curve(y_true, y_score_slope)
            roc_auc_slope = auc(fpr_slope, tpr_slope)
            tpr_at_target_slope = np.interp(target_fpr, fpr_slope, tpr_slope)
            slope_seq_len_alpha_aucs[title][alpha] = roc_auc_slope
            slope_seq_len_alpha_tprs[title][alpha] = tpr_at_target_slope

            ax_slope.plot(fpr_slope, tpr_slope, label=f'α={alpha} (AUC={roc_auc_slope:.3f})', color=colors[j])
            ax_slope.axhline(y=tpr_at_target_slope, color=colors[j], linestyle='--', linewidth=0.8,
                             label=f'TPR @ {target_fpr * 100}% FPR ({title}) = {tpr_at_target_slope:.4f} ')

        for ax in [ax_iqr, ax_slope]:
            ax.set_title(title)
            ax.set_xlabel('FPR')
            ax.grid(True)
            if ax == axes_iqr[0] or ax == axes_slope[0]:
                ax.set_ylabel('TPR')
            ax.legend(loc='lower right', fontsize='small')

        dataset_ema_dict[title] = ema_df

    for k in range(len(dataset_dict), len(axes_iqr)):
        axes_iqr[k].axis('off')
        axes_slope[k].axis('off')

    fig_iqr.tight_layout()
    fig_slope.tight_layout()
    plt.show()

    return iqr_seq_len_alpha_aucs, iqr_seq_len_alpha_tprs, slope_seq_len_alpha_aucs, slope_seq_len_alpha_tprs

## Plot S2Conv ROC

def plot_steps_to_convergence(dataset_dict, loss_columns, dataset_name, target_fpr = 0.01, epsilon = 0.05, final_fraction = 0.1, colors=CB_color_cycle):
    fig = plt.figure(figsize=(8,6))

    seq_len_aucs = {}
    seq_len_tpr_at_target = {}

    for idx, (title, df) in enumerate(dataset_dict.items()):

        auc_ttc = df.apply(lambda row: compute_steps_to_convergence_row(row, loss_columns, epsilon, final_fraction), axis=1)
        df['s2conv'] = auc_ttc

        y_true = df['label']
        y_score = df['s2conv']

        color = colors[idx % len(colors)] 

        fpr, tpr, _ = roc_curve(y_true, -y_score)
        roc_auc = auc(fpr, tpr)
        tpr_at_target = np.interp(target_fpr, fpr, tpr)
        seq_len_aucs[title] = roc_auc
        seq_len_tpr_at_target[title] = tpr_at_target

        plt.plot(fpr, tpr, color=color, label=f'{title} (AUC = {roc_auc:.3f})')
        # plt.axhline(y=tpr_at_target, color=color, linestyle='--', label = f'TPR @ {target_fpr * 100}% FPR ({title}) = {tpr_at_target:.4f}')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'ROC Curve for S2Conv of {dataset_name}', fontsize=16)
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.show()

    return seq_len_aucs, seq_len_tpr_at_target