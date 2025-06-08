import os, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm
from plottable import Table, ColumnDefinition
from plottable.cmap import normed_cmap
import seaborn as sns
from scipy.stats import bootstrap
from pathlib import Path

from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

def get_mccv_loaders(dataset, 
                     collate_fn,
                     n_splits, 
                     test_size=0.8, 
                     train_batch_size=32, 
                     val_batch_size=128, 
                     device='cpu', 
                     random_state=6262):
    """
    Returns loaders designed to perform Monte Carlo Cross Validation (MCCV).
    """
    indices = list(dataset.ind2patient.keys())
    indices_arr = np.zeros((len(indices),1))
    
    mccv =  ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    for train_idx, val_idx in mccv.split(indices_arr):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=train_batch_size, collate_fn=collate_fn, shuffle=True, generator=torch.Generator(device=dataset.device))
        val_loader = DataLoader(val_subset, batch_size=val_batch_size, collate_fn=collate_fn, shuffle=False, generator=torch.Generator(device=dataset.device))
        yield train_loader, val_loader


def train_GBT(splitter, X, y_reg, y_cat, prefix):
    # se_sums = np.zeros(2)
    # f1_rec_sums = np.zeros(1)
    # f1_mgmt_sums = np.zeros(1)

    all_rmse1 = []
    all_rmse2 = []
    all_f1_rec  = []
    all_f1_mgmt  = []
    
    n_splits = splitter.get_n_splits(X)
    t0 = time.time()
    for i, (train_idx, test_idx) in enumerate(splitter.split(X), 1):
        X_train, X_test = X[train_idx].cpu().detach(), X[test_idx].cpu().detach()
        y_reg_train, y_reg_test = y_reg[train_idx].cpu().detach(), y_reg[test_idx].cpu().detach()
        y_rec_train, y_rec_test = torch.argmax(y_cat[train_idx,:2], dim=1).cpu().detach(), torch.argmax(y_cat[test_idx,:2], dim=1).cpu().detach()
        y_mgmt_train, y_mgmt_test = torch.argmax(y_cat[train_idx,2:], dim=1).cpu().detach(), torch.argmax(y_cat[test_idx,2:], dim=1).cpu().detach()
    
        reg_mt = MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=100, early_stopping=True),
            n_jobs=-1)
        clf_rec = HistGradientBoostingClassifier(max_iter=100, early_stopping=True)
        clf_mgmt = HistGradientBoostingClassifier(max_iter=100, early_stopping=True)
    
        # Training
        reg_mt.fit(X_train, y_reg_train)
        clf_rec.fit(X_train, y_rec_train)
        clf_mgmt.fit(X_train, y_mgmt_train)
    
        # Prediction
        y_reg_pred = reg_mt.predict(X_test)
        y_rec_pred = clf_rec.predict(X_test)
        y_mgmt_pred = clf_mgmt.predict(X_test)
    
        # Accumulation of metrics
        for k in range(2):
            rmse = np.sqrt(np.mean((y_reg_pred[:, k] - y_reg_test[:, k].cpu().numpy())**2))
            if k == 0:
                all_rmse1.append(rmse)
            else:
                all_rmse2.append(rmse)
            # se_sums[k] += np.sum((y_reg_pred[:, k] - y_reg_test[:, k].cpu().numpy())**2)

        all_f1_rec.append(f1_score(y_rec_test, y_rec_pred, average='weighted', zero_division=0))
        all_f1_mgmt.append(f1_score(y_mgmt_test, y_mgmt_pred, average='weighted', zero_division=0))
        
        # f1_rec_sums += f1_score(y_rec_test, y_rec_pred, average='weighted', zero_division=0)
        # f1_mgmt_sums += f1_score(y_mgmt_test, y_mgmt_pred, average='weighted', zero_division=0)
    
        print(f"Iteration {i}/{n_splits}", end="\r")
    
    # # Averages over all Monte Carlo iterations
    # avg_rmse = np.sqrt(se_sums / (n_splits * len(test_idx)))
    # avg_f1_rec = f1_rec_sums / n_splits
    # avg_f1_mgmt = f1_mgmt_sums / n_splits
    # print("\nMultitâche CV → RMSE:", avg_rmse, "— F1 Recurrency:", avg_f1_rec, "— F1 MGMT Methylation:", avg_f1_mgmt)
    runtime = time.time() - t0
    run_name = f"{prefix}_gbt"
    reg_dict = {'os':all_rmse1, 'pfs':all_rmse2}
    clf_dict = {'recurrency':all_f1_rec, 'mgmt':all_f1_mgmt}
    return get_cv_results(reg_dict, clf_dict, run_name, time=runtime, save=True)
    
def get_ci(data, n_resamples: int = 10000, confidence: float = 0.95, random_state: int = 6262):
    ci_results = bootstrap((data,), 
              np.mean, 
              n_resamples=n_resamples, 
              vectorized=True, 
              confidence_level=confidence,
              random_state=random_state).confidence_interval
    low, high = ci_results.low, ci_results.high
    return low, np.mean(data), high

def get_cv_results(reg_dict, 
                   clf_dict, 
                   run_name, 
                   reg_metric='rmse', 
                   clf_metric='f1', 
                   time=None, 
                   save=False, 
                   result_file='results/cv_results.csv'):
    n_folds = len(list(reg_dict.values())[0])
    if not time:
        time = np.nan
    df_dict = {"run":[run_name], 
               "n_folds":[n_folds], 
               "time":[time]}
    reg_avg_list = []
    clf_avg_list = []
    reg_rel_width_list = []
    clf_rel_width_list = []
    for i, eval_dict in enumerate([reg_dict, clf_dict]):
        if i == 0:
            metric = reg_metric
        else:
            metric = clf_metric
        for task in eval_dict.keys():
            low, avg, up = get_ci(eval_dict[task])
            if i == 0:
                reg_avg_list.append(avg)
                reg_rel_width_list.append((up - low) / avg)
            else:
                clf_avg_list.append(avg)
                clf_rel_width_list.append((up - low) / avg)
            low_col_name = f"low_{metric}_{task}"
            avg_col_name = f"avg_{metric}_{task}"
            up_col_name = f"up_{metric}_{task}"
            df_dict[low_col_name] = [low]
            df_dict[avg_col_name] = [avg]
            df_dict[up_col_name] = [up]
    df_dict["sum_reg_avg"] = np.sum(reg_avg_list).item()
    df_dict["sum_clf_avg"] = np.sum(clf_avg_list).item()
    df_dict["sum_reg_ci_relative_width"] = np.sum(reg_rel_width_list).item()
    df_dict["sum_clf_ci_relative_width"] = np.sum(clf_rel_width_list).item()
    df = pd.DataFrame(df_dict)
    if save:
        # Assumes the saving directory is not in the gbmhackathon module
        save_path = '/'.join(os.path.abspath(__file__).split('/')[:-3]) + '/' + result_file
        if os.path.exists(save_path):
            result_bank = pd.read_csv(save_path)
            
            result_bank = result_bank[result_bank["run"] != run_name]
            
            result_bank = pd.concat([result_bank, df], axis=0)
            result_bank.to_csv(save_path, index=False)
        else:
            df.to_csv(save_path, index=False)
    return df

def get_upper(char):
    return char.upper()
    
def plot_cv_table(result_file, 
                  df=None, 
                  clip_runs=False,
                  max_len=40,
                  save=False, 
                  plot=True, 
                  out_file="results/cv_results_table.pdf",
                  figsize=(25, 6)):
    """
    Reads `result_file` (cv_results.csv), wraps it in a plottable.Table,
    applies a clean “striped” theme, and displays (or saves) it.

    - Striped rows (alternating light gray / white)
    - Bold header row, centered text
    - Adjustable column widths based on content
    """
    if df is None:
        # 1) Read CSV into DataFrame
        df = pd.read_csv(result_file)
    df['run'] = df['run'].apply(get_upper)
    if clip_runs:
        df['run'] = [name[:max_len] + '..' for name in df['run']]
    df['to_rank'] = df[['sum_reg_avg']+[col for col in df.columns if 'up_rmse' in col]].sum(axis=1) + 1/(df[['sum_clf_avg']+[col for col in df.columns if 'low_f1' in col]].sum(axis=1) + 1e-4)
    cols = [col for col in df.columns if 'up_f1' not in col and 'low_rmse' not in col and 'ci' not in col and 'sum' not in col and col not in ['run', 'to_rank']]

    fig, ax = plt.subplots(figsize=figsize)

    col_defs = ([
        ColumnDefinition(
            name=col,
            textprops={"ha": "center"},
            width=1.5,
            border="left",
            cmap=normed_cmap(df[col], cmap=plt_cm.RdYlGn_r)
        ) for col in cols if col in ['avg_rmse_os', 'avg_rmse_pfs', 'sum_reg_avg'] + [col for col in df.columns if 'up_rmse' in col]]
               + [
        ColumnDefinition(
            name=col,
            textprops={"ha": "center"},
            width=1.5,
            border="both",
            cmap=normed_cmap(df[col], cmap=plt_cm.RdYlGn)
        ) for col in cols if col in ['avg_f1_recurrency', 'avg_f1_mgmt', 'sum_clf_avg'] + [col for col in df.columns if 'low_f1' in col]]
               )
    
    tab = Table(df.set_index("run").round(3).sort_values(by=['to_rank'])[cols],
               column_definitions=col_defs,
               row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
               col_label_divider=True,
               col_label_divider_kw={"linewidth": 2, "linestyle": "-"},
               column_border_kw={"linewidth": 2, "linestyle": "-"},
               textprops={'fontsize':10, 'fontweight':'bold'})

    # 5) Display or save
    if save:
        save_path = '/'.join(os.path.abspath(__file__).split('/')[:-3]) + '/' + out_file
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if plot:
        plt.show()

def plot_results_cv(result_file, 
                    run_filter: str = None, 
                    baseline='random_baseline',
                    include_baseline=True,
                    clip_runs=False,
                    max_len=40,
                    second_plot=False, 
                    save=False, 
                    figsize=(10,8),
                    result_plot_file='results/plot_cv_results.pdf'):
    """
    Reads a CSV file containing cross-validation results and creates two horizontal bar plots:
    1. Horizontal bar chart of avg values with error bars for each run, grouped by task, 
       annotated with lower bound (white, bold), and upper bound (black, bold).
    2. Horizontal grouped bar chart of sum_reg_avg, sum_reg_ci_relative_width, sum_clf_avg,
       and sum_clf_ci_relative_width, annotated with the bar values at the center.
    """
    # Ensure the file exists
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"{result_file} not found. Please provide a valid path.")

    # Read the results
    df = pd.read_csv(result_file)
    if run_filter is not None:
        if baseline not in run_filter and include_baseline:
            run_filter += f'|{baseline}'
        df = df[df['run'].str.contains(run_filter)]
    if clip_runs:
        df['run'] = [(name[:max_len] + '..' if len(name) >= max_len else name) for name in df['run']]
    # Sort values
    df['to_rank'] = df[['sum_reg_avg']+[col for col in df.columns if 'up_rmse' in col]].sum(axis=1) + 1/(df[['sum_clf_avg']+[col for col in df.columns if 'low_f1' in col]].sum(axis=1) + 1e-4)
    df = df.set_index("run").round(3).sort_values(by=['to_rank'])
    df['run'] = df.index
    # Extract run names
    runs = df['run'].tolist()
    n_runs = len(runs)

    # Identify all columns matching avg_<metric>_<task>
    avg_cols = [col for col in df.columns if col.startswith("avg_")]
    metric_task_pairs = [tuple(col.split('_')[1:]) for col in avg_cols]  # [(metric, task), ...]
    task_labels = [f"{task} ({metric})" for metric, task in metric_task_pairs]
    n_tasks = len(metric_task_pairs)

    # Prepare data for the first plot
    plot_data = []
    for idx, run in enumerate(runs):
        row = df.iloc[idx]
        avgs = []
        lows = []
        ups = []
        err_lowers = []
        err_uppers = []
        for metric, task in metric_task_pairs:
            avg_val = row[f"avg_{metric}_{task}"]
            low_val = row[f"low_{metric}_{task}"]
            up_val = row[f"up_{metric}_{task}"]
            avgs.append(avg_val)
            lows.append(low_val)
            ups.append(up_val)
            err_lowers.append(avg_val - low_val)
            err_uppers.append(up_val - avg_val)
        plot_data.append({
            'run': run,
            'avg': avgs,
            'low': lows,
            'up': ups,
            'err_low': err_lowers,
            'err_up': err_uppers
        })

    # Create the first plot: horizontal bars with error bars and annotations
    sns.set(style="whitegrid")
    fig1, ax1 = plt.subplots(figsize=figsize)
    y = np.arange(n_tasks)
    total_height = 0.8
    bar_height = total_height / n_runs

    palette = sns.color_palette("viridis", n_runs)
    bar_containers = []
    if baseline in runs:
        baseline_idx = runs.index(baseline)
    else:
        baseline_idx = None
    for i, pdata in enumerate(plot_data):
        offsets = y - total_height / 2 + (i + 0.5) * bar_height
        if (baseline_idx is not None) and (i == baseline_idx):
            c = 'black'
        else:
            c = palette[i]
        bars = ax1.barh(
            offsets,
            pdata['avg'],
            height=bar_height,
            xerr=[pdata['err_low'], pdata['err_up']],
            label=pdata['run'],
            color=c,
            capsize=5
        )
        bar_containers.append((bars, pdata))

    ax1.set_yticks(y)
    ax1.set_yticklabels(task_labels)
    ax1.set_ylabel("Tasks (Metric)")
    ax1.set_xlabel("Average Metric Value")
    ax1.set_title("Cross-Validation Results by Task and Run")
    ax1.legend(title="Runs", bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size':8})

    # Annotate each bar with lower and upper values
    for bars, pdata in bar_containers:
        lows = pdata['low']
        ups = pdata['up']
        for bar, low_val, up_val in zip(bars, lows, ups):
            y_center = bar.get_y() + bar.get_height() / 2
            # Annotate lower bound (white, bold) at low_val
            ax1.text(
                low_val - 0.05,
                y_center,
                f"{low_val:.2f}",
                color="white",
                fontweight="bold",
                fontsize=8,
                ha="center",
                va="center"
            )
            # Annotate upper bound (black, bold) at up_val
            ax1.text(
                up_val + 0.05,
                y_center,
                f"{up_val:.2f}",
                color="black",
                fontweight="bold",
                fontsize=8,
                ha="center",
                va="center"
            )
    plt.xlim(0, np.max(pdata["up"]) + 0.45)
    plt.tight_layout()
    plot_cv_table('..', df, save=False, plot=False, figsize=(25, 6))
    if save:
        # Save the second figure (last active) to the specified file path
        suffix = '_metrics'
        save_path = '/'.join(os.path.abspath(__file__).split('/')[:-3]) + '/' + result_plot_file[:-4] + suffix + '.pdf'
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        
    if second_plot:
        # Prepare data for the second plot
        sum_reg_avgs = df['sum_reg_avg'].tolist()
        sum_clf_avgs = df['sum_clf_avg'].tolist()
        sum_reg_dims = df['sum_reg_ci_relative_width'].tolist()
        sum_clf_dims = df['sum_clf_ci_relative_width'].tolist()
    
        # Create the second plot: horizontal grouped bars for sum values, annotated at bar centers
        fig2, ax2 = plt.subplots(figsize=figsize)
        y2 = np.arange(n_runs)
        total_height2 = 0.8
        bar_height2 = total_height2 / 4
    
        # Use distinct colors for the four bar sets
        second_palette = sns.color_palette("inferno", 4)
        bars1 = ax2.barh(y2 + bar_height2 * 1.5, sum_reg_avgs, height=bar_height2, label="sum_reg_avg", color=second_palette[0])
        bars2 = ax2.barh(y2 + bar_height2 * 0.5, sum_reg_dims, height=bar_height2, label="sum_reg_ci_relative_width", color=second_palette[1])
        bars3 = ax2.barh(y2 - bar_height2 * 0.5, sum_clf_avgs, height=bar_height2, label="sum_clf_avg", color=second_palette[2])
        bars4 = ax2.barh(y2 - bar_height2 * 1.5, sum_clf_dims, height=bar_height2, label="sum_clf_ci_relative_width", color=second_palette[3])
    
        ax2.set_yticks(y2)
        ax2.set_yticklabels(runs)
        ax2.set_ylabel("Run")
        ax2.set_xlabel("Value")
        ax2.set_title("Sum of Averages and Relative CI Width per Run")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size':8})
    
        # Annotate bars at their centers
        for bar in bars1 + bars2 + bars3 + bars4:
            width = bar.get_width()
            y_center = bar.get_y() + bar.get_height() / 2
            ax2.text(
                width / 2,
                y_center - 0.015,
                f"{width:.2f}",
                color="white",
                fontweight="bold",
                fontsize=7,
                ha="center",
                va="center"
            )
    
        plt.tight_layout()
        if save:
            # Save the second figure (last active) to the specified file path
            suffix = '_summary'
            save_path = '/'.join(os.path.abspath(__file__).split('/')[:-3]) + '/' + result_plot_file[:-4] + suffix + '.pdf'
            plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.show()