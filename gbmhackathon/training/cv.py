import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bootstrap
from pathlib import Path

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
            if run_name in result_bank["run"]:
                result_bank = result_bank[result_bank["run"] != run_name]
            
            result_bank = pd.concat([result_bank, df], axis=0)
            result_bank.to_csv(save_path, index=False)
        else:
            df.to_csv(save_path, index=False)
    return df


# def plot_results_cv(result_file, save=False, result_plot_file='results/plot_cv_results.pdf'):
#     """
#     Reads a CSV file containing cross-validation results and creates two plots:
#     1. Bar chart of avg values with error bars for each run, grouped by task, 
#        annotated with lower bound (white, bold), average (black, bold), and upper bound (black, bold).
#     2. Grouped bar chart of sum_avg and sum_ci_relative_width for each run,
#        annotated with the bar values at the center.
#     """
#     # Ensure the file exists
#     if not os.path.exists(result_file):
#         raise FileNotFoundError(f"{result_file} not found. Please provide a valid path.")

#     # Read the results
#     df = pd.read_csv(result_file)

#     # Extract run names
#     runs = df['run'].tolist()
#     n_runs = len(runs)

#     # Identify all columns matching avg_<metric>_<task>
#     avg_cols = [col for col in df.columns if col.startswith("avg_")]
#     metric_task_pairs = [tuple(col.split('_')[1:]) for col in avg_cols]  # [(metric, task), ...]
#     task_labels = [f"{task}\n({metric})" for metric, task in metric_task_pairs]
#     n_tasks = len(metric_task_pairs)

#     # Prepare data for the first plot
#     plot_data = []
#     for idx, run in enumerate(runs):
#         row = df.iloc[idx]
#         avgs = []
#         lows = []
#         ups = []
#         err_lowers = []
#         err_uppers = []
#         for metric, task in metric_task_pairs:
#             avg_val = row[f"avg_{metric}_{task}"]
#             low_val = row[f"low_{metric}_{task}"]
#             up_val = row[f"up_{metric}_{task}"]
#             avgs.append(avg_val)
#             lows.append(low_val)
#             ups.append(up_val)
#             err_lowers.append(avg_val - low_val)
#             err_uppers.append(up_val - avg_val)
#         plot_data.append({
#             'run': run,
#             'avg': avgs,
#             'low': lows,
#             'up': ups,
#             'err_low': err_lowers,
#             'err_up': err_uppers
#         })

#     # Create the first plot: avg values with error bars and annotations
#     sns.set(style="whitegrid")
#     fig1, ax1 = plt.subplots(figsize=(12, 6))
#     x = np.arange(n_tasks)
#     total_width = 0.8
#     bar_width = total_width / n_runs

#     palette = sns.color_palette("viridis", n_runs)
#     bar_containers = []
#     for i, pdata in enumerate(plot_data):
#         offsets = x - total_width/2 + (i + 0.5) * bar_width
#         bars = ax1.bar(
#             offsets,
#             pdata['avg'],
#             width=bar_width,
#             yerr=[pdata['err_low'], pdata['err_up']],
#             label=pdata['run'],
#             color=palette[i],
#             capsize=5
#         )
#         bar_containers.append((bars, pdata))

#     ax1.set_xticks(x)
#     ax1.set_xticklabels(task_labels, rotation=45, ha='right')
#     ax1.set_xlabel("Tasks (Metric)")
#     ax1.set_ylabel("Average Metric Value")
#     ax1.set_title("Cross-Validation Results by Task and Run")
#     ax1.legend(title="Run", bbox_to_anchor=(1.05, 1), loc='upper left')

#     # Annotate each bar with lower, average, and upper values
#     for bars, pdata in bar_containers:
#         avgs = pdata['avg']
#         lows = pdata['low']
#         ups = pdata['up']
        
#         for i, values in enumerate(zip(bars, avgs, lows, ups)):
#             bar, avg_val, low_val, up_val = values
#             sign = 1 if i % 2 == 0 else -1
#             x_center = bar.get_x() + bar.get_width() / 2
#             # Annotate lower bound (white, bold)
#             ax1.text(
#                 x_center, low_val - 0.03,
#                 f"{low_val:.2f}",
#                 color="white",
#                 fontweight="bold",
#                 fontsize=10,
#                 ha="center",
#                 va="center"
#             )
#             # # Annotate average (black, bold)
#             # ax1.text(
#             #     x_center + sign * 0.05, avg_val + 0.02,
#             #     f"{avg_val:.2f}",
#             #     color="black",
#             #     fontweight="bold",
#             #     fontsize=10,
#             #     ha=("left" if sign == 1 else "right"),
#             #     va="center"
#             # )
#             # Annotate upper bound (black, bold)
#             ax1.text(
#                 x_center, up_val + 0.03,
#                 f"{up_val:.2f}",
#                 color="black",
#                 fontweight="bold",
#                 fontsize=10,
#                 ha="center",
#                 va="center"
#             )

#     plt.tight_layout()
    
#     # Prepare data for the second plot
#     sum_reg_avgs = df['sum_reg_avg'].tolist()
#     sum_clf_avgs = df['sum_clf_avg'].tolist()
#     sum_reg_dims = df['sum_reg_ci_relative_width'].tolist()
#     sum_clf_dims = df['sum_clf_ci_relative_width'].tolist()

#     # Create the second plot: sum_avg and sum_ci_relative_width for each run, annotated at bar centers
#     fig2, ax2 = plt.subplots(figsize=(10, 5))
#     x2 = np.arange(n_runs)
#     width2 = 0.35

#     bars1 = ax2.bar(x2 - width2/4, sum_reg_avgs, width2, label="sum_reg_avg", color=palette[0])
#     bars2 = ax2.bar(x2 + width2/4, sum_reg_dims, width2, label="sum_ci_relative_width", color=palette[1 if n_runs > 1 else 0])
#     bars3 = ax2.bar(x2 - 2*width2/4, sum_clf_avgs, width2, label="sum_clf_avg", color=palette[2 if n_runs > 1 else 1])
#     bars4 = ax2.bar(x2 + 2*width2/4, sum_clf_dims, width2, label="sum_ci_relative_width", color=palette[3 if n_runs > 1 else 1])

#     ax2.set_xticks(x2)
#     ax2.set_xticklabels(runs, rotation=45, ha='right')
#     ax2.set_xlabel("Run")
#     ax2.set_ylabel("Value")
#     ax2.set_title("Sum of Averages and Relative CI Width per Run")
#     ax2.legend()

#     # Annotate bars at their centers
#     for bar in bars1 + bars2 + bars3 + bars4:
#         height = bar.get_height()
#         x_center = bar.get_x() + bar.get_width() / 2
#         y_center = height / 2
#         ax2.text(
#             x_center, y_center,
#             f"{height:.2f}",
#             color="white",
#             fontweight="bold",
#             fontsize=10,
#             ha="center",
#             va="center"
#         )

#     plt.tight_layout()
#     if save:
#         # Assumes the saving directory is not in the gbmhackathon module
#         suffix = 'summary'
#         save_path = '/'.join(os.path.abspath(__file__).split('/')[:-3]) + '/' + result_plot_file[:-4] + suffix + '.pdf'
#         plt.savefig(save_path, bbox_inches='tight', format='pdf')
#     plt.show()

def plot_results_cv(result_file, save=False, result_plot_file='results/plot_cv_results.pdf'):
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
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    y = np.arange(n_tasks)
    total_height = 0.8
    bar_height = total_height / n_runs

    palette = sns.color_palette("viridis", n_runs)
    bar_containers = []
    for i, pdata in enumerate(plot_data):
        offsets = y - total_height / 2 + (i + 0.5) * bar_height
        bars = ax1.barh(
            offsets,
            pdata['avg'],
            height=bar_height,
            xerr=[pdata['err_low'], pdata['err_up']],
            label=pdata['run'],
            color=palette[i],
            capsize=5
        )
        bar_containers.append((bars, pdata))

    ax1.set_yticks(y)
    ax1.set_yticklabels(task_labels)
    ax1.set_ylabel("Tasks (Metric)")
    ax1.set_xlabel("Average Metric Value")
    ax1.set_title("Cross-Validation Results by Task and Run")
    ax1.legend(title="Runs", bbox_to_anchor=(1.05, 1), loc='upper left')

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
    plt.xlim(0, np.max(pdata["up"]) + 0.15)
    plt.tight_layout()
    if save:
        # Save the second figure (last active) to the specified file path
        suffix = '_metrics'
        save_path = '/'.join(os.path.abspath(__file__).split('/')[:-3]) + '/' + result_plot_file[:-4] + suffix + '.pdf'
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        
    # Prepare data for the second plot
    sum_reg_avgs = df['sum_reg_avg'].tolist()
    sum_clf_avgs = df['sum_clf_avg'].tolist()
    sum_reg_dims = df['sum_reg_ci_relative_width'].tolist()
    sum_clf_dims = df['sum_clf_ci_relative_width'].tolist()

    # Create the second plot: horizontal grouped bars for sum values, annotated at bar centers
    fig2, ax2 = plt.subplots(figsize=(8, 6))
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
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

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

        train_loader = DataLoader(train_subset, batch_size=train_batch_size, collate_fn=collate_fn, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=val_batch_size, collate_fn=collate_fn, shuffle=False)
        yield train_loader, val_loader