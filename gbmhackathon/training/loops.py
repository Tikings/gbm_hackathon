import numpy as np
import pandas as pd
import seaborn as sns
import pickle as pkl
import shutil, os

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import warnings
import inspect
import matplotlib.pyplot as plt

from gbmhackathon.training.predictive import collate_predictive
from gbmhackathon.viz.viz_experiment import see_emb, make_matplotlib_gif, new_framework_inference
from gbmhackathon.utils.analysis import compute_embedding_quality_metrics
from gbmhackathon.models.mme import concat_modality_embeddings
from gbmhackathon.utils.loss_functions import RegularizationModule

def modality_wise_contrastive_learning(mme, 
                          dataloader,
                          epochs, 
                          base_lr, 
                          optimizer_dict, # One optimizer instance per modality
                          contrastive_loss_dict, # One loss instance per modality
                          scheduler_dict, # One scheduler instance per modality
                          regularizer_dict=None, #One regularizer method per modality
                          batch_all=None,
                          make_gifs=False,
                          analyze_emb=False,
                          device=None,
                          ):
    device = device if device is not None else mme.device
    # Initialize metric collectors
    embedding_metrics = []
    
    
    # Phase I training loop
    EPOCHS_I_LOSSES = []
    modalities = next(iter(dataloader))[2].keys()
    
    # Logging of modality losses at each batch
    modality_batch_loss = {modality:[] for modality in modalities}
    
    # Logging of modality losses per epoch
    modality_avg_loss = {modality:[] for modality in modalities}

    if make_gifs and batch_all is None:
        print("No batch_all specified, setting make_gifs to False.")
        make_gifs = False
        
    if make_gifs:
        # Save embedding plots
        emb_frame_dirs = ['emb_mme', 'emb_by_mod', 'emb_by_patient']
                    
        for d in emb_frame_dirs:
            os.makedirs(d, exist_ok=True)

    if analyze_emb:
        quality_histograms = []
        quality_dict = {
        'avg_distance': [],
        'min_distance': [],
        'rankme': [],
        'zero_ratio_per_row': []
    }
        
    for epoch in range(1, epochs+1):
        epoch_loss = []
        
        for idx, batch in enumerate(dataloader):
            # Get batch
            patient_ids, modalities, X_dict, avail_mods, batch_targets, targets_names = batch
            # print(batch_targets.size())
            # Forward pass
            contrastive_outputs = mme(X_dict)
            # print(predictive_outputs.size())
            contrastive_loss_batch = (contrastive_outputs, patient_ids, avail_mods)
    
            # Loss computation 
            for modality in contrastive_outputs.keys():
                optimizer_dict[modality].zero_grad()
                contrastive_loss_batch = ({modality:contrastive_outputs[modality]}, patient_ids, avail_mods)
                modality_loss = contrastive_loss_dict[modality](contrastive_loss_batch)
                if regularizer_dict:
                    regularization_term = regularizer_dict[modality](contrastive_outputs[modality])
                    modality_loss = torch.add(regularization_term, modality_loss)
                modality_loss.backward()
                optimizer_dict[modality].step()
                scheduler_dict[modality].step()
                optimizer_dict[modality].zero_grad()
                modality_batch_loss[modality].append(modality_loss.item())
    
        for modality in modality_batch_loss.keys():
            modality_avg_loss[modality].append(np.mean(modality_batch_loss[modality]))
            # print(f"Epoch {epoch} {modality} total loss: {modality_avg_loss[modality][-1]:.4f}".upper())
    
        total_cross_modality_avg_loss = np.mean([epoch_losses[-1] for epoch_losses in modality_avg_loss.values()])
        print(f"Epoch {epoch} average total loss across modalities: {total_cross_modality_avg_loss:.4f}".upper())
    
        EPOCHS_I_LOSSES.append(total_cross_modality_avg_loss)

        if make_gifs:
            # capture embeddings figures instead of immediate plotting
            figs = see_emb(batch_all, mme, reducer='tsne')
            for i, fig in enumerate(figs):
                path = os.path.join(emb_frame_dirs[i], f'epoch_{epoch:03d}.png')
                fig.savefig(path, dpi=150, bbox_inches='tight')
                plt.close(fig)
        if analyze_emb:
            mme_embeddings = concat_modality_embeddings(new_framework_inference(batch_all, mme)[0])

            if epoch == 1 or epoch == epochs:
                hist = True
            else:
                hist = False
            quality_dict_epoch = compute_embedding_quality_metrics(mme_embeddings,
                                              zero_threshold = 1e-3,
                                              hist=hist,
                                              hist_bins = 100)
            if epoch == 1 or epoch == epochs:
                quality_histograms.append([quality_dict_epoch['hist_counts'], 
                                           quality_dict_epoch['hist_bins'],
                                           quality_dict_epoch['hist_values']
                                          ])
            for quality_key in quality_dict.keys():
                quality_dict[quality_key].append(quality_dict_epoch[quality_key])
            # with torch.no_grad():
            #     # Analyser les embeddings tous les N epochs
            #     if epoch % 5 == 0 or epoch == epochs - 1:
            #         save_dir = f"embedding_analysis/epoch_{epoch}"
            #         os.makedirs(save_dir, exist_ok=True)
            #         modality_keys = mme.modality_net_map.keys()
            #         metrics = analyze_embeddings_after_epoch(
            #             model=mme,
            #             dataloader=dataloader,
            #             modality_keys=modality_keys,
            #             patient_map=patient_map,
            #             device=mme.modality_net_map[list(mme.modality_net_map.keys())[0]].device,
            #             num_batches=-1,  # Analyser 2 batchs
            #             visualize=True,
            #             save_dir=save_dir
            #         )
                    
            #         embedding_metrics.append({
            #             'epoch': epoch,
            #             'metrics': metrics
            #         })
                    
            #         # Visualiser l'évolution des métriques
            #         if len(embedding_metrics) > 1:
            #             visualize_metrics_evolution(embedding_metrics, save_dir=save_dir)
    # Prepare data
    epochs_range = list(range(1, len(EPOCHS_I_LOSSES) + 1))
    global_avg = EPOCHS_I_LOSSES
    
    # Plot style
    plt.style.use('seaborn-v0_8-whitegrid')  # clean, modern grid
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot per-modality curves
    for modality, losses in modality_avg_loss.items():
        ax.plot(epochs_range, losses, marker='o', linewidth=2, alpha=0.8, label=f'{modality} Loss')
    
    # Plot global cross-modality average
    ax.plot(epochs_range, global_avg,
            marker='s', linewidth=3, linestyle='--',
            color='black', label='Global Avg Loss')
    
    # Aesthetic tweaks
    ax.set_title('Epoch-wise Contrastive Loss per Modality & Global Average', fontsize=16, weight='bold')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Average Loss', fontsize=14)
    ax.set_xticks(epochs_range)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(title='Legend', fontsize=12, title_fontsize=13, loc='upper right', frameon=True)
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    if analyze_emb:
        # Histogram values
        counts = quality_histograms[0][0]
        bins = quality_histograms[0][1]
        values = quality_histograms[0][2]
        plt.bar(bins[:-1], counts, width=np.diff(bins))
        M, m = bins[-1], bins[0]
        p = 95
        median, p_val = np.median(values), np.percentile(values, p)
        plt.title(f"Histogram of embedding values: Epoch 1\nMax: {M:.4f}, Min: {m:.4f}, Median: {median:.4f}, {p}th: {p_val:.4f} \nAvg near zero ratio = {quality_dict['zero_ratio_per_row'][0]:.4f}, RankMe = {quality_dict['rankme'][0]:.4f}")
        plt.show()
        
        counts = quality_histograms[1][0]
        bins = quality_histograms[1][1]
        values = quality_histograms[1][2]
        plt.bar(bins[:-1], counts, width=np.diff(bins))
        M, m = bins[-1], bins[0]
        median, p_val = np.median(values), np.percentile(values, p)
        plt.title(f"Histogram of embedding values: Epoch {epochs}\nMax: {M:.4f}, Min: {m:.4f}, Median: {median:.4f}, {p}th: {p_val:.4f} \nAvg near zero ratio = {quality_dict['zero_ratio_per_row'][-1]:.4f}, RankMe = {quality_dict['rankme'][-1]:.4f}")
        plt.show()
        
        quality_df = pd.DataFrame(quality_dict)
        for metric in quality_df.columns:
            sns.lineplot(quality_df, x=range(epochs), y=metric)
            plt.title(f"Evolution of {metric} over epochs")
            plt.show()
    if make_gifs:
        make_matplotlib_gif('emb_mme', 'emb_mme_animation.gif', fps=4, max_frames=epochs)
        make_matplotlib_gif('emb_by_mod', 'emb_by_modality_animation.gif', fps=4, max_frames=epochs)
        make_matplotlib_gif('emb_by_patient', 'emb_by_patient_animation.gif', fps=4, max_frames=epochs)

def new_predictive_learning(mme, clm, 
                          t_dataloader,
                          epochs, 
                          base_lr, 
                          optimizer,
                          scheduler,
                          clf_targets_dict,
                          v_dataloader=None,
                          reg_loss_fn = nn.MSELoss(),
                          clf_loss_fn = nn.CrossEntropyLoss(),
                          batch_all=None,
                          make_gifs=False,
                          show_plots=False,
                          device=None,
                          mme_emb_path=None,
                          ):
    device = device if device is not None else clm.device
    tasks = list(clm.config['predictive_cfg']['heads_configs'].keys())
    clf_targets = list(clf_targets_dict.keys())
    assert set(clf_targets).intersection(set(tasks)) == set(clf_targets), f"All classification targets ({clf_targets}) must be in tasks ({tasks})."

    device = clm.device
    t_epoch_losses_dict = {key:[] for key in tasks}
    t_epochs_losses = []
    if v_dataloader is not None:
        v_epoch_losses_dict = {key:[] for key in tasks}
        v_epochs_losses = []
    if make_gifs and batch_all is None:
        print("No batch_all specified, setting make_gifs to False.")
        make_gifs = False
        
    if make_gifs:
        # Save embedding plots
        emb_frame_dirs = ['emb_clm']
        for d in emb_frame_dirs:
            os.makedirs(d, exist_ok=True)
    if mme is None:
        with open(mme_emb_path, "rb") as f:
            contrastive_outputs = pkl.load(f)
            
    for epoch in range(1, epochs+1):
        t_batch_losses = {key:[] for key in tasks}
        t_overall_losses = []
        if v_dataloader is not None:
            v_batch_losses = {key:[] for key in tasks}
            v_overall_losses = []
        # for idx, batch in enumerate(t_dataloader):
        #     # Get batch
        #     patient_ids, modalities, X_dict, avail_mods, batch_targets, targets_names = batch
        #     # print(batch_targets.size())
        #     # Forward pass
        #     if mme is not None:
        #         mme.eval()
        #         contrastive_outputs = mme(X_dict)
            
        #     clm.train()
        #     task_outputs, patient_embeddings = clm(contrastive_outputs, avail_mods)
            
        #     overall_loss = torch.tensor(0, device=device)
        #     # Regression losses
        #     for i, key in enumerate(task_outputs.keys()):
        #         if key not in clf_targets:
        #             mse_loss = reg_loss_fn(task_outputs[key].squeeze(), batch_targets[:,i])
        #             overall_loss = torch.add(overall_loss, mse_loss)
        #             batch_losses[key].append(mse_loss.item())
                    
        #     # Classification losses
        #     for i, key in enumerate(clf_targets):
        #         if i == 0:
        #             start_idx = len(task_outputs.keys()) - len(clf_targets)
        #         else:
        #             start_idx = end_idx - 1 # last end_idx
        #         # clf_targets_dict is a dict that maps each clf target to the number of classes it holds
        #         # print("key", key)
        #         end_idx = start_idx + clf_targets_dict[key]
        #         # print("start/end", start_idx, end_idx)
        #         # print("batch targets", batch_targets[:,start_idx:end_idx])
        #         # print("model output", task_outputs[key])
        #         clf_loss = clf_loss_fn(task_outputs[key], batch_targets[:,start_idx:end_idx])
        #         overall_loss = torch.add(overall_loss, clf_loss)
        #         batch_losses[key].append(clf_loss.item())
    
        #     overall_loss.backward()
        #     overall_losses.append(overall_loss.item())
            
        #     optimizer.step()
        #     scheduler.step()
            
        # for key in task_outputs.keys():
        #     epoch_losses[key].append(np.mean(batch_losses[key]))
        # EPOCHS_II_LOSSES.append(np.mean(overall_losses))
        # print(f"Epoch {epoch} total loss: {EPOCHS_II_LOSSES[-1]:.4f}".upper())
        predictive_step(epoch, 
                    t_dataloader, 
                    'train',
                    mme, 
                    clm, 
                    clf_targets, 
                    clf_targets_dict,
                    t_epoch_losses_dict, 
                    t_epochs_losses, 
                    reg_loss_fn, 
                    clf_loss_fn, 
                    t_batch_losses, 
                    t_overall_losses,
                    device, 
                    optimizer=optimizer, 
                    scheduler=scheduler, 
                    )

        predictive_step(epoch, 
                    v_dataloader, 
                    'eval',
                    mme, 
                    clm, 
                    clf_targets, 
                    clf_targets_dict,
                    v_epoch_losses_dict, 
                    v_epochs_losses, 
                    reg_loss_fn, 
                    clf_loss_fn, 
                    v_batch_losses, 
                    v_overall_losses,
                    device, 
                    optimizer=None, 
                    scheduler=None, 
                    )
        print(f"Epoch {epoch} total training loss: {t_epochs_losses[-1]:.4f}\nEpoch {epoch} total validation loss: {v_epochs_losses[-1]:.4f}".upper(), end='\r')
        if make_gifs:
            # Inference on all original patient samples
            patient_ids, modalities, X_dict, avail_mods, batch_targets, targets_names = batch_all
            if mme is not None:
                mme.eval()
                contrastive_outputs = mme(X_dict)
            clm.eval()
            _, true_patient_embeddings = clm(contrastive_outputs, avail_mods)
            figs = see_emb(reducer='tsne', patient_embs=true_patient_embeddings)
            for i, fig in enumerate(figs):
                path = os.path.join(emb_frame_dirs[i], f'epoch_{epoch:03d}.png')
                fig.savefig(path, dpi=150, bbox_inches='tight')
                plt.close(fig)
    # # Prepare data
    # epochs_range = list(range(1, len(EPOCHS_II_LOSSES) + 1))
    # global_loss = EPOCHS_II_LOSSES
    
    # # Plot style
    # plt.style.use('seaborn-v0_8-whitegrid')  # clean, modern grid
    
    # # Plot per-modality curves
    # for task, losses in epoch_losses.items():
    #     plt.figure(figsize=(10,6))
    #     plt.plot(epochs_range, losses, marker='o', linewidth=2, alpha=0.8, label=f'{task} Loss')
    #     plt.title(f'Epoch-wise {task} Loss', fontsize=16, weight='bold')
    #     plt.xlabel('Epoch', fontsize=14)
    #     plt.ylabel('Average Loss', fontsize=14)
    #     plt.xticks(epochs_range)
    #     plt.tick_params(axis='both', which='major', labelsize=12)
    #     plt.legend(title='Legend', fontsize=12, title_fontsize=13, loc='upper right', frameon=True)
    #     plt.grid(True, which='major', linestyle='-', alpha=0.3)
        
    #     plt.tight_layout()
    #     plt.show()
    
    # # Plot global loss
    # plt.figure(figsize=(10,6))
    # plt.plot(epochs_range, global_loss,
    #         marker='s', linewidth=3, linestyle='--',
    #         color='black', label='Global Loss (Sum)')
    
    # # Aesthetic tweaks
    # plt.title('Epoch-wise Global Loss', fontsize=16, weight='bold')
    # plt.xlabel('Epoch', fontsize=14)
    # plt.ylabel('Average Loss', fontsize=14)
    # plt.xticks(epochs_range)
    # plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.legend(title='Legend', fontsize=12, title_fontsize=13, loc='upper right', frameon=True)
    # plt.grid(True, which='major', linestyle='-', alpha=0.3)
    
    # plt.tight_layout()
    # plt.show()
    if show_plots:
        predictive_plots(t_epoch_losses_dict, t_epochs_losses, mode='train')
        if v_dataloader is not None:
            predictive_plots(v_epoch_losses_dict, v_epochs_losses, mode='val')
    
    if make_gifs:
        make_matplotlib_gif('emb_clm', 'emb_clm_animation.gif', fps=4, max_frames=epochs)

def predictive_step(epoch, 
                    dataloader, 
                    mode,
                    mme, 
                    clm, 
                    clf_targets, 
                    clf_targets_dict,
                    epoch_losses_dict, 
                    epochs_losses, 
                    reg_loss_fn, 
                    clf_loss_fn, 
                    batch_losses, 
                    overall_losses,
                    device, 
                    optimizer=None, 
                    scheduler=None, 
                    ):
    prefix = 'training' if mode == 'train' else 'validation'
    for idx, batch in enumerate(dataloader):
        # Get batch
        patient_ids, modalities, X_dict, avail_mods, batch_targets, targets_names = batch
        # print(batch_targets.size())
        # Forward pass
        if mme is not None:
            mme.eval()
            contrastive_outputs = mme(X_dict)

        if mode == 'train':
            clm.train()

            task_outputs, patient_embeddings = clm(contrastive_outputs, avail_mods)
        
            overall_loss = torch.tensor(0, device=device)
            # Regression losses
            for i, key in enumerate(task_outputs.keys()):
                if key not in clf_targets:
                    mse_loss = reg_loss_fn(task_outputs[key].squeeze(), batch_targets[:,i])
                    overall_loss = torch.add(overall_loss, mse_loss)
                    batch_losses[key].append(mse_loss.item())
                    
            # Classification losses
            for i, key in enumerate(clf_targets):
                if i == 0:
                    start_idx = len(task_outputs.keys()) - len(clf_targets)
                else:
                    start_idx = end_idx - 1 # last end_idx
                # clf_targets_dict is a dict that maps each clf target to the number of classes it holds
                # print("key", key)
                end_idx = start_idx + clf_targets_dict[key]
                # print("start/end", start_idx, end_idx)
                # print("batch targets", batch_targets[:,start_idx:end_idx])
                # print("model output", task_outputs[key])
                clf_loss = clf_loss_fn(task_outputs[key], batch_targets[:,start_idx:end_idx])
                overall_loss = torch.add(overall_loss, clf_loss)
                batch_losses[key].append(clf_loss.item())
    
            overall_loss.backward()
            overall_losses.append(overall_loss.item())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        else:
            clm.eval()
            with torch.no_grad():
                task_outputs, patient_embeddings = clm(contrastive_outputs, avail_mods)
        
                overall_loss = torch.tensor(0, device=device)
                # Regression losses
                for i, key in enumerate(task_outputs.keys()):
                    if key not in clf_targets:
                        mse_loss = reg_loss_fn(task_outputs[key].squeeze(), batch_targets[:,i])
                        overall_loss = torch.add(overall_loss, mse_loss)
                        batch_losses[key].append(mse_loss.item())
                        
                # Classification losses
                for i, key in enumerate(clf_targets):
                    if i == 0:
                        start_idx = len(task_outputs.keys()) - len(clf_targets)
                    else:
                        start_idx = end_idx - 1 # last end_idx
                    # clf_targets_dict is a dict that maps each clf target to the number of classes it holds
                    # print("key", key)
                    end_idx = start_idx + clf_targets_dict[key]
                    # print("start/end", start_idx, end_idx)
                    # print("batch targets", batch_targets[:,start_idx:end_idx])
                    # print("model output", task_outputs[key])
                    clf_loss = clf_loss_fn(task_outputs[key], batch_targets[:,start_idx:end_idx])
                    overall_loss = torch.add(overall_loss, clf_loss)
                    batch_losses[key].append(clf_loss.item())
        
                overall_losses.append(overall_loss.item())
        
    for key in task_outputs.keys():
        epoch_losses_dict[key].append(np.mean(batch_losses[key]))
    epochs_losses.append(np.mean(overall_losses))
    # print(f"Epoch {epoch} total {prefix} loss: {epochs_losses[-1]:.4f}".upper(), end='\r')

def predictive_plots(epoch_losses_dict, epochs_losses, mode):
    # Prepare data
    epochs_range = list(range(1, len(epochs_losses) + 1))
    global_loss = epochs_losses
    prefix = 'training' if mode == 'train' else 'validation'
    # Plot style
    plt.style.use('seaborn-v0_8-whitegrid')  # clean, modern grid
    
    # Plot per-modality curves
    for task, losses in epoch_losses_dict.items():
        plt.figure(figsize=(10,6))
        plt.plot(epochs_range, losses, marker='o', linewidth=2, alpha=0.8, label=f'{task} Loss')
        plt.title(f'Epoch-wise {task} {prefix} Loss', fontsize=16, weight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Average Loss', fontsize=14)
        plt.xticks(epochs_range)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(title='Legend', fontsize=12, title_fontsize=13, loc='upper right', frameon=True)
        plt.grid(True, which='major', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Plot global loss
    plt.figure(figsize=(10,6))
    plt.plot(epochs_range, global_loss,
            marker='s', linewidth=3, linestyle='--',
            color='black', label='Global Loss (Sum)')
    
    # Aesthetic tweaks
    plt.title(f'Epoch-wise Global {prefix} Loss', fontsize=16, weight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Average Loss', fontsize=14)
    plt.xticks(epochs_range)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(title='Legend', fontsize=12, title_fontsize=13, loc='upper right', frameon=True)
    plt.grid(True, which='major', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
def build_run_name(model,
                   epochs,
                   base_lr,
                   eta_min_coef,
                   scheduler_class,
                   scheduler_cfg,
                   contrastive_loss_cfg,
                   modality_list=None,
                   max_len=250):
    """
    Build a concise run name encoding key training & model parameters.
    """
    parts = []
    # 1) Model
    parts.append(model.__class__.__name__)
    # 2) Epochs & LR
    parts.append(f"E{epochs}")
    parts.append(f"lr{base_lr:.0e}")
    # 3) Scheduler
    sch_name = scheduler_class.__name__
    parts.append(sch_name)
    # include eta_min
    if "eta_min" in scheduler_cfg:
        # eta_min might be float or dynamic*coef
        parts.append(f"etamin{scheduler_cfg['eta_min']:.0e}")
    # 4) Contrastive loss core params
    temp = contrastive_loss_cfg.get("temperature", None)
    sim  = contrastive_loss_cfg.get("similarity", None)
    if temp is not None:
        parts.append(f"t{temp}")
    if sim:
        parts.append(sim.replace("-",""))
    # 5) Modalities (either list or count)
    if modality_list:
        # if many, just count
        if len(modality_list) <= 3:
            parts.append("mod" + "-".join(modality_list))
        else:
            parts.append(f"mod{len(modality_list)}")
    # join
    name = "_".join(parts)
    # enforce max length
    if len(name) > max_len:
        # truncate at max_len (but try not to cut in the middle of a token)
        name = name[:max_len].rsplit("_", 1)[0]
    return name
    
def train_step(module: nn.Module, batch: tuple, batch_idx: int, mode: str, criterion: nn.Module | Any, optimizer: torch.optim.Optimizer):
    """
    Performs a single training step.

    Args:
        module: The model being trained.
        batch: A batch of data.
        batch_idx: Index of the batch.
        criterion: Loss function.
        optimizer: Optimizer object.

    Returns:
        Updated module and computed loss.
    """
    if mode == "patient":
        # Retrieve data (should be from the PatientWiseDataloader)
        x_dict, patient_list, available_mods_batch = batch

        # Compute Embeddings
        model_outputs = module(x_dict)

        # Prepare batch for loss computation
        batch_for_criterion = (model_outputs, patient_list, available_mods_batch)
    elif mode == "predictive":
        # Retrieve data (should be from the PredictiveDataloader)

        # Compute Embeddings

        # Prepare batch for loss computation
        pass
    else:
        raise ValueError(f"Unrecognized mode '{mode}'. Must be either 'patient' or 'predictive'")

    loss = criterion(batch_for_criterion)
    print(f"\n\033[1;37mBatch loss {batch_idx + 1} : {loss.item()}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return module, loss


def eval_step(module: nn.Module, batch: tuple, batch_idx: int, mode: str, criterion: nn.Module | Any, training=True):
    """
    Performs an evaluation step.

    Args:
        module: The model being evaluated.
        batch: A batch of data.
        batch_idx: Index of the batch.
        criterion: Loss function.
        training: True for validation, False for testing.

    Returns:
        Module, loss, and optionally model outputs and targets.
    """
    with torch.no_grad():
        if mode == "patient":
            # Retrieve data (should be from the PatientWiseDataloader)
            x_dict, patient_list, available_mods_batch = batch
            y = None  # Self Supervised learning

            # Compute Embeddings
            model_outputs = module(x_dict)

            # Prepare batch for loss computation
            batch_for_criterion = (model_outputs, patient_list, available_mods_batch)
        elif mode == "predictive":
            # Retrieve data (should be from the PredictiveDataloader)

            # Compute Embeddings

            # Prepare batch for loss computation
            pass
        else:
            raise ValueError(f"Unrecognized mode '{mode}'. Must be either 'patient' or 'predictive'")

        loss = criterion(batch_for_criterion)
        if training:
            print(f"\n\033[1;32mValidation Batch loss {batch_idx + 1} : {loss.item()}")
            return module, loss
        else:
            print(f"\n\033[1;32mTest Batch loss {batch_idx + 1} : {loss.item()}")
            return module, loss, model_outputs, y


def train_loop(
    module: nn.Module,
    EPOCHS: int,
    train_dataset: Dataset,
    val_dataset: Dataset,
    criterion: nn.Module | Any,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    batch_level_scheduler: bool = False,
    n_batches: int = 1,
    save_per_epoch: bool = False,
    save_path: str | None = None,
    return_best_epoch_idx: bool = False,
):
    """
    Executes the full training loop.

    Args:
        module: Model to train.
        EPOCHS: Number of training epochs.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        lr_scheduler: Learning rate scheduler (optional).

    Returns:
        Trained model.
    """
    best_val_loss = 0
    best_epoch = 0
    for epoch in range(EPOCHS):
        module.train(True)
        train_batch_losses = []
        for batch_idx in range(len(train_dataset)):
            batch = next(iter(train_dataset))

            module, loss = train_step(module, batch, batch_idx, criterion, optimizer)
            train_batch_losses.append(loss.item())

            # if we want to have a Learning rate schedule at batch granularity
            if lr_scheduler is not None and batch_level_scheduler:
                if batch_idx % n_batches == 0:
                    print("New LR:", lr_scheduler.get_last_lr()[0])
                    lr_scheduler.step()

        # if we want to have a Learning rate schedule at epoch granularity
        if lr_scheduler is not None and not batch_level_scheduler:
            print(f"Epoch {epoch} LR:", lr_scheduler.get_last_lr()[0])
            lr_scheduler.step()

        # Compute training epoch loss
        train_loss = np.mean(train_batch_losses)

        # Validation
        module.train(False)
        val_batch_losses = []
        for batch_idx in range(len(val_dataset)):
            batch = next(iter(val_dataset))
            module, loss = eval_step(module, batch, batch_idx, criterion)
            val_batch_losses.append(loss.item())

        # Compute validation epoch loss
        val_loss = np.mean(val_batch_losses)

        # Logic to monitor if we are at the current best epoch or not based on validation
        old_best_epoch = best_epoch if epoch > 0 else None
        if epoch == 0:
            best_val_loss = val_loss
            best_epoch = epoch + 1
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1

        # Logging of losses
        module.train_log(train_batch_losses, val_batch_losses, train_loss, val_loss)
        print(f"\n\033[1;33mEpoch {epoch + 1} :\n\033[1;37mTraining Loss : {train_loss}")
        print(f"\033[1;32mValidation Loss : {val_loss}")

        # If we want to save models at each best epoch
        if save_per_epoch and ((old_best_epoch is None) or (old_best_epoch != best_epoch)):
            if save_path is not None:
                if old_best_epoch is not None:
                    print("\nRemoving previous best..")
                    shutil.rmtree(save_path + f"_{old_best_epoch}")  # removes folder and files of previous sved state

                print("\nSaving current best..")
                save_path_epoch = save_path + f"_{best_epoch}"
                module.save_model(save_path_epoch)
            else:
                raise ValueError("save_per_epoch was set to True but save_path is None. You must specify a save_path.")
    if return_best_epoch_idx:
        return module, best_epoch
    return module