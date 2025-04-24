import numpy as np
import shutil

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import warnings


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
