import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import warnings

from torchviz import make_dot

# INFONCE LOSS PSEUDOCODE FOR OUR IDEA
# X_dict, patient_ids, available_modalities = batch
# bank = concatenation of all 6 modality tensors of the batch (batch_size * 6, model_output_size)
# bank_ids = same for the ids
# batch_loss = 0
# for patient in patient_ids:
#     missing_positions = Where in the bank tensor are stored missing tensors for the patient
#     positive_mask = If id is the same as patient and not missing
#     negative_mask = everything else

#     unfiltered_similarities = compute exp(dot product on each row of bank for each modality emb)
#     # = tensor of size batch_size * 6, n_patient_modalities

#     # => dot products of 0 will correspond to where we computed similarity between present and missing modalities
#     # or with completely orthogonal embeddings

#     # => this means that in our tensor, absolute 1 (exp(0)) need to be distinguished between actual orthogonal embeddings
#     # and simply missing modalities artifcats

#     # To do that set 1 to 0 on missing_positions for each of the n_patient_modalities columns

#     # Not necessary if we filter out missing modalities from the similarity computation step !

#     for modality in patient_modalities:
#         # We can only compute loss for one possible positive reference or for all of them
#         numerator = sample one positive similarity (among positive indices, choose randomly one values in the columns)
#         denominator = sum the modality column but mask out the patient itself as well as other patient modalities
#         denominator += numerator
#         loss = - log(numerator/denominator)
#         batch_loss += loss
# batch_loss.backward()


class NoPositivePairWarning(UserWarning):
    pass


class InfoNCELoss(nn.Module):
    """
    Implementation of InfoNCE loss for multimodal contrastive learning with missing modalities support.

    This implementation handles batches of patients with multiple modalities, some of which may be missing.
    It computes contrastive loss between modalities from the same patient (positives) vs. different patients (negatives).
    """

    def __init__(self, modalities: List[str], 
                 patient_map: Dict[str, int] = None, 
                 temperature: float = 0.07, 
                 similarity: str = "original", 
                 use_all_positives: bool = True):
        """
        Args:
            temperature: Scaling factor for the similarity scores
            use_all_positives: If True, use all positive pairs; if False, sample one positive pair per modality
        """
        super().__init__()
        self.temperature = temperature
        self.similarity = similarity

        if temperature is None and similarity == "original":
            raise ValueError("Cannot use temperature == None (logit-scaling) when similarity is original.")
        self.use_all_positives = use_all_positives
        self.modality_keys = modalities
        self.patient_map = patient_map # to translate str patient_ids to unique numbers
        self.eps = 1e-8  # For numerical stability

    def forward(self, batch) -> torch.Tensor:
        """
        Forward pass to compute the InfoNCE loss.

        Args:
            batch: A tuple containing (X_dict, patient_ids, available_modalities)
                X_dict: Dictionary with modality names as keys and embedding tensors
                patient_ids: List of patient IDs for the batch
                available_modalities: Tensor of shape [batch_size, num_modalities] indicating available modalities

        Returns:
            torch.Tensor: The computed InfoNCE loss
        """
        X_dict, patient_ids, available_modalities = batch

        # Create a unified embedding bank from all modalities
        bank = []
        bank_ids = []
        bank_modality_indices = []

        batch_size = len(patient_ids)
        device = next(iter(X_dict.values())).device

        # Gather all embeddings and their metadata
        for mod_idx, modality in enumerate(self.modality_keys):
            embeddings = X_dict[modality]  # Shape: [batch_size, embedding_dim]

            for patient_idx in range(batch_size):
                # Check if this modality is available for this patient
                if available_modalities[patient_idx, mod_idx] == 1:
                    bank.append(embeddings[patient_idx].view(-1))
                    bank_ids.append(patient_ids[patient_idx])
                    bank_modality_indices.append(mod_idx)

        # Convert lists to tensors
        bank = torch.stack(bank, dim=0)  # Shape: [total_present_modalities, embedding_dim]
        bank_ids = torch.tensor([self.patient_map[patient_id] for patient_id in bank_ids], device=device)
        bank_modality_indices = torch.tensor(bank_modality_indices, device=device)

        if bank.size(0) <= 1:
            # If there's only one modality tensor in the bank, we can't compute contrastive loss
            raise ValueError("Only one modality found in the bank. There is something wrong with the batch.")

        # Compute similarity matrix
        if self.similarity == "original": # Original InfoNCE uses dot product similarity
            similarities = torch.matmul(bank, bank.T) 
        else: # cosine similarity (used in NT-Xent loss which is a variant of the InfoNCELoss)
            norms = bank.norm(dim=1, keepdim=True).clamp(min=self.eps) # comute per-vector norms
            bank_normed = bank / norms # normalize each vector by its norm
            similarities = bank_normed @ bank_normed.T # compute similarity
            
        similarities = similarities / self.temperature  # [total_present_modalities, total_present_modalities]
        sim_exp = torch.exp(similarities)
            
        # print("Exp similarity matrix", sim_exp)
        batch_loss = torch.tensor(0.0, device=device)
        processed_pairs = 0

        # Process each patient separately
        for patient_id in patient_ids:
            # Find embeddings for this patient
            patient_mask = bank_ids == self.patient_map[patient_id]
            patient_indices = patient_mask.nonzero()
            patient_modality_indices = bank_modality_indices[patient_mask]

            # If no positive pairs are possible for this patient, skip
            if patient_mask.sum() == 0:
                warnings.warn(f"No possible positive pair found for patient {patient_id}", NoPositivePairWarning)
                continue

            # Process each available modality for this patient
            for idx, mod_idx in zip(patient_indices, patient_modality_indices):
                # Create positive mask (same patient, different modality)
                positive_mask = patient_mask.clone()
                positive_mask[idx] = False  # Exclude self

                # Negative mask (different patients)
                negative_mask = ~patient_mask

                # Compute numerator: sum of exp similarities with positive pairs
                if self.use_all_positives:
                    # Use all positive pairs
                    numerator = sim_exp[idx].view(-1)[positive_mask].sum()
                else:
                    # Sample one positive randomly
                    random_pos_idx = patient_indices[torch.randint(0, len(patient_indices), (1,))]
                    numerator = sim_exp[idx].view(-1)[random_pos_idx]

                # Compute denominator: sum of all similarities (excluding self)
                all_mask = positive_mask | negative_mask
                denominator = sim_exp[idx].view(-1)[all_mask].sum()

                # print("NUMERATOR", numerator)
                # print("DENOMINATOR", denominator)
                # print("RATIO", numerator / (denominator + self.eps))
                # Compute loss
                loss = -torch.log(numerator / (denominator + self.eps))
                # print("-LOG (loss)", loss)
                batch_loss = torch.add(batch_loss,loss)
                processed_pairs += 1

        # Average loss over all processed pairs
        # print(batch_loss)
        if processed_pairs > 0:
            batch_loss = torch.div(batch_loss,processed_pairs)
            # print(batch_loss)
        else:
            raise ValueError("Something went wrong, no pairs were processed.")
        return batch_loss

# class InfoNCELoss(nn.Module):
#     """
#     Implementation of InfoNCE loss for multimodal contrastive learning with missing modalities support,
#     with debug prints to catch where gradients might be lost.
#     """

#     def __init__(self, modalities, patient_map=None, temperature=0.1, use_all_positives=True):
#         super().__init__()
#         self.temperature = temperature
#         self.use_all_positives = use_all_positives
#         self.modality_keys = modalities
#         self.patient_map = patient_map
#         self.eps = 1e-8

#     def forward(self, batch):
#         X_dict, patient_ids, available_modalities = batch
#         device = next(iter(X_dict.values())).device

#         # Build bank
#         bank, bank_ids, bank_mods = [], [], []
#         for m_idx, m in enumerate(self.modality_keys):
#             emb = X_dict[m]
#             for p_idx in range(len(patient_ids)):
#                 if available_modalities[p_idx, m_idx] == 1:
#                     v = emb[p_idx].view(-1)
#                     v.retain_grad()
#                     bank.append(v)
#                     bank_ids.append(self.patient_map[patient_ids[p_idx]])
#                     bank_mods.append(m_idx)
#         bank = torch.stack(bank, 0).to(device)
#         bank.retain_grad()
#         bank_ids = torch.tensor(bank_ids, device=device)
#         bank_mods = torch.tensor(bank_mods, device=device)

#         # Early exit
#         if bank.size(0) <= 1:
#             raise ValueError("Need ≥2 embeddings in bank")

#         # Similarity & exp
#         sim = (bank @ bank.T) / self.temperature
#         sim.retain_grad()
#         exp_sim = torch.exp(sim)
#         exp_sim.retain_grad()

#         total_loss = torch.tensor(0., device=device, requires_grad=True)
#         total_loss.retain_grad()
#         count = 0

#         # For each patient
#         for pid in patient_ids:
#             pid_num = self.patient_map[pid]
#             mask = bank_ids == pid_num
#             idxs = mask.nonzero(as_tuple=False).view(-1)

#             if idxs.numel() < 2:
#                 warnings.warn(f"No pos pairs for {pid}")
#                 continue

#             # For each modality embedding of this patient
#             for idx in idxs:
#                 # Positive mask (same patient, different modality)
#                 pos_mask = mask.clone()
#                 pos_mask[idx] = False
#                 # Negative mask (everything else)
#                 neg_mask = ~mask

#                 # Numerator: sum over positives (or sample one)
#                 pos_vals = exp_sim[idx][pos_mask]
#                 pos_vals.retain_grad()
#                 if self.use_all_positives:
#                     numer = pos_vals.sum()
#                 else:
#                     # sample one positive
#                     weights = pos_vals.detach()
#                     choice = torch.multinomial(weights, 1)
#                     numer = pos_vals[choice]
#                 numer.retain_grad()

#                 # Denominator: sum over all except self, mask missing
#                 all_vals = exp_sim[idx][pos_mask | neg_mask]
#                 all_vals.retain_grad()
#                 denom = all_vals.sum() + self.eps
#                 denom.retain_grad()

#                 loss_ij = -torch.log(numer / denom + self.eps)
#                 loss_ij.retain_grad()
#                 total_loss = torch.add(total_loss,loss_ij)
#                 total_loss.retain_grad()
#                 count += 1

#                 # --- DEBUG: backward this partial loss ---
#                 total_loss.backward(torch.ones_like(total_loss), retain_graph=True)

#                 print(f"\nAfter backward of loss for bank idx {idx.item()}:")
#                 print(" bank.grad_fn:", bank.grad_fn)
#                 print(" sim.grad_fn:", sim.grad_fn)
#                 print(" exp_sim.grad_fn:", exp_sim.grad_fn)
#                 print(" numer.grad_fn:", numer.grad_fn if hasattr(numer, 'grad_fn') else numer.grad)
#                 print(" denom.grad_fn:", denom.grad_fn if hasattr(denom, 'grad_fn') else denom.grad)
#                 print(" pos_vals.grad:", pos_vals.grad)
#                 print(" all_vals.grad:", all_vals.grad)
#                 print(" loss_ij.grad_fn:", loss_ij.grad_fn)
#                 print(" total_loss.grad:", total_loss.grad)

#                 # zero grads before next iteration
#                 self.zero_grad()

class RegularizedInfoNCELoss(Module):
    def __init__(self, alpha: float,
                 modalities: List[str], 
                 patient_map: Dict[str, int] = None, 
                 temperature: float = 0.07, 
                 similarity: str = "original", 
                 use_all_positives: bool = True):
        super().__init__()
        self.infonce = InfoNCELoss(modalities=modalities, 
                                   patient_map=patient_map, 
                                   temperature=temperature, 
                                   similarity=similarity, 
                                   use_all_positives=use_all_positives)
        self.alpha = alpha

    def forward(self, batch) -> torch.Tensor:
        nce_loss = self.infonce(batch)

        X_dict, _, _ = batch
        for mod in X_dict.keys():
            