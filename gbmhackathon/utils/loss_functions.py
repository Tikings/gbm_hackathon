import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import warnings

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

    def __init__(self, modalities: List[str], patient_map: Dict[str, int] = None, temperature: float = 0.1, use_all_positives: bool = True):
        """
        Args:
            temperature: Scaling factor for the similarity scores
            use_all_positives: If True, use all positive pairs; if False, sample one positive pair per modality
        """
        super().__init__()
        self.temperature = temperature
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
        similarities = torch.matmul(bank, bank.T) / self.temperature  # [total_present_modalities, total_present_modalities]
        sim_exp = torch.exp(similarities)

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
                    pos_indices = torch.where(positive_mask)[0]
                    random_pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,))]
                    numerator = sim_exp[idx][random_pos_idx]

                # Compute denominator: sum of all similarities (excluding self)
                all_mask = positive_mask | negative_mask
                denominator = sim_exp[idx].view(-1)[all_mask].sum()

                # Compute loss
                loss = -torch.log(numerator / (denominator + self.eps))
                batch_loss = batch_loss + loss
                processed_pairs += 1

        # Average loss over all processed pairs
        if processed_pairs > 0:
            batch_loss = batch_loss / processed_pairs
        else:
            raise ValueError("Something went wrong, no pairs were processed.")
        return batch_loss
