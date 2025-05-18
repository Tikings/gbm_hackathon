import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Callable
import warnings

from gbmhackathon.utils.module_functions import enforce_signature_types
from gbmhackathon.s3_loader import load_s3
from torchviz import make_dot

# INFONCE LOSS PSEUDOCODE FOR OUR IDEA
# out_dict, patient_ids, available_modalities = batch
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

def make_patient_map(dataset, missing_mods_path: str = "s3://abstra-project-storage-lttemftb/1b75dc89-ad27-4a65-9e7f-877d1b4f36fc/missing_mod_per_samples.pkl"):
    missing_mods = load_s3(missing_mods_path)
    all_ids = list(missing_mods.keys())
    patient_map = {key: i for key, i in zip(all_ids, [k for k in range(1,len(all_ids)+1)])}
    for patient_id, idx in patient_map.items():
        if patient_id.endswith('b'): # pour les rechutes on met le même index que le sample original
            patient_map[patient_id] = idx - 1
    patient_number = []
    for pid in dataset.ind2patient.values():
        if 'd' in pid:
            pid = pid[:pid.index('_d')]
        patient_number.append(patient_map[pid])
    patient_map = dict(zip(dataset.ind2patient.values(), patient_number))
    idx_compatible_mapping = dict(zip(list(set(patient_map.values())), [k for k in range(len(set(patient_map.values())))]))
    for pid in patient_map.keys():
        patient_map[pid] = idx_compatible_mapping[patient_map[pid]]
    return patient_map
    
class InfoNCELoss(nn.Module):
    """
    Implementation of InfoNCE loss for multimodal contrastive learning with missing modalities support.

    This implementation handles batches of patients with multiple modalities, some of which may be missing.
    It computes contrastive loss between modalities from the same patient (positives) vs. different patients (negatives).
    """
    @enforce_signature_types
    def __init__(self, modalities: List[str], 
                 patient_map: Dict[str, int] = None, 
                 mode: str = 'modality_scale',#'patient_scale', #
                 temperature: float = 0.07, 
                 similarity: str = "original", 
                 use_all_positives: bool = True,
                 eps: float = 1e-7,
                 warn: bool = True):
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
        self.inverse_patient_map = {val:key for key, val in self.patient_map.items()}
        self.eps = eps  # For numerical stability
        self.mode = mode
        self.warn = warn
        self.pos_alignments = []
        self.neg_alignments = []
        
    def clear_alignments(self):
        self.pos_alignments = []
        self.neg_alignments = []
        print("Alignments cleared")
        
    def forward(self, batch) -> torch.Tensor:
        """
        Forward pass to compute the InfoNCE loss.

        Args:
            batch: A tuple containing (out_dict, patient_ids, available_modalities)
                out_dict: Dictionary with modality names as keys and embedding tensors
                patient_ids: List of patient IDs for the batch
                available_modalities: Tensor of shape [batch_size, num_modalities] indicating available modalities

        Returns:
            torch.Tensor: The computed InfoNCE loss
        """
        out_dict, patient_ids, available_modalities = batch

        # Create a unified embedding bank from all modalities
        bank = []
        bank_ids = []
        bank_modality_indices = []

        batch_size = len(patient_ids)
        device = next(iter(out_dict.values())).device

        if self.mode == 'modality_scale':
            # Gather all embeddings and their metadata
            for mod_idx, modality in enumerate(self.modality_keys):
                embeddings = out_dict[modality]  # Shape: [batch_size, embedding_dim]
    
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
                similarities = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)(bank[None,:,:], bank[:,None,:])
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
                if patient_mask.sum() == 0 and self.warn:
                    warnings.warn(f"No possible positive pair found for patient {patient_id}", NoPositivePairWarning)
                    continue
    
                # Process each available modality for this patient
                for idx, mod_idx in zip(patient_indices, patient_modality_indices):
                    # Create positive mask (same patient, different modality)
                    positive_mask = patient_mask.clone()
                    positive_mask[idx] = False  # Exclude self
    
                    # Negative mask (different patients)
                    negative_mask = ~patient_mask
    
                    # Choose positives
                    if self.use_all_positives:
                        pos_idxs = positive_mask.nonzero(as_tuple=False).view(-1)
                    else:
                        pos_idxs = patient_indices[torch.randint(len(patient_indices), (1,))]
                    neg_idxs = negative_mask.nonzero(as_tuple=False).view(-1)
    
                    # Numerator: sum of positive exponentials
                    numerator = sim_exp[idx, pos_idxs].sum()
                    # print("Numerator", numerator)
                    # print(f"Difference ({patient_id} - {self.inverse_patient_map[bank_ids[pos_idxs].item()]})", bank[idx] - bank[pos_idxs])
                    
                    # Store raw alignment
                    pos_sims = torch.log(sim_exp[idx, pos_idxs]) * self.temperature
                    neg_sims = torch.log(sim_exp[idx, neg_idxs]) * self.temperature
                    self.pos_alignments.extend(pos_sims.detach().cpu().tolist())
                    self.neg_alignments.extend(neg_sims.detach().cpu().tolist())
                        
                    # Compute denominator: sum of all negative similarities + the numerator (excluding self)
                    # all_mask = positive_mask | negative_mask
                    denominator = sim_exp[idx].view(-1)[negative_mask].sum() + numerator
                    # print("Denominator", denominator)
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
        else:
            # Gather all embeddings
            for patient_idx in range(batch_size):
                patient_tensors = [out_dict[mod][patient_idx].view(-1) for mod in out_dict.keys()]
                patient_emb = torch.cat(patient_tensors)
                bank.append(patient_emb)
                bank_ids.append(patient_ids[patient_idx])
    
            # Convert lists to tensors
            bank = torch.stack(bank, dim=0)  # Shape: [total_present_modalities, embedding_dim]
            bank_ids = torch.tensor([self.patient_map[patient_id] for patient_id in bank_ids], device=device)
    
            # Compute similarity matrix
            if self.similarity == "original": # Original InfoNCE uses dot product similarity
                similarities = torch.matmul(bank, bank.T) 
            else: # cosine similarity (used in NT-Xent loss which is a variant of the InfoNCELoss)
                similarities = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)(bank[None,:,:], bank[:,None,:])
            similarities = similarities / self.temperature  # [total_present_modalities, total_present_modalities]
                
            # print("Exp similarity matrix", sim_exp)
            batch_loss = torch.tensor(0.0, device=device)
            processed_pairs = 0
            
            # Process each patient separately
            for i, patient_id in enumerate(patient_ids):
                # Find embeddings for this patient
                mask = bank_ids != self.patient_map[patient_id]
                # Compute maximum distance between the patient and another patient
                batch_loss = torch.add(batch_loss, similarities[i, mask].max())
            return batch_loss
        
class SmoothingFunction(nn.Module):
    @enforce_signature_types
    def __init__(self, bound: float = -10.0, 
                 slope: float = 0.05, 
                 rate: float = -2.0):
        super().__init__()
        self.bound = bound
        self.slope = slope
        self.rate = rate

    def forward(self, x):
        return 0.5 * self.bound - (self.bound / (1 + torch.exp(x / self.rate))) + self.slope * x

def boundary_loss(outputs, min_val=-10, max_val=10):
    # Penalize values below min_val
    below_min = torch.relu(min_val - outputs)
    # Penalize values above max_val
    above_max = torch.relu(outputs - max_val)
    return torch.mean(below_min + above_max)

class RankMe(nn.Module):
    """
    Computes the effective rank (RankMe) of a batch of embeddings.
    RankMe = exp( - sum_k p_k log p_k ),
    where p_k = sigma_k / sum_j sigma_j are normalized singular values.
    """
    @enforce_signature_types
    def __init__(self, eps: float = 1e-12):
        """
        Args:
            eps: small constant to avoid log(0) or division by zero.
        """
        super().__init__()
        self.eps = eps

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: Tensor of shape [B, D] or [B, ... , D], where B is batch size
                        and D is embedding dimensionality. Any leading dimensions
                        will be flattened into the batch.
        
        Returns:
            rankme: scalar tensor, the effective rank of the batch.
        """
        # flatten any extra dims into batch
        x = embeddings.view(-1, embeddings.size(-1))  # [N, D]
        N, D = x.shape

        # compute singular values via SVD
        # Note: torch.linalg.svd may be faster; here we use torch.svd for compatibility
        # U, S, V = torch.svd(x, some=False)
        # Use torch.linalg.svd for newer PyTorch:
        S = torch.linalg.svdvals(x)  # shape [min(N, D)]

        # normalize singular values to get a probability distribution
        s_sum = S.sum() + self.eps
        p = S / s_sum

        # compute entropy of spectrum
        entropy = -(p * torch.log(p + self.eps)).sum()

        # effective rank = exp(entropy)
        rankme = torch.exp(entropy)

        return rankme
        
class RegularizedInfoNCELoss(nn.Module):
    @enforce_signature_types
    def __init__(self,
                 modalities: List[str], 
                 patient_map: Dict[str, int] = None, 
                 temperature: float = 0.07, 
                 similarity: str = "nt-xent", 
                 use_all_positives: bool = True,
                 warn: bool =True, 
                 alpha: float = 0.1,
                 eps: float = 1e-8,
                 bound: float = -10, 
                 slope: float = 0.05, 
                 rate: float = -2,
                 smoothing_func: bool = True
                ):
        super().__init__()
        self.infonce = InfoNCELoss(modalities=modalities, 
                                   patient_map=patient_map, 
                                   temperature=temperature, 
                                   similarity=similarity, 
                                   use_all_positives=use_all_positives,
                                  warn=warn)
        self.bound = bound
        if smoothing_func:
            self.smoothing_func = SmoothingFunction(bound=self.bound, slope=slope, rate=rate)
        else:
            self.smoothing_func = nn.Identity()
        self.alpha = alpha
        self.eps = eps
        # self.rankme_func = RankMe()
        
    def forward(self, batch) -> torch.Tensor:
        nce_loss = self.infonce(batch)
        # print(nce_loss)
        if self.alpha != 0:
            out_dict, _, _ = batch
            N = list(out_dict.values())[0].size(0)  # Number of patients
            
            # Stack all embeddings for vectorized computation
            # Shape: [num_modalities, batch_size, embedding_dim]
            all_embeddings = torch.stack([out_dict[mod].squeeze() for mod in out_dict.keys()])
            
            # Calculate zero-activation penalty - vectorized across all modalities and patients
            # Shape after comparison: [num_modalities, batch_size, embedding_dim]
            zero_mask = (all_embeddings.abs() <= self.eps)
            
            # Count zeros for each modality-patient pair and normalize by embedding size
            # Shape: [num_modalities, batch_size]  
            embedding_sizes = torch.tensor([all_embeddings.shape[2]] * all_embeddings.shape[0], 
                                          device=all_embeddings.device)[:, None]
            zero_ratios = zero_mask.sum(dim=2) / embedding_sizes
            
            # Calculate L2 norms - vectorized across all modalities and patients
            # Shape: [num_modalities, batch_size]
            norm_penalties = torch.norm(all_embeddings, p=2, dim=2)
    
            zero_ratios_per_mod = zero_ratios.t().mean(dim=0)
            # print(zero_ratios_per_mod)
            # Combine penalties (sum of zero ratio and norm penalty)
            # Shape: [num_modalities, batch_size]
            combined_penalties = zero_ratios + zero_ratios_per_mod.sum() - norm_penalties
            # torch.clamp(combined_penalties, max=100)
            reg_loss = combined_penalties.sum()
    
            return self.smoothing_func(nce_loss - self.alpha * reg_loss)
        return self.smoothing_func(nce_loss)
            