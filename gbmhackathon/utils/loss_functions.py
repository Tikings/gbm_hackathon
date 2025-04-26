import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import warnings

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
                 use_all_positives: bool = True,
                 eps: float = 1e-8):
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
        self.eps = eps  # For numerical stability

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
#         out_dict, patient_ids, available_modalities = batch
#         device = next(iter(out_dict.values())).device

#         # Build bank
#         bank, bank_ids, bank_mods = [], [], []
#         for m_idx, m in enumerate(self.modality_keys):
#             emb = out_dict[m]
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
class SmoothingFunction(nn.Module):
    def __init__(self, bound: float = -10, 
                 slope: float = 0.05, 
                 rate: float = -2):
        super().__init__()
        self.bound = bound
        self.slope = slope
        self.rate = rate

    def forward(self, x):
        return self.bound - (self.bound / (1 + torch.exp(x / self.rate))) + self.slope * x

def boundary_loss(outputs, min_val=-10, max_val=10):
    # Penalize values below min_val
    below_min = torch.relu(min_val - outputs)
    # Penalize values above max_val
    above_max = torch.relu(outputs - max_val)
    return torch.mean(below_min + above_max)
    
class RegularizedInfoNCELoss(nn.Module):
    def __init__(self,
                 modalities: List[str], 
                 patient_map: Dict[str, int] = None, 
                 temperature: float = 0.07, 
                 similarity: str = "original", 
                 use_all_positives: bool = True,
                 alpha: float = 0.1,
                 beta: float = 0.2,
                 nce_eps: float = 1e-8,
                 reg_eps: float = 1e-8,
                 bound: float = -10, 
                 slope: float = 0.05, 
                 rate: float = -2):
        super().__init__()
        self.infonce = InfoNCELoss(modalities=modalities, 
                                   patient_map=patient_map, 
                                   temperature=temperature, 
                                   similarity=similarity, 
                                   use_all_positives=use_all_positives)
        self.bound = bound
        self.smoothing_func = SmoothingFunction(bound=self.bound, slope=slope, rate=rate)
        
        self.alpha = alpha
        self.beta = beta
        self.reg_eps = reg_eps
    def forward(self, batch) -> torch.Tensor:
        nce_loss = self.infonce(batch)
        
        out_dict, _, _ = batch
        N = list(out_dict.values())[0].size(0)  # Number of patients
        
        # Stack all embeddings for vectorized computation
        # Shape: [num_modalities, batch_size, embedding_dim]
        all_embeddings = torch.stack([out_dict[mod].squeeze() for mod in out_dict.keys()])
        
        # Calculate zero-activation penalty - vectorized across all modalities and patients
        # Shape after comparison: [num_modalities, batch_size, embedding_dim]
        zero_mask = (all_embeddings.abs() <= self.reg_eps)
        
        # Count zeros for each modality-patient pair and normalize by embedding size
        # Shape: [num_modalities, batch_size]  
        embedding_sizes = torch.tensor([all_embeddings.shape[2]] * all_embeddings.shape[0], 
                                      device=all_embeddings.device)[:, None]
        zero_ratios = zero_mask.sum(dim=2) / embedding_sizes
        
        # Calculate L2 norms - vectorized across all modalities and patients
        # Shape: [num_modalities, batch_size]
        norm_penalties = torch.norm(all_embeddings, p=2, dim=2)

        zero_ratios_per_mod = zero_ratios.t().mean(dim=0)
        print(zero_ratios_per_mod)
        # Combine penalties (sum of zero ratio and norm penalty)
        # Shape: [num_modalities, batch_size]
        combined_penalties = zero_ratios + zero_ratios_per_mod.sum() + norm_penalties
        # combined_penalties = combined_penalties + self.beta * boundary_loss(all_embeddings, min_val=self.bound, max_val=-self.bound)

        
        # --- Original regularization loss ---
        # Sum across all modalities and patients, then average by number of patients
        reg_loss = combined_penalties.sum() / N
        

        # # Compute variance for each patient across modalities
        # # Shape: [batch_size]
        # patient_variance = torch.var(patient_zero_ratios, dim=1)
        
        # # Average variance across the batch
        # mean_variance = patient_variance.mean()
        
        # # Compute attention scores across modalities
        # attention_scores = torch.softmax(all_embeddings.mean(dim=2), dim=0)  # [num_modalities, batch_size]
        # ideal_score = 1.0 / len(out_dict.keys())
        # attention_imbalance = torch.sum(torch.abs(attention_scores - ideal_score))

        # # Compute correlation matrix between modality embeddings
        # flattened_embeddings = all_embeddings.view(len(out_dict.keys()), N, -1)
        # normalized_embeddings = F.normalize(flattened_embeddings, p=2, dim=2)
        # correlation_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.transpose(1, 2))
        # correlation_penalty = torch.mean(torch.abs(correlation_matrix - torch.eye(N, device=correlation_matrix.device)))

        # # Calculate average activation per modality
        # modality_activations = 1 - zero_ratios.mean(dim=1)  # [num_modalities]
        
        # # Apply stronger regularization to more active modalities
        # modality_weights = F.softmax(modality_activations, dim=0)
        # weighted_regularization = torch.sum(modality_weights * combined_penalties.sum(dim=1))

        # # Compute cross-correlation matrix between modality embeddings
        # z1 = all_embeddings[0]  # [batch_size, embedding_dim]
        # z2 = all_embeddings[1]  # [batch_size, embedding_dim]
        # c = torch.matmul(z1.T, z2) / N  # [embedding_dim, embedding_dim]
        
        # # Target: identity matrix to reduce redundancy
        # on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        # off_diag = torch.sum(c**2) - torch.sum(torch.diagonal(c)**2)
        # barlow_loss = on_diag + 0.005 * off_diag

        # # Calculate KL divergence from uniform distribution
        # uniform_target = torch.ones_like(zero_ratios) / len(out_dict.keys())
        # activation_distribution = zero_ratios / zero_ratios.sum(dim=0, keepdim=True)
        # uniform_loss = F.kl_div(activation_distribution.log(), uniform_target, reduction='batchmean')

        # modality_collapse_loss = 0

        # add_mean_variance = 0
        # if add_mean_variance == 1:
        #     modality_collapse_loss = modality_collapse_loss + mean_variance
        #     print("MEAN VAR:", mean_variance)

        # add_correlation_penalty = 0
        # if add_correlation_penalty == 1:
        #     modality_collapse_loss = modality_collapse_loss + correlation_penalty
        #     print("CORR PEN:", correlation_penalty)
            
        # add_attention_imbalance = 0
        # if add_attention_imbalance == 1:
        #     modality_collapse_loss = modality_collapse_loss + attention_imbalance
        #     print("ATTENTION IMB:", attention_imbalance)

        # add_weighted_regularization = 1
        # if add_weighted_regularization == 1:
        #     modality_collapse_loss = modality_collapse_loss + weighted_regularization
        #     print("ADAPTIVE REG:", weighted_regularization)

        # add_barlow_loss = 0
        # if add_barlow_loss == 1:
        #     modality_collapse_loss = modality_collapse_loss - barlow_loss
        #     print("BARLOW:", barlow_loss)

        # add_uniform_loss = 0
        # if add_uniform_loss == 1:
        #     modality_collapse_loss = modality_collapse_loss + uniform_loss
        #     print("UNIFORM:", uniform_loss)
    
        return self.smoothing_func(nce_loss - self.alpha * reg_loss) #+ self.beta * modality_collapse_loss)
        
    # def forward(self, batch) -> torch.Tensor:
    #     nce_loss = self.infonce(batch)
        
    #     out_dict, _, _ = batch
    #     N = list(out_dict.values())[0].size(0)  # Number of patients
        
    #     # Stack all embeddings for vectorized computation
    #     # Shape: [num_modalities, batch_size, embedding_dim]
    #     all_embeddings = torch.stack([out_dict[mod].squeeze() for mod in out_dict.keys()])
    #     # print(all_embeddings.size())

    #     max_value = torch.max(all_embeddings)
        
    #     # Calculate zero-activation penalty - vectorized across all modalities and patients
    #     # Shape after comparison: [num_modalities, batch_size, embedding_dim]
    #     zero_mask = (all_embeddings.abs() <= self.reg_eps)
    #     # print(zero_mask.size())
        
    #     # Count zeros for each modality-patient pair and normalize by embedding size
    #     # Shape: [num_modalities, batch_size]  
    #     embedding_sizes = torch.tensor([all_embeddings.shape[2]] * all_embeddings.shape[0], 
    #                                   device=all_embeddings.device)[:, None]
    #     # print(embedding_sizes.size())
    #     zero_penalties = zero_mask.sum(dim=2) / embedding_sizes
    #     # print(zero_penalties.size())
        
    #     # Calculate L2 norms - vectorized across all modalities and patients
    #     # Shape: [num_modalities, batch_size]
    #     norm_penalties = torch.norm(all_embeddings, p=2, dim=2)
        
    #     # Combine penalties (sum of zero penalty and norm penalty)
    #     # Shape: [num_modalities, batch_size]
    #     combined_penalties = zero_penalties + norm_penalties
        
    #     # Sum across all modalities and patients, then average by number of patients
    #     # Shape: scalar
    #     reg_loss = combined_penalties.sum() / N
    #     return self.smoothing_func(nce_loss - self.alpha * reg_loss)