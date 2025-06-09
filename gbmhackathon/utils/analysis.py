import numpy as np
import pickle as pkl
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from typing import Dict, List, Tuple, Union
from matplotlib.colors import ListedColormap
import scipy.spatial.distance as distance
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.manifold import trustworthiness
from scipy.spatial import procrustes

from gbmhackathon.utils.loss_functions import RankMe
from gbmhackathon.viz.viz_experiment import visualize_embeddings
from gbmhackathon.models.mme import concat_modality_embeddings
from gbmhackathon.utils.model_saving import CPU_Unpickler

def prepare_embeddings(emb_dict, batch_all):
    patient_list, modality_list, avail_mods = batch_all[0], batch_all[1], batch_all[3]
    return get_filtered_info(emb_dict, avail_mods, patient_list, modality_list)

def get_modality_keys(row, avail_mods, modality_list):
    avail_mod_row = avail_mods[row]
    return [mod for i, mod in enumerate(modality_list) if avail_mod_row[i] == 1]

def get_row_id(i, patient_list):
    return patient_list[i]
    
def get_filter(tensor, modality, avail_mods, modality_list):
    keep_row = []
    for i in range(tensor.size(0)):
        if modality in get_modality_keys(i, avail_mods, modality_list):
            keep_row.append(i)
    return keep_row

def filter_embeddings(mme_embeddings, modality, avail_mods, modality_list):
    return mme_embeddings[get_filter(mme_embeddings, modality, avail_mods, modality_list),:]

def get_filtered_info(mme_emb_dict, avail_mods, patient_list, modality_list):
    filtered_dict = {}
    filtered_ids_dict = {}
    for key in mme_emb_dict.keys():
        filtered_dict[key] = filter_embeddings(mme_emb_dict[key], key, avail_mods, modality_list)
        row_filter = get_filter(mme_emb_dict[key], key, avail_mods, modality_list)
        filtered_ids_dict[key] = [get_row_id(i, patient_list) for i in row_filter]
    return filtered_dict, filtered_ids_dict

os.environ["SCIPY_ARRAY_API"]="1"
def linear_probing_clf(X, y, model=LogisticRegression, cv=5):
    """
    Performs linear probing using a logistic regression model (or compatible classifier)
    to evaluate the predictive power of features with respect to a binary target.

    Parameters:
    ----------
    X : array-like or tensor
        Feature matrix. If `X` is a PyTorch tensor, it will be converted to a NumPy array.
    y : array-like or pandas Series
        Binary target labels. If `y` is a pandas Series, it will be converted to a NumPy array.
    model : sklearn-like classifier, optional
        A classifier class with scikit-learn interface. Defaults to `LogisticRegression`.
    cv : int, optional
        Number of cross-validation folds. Defaults to 5.

    Returns:
    -------
    acc : float
        Accuracy score on the held-out test set.
    roc : float
        ROC AUC score on the held-out test set.
    cr : str
        Text summary of the classification report.
    scores : ndarray
        Array of cross-validated ROC AUC scores.

    Notes:
    -----
    - Uses `train_test_split` to divide the data into 80% training and 20% test set.
    - The classifier is trained with class balancing (`class_weight='balanced'`) and max_iter=1000.
    - The model is evaluated using Accuracy, ROC AUC, and a detailed classification report.
    - Cross-validation scores are based on ROC AUC.
    """
    if hasattr(X, 'numpy'):
        X = X.detach().numpy()
    if hasattr(y, 'values'):
        y = y.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = model(max_iter=1000, class_weight='balanced', C=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    cr = classification_report(y_test, y_pred)
    print("Accuracy:", acc)
    print("ROC AUC:", roc)
    print("Classification Report:\n", cr)

    scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    print("Mean ROC AUC (CV):", scores.mean())

    return acc, roc, cr, scores

def linear_probing_reg(X, y, model=LinearRegression, cv=5):
    """
    Performs linear probing using a regression model (default is LinearRegression)
    to evaluate how well the features predict a continuous target variable.

    Parameters:
    ----------
    X : array-like or tensor
        Feature matrix. If `X` is a PyTorch tensor, it will be converted to a NumPy array.
    y : array-like or pandas Series
        Continuous target values. If `y` is a pandas Series, it will be converted to a NumPy array.
    model : sklearn-like regressor, optional
        A regression model class with scikit-learn interface. Defaults to `LinearRegression`.
    cv : int, optional
        Number of cross-validation folds. Defaults to 5.

    Returns:
    -------
    mse : float
        Mean Squared Error on the test set.
    mae : float
        Mean Absolute Error on the test set.
    rmse : float
        RMSE score on the test set.
    scores : ndarray
        Array of cross-validated R² scores.

    Notes:
    -----
    - This function uses `train_test_split` to divide the dataset into 80% training and 20% testing.
    - It fits the regression model on the training data and evaluates performance on the test data.
    - Performance is assessed using MSE, MAE, and R² score.
    - It also performs k-fold cross-validation (default 5 folds) using the R² score.
    - If inputs are PyTorch tensors or pandas Series, they are automatically converted to NumPy arrays.
    """
    if hasattr(X, 'numpy'):
        X = X.detach().numpy()
    if hasattr(y, 'values'):
        y = y.values

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X = x_scaler.fit_transform(X.reshape(-1, X.shape[-1]) if X.ndim == 1 else X)
    y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg = model()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("RMSE Score:", rmse)
    print(f"Y_min: {y.min()}, Y_max: {y.max()}")

    scores = cross_val_score(reg, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    print("Mean RMSE (CV):", -scores.mean())

    return mse, mae, rmse, -scores

def find_key(dictionary, value_to_find):
    keys = [k for k, v in dictionary.items() if v == value_to_find]
    return keys[0]

def evaluate_embedding_quality(X_raw, Y_raw, n_neighbors=5):
    """
    Evaluates the quality of the embedding Y with respect to the original binary input X.

    Parameters:
        X_raw (np.ndarray): Original binary input matrix (samples x features)
        Y_raw (np.ndarray): Continuous embedding (samples x embedding_dim)
        n_neighbors (int): Number of neighbors for trustworthiness computation

    Returns:
        dict containing metrics: MSE, R², trustworthiness, Procrustes disparity
    """
    # Scaling
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_raw)
    Y_scaled = scaler_y.fit_transform(Y_raw)

    metrics = {}

    ## 1. Regression Y → X
    reg = Lasso() #Ridge()
    reg.fit(Y_scaled, X_scaled)
    X_pred = reg.predict(Y_scaled)

    metrics["mse"] = [mean_squared_error(X_scaled, X_pred)]
    metrics["rmse"] = [root_mean_squared_error(X_scaled, X_pred)]

    ## 2. Trustworthiness (local neighborhood preservation)
    try:
        trust = trustworthiness(X_scaled, Y_scaled, n_neighbors=n_neighbors)
        metrics["trustworthiness"] = [trust]
    except Exception as e:
        metrics["trustworthiness"] = [f"Error: {str(e)}"]

    ## 3. Procrustes (global geometric alignment)
    min_dim = min(X_scaled.shape[1], Y_scaled.shape[1])
    _, _, disparity = procrustes(X_scaled[:, :min_dim], Y_scaled[:, :min_dim])
    metrics["procrustes_disparity"] = [disparity]

    return metrics
    
def analyze_embeddings(wes_data: pd.DataFrame, 
                       bulk_data: pd.DataFrame, 
                       batch_all: tuple,
                       prefix: str,
                       path_folder:str,
                       emb_path: Union[str, None] = None, 
                       batch: Union[torch.Tensor, None] = None,
                       show: bool = True,
                      ):
    assert emb_path is not None or batch is not None, "You mus either provide a path or the compete batch tensor"
    os.makedirs(path_folder, exist_ok=True) # To ensure that we can save 
    
    # Loading section
    if emb_path is not None:
        with open(emb_path, 'rb') as f:
            embeddings = CPU_Unpickler(f).load()
    elif batch is not None:
        modality_list = batch_all[1]
        embeddings = {}
        emb_size = int(batch.size(1)/len(modality_list))
        for i, key in enumerate(modality_list):
            embeddings[key] = batch[:,i*emb_size:(i+1)*emb_size]
    rdy_embs, rdy_pids = prepare_embeddings(embeddings, batch_all)
    full_embeddings = concat_modality_embeddings(embeddings)
    full_patient_list = batch_all[0]
    
    # Visualize the whole dataset
    visualize_embeddings(concat_modality_embeddings(embeddings), 
                         method='pca', 
                         cluster_method='optics',
                         return_fig=True,
                        )
    suffix = f'{prefix}_dataset_pca_optics.pdf' if path_folder.endswith('/') else f'/{prefix}_dataset_pca_optics.pdf'
    plt.savefig(path_folder + suffix, bbox_inches='tight', format='pdf')
    if show:
        plt.show()
    visualize_embeddings(concat_modality_embeddings(embeddings), 
                         method='umap', 
                         cluster_method='optics',
                         return_fig=True
                        )
    suffix = f'{prefix}_dataset_umap_optics.pdf' if path_folder.endswith('/') else f'/{prefix}_dataset_umap_optics.pdf'
    plt.savefig(path_folder + suffix, bbox_inches='tight', format='pdf')
    if show:
        plt.show()

    # List of genes to evaluate
    genes = ["TP53", # Common tumour marker. This gene ie for the protein that "guards" cell DNA against degradation.
             "EGFR", # Often overexpressed when amplified
             "PTEN", # Tumour supressor, regulates important pathways. Lost or mutated in GBM cells
             "NF1", # Tumour supressor, RAS signaling inhibition, when lost, leads to uncontrolled cell growth in GBM
             "PIK3CA", # Catalityc subunit of PI3K, promotes cell survival in GBM tumours when activated
             "PIK3R1"] # Regulatory subunit of PI3K, same as PI3KCA
    
    results = []
    wes_idx = [i for i in range(full_embeddings.size(0)) if full_patient_list[i] in rdy_pids["wes"]]
    X_wes = full_embeddings[wes_idx,:]
    # Run linear probing for each gene and collect metrics
    for gene in genes:
        y_wes = wes_data.loc[rdy_pids['wes'],:][gene].astype(int)
    
        acc, roc, cr, scores = linear_probing_clf(X_wes, y_wes)
    
        results.append({
            "Gene": gene,
            "Accuracy": round(acc, 4),
            "ROC AUC": round(roc, 4),
            "CV Mean ROC AUC": round(scores.mean(), 4),
            "CV Std ROC AUC": round(scores.std(), 4)
        })
    
    results_df_wes = pd.DataFrame(results)
    plt.figure(figsize=(10, 4))
    sns.set(style="whitegrid")
    
    sns.heatmap(
        results_df_wes.set_index("Gene"),
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        cbar=False,
        linewidths=0.6,
        linecolor="lightgray",
        annot_kws={"fontsize": 12}
    )
    
    plt.title(f"Linear Probing Metrics per Gene", fontsize=14, weight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    suffix = f'{prefix}_linear_probing_wes.pdf' if path_folder.endswith('/') else f'/{prefix}_linear_probing_wes.pdf'
    plt.savefig(path_folder + suffix, bbox_inches='tight', format='pdf')
    plt.show()

    genes_ensembl_ids = {
    "EGFR": "ENSG00000146648", # Often overexpressed when amplified
    "PDGFRA": "ENSG00000134853", # Not in original list, but important in proneural subtype
    "NF1": "ENSG00000196712", # Lower expression linked to mesenchymal subtype
    "CDKN2A": "ENSG00000147889", # Often deleted; absence confirmed by low/no expression
    "IDH1": "ENSG00000138413", # Wild-type shows baseline; mutant IDH1 may be expressed in low-grade gliomas
    "MGMT": "ENSG00000170430", # Expression level predicts response to alkylating agents
    "TERT": "ENSG00000164362", # Promoter mutation leads to upregulation, measurable by RNA-seq
    "MDM2": "ENSG00000135679", # Overexpression can suppress p53
    "CHI3L1": "ENSG00000133048", # Not on original list, but highly upregulated in mesenchymal GBM
    "VEGFA": "ENSG00000112715", # Angiogenesis-related, overexpressed in GBM
    "SOX2": "ENSG00000181449", # Stemness marker; overexpressed in proneural subtype
    "OLIG2": "ENSG00000205927", # Highly expressed in proneural tumors
    "CD44": "ENSG00000026508", # Mesenchymal subtype marker, elevated in RNA-seq
    }
    genes = list(genes_ensembl_ids.values())

    results = []
    bulk_idx = [i for i in range(full_embeddings.size(0)) if full_patient_list[i] in rdy_pids["bulk"]]
    X_bulk = full_embeddings[bulk_idx,:]
    pids_bulk = [pid + '_mRNA' for pid in rdy_pids['bulk']]
    for gene in genes:
        y_bulk = bulk_data.T.loc[pids_bulk,:][gene].astype(float)
        scaled_y = StandardScaler().fit_transform(y_bulk.values.reshape(-1,1))
        mse, mae, rmse, scores = linear_probing_reg(X_bulk, y_bulk, model=Lasso)
    
        results.append({
            "Gene": find_key(genes_ensembl_ids, gene),
            "MSE": round(mse, 4),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "Y_min": round(scaled_y.min(), 4),
            "Y_avg": round(scaled_y.mean(), 4),
            "Y_max": round(scaled_y.max(), 4),
            "CV Mean RMSE": round(scores.mean(), 4),
            "CV Std RMSE": round(scores.std(), 4)
        })
    
    results_df_bulk = pd.DataFrame(results)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    sns.set(style="whitegrid")
    
    # Ensure the correct numeric format and index setup
    sns.heatmap(
        results_df_bulk.set_index("Gene"),
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        cbar=True,
        linewidths=0.6,
        linecolor="lightgray",
        annot_kws={"fontsize": 11}
    )
    
    plt.title("Linear Regression Probing Metrics per Gene", fontsize=14, weight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    suffix = f'{prefix}_linear_probing_bulk.pdf' if path_folder.endswith('/') else f'/{prefix}_linear_probing_bulk.pdf'
    plt.savefig(path_folder + suffix, bbox_inches='tight', format='pdf')
    plt.show()

    y_bulk = bulk_data.T.loc[pids_bulk,:].astype(int)
    dico = evaluate_embedding_quality(X_raw = X_bulk.cpu().detach().numpy(), Y_raw = y_bulk)

    results = {'lp_wes':results_df_wes, 'lp_bulk':results_df_bulk, 'quality_embeddings':pd.DataFrame(dico)}
    suffix = f'{prefix}_results.pkl' if path_folder.endswith('/') else f'/{prefix}_results.pkl'
    with open(path_folder + suffix, 'wb') as f:
        pkl.dump(results, f)
    return results
    
def compute_embedding_quality_metrics(
    embeddings: torch.Tensor,
    zero_threshold: float = 1e-2,
    hist: bool = True,
    hist_bins: int = 100,
    auto_bin_ratio: float = 0.95,
    custom_bins: bool = False
) -> dict:
    """
    Compute various quality metrics for a batch of embeddings.

    Args:
        embeddings (torch.Tensor): Tensor of shape [B, E], where B is batch size and E is embedding dimensionality.
        zero_threshold (float): Range around zero to consider as "close to zero" for collapse detection.
        hist_bins (int): Number of bins for the overall histogram of embedding values.

    Returns:
        A dictionary containing:
            - avg_distance (float): Average pairwise distance between embedding rows.
            - min_distance (float): Minimum non-zero pairwise distance.
            - rankme (torch.Tensor): Effective rank of the batch (scalar tensor).
            - histogram (torch.Tensor): Histogram counts of embedding values (length hist_bins).
            - zero_ratio_per_row (torch.Tensor): Tensor of shape [B], ratio of values per row within +/- zero_threshold.
    """
    assert embeddings.ndim == 2, "Embeddings must be a 2D tensor of shape [B, E]"
    B, E = embeddings.shape

    # Pairwise distances (Euclidean)
    # Compute full distance matrix
    with torch.no_grad():
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # [B, B]
        # Mask out zeros on diagonal
        mask = ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
        distances = dist_matrix[mask]
        avg_distance = distances.mean().item()
        min_distance = distances.min().item()

    # RankMe
    rankme_module = RankMe()
    rankme_value = rankme_module(embeddings).item()

    if hist:
        # Histogram of values with edges
        flat_vals = embeddings.view(-1).to("cpu")
        if custom_bins:
            # Compute percentiles
            q_start = float(torch.quantile(flat_vals, 0.0))
            q_low = float(torch.quantile(flat_vals, 1-auto_bin_ratio))
            q_mid = float(torch.quantile(flat_vals, auto_bin_ratio))
            q_high = float(torch.quantile(flat_vals, 1.0))
            # Number of bins in each segment
            n_body = int(hist_bins * auto_bin_ratio)
            n_tail = int(0.5 * (hist_bins - n_body))
            # Edges for each segment
            edges_start = torch.linspace(q_start, q_low, n_tail + 1)
            edges_low = torch.linspace(q_low, q_mid, n_body + 1)
            edges_high = torch.linspace(q_mid, q_high, n_tail + 1)
            # Combine, avoiding duplicate q_mid
            bin_edges = torch.cat([edges_start[:-1], edges_low[:-1], edges_high])
            hist_counts, hist_bins = torch.histogram(flat_vals, bins=bin_edges)
        else:
            hist_counts, hist_bins = torch.histogram(flat_vals)
        
    # Ratio of near-zero values per row
    zero_mask = embeddings.abs() <= zero_threshold
    zero_counts = zero_mask.sum(dim=1).to(torch.float32)
    zero_ratio_per_row = zero_counts / E
    zero_ratio_per_row = zero_ratio_per_row.mean().item()

    if hist:
        return {
            'avg_distance': avg_distance,
            'min_distance': min_distance,
            'rankme': rankme_value,
            'hist_counts': hist_counts.detach().numpy(),
            'hist_bins':hist_bins.detach().numpy(),
            'hist_values':flat_vals.detach().numpy(),
            'zero_ratio_per_row': zero_ratio_per_row
        }
    else:
        return {
            'avg_distance': avg_distance,
            'min_distance': min_distance,
            'rankme': rankme_value,
            'zero_ratio_per_row': zero_ratio_per_row
        }
    
class EmbeddingAnalyzer:
    """
    Classe pour analyser les embeddings issus d'un apprentissage par contrastive learning
    """
    def __init__(self, embeddings: Dict[str, torch.Tensor], patient_ids: List[str], 
                 available_modalities: torch.Tensor, modality_keys: List[str],
                 patient_map: Dict[str, int] = None, device='cpu'):
        """
        Initialise l'analyseur d'embeddings
        
        Args:
            embeddings: Dictionnaire avec les noms des modalités comme clés et les tenseurs d'embedding
            patient_ids: Liste des ID patients pour le batch
            available_modalities: Tenseur de forme [batch_size, num_modalities] indiquant les modalités disponibles
            modality_keys: Liste des noms des modalités
            patient_map: Dictionnaire pour traduire les patient_ids en numéros uniques
            device: Appareil sur lequel exécuter les calculs ('cpu' ou 'cuda')
        """
        self.device = device
        self.modality_keys = modality_keys
        
        # Créer un dictionnaire patient_map si non fourni
        if patient_map is None:
            unique_patients = sorted(list(set(patient_ids)))
            self.patient_map = {pid: i for i, pid in enumerate(unique_patients)}
        else:
            self.patient_map = patient_map
            
        with torch.no_grad():
            # Construire la banque d'embeddings
            self.bank = []  # Liste des embeddings
            self.bank_ids = []  # Liste des IDs patients correspondants
            self.bank_modality_indices = []  # Liste des indices de modalités correspondants
            self.bank_modality_names = []  # Liste des noms de modalités correspondants
            
            batch_size = len(patient_ids)
            
            # Rassembler tous les embeddings et leurs métadonnées
            for mod_idx, modality in enumerate(modality_keys):
                emb = embeddings[modality]  # Forme: [batch_size, embedding_dim]
                
                for patient_idx in range(batch_size):
                    # Vérifier si cette modalité est disponible pour ce patient
                    if available_modalities[patient_idx, mod_idx] == 1:
                        self.bank.append(emb[patient_idx].detach().cpu().numpy())
                        self.bank_ids.append(patient_ids[patient_idx])
                        self.bank_modality_indices.append(mod_idx)
                        self.bank_modality_names.append(modality)
            
            # Convertir en tableaux numpy
            self.bank = np.array(self.bank)
            self.patient_ids_array = np.array(self.bank_ids)
            self.modality_indices = np.array(self.bank_modality_indices)
            self.modality_names = np.array(self.bank_modality_names)
            
            # Calculer la matrice de distance
            self.distance_matrix = euclidean_distances(self.bank)
            
            # Créer des tableaux numériques des IDs patients
            self.patient_ids_num = np.array([self.patient_map[pid] for pid in self.bank_ids])
            
            # Générer des couleurs pour les patients
            self.unique_patients = np.unique(self.patient_ids_num)
            self.patient_colors = plt.cm.tab20(np.linspace(0, 1, len(self.unique_patients)))
        
    def compute_intra_inter_distances(self) -> Tuple[float, float, float]:
        """
        Calcule les distances moyennes intra-patient et inter-patients
        
        Returns:
            Tuple contenant (distance_intra_moyenne, distance_inter_moyenne, ratio)
        """
        intra_distances = []
        inter_distances = []
        
        # Pour chaque paire d'embeddings
        for i in range(len(self.bank)):
            for j in range(i+1, len(self.bank)):
                dist = self.distance_matrix[i, j]
                
                # Si les embeddings appartiennent au même patient
                if self.patient_ids_num[i] == self.patient_ids_num[j]:
                    intra_distances.append(dist)
                else:
                    inter_distances.append(dist)
        
        intra_mean = np.mean(intra_distances) if intra_distances else float('inf')
        inter_mean = np.mean(inter_distances) if inter_distances else float('inf')
        ratio = intra_mean / inter_mean if inter_mean != 0 else float('inf')
        
        return intra_mean, inter_mean, ratio
        
    def compute_worst_case_ratio(self) -> float:
        """
        Calcule le ratio entre la plus grande distance intra-patient et la plus petite distance inter-patients
        
        Returns:
            float: Ratio max_intra / min_inter
        """
        max_intra = -float('inf')
        min_inter = float('inf')
        
        # Pour chaque paire d'embeddings
        for i in range(len(self.bank)):
            for j in range(i+1, len(self.bank)):
                dist = self.distance_matrix[i, j]
                
                # Si les embeddings appartiennent au même patient
                if self.patient_ids_num[i] == self.patient_ids_num[j]:
                    max_intra = max(max_intra, dist)
                else:
                    min_inter = min(min_inter, dist)
        
        # Si aucune distance intra ou inter n'a été trouvée
        if max_intra == -float('inf'):
            max_intra = 0
        if min_inter == float('inf'):
            min_inter = 0
            
        return max_intra / (min_inter + 1e-8)

    def compute_silhouette_score(self) -> float:
        """
        Calcule le score de silhouette en utilisant les IDs patients comme labels
        
        Returns:
            float: Score de silhouette
        """
        if len(np.unique(self.patient_ids_num)) <= 1:
            return 0.0  # Cas spécial: un seul cluster
            
        return silhouette_score(self.bank, self.patient_ids_num)
    
    def compute_knn_accuracy(self, k: int = 3) -> float:
        """
        Évalue la précision d'un classifieur kNN pour prédire à quel patient appartient chaque modalité
        
        Args:
            k: Nombre de voisins pour le classifieur kNN
            
        Returns:
            float: Précision du classifieur
        """
        if len(self.bank) <= 1:
            return 0.0
            
        # Utiliser leave-one-out pour évaluer la précision
        classifier = KNeighborsClassifier(n_neighbors=min(k, len(self.bank)-1))
        correct = 0
        
        for i in range(len(self.bank)):
            # Construire l'ensemble d'entraînement en excluant l'exemple courant
            train_data = np.delete(self.bank, i, axis=0)
            train_labels = np.delete(self.patient_ids_num, i)
            
            # Entraîner le classifieur
            classifier.fit(train_data, train_labels)
            
            # Prédire le label pour l'exemple courant
            predicted = classifier.predict([self.bank[i]])
            
            if predicted[0] == self.patient_ids_num[i]:
                correct += 1
                
        return correct / len(self.bank)
    
    def visualize_mds(self, figsize=(10, 8), title="Projection MDS des embeddings"):
        """
        Visualise les embeddings en utilisant MDS (Multidimensional Scaling)
        """
        # Appliquer MDS
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        pos = mds.fit_transform(self.distance_matrix)
        
        # Créer la figure
        plt.figure(figsize=figsize)
        
        # Créer un scatter plot coloré par patient
        for i, patient_id in enumerate(self.unique_patients):
            mask = self.patient_ids_num == patient_id
            plt.scatter(pos[mask, 0], pos[mask, 1], c=[self.patient_colors[i]], 
                      label=f"Patient {self.bank_ids[np.where(mask)[0][0]]}")
        
        plt.title(title)
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_distance_heatmap(self, figsize=(12, 10), annotate=False, 
                                title="Heatmap des distances entre embeddings"):
        """
        Visualise la matrice de distances sous forme de heatmap, regroupée par patient
        """
        # Créer un DataFrame pour faciliter la manipulation
        df = pd.DataFrame({
            'patient_id': self.bank_ids,
            'modality': self.modality_names,
            'patient_num': self.patient_ids_num
        })
        
        # Trier les indices par patient puis par modalité
        sorted_indices = df.sort_values(['patient_num', 'modality']).index.values
        
        # Réorganiser la matrice de distances
        sorted_distances = self.distance_matrix[sorted_indices][:, sorted_indices]
        
        # Créer des étiquettes pour la heatmap
        labels = [f"{df.iloc[i]['patient_id']}_{df.iloc[i]['modality']}" for i in sorted_indices]
        
        # Créer la heatmap
        plt.figure(figsize=figsize)
        ax = sns.heatmap(sorted_distances, xticklabels=labels, yticklabels=labels,
                    cmap="viridis", annot=annotate, fmt=".2f" if annotate else None)
        
        # Rotation des étiquettes pour une meilleure lisibilité
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        
        plt.title(title)
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_pos_neg_distribution(self, figsize=(10, 6), 
                                     title="Distribution des distances positives vs négatives"):
        """
        Visualise les distributions des distances positives (même patient) vs négatives (patients différents)
        """
        pos_distances = []
        neg_distances = []
        
        # Pour chaque paire d'embeddings
        for i in range(len(self.bank)):
            for j in range(i+1, len(self.bank)):
                dist = self.distance_matrix[i, j]
                
                # Si les embeddings appartiennent au même patient
                if self.patient_ids_num[i] == self.patient_ids_num[j]:
                    pos_distances.append(dist)
                else:
                    neg_distances.append(dist)
        
        # Créer la figure
        plt.figure(figsize=figsize)
        
        # Tracer les histogrammes
        plt.hist(pos_distances, alpha=0.5, label="Distances positives (même patient)", color="green",
               bins=20, density=True)
        plt.hist(neg_distances, alpha=0.5, label="Distances négatives (patients différents)", color="red",
               bins=20, density=True)
        
        plt.xlabel("Distance")
        plt.ylabel("Densité")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def quality_summary(self, viz=False):
        """
        Affiche un résumé des métriques calculées
        """
        # Affichages des graphiques
        if viz:
            self.visualize_mds()
            plt.show()
            self.visualize_distance_heatmap(annotate=True)
            plt.show()
            self.visualize_pos_neg_distribution()
            plt.show()
        
        # Calcul des métriques
        intra_mean, inter_mean, ratio = self.compute_intra_inter_distances()
        worst_ratio = self.compute_worst_case_ratio()
        silhouette = self.compute_silhouette_score()
        knn_acc = self.compute_knn_accuracy()
        
        # Affichage des résultats
        print("=============== MÉTRIQUES D'EMBEDDINGS ===============")
        print(f"Distance moyenne intra-patient: {intra_mean:.4f}")
        print(f"Distance moyenne inter-patients: {inter_mean:.4f}")
        print(f"Ratio distances moyennes (intra/inter): {ratio:.4f}")
        print(f"Ratio pire cas (max intra / min inter): {worst_ratio:.4f}")
        print(f"Score de silhouette: {silhouette:.4f}")
        print(f"Précision kNN: {knn_acc:.4f}")
        print("=====================================================")
        
        return {
            "intra_mean": intra_mean,
            "inter_mean": inter_mean,
            "ratio": ratio,
            "worst_ratio": worst_ratio,
            "silhouette": silhouette,
            "knn_acc": knn_acc
        }


# Fonction d'aide pour intégrer l'analyse dans la boucle d'entraînement
def analyze_embeddings_after_epoch(model, dataloader, modality_keys, patient_map, 
                                 device='cpu', num_batches=1, visualize=True, save_dir=None):
    """
    Analyse les embeddings du modèle sur un sous-ensemble du dataloader
    
    Args:
        model: Le modèle qui génère les embeddings
        dataloader: Le dataloader contenant les données
        modality_keys: Liste des noms des modalités
        patient_map: Dictionnaire pour traduire les patient_ids en numéros uniques
        device: Appareil sur lequel exécuter les calculs ('cpu' ou 'cuda')
        num_batches: Nombre de batchs à analyser
        visualize: Si True, génère et sauvegarde les visualisations
        save_dir: Répertoire où sauvegarder les visualisations
        
    Returns:
        dict: Dictionnaire des métriques calculées
    """
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if num_batches != -1: # Use all batches
                if batch_idx >= num_batches:
                    break
                
            # Obtenir le batch
            patient_ids, modalities, X_dict, avail_mods, batch_targets, targets_names = batch
            
            # Forward pass pour obtenir les embeddings
            contrastive_outputs, *_ = model(X_dict)
            
            # Créer l'analyseur
            analyzer = EmbeddingAnalyzer(
                embeddings=contrastive_outputs,
                patient_ids=patient_ids,
                available_modalities=avail_mods,
                modality_keys=modality_keys,
                patient_map=patient_map,
                device=device
            )
            
            # Calculer les métriques
            metrics = analyzer.quality_summary(viz=False)
            all_metrics.append(metrics)
            
            # Générer et sauvegarder les visualisations
            if visualize:
                if save_dir is None:
                    save_dir = "."
                    
                # MDS
                fig_mds = analyzer.visualize_mds()
                fig_mds.savefig(f"{save_dir}/mds_batch_{batch_idx}.png")
                
                # Heatmap
                fig_heatmap = analyzer.visualize_distance_heatmap()
                fig_heatmap.savefig(f"{save_dir}/heatmap_batch_{batch_idx}.png")
                
                # Distribution
                fig_dist = analyzer.visualize_pos_neg_distribution()
                fig_dist.savefig(f"{save_dir}/distribution_batch_{batch_idx}.png")
                
                plt.close('all')
    
    # Agréger les métriques sur tous les batchs
    if all_metrics:
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
        print("\n=============== MÉTRIQUES MOYENNES ===============")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.4f}")
        print("==================================================\n")
        return avg_metrics
    
    return {}


# Exemple d'intégration dans la boucle d'entraînement
def integrate_in_training_loop(EPOCHS_I, dataloader, gbmnet, optimizer_I_fn, base_lr_I, 
                             scheduler_I_cfg, contrastive_loss_cfg, RegularizedInfoNCELoss,
                             modality_keys, patient_map, pretrained_mme=False):
    """
    Intègre l'analyse des embeddings dans la boucle d'entraînement
    """
    # Initialize metric collectors
    pos_aligns_I, neg_aligns_I, distance_aligns_I = [], [], []
    embedding_metrics = []
    
    if pretrained_mme:
        EPOCHS_I_LOSSES = None
    else:
        # Optimization for Phase I
        optimizer_I = optimizer_I_fn(gbmnet.mme.parameters(), lr=base_lr_I)
        scheduler_I_cfg["optimizer"] = optimizer_I
        if "eta_min" in scheduler_I_cfg.keys():
            if scheduler_I_cfg["eta_min"] == "dynamic":
                scheduler_I_cfg["eta_min"] = eta_min_coef * base_lr_I
        scheduler_I = instantiate(scheduler_I_cfg, scheduler_I)
    
        assert contrastive_loss_cfg is not None, "You must provide a config for the contrastive loss"
        contrastive_loss_fn = instantiate(contrastive_loss_cfg, RegularizedInfoNCELoss)
        
        # Phase I training loop
        EPOCHS_I_LOSSES = []
        for epoch in range(EPOCHS_I):
            epoch_loss = []
            
            for idx, batch in enumerate(dataloader):
                # Get batch
                patient_ids, modalities, X_dict, avail_mods, batch_targets, targets_names = batch
                
                # Forward pass
                contrastive_outputs, *_ = gbmnet(X_dict)
                contrastive_loss_batch = (contrastive_outputs, patient_ids, avail_mods)
        
                # Loss computation
                contrastive_loss = contrastive_loss_fn(contrastive_loss_batch)
                loss = contrastive_loss
        
                # Backward pass
                loss.backward()
                optimizer_I.step()
                
        
                # Learning schedule
                before_lr_I = optimizer_I.param_groups[0]["lr"]
                scheduler_I.step()

                # Reset gradients
                optimizer_I.zero_grad()
        
                # Store losses
                epoch_loss.append(loss.item())
        
            print(f"\n\nLearning Rate: {before_lr_I}")
            # For monitoring
            avg_loss = np.mean(epoch_loss)
            pos_align = np.mean(contrastive_loss_fn.infonce.pos_alignments)
            neg_align = np.mean(contrastive_loss_fn.infonce.neg_alignments)
            distance = pos_align - neg_align
            
            print(f"Epoch {epoch} total loss: {np.mean(epoch_loss):.4f}".upper())
            
            pos_aligns_I.append(pos_align)
            neg_aligns_I.append(neg_align)
            distance_aligns_I.append(distance)
            contrastive_loss_fn.infonce.clear_alignments()
            EPOCHS_I_LOSSES.append(avg_loss)
            
            # Analyser les embeddings tous les N epochs
            if epoch % 5 == 0 or epoch == EPOCHS_I - 1:
                save_dir = f"embedding_analysis/epoch_{epoch}"
                os.makedirs(save_dir, exist_ok=True)
                
                metrics = analyze_embeddings_after_epoch(
                    model=gbmnet,
                    dataloader=dataloader,
                    modality_keys=modality_keys,
                    patient_map=patient_map,
                    device=gbmnet.device,
                    num_batches=2,  # Analyser 2 batchs
                    visualize=True,
                    save_dir=save_dir
                )
                
                embedding_metrics.append({
                    'epoch': epoch,
                    'metrics': metrics
                })
                
                # Visualiser l'évolution des métriques
                if len(embedding_metrics) > 1:
                    visualize_metrics_evolution(embedding_metrics, save_dir=save_dir)
    
    return EPOCHS_I_LOSSES, pos_aligns_I, neg_aligns_I, distance_aligns_I, embedding_metrics

# Visualiser l'évolution des métriques au cours de l'entraînement
def visualize_metrics_evolution(embedding_metrics, save_dir="."):
    """
    Visualise l'évolution des métriques d'embeddings au cours de l'entraînement
    """
    epochs = [entry['epoch'] for entry in embedding_metrics]
    metrics_names = list(embedding_metrics[0]['metrics'].keys())
    
    # Créer un subplot pour chaque métrique
    fig, axes = plt.subplots(len(metrics_names), 1, figsize=(10, 3*len(metrics_names)))
    
    for i, metric_name in enumerate(metrics_names):
        metric_values = [entry['metrics'][metric_name] for entry in embedding_metrics]
        
        ax = axes[i] if len(metrics_names) > 1 else axes
        ax.plot(epochs, metric_values, 'o-', linewidth=2)
        ax.set_title(f"Évolution de {metric_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/metrics_evolution.png")
    plt.close()