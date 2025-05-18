import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from typing import Dict, List, Tuple
from matplotlib.colors import ListedColormap
import scipy.spatial.distance as distance
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


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