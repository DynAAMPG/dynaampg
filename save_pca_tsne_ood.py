import torch
from torch_geometric.loader import DataLoader
from session_dataset import SessionDataset
from dynaampg import DynAAMPG
from config import *
from utils import *
import numpy as np
import string


def save_pca_tsne(dataset_name, dataset_labels, method, num_ood_samples, dist_thres):
    reduced_features = np.load(os.path.join(SAVED_PCA_TSNE_DIR, f"{dataset_name.lower()}_{method.lower()}.npy"))
    labels = np.load(os.path.join(SAVED_PCA_TSNE_DIR, f"{dataset_name.lower()}_labels.npy"))

    # Generate OOD samples
    ood_samples = []   

    attempts = 0
    max_attempts = 1000  # Prevent infinite loop

    while len(ood_samples) < num_ood_samples and attempts < max_attempts:
        new_x = np.random.uniform(reduced_features[:, 0].min(), reduced_features[:, 0].max())
        new_y = np.random.uniform(reduced_features[:, 1].min(), reduced_features[:, 1].max())
        
        # Calculate distances to all blue points
        distances = np.sqrt((reduced_features[:, 0] - new_x) ** 2 + (reduced_features[:, 1] - new_y)**2)
        
        # If minimum distance is >= 2, accept the point
        if np.min(distances) >= dist_thres:
            ood_samples.append([new_x, new_y])
        
        attempts += 1
    node_features = np.array(node_features)

    pca = PCA(n_components=2)
    reduced_features_pca = pca.fit_transform(node_features)
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features_tsne = tsne.fit_transform(node_features)

    np.save(os.path.join(SAVED_PCA_TSNE_DIR, f"{dataset_name.lower()}_pca.npy"), reduced_features_pca)
    np.save(os.path.join(SAVED_PCA_TSNE_DIR, f"{dataset_name.lower()}_t-sne.npy"), reduced_features_tsne)
    np.save(os.path.join(SAVED_PCA_TSNE_DIR, f"{dataset_name.lower()}_labels.npy"), labels)


if __name__ == "__main__":

    # iscx_vpn_dataset = SessionDataset(root=ISCX_VPN_DATASET_DIR, class_labels=iscx_vpn_get_unique_labels())
    # vnat_dataset = SessionDataset(root=VNAT_DATASET_DIR, class_labels=vnat_get_unique_labels())
    iscx_tor_dataset = SessionDataset(root=ISCX_TOR_DATASET_DIR, class_labels=iscx_tor_get_unique_labels())

    # save_pca_tsne(iscx_vpn_dataset, iscx_vpn_get_unique_labels(), 'ISCX-VPN')
    # save_pca_tsne(vnat_dataset, vnat_get_unique_labels(), 'VNAT')
    save_pca_tsne(iscx_tor_dataset, iscx_tor_get_unique_labels(), 'ISCX-TOR')

