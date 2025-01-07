from config import *
from utils import *
import numpy as np



def save_pca_tsne(dataset, dataset_labels, dataset_name):
    node_features = []
    labels = []
    labels_int = []

    for data in dataset:
        node_features.append(data.x.view(-1).cpu().numpy())
        label = (data.y[0] == 1.0).nonzero(as_tuple=False).item()
        labels.append(dataset_labels[label])
        labels_int.append(label)

    node_features = np.array(node_features)

    pca = PCA(n_components=2)
    reduced_features_pca = pca.fit_transform(node_features)
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features_tsne = tsne.fit_transform(node_features)
    umap = UMAP(n_components=2, random_state=42)
    reduced_features_umap = umap.fit_transform(node_features)

    np.save(os.path.join(SAVED_PCA_TSNE_DIR, f"{dataset_name.lower()}_pca.npy"), reduced_features_pca)
    np.save(os.path.join(SAVED_PCA_TSNE_DIR, f"{dataset_name.lower()}_t-sne.npy"), reduced_features_tsne)
    np.save(os.path.join(SAVED_PCA_TSNE_DIR, f"{dataset_name.lower()}_umap.npy"), reduced_features_umap)
    np.save(os.path.join(SAVED_PCA_TSNE_DIR, f"{dataset_name.lower()}_labels.npy"), labels_int)


if __name__ == "__main__":

    iscx_vpn_dataset = SessionDataset(root=ISCX_VPN_DATASET_DIR, class_labels=iscx_vpn_get_unique_labels())
    vnat_dataset = SessionDataset(root=VNAT_DATASET_DIR, class_labels=vnat_get_unique_labels())
    iscx_tor_dataset = SessionDataset(root=ISCX_TOR_DATASET_DIR, class_labels=iscx_tor_get_unique_labels()) 

    save_pca_tsne(iscx_vpn_dataset, iscx_vpn_get_unique_labels(), 'ISCX-VPN')
    save_pca_tsne(vnat_dataset, vnat_get_unique_labels(), 'VNAT')
    save_pca_tsne(iscx_tor_dataset, iscx_tor_get_unique_labels(), 'ISCX-Tor')

