import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch_geometric.data import DataLoader
from config import *
from session_dataset import SessionDataset
from utils import *
import numpy as np
from matplotlib.colors import ListedColormap
from config import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman'
})




def plot_dataset_distribution(dataset, dataset_name, dataset_labels, file_name, method='PCA', scale=1.0):
    node_features = []
    labels = []
    labels_int = []

    for data in dataset:
        node_features.append(data.x.view(-1).cpu().numpy())
        label = (data.y[0] == 1.0).nonzero(as_tuple=False).item()
        labels.append(dataset_labels[label])
        labels_int.append(label)

    node_features = np.array(node_features)

    # Change to 3 components
    if method == 'PCA':
        pca = PCA(n_components=3)
        reduced_features = pca.fit_transform(node_features)
    elif method == 't-SNE':
        n_samples = len(node_features)
        perplexity = min(30, n_samples // 4)
        tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
        reduced_features = tsne.fit_transform(node_features)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Custom colors for better visibility in 3D
    custom_colors = [
        '#e6194B',  # Red
        '#3cb44b',  # Green
        '#4363d8',  # Blue
        '#f58231',  # Orange
        '#911eb4',  # Purple
        '#42d4f4',  # Cyan
        '#f032e6',  # Magenta
        '#bfef45',  # Lime
        '#fabed4',  # Pink
        '#469990',  # Teal
        '#dcbeff',  # Lavender
        '#9A6324',  # Brown
        '#fffac8',  # Beige
        '#800000',  # Maroon
        '#aaffc3',  # Mint
    ]
    
    scatter = ax.scatter(reduced_features[:, 0]*scale, 
                        reduced_features[:, 1]*scale, 
                        reduced_features[:, 2]*scale,
                        c=labels_int, 
                        cmap=ListedColormap(custom_colors), 
                        s=70,
                        alpha=0.6)  # Added some transparency

    ax.set_xlabel(f'{method} Component 1', labelpad=10)
    ax.set_ylabel(f'{method} Component 2', labelpad=10)
    ax.set_zlabel(f'{method} Component 3', labelpad=10)
    plt.title(f'{dataset_name} Distribution ({method})', pad=20)
    
    # Add legend with class labels
    legend1 = ax.legend(scatter.legend_elements()[0], 
                       dataset_labels,
                       title="Classes",
                       loc="center left",
                       bbox_to_anchor=(1.15, 0.5))
    ax.add_artist(legend1)
    
    # Adjust the view angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # plt.savefig(file_name, dpi=600, bbox_inches='tight')
    plt.show()



iscx_vpn_dataset = SessionDataset(root=ISCX_VPN_DATASET_DIR, class_labels=iscx_vpn_get_unique_labels())
vnat_dataset = SessionDataset(root=VNAT_DATASET_DIR, class_labels=vnat_get_unique_labels())

# plot_dataset_distribution(iscx_vpn_dataset, dataset_name='ISCX-VPN', dataset_labels=iscx_vpn_get_unique_labels(), file_name='visualization/fig_iscx_distribution_pca.png', method='PCA')
plot_dataset_distribution(iscx_vpn_dataset, dataset_name='ISCX-VPN', dataset_labels=iscx_vpn_get_unique_labels(), file_name='visualization/fig_iscx_distribution_tsne.png', method='t-SNE', scale=3.0)  

# plot_dataset_distribution(vnat_dataset, dataset_name='VNAT', dataset_labels=vnat_get_unique_labels(), file_name='visualization/fig_vnat_distribution_pca.png', method='PCA')  
# plot_dataset_distribution(vnat_dataset, dataset_name='VNAT', dataset_labels=vnat_get_unique_labels(), file_name='visualization/fig_vnat_distribution_tsne.png', method='t-SNE')  