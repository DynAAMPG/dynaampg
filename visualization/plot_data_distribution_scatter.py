import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from config import *
from utils import *
import numpy as np
from config import *



# Before running this script you should run save_pca_tsne.py to save the PCA and t-SNE results in the SAVED_PCA_TSNE_DIR directory.



plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman'
})




def plot_dataset_distribution(dataset_name, dataset_labels, file_name, method='PCA'):

    reduced_features = np.load(os.path.join(SAVED_PCA_TSNE_DIR, f"{dataset_name.lower()}_{method.lower()}.npy"))
    labels = np.load(os.path.join(SAVED_PCA_TSNE_DIR, f"{dataset_name.lower()}_labels.npy"))

    # Plot the 2D reduced features
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap="tab20", s=50, alpha=0.5)
    plt.legend(scatter.legend_elements()[0], dataset_labels)
    plt.title(f'{dataset_name} Distribution ({method})')
    plt.xlabel('Principal Component 1' if method == 'PCA' else 't-SNE Component 1' if method == 't-SNE' else 'UMAP Component 1')
    plt.ylabel('Principal Component 2' if method == 'PCA' else 't-SNE Component 2' if method == 't-SNE' else 'UMAP Component 2')
    plt.savefig(file_name, dpi=600)
    plt.show()


if __name__ == "__main__":

    plot_dataset_distribution(dataset_name='ISCX-VPN', dataset_labels=iscx_vpn_get_unique_labels(), file_name='visualization/fig_iscx_distribution_pca.png', method='PCA')
    plot_dataset_distribution(dataset_name='ISCX-VPN', dataset_labels=iscx_vpn_get_unique_labels(), file_name='visualization/fig_iscx_distribution_tsne.png', method='t-SNE') 
    plot_dataset_distribution(dataset_name='ISCX-VPN', dataset_labels=iscx_vpn_get_unique_labels(), file_name='visualization/fig_iscx_distribution_umap.png', method='UMAP') 

    plot_dataset_distribution(dataset_name='VNAT', dataset_labels=vnat_get_unique_labels(), file_name='visualization/fig_vnat_distribution_pca.png', method='PCA')  
    plot_dataset_distribution(dataset_name='VNAT', dataset_labels=vnat_get_unique_labels(), file_name='visualization/fig_vnat_distribution_tsne.png', method='t-SNE') 
    plot_dataset_distribution(dataset_name='VNAT', dataset_labels=vnat_get_unique_labels(), file_name='visualization/fig_vnat_distribution_umap.png', method='UMAP') 

    plot_dataset_distribution(dataset_name='ISCX-Tor', dataset_labels=iscx_tor_get_unique_labels(), file_name='visualization/fig_iscx_tor_distribution_pca.png', method='PCA') 
    plot_dataset_distribution(dataset_name='ISCX-Tor', dataset_labels=iscx_tor_get_unique_labels(), file_name='visualization/fig_iscx_tor_distribution_tsne.png', method='t-SNE') 
    plot_dataset_distribution(dataset_name='ISCX-Tor', dataset_labels=iscx_tor_get_unique_labels(), file_name='visualization/fig_iscx_tor_distribution_umap.png', method='UMAP')    


    