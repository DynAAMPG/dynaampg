import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import DataLoader
from session_dataset import SessionDataset
from dynaampg import DynAAMPG
from config import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt



def plot_logit_distribution(logits, labels, label_names, dataset_name = 'ISCX-VPN', method='PCA', file_name='logit_distribution.png'):
    # Plot the 2D reduced features
    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(logits[:, 0], logits[:, 1], c=labels, cmap="tab20", s=50)
    plt.legend(scatter.legend_elements()[0], label_names)
    plt.title(f'{dataset_name} Distribution ({method})')
    plt.xlabel('Principal Component 1' if method == 'PCA' else 't-SNE Component 1')
    plt.ylabel('Principal Component 2' if method == 'PCA' else 't-SNE Component 2')
    plt.savefig(file_name, dpi=600)
    plt.show()




if __name__ == "__main__":

    plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman'})

    iscx_vpn_labels = iscx_vpn_get_unique_labels()
    labels = np.load(os.path.join(SAVED_LOGITS_DIR, "labels.npy"))   
    actual_logits_pca = np.load(os.path.join(SAVED_LOGITS_DIR, "actual_logits_pca.npy")) 
    actual_logits_tsne = np.load(os.path.join(SAVED_LOGITS_DIR, "actual_logits_tsne.npy"))
    modified_logits_pca = np.load(os.path.join(SAVED_LOGITS_DIR, "modified_logits_pca.npy"))
    modified_logits_tsne = np.load(os.path.join(SAVED_LOGITS_DIR, "modified_logits_tsne.npy"))

    plot_logit_distribution(actual_logits_pca, labels, label_names=iscx_vpn_labels, dataset_name='Actual logits - ISCX-VPN', method='PCA', file_name='visualization/actual_logits_pca.png')
    plot_logit_distribution(actual_logits_tsne, labels, label_names=iscx_vpn_labels, dataset_name='Actual logits - ISCX-VPN', method='t-SNE', file_name='visualization/actual_logits_tsne.png')
    plot_logit_distribution(modified_logits_pca, labels, label_names=iscx_vpn_labels, dataset_name='Modified logits - ISCX-VPN', method='PCA', file_name='visualization/modified_logits_pca.png')
    plot_logit_distribution(modified_logits_tsne, labels, label_names=iscx_vpn_labels, dataset_name='Modified logits - ISCX-VPN', method='t-SNE', file_name='visualization/modified_logits_tsne.png')
    
