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
import random
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns


plt.rcParams.update({
    "font.family": 'Times New Roman',
    "font.size": 18
})




def plot_dataset_distribution(axis, dataset_name, dataset_labels, method='PCA', num_ood_samples=5, dist_thres=0.8, color_dict=None):

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

    ood_samples = np.array(ood_samples)

    # Create custom colormap from color dictionary
    unique_labels = np.unique(labels)
    label_colors = [color_dict[dataset_labels[i]] for i in range(len(unique_labels))]
    custom_cmap = ListedColormap(label_colors)

    # Update scatter plot to use consistent colors
    scatter = axis.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                         c=labels, cmap=custom_cmap, s=50)
    axis.scatter(ood_samples[:, 0], ood_samples[:, 1], 
                c='darkred', marker='x', label='OOD Samples', s=50)
    

    # Create legend with correct label ordering
    legend_elements = []
    for i in unique_labels:
        legend_elements.append(plt.scatter([], [], c=color_dict[dataset_labels[int(i)]], s=50))
    legend_elements.append(plt.scatter([], [], c='darkred', marker='x', s=50))
    
    # Use the original dataset labels in the correct order
    axis.legend(legend_elements, [dataset_labels[int(i)] for i in unique_labels] + ['OOD sample'], labelspacing=0.2) 


    # plt.legend(scatter.legend_elements()[0], dataset_labels + ['OOD Samples'])
    axis.set_title(f'{dataset_name} Feature Distribution ({method}) with OOD Samples')
    axis.set_xlabel('Principal Component 1' if method == 'PCA' else 't-SNE Component 1')
    axis.set_ylabel('Principal Component 2' if method == 'PCA' else 't-SNE Component 2')





def plot():
    # Updated ISCX2016 distribution and labels
    iscx_vpn_distribution = [
        14621, 21610, 3752, 138549, 399893, 4996, 
        596, 8058, 1318, 2040, 7730, 954
    ]
    iscx_vpn_labels = [
        'email', 'chat', 'stream', 'ft', 'voip', 'p2p',
        'vpn_email', 'vpn_chat', 'vpn_stream', 'vpn_ft', 'vpn_voip', 'vpn_p2p'
    ]

    # Updated VNAT-PN distribution and labels
    vnat_distribution = [32826, 27182, 3518, 3052, 712, 18, 16, 10]
    vnat_labels = [
        'ft', 'p2p', 'stream', 'voip', 'vpn_voip', 
        'vpn_ft', 'vpn_p2p', 'vpn_stream'
    ]

    # Updated Tor distribution and labels   
    iscx_tor_distribution = [55660,700,1008,2016,3774,3104,2902,45838,68,12,42,46,294,12,28,40]    
    iscx_tor_labels = [
        'browse', 'email','chat','audio','video','ft','voip','p2p','tor_browse','tor_email',
        'tor_chat','tor_audio','tor_video','tor_ft','tor_voip','tor_p2p'
    ]

    # Create a consistent color mapping for all classes
    all_classes = list(set(iscx_vpn_labels + vnat_labels + iscx_tor_labels))
    color_palette = plt.cm.tab20(np.linspace(0, 1, len(all_classes)))
    color_dict = dict(zip(all_classes, color_palette))

    # Create DataFrames with color information
    iscx_vpn_df = pd.DataFrame({
        'Labels': iscx_vpn_labels, 
        'Count': iscx_vpn_distribution,
        'Colors': [color_dict[label] for label in iscx_vpn_labels]
    })
    vnat_df = pd.DataFrame({
        'Labels': vnat_labels, 
        'Count': vnat_distribution,
        'Colors': [color_dict[label] for label in vnat_labels]
    })
    iscx_tor_df = pd.DataFrame({
        'Labels': iscx_tor_labels, 
        'Count': iscx_tor_distribution,
        'Colors': [color_dict[label] for label in iscx_tor_labels]
    })

    # Create a figure with 2 rows, 3 columns
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(28, 16))

    

   
   
    # Function to add value labels on bars
    def add_value_labels(ax, fontsize=15):
        for container in ax.containers:
            ax.bar_label(container, fontsize=fontsize, padding=3, rotation=0)

    
    
    # First row plots (distribution plots)
    # Plot bar chart for ISCX2016 using seaborn
    sns.barplot(x='Labels', y='Count', hue='Labels', data=iscx_vpn_df, ax=ax1, 
                palette=[tuple(c) for c in iscx_vpn_df['Colors']], legend=False)    
    add_value_labels(ax1)
    ax1.tick_params(axis='x', rotation=90)
    ax1.set_ylim(0, max(iscx_vpn_distribution) * 1.15)
    ax1.set_title('ISCX-VPN Class Distribution', pad=10)
    ax1.set_xlabel('Traffic Classes')  # Set consistent x-label
    ax1.set_ylabel('Number of Samples')  # Set consistent y-label
    
    # Plot bar chart for VNAT-PN using seaborn
    sns.barplot(x='Labels', y='Count', hue='Labels', data=vnat_df, ax=ax2, 
                palette=[tuple(c) for c in vnat_df['Colors']], legend=False)
    
    add_value_labels(ax2)
    ax2.tick_params(axis='x', rotation=90)
    ax2.set_ylim(0, max(vnat_distribution) * 1.15)
    ax2.set_title('VNAT Class Distribution', pad=10)
    ax2.set_xlabel('Traffic Classes')  # Set consistent x-label
    ax2.set_ylabel('Number of Samples')  # Set consistent y-label
    
    # Plot bar chart for Tor using seaborn  
    sns.barplot(x='Labels', y='Count', hue='Labels', data=iscx_tor_df, ax=ax3, 
                palette=[tuple(c) for c in iscx_tor_df['Colors']], legend=False)
    
    add_value_labels(ax3)
    ax3.tick_params(axis='x', rotation=90)
    ax3.set_ylim(0, max(iscx_tor_distribution) * 1.15)
    ax3.set_title('ISCX-Tor Class Distribution', pad=10)
    ax3.set_xlabel('Traffic Classes')  # Set consistent x-label
    ax3.set_ylabel('Number of Samples')  # Set consistent y-label



    # Second row plots (PCA/t-SNE plots)
    plot_dataset_distribution(ax4, dataset_name='ISCX-VPN', dataset_labels=iscx_vpn_labels, 
                            method='t-SNE', num_ood_samples=100, dist_thres=2, color_dict=color_dict)
    plot_dataset_distribution(ax5, dataset_name='VNAT', dataset_labels=vnat_labels, 
                            method='t-SNE', num_ood_samples=100, dist_thres=5, color_dict=color_dict)
    plot_dataset_distribution(ax6, dataset_name='ISCX-TOR', dataset_labels=iscx_tor_labels, 
                            method='t-SNE', num_ood_samples=100, dist_thres=5, color_dict=color_dict)

    plt.tight_layout(pad=2.0)
    plt.savefig('visualization/fig_two_problems.png')
    plt.show()



if __name__ == "__main__":

    plot()
    