import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils import *

# Set the default font to Times New Roman
plt.rcParams.update({
    "font.family": 'Times New Roman',
    "font.size": 22
})

colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#808000', 
        '#008000', '#800080', '#008080', '#000080', '#FFA500', '#A52A2A', '#8A2BE2', '#5F9EA0', 
        '#D2691E', '#FF7F50', '#6495ED', '#DC143C'
    ]

def plot_class_distribution():
    iscx_vpn_distribution = list(iscx_vpn_class_counts.values())
    iscx_vpn_labels = iscx_vpn_get_short_labels()

    vnat_distribution = list(vnat_class_counts.values())
    vnat_labels = vnat_get_short_labels()

    iscx_tor_distribution = list(iscx_tor_class_counts.values())  
    iscx_tor_labels = iscx_tor_get_short_labels()

    mendeley_distribution = list(mendeley_network_traffic_class_counts.values())  
    mendeley_labels = mendeley_network_traffic_get_unique_labels()

    custom_distribution = list(realtime_class_counts.values())  
    custom_labels = realtime_get_unique_labels()


    # Create a DataFrame for each dataset
    iscx_vpn_df = pd.DataFrame({'Labels': iscx_vpn_labels, 'Count': iscx_vpn_distribution})
    vnat_df = pd.DataFrame({'Labels': vnat_labels, 'Count': vnat_distribution})
    iscx_tor_df = pd.DataFrame({'Labels': iscx_tor_labels, 'Count': iscx_tor_distribution})
    mendeley_df = pd.DataFrame({'Labels': mendeley_labels, 'Count': mendeley_distribution})
    custom_df = pd.DataFrame({'Labels': custom_labels, 'Count': custom_distribution})

    # Create a figure with 3 subplots in one column
    fig, axes = plt.subplots(5, 1, figsize=(12, 30))

    # Function to add value labels on bars
    def add_value_labels(ax, fontsize=17):
        for container in ax.containers:
            ax.bar_label(container, fontsize=fontsize, padding=3, rotation=0)

    # Plot bar chart for ISCX2016 using seaborn
    sns.barplot(x='Labels', y='Count', data=iscx_vpn_df, ax=axes[0], 
                palette=colors[:len(iscx_vpn_df)], legend=False)
    add_value_labels(axes[0])
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].set_ylim(0, max(iscx_vpn_distribution) * 1.15)
    axes[0].set_title('ISCX-VPN Dataset Distribution', pad=10)
    axes[0].set_xlabel('Traffic Classes')  # Set consistent x-label
    axes[0].set_ylabel('Number of Samples')  # Set consistent y-label
    
    # Plot bar chart for VNAT-PN using seaborn
    sns.barplot(x='Labels', y='Count', data=vnat_df, ax=axes[1], 
                palette=colors[:len(vnat_df)], legend=False)
    add_value_labels(axes[1])
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].set_ylim(0, max(vnat_distribution) * 1.15)
    axes[1].set_title('VNAT Dataset Distribution', pad=10)
    axes[1].set_xlabel('Traffic Classes')  # Set consistent x-label
    axes[1].set_ylabel('Number of Samples')  # Set consistent y-label
    
    # Plot bar chart for Tor using seaborn
    sns.barplot(x='Labels', y='Count', data=iscx_tor_df, ax=axes[2], 
                palette=colors[:len(iscx_tor_df)], legend=False)
    add_value_labels(axes[2])
    axes[2].tick_params(axis='x', rotation=90)
    axes[2].set_ylim(0, max(iscx_tor_distribution) * 1.15)
    axes[2].set_title('ISCX-Tor Dataset Distribution', pad=10)
    axes[2].set_xlabel('Traffic Classes')  # Set consistent x-label
    axes[2].set_ylabel('Number of Samples')  # Set consistent y-label
    





    sns.barplot(x='Labels', y='Count', data=mendeley_df, ax=axes[3], 
                palette=colors[:len(mendeley_df)], legend=False)
    add_value_labels(axes[3])
    axes[3].tick_params(axis='x', rotation=90)
    axes[3].set_ylim(0, max(mendeley_distribution) * 1.15)
    axes[3].set_title('Mendeley NetworkTraffic Dataset Distribution', pad=10)
    axes[3].set_xlabel('Traffic Classes')  # Set consistent x-label
    axes[3].set_ylabel('Number of Samples')  # Set consistent y-label



    sns.barplot(x='Labels', y='Count', data=custom_df, ax=axes[4], 
                palette=colors[:len(custom_df)], legend=False)
    add_value_labels(axes[4])
    axes[4].tick_params(axis='x', rotation=90)
    axes[4].set_ylim(0, max(custom_distribution) * 1.15)
    axes[4].set_title('Custom Dataset Distribution', pad=10)
    axes[4].set_xlabel('Traffic Classes')  # Set consistent x-label
    axes[4].set_ylabel('Number of Samples')  # Set consistent y-label

















    plt.tight_layout(pad=2.0)
    plt.savefig('visualization/fig_dataset_distribution_barplot.png', dpi=300)
    plt.show()

# Call the function to plot the distributions
plot_class_distribution()