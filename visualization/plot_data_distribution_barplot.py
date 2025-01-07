import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils import iscx_vpn_class_counts, vnat_class_counts

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
    # Updated ISCX-VPN distribution and labels

    iscx_vpn_distribution = list(iscx_vpn_class_counts.values())



    iscx_vpn_labels = [
        'email', 'chat', 'stream', 'ft', 'voip', 'p2p',
        'vpn_email', 'vpn_chat', 'vpn_stream', 'vpn_ft', 'vpn_voip', 'vpn_p2p'
    ]

    # Updated VNAT distribution and labels
    vnat_distribution = list(vnat_class_counts.values())
    vnat_labels = [
        'stream', 'voip', 'ft', 'p2p', 'vpn_stream', 
        'vpn_voip', 'vpn_ft', 'vpn_p2p'
    ]




    # Updated Tor distribution and labels
    iscx_tor_distribution = [55660, 700, 1008, 2016, 3774, 3104, 2902, 45838, 68, 12, 42, 46, 294, 12, 28, 40]    
    iscx_tor_labels = [
        'browse', 'email','chat','audio','video','ft','voip','p2p','tor_browse','tor_email',
        'tor_chat','tor_audio','tor_video','tor_ft','tor_voip','tor_p2p'
    ]


    # Create a DataFrame for each dataset
    iscx_vpn_df = pd.DataFrame({'Labels': iscx_vpn_labels, 'Count': iscx_vpn_distribution})
    vnat_df = pd.DataFrame({'Labels': vnat_labels, 'Count': vnat_distribution})
    iscx_tor_df = pd.DataFrame({'Labels': iscx_tor_labels, 'Count': iscx_tor_distribution})

    # Create a figure with 3 subplots in one column
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

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
    # sns.barplot(x='Labels', y='Count', data=iscx_tor_df, ax=axes[2], 
    #             colors=colors, legend=False)
    # add_value_labels(axes[2])
    # axes[2].tick_params(axis='x', rotation=90)
    # axes[2].set_ylim(0, max(iscx_tor_distribution) * 1.15)
    # axes[2].set_title('ISCX-Tor Dataset Distribution', pad=10)
    # axes[2].set_xlabel('Traffic Classes')  # Set consistent x-label
    # axes[2].set_ylabel('Number of Samples')  # Set consistent y-label
    
    plt.tight_layout(pad=2.0)
    plt.savefig('visualization/fig_dataset_distribution_barplot.png', dpi=300)
    plt.show()

# Call the function to plot the distributions
plot_class_distribution()