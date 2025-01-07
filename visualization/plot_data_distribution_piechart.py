import matplotlib.pyplot as plt

# Set the default font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
font_size = 18

def plot_class_distribution():
    # Updated ISCX2016 distribution and labels
    iscx2016_distribution = [
        14621, 21610, 3752, 138549, 399893, 4996, 
        596, 8058, 1318, 2040, 7730, 954
    ]
    iscx2016_labels = [
        'email', 'chat', 'streaming', 'file_transfer', 'voip', 'p2p',
        'vpn_email', 'vpn_chat', 'vpn_streaming', 'vpn_file_transfer', 'vpn_voip', 'vpn_p2p'
    ]

    # Updated VNAT-PN distribution and labels
    vnat_vpn_distribution = [32826, 27182, 3518, 3052, 712, 18, 16, 10]
    vnat_vpn_labels = [
        'file_transfer', 'p2p', 'streaming', 'voip', 'vpn_voip', 
        'vpn_file_transfer', 'vpn_p2p', 'vpn_streaming'
    ]

 	# Updated Tor distribution and labels
    tor_distribution = [2645, 497, 485, 1026, 1529, 1663, 4524, 2139]
    tor_labels = [
        'browsing', 'email', 'chat', 'audio_stream', 
        'video_stream', 'file_transfer', 'voip', 'p2p' 
    ]

    alpha = 0.5
    # Define color palettes for each dataset
    iscx2016_colors = plt.cm.tab20.colors  # Use a colormap for more categories
    vnat_vpn_colors = plt.cm.tab10.colors  # Use a different colormap
    tor_colors = plt.cm.tab10.colors  # Use a different colormap

    iscx2016_colors_alpha = [(r, g, b, alpha) for r, g, b in plt.cm.tab20.colors]
    vnat_vpn_colors_alpha = [(r, g, b, alpha) for r, g, b in plt.cm.tab10.colors]
    tor_colors_alpha = [(r, g, b, alpha) for r, g, b in plt.cm.tab10.colors]

    # Create a figure with 2 subplots in one column
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Plot pie chart for ISCX2016
    wedges, _ = axes[0].pie(iscx2016_distribution, radius=1.05, startangle=90, colors=iscx2016_colors_alpha)
    for wedge, color in zip(wedges, iscx2016_colors):
        wedge.set_edgecolor(color[:3] + (1.0,))  # Set edge color to match fill color with full opacity

    axes[0].set_title('ISCX-VPN Class Distribution', fontsize=font_size)
    iscx2016_percentages = [f"{label} ({value / sum(iscx2016_distribution) * 100:.2f}%)" for label, value in zip(iscx2016_labels, iscx2016_distribution)]
    axes[0].legend(iscx2016_percentages, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=font_size, ncol=2)  # Move legend to the right and center vertically

    # Plot pie chart for VNAT-PN
    wedges, _ = axes[1].pie(vnat_vpn_distribution, radius=1.05, startangle=90, colors=vnat_vpn_colors_alpha)
    for wedge, color in zip(wedges, vnat_vpn_colors):
        wedge.set_edgecolor(color[:3] + (1.0,))

    # axes[1].pie(vnat_vpn_distribution, radius=1.05, startangle=90, colors=vnat_vpn_colors_alpha, wedgeprops={'edgecolor': (0.5,0.5,0.5, 0.3), 'aa': True}) 
    axes[1].set_title('VNAT Class Distribution', fontsize=font_size)
    vnat_vpn_percentages = [f"{label} ({value / sum(vnat_vpn_distribution) * 100:.2f}%)" for label, value in zip(vnat_vpn_labels, vnat_vpn_distribution)]
    axes[1].legend(vnat_vpn_percentages, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=font_size, ncol=2)  # Move legend to the right and center vertically

    # Plot pie chart for Tor
    wedges, _ = axes[2].pie(tor_distribution, radius=1.05, startangle=90, colors=tor_colors_alpha)
    for wedge, color in zip(wedges, tor_colors):
        wedge.set_edgecolor(color[:3] + (1.0,)) 

    # axes[2].pie(tor_distribution, radius=1.05, startangle=90, colors=tor_colors_alpha, wedgeprops={'edgecolor': (0.5,0.5,0.5, 0.3), 'aa': True}) 
    axes[2].set_title('ISCX-Tor Class Distribution', fontsize=font_size)
    tor_percentages = [f"{label} ({value / sum(tor_distribution) * 100:.2f}%)" for label, value in zip(tor_labels, tor_distribution)]
    axes[2].legend(tor_percentages, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=font_size, ncol=2)  # Move legend to the right and center vertically

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    plt.savefig('visualization/fig_dataset_distribution.png')
    plt.show()

# Call the function to plot the distributions
plot_class_distribution()