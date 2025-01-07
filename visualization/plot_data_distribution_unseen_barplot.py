import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set the default font to Times New Roman
plt.rcParams.update({
    "font.family": 'Times New Roman',
    "font.size": 12
})

def plot_class_distribution():

    mendeley_distribution = [ 11048, 496, 588, 1704 ]
    mendeley_labels = ['browse', 'ft', 'p2p', 'stream']

    custom_distribution = [ 4298, 696, 10428, 15838 ]
    custom_labels = ['stream', 'chat', 'voip', 'game']


    mendeley_df = pd.DataFrame({'Labels': mendeley_labels, 'Count': mendeley_distribution})
    custom_df = pd.DataFrame({'Labels': custom_labels, 'Count': custom_distribution})


    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Function to add value labels on bars
    def add_value_labels(ax, fontsize=10):
        for container in ax.containers:
            ax.bar_label(container, fontsize=fontsize, padding=3, rotation=0)

    # Plot bar chart for ISCX2016 using seaborn
    sns.barplot(x='Labels', y='Count', data=mendeley_df, ax=axes[0], 
                hue='Labels', legend=False, palette='tab20')
    add_value_labels(axes[0])
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].set_ylim(0, max(mendeley_distribution) * 1.15)
    axes[0].set_title('Mendeley NetworkTraffic Dataset', pad=10)
    axes[0].set_xlabel('Traffic Classes')  # Set consistent x-label
    axes[0].set_ylabel('Number of Samples')  # Set consistent y-label
    
    # Plot bar chart for VNAT-PN using seaborn
    sns.barplot(x='Labels', y='Count', data=custom_df, ax=axes[1], 
                hue='Labels', legend=False, palette='tab10')
    add_value_labels(axes[1])
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].set_ylim(0, max(custom_distribution) * 1.15)
    axes[1].set_title('Custom Dataset', pad=10)
    axes[1].set_xlabel('Traffic Classes')  # Set consistent x-label
    axes[1].set_ylabel('Number of Samples')  # Set consistent y-label
    
    
    plt.tight_layout(pad=2.0)
    plt.savefig('visualization/fig_unseen_data_distribution_barplot.png')
    plt.show()

# Call the function to plot the distributions
plot_class_distribution()