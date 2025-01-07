from utils import *
import matplotlib.pyplot as plt
import numpy as np


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


def plot_usmobileapp_distribution():
    # Prepare data for plotting
    countries = list(usmobileapp_class_counts.keys())
    android_counts = [usmobileapp_class_counts[country]['android'] for country in countries]
    ios_counts = [usmobileapp_class_counts[country]['ios'] for country in countries]

    # Set width of bars and positions of the bars
    bar_width = 0.35
    x = np.arange(len(countries))

    # Create bars
    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_width/2, android_counts, bar_width, label='Android', color='#3ddc84')
    plt.bar(x + bar_width/2, ios_counts, bar_width, label='iOS', color='#555555')

    # Customize the plot
    plt.xlabel('Countries')
    plt.ylabel('Number of Samples')
    plt.title('US Mobile App Dataset Distribution', pad=10)
    plt.xticks(x, countries)
    plt.ylim(0, max(android_counts) * 1.15)
    plt.legend()

    # Add value labels on top of each bar
    for i, v in enumerate(android_counts):
        plt.text(i - bar_width/2, v, str(v), ha='center', va='bottom')
    for i, v in enumerate(ios_counts):
        plt.text(i + bar_width/2, v, str(v), ha='center', va='bottom')

    # Adjust layout and display
    plt.tight_layout()
    plt.savefig('usmobileapp_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()





def plot_distributions():
    # Create figure and axes for 2x2 subplot grid with increased height
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 20))

    # Common function to create bars and format axes
    def create_bar_plot(ax, data_dict, title):
        labels = list(data_dict.keys())
        values = list(data_dict.values())
        bars = ax.bar(labels, values, color=colors[:len(labels)])
        ax.set_xlabel('Traffic Classes', fontsize=24, labelpad=10)
        ax.set_ylabel('Number of Samples', fontsize=24, labelpad=10)
        ax.set_title(title, fontsize=24, pad=20)
        # Rotate labels and adjust their position
        ax.tick_params(axis='x', rotation=45, labelsize=24)
        ax.tick_params(axis='y', labelsize=24)
        plt.setp(ax.get_xticklabels(), ha='right')
        ax.set_ylim(0, max(values) * 1.15)
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=24)
        return bars

    # Plot first three datasets using the common function
    create_bar_plot(ax1, iscx_tor_class_counts, 'ISCX-Tor Dataset Distribution')
    create_bar_plot(ax2, mendeley_network_traffic_class_counts, 'Mendeley NetworkTraffic Dataset Distribution')
    

    # Add US Mobile App distribution in the fourth plot
    countries = list(usmobileapp_class_counts.keys())
    android_counts = [usmobileapp_class_counts[country]['android'] for country in countries]
    ios_counts = [usmobileapp_class_counts[country]['ios'] for country in countries]

    bar_width = 0.35
    x = np.arange(len(countries))
    
    # Create bars for US Mobile App plot
    ax3.bar(x - bar_width/2, android_counts, bar_width, label='Android', color='#3ddc84')
    ax3.bar(x + bar_width/2, ios_counts, bar_width, label='iOS', color='#555555')
    
    # Customize the fourth plot
    ax3.set_xlabel('Countries', fontsize=24, labelpad=10)
    ax3.set_ylabel('Number of Samples', fontsize=24, labelpad=10)
    ax3.set_title('US Mobile App Dataset Distribution', fontsize=24, pad=20)
    ax3.set_xticks(x)
    ax3.set_xticklabels(countries, rotation=45, ha='right', fontsize=24)
    ax3.tick_params(axis='y', labelsize=24)
    ax3.set_ylim(0, max(android_counts) * 1.15)
    ax3.legend(fontsize=24)

    # Add value labels for US Mobile App plot
    for i, v in enumerate(android_counts):
        ax3.text(i - bar_width/2, v, str(v), ha='center', va='bottom', fontsize=24)
    for i, v in enumerate(ios_counts):
        ax3.text(i + bar_width/2, v, str(v), ha='center', va='bottom', fontsize=24)


    create_bar_plot(ax4, realtime_class_counts, 'Custom Dataset Distribution')

    # Adjust layout with more space between subplots
    plt.tight_layout(pad=3.0)
    plt.savefig('FigDatasetDistribution.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':

    plot_distributions()

