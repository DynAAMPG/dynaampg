import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cProfile import label
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from config import SAVED_EVALS_DIR
from utils import colors, iscx_vpn_get_unique_labels, vnat_get_unique_labels

plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "font.size": 22
})

def plot_pr_trend(source_file, dataset, title, label, savefile):   
    df = pd.read_csv(source_file)
    
    # Get precision values
    precisions = df[dataset].values
    epochs = range(1, len(precisions) + 1)


    plt.plot(epochs, precisions, 
                color='red',
                label=label)
        
    
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower right', fontsize=18)
    plt.grid(True)
    plt.savefig(savefile, dpi=300)
    plt.show()





def plot_per_class_pr_trend(source_file, dataset, labels, title, savefile):   
    df = pd.read_csv(source_file)
    plt.figure(figsize=(10, 8))
    
    for i, class_label in enumerate(labels):
        precisions = df[class_label].values
        epochs = range(1, len(precisions) + 1)

        plt.plot(epochs, precisions, color=colors[i], label=f'{class_label} (AP = {max(precisions):.3f})', linewidth=2)
        
    
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower right', fontsize=18)
    plt.grid(True)
    plt.savefig(savefile, dpi=300)
    plt.show()




if __name__ == '__main__':
    
    plot_per_class_pr_trend(source_file=os.path.join(SAVED_EVALS_DIR, 'iscx_vpn_per_class_pr_nomargin.csv'), dataset='ISCX-VPN', labels=iscx_vpn_get_unique_labels(), title='AUC-PR Trend (No Margin) - ISCX-VPN', savefile='visualization/iscx_vpn_pr_trend_nomargin.png')
    plot_per_class_pr_trend(source_file=os.path.join(SAVED_EVALS_DIR, 'vnat_per_class_pr_nomargin.csv'), dataset='VNAT', labels=vnat_get_unique_labels(), title='AUC-PR Trend (No Margin) - VNAT', savefile='visualization/vnat_pr_trend_nomargin.png')

    plot_per_class_pr_trend(source_file=os.path.join(SAVED_EVALS_DIR, 'iscx_vpn_per_class_pr_dynaam.csv'), dataset='ISCX-VPN', labels=iscx_vpn_get_unique_labels(), title='AUC-PR Trend (DynAAM) - ISCX-VPN', savefile='visualization/iscx_vpn_pr_trend_dynaam.png')
    plot_per_class_pr_trend(source_file=os.path.join(SAVED_EVALS_DIR, 'vnat_per_class_pr_dynaam.csv'), dataset='VNAT', labels=vnat_get_unique_labels(), title='AUC-PR Trend (DynAAM) - VNAT', savefile='visualization/vnat_pr_trend_dynaam.png')
    
