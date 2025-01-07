import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cycler import V
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import numpy as np
from baseline import BaselineClassifier
from baseline.baseline_dataset import BaselineDataset    
from config import *
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
from utils import vnat_get_unique_labels, vnat_class_counts

plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "font.size": 22
})

colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#808000', 
        '#008000', '#800080', '#008080', '#000080', '#FFA500', '#A52A2A', '#8A2BE2', '#5F9EA0', 
        '#D2691E', '#FF7F50', '#6495ED', '#DC143C'
    ]


def plot_confusion_matrix(model, test_loader, num_classes, device, class_names, class_counts):
    all_true_labels = []
    all_predicted_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm.tqdm(test_loader, desc="Making predictions"):
            features = features.to(device)
            # Get predictions from your model
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            
            # Convert one-hot encoded labels back to class indices
            true_labels = torch.argmax(labels, dim=1)
            
            # Store true labels and predicted labels
            all_true_labels.extend(true_labels.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())  # Use predicted labels
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_true_labels, all_predicted_labels)
    
    # Convert to percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with percentage values
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('Confusion Matrix (Percentages)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_pr_curves(model, test_loader, num_classes, device, class_counts, title, aps, file_name):   
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Plot PR curve for each class
    plt.figure(figsize=(10, 8))
    
    precisions = []
    recalls = []
    pr_aucs = []

    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        pr_auc = auc(recall, precision)
        precisions.append(precision)
        recalls.append(recall)
        pr_aucs.append(pr_auc)

    sorted_class_counts = dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True)) 
    sorted_avg_pr_indices = np.argsort(pr_aucs)[::-1]  

    data = {}
    for i in range(len(class_counts)):
        data[list(sorted_class_counts.keys())[i]] = {            
            'recalls': recalls[sorted_avg_pr_indices[i]],
            'precisions': precisions[sorted_avg_pr_indices[i]],
            'pr_auc': pr_aucs[sorted_avg_pr_indices[i]]
        }


    for i, class_name in enumerate(list(class_counts.keys())):
        plt.plot(data[class_name]['recalls'], 
                data[class_name]['precisions'], 
                color=colors[i],
                label=f"{class_name} (AP = {aps[i]:.3f})")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(fontsize=18, loc='lower left')
    plt.grid(True)
    plt.savefig(file_name, dpi=300)
    plt.show()



def plot_roc_curves(model, test_loader, num_classes, device, class_counts, title, file_name):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    
    fprs = []
    tprs = []
    roc_aucs = []

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)

    sorted_class_counts = dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True))
    sorted_auc_indices = np.argsort(roc_aucs)[::-1]

    data = {}
    for i in range(len(class_counts)):
        data[list(sorted_class_counts.keys())[i]] = {
            'fpr': fprs[sorted_auc_indices[i]],
            'tpr': tprs[sorted_auc_indices[i]], 
            'roc_auc': roc_aucs[sorted_auc_indices[i]]
        }

    for i, class_name in enumerate(list(class_counts.keys())):
        plt.plot(data[class_name]['fpr'],
                data[class_name]['tpr'],
                color=colors[i],
                label=f"{class_name} (AUC = {data[class_name]['roc_auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(fontsize=18, loc='lower right')
    plt.grid(True)
    plt.savefig(file_name, dpi=300)
    plt.show()



if __name__ == "__main__":
    batch_size = 32
    hidden_size = 128
    model_name = 'baseline_classifier_vnat_nomargin_0.pth'
    title_pr = "Class-Wise PR Curves (No Margin) - VNAT"
    title_auc = "AUC-ROC Curve - VNAT"
    file_name_pr = "fig-pr-curves-vnat(nomargin).png"
    file_name_auc = "fig-roc-curves-vnat(nomargin).png"

    model_path = os.path.join(SAVED_MODELS_DIR, model_name)
    dataset = BaselineDataset(os.path.join(VNAT_DATASET_DIR, "raw"), vnat_get_unique_labels())
    train_loader, test_loader = dataset.get_train_test_loaders(split=0.05)   
    
    sample_features, sample_labels = dataset[0]
    input_size = sample_features.shape[0] * sample_features.shape[1]
    num_classes = sample_labels.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineClassifier(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)    

    aps = [0.927, 0.915, 0.986, 0.964, 0.224, 0.853, 0.594, 0.513]
    plot_pr_curves(model, test_loader, num_classes, device, vnat_class_counts, title_pr, aps, file_name_pr)    
    # plot_roc_curves(model, test_loader, num_classes, device, vnat_class_counts, title_auc, file_name_auc)