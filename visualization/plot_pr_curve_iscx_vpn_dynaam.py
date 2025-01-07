import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
from baseline.baseline_alt_2 import BaselineAlt2Classifier
from baseline.baseline_dataset import BaselineDataset
from config import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
from utils import iscx_vpn_get_unique_labels, iscx_vpn_class_counts


plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "font.size": 16
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

def plot_pr_curves(model, test_loader, num_classes, device, class_counts, title, file_name):   
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
                label=f"{class_name} (AP = {data[class_name]['pr_auc']:.3f})")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(fontsize=14, loc='best')
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()

if __name__ == "__main__":
    batch_size = 32
    hidden_size = 128
    model_name = 'baseline_classifier_iscx_vpn_dynaam_9.pth'
    title = "Precision-Recall Curve (DynAAM) - ISCX-VPN"
    file_name = "fig-pr-curves-iscx-vpn(dynaam).png"

    model_path = os.path.join(SAVED_MODELS_DIR, 'baseline_classifier_iscx_vpn_dynaam_99.pth')
    dataset = BaselineDataset(os.path.join(ISCX_VPN_DATASET_DIR, "raw"), iscx_vpn_get_unique_labels())
    train_loader, test_loader = dataset.get_train_test_loaders()    
    
    sample_features, sample_labels = dataset[0]
    input_size = sample_features.shape[0] * sample_features.shape[1]
    num_classes = sample_labels.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineAlt2Classifier(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)    

    plot_pr_curves(model, test_loader, num_classes, device, iscx_vpn_class_counts, title, file_name)    
    # plot_confusion_matrix(model, test_loader, num_classes, device, class_counts)