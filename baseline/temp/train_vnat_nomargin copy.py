import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from baseline_dataset import BaselineDataset
from baseline import BaselineClassifier
from config import *
from utils import vnat_get_unique_labels
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score
import numpy as np

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter('runs/vnat_baseline_training')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, actual = torch.max(labels.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == actual).sum().item()
            
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(actual.cpu().numpy())
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, actual = torch.max(labels.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == actual).sum().item()
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(actual.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Calculate precision for each class
        train_precision = precision_score(all_train_labels, all_train_preds, average=None)
        val_precision = precision_score(all_val_labels, all_val_preds, average=None)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
        
        # Log precision for each class
        for i, (train_prec, val_prec) in enumerate(zip(train_precision, val_precision)):
            writer.add_scalar(f'Precision/train_class_{i}', train_prec, epoch)
            writer.add_scalar(f'Precision/val_class_{i}', val_prec, epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        torch.save(model.state_dict(), f'{SAVED_MODELS_DIR}/baseline_classifier_vnat_nomargin_{epoch}.pth')
    
    writer.close()

if __name__ == "__main__":
    epochs = 500
    batch_size = 32
    hidden_size = 128

    dataset = BaselineDataset(os.path.join(VNAT_DATASET_DIR, "raw"), vnat_get_unique_labels())
    train_loader, val_loader = dataset.get_train_test_loaders()    
    
    # Get input size from first sample
    sample_features, sample_labels = dataset[0]
    input_size = sample_features.shape[0] * sample_features.shape[1]
    num_classes = sample_labels.shape[0]

    model = BaselineClassifier(input_size, hidden_size, num_classes)
    train_model(model, train_loader, val_loader, num_epochs=epochs)

