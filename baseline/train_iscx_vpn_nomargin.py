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
from utils import iscx_vpn_get_unique_labels

def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
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
            total += labels.size(0)
            correct += (predicted == actual).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

        torch.save(model.state_dict(), f'{SAVED_MODELS_DIR}/baseline_classifier_iscx_vpn_nomargin_{epoch}.pth')


if __name__ == "__main__":
    epochs = 100
    batch_size = 32
    hidden_size = 128

    dataset = BaselineDataset(os.path.join(ISCX_VPN_DATASET_DIR, "raw"), iscx_vpn_get_unique_labels())
    train_loader, test_loader = dataset.get_train_test_loaders()    
    
    # Get input size from first sample
    sample_features, sample_labels = dataset[0]
    input_size = sample_features.shape[0] * sample_features.shape[1]
    num_classes = sample_labels.shape[0]

    model = BaselineClassifier(input_size, hidden_size, num_classes)
    train_model(model, train_loader, num_epochs=epochs)

