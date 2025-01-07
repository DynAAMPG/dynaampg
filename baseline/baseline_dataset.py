import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import json
import os
import numpy as np
import tqdm
from sklearn.preprocessing import OneHotEncoder
from config import *
from utils import iscx_vpn_get_unique_labels

class BaselineDataset(Dataset):
    def __init__(self, data_dir, class_labels):
        self.data_dir = data_dir
        self.class_labels = class_labels
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        
        # Get all unique classes for one-hot encoding
        all_classes = []
        # Pre-load all data into memory
        self.cached_data = []
        
        for file_name in tqdm.tqdm(self.file_list, desc="Loading dataset"):
            with open(os.path.join(data_dir, file_name), 'r') as f:
                data = json.load(f)
                all_classes.append(data['class'])
                # Store the processed tensors directly
                features = torch.tensor(data['features'], dtype=torch.float32)
                self.cached_data.append(data)
        
        # Initialize and fit one-hot encoder
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(np.array(all_classes).reshape(-1, 1))
        
        # Pre-compute all labels
        self.cached_labels = []
        for data in self.cached_data:
            one_hot = self.encoder.transform([[data['class']]])
            label = torch.tensor(one_hot, dtype=torch.float32).squeeze()
            self.cached_labels.append(label)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Return pre-loaded data instead of reading from disk
        features = torch.tensor(self.cached_data[idx]['features'], dtype=torch.float32)
        return features, self.cached_labels[idx]
    


    def get_train_test_loaders(self, batch_size=32, split=0.7):
        # Get indices for each class
        class_indices = {}
        for idx, data in enumerate(self.cached_data):
            class_label = data['class']
            if class_label not in class_indices:
                class_indices[class_label] = []
            class_indices[class_label].append(idx)
            
        train_indices = []
        test_indices = []
        
        # For each class, split indices into train and test
        for class_label in class_indices:
            indices = class_indices[class_label]
            np.random.shuffle(indices)
            
            split_idx = int(len(indices) * split)
            train_indices.extend(indices[:split_idx])
            test_indices.extend(indices[split_idx:])
            
        # Create train and test datasets using subset
        train_dataset = Subset(self, train_indices)
        test_dataset = Subset(self, test_indices)
        
        # Create and return data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader




    def get_instances_per_class(self):
        """
        Returns a dictionary containing the count of instances for each class label.
        
        Returns:
            dict: A dictionary where keys are class names and values are instance counts
        """
        class_counts = {}
        
        for data in self.cached_data:
            class_label = data['class']
            if class_label not in class_counts:
                class_counts[class_label] = 0
            class_counts[class_label] += 1
            
        return class_counts
    
    def get_class_names(self):
        return self.encoder.categories_[0]
    

    def get_loader_class_count(self, loader):

        class_counts = {}
        class_names = self.get_class_names()
        
        # Initialize counts for all classes to 0
        for class_name in class_names:
            class_counts[class_name] = 0
            
        # Count instances
        for _, labels in loader:
            # Convert one-hot encoded labels back to class indices
            class_indices = torch.argmax(labels, dim=1)
            
            # Count occurrences of each class
            for idx in class_indices:
                class_name = class_names[idx]
                class_counts[class_name] += 1
                
        return class_counts


if __name__ == "__main__":
    dataset = BaselineDataset(os.path.join(ISCX_VPN_DATASET_DIR, "raw"), iscx_vpn_get_unique_labels())
    train_loader, test_loader = dataset.get_train_test_loaders()
    train_counts = dataset.get_loader_class_count(train_loader)
    test_counts = dataset.get_loader_class_count(test_loader)

    print("Train counts:")
    for class_name, count in train_counts.items():
        print(count)

    print("\nTest counts:")
    for class_name, count in test_counts.items():
        print(count)
