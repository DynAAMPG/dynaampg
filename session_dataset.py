import torch
from torch_geometric.data import InMemoryDataset, Dataset, Data

from tqdm import tqdm
import torch_geometric.transforms as T
from pathlib import Path
import json
import numpy as np
from utils import *
import os
from config import *
from utils import *
from torch_geometric.loader import DataLoader
from torch.utils.data import SubsetRandomSampler


class SessionDataset(InMemoryDataset):
    def __init__(self, root, class_labels, exclude_classes=None, transform=None, pre_transform=None, pre_filter=None):
        self.exclude_classes = exclude_classes
        self.class_labels = class_labels
        super(SessionDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root        
                
        self.new_class_labels = []

    @property
    def raw_file_names(self):
        raw_files = list(Path(self.root + '\\raw').rglob('*.json'))
        raw_files = [item.name for item in raw_files]
        return raw_files

    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def download(self):
        pass

    def process(self):

        if self.exclude_classes is not None and len(self.exclude_classes) > 0:
            self.new_class_labels = [label for label in self.class_labels if label not in self.exclude_classes]
        else:
            self.new_class_labels = self.class_labels   


        pbar = tqdm(total=len(self.raw_paths), desc='Files Done: ')

        data_list = []
        for file_number, raw_file in enumerate(self.raw_paths):

            with open(raw_file, 'r') as file_handle:
                json_data = json.load(file_handle)
                
                features = json_data["features"]
                edge_indices = json_data["edge_indices"]
                class_label = json_data["class"]

                class_vector = np.zeros(len(self.new_class_labels), dtype=int)
                index = self.new_class_labels.index(class_label)
                class_vector[index] = 1

                edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)
                x = torch.tensor(features, dtype=torch.float)
                y = torch.tensor(np.array([class_vector], dtype=np.float32), dtype=torch.float)
                graph = Data(x=x, edge_index=edge_index, y=y)

                # self.class_counts[class_label] += 1

                data_list.append(graph)

            pbar.update(1)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @classmethod
    def get_random_session(cls, num_sessions=1):
        sessions = {}
        
        x = torch.rand((5,1500))
        edge_indices = [[0,1,2,3],[1,2,3,4]]
        y = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        data = Data(x=x, edge_indices=edge_indices, y=y)

        return sessions
    

    
    def get_class_distribution(self):
        class_counts = {label: 0 for label in self.class_labels}
        instances = self.len()

        for i in range(instances):
            instance = self.get(i)
            y = instance.y
            index = torch.argmax(y).item()
            
            class_counts[self.class_labels[index]] += 1

        return class_counts
    

    def get_train_test_loaders(self, batch_size=32, split=0.7):
        # Get indices for each class
        class_indices = {label: [] for label in self.class_labels}
        for i in range(self.len()):
            instance = self.get(i)
            y = instance.y
            index = torch.argmax(y).item()
            class_indices[self.class_labels[index]].append(i)
        
        # Split indices for each class according to ratio
        train_indices = []
        test_indices = []
        for label in self.class_labels:
            indices = class_indices[label]
            n_train = int(len(indices) * split)
            
            # Shuffle indices
            indices = np.random.permutation(indices)
            
            train_indices.extend(indices[:n_train])
            test_indices.extend(indices[n_train:])
        
        # Create samplers for train and test
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        
        # Create data loaders
        train_loader = DataLoader(self, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(self, batch_size=batch_size, sampler=test_sampler)
        
        return train_loader, test_loader
    

    @classmethod
    def get_class_count(cls, loader, class_labels):
        class_counts = {label: 0 for label in class_labels}
        for batch in loader:
            indices = torch.argmax(batch.y, dim=1)
            for idx in indices:
                class_counts[class_labels[idx.item()]] += 1

        return class_counts


if __name__ == '__main__':

    dataset = SessionDataset(root=REALTIME_DATASET_DIR, class_labels=realtime_get_unique_labels())

    class_counts = dataset.get_class_distribution()
    for key, value in class_counts.items():
        print(value)

    # train_loader, test_loader = dataset.get_train_test_loaders(batch_size=32, split=0.7)

    # train_class_counts = dataset.get_class_count(train_loader, dataset.class_labels)
    # print('Train Distribution:')
    # for key, value in train_class_counts.items():
    #     print(value)

    # test_class_counts = dataset.get_class_count(test_loader, dataset.class_labels)
    # print('Test Distribution:')
    # for key, value in test_class_counts.items():
    #     print(value)
