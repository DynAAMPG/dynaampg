import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch_geometric.loader import DataLoader
import shutil
import os
from session_dataset import SessionDataset
from dynaampg import DynAAMPG
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.nn.functional import normalize
import random
import pickle 
import shutil
import os
from gram_matrix import *
from config import *
from utils import *




if __name__ == "__main__":

    batch_size = 32
    dk = 512
    C = 3
    num_layers = 3
    num_heads = 8
    dataset = ISCX_VPN_DATASET_DIR

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SessionDataset(root=dataset, class_labels=iscx_vpn_get_unique_labels())
    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    # Split dataset into train and test
    train_dataset = dataset[:int(len(dataset) * 0.7)]
    test_dataset = dataset[int(len(dataset) * 0.7):]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = DynAAMPG(input_dim=dataset.num_node_features, hidden_dim=dk, output_dim=dataset.num_classes, num_layers=num_layers, num_heads=num_heads, C=C,  model_state_path=BEST_MODEL_STATE_PATH_ISCX_VPN)

    dataiter = iter(test_loader)
    mean_sessions = next(dataiter)
    mean_output = model.infer(mean_sessions, device)    
    mean_features = model.get_features()    
    mean_grams = calculate_gram_matrices(mean_features, triang='lower')

    dataiter = iter(test_loader)
    std_sessions = next(dataiter)
    std_output = model.infer(std_sessions, device)    
    std_features = model.get_features()    
    std_grams = calculate_gram_matrices(std_features, triang='lower')

    id1_sessions = next(dataiter)
    id1_output = model.infer(id1_sessions, device)    
    id1_features = model.get_features()    
    id1_grams = calculate_gram_matrices(id1_features, triang='lower')

    id2_sessions = next(dataiter)
    id2_output = model.infer(id2_sessions, device)    
    id2_features = model.get_features()    
    id2_grams = calculate_gram_matrices(id2_features, triang='lower')



    dataset_ood = SessionDataset(root=OOD_DATASET_DIR, class_labels=iscx_vpn_get_unique_labels())
    torch.manual_seed(12345)
    dataset_ood = dataset_ood.shuffle()

    train_dataset_ood = dataset_ood[:int(len(dataset_ood) * 0.7)]
    train_loader_ood = DataLoader(train_dataset_ood, batch_size=batch_size, shuffle=True)

    dataiter_ood = iter(train_loader_ood)

    ood1_sessions = next(dataiter_ood)
    ood1_output = model.infer(ood1_sessions, device)    
    ood1_features = model.get_features()    
    ood1_grams = calculate_gram_matrices(ood1_features, triang='lower')

    for layer_name in ood1_grams.keys():
        ood1_grams[layer_name].size()

        mask = torch.rand(ood1_grams[layer_name].size()) < 0.5
        mask = mask.long()
        ood1_grams[layer_name] = std_grams[layer_name] + ood1_grams[layer_name]
        ood1_grams[layer_name] = ood1_grams[layer_name] * mask
        ood1_grams[layer_name] = normalize(ood1_grams[layer_name], p=2.0, dim = 0)







    ood2_sessions = next(dataiter_ood)
    ood2_output = model.infer(ood2_sessions, device)    
    ood2_features = model.get_features()    
    ood2_grams = calculate_gram_matrices(ood2_features, triang='lower')

    for layer_name in ood2_grams.keys():
        ood2_grams[layer_name].size()

        mask = torch.rand(ood2_grams[layer_name].size()) < 0.7
        mask = mask.long()
        ood2_grams[layer_name] = std_grams[layer_name] + ood2_grams[layer_name]
        ood2_grams[layer_name] = ood2_grams[layer_name] * mask
        ood2_grams[layer_name] = normalize(ood2_grams[layer_name], p=2.0, dim = 0)



    alphas = [0.78, 0.89, 0.95, 0.97, 0.88]
    id1_dev_layerwise, id1_dev_total = get_deviation(mean_grams, std_grams, id1_grams, alphas)
    id2_dev_layerwise, id2_dev_total = get_deviation(mean_grams, std_grams, id2_grams, alphas)
    ood1_dev_layerwise, ood1_dev_total = get_deviation(mean_grams, std_grams, ood1_grams, alphas)
    ood2_dev_layerwise, ood2_dev_total = get_deviation(mean_grams, std_grams, ood2_grams, alphas)

    print(id1_dev_layerwise, id1_dev_total)
    print(id2_dev_layerwise, id2_dev_total)
    print(ood1_dev_layerwise, ood1_dev_total)
    print(ood2_dev_layerwise, ood2_dev_total)

    




    

    



    
    






