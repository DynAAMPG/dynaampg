import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib
from session_dataset import SessionDataset
import torch
from config import *


import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm

def myconverter(o):
    if isinstance(o, np.float32):
        return float(o)

def preprocess(dataset_path):
    files = os.listdir(dataset_path)
    pbar = tqdm(total=len(files), desc='Files Done: ')

    for file_name in files:
        classs = ''
        class_vector = []
        edge_indices = []
        features = []
        id = 0

        json_str = {}

        with open(os.path.join(dataset_path, file_name), 'r') as file:

            data = json.load(file)

            classs = data['class']
            class_vector = data['class_vector']
            edge_indices = data['edge_indices']
            id = data['id']

            features = torch.rand((10,1500))
            tensor = torch.rand((10,1500)) < 0.7
            tensor = tensor.long()
            features = features * tensor

            features = features.numpy().tolist()
 
        with open(os.path.join(dataset_path, file_name), 'w') as file:
            json_str['id'] = id
            json_str['features'] = features
            json_str['edge_indices'] = edge_indices
            json_str['class'] = classs
            json_str['class_vector'] = class_vector
            file.write(json.dumps(json_str, default=myconverter))


        # Update progress bar
        pbar.update(1)


        
        


if __name__=='__main__':

    # Process OOD dataset
    preprocess(os.path.join(OOD_DATASET_DIR, 'raw'))
 
    print('OOD Datasets Generate. ALL DONE!!!!!')


